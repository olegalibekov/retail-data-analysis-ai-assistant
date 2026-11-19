"""Live checks against BigQuery and Gemini. Require --live and credentials (.env).

pytest tests/test_live.py --live -v
"""

import json
import os

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from agent.config import load_users, settings
from agent.core.llm import GeminiClient
from agent.core.prompts import build_system_prompt
from agent.core.tools import tool_schemas
from agent.guardrails.pii import PII_COLUMNS
from agent.guardrails.sql_guard import GuardViolation, guard
from agent.observability import Trace
from agent.warehouse.bq_client import BigQueryRunner, SQLSyntaxError
from agent.warehouse.schema import SCHEMA

needs_bq = pytest.mark.skipif(
    not (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GCP_PROJECT_ID")),
    reason="BigQuery credentials not configured",
)
needs_gemini = pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not configured")

USERS = load_users()
RAW = "`bigquery-public-data.thelook_ecommerce`"


@pytest.fixture(scope="module")
def bq():
    return BigQueryRunner()


@needs_bq
def test_schema_contract_matches_the_live_tables(bq):
    for table, columns in SCHEMA.items():
        live = {c["name"] for c in bq.get_table_schema(table, include_pii=True)}
        assert set(columns) <= live, f"{table}: missing {set(columns) - live}"
    assert not {c["name"] for c in bq.get_table_schema("users")} & PII_COLUMNS
    assert {"email", "first_name", "street_address"} <= {
        c["name"] for c in bq.get_table_schema("users", include_pii=True)
    }


@needs_bq
def test_every_golden_query_is_valid_bigquery_sql(bq):
    for trio in json.loads(settings.trios_path.read_text()):
        for user in (USERS["admin"], USERS["alice"]):
            assert bq.dry_run(guard(trio["sql"], user)) > 0, trio["question"]


@needs_bq
def test_configured_scopes_match_real_products(bq):
    for user in USERS.values():
        count = bq.execute_query(guard("SELECT COUNT(*) AS n FROM products", user))["n"][0]
        assert count > 0, f"{user.id}: scope {user.scope} matches no products"


@needs_bq
def test_scope_and_pii_hold_on_real_data(bq):
    alice = USERS["alice"]
    brands = bq.execute_query(guard("SELECT DISTINCT brand FROM products", alice))["brand"].tolist()
    assert brands == list(alice.scope["brand"])

    people = bq.execute_query(guard("SELECT id FROM users LIMIT 5", alice))
    assert len(people) == 5 and not set(people.columns) & PII_COLUMNS

    scoped = bq.execute_query(
        guard("SELECT (SELECT COUNT(*) FROM order_items) AS items, (SELECT COUNT(*) FROM users) AS buyers", alice)
    )
    raw = bq.execute_query(
        f"SELECT COUNT(*) AS items, COUNT(DISTINCT oi.user_id) AS buyers FROM {RAW}.order_items AS oi "
        f"JOIN {RAW}.products AS p ON p.id = oi.product_id WHERE p.brand = 'Calvin Klein Jeans'"
    )
    assert (scoped["items"][0], scoped["buyers"][0]) == (raw["items"][0], raw["buyers"][0])


@needs_bq
def test_basket_size_counts_only_the_items_in_scope_on_real_data(bq):
    alice = USERS["alice"]
    scoped = bq.execute_query(
        guard(
            "SELECT (SELECT SUM(num_of_item) FROM orders) AS basket_items, (SELECT COUNT(*) FROM order_items) AS items",
            alice,
        )
    )
    raw = bq.execute_query(
        f"SELECT SUM(o.num_of_item) AS basket_items FROM {RAW}.orders AS o WHERE o.order_id IN ("
        f"SELECT oi.order_id FROM {RAW}.order_items AS oi JOIN {RAW}.products AS p ON p.id = oi.product_id "
        "WHERE p.brand = 'Calvin Klein Jeans')"
    )
    assert scoped["basket_items"][0] == scoped["items"][0]  # every basket adds up to her items, no more
    assert scoped["basket_items"][0] <= raw["basket_items"][0]  # the raw column also counts items of other brands


@needs_bq
def test_bare_table_name_reaches_no_table_without_a_default_dataset(bq):
    # The second line behind sql_guard: only its views name a dataset, so no other name resolves to a table.
    with pytest.raises(SQLSyntaxError, match="dataset"):
        bq.dry_run("SELECT COUNT(*) FROM events")


@needs_bq
def test_pii_is_unreachable_even_when_the_name_check_is_bypassed(bq):
    with pytest.raises(GuardViolation, match="aggregate-only"):
        bq.run(guard("SELECT (SELECT AS STRUCT u.*).email FROM users AS u", USERS["admin"]))


@needs_bq
def test_query_errors_are_classified_as_repairable(bq):
    with pytest.raises(SQLSyntaxError, match="nope"):
        bq.run(guard("SELECT nope FROM order_items", USERS["admin"]))


@needs_gemini
def test_gemini_accepts_the_tools_and_writes_a_valid_query():
    trace = Trace("alice", "live", "q")
    messages = [
        SystemMessage(build_system_prompt(USERS["alice"], [])),
        HumanMessage("What was our total revenue last month?"),
    ]
    reply = GeminiClient(tool_schemas()).invoke(messages, trace)
    assert reply.tool_calls and reply.tool_calls[0]["name"] == "run_sql", reply
    guard(reply.tool_calls[0]["args"]["sql"], USERS["alice"])
    assert trace.spans[-1]["tokens_in"] > 0
