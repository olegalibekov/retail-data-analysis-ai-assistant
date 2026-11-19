"""Retrieval of golden examples, and validity of their SQL."""

import json
import sqlite3

import pytest
import sqlglot
from sqlglot import exp

from agent.config import settings
from agent.guardrails.sql_guard import guard
from agent.stores.golden import GoldenStore
from conftest import ADMIN


def test_golden_retrieval_finds_the_relevant_trio():
    store = GoldenStore(settings.trios_path)
    assert "churn" in store.retrieve("Why did churn go up in March?")[0]["question"].lower()
    assert "Texas" in store.retrieve("Why are users in Ohio underspending compared to Florida?")[0]["question"]
    assert "report for Q1" in store.retrieve("create a report for Q3 with action items")[0]["question"]
    assert store.retrieve("hello there") == []


def test_every_golden_sql_passes_the_guard():
    trios = json.loads(settings.trios_path.read_text())
    assert len({trio["id"] for trio in trios}) == len(trios)
    for trio in trios:
        assert trio["provenance"] == "sample"
        assert all(isinstance(trio[key], str) and trio[key] for key in ("id", "question", "sql", "report"))
        guard(trio["sql"], ADMIN)


@pytest.mark.parametrize(
    "purchases, expected",
    [
        ([], (0, 0, None)),
        ([(1, "2026-08-01", "Complete")], (1, 1, 1.0)),
        ([(1, "2026-08-01", "Complete")] * 2, (1, 1, 1.0)),
        ([(1, "2026-08-01", "Complete"), (1, "2026-10-30", "Complete")], (1, 0, 0.0)),
        ([(1, "2026-08-01", "Complete"), (1, "2026-10-31", "Complete")], (1, 1, 1.0)),
        ([(1, "2026-08-01", "Complete"), (1, "2026-10-15", "Cancelled")], (1, 1, 1.0)),
        ([(1, "2026-08-01", "Complete"), (1, "2026-10-15", "Returned")], (1, 1, 1.0)),
        ([(1, "2026-08-01", "Returned")], (0, 0, None)),
    ],
)
def test_churn_boundaries_and_empty_months(purchases, expected):
    """Execute the actual example SQL on small fixtures, with a fixed calendar."""
    query = next(trio["sql"] for trio in json.loads(settings.trios_path.read_text()) if "churn" in trio["question"])
    tree = sqlglot.parse_one(query.replace("CURRENT_DATE()", "DATE('2026-11-01')"), read="bigquery")
    months = next(cte for cte in tree.find_all(exp.CTE) if cte.alias == "months")
    months.set(
        "this",
        sqlglot.parse_one(" UNION ALL ".join(f"SELECT '2026-{month:02d}-01' AS month_start" for month in range(5, 11))),
    )
    with sqlite3.connect(":memory:") as connection:
        connection.create_function("DATE_TRUNC", 2, lambda unit, day: day[:7] + "-01")
        connection.execute("CREATE TABLE order_items (user_id INTEGER, created_at TEXT, status TEXT)")
        connection.executemany("INSERT INTO order_items VALUES (?, ?, ?)", purchases)
        rows = connection.execute(tree.sql(dialect="sqlite")).fetchall()
    assert len(rows) == 6
    assert rows[-1] == ("2026-10-01", *expected)
