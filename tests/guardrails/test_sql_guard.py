"""Requirement 2: read-only access, user scope and PII, enforced in code."""

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.executor import execute

from agent.config import DATASET
from agent.guardrails import sql_guard
from agent.guardrails.pii import PII_COLUMNS
from agent.guardrails.sql_guard import GuardViolation, guard
from agent.warehouse.schema import SCHEMA
from conftest import ADMIN, ALICE, BOB, User


def qualified_tables(sql: str) -> list[str]:
    return [t.name for t in sqlglot.parse_one(sql, read="bigquery").find_all(exp.Table) if t.db]


def top_level_ctes(sql: str) -> list[str]:
    """The guard's views followed by the model's own top-level CTEs."""
    return [cte.alias for cte in sqlglot.parse_one(sql, read="bigquery").find(exp.With).expressions]


def view_body(sql: str, name: str) -> exp.Select:
    return next(cte.this for cte in sqlglot.parse_one(sql, read="bigquery").find_all(exp.CTE) if cte.alias == name)


def row(table: str, **values) -> dict:
    return {**dict.fromkeys(SCHEMA[table]), **values}


def run_offline(sql: str, **rows: list[dict]) -> list[tuple]:
    """Run a guarded query on a few in-memory rows with sqlglot's own executor."""
    catalog, dataset = DATASET.split(".")
    return execute(sql, dialect="bigquery", tables={catalog: {dataset: rows}}).rows


def rejection(sql: str, user: User = ALICE) -> str:
    with pytest.raises(GuardViolation) as error:
        guard(sql, user)
    return error.value.kind


def test_query_is_wrapped_in_scoped_views():
    out = guard(
        "SELECT category, SUM(sale_price) FROM order_items oi JOIN products p ON p.id = oi.product_id GROUP BY 1", ALICE
    )
    assert "brand IN ('Brand A')" in out
    assert "product_id IN (SELECT id FROM products)" in out
    # The only references to real tables are the definitions of the views this query needs.
    assert sorted(qualified_tables(out)) == ["order_items", "products"]


def test_only_the_needed_views_are_added_in_dependency_order():
    def views(sql, user=ALICE):
        return [cte.alias for cte in sqlglot.parse_one(guard(sql, user), read="bigquery").find_all(exp.CTE)]

    assert views("SELECT COUNT(*) FROM products") == ["products"]
    assert views("SELECT COUNT(*) FROM orders") == ["products", "order_items", "orders"]
    assert views("SELECT COUNT(*) FROM users") == ["products", "order_items", "users"]
    assert views("SELECT COUNT(*) FROM users", ADMIN) == ["users"]  # no scope: nothing to depend on
    assert guard("SELECT 1", ALICE) == "SELECT 1"


def test_qualified_references_are_redirected_to_the_views():
    for ref in (
        "`bigquery-public-data.thelook_ecommerce.users`",
        "thelook_ecommerce.users",
        "`bigquery-public-data`.thelook_ecommerce.users",
    ):
        out = guard(f"SELECT COUNT(*) FROM {ref}", ALICE)
        assert sorted(qualified_tables(out)) == ["order_items", "products", "users"], ref  # view definitions only
        assert out.endswith(("SELECT COUNT(*) FROM users", "SELECT COUNT(*) FROM `users`"))  # the view, not the table


def test_unrestricted_user_gets_no_row_filter_but_still_no_pii():
    out = guard("SELECT id FROM users", ADMIN)
    assert "WHERE" not in out
    assert not any(col in out.lower() for col in PII_COLUMNS)


def test_empty_scope_list_grants_nothing_and_stays_valid_sql():
    nobody = User("x", "X", {"brand": ()})
    out = guard("SELECT COUNT(*) FROM products", nobody)
    assert "WHERE FALSE" in out and "IN ()" not in out
    sqlglot.parse_one(out, read="bigquery")


def test_scope_values_are_escaped():
    carol = User("carol", "Carol", {"brand": ("Levi's", "x') OR 1=1 --")})
    out = guard("SELECT COUNT(*) FROM products", carol)
    sqlglot.parse_one(out, read="bigquery")  # still one valid statement
    assert "Levi\\'s" in out and "OR 1 = 1" not in out


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM users",
        "SELECT u.* FROM users AS u",
        "SELECT TO_JSON_STRING(u) FROM users AS u",
        "SELECT * EXCEPT (id) FROM users",
        "SELECT * FROM orders o JOIN users u ON u.id = o.user_id",
    ],
)
def test_star_queries_cannot_reach_pii(sql):
    assert rejection(sql) == "customer_privacy"


@pytest.mark.parametrize(
    "sql, kind",
    [
        ("SELECT email FROM users", "pii_column"),
        ("SELECT u.first_name, u.last_name FROM users u", "pii_column"),
        ("SELECT id FROM users WHERE email LIKE '%@gmail.com'", "pii_column"),
        ("SELECT id FROM users ORDER BY latitude", "pii_column"),
        ("SELECT COUNT(DISTINCT street_address) FROM users", "pii_column"),
        ("DELETE FROM orders WHERE TRUE", "not_read_only"),
        ("UPDATE users SET age = 1 WHERE TRUE", "not_read_only"),
        ("INSERT INTO orders (order_id) VALUES (1)", "not_read_only"),
        ("CREATE TABLE t AS SELECT * FROM orders", "not_read_only"),
        ("DROP TABLE orders", "not_read_only"),
        ("SELECT * INTO t FROM users", "not_read_only"),
        ("SELECT 1; SELECT 2", "not_single_statement"),
        ("SELECT 1; DROP TABLE orders", "not_single_statement"),
        ("SELECT * FROM events", "table_not_allowed"),
        ("SELECT * FROM `bigquery-public-data.thelook_ecommerce.events`", "table_not_allowed"),
        ("SELECT * FROM other_dataset.users", "table_not_allowed"),
        ("SELECT * FROM `other-project.thelook_ecommerce.users`", "table_not_allowed"),
        ("SELECT * FROM `region-us`.INFORMATION_SCHEMA.TABLES", "table_not_allowed"),
        ("SELECT * FROM EXTERNAL_QUERY('conn', 'SELECT 1')", "table_function"),
        ("WITH users AS (SELECT 1 AS id) SELECT * FROM users", "reserved_name"),
        (
            "WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 5) SELECT * FROM t",
            "recursive_cte",
        ),
        (
            "SELECT * FROM "
            "(WITH RECURSIVE t AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 5) SELECT * FROM t) AS a",
            "recursive_cte",
        ),
        ("SELEC 1", "parse_error"),
        ("", "not_single_statement"),
    ],
)
def test_rejections(sql, kind):
    with pytest.raises(GuardViolation) as error:
        guard(sql, ALICE)
    assert error.value.kind == kind


def test_model_ctes_unions_and_subqueries_are_kept():
    sql = """
        WITH monthly AS (SELECT DATE_TRUNC(DATE(created_at), MONTH) AS m, SUM(sale_price) AS rev FROM order_items GROUP BY 1)
        SELECT m, rev FROM monthly WHERE rev > (SELECT AVG(rev) FROM monthly)
        UNION ALL
        SELECT m, rev FROM monthly ORDER BY m LIMIT 5
    """
    out = guard(sql, ALICE)
    names = [cte.alias for cte in sqlglot.parse_one(out, read="bigquery").find_all(exp.CTE)]
    assert names == ["products", "order_items", "monthly"]
    assert "LIMIT 5" in out


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 FROM orders;",
        "SELECT 1 FROM orders; -- done",
        "/* note */ SELECT 1 FROM orders -- ; DROP TABLE orders",
    ],
)
def test_trailing_semicolons_and_comments_are_harmless(sql):
    out = guard(sql, ALICE)
    assert "SELECT 1 FROM orders" in out
    statements = sqlglot.parse(out, read="bigquery")
    assert len(statements) == 1 and isinstance(statements[0], exp.Select)  # the comment stays a comment


def test_struct_tricks_cannot_export_customer_profiles():
    assert rejection("SELECT (SELECT AS STRUCT u.*).email FROM users AS u") == "customer_privacy"


@pytest.mark.parametrize(
    "sql",
    [
        # The CTE lives in a subquery; the same name outside it is the real table.
        "SELECT * FROM (WITH events AS (SELECT 1 AS x) SELECT * FROM events) AS a CROSS JOIN events AS e",
        "SELECT * FROM (WITH inventory_items AS (SELECT 1 AS x) SELECT * FROM inventory_items) AS a "
        "CROSS JOIN inventory_items AS e",
        "SELECT * FROM events WHERE EXISTS (WITH events AS (SELECT 1 AS x) SELECT x FROM events)",
        "(WITH events AS (SELECT 1 AS x) SELECT * FROM events) UNION ALL SELECT * FROM events",
        # A CTE is not visible inside its own definition, nor in the CTEs declared before it.
        "WITH events AS (SELECT * FROM events) SELECT * FROM events",
        "WITH early AS (SELECT * FROM events), events AS (SELECT 1 AS x) SELECT * FROM early",
    ],
)
def test_cte_does_not_vouch_for_the_real_table_of_the_same_name_outside_its_scope(sql):
    for user in (ALICE, ADMIN):
        assert rejection(sql, user) == "table_not_allowed"


@pytest.mark.parametrize(
    "sql",
    [
        # An alias names the columns of a source. BigQuery still looks the same word up as a table.
        "SELECT * FROM orders AS events CROSS JOIN events",
        "SELECT * FROM (SELECT 1 AS x) AS events CROSS JOIN events",
        "SELECT * FROM UNNEST([1]) AS events CROSS JOIN events",
        "SELECT * FROM orders AS events WHERE EXISTS (SELECT 1 FROM events)",
    ],
)
def test_alias_does_not_vouch_for_the_real_table_of_the_same_name(sql):
    assert rejection(sql) == "table_not_allowed"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM (WITH inner_cte AS (SELECT 1 AS x) SELECT * FROM inner_cte) AS a CROSS JOIN inner_cte",
        "WITH outer_cte AS (WITH inner_cte AS (SELECT 1 AS x) SELECT * FROM inner_cte) SELECT * FROM inner_cte",
        "SELECT (SELECT COUNT(*) FROM inner_cte) FROM orders WHERE order_id IN "
        "(WITH inner_cte AS (SELECT order_id FROM order_items) SELECT order_id FROM inner_cte)",
    ],
)
def test_cte_referenced_outside_the_subquery_that_defines_it_is_rejected(sql):
    assert rejection(sql) == "table_not_allowed"


@pytest.mark.parametrize(
    "sql",
    [
        # sqlglot's scope analysis treats these CTEs as visible in the outer query; in BigQuery they are not.
        "SELECT * FROM events WHERE EXISTS (WITH events AS (SELECT 1 AS x) (SELECT x FROM events) LIMIT 1)",
        "SELECT (WITH events AS (SELECT 1 AS x) (SELECT x FROM events) ORDER BY x LIMIT 1) AS v FROM events",
        "SELECT * FROM (WITH events AS (SELECT 1 AS x) (SELECT x FROM events) LIMIT 1) CROSS JOIN events",
        "SELECT x FROM events UNION ALL (WITH events AS (SELECT 1 AS x) (SELECT x FROM events) LIMIT 1)",
        # Here it does not look at the WITH clause at all, so the CTE would pass for a real table of that name.
        "SELECT * FROM (WITH recent AS (SELECT 1 AS x) (SELECT x FROM recent) LIMIT 1) AS a",
        "WITH a AS (WITH recent AS (SELECT 1 AS x) (SELECT x FROM recent) LIMIT 1) SELECT * FROM a",
        "SELECT * FROM UNNEST((WITH recent AS (SELECT [1] AS x) (SELECT x FROM recent) LIMIT 1)) AS v",
    ],
)
def test_with_clause_in_front_of_a_parenthesised_query_is_rejected_inside_a_larger_query(sql):
    assert rejection(sql) == "unsupported_query"


@pytest.mark.parametrize(
    "sql",
    [
        # Pipe syntax `|> AS name`: sqlglot writes the name after FROM, but the tree holds an alias there, no table.
        "FROM orders |> AS `bigquery-public-data.thelook_ecommerce.users` |> SELECT *",
        "FROM orders |> AS `bigquery-public-data.thelook_ecommerce.events`",
        "SELECT COUNT(*) FROM users WHERE EXISTS (FROM orders |> AS `INFORMATION_SCHEMA.SCHEMATA`)",
        "FROM orders |> AS recent |> WHERE recent.status = 'Complete'",
        # The path after LATERAL is no table in the tree either, with or without a join in front of it.
        "SELECT * FROM orders, LATERAL thelook_ecommerce.events",
        "SELECT * FROM orders CROSS JOIN (LATERAL thelook_ecommerce.events)",
        "SELECT * FROM orders LATERAL thelook_ecommerce.events",
        # Harmless, but neither BigQuery nor one of the sources the guard knows how to check.
        "SELECT * FROM orders CROSS JOIN (VALUES (1, 2)) AS v",
    ],
)
def test_from_or_join_that_reads_from_anything_but_a_table_a_query_or_unnest_is_rejected(sql):
    for user in (ALICE, ADMIN):
        assert rejection(sql, user) == "unsupported_query"


def test_table_reference_that_belongs_to_no_scope_is_rejected():
    assert rejection("SELECT * FROM (events) AS e") == "unsupported_query"
    assert rejection("SELECT * FROM (orders) AS o") == "unsupported_query"  # even an allowed table: it was not analysed


def test_failure_of_the_scope_analysis_rejects_the_query(monkeypatch):
    def broken(tree):
        raise RuntimeError("cannot analyse")

    monkeypatch.setattr(sql_guard, "traverse_scope", broken)
    assert rejection("SELECT COUNT(*) FROM orders") == "unsupported_query"


@pytest.mark.parametrize(
    "sql",
    [
        # A qualified name is the real table, whatever the query calls its CTEs.
        "WITH events AS (SELECT 1 AS x) SELECT * FROM thelook_ecommerce.events",
        "WITH events AS (SELECT 1 AS x) SELECT * FROM `bigquery-public-data.thelook_ecommerce.events`",
        # A name with a dot or a wildcard can be a path to a real table, so a CTE of that name does not cover it.
        "WITH `INFORMATION_SCHEMA.TABLES` AS (SELECT 1 AS x) SELECT * FROM INFORMATION_SCHEMA.TABLES",
        "WITH `events_*` AS (SELECT 1 AS x) SELECT * FROM `events_*`",
        # sqlglot takes what is attached to a dash for one name and writes it back unquoted: two tables to BigQuery.
        "WITH `a-b,events` AS (SELECT 1 AS x) SELECT * FROM a-b,events",
        # Names are compared exactly, which is at least as strict as BigQuery.
        "WITH Monthly AS (SELECT 1 AS x) SELECT * FROM monthly",
    ],
)
def test_only_a_bare_plain_name_spelled_like_the_cte_refers_to_it(sql):
    assert rejection(sql) == "table_not_allowed"


@pytest.mark.parametrize(
    "sql, model_ctes, views",
    [
        (
            "WITH placed AS (SELECT user_id FROM orders), buyers AS (SELECT DISTINCT user_id FROM placed) "
            "SELECT COUNT(*) FROM buyers",
            ["placed", "buyers"],
            ["products", "order_items", "orders"],
        ),
        (
            "WITH placed AS (SELECT user_id FROM orders) SELECT COUNT(*) FROM users WHERE id IN "
            "(WITH buyers AS (SELECT user_id FROM placed) SELECT user_id FROM buyers)",
            ["placed"],
            ["products", "order_items", "orders", "users"],
        ),
        (
            "SELECT COUNT(*) FROM users AS u WHERE EXISTS (SELECT 1 FROM orders AS o WHERE o.user_id = u.id)",
            [],
            ["products", "order_items", "orders", "users"],
        ),
        (
            "SELECT user_id FROM orders UNION ALL SELECT id FROM users",
            [],
            ["products", "order_items", "orders", "users"],
        ),
        (
            "SELECT day, COUNT(o.order_id) FROM UNNEST(GENERATE_DATE_ARRAY('2026-01-01', '2026-01-31')) AS day "
            "LEFT JOIN orders AS o ON DATE(o.created_at) = day GROUP BY day",
            [],
            ["products", "order_items", "orders"],
        ),
        (
            "SELECT * FROM (SELECT order_id, user_id FROM thelook_ecommerce.orders) AS o WHERE EXISTS "
            "(SELECT 1 FROM `bigquery-public-data.thelook_ecommerce.users` AS u WHERE u.id = o.user_id)",
            [],
            ["products", "order_items", "orders", "users"],
        ),
        (
            "SELECT * FROM (SELECT COUNT(*) AS placed FROM orders) CROSS JOIN (SELECT COUNT(*) AS buyers FROM users)",
            [],
            ["products", "order_items", "orders", "users"],
        ),
        (
            "SELECT COUNT(*) FROM (orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id) "
            "LEFT JOIN users AS u ON u.id = o.user_id",
            [],
            ["products", "order_items", "orders", "users"],
        ),
        (
            "SELECT * FROM (SELECT status, num_of_item FROM orders) "
            "PIVOT(SUM(num_of_item) FOR status IN ('Complete', 'Cancelled'))",
            [],
            ["products", "order_items", "orders"],
        ),
        (
            "SELECT user_id FROM orders AS o "
            "WHERE EXISTS (SELECT 1 FROM order_items AS oi WHERE oi.order_id = o.order_id) "
            "QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) = 1",
            [],
            ["products", "order_items", "orders"],
        ),
    ],
    ids=[
        "cte chain",
        "inner WITH uses outer CTE",
        "correlated subquery",
        "union",
        "unnest",
        "qualified names",
        "unaliased subqueries",
        "parenthesised join",
        "pivot",
        "exists with qualify",
    ],
)
def test_legitimate_shapes_pass_and_every_table_reference_is_redirected_to_a_view(sql, model_ctes, views):
    out = guard(sql, ALICE)
    assert top_level_ctes(out) == views + model_ctes
    assert sorted(qualified_tables(out)) == sorted(views)  # real tables appear only inside the views


def test_restricted_orders_view_counts_items_from_the_scoped_order_items_view():
    orders = view_body(guard("SELECT AVG(num_of_item) FROM orders", ALICE), "orders")
    assert orders.named_selects == list(SCHEMA["orders"])  # same columns, same order

    raw_orders = next(table for table in orders.find_all(exp.Table) if table.db)
    counted = orders.find(exp.Subquery)
    count = next(select for select in counted.this.selects if select.alias == "num_of_item")
    assert isinstance(count.this, exp.Count) and isinstance(count.this.this, exp.Star)
    assert [(table.name, table.db) for table in counted.this.find_all(exp.Table)] == [("order_items", "")]  # the view
    assert [group.name for group in counted.this.args["group"].expressions] == ["order_id"]

    # num_of_item is taken from that count and never from the raw table.
    assert counted.alias != raw_orders.alias
    assert {column.table for column in orders.find_all(exp.Column) if column.name == "num_of_item"} == {counted.alias}

    # An inner join on the order: orders without an item in scope drop out.
    (join,) = orders.args["joins"]
    assert join.this is counted and not join.side
    assert {column.name for column in join.args["on"].find_all(exp.Column)} == {"order_id"}


def test_mixed_order_shows_a_restricted_user_only_the_items_in_scope():
    tables = {
        "products": [row("products", id=1, brand="Brand A"), row("products", id=2, brand="Brand B")],
        "order_items": [
            row("order_items", id=10, order_id=100, product_id=1),
            row("order_items", id=11, order_id=100, product_id=2),
            row("order_items", id=12, order_id=100, product_id=2),
            row("order_items", id=13, order_id=101, product_id=2),
        ],
        "orders": [row("orders", order_id=100, num_of_item=3), row("orders", order_id=101, num_of_item=1)],
    }
    sql = "SELECT order_id, num_of_item FROM orders ORDER BY order_id"
    assert run_offline(guard(sql, ALICE), **tables) == [(100, 1)]  # not 3, and the Brand B-only order is not hers
    assert run_offline(guard(sql, BOB), **tables) == [(100, 2), (101, 1)]
    assert run_offline(guard(sql, ADMIN), **tables) == [(100, 3), (101, 1)]  # unrestricted: the raw count


def test_unrestricted_user_views_stay_plain_projections_of_the_tables():
    out = guard(
        "SELECT COUNT(*) FROM orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id "
        "JOIN products AS p ON p.id = oi.product_id JOIN users AS u ON u.id = o.user_id",
        ADMIN,
    )
    for table, columns in SCHEMA.items():
        assert view_body(out, table).sql(dialect="bigquery") == f"SELECT {', '.join(columns)} FROM `{DATASET}.{table}`"


@pytest.mark.parametrize("user", [ALICE, ADMIN])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, age FROM users",
        "SELECT user_id, gender FROM orders",
        "SELECT TO_JSON_STRING(u) FROM users AS u",
        "SELECT ARRAY_AGG(STRUCT(id, age)) FROM users",
        "WITH people AS (SELECT id AS customer, age AS years FROM users) SELECT customer, years FROM people",
        "SELECT age FROM users WHERE id = 123",
        "SELECT AVG(age) FROM users WHERE id = 123",
        "SELECT age, COUNT(IF(id = 123, 1, NULL)) FROM users GROUP BY age",
        "SELECT age, COUNT(*) OVER () FROM users",
        "SELECT age FROM users UNION ALL SELECT age FROM users",
        "SELECT age, SUM(id) FROM users GROUP BY age",
    ],
)
def test_individual_demographic_profiles_are_blocked(sql, user):
    assert rejection(sql, user) == "customer_privacy"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT age, COUNT(*) FROM users GROUP BY age",
        "SELECT state, COUNT(DISTINCT id) FROM users GROUP BY state",
        "SELECT u.state, SUM(oi.sale_price), COUNT(DISTINCT u.id) FROM users u "
        "JOIN order_items oi ON oi.user_id = u.id GROUP BY u.state",
        "SELECT u.id, SUM(oi.sale_price) FROM users u JOIN order_items oi ON oi.user_id = u.id GROUP BY u.id",
    ],
)
def test_aggregate_demographics_and_separate_customer_spend_are_allowed(sql):
    assert "brand IN ('Brand A')" in guard(sql, ALICE)


@pytest.mark.parametrize("user", [ALICE, ADMIN])
def test_churn_cohort_can_be_aggregated_by_traffic_source(user):
    sql = """
        WITH purchases AS (
          SELECT DISTINCT user_id AS customer, DATE(created_at) AS purchased
          FROM order_items WHERE status NOT IN ('Cancelled', 'Returned')
        ), gaps AS (
          SELECT customer, purchased,
            LEAD(purchased) OVER (PARTITION BY customer ORDER BY purchased) AS next_purchase
          FROM purchases
        ), churned AS (
          SELECT customer, DATE_ADD(purchased, INTERVAL 90 DAY) AS churn_date FROM gaps
          WHERE (next_purchase IS NULL OR DATE_DIFF(next_purchase, purchased, DAY) > 90)
            AND DATE_ADD(purchased, INTERVAL 90 DAY) >= DATE('2026-10-01')
            AND DATE_ADD(purchased, INTERVAL 90 DAY) < DATE('2026-11-01')
        )
        SELECT u.traffic_source, COUNT(DISTINCT c.customer) AS churned_customers
        FROM churned c JOIN users u ON u.id = c.customer
        GROUP BY u.traffic_source ORDER BY churned_customers DESC
    """
    assert "traffic_source" in guard(sql, user)


@pytest.mark.parametrize("user", [ALICE, ADMIN])
@pytest.mark.parametrize("projection", ["c.customer", "SUM(c.customer)", "ARRAY_AGG(c.customer)"])
def test_cohort_alias_cannot_expose_identified_demographics(user, projection):
    sql = f"""
        WITH cohort AS (SELECT user_id AS customer FROM order_items)
        SELECT u.traffic_source, {projection}
        FROM cohort c JOIN users u ON u.id = c.customer GROUP BY u.traffic_source, c.customer
    """
    assert rejection(sql, user) == "customer_privacy"


@pytest.mark.parametrize("user", [ALICE, ADMIN])
def test_cohort_cannot_filter_to_an_individual_before_demographic_aggregation(user):
    sql = """
        WITH cohort AS (SELECT user_id AS customer FROM order_items WHERE user_id = 123)
        SELECT u.traffic_source, COUNT(*)
        FROM cohort c JOIN users u ON u.id = c.customer GROUP BY u.traffic_source
    """
    assert rejection(sql, user) == "customer_privacy"


def test_union_cannot_hide_identifiers_in_another_demographic_branch():
    sql = "SELECT age, COUNT(*) FROM users GROUP BY age UNION ALL SELECT user_id, COUNT(*) FROM order_items GROUP BY user_id"
    assert rejection(sql) == "customer_privacy"


@pytest.mark.parametrize("user", [ALICE, ADMIN])
@pytest.mark.parametrize(
    "sql",
    [
        "WITH target AS (SELECT 123 AS customer) SELECT AVG(u.age) FROM users u JOIN target t ON u.id=t.customer",
        "SELECT AVG(u.age) FROM users u JOIN products p ON u.id=p.id",
        "WITH c AS (SELECT user_id AS x, order_id AS x FROM order_items UNION ALL SELECT id,id FROM users) SELECT COUNT(*) FROM c",
        "WITH c AS (SELECT age FROM users) SELECT * FROM c",
        "WITH c AS (SELECT age, COUNT(*) OVER () AS n FROM users) SELECT * FROM c",
        "WITH c AS (SELECT TO_HEX(SHA256(CAST(user_id AS STRING))) AS customer FROM order_items) SELECT c.customer,u.age,COUNT(*) FROM c CROSS JOIN users u GROUP BY c.customer,u.age",
    ],
)
def test_privacy_corner_cases_fail_closed(sql, user):
    assert rejection(sql, user) == "customer_privacy"


@pytest.mark.parametrize("user", [ALICE, ADMIN])
@pytest.mark.parametrize(
    "sql",
    [
        "WITH a AS (SELECT traffic_source,COUNT(*) AS n FROM users GROUP BY traffic_source) SELECT * FROM a ORDER BY n DESC LIMIT 3",
        "WITH a AS (SELECT traffic_source,COUNT(*) AS n FROM users GROUP BY traffic_source), b AS (SELECT * FROM a WHERE n>2) SELECT * FROM b",
        "WITH c AS (SELECT user_id AS customer FROM order_items UNION ALL SELECT id FROM users) SELECT u.traffic_source,COUNT(DISTINCT c.customer) FROM c JOIN users u ON c.customer=u.id GROUP BY u.traffic_source",
        "WITH c AS (SELECT user_id AS id FROM order_items) SELECT traffic_source,COUNT(*) FROM c JOIN users USING(id) GROUP BY traffic_source",
        "WITH c AS (SELECT user_id,MAX(created_at) AS last_purchase FROM order_items GROUP BY user_id) SELECT traffic_source,COUNT(*) FROM c JOIN users u ON c.user_id=u.id WHERE c.last_purchase<TIMESTAMP('2026-01-01') GROUP BY traffic_source",
    ],
)
def test_aggregate_cohort_corner_cases_are_allowed(sql, user):
    assert "users" in guard(sql, user)


@pytest.mark.parametrize("user", [ALICE, ADMIN])
def test_identifier_cohort_membership_allows_aggregate_traffic_sources(user):
    sql = "SELECT traffic_source,COUNT(*) FROM users WHERE id IN (SELECT user_id FROM order_items WHERE status='Complete') GROUP BY traffic_source"
    assert "traffic_source" in guard(sql, user)


@pytest.mark.parametrize(
    "subquery",
    ["SELECT 123", "SELECT user_id FROM order_items WHERE user_id=123", "SELECT COUNT(user_id) FROM order_items"],
)
def test_membership_cannot_hide_an_individual_filter(subquery):
    assert rejection(f"SELECT AVG(age) FROM users WHERE id IN ({subquery})") == "customer_privacy"
