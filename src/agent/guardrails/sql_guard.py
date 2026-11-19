"""Deterministic SQL guard: read-only, table allow-list, user scope and PII.

The model writes plain SQL against four bare table names. Before execution this
module (1) validates the statement and (2) prepends code-generated CTEs that shadow
the tables it uses with scoped, PII-free versions. The model's query can therefore only
ever see rows the user is entitled to and columns that are not PII, regardless of
what the prompt or a prompt injection says.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from agent.config import DATASET, User
from agent.guardrails.pii import PII_COLUMNS
from agent.warehouse.schema import SCHEMA, SCOPE_KEYS

SAFE_COLUMNS: dict[str, tuple[str, ...]] = {table: tuple(cols) for table, cols in SCHEMA.items()}
_CATALOG, _DB = DATASET.split(".")

# A name made of letters, digits and underscores. Only such a name may refer to a CTE. BigQuery
# can read anything else (a dot, a wildcard) as a path to a real table, and sqlglot takes what
# is attached to a dash for part of the name and writes it back unquoted: `a-b,events` is one
# name to sqlglot and two tables to BigQuery.
_PLAIN_NAME = re.compile(r"\w+")

_UNSUPPORTED = (
    "This query's structure is not supported. Use plain SELECT statements, joins, subqueries and CTEs, "
    "and do not put parentheses around a single table or around the query that follows a WITH clause."
)

_FORBIDDEN = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,
    exp.Into,
    exp.Set,
    exp.Use,
    exp.Transaction,
)


class GuardViolation(Exception):
    """The query is not allowed. `kind` is recorded in the trace."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _scoped_orders() -> str:
    """The orders view of a restricted user.

    The raw num_of_item counts every item of an order, including items outside the user's
    scope. The view takes it from a count over the scoped order_items view instead, and the
    join keeps only the orders that have such an item.
    """
    columns = []
    for column in SAFE_COLUMNS["orders"]:
        source = "scoped_items" if column == "num_of_item" else "raw_orders"
        columns.append(f"{source}.{column}")
    return (
        f"SELECT {', '.join(columns)} FROM `{DATASET}.orders` AS raw_orders "
        "JOIN (SELECT order_id, COUNT(*) AS num_of_item FROM order_items GROUP BY order_id) AS scoped_items "
        "ON scoped_items.order_id = raw_orders.order_id"
    )


def _scoped_ctes(user: User, used: set[str]) -> list[tuple[str, str]]:
    """The guard's views for the tables a query uses, in dependency order.

    products     the user's products
    order_items  items of those products
    orders       orders that contain such items; num_of_item counts only those items
    users        customers who bought such items, without PII columns

    An unrestricted user gets all rows of every table, still without the PII columns.
    """

    def view(table: str, where: str = "") -> str:
        return f"SELECT {', '.join(SAFE_COLUMNS[table])} FROM `{DATASET}.{table}`" + (
            f" WHERE {where}" if where else ""
        )

    filters = []
    for col, values in user.scope.items():
        if col not in SCOPE_KEYS:
            raise ValueError(f"Unsupported scope key: {col}")
        literals = ", ".join(exp.Literal.string(v).sql(dialect="bigquery") for v in values)
        filters.append(f"{col} IN ({literals})" if values else "FALSE")  # an empty list grants nothing
    if not filters:  # unrestricted user: column projection only
        return [(table, view(table)) for table in SAFE_COLUMNS if table in used]

    needed = set(used)
    if needed & {"orders", "users"}:
        needed.add("order_items")
    if "order_items" in needed:
        needed.add("products")
    views = {
        "products": view("products", " AND ".join(filters)),
        "order_items": view("order_items", "product_id IN (SELECT id FROM products)"),
        "orders": _scoped_orders(),
        "users": view("users", "id IN (SELECT user_id FROM order_items)"),
    }
    return [(table, body) for table, body in views.items() if table in needed]


def _is_cte_reference(table: exp.Table, scope: Scope) -> bool:
    """Whether `table` names a CTE that is visible in `scope`, the scope the reference is written in.

    Only a bare, plain name can. A qualified name is always a real table, and sqlglot keeps
    `INFORMATION_SCHEMA.TABLES` as a single identifier although it is a path to a real view.
    """
    qualified = table.args.get("db") is not None or table.args.get("catalog") is not None
    plain = _PLAIN_NAME.fullmatch(table.name) is not None
    return not qualified and plain and isinstance(scope.cte_sources.get(table.name), Scope)


def _declared_by(cte: exp.CTE, scope: Scope) -> bool:
    """Whether `cte` belongs to the WITH clause of the scope's own query."""
    with_clause = cte.parent
    return with_clause.parent is scope.expression


def _is_checked_source(source: exp.Expression) -> bool:
    """Whether a FROM or JOIN reads from a table, from UNNEST or from a query.

    A table is checked by its name, and whatever UNNEST or a query reads is checked in turn.
    sqlglot can put other things there and write them as a name, which BigQuery would look
    up as a table although the tree holds no table for it: the alias that pipe syntax
    `|> AS name` leaves in FROM, or the path that follows LATERAL.
    """
    if isinstance(source, exp.Subquery):
        inside = source.unnest()  # what the parentheses hold, however many there are
        return isinstance(inside, (exp.Select, exp.SetOperation, exp.Table, exp.Unnest))
    return isinstance(source, (exp.Table, exp.Unnest))


def _real_tables(tree: exp.Expression) -> list[exp.Table]:
    """The table references in `tree` that name a real table, not one of the query's own CTEs.

    A CTE is visible only inside the query that declares it. Each reference is therefore
    judged in its own scope, with sqlglot's scope analysis, and never by name alone:
    otherwise a CTE declared in one subquery would vouch for a real table of the same name
    used outside it.

    Fails closed. The query is rejected when sqlglot cannot analyse it, when a table
    reference belongs to no scope, when a FROM or JOIN reads from something that is not
    checked here, and when a CTE was not analysed as part of the query whose WITH clause
    declares it. The last one is `WITH t AS (...) (SELECT ...) LIMIT 1` inside a larger
    query: sqlglot either makes `t` visible in the whole surrounding query or does not look
    at the WITH clause at all, while in BigQuery `t` exists in exactly the query the WITH
    belongs to.
    """
    try:
        scopes = traverse_scope(tree)
        placed = [(table, scope) for scope in scopes for table in scope.tables]
        declared_ids = {id(cte) for scope in scopes for cte in scope.ctes if _declared_by(cte, scope)}
    except Exception as error:  # nothing that sqlglot could not analyse may run
        raise GuardViolation("unsupported_query", _UNSUPPORTED) from error
    placed_ids = {id(table) for table, _ in placed}
    unplaced_tables = [table for table in tree.find_all(exp.Table) if id(table) not in placed_ids]
    undeclared_ctes = [cte for cte in tree.find_all(exp.CTE) if id(cte) not in declared_ids]
    sources = [clause.this for clause in tree.find_all(exp.From, exp.Join)]
    sources.extend(tree.find_all(exp.Lateral))  # sqlglot also hangs LATERAL off a table, without a JOIN
    unchecked_sources = [source for source in sources if not _is_checked_source(source)]
    if unplaced_tables or undeclared_ctes or unchecked_sources:
        raise GuardViolation("unsupported_query", _UNSUPPORTED)
    return [table for table, scope in placed if not _is_cte_reference(table, scope)]


def _guard_demographics(tree: exp.Expression, used: set[str]) -> None:
    """Demographics are group-level only, including for the CEO.

    Qualify a copy to expand stars and resolve source aliases. In a demographic query,
    identifier-only cohort CTEs may carry keys internally. In demographic scopes,
    identifiers may only join tables or count customers, never expose profiles or
    select an individual. Resolve aliases so CTEs cannot bypass that restriction.
    """
    if not used & {"users", "orders"}:
        return
    message = (
        "Customer demographics are aggregate-only. Do not combine age, gender, location or other "
        "customer attributes with an individual ID, or filter to an individual. Use a grouped "
        "aggregate query; cohort CTEs may carry IDs internally, but demographic scopes must use IDs "
        "only in joins, identifier-cohort IN subqueries or COUNT. For traffic-source breakdowns, build an "
        "identifier-only cohort, then join users in the final SELECT and GROUP BY traffic_source. "
        "Aggregate traffic-source analysis is allowed. Query customer spend separately without demographics."
    )
    try:
        checked = qualify(tree.copy(), dialect="bigquery", schema=SCHEMA, quote_identifiers=False)
        scopes = traverse_scope(checked)
    except Exception as error:
        raise GuardViolation("customer_privacy", message) from error

    if checked.find(exp.TableColumn):
        raise GuardViolation("customer_privacy", message)

    # Resolve identifier lineage through cohort CTEs and aliases. An internal
    # customer key is permitted; projecting it beside demographics is not.
    identity, raw_demographic, aggregate_demographic = 1, 2, 4
    demographic = raw_demographic | aggregate_demographic
    outputs: dict[int, dict[str, int]] = {}
    columns: dict[int, int] = {}

    def flags(node):
        if isinstance(node, exp.Column):
            return columns.get(id(node), 0)
        if isinstance(node, exp.Window):
            # Partition/order keys do not become the window's returned value.
            return flags(node.this)
        if isinstance(node, exp.Count):
            argument = node.this
            if isinstance(argument, exp.Column) or (
                isinstance(argument, exp.Distinct)
                and len(argument.expressions) == 1
                and isinstance(argument.expressions[0], exp.Column)
            ):
                return flags(argument) & demographic
        result = 0
        for child in node.iter_expressions():
            result |= flags(child)
        return result

    for scope in scopes:
        for column in scope.columns:
            owner = scope
            source = owner.sources.get(column.table)
            while source is None and owner.parent is not None:
                owner = owner.parent
                source = owner.sources.get(column.table)
            value = 0
            if isinstance(source, exp.Table):
                table, name = source.name.lower(), column.name.lower()
                if (table == "users" and name != "id") or (table == "orders" and name == "gender"):
                    value |= raw_demographic
                if (table == "users" and name == "id") or (
                    table in {"orders", "order_items"} and name in {"id", "user_id", "order_id", "inventory_item_id"}
                ):
                    value |= identity
            elif isinstance(source, Scope):
                if id(source) not in outputs or column.name not in outputs[id(source)]:
                    raise GuardViolation("customer_privacy", message)
                value = outputs[id(source)][column.name]
            columns[id(column)] = value
        names = scope.expression.named_selects
        if len(names) != len(set(names)):
            raise GuardViolation("customer_privacy", message)
        if scope.set_operation_scopes:
            branches = [list(outputs[id(branch)].values()) for branch in scope.set_operation_scopes]
            if any(len(branch) != len(names) for branch in branches):
                raise GuardViolation("customer_privacy", message)
            outputs[id(scope)] = {name: branches[0][i] | branches[1][i] for i, name in enumerate(names)}
        else:
            grouped = scope.expression.args.get("group") is not None
            aggregated = any(
                aggregate.find_ancestor(exp.Select) is scope.expression and not aggregate.find_ancestor(exp.Window)
                for aggregate in scope.expression.find_all(exp.AggFunc)
            )
            result = {item.alias_or_name: flags(item) for item in scope.expression.selects}
            # Preserve that these attributes already describe groups when an
            # outer query sorts, limits or presents an aggregate CTE.
            if grouped or aggregated:
                result = {
                    name: (value & ~raw_demographic) | (aggregate_demographic if value & raw_demographic else 0)
                    for name, value in result.items()
                }
            outputs[id(scope)] = result

    if not any(value & demographic for value in columns.values()):
        return

    if any(value & (identity | raw_demographic) for value in outputs[id(scopes[-1])].values()):
        raise GuardViolation("customer_privacy", message)

    for scope in scopes:
        has_demographics = any(columns.get(id(col), 0) & demographic for col in scope.columns)
        for column in scope.columns:
            if not columns.get(id(column), 0) & identity:
                continue
            ancestor = column.parent
            permitted = False
            while ancestor is not None and ancestor is not scope.expression:
                if isinstance(ancestor, exp.Count):
                    argument = ancestor.this
                    permitted = argument is column or (
                        isinstance(argument, exp.Distinct) and argument.expressions == [column]
                    )
                    break
                if (
                    isinstance(ancestor, exp.EQ)
                    and all(
                        isinstance(side, exp.Column) and columns.get(id(side), 0) & identity
                        for side in (ancestor.this, ancestor.expression)
                    )
                    and ancestor.find_ancestor(exp.Join)
                ):
                    permitted = True
                    break
                if isinstance(ancestor, exp.In) and ancestor.this is column:
                    query = ancestor.args.get("query")
                    target = query.this if isinstance(query, exp.Subquery) else query
                    cohort = next((item for item in scopes if item.expression is target), None)
                    permitted = cohort is not None and list(outputs[id(cohort)].values()) == [identity]
                    break
                # Never allow a cohort to select an identified customer, even
                # when that cohort is later joined to an aggregate breakdown.
                if isinstance(ancestor, (exp.Where, exp.Having, exp.Qualify, exp.Join)):
                    break
                ancestor = ancestor.parent
            else:
                # Only identifier-only intermediate scopes may carry keys or
                # group/partition by them. Demographic scopes remain aggregate-only.
                permitted = not has_demographics and scope is not scopes[-1]
            if not permitted:
                raise GuardViolation("customer_privacy", message)


def guard(sql: str, user: User) -> str:
    """Validate `sql` and return the scoped, PII-safe query to execute."""
    try:
        # A trailing semicolon or comment parses as an empty statement; ignore those.
        statements = [
            s for s in sqlglot.parse(sql, read="bigquery") if s is not None and not isinstance(s, exp.Semicolon)
        ]
    except SqlglotError as e:
        raise GuardViolation("parse_error", f"SQL could not be parsed: {e}") from e
    if len(statements) != 1:
        raise GuardViolation("not_single_statement", "Exactly one SQL statement is allowed.")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.SetOperation)) or tree.find(*_FORBIDDEN):
        raise GuardViolation("not_read_only", "Only read-only SELECT queries are allowed.")

    with_ = tree.args.get("with_") or tree.args.get("with")
    # At any depth: the visibility rules that _real_tables relies on hold for non-recursive CTEs only.
    if any(clause.args.get("recursive") for clause in tree.find_all(exp.With)):
        raise GuardViolation("recursive_cte", "Recursive CTEs are not allowed.")

    model_ctes = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}
    if clash := model_ctes & SAFE_COLUMNS.keys():
        raise GuardViolation("reserved_name", f"CTE names {sorted(clash)} are reserved table names; rename them.")

    if any(not isinstance(table.this, exp.Identifier) for table in tree.find_all(exp.Table)):
        raise GuardViolation("table_function", "Table functions are not allowed.")

    used: set[str] = set()
    for table in _real_tables(tree):
        name, db, catalog = table.name.lower(), table.db, table.catalog
        qualified = bool(db or catalog)
        in_dataset = not qualified or (db == _DB and catalog in ("", _CATALOG))
        if name not in SAFE_COLUMNS or not in_dataset:
            raise GuardViolation(
                "table_not_allowed",
                f"Table '{table.sql(dialect='bigquery')}' is not available. Use only: {', '.join(SAFE_COLUMNS)}.",
            )
        # Point the reference at the scoped CTE of the same name.
        table.set("this", exp.to_identifier(name))
        table.set("db", None)
        table.set("catalog", None)
        used.add(name)

    for column in tree.find_all(exp.Column):
        if column.name.lower() in PII_COLUMNS:
            raise GuardViolation(
                "pii_column",
                f"Column '{column.name}' is personal data and cannot be queried. Use users.id and aggregates instead.",
            )

    _guard_demographics(tree, used)

    if views := _scoped_ctes(user, used):
        carrier = exp.select("1")
        for name, body in views:
            carrier = carrier.with_(name, as_=body, dialect="bigquery")
        scoped_with = carrier.args.get("with_") or carrier.args.get("with")
        if with_ is not None:
            with_.set("expressions", scoped_with.expressions + with_.expressions)
        else:
            tree.set("with_" if "with_" in exp.Select.arg_types else "with", scoped_with)

    return tree.sql(dialect="bigquery")
