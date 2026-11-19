"""Agent tools.

A tool is a pydantic args model (its docstring and field descriptions are what the
model sees) plus a handler, registered with @tool. Adding a capability such as
charts, e-mail or web search means adding one more entry here; the graph and the
CLI do not change.

Handlers never raise and never touch the UI. They return a ToolResult whose text
goes back to the model, so every failure is something the model can react to.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError

from agent.config import User, settings
from agent.guardrails.grounding import remove_unsupported, replace_with_marker, ungrounded_figures
from agent.guardrails.pii import mask_dataframe, scan_text
from agent.guardrails.sql_guard import GuardViolation, guard
from agent.observability import Trace, loggable
from agent.stores.reports import ReportStore
from agent.warehouse.bq_client import BigQueryRunner, BQNotConfiguredError, BQUnavailableError, SQLSyntaxError

log = logging.getLogger(__name__)


@dataclass
class ToolContext:
    user: User
    thread_id: str
    trace: Trace
    bq: BigQueryRunner
    reports: ReportStore
    span: dict  # this call's trace span, for the handler to enrich
    # What the turn has used so far, earlier calls of the same step included:
    sql_failures: int = 0  # failed queries
    empty_results: int = 0  # queries that returned no data
    sql_queries: int = 0  # queries sent to BigQuery
    bytes_scanned: int = 0  # by those queries
    grounding_checked: bool = False  # the turn's one repair round for unsupported figures was used before this step
    # What the conversation holds, earlier calls of the same step included:
    evidence: list[str] = field(default_factory=list)  # query results: what a figure may be taken from
    user_texts: list[str] = field(default_factory=list)  # the user's own messages

    @property
    def known(self) -> list[str]:
        """The texts a name in model-written text may come from: query results and the user's own words."""
        return [*self.evidence, *self.user_texts]


@dataclass
class ToolResult:
    content: str = ""
    sql_failed: bool = False  # counts against the per-turn repair budget
    stop_sql: bool = False  # no further queries this turn (warehouse is down)
    empty: bool = False
    sql_sent: bool = False  # the query went to BigQuery: counts against the per-turn query cap
    bytes_scanned: int = 0  # by that query: counts against the per-turn scan cap
    evidence: bool = False  # rows of a successful query: the only text a figure may be taken from
    grounding_checked: bool = False  # this call used the turn's one repair round for unsupported figures
    pending_delete: dict | None = None  # deletion that must be confirmed by the user


@dataclass(frozen=True)
class Tool:
    name: str
    args: type[BaseModel]
    handler: Callable[[ToolContext, BaseModel], ToolResult]
    label: str  # progress text shown while the tool runs


TOOLS: dict[str, Tool] = {}


def tool(name: str, args: type[BaseModel], label: str):
    def register(handler):
        TOOLS[name] = Tool(name, args, handler, label)
        return handler

    return register


def tool_schemas() -> list[StructuredTool]:
    """Tool declarations for the model. Execution goes through `execute`."""
    return [
        StructuredTool(
            name=t.name, description=" ".join((t.args.__doc__ or "").split()), args_schema=t.args, func=lambda **_: None
        )
        for t in TOOLS.values()
    ]


def execute(name: str, raw_args: dict | None, ctx: ToolContext) -> ToolResult:
    """Run one tool call. Never raises."""
    spec = TOOLS.get(name)
    if spec is None:
        ctx.span.update(outcome="unknown_tool")
        return ToolResult(f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS)}.")
    try:
        args = spec.args.model_validate(raw_args or {})
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        ctx.span.update(outcome="bad_arguments", error=loggable(problems, 300))
        return ToolResult(f"Invalid arguments for {name}: {problems}")
    try:
        return spec.handler(ctx, args)
    except Exception as e:  # a tool bug must not crash the turn
        log.exception("Tool %s failed", name)
        ctx.span.update(status="error", outcome="internal_error", error=loggable(f"{type(e).__name__}: {e}", 300))
        ctx.trace.mark("degraded")
        return ToolResult(
            f"The {name} tool failed unexpectedly. Do not retry it; tell the user this step could not be completed."
        )


# --------------------------------------------------------------------------- data


class RunSQL(BaseModel):
    """Run one read-only BigQuery Standard SQL SELECT and get the rows back as CSV
    (at most 200 rows, so aggregate in SQL). Use only the tables orders, order_items,
    products and users by their bare names. Personal-data columns do not exist. The
    data is already limited to the products the current user is responsible for."""

    sql: str = Field(description="A single SELECT statement. CTEs are allowed.")
    purpose: str = Field(default="", description="A few words on what this query is meant to find out.")


_POSITION = re.compile(r"\s+at \[\d+:\d+\]", re.I)  # positions refer to the rewritten SQL; they would mislead

MAX_EMPTY_RESULTS = 2  # an empty result gets one guided retry; a second one ends querying for the turn

_USE_WHAT_YOU_HAVE = "Answer with the results you already have and say plainly what could not be computed."

# The free text of a query, found without parsing it, because SQL that the guard rejected may not parse.
# A backticked name is matched only to be kept: a quote inside it does not start a literal.
_SQL_FREE_TEXT = re.compile(
    r"""
      (?P<name>`[^`\n]*`)
    | (?P<comment>--[^\n]*|\#[^\n]*|/\*.*?(?:\*/|\Z))
    | (?P<literal>'''(?:[^\\]|\\.)*?'''|\"\"\"(?:[^\\]|\\.)*?\"\"\"|'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
    | (?P<open_quote>['"].*)
    """,
    re.DOTALL | re.VERBOSE,
)


def _placeholder(free_text: re.Match[str]) -> str:
    if free_text.group("name") is not None:
        return free_text.group()
    if free_text.group("comment") is not None:
        return " "
    return "'?'"


def _without_literals_and_comments(sql: str) -> str:
    """A query with every string literal replaced by '?' and every comment removed.

    A quote that is never closed (an apostrophe in 'O'Brien', say) leaves it open which quotes
    belong together, so everything from the first quote on is then replaced.
    """
    free_texts = list(_SQL_FREE_TEXT.finditer(sql))
    if any(free_text.group("open_quote") for free_text in free_texts):
        quoted = [free_text for free_text in free_texts if free_text.group("literal") or free_text.group("open_quote")]
        return _SQL_FREE_TEXT.sub(_placeholder, sql[: quoted[0].start()]) + "'?'"
    return _SQL_FREE_TEXT.sub(_placeholder, sql)


def _unchecked_sql_for_trace(sql: str) -> str:
    """SQL that the guard did not accept, as it may be written to the trace.

    Nothing has checked what its literals and comments say, and a filter on a personal-data
    column carries the personal data in a literal. The trace keeps the structure only.
    """
    return scan_text(_without_literals_and_comments(sql), known=())[0]


def _attempts(left: int) -> str:
    if left > 0:
        return f"Fix the query and try again ({left} attempt(s) left for this question)."
    return "No attempts are left for this question: stop querying, tell the user what could not be computed and suggest a simpler question."


def _budget_refusal(ctx: ToolContext) -> str | None:
    """Why this turn may not send another query to BigQuery, or None if it may.

    Checked before the guard and the warehouse, for every call on its own. The limits therefore
    hold whatever the model does, several queries in one step included: each call sees what the
    earlier ones used.
    """
    if ctx.sql_failures >= settings.max_sql_failures:
        return "No more queries can be run for this question. " + _attempts(0)
    if ctx.empty_results >= MAX_EMPTY_RESULTS:
        return (
            f"No more queries can be run for this question: {MAX_EMPTY_RESULTS} queries have already returned no "
            "data. Stop querying and tell the user that no matching data was found within their data scope."
        )
    if ctx.sql_queries >= settings.max_queries_per_turn:
        return (
            f"No more queries can be run for this question: its limit of queries ({settings.max_queries_per_turn}) "
            f"is reached. {_USE_WHAT_YOU_HAVE}"
        )
    if ctx.bytes_scanned >= settings.bq_max_bytes_per_turn:
        return (
            "No more queries can be run for this question: its limit on the amount of data scanned is reached. "
            f"{_USE_WHAT_YOU_HAVE}"
        )
    return None


def _render(df: pd.DataFrame, truncated: bool) -> str:
    csv = df.round(4).to_csv(index=False)
    note = ""
    if len(csv) > settings.max_result_chars:
        csv = csv[: settings.max_result_chars].rsplit("\n", 1)[0] + "\n"
        note = " The output was cut for length; aggregate or filter further if more is needed."
    elif truncated:
        note = f" Only the first {settings.bq_max_rows} rows are shown; aggregate in SQL rather than listing rows."
    return f"{len(df)} row(s).{note}\n{csv}"


@tool("run_sql", RunSQL, "Querying the data warehouse")
def run_sql(ctx: ToolContext, args: RunSQL) -> ToolResult:
    span = ctx.span
    span["purpose"] = scan_text(args.purpose, known=())[0]
    if refusal := _budget_refusal(ctx):
        span.update(outcome="budget_exhausted", sql=_unchecked_sql_for_trace(args.sql))
        return ToolResult(refusal)

    left = settings.max_sql_failures - ctx.sql_failures
    try:
        executed_sql = guard(args.sql, ctx.user)
    except GuardViolation as e:
        # The guard's message quotes the query (the place where parsing failed, the table expression),
        # literals included, so the trace keeps the kind of violation and the model gets the message.
        span.update(outcome="rejected", guard=e.kind, sql=_unchecked_sql_for_trace(args.sql))
        ctx.trace.event("guard_block", guard=e.kind)
        return ToolResult(f"QUERY REJECTED ({e.kind}): {e} {_attempts(left - 1)}", sql_failed=True)

    span.update(sql=scan_text(args.sql)[0], executed_sql=scan_text(executed_sql)[0])
    result = _query_warehouse(ctx, executed_sql, left)
    result.sql_sent = True
    return result


def _query_warehouse(ctx: ToolContext, executed_sql: str, attempts_left: int) -> ToolResult:
    """Run a query that the guard accepted and describe the outcome to the model."""
    span = ctx.span
    try:
        outcome = ctx.bq.run(executed_sql, trace_id=ctx.trace.trace_id)
    except SQLSyntaxError as e:
        message = _POSITION.sub("", str(e))
        span.update(outcome="sql_error", error=loggable(message, 500))
        return ToolResult(f"SQL ERROR: {message} {_attempts(attempts_left - 1)}", sql_failed=True)
    except BQNotConfiguredError as e:
        span.update(outcome="not_configured", error=loggable(str(e), 500))
        ctx.trace.event("warehouse_not_configured", detail=loggable(str(e), 300))
        ctx.trace.mark("degraded")
        return ToolResult(
            "DATA WAREHOUSE NOT CONFIGURED. This installation has no BigQuery credentials, so no data can be queried and "
            "retrying will not help. Do not run more queries. Tell the user plainly that the data source has not been "
            "set up yet: whoever installed the assistant needs to add BigQuery credentials (see the README, section "
            "Setup) and restart it. Do not guess any numbers.",
            stop_sql=True,
        )
    except BQUnavailableError as e:
        span.update(outcome="unavailable", error=loggable(str(e), 500))
        ctx.trace.event("warehouse_unavailable", detail=loggable(str(e), 300))
        ctx.trace.mark("degraded")
        return ToolResult(
            "DATA WAREHOUSE UNAVAILABLE. Do not run more queries in this turn. Tell the user the data source cannot "
            "be reached right now and that they can try again shortly. Do not guess any numbers.",
            stop_sql=True,
        )

    df, masked = mask_dataframe(outcome.df)
    if masked:
        ctx.trace.event("pii_redacted", where="query_result", detail=masked)
    scanned = outcome.bytes_processed
    span.update(
        rows=len(df),
        columns=list(df.columns),
        bytes=scanned,
        job_id=outcome.job_id,
        bq_ms=outcome.duration_ms,
        truncated=outcome.truncated,
    )
    if df.empty or df.isna().all().all():  # no rows, or an aggregate over nothing (a single row of NULLs)
        span["outcome"] = "empty"
        if ctx.empty_results:
            return ToolResult(
                "The query returned no data again. Do not retry; tell the user no matching data was found within their data scope.",
                empty=True,
                bytes_scanned=scanned,
            )
        return ToolResult(
            "The query ran but returned no data (0 rows, or only NULLs). Likely causes: a filter value that does not exist (check spelling and "
            "case with a DISTINCT query), a date range without data, or data outside this user's product scope. You may "
            "run ONE corrected or diagnostic query; if it is still empty, tell the user plainly that nothing matched.",
            empty=True,
            bytes_scanned=scanned,
        )
    span["outcome"] = "ok"
    notes = [f"Removed for privacy: {event}." for event in masked if event.startswith("dropped columns")]
    if df.isna().any().any():
        notes.append("Blank cells are NULL (no data), not zero.")
    return ToolResult(_render(df, outcome.truncated) + "\n".join(notes), bytes_scanned=scanned, evidence=True)


# ------------------------------------------------------------------------ reports


class SaveReport(BaseModel):
    """Save a finished report to the user's Saved Reports library. Call this only
    when the user asks to create or save a report, after you have the numbers."""

    title: str = Field(description="Short descriptive title.")
    content: str = Field(description="The full report in Markdown: summary, key metrics, insights, action items.")


@tool("save_report", SaveReport, "Saving the report")
def save_report(ctx: ToolContext, args: SaveReport) -> ToolResult:
    if not args.title.strip() or not args.content.strip():
        return ToolResult("A report needs both a title and content. Write the full report and save it again.")
    title, content = args.title.strip()[:200], args.content

    # A saved report is shown to the user in full, so its figures are checked here, where it is saved.
    unsupported = ungrounded_figures(f"{title}\n{content}", ctx.evidence)
    if unsupported:
        ctx.trace.event(
            "ungrounded_figures", figures=unsupported, where="saved_report", after_repair=ctx.grounding_checked
        )
    if unsupported and not ctx.grounding_checked:
        # The turn's one repair round: nothing is saved, and the model is asked for a corrected report.
        ctx.span.update(outcome="not_saved", figures=unsupported)
        return ToolResult(
            "REPORT NOT SAVED. These figures are not in any query result of this conversation: "
            f"{', '.join(unsupported)}. Do not calculate them yourself. Either compute them with run_sql and use the "
            "returned values, or remove them. Then call save_report again with the complete report.",
            grounding_checked=True,
        )
    if unsupported:
        title, content = replace_with_marker(title, unsupported), remove_unsupported(content, unsupported)
        ctx.span.update(outcome="figures_removed", figures=unsupported)
        ctx.trace.event("figures_removed", figures=unsupported, where="saved_report")
        ctx.trace.mark("degraded")

    title, found_title = scan_text(title, known=ctx.known)
    content, found_body = scan_text(content, known=ctx.known)
    if found := found_title + found_body:
        ctx.trace.event("pii_redacted", where="saved_report", detail=found)
    report = ctx.reports.save(ctx.user.id, ctx.thread_id, title, content)
    ctx.trace.change(f"Saved report #{report.id} “{title}”")
    ctx.span.update(report_id=report.id, title=title)
    saved = f"Saved as report #{report.id}: {title}."
    if unsupported:
        saved += (
            f" {len(unsupported)} unverified figure(s) were removed from the saved text and are marked "
            "[figure removed] there; tell the user."
        )
    return ToolResult(f"{saved} The full report is now displayed to the user; do not repeat it.")


class ListReports(BaseModel):
    """List the current user's saved reports (id, title, creation time)."""

    mentions: str = Field(default="", description="Only reports whose title or text mentions this.")
    this_conversation: bool = Field(default=False, description="Only reports created in the current conversation.")


@tool("list_reports", ListReports, "Looking up saved reports")
def list_reports(ctx: ToolContext, args: ListReports) -> ToolResult:
    found = ctx.reports.find(
        ctx.user.id, thread_id=ctx.thread_id if args.this_conversation else None, mentions=args.mentions or None
    )
    ctx.span["count"] = len(found)
    if not found:
        return ToolResult("No saved reports match.")
    return ToolResult(
        "\n".join(f"#{report.id} | {report.title} | {report.summary()['created_at']} UTC" for report in found)
    )


class DeleteReports(BaseModel):
    """Request deletion of the user's saved reports. Nothing is deleted by this call:
    the system shows the user the exact list of matching reports and asks them to
    confirm. Give at least one criterion; several criteria are combined with AND."""

    report_ids: list[int] = Field(default_factory=list, description="Specific report ids.")
    mentions: str = Field(default="", description="Reports whose title or text mentions this, e.g. a client name.")
    this_conversation: bool = Field(default=False, description="Reports created in the current conversation.")
    all_reports: bool = Field(
        default=False, description="Every report the user owns. Only when the user explicitly asks for all."
    )


def _describe_criteria(args: DeleteReports) -> str:
    """The user's deletion criteria in words, for the confirmation dialog and the trace."""
    parts = []
    if args.report_ids:
        parts.append(f"ids {args.report_ids}")
    if args.mentions:
        parts.append(f"mentioning '{args.mentions}'")
    if args.this_conversation:
        parts.append("created in this conversation")
    return ", ".join(parts) or "all reports"


@tool("delete_reports", DeleteReports, "Finding the reports to delete")
def delete_reports(ctx: ToolContext, args: DeleteReports) -> ToolResult:
    if not (args.report_ids or args.mentions or args.this_conversation or args.all_reports):
        return ToolResult("Say which reports to delete: report_ids, mentions, this_conversation or all_reports.")
    matches = ctx.reports.find(
        ctx.user.id,
        thread_id=ctx.thread_id if args.this_conversation else None,
        mentions=args.mentions or None,
        ids=args.report_ids or None,
    )
    # The search term can be an e-mail address. The search uses it as typed; what is shown and logged is scanned.
    criteria = scan_text(_describe_criteria(args), known=ctx.known)[0]
    ctx.span.update(criteria=criteria, matched=[report.id for report in matches])
    if not matches:
        return ToolResult(f"The user has no saved reports matching: {criteria}. Nothing to delete.")
    return ToolResult(
        pending_delete={
            "criteria": criteria,
            "ids": [report.id for report in matches],
            "reports": [report.summary() for report in matches],
        }
    )
