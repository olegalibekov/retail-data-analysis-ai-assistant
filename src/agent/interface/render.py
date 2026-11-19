"""How things look in the terminal: panels, tables and the trace tree.

Pure presentation. Every function takes plain data and returns something Rich
can print; nothing here reads input, calls the agent or touches a store.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from agent.config import User

HELP = """\
[bold]Ask anything about sales, customers, products and orders[/], for example:
  - What was our revenue by month over the last year?
  - Why are customers in Texas underspending compared with California?
  - Create a report for Q1 with insights and action items for Q2.
  - Delete all the reports we made in this conversation.

[bold]Commands[/]
  /trace \\[id]    what happened in the last turn (or in a given trace)
  /metrics       agent metrics for this session
  /reports       your saved reports
  /new           start a new conversation
  /help  /quit"""


_STATUS_STYLE = {"ok": "green", "blocked": "yellow", "cancelled": "yellow", "degraded": "yellow", "error": "red"}
_FAILED_OUTCOMES = ("sql_error", "rejected", "unavailable", "not_configured", "internal_error")
_PREVIEW_ROWS = 30


# ------------------------------------------------------------------- conversation


def banner(user: User, thread_id: str) -> Panel:
    return Panel(
        f"Signed in as [bold]{user.name}[/] · data scope: [bold]{escape(user.describe_scope())}[/]\n"
        f"Conversation {thread_id} · type [bold]/help[/] for examples and commands",
        title="Retail data analysis assistant",
        border_style="cyan",
    )


def answer_panel(answer: str) -> Panel:
    return Panel(Markdown(answer), title="assistant", title_align="left", border_style="cyan")


def report_panel(report) -> Panel:
    title = f"report #{report.id} · {escape(report.title)}"
    return Panel(Markdown(report.body), title=title, title_align="left", border_style="green")


def turn_footer(record: dict) -> str:
    """The dim line under every answer: trace id, effort, time, and the status if it was not ok."""
    queries = "query" if record["sql_queries"] == 1 else "queries"
    line = (
        f"[dim]trace {record['trace_id']} · {record['model_calls']} model call(s) · "
        f"{record['sql_queries']} {queries} · {record['duration_ms'] / 1000:.1f}s[/]"
    )
    if record["status"] != "ok":
        line += f" · [{_STATUS_STYLE[record['status']]}]{record['status']}[/]"
    return line


def delete_preview(request: dict) -> RenderableType:
    """The exact list of reports a deletion would remove, shown before the user is asked to confirm."""
    reports = request["reports"]
    table = Table()
    table.add_column("ID", justify="right")
    table.add_column("Title")
    table.add_column("Created (UTC)")
    for report in reports[:_PREVIEW_ROWS]:
        table.add_row(str(report["id"]), report["title"], report["created_at"])
    parts: list[RenderableType] = [
        f"[bold red]About to delete {len(reports)} saved report(s)[/] ({escape(request['criteria'])}):",
        table,
    ]
    if len(reports) > _PREVIEW_ROWS:
        parts.append(f"… and {len(reports) - _PREVIEW_ROWS} more (see /reports for the full list).")
    return Group(*parts)


# ------------------------------------------------------------------------ commands


def reports_table(reports: list, owner_name: str, thread_id: str) -> Table:
    table = Table(title=f"Saved reports of {owner_name}")
    table.add_column("ID", justify="right")
    table.add_column("Title")
    table.add_column("Created (UTC)")
    table.add_column("This conversation")
    for report in reports:
        in_this_conversation = "yes" if report.thread_id == thread_id else ""
        table.add_row(str(report.id), report.title, report.summary()["created_at"], in_this_conversation)
    return table


def metrics_table(summary: dict) -> Table:
    table = Table(title="Agent metrics (this session)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for name, value in summary.items():
        if value is None:
            shown = "–"
        elif isinstance(value, int):
            shown = f"{value:,}"
        else:
            shown = str(value)
        table.add_row(name.replace("_", " "), shown)
    return table


# --------------------------------------------------------------------------- setup


def gemini_setup_panel() -> Panel:
    return Panel(
        "[bold]No Gemini API key, so the assistant cannot answer.[/]\n"
        "Fix: copy .env.example to .env, set GOOGLE_API_KEY (https://aistudio.google.com/apikey) and start again.",
        title="Setup needed: Gemini",
        border_style="yellow",
    )


def bigquery_setup_panel(problem: str) -> Panel:
    return Panel(
        "[bold]BigQuery is not set up, so questions about data cannot be answered yet.[/]\n"
        f"Reason: {escape(problem)}\n"
        "Fix: in .env set GOOGLE_APPLICATION_CREDENTIALS to the path of a service-account key and "
        "GCP_PROJECT_ID to your project id (or run `gcloud auth application-default login`), then start "
        "again. See README, section Setup. Saved reports and the commands below still work.",
        title="Setup needed: BigQuery",
        border_style="yellow",
    )


# --------------------------------------------------------------------------- trace


def _span_line(span: dict) -> str:
    """One line of the trace tree for a graph node, a model call or a tool call."""
    seconds = f"{span.get('duration_ms', 0) / 1000:.2f}s"
    if span["kind"] == "node":
        return _node_line(span, seconds)
    if span["kind"] == "llm":
        return _model_line(span, seconds)
    return _tool_line(span, seconds)


def _node_line(span: dict, seconds: str) -> str:
    line = f"[cyan]{span['name']}[/] {seconds}"
    if span.get("examples"):
        line += f" · {len(span['examples'])} golden example(s)"
    if span.get("status") == "paused":
        line += " [yellow](paused: waiting for the user)[/]"
    return line


def _model_line(span: dict, seconds: str) -> str:
    failed = span.get("status") == "error"
    line = f"[{'red' if failed else 'magenta'}]model[/] {escape(span.get('model', ''))}"
    if span.get("attempt", 1) > 1:
        line += f" (attempt {span['attempt']})"
    line += f" {seconds}"
    if not failed:
        line += f" · {span.get('tokens_in', 0):,} in / {span.get('tokens_out', 0):,} out"
    if span.get("tool_calls"):
        line += escape(f" → {', '.join(span['tool_calls'])}")
    return line


def _tool_line(span: dict, seconds: str) -> str:
    outcome = span.get("outcome", span.get("status", ""))
    if outcome in ("ok", ""):
        style = "green"
    elif span.get("status") == "error" or outcome in _FAILED_OUTCOMES:
        style = "red"
    else:
        style = "yellow"

    facts = []
    if "rows" in span:
        facts.append(f"{span['rows']} row(s)")
    if span.get("bytes"):
        facts.append(f"{span['bytes'] / 1e6:.1f} MB scanned")
    if span.get("purpose"):
        facts.append(escape(span["purpose"]))
    if span.get("criteria"):
        facts.append(escape(f"{span['criteria']} → ids {span.get('matched')}"))
    if span.get("report_id"):
        facts.append(f"report #{span['report_id']}")

    line = f"[bold]{span['name']}[/] [{style}]{outcome}[/] {seconds}"
    return " · ".join([line, *facts])


def trace_tree(record: dict) -> Tree:
    """A turn as a tree: nodes, model calls, tool calls, then incidents."""
    style = _STATUS_STYLE.get(record["status"], "white")
    tree = Tree(
        f"[bold]trace {record['trace_id']}[/] · {record['started_at']} · user {record['user_id']} · conversation {record['thread_id']} · "
        f"[{style}]{record['status']}[/] · {record['duration_ms'] / 1000:.1f}s · {record['model_calls']} model call(s) · "
        f"{record['tokens_in']:,} in / {record['tokens_out']:,} out tokens"
    )
    tree.add(f"[dim]question:[/] {escape(record['question'])}")
    if meta := record.get("meta"):
        versions = ", ".join(f"{name} {digest}" for name, digest in meta.get("prompt", {}).items())
        tree.add(f"[dim]prompt:[/] {versions} [dim]· models:[/] {escape(', '.join(meta.get('models', [])))}")
    branches: dict[int | None, Tree] = {None: tree}
    for index, span in enumerate(record["spans"]):
        branch = branches[index] = branches.get(span.get("parent"), tree).add(_span_line(span))
        if span.get("sql"):
            branch.add(f"[dim]{escape(' '.join(span['sql'].split()))}[/]")
        if span.get("error"):
            branch.add(f"[red]{escape(str(span['error']))}[/]")
    if record["events"]:
        events = tree.add("[bold]events[/]")
        for event in record["events"]:
            details = ", ".join(f"{k}={v}" for k, v in event.items() if k not in ("kind", "at_ms"))
            events.add(f"[yellow]{event['kind']}[/] at {event['at_ms'] / 1000:.2f}s {escape(details)}")
    if record.get("error"):
        tree.add(f"[red]error: {escape(record['error'])}[/]")
    return tree
