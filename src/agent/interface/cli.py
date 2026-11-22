"""Command-line chat interface.

The CLI owns the interaction: the prompt, progress, slash commands and the
confirmation dialog for deletions. What things look like is in render.py. The
graph never prints.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich.console import Console
from rich.markup import escape

from agent.config import User, load_users, settings
from agent.core.graph import build_graph
from agent.core.llm import GeminiClient
from agent.core.nodes import Nodes, with_changes
from agent.core.state import TurnContext, new_turn, text_of
from agent.core.tools import tool_schemas
from agent.guardrails.pii import scan_text
from agent.interface import render
from agent.observability import Trace, load_traces, summarise
from agent.stores.golden import GoldenStore
from agent.stores.reports import ReportStore
from agent.warehouse.bq_client import BigQueryRunner

log = logging.getLogger(__name__)

FAILURE = "Something went wrong on my side and I couldn't complete that."


def failure(changes: list[str]) -> str:
    """What the user reads after an unexpected failure, including what the turn did before it failed."""
    message = with_changes(FAILURE, changes)
    if changes:
        return f"{message} Check before you retry, so that it is not done twice."
    return f"{message} Please try again."


class ChatApp:
    def __init__(
        self,
        user: User,
        thread_id: str,
        console: Console,
        graph: Any = None,
        reports: ReportStore | None = None,
        bq: BigQueryRunner | None = None,
    ):
        self.user, self.thread_id, self.console = user, thread_id, console
        self.reports = reports or ReportStore(settings.reports_db_path)
        self.bq = bq if bq is not None or graph is not None else BigQueryRunner()
        self.graph = graph or self._build_graph()
        self.session: list[dict] = []  # trace records of this session

    def _build_graph(self):
        nodes = Nodes(GeminiClient(tool_schemas()), self.bq, self.reports, GoldenStore(settings.trios_path))
        return build_graph(nodes, InMemorySaver())

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 50}

    # ------------------------------------------------------------------ turns

    def ask(self, question: str) -> dict:
        trace = Trace(self.user.id, self.thread_id, scan_text(question)[0], path=settings.trace_path)
        ctx = TurnContext(self.user, self.thread_id, trace)
        answer, error = "", None
        try:
            result = self._invoke(new_turn(question), ctx)
            while waiting := result.get("__interrupt__"):
                result = self._invoke(Command(resume=self._confirm(waiting[0].value)), ctx)
            answer = text_of(result["messages"][-1])
        except KeyboardInterrupt:
            trace.mark("cancelled")
            error = "cancelled by user"
        except Exception as e:  # the UI must survive anything
            log.exception("Turn %s failed", trace.trace_id)
            trace.mark("error")
            error, answer = f"{type(e).__name__}: {e}"[:500], failure(trace.changes)
        record = trace.finish(answer=answer, error=error)
        self.session.append(record)
        self._show_answer(record)
        return record

    def _invoke(self, payload: Any, ctx: TurnContext) -> dict:
        if not self.console.is_terminal or self.console.is_dumb_terminal:
            self.console.print("[dim]Processing…[/]")
        with self.console.status("Processing…", spinner="dots") as status:
            ctx.trace.listener = lambda label: status.update(f"{escape(label)}…")
            try:
                return self.graph.invoke(payload, self.config, context=ctx)
            finally:
                ctx.trace.listener = None

    def _confirm(self, request: dict) -> dict:
        """Ask the user to approve a deletion. Only a typed 'yes' or 'y' confirms."""
        self.console.print(render.delete_preview(request))
        try:  # keys typed before this dialog appeared are dropped, so they cannot confirm it
            reply = self._input(
                "[bold red]Delete these reports?[/] Type [bold]yes[/] to confirm, anything else to cancel: "
            )
        except (EOFError, KeyboardInterrupt):
            reply = ""
        return {"confirmed": reply.strip().lower() in ("y", "yes")}

    def _input(self, prompt: str) -> str:
        if discard_typeahead():
            self.console.print("[dim](Ignored what was typed while I was working.)[/]")
        return self.console.input(prompt)

    def _show_answer(self, record: dict) -> None:
        if record["status"] == "cancelled":
            self.console.print("[yellow]Cancelled.[/]")
        elif record["answer"]:
            for report in self._reports_saved_in(record):  # shown in full, so the model need not repeat them
                self.console.print(render.report_panel(report))
            self.console.print(render.answer_panel(record["answer"]))
        self.console.print(render.turn_footer(record))

    def _reports_saved_in(self, record: dict) -> list:
        saved_ids = []
        for span in record["spans"]:
            if span["kind"] == "tool" and span["name"] == "save_report" and span.get("report_id"):
                saved_ids.append(span["report_id"])
        return self.reports.find(self.user.id, ids=saved_ids) if saved_ids else []

    # --------------------------------------------------------------- commands

    def command(self, line: str) -> bool:
        """Handle a slash command. Returns False when the app should exit."""
        name, _, arg = line.partition(" ")
        arg = arg.strip()
        if name in ("/quit", "/exit", "/q"):
            return False
        if name == "/help":
            self.console.print(render.HELP)
        elif name == "/trace":
            self._show_trace(arg)
        elif name == "/metrics":
            self.console.print(render.metrics_table(summarise(self.session)))
        elif name == "/reports":
            self._show_reports()
        elif name == "/new":
            self.thread_id = uuid.uuid4().hex[:8]
            self.console.print(f"Started a new conversation ({self.thread_id}).")
        else:
            self.console.print(f"Unknown command {escape(name)}. Type /help.")
        return True

    def _show_trace(self, trace_id: str) -> None:
        if trace_id:
            # Traces hold questions and SQL, so a user may look up only their own. Another user's id
            # gets the same answer as an unknown one, which does not reveal that it exists.
            own = [r for r in load_traces(settings.trace_path) if r["user_id"] == self.user.id]
            record = next((r for r in reversed(own) if r["trace_id"].startswith(trace_id)), None)
        else:
            record = self.session[-1] if self.session else None
        if record is None:
            self.console.print("No such trace." if trace_id else "No turns yet in this session.")
            return
        self.console.print(render.trace_tree(record))

    def _show_reports(self) -> None:
        reports = self.reports.find(self.user.id)
        if reports:
            self.console.print(render.reports_table(reports, self.user.name, self.thread_id))
        else:
            self.console.print("You have no saved reports.")

    # ------------------------------------------------------------------- loop

    def check_setup(self) -> None:
        """Say at startup what is not configured, instead of failing on the first question."""
        if not os.getenv("GOOGLE_API_KEY"):
            self.console.print(render.gemini_setup_panel())
        if self.bq is None:
            return
        with self.console.status("Checking access to BigQuery…", spinner="dots"):
            problem = self.bq.setup_problem()
        if problem:
            self.console.print(render.bigquery_setup_panel(problem))

    def run(self) -> None:
        self.console.print(render.banner(self.user, self.thread_id))
        self.check_setup()
        while True:
            try:
                line = self._input("\n[bold green]you ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
            if not line:
                continue
            if line.startswith("/"):
                try:
                    if not self.command(line):
                        break
                except Exception:  # a failing command must not end the chat
                    log.exception("Command %s failed", line)
                    self.console.print("[red]That command failed.[/] The details are in logs/agent.log.")
                continue
            self.ask(line)
        self.console.print("[dim]Goodbye.[/]")


# ---------------------------------------------------------------------- terminal


def discard_typeahead(stream: Any = None) -> bool:
    """Drop keys typed while the assistant was busy. Returns True if whole lines were dropped.

    Otherwise every line typed during a turn would be taken as a further question,
    each costing model calls, and a stray "yes" could answer a confirmation dialog
    before it is even shown. Piped or scripted input is deliberate and left alone.
    """
    stream = stream or sys.stdin
    try:
        if not stream.isatty():
            return False
        import fcntl
        import struct
        import termios

        waiting = struct.unpack("i", fcntl.ioctl(stream, termios.FIONREAD, b"\0\0\0\0"))[0]
        termios.tcflush(stream, termios.TCIFLUSH)
        return waiting > 0
    except Exception:  # a terminal quirk must never break the chat
        return False


# ---------------------------------------------------------------------------- main


def _setup_logging() -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=settings.log_path, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.captureWarnings(True)  # library warnings go to the log, not into the chat
    for noisy in ("httpx", "httpx2", "httpcore", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    try:
        users = load_users()
    except (OSError, ValueError, KeyError, TypeError) as e:
        Console().print(f"[red]Cannot start: config/users.yaml is not valid.[/] {escape(str(e))}")
        return 2
    parser = argparse.ArgumentParser(prog="main.py", description="Chat with the retail data analysis assistant.")
    parser.add_argument(
        "--user",
        default="alice",
        choices=sorted(users),
        help="who you are (see config/users.yaml); decides which products you can analyse",
    )
    args = parser.parse_args(argv)

    _setup_logging()
    console = Console()
    ChatApp(users[args.user], uuid.uuid4().hex[:8], console).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
