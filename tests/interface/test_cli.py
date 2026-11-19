"""The CLI: confirmation dialog, failure containment, setup check and views."""

import io
import os
import select

import pytest
from rich.console import Console

from agent.config import settings
from agent.interface.cli import ChatApp, discard_typeahead
from agent.observability import load_traces
from conftest import FakeBQ, FakeLLM, ai, tool_call


@pytest.fixture
def cli(make, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "trace_path", tmp_path / "logs" / "traces.jsonl")

    def _cli(llm, bq=None, replies=()):
        h = make(llm, bq)
        answers = iter(replies)
        monkeypatch.setattr("builtins.input", lambda *args: next(answers))
        console = Console(file=io.StringIO(), width=200)
        return ChatApp(h.user, h.thread, console, graph=h.graph, reports=h.reports), h, console.file

    return _cli


def test_cli_delete_needs_a_typed_yes(cli):
    delete = ai("", tool_call("delete_reports", all_reports=True))
    app, h, out = cli(FakeLLM(delete, ai("Kept."), delete, ai("Deleted.")), replies=["maybe", "yes"])
    h.reports.save("alice", "t1", "Q1 report", "…")

    assert app.ask("Delete all my reports")["answer"] == "Kept."
    assert len(h.reports.find("alice")) == 1
    assert "About to delete 1 saved report(s) (all reports):" in out.getvalue() and "Q1 report" in out.getvalue()

    assert app.ask("Delete all my reports")["answer"] == "Deleted."
    assert h.reports.find("alice") == []


def test_saved_report_is_shown_in_full_by_the_cli(cli):
    llm = FakeLLM(
        ai("", tool_call("save_report", title="Q3 review", content="# Q3\nRevenue grew.")), ai("Saved as report #1.")
    )
    app, h, out = cli(llm)
    app.ask("Create a report for Q3")
    text = out.getvalue()
    assert "report #1 · Q3 review" in text and "Revenue grew." in text and "Saved as report #1." in text


def test_cli_survives_an_unexpected_failure(cli):
    app, h, out = cli(FakeLLM(RuntimeError("kaboom")))
    record = app.ask("Revenue?")
    assert record["status"] == "error" and "RuntimeError: kaboom" in record["error"]
    assert "Something went wrong on my side" in out.getvalue() and "Traceback" not in out.getvalue()
    assert load_traces(settings.trace_path)[-1]["trace_id"] == record["trace_id"]  # the failure is on record

    h.llm.script = [ai("Recovered.")]
    assert app.ask("Again?")["answer"] == "Recovered."
    roles = [type(m).__name__ for m in h.llm.calls[-1][0]]
    assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]  # the unanswered turn is patched


def test_cli_recovers_when_a_turn_dies_after_running_tools(cli):
    app, h, out = cli(FakeLLM(ai("", tool_call("run_sql", sql="SELECT 1 FROM orders")), RuntimeError("kaboom")))
    assert app.ask("Revenue?")["status"] == "error" and len(h.bq.queries) == 1

    h.llm.script, h.llm.calls = [ai("Recovered.")], []
    assert app.ask("Again?")["answer"] == "Recovered."
    assert len(h.llm.calls) == 1 and len(h.bq.queries) == 1  # the dead turn's work is not replayed
    roles = [type(m).__name__ for m in h.llm.calls[0][0]]
    assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"]


def test_startup_check_names_what_is_not_configured(cli, monkeypatch):
    class NoCredentials:
        def setup_problem(self):
            return "Your default credentials were not found."

    class Ready:
        def setup_problem(self):
            return None

    app, h, out = cli(FakeLLM(ai("ok")))
    monkeypatch.setenv("GOOGLE_API_KEY", "set")
    app.bq = Ready()
    app.check_setup()
    assert "Setup needed" not in out.getvalue()  # nothing to say when all is in place

    monkeypatch.delenv("GOOGLE_API_KEY")
    app.bq = NoCredentials()
    app.check_setup()
    text = " ".join(out.getvalue().split())
    assert "Setup needed: Gemini" in text and "Setup needed: BigQuery" in text
    assert "Your default credentials were not found." in text and "GOOGLE_APPLICATION_CREDENTIALS" in text


@pytest.mark.skipif(os.name == "nt", reason="POSIX terminals only")
def test_keys_typed_while_busy_are_dropped_on_a_terminal_but_not_on_a_pipe():
    import pty

    master, slave = pty.openpty()
    try:
        with os.fdopen(slave, closefd=False) as terminal:
            os.write(master, b"1\n2\nyes\n")  # typed while the assistant was working
            assert select.select([slave], [], [], 1)[0]  # ... and waiting to be read
            assert discard_typeahead(terminal) is True
            assert select.select([slave], [], [], 0.1)[0] == []  # gone: not the next question, not a "yes"
            assert discard_typeahead(terminal) is False
    finally:
        os.close(master)
        os.close(slave)

    read_end, write_end = os.pipe()
    with os.fdopen(read_end) as piped:  # scripted input is deliberate: kept
        os.write(write_end, b"What was revenue?\n")
        os.close(write_end)
        assert discard_typeahead(piped) is False and piped.readline() == "What was revenue?\n"


def test_failing_command_does_not_end_the_chat(cli):
    app, h, out = cli(FakeLLM(ai("ok")), replies=["/reports", "/quit"])
    h.reports.path = "/nonexistent-dir/reports.db"  # the report database is gone
    app.run()  # must return normally
    text = out.getvalue()
    assert "That command failed." in text and "Goodbye." in text and "Traceback" not in text


def test_trace_metrics_and_reports_views(cli):
    import pandas as pd

    llm = FakeLLM(
        ai("", tool_call("run_sql", sql="SELECT [1, 2] AS a FROM orders", purpose="check [brackets]")), ai("**Done.**")
    )
    app, h, out = cli(llm, FakeBQ(pd.DataFrame({"n": [1]})))
    h.reports.save("alice", "t1", "Q1 report", "…")
    app.ask("How did revenue develop?")
    for command in (
        "/trace",
        "/metrics",
        "/reports",
        "/help",
        "/unknown",
        f"/trace {app.session[-1]['trace_id'][:6]}",
    ):
        assert app.command(command)
    text = out.getvalue()
    for expected in (
        "run_sql",
        "ok",
        "SELECT [1, 2] AS a FROM orders",
        "check [brackets]",
        "sql first pass rate",
        "Q1 report",
        "Unknown command /unknown",
    ):
        assert expected in text, expected
    assert app.command("/quit") is False
