"""Offline test harness: the real graph with a scripted model and a fake warehouse."""

from __future__ import annotations

import itertools

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.config import User, settings
from agent.core.graph import build_graph
from agent.core.nodes import Nodes
from agent.core.state import TurnContext, new_turn
from agent.observability import Trace
from agent.stores.golden import GoldenStore
from agent.stores.reports import ReportStore
from agent.warehouse.bq_client import QueryOutcome

ALICE = User("alice", "Alice", {"brand": ("Brand A",)})
BOB = User("bob", "Bob", {"brand": ("Brand B",)})
ADMIN = User("admin", "Dana", {})

_ids = itertools.count()


def tool_call(name: str, **args) -> dict:
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


def ai(text: str = "", *calls: dict) -> AIMessage:
    return AIMessage(content=text, tool_calls=list(calls))


def _next(script: list):
    return script.pop(0) if len(script) > 1 else script[0]  # the last step repeats


class FakeLLM:
    """Plays a script of AIMessages, exceptions, or callables(messages) -> AIMessage."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[tuple[list, bool]] = []

    def invoke(self, messages, trace, use_tools=True):
        self.calls.append((messages, use_tools))
        with trace.span("llm", kind="llm", model="fake"):
            step = _next(self.script)
            if isinstance(step, Exception):
                raise step
            reply = step(messages) if callable(step) else step
            return AIMessage(
                content=reply.content, tool_calls=reply.tool_calls
            )  # a fresh message, as a real model returns


class FakeBQ:
    """Plays a script of DataFrames or exceptions; records the SQL it was given."""

    def __init__(self, *script, bytes_per_query: int = 1234):
        self.script = list(script) or [pd.DataFrame({"n": [1]})]
        self.bytes_per_query = bytes_per_query
        self.queries: list[str] = []

    def run(self, sql, trace_id=""):
        self.queries.append(sql)
        step = _next(self.script)
        if isinstance(step, Exception):
            raise step
        return QueryOutcome(
            df=step, bytes_processed=self.bytes_per_query, job_id="job_1", truncated=False, duration_ms=5
        )

    def get_table_schema(self, table_name, include_pii=False):
        return [{"name": "id", "type": "INTEGER", "mode": "NULLABLE", "description": ""}]


class Harness:
    def __init__(self, tmp_path, llm, bq=None, user=ALICE, thread="t1"):
        self.llm, self.bq, self.user, self.thread = llm, bq or FakeBQ(), user, thread
        self.reports = ReportStore(tmp_path / "reports.db")
        self.graph = build_graph(Nodes(llm, self.bq, self.reports, GoldenStore(settings.trios_path)), InMemorySaver())
        self.trace: Trace | None = None

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread}}

    def _context(self) -> TurnContext:
        return TurnContext(self.user, self.thread, self.trace)

    def ask(self, question: str) -> dict:
        self.trace = Trace(self.user.id, self.thread, question)
        return self.graph.invoke(new_turn(question), self.config, context=self._context())

    def resume(self, decision) -> dict:
        return self.graph.invoke(Command(resume=decision), self.config, context=self._context())

    def events(self, kind: str) -> list[dict]:
        return [e for e in self.trace.events if e["kind"] == kind]

    def tool_spans(self, name: str) -> list[dict]:
        return [s for s in self.trace.spans if s["kind"] == "tool" and s["name"] == name]


@pytest.fixture
def make(tmp_path):
    def _make(llm, bq=None, **kwargs) -> Harness:
        return Harness(tmp_path, llm, bq, **kwargs)

    return _make


def pytest_addoption(parser):
    parser.addoption(
        "--live", action="store_true", default=False, help="Run live BigQuery/Gemini checks (requires credentials)."
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="Live API checks are opt-in; pass --live to enable them")
    for item in items:
        if item.path.name == "test_live.py":
            item.add_marker(skip)
