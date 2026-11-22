"""Web demo: the same agent graph behind a small HTTP API and one page.

It exists so that a reviewer can try the assistant without installing anything.
The graph, guardrails and stores are exactly those of the CLI; this module only
replaces the terminal: sessions instead of a process, a confirmation request
instead of a typed "yes", JSON instead of panels.

A public address spends the owner's model quota and query budget, so the demo is
closed by default: it needs an access code (DEMO_ACCESS_CODE), and it limits the
turns per session and per minute. The demo identities are a stand-in for single
sign-on, as in the CLI. Conversations and sessions live in memory, so the service
must run as a single instance.
"""

from __future__ import annotations

import hmac
import io
import logging
import os
import secrets
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field
from rich.console import Console

from agent.config import User, load_users, settings
from agent.core.graph import build_graph
from agent.core.llm import GeminiClient
from agent.core.nodes import Nodes
from agent.core.state import TurnContext, new_turn, text_of
from agent.core.tools import tool_schemas
from agent.guardrails.pii import scan_text
from agent.interface import render
from agent.interface.cli import failure
from agent.observability import Trace, load_traces, summarise
from agent.stores.golden import GoldenStore
from agent.stores.reports import ReportStore
from agent.warehouse.bq_client import BigQueryRunner

log = logging.getLogger(__name__)

PAGE = Path(__file__).parent / "static" / "index.html"
MAX_SESSIONS = 200
MAX_TURNS_PER_SESSION = 40
MAX_TURNS_PER_MINUTE = 20  # across all sessions: the model quota is shared


@dataclass
class Session:
    user: User
    thread_id: str
    turns: int = 0
    records: list[dict] = field(default_factory=list)  # trace records of this session
    pending: Trace | None = None  # the turn that is waiting for the user's confirmation
    lock: threading.Lock = field(default_factory=threading.Lock)  # one turn at a time


class Service:
    """Everything the endpoints share: the graph, the stores and the open sessions."""

    def __init__(self, graph: Any = None, reports: ReportStore | None = None, access_code: str | None = None):
        self.users = load_users()
        self.reports = reports or ReportStore(settings.reports_db_path)
        self.graph = graph or self._build_graph()
        self.access_code = access_code if access_code is not None else os.getenv("DEMO_ACCESS_CODE", "")
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self._recent_turns: deque[float] = deque()
        self._guard = threading.Lock()

    def _build_graph(self):
        nodes = Nodes(GeminiClient(tool_schemas()), BigQueryRunner(), self.reports, GoldenStore(settings.trios_path))
        return build_graph(nodes, InMemorySaver())

    # --------------------------------------------------------------- sessions

    def open_session(self, code: str, user_id: str) -> tuple[str, Session]:
        if not self.access_code:
            raise HTTPException(503, "The demo is not configured: DEMO_ACCESS_CODE is not set.")
        if not hmac.compare_digest(code.encode(), self.access_code.encode()):
            raise HTTPException(403, "Wrong access code.")
        if user_id not in self.users:
            raise HTTPException(400, "Unknown user.")
        token = secrets.token_urlsafe(24)
        with self._guard:
            self.sessions[token] = Session(self.users[user_id], uuid.uuid4().hex[:8])
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)  # the oldest session makes room
        return token, self.sessions[token]

    def switch_user(self, token: str, user_id: str) -> tuple[str, Session]:
        """Continue as another demo user: a fresh session, so nothing of the previous user carries over.

        The visitor has already shown the access code, so it is not asked again. The old
        session ends, and with it its conversation.
        """
        self.session(token)
        if user_id not in self.users:
            raise HTTPException(400, "Unknown user.")
        fresh_token = secrets.token_urlsafe(24)
        with self._guard:
            self.sessions.pop(token, None)
            self.sessions[fresh_token] = Session(self.users[user_id], uuid.uuid4().hex[:8])
        return fresh_token, self.sessions[fresh_token]

    def session(self, token: str) -> Session:
        found = self.sessions.get(token)
        if found is None:
            raise HTTPException(401, "The session has expired. Sign in again.")
        return found

    def _spend_a_turn(self, session: Session) -> None:
        """Refuse a turn that would exceed the limits, before it costs a model call."""
        now = time.monotonic()
        with self._guard:
            while self._recent_turns and now - self._recent_turns[0] > 60:
                self._recent_turns.popleft()
            if session.turns >= MAX_TURNS_PER_SESSION:
                raise HTTPException(429, "This demo session has reached its limit of questions. Sign in again.")
            if len(self._recent_turns) >= MAX_TURNS_PER_MINUTE:
                raise HTTPException(429, "The demo is busy. Please try again in a minute.")
            self._recent_turns.append(now)
            session.turns += 1

    # ------------------------------------------------------------------ turns

    def ask(self, session: Session, question: str) -> dict:
        if session.pending is not None:
            raise HTTPException(409, "Answer the pending confirmation first.")
        self._spend_a_turn(session)
        trace = Trace(session.user.id, session.thread_id, scan_text(question)[0], path=settings.trace_path)
        return self._run(session, trace, new_turn(question))

    def confirm(self, session: Session, confirmed: bool) -> dict:
        """Continue the turn that paused for a deletion. Only an explicit True deletes."""
        trace = session.pending
        if trace is None:
            raise HTTPException(409, "Nothing is waiting for confirmation.")
        session.pending = None
        return self._run(session, trace, Command(resume={"confirmed": confirmed is True}))

    def _run(self, session: Session, trace: Trace, payload: Any) -> dict:
        config = {"configurable": {"thread_id": session.thread_id}, "recursion_limit": 50}
        context = TurnContext(session.user, session.thread_id, trace)
        answer, error = "", None
        try:
            with session.lock:
                result = self.graph.invoke(payload, config, context=context)
            if waiting := result.get("__interrupt__"):
                session.pending = trace  # the same trace continues after the user's decision
                return {"confirmation": waiting[0].value}
            answer = text_of(result["messages"][-1])
        except Exception as e:  # the page must get an answer whatever happens
            log.exception("Turn %s failed", trace.trace_id)
            trace.mark("error")
            error, answer = f"{type(e).__name__}: {e}"[:500], failure(trace.changes)
        record = trace.finish(answer=answer, error=error)
        session.records.append(record)
        return {
            "answer": answer,
            "status": record["status"],
            "trace_id": record["trace_id"],
            "model_calls": record["model_calls"],
            "sql_queries": record["sql_queries"],
            "seconds": round(record["duration_ms"] / 1000, 1),
            "changes": record["changes"],
            "reports": self._reports_saved_in(session, record),
        }

    def _reports_saved_in(self, session: Session, record: dict) -> list[dict]:
        """Reports saved in this turn, in full, read from the store: the model need not repeat them."""
        saved_ids = [
            span["report_id"]
            for span in record["spans"]
            if span["kind"] == "tool" and span["name"] == "save_report" and span.get("report_id")
        ]
        if not saved_ids:
            return []
        return [
            {"id": report.id, "title": report.title, "body": report.body}
            for report in self.reports.find(session.user.id, ids=saved_ids)
        ]

    # ------------------------------------------------------------------ views

    def trace_text(self, session: Session, trace_id: str | None) -> str:
        """A trace as the CLI shows it. Only the session user's own traces can be looked up."""
        if trace_id:
            own = [r for r in load_traces(settings.trace_path) if r["user_id"] == session.user.id]
            record = next((r for r in reversed(own) if r["trace_id"].startswith(trace_id)), None)
        else:
            record = session.records[-1] if session.records else None
        if record is None:
            raise HTTPException(404, "No such trace.")
        return _as_text(render.trace_tree(record))


def _as_text(renderable: Any) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=110, color_system=None).print(renderable)
    return buffer.getvalue()


# ---------------------------------------------------------------------------- HTTP


class SignIn(BaseModel):
    code: str = Field(max_length=200)
    user: str = Field(max_length=50)


class Switch(BaseModel):
    user: str = Field(max_length=50)


class Question(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class Decision(BaseModel):
    confirmed: bool


def create_app(service: Service | None = None) -> FastAPI:
    app = FastAPI(title="Retail data analysis assistant", docs_url=None, redoc_url=None, openapi_url=None)
    state: dict[str, Service] = {}

    def shared() -> Service:
        if "service" not in state:  # built on first use, so the page loads even if the model key is missing
            state["service"] = service or Service()
        return state["service"]

    def current(authorization: str = Header(default=""), svc: Service = Depends(shared)) -> Session:
        return svc.session(authorization.removeprefix("Bearer ").strip())

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(PAGE)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/users")
    def users(svc: Service = Depends(shared)) -> list[dict]:
        return [{"id": user.id, "name": user.name} for user in svc.users.values()]

    @app.post("/api/session")
    def sign_in(body: SignIn, svc: Service = Depends(shared)) -> dict:
        token, session = svc.open_session(body.code, body.user)
        return {"token": token, "name": session.user.name, "scope": session.user.describe_scope()}

    @app.post("/api/switch")
    def switch(body: Switch, authorization: str = Header(default=""), svc: Service = Depends(shared)) -> dict:
        token, session = svc.switch_user(authorization.removeprefix("Bearer ").strip(), body.user)
        return {"token": token, "name": session.user.name, "scope": session.user.describe_scope()}

    @app.post("/api/ask")
    def ask(body: Question, session: Session = Depends(current), svc: Service = Depends(shared)) -> dict:
        return svc.ask(session, body.message)

    @app.post("/api/confirm")
    def confirm(body: Decision, session: Session = Depends(current), svc: Service = Depends(shared)) -> dict:
        return svc.confirm(session, body.confirmed)

    @app.post("/api/new")
    def new_conversation(session: Session = Depends(current)) -> dict:
        session.thread_id, session.pending = uuid.uuid4().hex[:8], None
        return {"thread_id": session.thread_id}

    @app.get("/api/reports")
    def reports(session: Session = Depends(current), svc: Service = Depends(shared)) -> list[dict]:
        return [
            {**report.summary(), "this_conversation": report.thread_id == session.thread_id}
            for report in svc.reports.find(session.user.id)
        ]

    @app.get("/api/metrics")
    def metrics(session: Session = Depends(current)) -> dict:
        return summarise(session.records)

    @app.get("/api/trace")
    def trace(id: str = "", session: Session = Depends(current), svc: Service = Depends(shared)) -> dict:
        return {"text": svc.trace_text(session, id.strip() or None)}

    return app


app = create_app()
