"""The web demo: access code, sessions, the confirmation of deletions over HTTP, and its limits."""

import pytest
from fastapi.testclient import TestClient

from agent.config import settings
from agent.interface import web
from agent.observability import Trace
from conftest import FakeLLM, ai, tool_call


@pytest.fixture
def site(make, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "trace_path", tmp_path / "logs" / "traces.jsonl")

    def _site(llm, access_code="open sesame"):
        h = make(llm)
        service = web.Service(graph=h.graph, reports=h.reports, access_code=access_code)
        return TestClient(web.create_app(service)), h

    return _site


def sign_in(client, user="alice", code="open sesame") -> dict:
    token = client.post("/api/session", json={"code": code, "user": user}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_page_and_user_list_are_public_but_nothing_else_is(site):
    client, _ = site(FakeLLM(ai("ok")))
    assert client.get("/").status_code == 200 and "Retail data analysis assistant" in client.get("/").text
    assert {"id": "alice", "name": "Alice (Manager)"} in client.get("/api/users").json()
    for path in ("/api/reports", "/api/metrics", "/api/trace"):
        assert client.get(path).status_code == 401
    assert client.post("/api/ask", json={"message": "Revenue?"}).status_code == 401


def test_session_needs_the_access_code(site):
    client, _ = site(FakeLLM(ai("ok")))
    assert client.post("/api/session", json={"code": "guess", "user": "alice"}).status_code == 403
    assert client.post("/api/session", json={"code": "open sesame", "user": "mallory"}).status_code == 400
    assert client.post("/api/session", json={"code": "open sesame", "user": "alice"}).status_code == 200


def test_demo_without_a_configured_access_code_admits_nobody(site):
    client, _ = site(FakeLLM(ai("ok")), access_code="")
    assert client.post("/api/session", json={"code": "", "user": "alice"}).status_code == 503


def test_question_is_answered_with_its_trace_summary(site):
    client, h = site(FakeLLM(ai("", tool_call("run_sql", sql="SELECT 1 FROM orders")), ai("Revenue is up.")))
    result = client.post("/api/ask", json={"message": "How is revenue?"}, headers=sign_in(client)).json()
    assert result["answer"] == "Revenue is up." and result["status"] == "ok"
    assert (result["model_calls"], result["sql_queries"]) == (2, 1) and result["trace_id"]


def test_deletion_waits_for_an_explicit_decision(site):
    delete = ai("", tool_call("delete_reports", all_reports=True))
    client, h = site(FakeLLM(delete, ai("Kept."), delete, ai("Deleted.")))
    headers = sign_in(client)
    h.reports.save("alice", "t1", "Q1 report", "…")

    first = client.post("/api/ask", json={"message": "Delete all my reports"}, headers=headers).json()
    assert [r["title"] for r in first["confirmation"]["reports"]] == ["Q1 report"] and "answer" not in first
    assert len(h.reports.find("alice")) == 1  # nothing is deleted while the question is open
    assert client.post("/api/ask", json={"message": "Never mind"}, headers=headers).status_code == 409

    assert client.post("/api/confirm", json={"confirmed": False}, headers=headers).json()["answer"] == "Kept."
    assert len(h.reports.find("alice")) == 1
    assert client.post("/api/confirm", json={"confirmed": True}, headers=headers).status_code == 409  # nothing pending

    client.post("/api/ask", json={"message": "Delete all my reports"}, headers=headers)
    done = client.post("/api/confirm", json={"confirmed": True}, headers=headers).json()
    assert h.reports.find("alice") == [] and done["answer"] == "Deleted."
    assert done["changes"] and "Q1 report" in done["changes"][0]  # the receipt comes from the store, not the model


def test_saved_report_is_returned_in_full(site):
    llm = FakeLLM(ai("", tool_call("save_report", title="Q3 review", content="# Q3\nRevenue grew.")), ai("Saved."))
    client, _ = site(llm)
    result = client.post("/api/ask", json={"message": "Create a report"}, headers=sign_in(client)).json()
    assert result["reports"] == [{"id": 1, "title": "Q3 review", "body": "# Q3\nRevenue grew."}]


def test_unexpected_failure_still_returns_an_answer(site):
    client, _ = site(FakeLLM(RuntimeError("kaboom")))
    result = client.post("/api/ask", json={"message": "Revenue?"}, headers=sign_in(client)).json()
    assert (
        result["status"] == "error" and "Nothing was changed" in result["answer"] and "kaboom" not in result["answer"]
    )


def test_only_own_traces_and_reports_are_visible(site):
    client, h = site(FakeLLM(ai("ok")))
    headers = sign_in(client)
    theirs = Trace("bob", "t9", "How is Brand B selling?", path=settings.trace_path).finish(answer="x")
    h.reports.save("bob", "t9", "Bob's report", "…")
    client.post("/api/ask", json={"message": "How is Brand A selling?"}, headers=headers)

    assert client.get(f"/api/trace?id={theirs['trace_id']}", headers=headers).status_code == 404
    assert "How is Brand A selling?" in client.get("/api/trace", headers=headers).json()["text"]
    assert client.get("/api/reports", headers=headers).json() == []
    assert client.get("/api/metrics", headers=headers).json()["turns"] == 1


def test_turns_per_session_are_limited(site, monkeypatch):
    monkeypatch.setattr(web, "MAX_TURNS_PER_SESSION", 2)
    client, _ = site(FakeLLM(ai("ok")))
    headers = sign_in(client)
    codes = [client.post("/api/ask", json={"message": "Q"}, headers=headers).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_turns_per_minute_are_limited_across_sessions(site, monkeypatch):
    monkeypatch.setattr(web, "MAX_TURNS_PER_MINUTE", 1)
    client, _ = site(FakeLLM(ai("ok")))
    assert client.post("/api/ask", json={"message": "Q"}, headers=sign_in(client)).status_code == 200
    assert client.post("/api/ask", json={"message": "Q"}, headers=sign_in(client, user="bob")).status_code == 429
