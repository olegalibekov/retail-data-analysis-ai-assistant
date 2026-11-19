"""Model retries and the fallback model."""

import pytest
from langchain_core.messages import AIMessage

from agent.config import settings
from agent.core.llm import GeminiClient, LLMUnavailable
from agent.observability import Trace

OK = AIMessage(content="hello", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
BUSY = Exception("429 RESOURCE_EXHAUSTED. Please retry in 3s.")
DOWN = Exception("503 UNAVAILABLE: the model is overloaded")
BAD_KEY = Exception("400 INVALID_ARGUMENT: API key not valid")


class Scripted:
    def __init__(self, *steps):
        self.steps, self.calls = list(steps), 0

    def invoke(self, messages):
        self.calls += 1
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step


def client(primary: Scripted, fallback: Scripted):
    sleeps: list[float] = []
    models = {"primary": primary, "fallback": fallback}
    return GeminiClient(
        [], models=list(models), factory=lambda name, with_tools: models[name], sleep=sleeps.append
    ), sleeps


def test_transient_error_is_retried_with_the_servers_delay():
    llm, sleeps = client(Scripted(BUSY, OK), Scripted(OK))
    trace = Trace("alice", "t1", "q")
    assert llm.invoke([], trace) is OK
    assert sleeps == [3.0]
    assert [(s["status"], s["attempt"], s["fallback"]) for s in trace.spans] == [("error", 1, False), ("ok", 2, False)]
    assert trace.spans[1]["tokens_in"] == 10 and trace.spans[0]["transient"] is True


def test_permanent_error_goes_straight_to_the_fallback_model():
    primary, fallback = Scripted(BAD_KEY), Scripted(OK)
    llm, sleeps = client(primary, fallback)
    trace = Trace("alice", "t1", "q")
    assert llm.invoke([], trace) is OK
    assert primary.calls == 1 and sleeps == [] and trace.spans[-1]["fallback"] is True
    assert trace.finish()["fallback_used"] is True


def test_long_quota_wait_switches_model_instead_of_sleeping():
    llm, sleeps = client(Scripted(Exception("429 RESOURCE_EXHAUSTED retryDelay: '3600s'")), Scripted(OK))
    assert llm.invoke([], Trace("alice", "t1", "q")) is OK and sleeps == []


def test_timeout_is_not_retried_on_the_same_model():
    primary, fallback = (
        Scripted(Exception("504 DEADLINE_EXCEEDED. Deadline expired before operation could complete.")),
        Scripted(OK),
    )
    llm, sleeps = client(primary, fallback)
    assert llm.invoke([], Trace("alice", "t1", "q")) is OK
    assert primary.calls == 1 and fallback.calls == 1 and sleeps == []


def test_total_outage_raises_after_bounded_attempts():
    primary, fallback = Scripted(DOWN), Scripted(DOWN)
    llm, sleeps = client(primary, fallback)
    with pytest.raises(LLMUnavailable, match="fallback: Exception: 503"):
        llm.invoke([], Trace("alice", "t1", "q"))
    assert primary.calls == fallback.calls == settings.llm_retries + 1
    assert len(sleeps) == 2 * settings.llm_retries and max(sleeps) <= 9


def test_misconfiguration_is_reported_as_unavailable():
    def no_key(name, with_tools):
        raise ValueError("GOOGLE_API_KEY is missing")

    with pytest.raises(LLMUnavailable, match="GOOGLE_API_KEY is missing"):
        GeminiClient([], models=["m"], factory=no_key).invoke([], Trace("alice", "t1", "q"))
