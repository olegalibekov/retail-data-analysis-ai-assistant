"""Model access with bounded retries and a fallback model.

Order of defence against provider trouble (rate limits, outages, timeouts):
retry the primary model a few times with backoff, honouring the server's
retry hint; then do the same with the fallback model, which has its own quota;
then raise LLMUnavailable, which the agent node turns into a polite message.
A timeout is not retried on the same model. Every attempt is a trace span.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage

from agent.config import settings
from agent.observability import Trace

log = logging.getLogger(__name__)

# (model name, tools enabled) -> object with .invoke(messages) -> AIMessage
ModelFactory = Callable[[str, bool], Any]

_TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
_TRANSIENT_TEXT = re.compile(
    r"RESOURCE_EXHAUSTED|UNAVAILABLE|DEADLINE_EXCEEDED|INTERNAL|overloaded|timed? ?out|timeout|connection|\b(429|500|502|503|504)\b",
    re.I,
)
_TIMEOUT_TEXT = re.compile(r"DEADLINE_EXCEEDED|timed? ?out|timeout|\b(408|504)\b", re.I)
_RETRY_HINT = re.compile(r"retry in ([\d.]+)\s*s|retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", re.I)


class LLMUnavailable(Exception):
    """No model could answer after retries and fallback."""


def classify(error: Exception) -> tuple[bool, float | None]:
    """(is it worth retrying the same model, seconds the server asked us to wait).

    Rate limits and overload are worth a short wait. A timeout is not: the user has
    already waited a full timeout, so the next attempt goes to the other model.
    """
    text = f"{type(error).__name__}: {error}"
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    timed_out = code in (408, 504) or isinstance(error, TimeoutError) or bool(_TIMEOUT_TEXT.search(text))
    transient = not timed_out and (code in _TRANSIENT_CODES or bool(_TRANSIENT_TEXT.search(text)))
    hint = _RETRY_HINT.search(text)
    return transient, float(hint.group(1) or hint.group(2)) if hint else None


class GeminiClient:
    def __init__(
        self,
        tools: Sequence[Any],
        models: Sequence[str] | None = None,
        factory: ModelFactory | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.tools = list(tools)
        self.models = tuple(dict.fromkeys(models or (settings.model, settings.fallback_model)))
        self._factory = factory or self._gemini
        self._sleep = sleep
        self._runnables: dict[tuple[str, bool], Any] = {}

    def _gemini(self, model: str, with_tools: bool) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI

        # max_retries=0: retries are done here, where they are bounded and traced.
        llm = ChatGoogleGenerativeAI(model=model, max_retries=0, timeout=settings.llm_timeout_s)
        # Tools stay declared even when disabled, so earlier tool calls in the history remain valid.
        return llm.bind_tools(self.tools, tool_choice="auto" if with_tools else "none")

    def _runnable(self, model: str, with_tools: bool) -> Any:
        key = (model, with_tools)
        if key not in self._runnables:
            self._runnables[key] = self._factory(model, with_tools)
        return self._runnables[key]

    def invoke(self, messages: list, trace: Trace, use_tools: bool = True) -> AIMessage:
        last_error = "no model configured"
        for index, model in enumerate(self.models):
            for attempt in range(1, settings.llm_retries + 2):
                with trace.span(
                    "llm", kind="llm", label="Processing", model=model, attempt=attempt, fallback=index > 0
                ) as span:
                    try:
                        reply = self._runnable(model, use_tools).invoke(messages)
                    except Exception as e:
                        last_error = f"{model}: {type(e).__name__}: {e}"[:400]
                        transient, hinted = classify(e)
                        span.update(status="error", error=last_error, transient=transient)
                        log.warning("Model call failed (%s, attempt %d): %s", model, attempt, last_error)
                    else:
                        usage = getattr(reply, "usage_metadata", None) or {}
                        span.update(
                            tokens_in=usage.get("input_tokens", 0),
                            tokens_out=usage.get("output_tokens", 0),
                            tool_calls=[call["name"] for call in reply.tool_calls],
                        )
                        return reply
                if not transient or attempt > settings.llm_retries:
                    break
                wait = hinted if hinted is not None else min(2**attempt, 8) + random.random()
                if wait > settings.llm_max_wait_s:
                    break  # e.g. daily quota exhausted: waiting will not help, the other model might
                trace.progress(f"The model is busy, retrying in {wait:.0f}s")
                self._sleep(wait)
        raise LLMUnavailable(last_error)
