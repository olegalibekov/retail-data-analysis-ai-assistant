"""Per-turn tracing and agent metrics.

One Trace per user turn. Graph nodes, model calls and tool calls each record a
span (name, kind, timing, attributes, error); notable incidents are recorded as
events (guard blocks, redactions, deletions, outages). What the turn changed for
the user (a saved report, a deletion) is recorded as a receipt, see Trace.change.
A finished trace is one JSON line in logs/traces.jsonl, which is what `/trace`
and `/metrics` read.

Result rows are never written to a trace, only row counts and column names.
Free text goes through `loggable` first.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from agent.guardrails.pii import scan_text

log = logging.getLogger(__name__)

_SEVERITY = {"ok": 0, "blocked": 1, "cancelled": 1, "degraded": 2, "error": 3}


def loggable(text: str, limit: int) -> str:
    """Free text (an error message, say) as it may be written to a trace: scanned, then cut to `limit`.

    Nothing in a log has to stay readable as a name, so the scan is the strictest one. The text
    is cut after the scan, because half of an e-mail address would no longer be recognised.
    """
    return scan_text(text, known=())[0][:limit]


@dataclass
class Trace:
    user_id: str
    thread_id: str
    question: str
    path: Path | None = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "ok"
    spans: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # what produced the answer: prompt versions, models, limits
    listener: Callable[[str], None] | None = None  # progress line for the UI

    def __post_init__(self) -> None:
        self._t0 = time.monotonic()
        self._stack: list[int] = []
        self.started_at = datetime.now(UTC).isoformat(timespec="seconds")

    def _ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    @property
    def current(self) -> dict:
        """The innermost open span."""
        return self.spans[self._stack[-1]]

    def progress(self, label: str) -> None:
        if self.listener:
            self.listener(label)

    @contextmanager
    def span(self, name: str, kind: str, label: str | None = None, **attrs: Any) -> Iterator[dict]:
        """Record a timed step. The yielded dict can be enriched by the caller."""
        span: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "parent": self._stack[-1] if self._stack else None,
            "start_ms": self._ms(),
            **attrs,
        }
        self.spans.append(span)
        self._stack.append(len(self.spans) - 1)
        if label:
            self.progress(label)
        try:
            yield span
        except BaseException as e:
            if "status" not in span:
                span["status"] = "error"
                span["error"] = loggable(f"{type(e).__name__}: {e}", 500)
            raise
        finally:
            span.setdefault("status", "ok")
            span["duration_ms"] = self._ms() - span["start_ms"]
            self._stack.pop()

    def event(self, kind: str, **attrs: Any) -> None:
        self.events.append({"kind": kind, "at_ms": self._ms(), **attrs})

    def change(self, text: str) -> None:
        """Record a receipt: something this turn changed for the user, in words they can read.

        A turn can save or delete reports and fail afterwards. The message about the failure is
        completed from these receipts, so it cannot claim that nothing was changed.
        """
        self.event("change", text=text)

    @property
    def changes(self) -> list[str]:
        """The receipts of this turn so far, in the order the changes were made."""
        return [event["text"] for event in self.events if event["kind"] == "change"]

    def mark(self, status: str) -> None:
        """Downgrade the turn status; the worst status wins."""
        if _SEVERITY[status] > _SEVERITY[self.status]:
            self.status = status

    def finish(self, answer: str = "", error: str | None = None) -> dict:
        if error:
            error = loggable(error, 500)
        llm = [span for span in self.spans if span["kind"] == "llm"]
        sql = [span for span in self.spans if span["kind"] == "tool" and span["name"] == "run_sql"]
        # Queries whose SQL was actually judged; an outage says nothing about SQL quality.
        judged = [span for span in sql if span.get("outcome") in ("ok", "empty", "sql_error", "rejected")]
        record = {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "status": self.status,
            "duration_ms": self._ms(),
            "question": self.question,
            "answer": answer,
            "error": error,
            "model_calls": len(llm),
            "model_errors": sum(span["status"] == "error" for span in llm),
            "fallback_used": any(span.get("fallback") and span["status"] == "ok" for span in llm),
            "tokens_in": sum(span.get("tokens_in", 0) for span in llm),
            "tokens_out": sum(span.get("tokens_out", 0) for span in llm),
            "sql_queries": len(sql),
            "sql_first_pass_ok": judged[0]["outcome"] in ("ok", "empty") if judged else None,
            "sql_failures": sum(span.get("outcome") in ("sql_error", "rejected") for span in sql),
            "sql_empty": sum(span.get("outcome") == "empty" for span in sql),
            "bytes_scanned": sum(span.get("bytes", 0) for span in sql),
            "guard_blocks": sum(event["kind"] == "guard_block" for event in self.events),
            "pii_redactions": sum(event["kind"] == "pii_redacted" for event in self.events),
            "changes": self.changes,
            "meta": self.meta,
            "spans": self.spans,
            "events": self.events,
        }
        if self.path is not None:
            try:  # losing a trace must never fail the user's request
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except OSError:
                log.warning("Could not write trace %s", self.trace_id, exc_info=True)
        return record


def load_traces(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn write must not break /metrics
    return records


def summarise(turns: list[dict]) -> dict:
    """Agent-level metrics over a set of turns (finished trace records)."""
    count = len(turns)
    if not count:
        return {"turns": 0}

    def total(field: str) -> int:
        return sum(turn[field] for turn in turns)

    def rate(part: int, whole: int) -> float | None:
        return round(part / whole, 3) if whole else None

    def share_with_status(status: str) -> float | None:
        return rate(sum(turn["status"] == status for turn in turns), count)

    latencies = sorted(turn["duration_ms"] for turn in turns)
    p95_index = min(count - 1, int(0.95 * count))
    judged = [turn for turn in turns if turn["sql_first_pass_ok"] is not None]

    return {
        "turns": count,
        "answered_ok_rate": share_with_status("ok"),
        "degraded_rate": share_with_status("degraded"),
        "error_rate": share_with_status("error"),
        "blocked_rate": share_with_status("blocked"),
        "latency_p50_s": round(median(latencies) / 1000, 1),
        "latency_p95_s": round(latencies[p95_index] / 1000, 1),
        "model_calls_per_turn": round(total("model_calls") / count, 2),
        "model_error_rate": rate(total("model_errors"), total("model_calls")),
        "fallback_model_rate": rate(sum(bool(turn["fallback_used"]) for turn in turns), count),
        "tokens_in": total("tokens_in"),
        "tokens_out": total("tokens_out"),
        "sql_queries": total("sql_queries"),
        "sql_first_pass_rate": rate(sum(turn["sql_first_pass_ok"] for turn in judged), len(judged)),
        "sql_failure_rate": rate(total("sql_failures"), total("sql_queries")),
        "sql_empty_rate": rate(total("sql_empty"), total("sql_queries")),
        "gb_scanned": round(total("bytes_scanned") / 1e9, 3),
        "guard_blocks": total("guard_blocks"),
        "pii_redactions": total("pii_redactions"),
    }
