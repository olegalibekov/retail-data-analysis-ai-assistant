"""Traces and the metrics computed from them."""

from agent.observability import Trace, load_traces, summarise


def test_trace_is_written_and_summarised(tmp_path):
    path = tmp_path / "logs" / "traces.jsonl"
    for outcome in ("ok", "sql_error"):
        trace = Trace("alice", "t1", "question", path=path)
        with trace.span("agent", kind="node"), trace.span("llm", kind="llm", model="m") as span:
            span.update(tokens_in=100, tokens_out=20)
        with trace.span("tools", kind="node"), trace.span("run_sql", kind="tool") as span:
            span.update(outcome=outcome, bytes=2_000_000, sql="SELECT 1")
        if outcome == "sql_error":
            trace.mark("degraded")
            trace.event("guard_block", guard="pii_column")
        record = trace.finish(answer="done")
        assert record["spans"][1]["parent"] == 0 and record["spans"][3]["parent"] == 2
    path.open("a").write("{torn line\n")

    records = load_traces(path)
    assert [r["status"] for r in records] == ["ok", "degraded"]
    summary = summarise(records)
    assert summary["turns"] == 2 and summary["answered_ok_rate"] == 0.5 and summary["degraded_rate"] == 0.5
    assert summary["sql_first_pass_rate"] == 0.5 and summary["sql_failure_rate"] == 0.5
    assert summary["tokens_in"] == 200 and summary["guard_blocks"] == 1 and summary["gb_scanned"] == 0.004
    assert summarise([]) == {"turns": 0}


def test_unwritable_trace_file_does_not_fail_the_turn(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    record = Trace("alice", "t1", "q", path=blocker / "traces.jsonl").finish(answer="still answered")
    assert record["answer"] == "still answered"


def test_span_records_errors_and_reraises():
    trace = Trace("alice", "t1", "q")
    try:
        with trace.span("boom", kind="tool"):
            raise ValueError("bad")
    except ValueError:
        pass
    assert trace.spans[0]["status"] == "error" and "ValueError: bad" in trace.spans[0]["error"]
    trace.mark("degraded")
    trace.mark("ok")
    assert trace.status == "degraded"
