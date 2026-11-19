"""The report store: ownership, filters and soft delete."""

from agent.stores.reports import ReportStore


def test_reports_are_private_to_their_owner(tmp_path):
    store = ReportStore(tmp_path / "r.db")
    mine = store.save("alice", "t1", "Q1 for Client X", "body")
    theirs = store.save("bob", "t1", "Q1 for Client X", "body")
    assert [r.id for r in store.find("alice")] == [mine.id]
    assert store.find("alice", ids=[theirs.id]) == []
    assert store.soft_delete("alice", [mine.id, theirs.id]) == [mine.id]  # bob's id is ignored
    assert [r.id for r in store.find("bob")] == [theirs.id]


def test_report_filters_and_soft_delete(tmp_path):
    store = ReportStore(tmp_path / "r.db")
    a = store.save("alice", "t1", "Client X review", "…")
    b = store.save("alice", "t2", "Monthly revenue", "mentions client x in the body")
    c = store.save("alice", "t2", "100% growth_plan", "…")
    assert [r.id for r in store.find("alice", mentions="CLIENT X")] == [a.id, b.id]
    assert [r.id for r in store.find("alice", thread_id="t2")] == [b.id, c.id]
    assert [r.id for r in store.find("alice", thread_id="t2", mentions="client x")] == [b.id]
    assert store.find("alice", mentions="%") == [c] and store.find("alice", mentions="_") == [
        c
    ]  # LIKE wildcards are literal
    assert store.find("alice", ids=[]) == []

    assert store.soft_delete("alice", [a.id, b.id]) == [a.id, b.id]
    assert [r.id for r in store.find("alice")] == [c.id]
    assert store.soft_delete("alice", [a.id]) == []  # already deleted
