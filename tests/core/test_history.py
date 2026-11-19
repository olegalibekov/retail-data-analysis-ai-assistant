"""Preparation of the conversation history sent to the model."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.config import settings
from agent.core.state import prepare_history
from conftest import ai, tool_call


def test_history_stubs_dangling_tool_calls():
    a, b = tool_call("run_sql", sql="SELECT 1"), tool_call("run_sql", sql="SELECT 2")
    messages = [HumanMessage("q1"), ai("", a, b), ToolMessage("1 row", tool_call_id=a["id"]), HumanMessage("q2")]
    out = prepare_history(messages)
    assert [type(m).__name__ for m in out] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "ToolMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert out[3].tool_call_id == b["id"] and "interrupted" in out[3].content  # missing tool result
    assert "No answer was produced" in out[4].content  # missing final answer


def test_history_is_windowed_at_a_user_message(monkeypatch):
    def turn(n):
        call = tool_call("run_sql", sql="SELECT 1")
        return [HumanMessage(f"q{n}"), ai("", call), ToolMessage("rows", tool_call_id=call["id"]), AIMessage(f"a{n}")]

    messages = [m for n in range(4) for m in turn(n)]
    assert prepare_history(messages) == messages  # short conversations go to the model unchanged

    monkeypatch.setattr(settings, "max_history_messages", 9)
    out = prepare_history(messages)
    assert isinstance(out[0], HumanMessage) and out[0].content == "q2" and len(out) == 8
