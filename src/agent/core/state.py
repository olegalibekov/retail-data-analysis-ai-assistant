"""Graph state, per-turn context and message helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from agent.config import User, settings
from agent.observability import Trace


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # the conversation (checkpointed)
    # Reset at the start of every turn:
    model_calls: int
    sql_failures: int
    empty_results: int
    sql_queries: int  # queries sent to BigQuery
    bytes_scanned: int  # by those queries
    grounding_checked: bool  # the one repair round for unsupported figures was used
    blocked: bool
    examples: list[dict]  # golden trios retrieved for this question
    pending_delete: list[dict] | None  # deletions waiting for the user's confirmation


@dataclass
class TurnContext:
    """Per-turn runtime context. Not checkpointed; passed on every invoke."""

    user: User
    thread_id: str
    trace: Trace


def new_turn(question: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=question)],
        "model_calls": 0,
        "sql_failures": 0,
        "empty_results": 0,
        "sql_queries": 0,
        "bytes_scanned": 0,
        "grounding_checked": False,
        "blocked": False,
        "examples": [],
        "pending_delete": None,
    }


def text_of(message: AnyMessage) -> str:
    """Plain text of a message whose content may be a string or a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    texts = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif block.get("type") == "text":
            texts.append(block.get("text", ""))
    return "".join(texts)


def query_results(messages: list[AnyMessage]) -> list[str]:
    """The query results of the whole conversation, as the model received them.

    They are the only evidence for a figure: the tools node marks a tool message as evidence
    when it carries the rows of a successful query. Errors, report listings, the user's
    questions and the model's own earlier answers are not evidence.
    """
    return [
        text_of(message)
        for message in messages
        if isinstance(message, ToolMessage) and message.additional_kwargs.get("evidence")
    ]


def user_texts(messages: list[AnyMessage]) -> list[str]:
    """What the user wrote in the whole conversation."""
    return [text_of(message) for message in messages if isinstance(message, HumanMessage)]


_STUB = "Not executed: the previous step was interrupted."
_UNANSWERED = "(No answer was produced for the previous message because of an error.)"


def prepare_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The conversation as the model should see it.

    - Only recent turns (bounded by MAX_HISTORY_MESSAGES, cut at a user message).
    - Leftovers of a turn that was interrupted or failed are repaired, because
      the model API rejects them: a tool call without a result gets a stub
      result, and a user message that never got an answer gets a stub answer.
    """
    humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    start = 0
    if len(messages) > settings.max_history_messages and humans:
        start = next((i for i in humans if len(messages) - i <= settings.max_history_messages), humans[-1])

    out: list[AnyMessage] = []
    i = start
    while i < len(messages):
        message = messages[i]
        if (
            isinstance(message, HumanMessage)
            and out
            and not (isinstance(out[-1], AIMessage) and not out[-1].tool_calls)
        ):
            out.append(AIMessage(content=_UNANSWERED))  # the previous turn never produced an answer
        out.append(message)
        i += 1
        if isinstance(message, AIMessage) and message.tool_calls:
            answered = set()
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                answered.add(messages[i].tool_call_id)
                out.append(messages[i])
                i += 1
            out.extend(
                ToolMessage(content=_STUB, tool_call_id=call["id"], name=call["name"])
                for call in message.tool_calls
                if call["id"] not in answered
            )
    return out
