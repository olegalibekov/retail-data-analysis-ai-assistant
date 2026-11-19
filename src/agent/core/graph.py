"""The agent as a LangGraph state machine.

    START -> input_guard -> agent <-> tools -> (confirm_delete) -> agent -> output_guard -> END

- input_guard    cheap deterministic checks; retrieves golden examples
- agent          one model call (bounded per turn; the last one must answer)
- tools          executes tool calls; SQL goes through sql_guard, results through pii
- confirm_delete pauses with interrupt() until the user confirms a deletion
- output_guard   final PII scan of the answer
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.core.nodes import Nodes
from agent.core.state import AgentState, TurnContext


def _after_guard(state: AgentState) -> str:
    return END if state.get("blocked") else "agent"


def _after_agent(state: AgentState) -> str:
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "output_guard"


def _after_tools(state: AgentState) -> str:
    return "confirm_delete" if state.get("pending_delete") else "agent"


def build_graph(nodes: Nodes, checkpointer=None):
    graph = StateGraph(AgentState, context_schema=TurnContext)
    graph.add_node("input_guard", nodes.input_guard)
    graph.add_node("agent", nodes.agent)
    graph.add_node("tools", nodes.tools)
    graph.add_node("confirm_delete", nodes.confirm_delete)
    graph.add_node("output_guard", nodes.output_guard)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", _after_guard, ["agent", END])
    graph.add_conditional_edges("agent", _after_agent, ["tools", "output_guard"])
    graph.add_conditional_edges("tools", _after_tools, ["confirm_delete", "agent"])
    graph.add_edge("confirm_delete", "agent")
    graph.add_edge("output_guard", END)
    return graph.compile(checkpointer=checkpointer)
