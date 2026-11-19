"""Graph nodes. Each is one small step with its own trace span.

Failures of the model, the warehouse or a tool become state (a message the model
or the user can read), never an exception that reaches the UI.
"""

from __future__ import annotations

import functools
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from agent.config import settings
from agent.core.llm import GeminiClient, LLMUnavailable
from agent.core.prompts import build_system_prompt, prompt_fingerprint
from agent.core.state import AgentState, TurnContext, prepare_history, query_results, text_of, user_texts
from agent.core.tools import TOOLS, ToolContext, ToolResult, execute
from agent.guardrails.grounding import remove_unsupported, ungrounded_figures
from agent.guardrails.pii import scan_text
from agent.observability import Trace, loggable
from agent.stores.golden import GoldenStore
from agent.stores.reports import ReportStore
from agent.warehouse.bq_client import BigQueryRunner

MAX_INPUT_CHARS = 4000

# A cheap tripwire for obvious attacks: it saves a model call and feeds the
# guard-block metric. It is not the security boundary; sql_guard and pii are.
_INJECTION = re.compile(
    r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier|your)\s+(?:instructions|rules|prompts?|guardrails)"
    r"|disregard\s+.{0,30}(?:instructions|rules|guardrails)"
    r"|(?:reveal|show|print|repeat|output|leak)\b.{0,40}(?:system\s+prompt|your\s+instructions|hidden\s+prompt)"
    r"|(?:disable|bypass|turn\s+off|switch\s+off)\b.{0,30}(?:pii|masking|guardrails?|safety|filters?|restrictions?)"
    r"|developer\s+mode|jailbreak|\bDAN\s+mode",
    re.I,
)

REFUSAL = (
    "I can't help with that request. I can analyse sales, customers, products and orders for your "
    "product area, and manage your saved reports. What would you like to look at?"
)
TOO_LONG = (
    f"That message is too long for me to process (the limit is {MAX_INPUT_CHARS:,} characters). Please shorten it."
)
MODEL_UNREACHABLE = "I can't reach the language model right now, so I couldn't finish this request."
UNGROUNDED = (
    "[Automatic check, not from the user] These figures are not in any query result of this conversation: "
    "{figures}. Do not calculate them yourself. Either compute them with run_sql and use the returned values, or "
    "leave them out. Then give the complete answer again."
)
NO_ANSWER_GIVEN = "I wasn't able to produce an answer to that."
NO_ANSWER = f"{NO_ANSWER_GIVEN} Could you rephrase the question?"


def with_changes(message: str, changes: list[str]) -> str:
    """Complete a message about a failed turn with what the turn changed before it failed.

    A turn can save or delete reports and fail afterwards. The user must then read what was
    done, not that nothing was changed. `changes` are the receipts of the turn (Trace.changes).
    """
    if not changes:
        return f"{message} Nothing was changed."
    return f"{message} Before that, this was done: {'; '.join(changes)}."


def model_down(changes: list[str]) -> str:
    """What the user reads when no model could be reached."""
    return with_changes(MODEL_UNREACHABLE, changes) + " Please try again in a minute."


def no_answer(changes: list[str]) -> str:
    """What the user reads when the model returned nothing.

    After a change it names the change instead of asking for the question again: repeating a
    request that was carried out would save a second report or find nothing left to delete.
    """
    if not changes:
        return NO_ANSWER
    return with_changes(NO_ANSWER_GIVEN, changes)


def traced(fn):
    """Wrap a node in a trace span. A pause for user confirmation is not an error."""

    @functools.wraps(fn)
    def wrapper(self, state: AgentState, runtime: Runtime[TurnContext]) -> dict:
        with runtime.context.trace.span(fn.__name__, kind="node") as span:
            try:
                return fn(self, state, runtime)
            except GraphBubbleUp:
                span["status"] = "paused"
                raise

    return wrapper


def _is_last_call(calls_made: int) -> bool:
    """Whether the next model call is the last one the turn may make: it must answer, without tools."""
    return calls_made >= settings.max_model_calls - 1


def _text_only(reply: AIMessage) -> AIMessage:
    """The reply to a call made without tools. A model that asks for a tool anyway is not obeyed."""
    if not reply.tool_calls:
        return reply
    return AIMessage(content=text_of(reply))


def _record_outage(trace: Trace, error: LLMUnavailable) -> None:
    trace.event("model_unavailable", detail=loggable(str(error), 300))
    trace.mark("degraded")


def _tool_message(call: dict, result: ToolResult) -> ToolMessage:
    """A tool result as a message of the conversation. Rows of a query are marked as evidence for figures."""
    marks = {"evidence": True} if result.evidence else {}
    return ToolMessage(content=result.content, tool_call_id=call["id"], name=call["name"], additional_kwargs=marks)


def _deletion_receipt(deleted: list[int], listed: dict[int, dict]) -> str:
    """What a confirmed deletion did: the reports the store deleted, with the titles the user saw in the preview."""
    named = ", ".join(f"#{report_id} “{listed[report_id]['title']}”" for report_id in deleted)
    receipt = f"Deleted {len(deleted)} report(s): {named}"
    gone = len(listed) - len(deleted)
    if gone:
        receipt += f"; {gone} listed report(s) no longer existed"
    return receipt


class Nodes:
    def __init__(self, llm: GeminiClient, bq: BigQueryRunner, reports: ReportStore, golden: GoldenStore):
        self.llm, self.bq, self.reports, self.golden = llm, bq, reports, golden

    @traced
    def input_guard(self, state: AgentState, runtime: Runtime[TurnContext]) -> dict:
        trace = runtime.context.trace
        question = text_of(state["messages"][-1])
        problem = None
        if len(question) > MAX_INPUT_CHARS:
            problem = "too_long"
        elif _INJECTION.search(question):
            problem = "prompt_injection"
        if problem:
            trace.event("guard_block", guard=problem)
            trace.mark("blocked")
            return {"messages": [AIMessage(content=TOO_LONG if problem == "too_long" else REFUSAL)], "blocked": True}
        examples = self.golden.retrieve(question)
        # Short follow-ups often contain no metric keyword. Recover examples
        # from recent user context, without carrying them across explicit topic changes.
        if not examples and re.match(r"^(yes\b|and\b|what about\b|same\b|those\b|that\b)", question.strip(), re.I):
            for previous in reversed(user_texts(state["messages"])[-4:-1]):
                examples = self.golden.retrieve(previous)
                if examples:
                    break
        trace.current["examples"] = [e["question"] for e in examples]
        return {"examples": examples}

    @traced
    def agent(self, state: AgentState, runtime: Runtime[TurnContext]) -> dict:
        ctx = runtime.context
        calls = state.get("model_calls", 0)
        final = _is_last_call(calls)
        if final:
            ctx.trace.event("step_limit")
            ctx.trace.mark("degraded")
        ctx.trace.meta.update(
            prompt=prompt_fingerprint(),
            models=list(getattr(self.llm, "models", ())),
            limits={
                "model_calls": settings.max_model_calls,
                "sql_failures": settings.max_sql_failures,
                "sql_queries": settings.max_queries_per_turn,
                "bq_bytes": settings.bq_max_bytes_billed,
                "bq_bytes_per_turn": settings.bq_max_bytes_per_turn,
            },
        )
        prompt = [
            SystemMessage(content=build_system_prompt(ctx.user, state.get("examples") or [], final_step=final)),
            *prepare_history(state["messages"]),
        ]
        try:
            reply = self.llm.invoke(prompt, ctx.trace, use_tools=not final)
        except LLMUnavailable as error:
            _record_outage(ctx.trace, error)
            return {"messages": [AIMessage(content=model_down(ctx.trace.changes))], "model_calls": calls + 1}
        if final:
            reply = _text_only(reply)
        reply, update = self._grounded(reply, prompt, state, ctx, calls + 1)
        if not reply.tool_calls and not text_of(reply).strip():
            ctx.trace.event("empty_answer")
            ctx.trace.mark("degraded")
            reply = AIMessage(content=no_answer(ctx.trace.changes))
        return {"messages": [reply], **update}

    def _grounded(
        self, reply: AIMessage, prompt: list, state: AgentState, ctx: TurnContext, calls: int
    ) -> tuple[AIMessage, dict]:
        """The reply with every checked figure of a final answer backed by a query result, and the state it leaves.

        The model gets one repair round per turn, shared with save_report, if a model call is
        left for it. Whatever is unsupported after that is removed by code: the user never
        reads a checked figure that is not in a query result. `calls` counts the model calls of the
        turn, the one that produced `reply` included.
        """
        repair_used = state.get("grounding_checked", False)
        unsupported = self._ungrounded(reply, state)
        if unsupported:
            ctx.trace.event("ungrounded_figures", figures=unsupported, where="final_answer", after_repair=repair_used)
        if unsupported and not repair_used and calls < settings.max_model_calls:
            reply = self._repaired(reply, prompt, unsupported, ctx.trace, last_call=_is_last_call(calls))
            calls, repair_used = calls + 1, True
            unsupported = self._ungrounded(reply, state)
            if unsupported:
                ctx.trace.event("ungrounded_figures", figures=unsupported, where="final_answer", after_repair=True)
        if unsupported:
            reply = AIMessage(content=remove_unsupported(text_of(reply), unsupported))
            ctx.trace.event("figures_removed", figures=unsupported, where="final_answer")
            ctx.trace.mark("degraded")
        return reply, {"model_calls": calls, "grounding_checked": repair_used}

    def _repaired(self, draft: AIMessage, prompt: list, figures: list[str], trace: Trace, last_call: bool) -> AIMessage:
        """The model's second attempt at an answer with unsupported figures, or the draft if there is none.

        The draft and the note that asks for the correction are not kept in the conversation.
        """
        note = HumanMessage(content=UNGROUNDED.format(figures=", ".join(figures)))
        try:
            reply = self.llm.invoke([*prompt, AIMessage(content=text_of(draft)), note], trace, use_tools=not last_call)
        except LLMUnavailable as error:
            _record_outage(trace, error)
            return draft
        if last_call:
            reply = _text_only(reply)
        if not reply.tool_calls and not text_of(reply).strip():
            return draft  # an empty second attempt is not better than the first
        return reply

    @staticmethod
    def _ungrounded(reply: AIMessage, state: AgentState) -> list[str]:
        """Checked figures in a final answer that no query result of the conversation supports."""
        if reply.tool_calls:
            return []  # an ordinary tool step: nothing is shown to the user yet
        return ungrounded_figures(text_of(reply), query_results(state["messages"]))

    @traced
    def tools(self, state: AgentState, runtime: Runtime[TurnContext]) -> dict:
        ctx = runtime.context
        messages = state["messages"]
        evidence, texts = query_results(messages), user_texts(messages)
        failures, empties = state.get("sql_failures", 0), state.get("empty_results", 0)
        queries, scanned = state.get("sql_queries", 0), state.get("bytes_scanned", 0)
        # Every call of a step sees the repair round as it stood before the step, so two reports
        # saved in one step are both sent back once instead of the second being cut without a chance.
        repair_used = state.get("grounding_checked", False)
        repair_requested = False
        results, pending = [], []
        for call in messages[-1].tool_calls:
            label = TOOLS[call["name"]].label if call["name"] in TOOLS else None
            purpose = (call.get("args") or {}).get("purpose")
            with ctx.trace.span(
                call["name"], kind="tool", label=f"{label}: {purpose}" if label and purpose else label
            ) as span:
                result = execute(
                    call["name"],
                    call.get("args"),
                    ToolContext(
                        user=ctx.user,
                        thread_id=ctx.thread_id,
                        trace=ctx.trace,
                        bq=self.bq,
                        reports=self.reports,
                        span=span,
                        sql_failures=failures,
                        empty_results=empties,
                        sql_queries=queries,
                        bytes_scanned=scanned,
                        grounding_checked=repair_used,
                        evidence=evidence,
                        user_texts=texts,
                    ),
                )
            failures = settings.max_sql_failures if result.stop_sql else failures + result.sql_failed
            empties += result.empty
            queries += result.sql_sent
            scanned += result.bytes_scanned
            repair_requested = repair_requested or result.grounding_checked
            if result.evidence:
                evidence = [*evidence, result.content]
            if result.pending_delete:
                pending.append({"tool_call_id": call["id"], **result.pending_delete})
            else:
                results.append(_tool_message(call, result))
        return {
            "messages": results,
            "sql_failures": failures,
            "empty_results": empties,
            "sql_queries": queries,
            "bytes_scanned": scanned,
            "grounding_checked": repair_used or repair_requested,
            "pending_delete": pending or None,
        }

    @traced
    def confirm_delete(self, state: AgentState, runtime: Runtime[TurnContext]) -> dict:
        """Human-in-the-loop gate for deletions.

        The report ids were resolved by code in the tools node and are read here
        from checkpointed state, so what the user confirms is exactly what is
        deleted. The decision comes from the UI through interrupt(); the model
        cannot supply it. Only an explicit `confirmed: True` deletes.

        What was deleted is recorded as a receipt before the model is asked to
        describe the outcome, so the user learns of it even if that call fails.
        """
        ctx = runtime.context
        pending = state["pending_delete"] or []
        reports = {}
        for request in pending:
            for report in request["reports"]:
                reports[report["id"]] = report
        ids = sorted(reports)
        decision = interrupt(
            {
                "action": "delete_reports",
                "criteria": "; ".join(request["criteria"] for request in pending),
                "reports": [reports[report_id] for report_id in ids],
            }
        )
        if isinstance(decision, dict) and decision.get("confirmed") is True:
            deleted = self.reports.soft_delete(ctx.user.id, ids)
            outcome = self._deletion_outcome(ctx, deleted, reports)
        else:
            ctx.trace.event("delete_declined", ids=ids, user_id=ctx.user.id)
            outcome = "DECLINED by the user in the confirmation dialog. Nothing was deleted. Tell them briefly that no report was deleted."
        return {
            "messages": [
                ToolMessage(content=outcome, tool_call_id=request["tool_call_id"], name="delete_reports")
                for request in pending
            ],
            "pending_delete": None,
        }

    @staticmethod
    def _deletion_outcome(ctx: TurnContext, deleted: list[int], listed: dict[int, dict]) -> str:
        """Record a confirmed deletion and say to the model what it did, with the same facts as the receipt."""
        titles = [listed[report_id]["title"] for report_id in deleted]
        ctx.trace.event("reports_deleted", ids=deleted, titles=titles, user_id=ctx.user.id)
        if not deleted:  # every listed report was deleted elsewhere while the dialog was open
            return (
                "CONFIRMED by the user in the confirmation dialog, but none of the listed reports existed any more, "
                "so nothing was deleted. Tell them briefly."
            )
        receipt = _deletion_receipt(deleted, listed)
        ctx.trace.change(receipt)
        return f"CONFIRMED by the user in the confirmation dialog. {receipt}. Tell them briefly."

    @traced
    def output_guard(self, state: AgentState, runtime: Runtime[TurnContext]) -> dict:
        """Last PII check on the text the user will see.

        A person's name may come from a query result (a brand named after its founder) or
        from the user's own words. One that is in neither was made up, and it is redacted.
        """
        messages = state["messages"]
        last = messages[-1]
        clean, found = scan_text(text_of(last), known=[*query_results(messages), *user_texts(messages)])
        if not found:
            return {}
        runtime.context.trace.event("pii_redacted", where="final_answer", detail=found)
        return {"messages": [last.model_copy(update={"content": clean})]}  # same id: replaces the original
