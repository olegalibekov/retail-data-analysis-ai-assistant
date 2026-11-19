# Technical explanation

How the system works, why it is built this way, and how each requirement is met. The architecture and diagrams are in [ARCHITECTURE.md](ARCHITECTURE.md); setup and usage are in the [README](../README.md).

Contents: [1 Approach](#1-approach) · [2 Business rules](#2-confirmed-business-rules-and-remaining-assumptions) · [3 Choices](#3-choices) · [4 One turn](#4-how-one-turn-works) · [5 Requirements](#5-the-requirements-one-by-one) · [6 Error handling](#6-error-handling-and-fallbacks) · [7 Threat model](#7-threat-model) · [8 Cost](#8-cost-controls) · [9 Extending](#9-extending-the-system) · [10 Tests](#10-what-is-tested) · [11 Limitations](#11-limitations-and-next-steps)

## 1. Approach

Three ideas shape the design.

1. **The model proposes; code decides.** The model writes SQL and requests actions. What runs against the warehouse, what the user sees and what gets deleted is decided by deterministic code. Warehouse access and deletion authorization do not depend on the model obeying its prompt; free-text safeguards remain heuristic.
2. **Every loop is bounded.** Self-correction is useful only if it cannot run away. Each turn has a fixed budget of model calls and failed queries, and model calls and query jobs have configurable timeouts. SDK transport retries can add latency beyond the query-job deadline.
3. **Every turn is explainable afterwards.** One trace per turn records each step, model call, query, retry and guard decision, and its id is shown to the user.

The agent is a small LangGraph state machine: `input_guard → agent ⇄ tools → (confirm_delete) → output_guard`. It has four tools: `run_sql`, `save_report`, `list_reports` and `delete_reports`.

## 2. Confirmed business rules and remaining assumptions

The following defines access, customer privacy, metric definitions and report ownership. Soft deletion and a global persona are design choices.

| Business rule | Implementation decision |
|---|---|
| How is "products related to a user" defined? | Brand only: a manager has one or more brands; only the CEO sees all brands. Managers cannot compare against company-wide totals or growth. Brand assignments in `config/users.yaml` use exact dataset values directly. Production uses an entitlement table |
| May customer-level results show a customer id? | Yes: internal `user_id` is used here; a stable pseudonymous ID is also acceptable. Spend is computed from authorized brands only |
| Are demographic aggregates (age, gender, state, city) acceptable? | Group-level analysis only: do not associate age, gender, location or other demographic attributes with an individual ID. Names, e-mail, street address, postal code and exact coordinates remain unavailable |
| How is revenue defined? | Sum of item sale prices excluding cancelled and returned items |
| Who may delete reports? | Only the owner. Deletion is soft, so it can be reversed |
| How is churn defined? | Default: no purchase for 90 days within authorized brands. Every churn answer states the window and scope; user overrides such as 60 days apply consistently to SQL and the active-customer denominator |
| Is the persona global or per team? | Global, set by the business |
| Data warehouse | BigQuery |

## 3. Choices

### Framework: LangGraph

The requirements map directly onto what LangGraph provides:

- **Confirmation of destructive actions** needs the run to stop, survive, and continue exactly where it stopped. `interrupt()` with a checkpointer does this, and with a durable checkpointer it also survives a process restart.
- **Bounded self-correction** is easier to reason about as explicit edges and counters in state than as a hidden loop inside a prebuilt agent.
- **Observability** needs one place where every step passes. Nodes are that place.
- **Extensibility**: a new capability is a new tool, or a new node if it needs its own control flow, without touching the rest.

I deliberately did not use the prebuilt ReAct agent. Five small nodes written by hand are short, and they make the guard, the budget and the confirmation gate visible in the graph instead of hidden in callbacks. Alternatives considered: Google ADK (good Gemini and BigQuery integration, but less explicit control flow), and PydanticAI (pleasant typing, but more orchestration to write by hand for interrupts and checkpoints).

**My experience with LangGraph:** I have built and operated LangGraph agents in production, so the choices here (hand-written nodes, checkpointed interrupts for approvals, explicit budgets in state) come from running such systems, not only from the documentation.

### Models

- **Primary: `gemini-3.5-flash-lite`** (`GEMINI_MODEL`). The prototype uses this model for tool calling, SQL generation and report writing. Its behavior is checked against live reference queries rather than assumed from model size.
- **Fallback: `gemini-3.1-flash-lite`** (`GEMINI_FALLBACK_MODEL`). The wrapper tries it after bounded primary retries. It can recover from some model-specific failures; it is not a guarantee against shared quota or provider outages.
- **Latency and quality tradeoff.** Start with a model suited to interactive tool use. Adopt a stronger model or task-specific routing only if held-out evaluations show enough improvement to justify the added latency and cost. No unrecorded comparative benchmark is claimed.
- **Sampling parameters** are left at SDK defaults. Prompt rules, golden examples and deterministic checks reduce errors but do not make model output deterministic. Traces record model names and prompt hashes.
- **Upgrade candidates:** Gemini 4 Argon for reasoning quality, or Gemini 3.8 Flash when cost is more important; retain the same guards and validate on this project's evaluations before switching. On 19 November 2026, the configured key hit quota limits for Argon, while Flash 3.8 returned timeouts/service-unavailable errors, so neither was adopted as a validated default. Model availability and pricing are listed in [Google's official documentation](https://aistudio.google.com/docs/models).
- All model access goes through `core/llm.py`. Gemini model names are configuration; another provider would require adapting that wrapper and rerunning evaluations.

### Cloud services

See the table in [ARCHITECTURE.md, section 3](ARCHITECTURE.md#3-building-blocks). In short: Cloud Run for the stateless agent service, Vertex AI for Gemini, BigQuery with authorized views and policy tags for the data, one Cloud SQL Postgres for all application state (checkpoints, reports, audit, preferences, vector index), Cloud Storage for the Golden Knowledge bucket, Langfuse for traces, prompts and evaluation data, and Cloud Monitoring for alerts.

## 4. How one turn works

1. The CLI creates a `Trace` and invokes the graph with the user's message, the conversation id and a per-turn context (user, trace).
2. `input_guard` rejects over-long input and obvious attacks without a model call, and retrieves similar golden examples.
3. `agent` builds the system prompt (fixed rules, schema, business definitions, persona, user scope, examples), prepares the history and calls Gemini with the tools.
4. If the model asks for tools, `tools` runs them. `run_sql` goes through the SQL guard, a free dry run, the capped execution and result masking. Tool results, including failures, go back to the model as text.
5. Steps 3 and 4 repeat until the model answers or the budget is used. The last allowed model call is made with tools disabled.
6. A deletion request takes the side path through `confirm_delete`, which pauses for the user.
7. `output_guard` scans the final text for personal data.
8. The CLI prints the answer and a one-line footer with the trace id, and the trace is written.

State is checkpointed after every step, which is what lets a turn pause for confirmation and continue exactly where it stopped. The prototype keeps checkpoints in memory; production uses Postgres, so a pause also survives a restart.

## 5. The requirements, one by one

### 5.1 Hybrid intelligence: the Golden Knowledge bucket (design, with a small working hook)

**Goal (production design).** Use expert-approved (question, SQL, analyst report) trios so the agent interprets new questions consistently. No real bucket or expert corpus was supplied. The local samples are authored examples; seeded retrieval is an optional prototype demonstration. The full format, lifecycle and scalable approval controls are defined in [Architecture section 8](ARCHITECTURE.md#8-golden-knowledge-life-cycle).

**At query time.**
1. The question is rewritten into a standalone form if it depends on earlier turns ("and for Texas?").
2. A hybrid search (embedding similarity plus keywords) over the trio index returns the best three to five matches. Before ranking, eligibility checks require active approved versions, compatible schema/metric definitions and valid dates. Every brand represented must be authorized for the user; only explicitly reviewed sanitized shared templates may cross scopes.
3. The matches are placed in the prompt as examples, with the instruction to adapt rather than copy. They give the model three things: the analysts' definitions ("underspending means revenue per customer"), proven query logic for this schema, and the expected report structure.
4. The model then writes its own SQL, which runs through the same guard as any other query. An example can never widen what a user sees.

**Keeping the bucket current.**
- *New trios from analysts*: submissions enter a candidate prefix and receive format, SQL, cost, privacy, scope and report-grounding checks. A domain analyst other than the author approves the exact content hash; sensitive metric/scope changes require a second approver. Only the publisher service may publish an approved immutable version, embed/index it and activate it for retrieval.
- *New trios from use*: answers that were rated well, saved as reports without edits, and pass an automated groundedness check become candidates. A candidate is never promoted automatically; it goes to an analyst review queue. This prevents the knowledge base from being poisoned or drifting.
- *Maintenance*: weekly and schema/metric-change jobs revalidate published trios and deactivate invalid versions; near-duplicate merges return through candidate review; trios that are never retrieved or lead to poor ratings are reviewed. When a definition changes (for example churn), trios carry a validity period rather than being silently overwritten.
- *Scalable review*: domain queues in Cloud SQL use leases, deduplication and impact/age priorities; Cloud Run workers use idempotent jobs, bounded retries and a dead-letter queue. Batch review retains a decision per candidate. Monitor queue age, workload and rejection rates; backlog never bypasses approval.
- The approved bucket is versioned and the index is derived. Catalog checks prevent stale index entries from serving revoked versions. Rollback uses only an approved, still-compatible version; edits invalidate previous approval.

**In the prototype.** `src/agent/stores/golden.py` ranks nine authored sample trios in `data/golden_trios.json` by word overlap and adds the best two to the prompt. This costs no model calls. For short follow-ups such as “yes” with no matching example, the input node searches up to three recent user questions for relevant examples. Explicit topic changes use their own retrieval results. Each sample has `provenance: sample`, and the prompt labels it as sample material rather than historical expert work. Its report field is illustrative analysis guidance, not observed results. The retrieved questions are recorded in the trace.

### 5.2 Safety and PII masking (built)

The rule is that personal data and out-of-scope rows must be unreachable, not merely forbidden. There are four layers; the first one carries the weight.

**Layer 1: the SQL guard** (`src/agent/guardrails/sql_guard.py`). Before any query runs, it is parsed with `sqlglot` in the BigQuery dialect and:

1. must be exactly one statement, and that statement a `SELECT` (CTEs and set operations allowed). Writes, DDL, scripts and procedure calls are rejected;
2. may reference only `orders`, `order_items`, `products` and `users`. Other tables, other datasets, `INFORMATION_SCHEMA`, table functions and recursive CTEs are rejected;
3. has every table reference redirected to a view of the same name that the guard itself prepends as a CTE (only the views the query needs, plus the ones those depend on):

```sql
WITH products    AS (SELECT <non-PII columns> FROM `…thelook_ecommerce.products` WHERE brand IN ('Calvin Klein Jeans')),
     order_items AS (SELECT … FROM `…order_items` WHERE product_id IN (SELECT id FROM products)),
     orders      AS (SELECT … FROM `…orders`      WHERE order_id IN (SELECT order_id FROM order_items)),
     users       AS (SELECT id, age, gender, state, country, city, traffic_source, created_at
                     FROM `…users` WHERE id IN (SELECT user_id FROM order_items))
<the model's query>
```

For restricted users, the orders view joins counts from scoped `order_items` and replaces `num_of_item` with that count; the SQL above abbreviates this projection.

The model's query can only see these views. PII columns do not exist in the `users` view, so references to `email` cannot reach the raw column. Whole-customer `SELECT *` is also rejected by the demographic guard because it pairs IDs with personal attributes. Rows outside the user's product scope are not in any view. The scope filter is generated by code from the user's entitlements and its values are escaped; the model never writes it. The SQL is regenerated from the parsed tree. Unsupported source structures are rejected; parser differences remain a reason to add independent warehouse enforcement in production.

As a courtesy, a query that names a PII column is rejected up front with a clear reason, which saves a round trip. That check is not the barrier; the view is.

**Aggregate-only demographics.** Before adding scoped views, the guard qualifies a copy of queries that read users/orders, expands stars and resolves aliases. Identifier-only cohort CTEs can carry customer keys, group purchases and compute gaps internally. The guard follows identifier aliases through CTEs and unions. Within scopes that read demographic attributes, individual identifiers may only appear in direct identifier-to-identifier join equalities, membership in an identifier-only cohort subquery, or plain `COUNT`/`COUNT(DISTINCT ...)`. Returning, transforming, grouping by or filtering on those identifiers is rejected. Demographic outputs must originate from aggregate/grouped queries; wrappers over those aggregates are allowed, but window counts alone are insufficient. Duplicate output aliases fail closed rather than losing lineage. Whole-row exports are rejected. This applies to the CEO too. This allows churn-cohort counts by traffic source while blocking individual filters inside those cohorts. Intermediate scopes that combine IDs with demographics remain conservatively restricted. This is not differential privacy: small-group inference is still a production concern.

**Layer 2: result masking** (`src/agent/guardrails/pii.py`). Before rows reach the model, columns with PII-like names are dropped and values that look like e-mail addresses, phone numbers or street addresses are redacted.

**Layer 3: output scan.** The final answer passes through the same redaction in `output_guard`. Report text is scanned when it is saved, and the question is scanned before it is written to a trace.

**Layer 4: scope of the assistant.** `input_guard` refuses over-long input and obvious attack phrases without calling the model. General off-topic refusal relies on the system prompt and is evaluated with live safety cases; it is not a deterministic guarantee. The SQL and report ownership controls operate independently of model compliance, while free-text PII scanning remains heuristic.

**Saved reports** are filtered by owner in every query of the store (`src/agent/stores/reports.py`), so one user cannot list, read or delete another's reports, whatever id the model passes.

**In production** the same guard stays, and the warehouse enforces the rules independently: authorized views and row-level access policies for scope, policy tags with no fine-grained read access for PII columns, read-only roles, and Sensitive Data Protection as a second scan of outgoing text. These controls reduce reliance on the application guard; their policies and scope-to-identity mapping must also be tested.

### 5.3 High-stakes oversight: deleting reports (built)

The flow for "Delete all reports mentioning Client X" or "Delete all the reports we made in this conversation":

1. The model calls `delete_reports` with criteria (`mentions`, `this_conversation`, `report_ids`, `all_reports`). The tool deletes nothing.
2. Code resolves the criteria to the user's own, not yet deleted reports. Every report was stored with `owner_id` and `thread_id`, which is what makes "in this conversation" answerable.
3. The resulting ids are written to the graph state, and the `confirm_delete` node calls `interrupt()`. The state, including the list, is checkpointed.
4. The CLI shows the exact list (id, title, date) and asks for a typed `yes`. Anything else, including Enter or Ctrl-C, cancels.
5. On `yes` the frozen ids are soft-deleted, with the owner checked again in the same SQL statement. An audit event with the ids is recorded in the trace.
6. The model receives "confirmed, deleted n" or "declined" and tells the user.

Why this is strict:
- The confirmation is a value passed from the UI into the graph. The model cannot produce it, and a prompt injection cannot either.
- What is deleted is what was shown. A report created between the preview and the confirmation is not touched (covered by a test).
- Only `confirmed: True` deletes; any other value is treated as "no".
- Keys typed before the dialog appears are discarded, so text typed ahead (or pasted) cannot confirm a deletion the user has not yet seen.

Why it does not break the experience:
- One confirmation covers the whole batch.
- If nothing matches, there is no dialog, just an answer.
- Deletion is soft: rows are marked as deleted, not removed, so a mistaken deletion can be recovered at database level.

In production the same `interrupt()` is shown as a confirmation card in the chat UI, deleted rows are purged after a retention window, and the audit event goes to an audit table in the same transaction as the deletion.

### 5.4 Continuous improvement (design)

**User level.** A preferences store keyed by user holds small facts such as "prefers tables", "wants short answers", "likes a chart with trends", each with a confidence and a source.
- *Explicit*: when a user says "always give me bullet points", the agent saves it immediately with a tool call.
- *Implicit*: after a conversation ends, a background job reads the transcript with an inexpensive model and extracts candidate preferences from behaviour (asked for more detail twice, converted answers to tables, ignored charts). A preference is applied only after it has been seen several times.
- The top preferences are injected into the prompt next to the persona. Company rules and persona win over personal preferences where they conflict.
- The user can ask what is remembered and reset it. In LangGraph this is the long-term `Store`, namespaced by user and backed by Postgres.

**System level.** The system improves through artefacts that are versioned and tested, never by silently changing its own rules.
- Feedback (thumbs, edits to saved reports, rephrased questions) is attached to traces.
- Well-rated interactions become candidate golden trios (section 5.1).
- Failures are clustered weekly: frequent SQL errors, empty results, guard rejections and low-rated answers. A recurring cause becomes a line in the semantic layer, a new trio or a prompt change.
- Every such change runs through the evaluation suite before it is promoted, so learning cannot regress safety or quality.

### 5.5 Resilience and graceful error handling (built)

- **Syntax errors** are caught by a BigQuery dry run, which is free. The error text goes back to the model, which fixes the query. At most three failed queries per question; after that further queries are refused without touching BigQuery and the model is told to explain what could not be computed.
- **Empty results** get one guided retry (check filter values, case and date range). A second empty result is reported to the user as "nothing matched".
- **Warehouse outage, timeout or permission error** is classified separately from a wrong query. Queries stop for this turn and the model is told not to guess numbers.
- **Model failures**: a rate limit or overload gets up to two retries with backoff, using the delay the server asks for when it gives one; then the fallback model; then a plain message. A wait longer than 20 seconds is not taken, and a call that timed out (60 seconds) is not repeated on the same model; in both cases the fallback model is tried instead.
- **Cost stays bounded**: at most eight model calls per question, the last one with tools disabled so it must answer; one gigabyte maximum bytes billed per query, enforced by BigQuery; 200 rows returned; a 60-second job timeout.
- **Missing setup is not treated as an outage.** Missing BigQuery credentials are detected once at startup, shown in a "Setup needed" box with the fix, and remembered, so later questions fail in milliseconds with an honest explanation ("the data source has not been set up yet") instead of a slow "try again later".
- **Stray input costs nothing.** Lines typed while a turn is running are discarded on a terminal. Without this, each of them became a further question with its own model calls.
- **Derived figures must come from data.** A percentage or explicit customer/order/item count in a final answer, or in a report about to be saved, that does not occur in any query result of the conversation (exactly for counts; as written or as a fraction for percentages) is sent back to the model once with the instruction to compute it in SQL or leave it out (`src/agent/guardrails/grounding.py`). The rejected draft is not kept in the conversation, and the incident is a trace event. This was added after a live session in which the model computed a growth rate in its head and got it wrong.
- **The interface never crashes.** Tool handlers cannot raise; the CLI catches anything else, shows a neutral message with the trace id and keeps the conversation usable. Leftovers of a failed turn in the history are repaired before the next model call.

The full matrix is in [section 6](#6-error-handling-and-fallbacks).

### 5.6 Quality assurance (design; tests and a small evaluation suite built)

**Before deployment**, an evaluation suite runs on every change to prompt, model, tools or definitions:

| Level | What is measured | How |
|---|---|---|
| Deterministic parts | Guard, masking, report store, budgets, confirmation | Unit and flow tests (in this repository) |
| SQL generation | Execution accuracy | For a set of questions derived from golden trios (held out from retrieval), compare the result set of the generated SQL with the result set of the analyst's SQL |
| Answers and reports | Intent match, groundedness, completeness, clarity, usefulness of action items | A judge model with a written rubric, calibrated against analyst labels on a sample; plus a programmatic check that every number in the answer appears in a query result of that turn |
| Safety | PII requests, scope escapes, injections in user text and in data, deletions without confirmation | A red-team set that must pass completely; any failure blocks the release |
| Robustness | Behaviour under injected faults | Simulated BigQuery outage, rate limits, malformed SQL |
| Conversation | Multi-turn success | Simulated users with a goal and a persona; measure whether the goal is reached and in how many turns |

**What exists today.** `python evals/run.py` runs twenty-two live cases: fifteen analysis cases (numeric reference queries plus schema coverage) and seven safety cases. It requires every safety case and at least 90% of analysis cases to pass. Provider outages fail their cases, including safety cases. The generated `evals/results/latest.json` records outcomes, runtime, exact brand scopes and source/configuration fingerprints. Partial runs cannot replace it. The churn cases check a 60-day override and a 90-day cohort breakdown by traffic source, current inactivity counts, and monthly churn in a follow-up for Bob, with reference counts and the stated brand scope.

The checks have limits: numeric matching is not a semantic proof, and the PII evaluation checks email patterns and prohibited column names rather than every possible personal attribute. Offline guard tests and the individual-demographic refusal case complement it. Report quality, broad multi-turn behavior and run-to-run consistency still need a larger held-out evaluation set.

**Does a report answer the user's intent?**
- The report states the question as understood and the definitions used, so a misreading is visible to the reader.
- The judge compares the report with the analyst report of the nearest golden trio.
- In the prototype, the groundedness check matches percentages to numeric values anywhere in the conversation's successful query results, allowing rounding and fraction forms. Explicit customer, order, item, user and purchase counts must match a result number exactly; dates, inactivity windows and top-N suggestions are excluded. It does not verify the metric, period, sign or attribution, does not validate all number formats, and always allows 0% and 100%. The broader semantic checks above are production design work.
- A sample of production reports, and all with a low judge score, go to analyst review.

**User experience.**
- Before launch: a pilot with a few executives, observed sessions, and a short task-based test (find X, compare Y, produce a report).
- Measured continuously: time to first answer and to final answer, turns needed per task, rephrase rate, abandonment, thumbs, how often reports are saved and edited, and how often the persona's length and format rules are met.
- Latency has a budget (for example p50 under 10 seconds, p95 under 30) that is part of the release gate.

### 5.7 Observability (built)

**Per turn** (`src/agent/observability.py`): one trace, written as one JSON line to `logs/traces.jsonl`.
- *Spans* for every graph node, every model attempt (model, attempt number, whether it was the fallback, tokens in and out, tool calls requested, error) and every tool call (for SQL: the model's query, the query actually executed, purpose, rows, columns, bytes scanned, BigQuery job id, outcome, error).
- *Events* for incidents: guard blocks with the reason, PII redactions and where they happened, warehouse or model outages, step limit reached, reports deleted or deletion declined.
- A *status* per turn: ok, blocked, cancelled, degraded or error.
- Result rows are never written to a trace. BigQuery jobs carry the trace id as a label, so a job can be matched to a turn and the reverse.

**Deep dive.** Every answer ends with its trace id. `/trace` shows the last turn as a tree; `/trace <id>` shows any recorded turn. Reading it answers: what was asked, which examples were retrieved, what the model requested at each step, which SQL ran and what came back, where it retried, and what was blocked or redacted. The application log with stack traces is in `logs/agent.log`.

**Agent-level metrics** (`/metrics`, computed from the traces):

| Area | Metrics |
|---|---|
| Outcome | share of turns answered normally, degraded, failed, blocked |
| Speed | latency p50 and p95 |
| Model | calls per turn, error rate, how often the fallback model answered, tokens in and out |
| SQL | queries, first-attempt success rate, failure rate, empty-result rate, gigabytes scanned |
| Safety | guard blocks, PII redactions |

**In production** the same spans go to Langfuse, and metrics to Cloud Monitoring. Additional metrics: cost per turn and per user, user feedback rate, judge scores on sampled traffic, retrieval hit rate, confirmation accept rate. Alerts: answer success rate below target, p95 latency above budget, model or warehouse error rate above a threshold, any PII redaction in a final answer (it means an earlier layer let something through), and unusual cost. Support can open a trace from the id the user sees, and replay it against a new prompt or model version.

### 5.8 Agility: persona management (design, with a working hook)

- The system prompt has two parts. The **core** (rules, method, safety) is code, reviewed and released like code. The **persona** (tone, format, length) is content.
- In production the persona lives in a prompt registry with a web editor. A non-developer edits it, saves a new version and labels it "production". The service fetches the labelled version with a short cache, so the change is live within a minute and no deployment happens.
- Safeguards: the persona block is limited in length and is inserted under a heading that says it affects tone and format only; a new version runs a quick automatic check (the safety set and a tone check) before the label moves; every version is kept, with author and time, and rollback is one click.
- In the prototype, [config/persona.md](../config/persona.md) is read on every model call. Edit the file and the next answer uses the new tone. A test covers this.

## 6. Error handling and fallbacks

| Failure | Detected by | What the system does | What the user sees |
|---|---|---|---|
| Wrong SQL (syntax, unknown column) | SQL validation or BigQuery dry run, before billed execution | Error text returned to the model; up to 3 failed queries per question | A normal answer after the repair |
| Query not allowed (write, other table, PII column) | SQL guard, before BigQuery | Reason returned to the model; counts as a failed query; guard event recorded | An explanation that this is not available |
| Query too expensive | Dry-run estimate, and BigQuery's own byte cap | Returned to the model as repairable: narrow the query | A normal answer, or a request to narrow the question |
| Empty result | Row count | One guided retry, then stop | "No matching data", with the likely reason |
| BigQuery down, timeout, access denied | Error class from the client | No more queries this turn; the model is told not to guess | "The data source cannot be reached right now" |
| BigQuery credentials or project missing | Credential lookup at startup; remembered | Startup box with the reason and the fix; queries fail at once; the model is told that retrying will not help | "The data source has not been set up yet", with where to look |
| Keys typed while a turn is running | Pending terminal input before each prompt | Discarded, with a one-line note | "(Ignored what was typed while I was working.)" |
| Model rate limited or overloaded | Error code and text | Retry with backoff (server hint respected), then fallback model | Usually nothing; a short "retrying" status |
| Model call times out | Deadline error after 60 seconds | No retry on the same model; fallback model | A longer wait, then the answer |
| No model available | Retries and fallback exhausted | Static reply; turn marked degraded | "I can't reach the language model right now… try again in a minute" |
| Model loops or returns nothing | Per-turn call limit; empty content | Final call with tools disabled; fallback text if still empty | An answer with what was found, or a request to rephrase |
| Malformed tool call | Argument validation | Validation message returned to the model | Nothing |
| Bug in a tool | Catch-all around each handler | Message to the model; logged with stack trace; turn marked degraded | The rest of the answer, with that step reported as failed |
| Unexpected error in the graph | Catch-all in the CLI | Logged; trace written with the error; history repaired on the next turn | "Something went wrong on my side…", and the chat continues |
| Trace cannot be written | Exception on write | Logged and ignored | Nothing |

Production adds a circuit breaker per dependency (after repeated failures, skip straight to the fallback for a while), a second region for the model, and per-user rate limits so one user cannot exhaust the shared quota.

### When a production control is unavailable

The services that exist only in the production design are safety controls, so the rule for each is whether the system fails closed (holds the work back) or fails open (carries on without it).

| Dependency down | Rule | What happens | What the user sees |
|---|---|---|---|
| Entitlement store (who may see which products) | Fail closed | No query is run for a user whose scope cannot be established. A scope may be served from a cache for a few minutes at most; after that it expires rather than being reused. There is never a fallback to "all products" | "I can't verify your data access right now, so I can't run queries. Your saved reports are still available." |
| PII inspection service | Fail closed for outgoing text | The answer or report is held back, not shown and not saved. The pattern scanner in the service is a first pass, never a substitute that releases text by itself. The call is retried a bounded number of times, then the turn ends | "I have an answer but could not complete the privacy check, so I'm not showing it. Please try again shortly." |
| Checkpoint store | Fail closed for actions | A turn that cannot be checkpointed does not start a deletion, because the frozen selection and the pause for confirmation live in the checkpoint | A plain failure message; nothing is changed |
| Report store | Degrade | Saving, listing and deleting are unavailable; analysis continues | The answer, with a note that reports are unavailable |
| Golden Knowledge index | Fail open | The question is answered without examples, and the turn is marked degraded in its trace | Nothing |
| Trace store | Fail open, buffered | Spans are queued in memory and dropped oldest-first if the outage lasts; an alert fires | Nothing |

Failing closed on the first three is deliberate: a wrong scope, an unchecked answer or an unconfirmed deletion each breaks a guarantee, while a missing answer only costs a retry.

## 7. Threat model

| Threat | Control |
|---|---|
| User asks for personal data | PII columns do not exist in the views the query runs against; results and output are scanned |
| User tries to see other products | Scope is applied by code to every query; the model cannot remove it |
| Prompt injection in the user's message | Cheap pattern check; more importantly, the model has no capability that an injection could abuse: it can run scoped read-only queries and request a deletion that a person must confirm |
| Injection hidden in data (a product name, a report body) | Same as above; the prompt marks tool output as data |
| Attempt to modify or drop data | Only a single `SELECT` passes the guard; the warehouse identity is read-only |
| Guard and BigQuery reading the SQL differently | The executed SQL is regenerated from the parsed tree; anything that does not parse is rejected |
| Deleting another user's reports | Owner filter in every statement of the report store |
| Deleting without consent | `interrupt()`; only an explicit confirmation from the UI deletes |
| Running up cost | Byte cap, row cap, timeouts, per-turn budgets, input length limit |
| Personal data leaking through logs | No result rows in traces; questions are scanned before logging |
| Re-identification through small groups (one customer in a city) | Not addressed in the prototype; production adds a minimum group size for customer-level breakdowns |

## 8. Cost controls

- **Model**: two to four calls for a typical question; hard limit of eight. Golden examples are retrieved without a model call. The fixed part of the prompt comes first, which suits provider-side prompt caching.
- **BigQuery**: free dry run before every execution; maximum bytes billed per query; job timeout; results limited to 200 rows, which also pushes the model towards aggregate queries. BigQuery's own result cache makes repeated identical queries free.
- **Production**: per-user and per-day budgets, project-level BigQuery quotas, a cache of recent question-to-result pairs within the same scope, and cost per turn as a monitored metric.

## 9. Extending the system

- **A new capability** (charts, e-mail, web search) is a new tool: an args model and a handler registered in `src/agent/core/tools.py`. It is declared to the model, traced and error-contained automatically. A README example shows the few lines needed.
- **A capability with outside effects** (sending e-mail) reuses the confirmation pattern of `confirm_delete`: the tool prepares the action, the graph pauses, the person approves.
- **A capability with its own flow** (a multi-step report builder, a chart renderer with a retry loop) becomes a node or a subgraph.
- **A new interface** (web, Slack) calls the same graph. The CLI contains only presentation and the confirmation dialog.
- **A new data source** gets its own guard and tool; the pattern of "model proposes, code scopes and executes" stays.

## 10. What is tested

Run `python -m pytest -q`. Live checks are skipped unless `--live` is supplied, even if credentials are configured. The offline suite uses the real graph with a scripted model and a fake warehouse, so it checks control flow and guards without network access:

- **SQL guard**: scoped rewriting, redirection of qualified table names, escaping of scope values, and rejection of writes, multiple statements, unknown tables, other datasets, PII columns, table functions and recursive CTEs.
- **PII**: masking of results, redaction of the final answer and of saved reports, and no false positives on dates, money and ids.
- **Deletion**: pause and preview, decline, confirm, frozen ids, other users' reports untouched, only an explicit `True` confirms, mixed tool calls.
- **Resilience**: SQL repair, repair budget, model-call budget, empty results, warehouse outage, model outage, retry and fallback logic, malformed tool calls, a crashing tool, a crashing turn, and recovery of the conversation afterwards.
- **BigQuery client**: dry run before execution, byte cap, row cap, job timeout and trace label on every query; classification of errors into repairable, outage and missing setup.
- **Observability**: span structure, statuses, metrics, and the `/trace` and `/metrics` views.
- **CLI**: the startup setup check, and discarding of keys typed while a turn is running (on a real pseudo-terminal).
- **Groundedness**: an unsupported checked figure triggers exactly one repair round, supported ones cost nothing, and a report with an unsupported checked figure is not saved as written.
- **Hooks**: golden example retrieval and persona reload.

`tests/test_live.py` holds checks against real BigQuery and Gemini: the schema contract, that every golden query is valid, that scope and PII protection hold on real data, error classification, and that Gemini accepts the tool definitions. Run them explicitly with `python -m pytest tests/test_live.py --live -v`. Individual checks are also skipped when their required credentials are not configured. These tests contact external services and may incur usage charges.

## 11. Limitations and next steps

- **Identity and entitlements are simulated.** Access is brand-based, while the CLI identity and example brand assignments are sample configuration.
- **Scope semantics follow the confirmed brand rules.** Orders are visible if they contain an in-scope item, and `orders.num_of_item` is recomputed from in-scope items. Customer demographics describe customers who bought those products; this is not a separate customer-ownership model.
- **Free-text PII detection uses patterns.** It detects common e-mail, phone, address, postal-code, coordinate and name patterns, but cannot guarantee recognition of every free-text format or language. The structural layer makes this a second line, and production adds a DLP service.
- **The attack-phrase check is English and deliberately narrow.** It is a cost saver, not a control.
- **Golden retrieval is lexical** and the examples are a small seeded set.
- **The learning loop is designed, not built, and the evaluation suite is small.** Twenty-two cases measure whether figures are right and safety rules hold; report quality, depth of "why" analyses and run-to-run consistency are not measured yet.
- **No token streaming** in the CLI; the spinner shows the current step instead.

Next steps, in order: validate the remaining production assumptions with deployment stakeholders; build the evaluation set from real trios and wire it into CI; move state to Postgres and deploy the service on Cloud Run behind SSO; add warehouse-side enforcement; then charts and report delivery as the first new tools.


Caching and concurrent query execution are optional optimizations, not missing prototype requirements. Their adoption criteria, scope/freshness rules and budget constraints are documented in [Architecture section 12](ARCHITECTURE.md#12-optional-improvements-after-measurement). Vector retrieval is already part of the target production design; the small prototype corpus uses lexical ranking.
