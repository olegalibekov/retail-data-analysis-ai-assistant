"""Settings, paths and demo-user scopes. Everything configurable lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agent.warehouse.schema import SCOPE_KEYS

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Normalise credentials so a half-filled .env behaves predictably.
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
_key_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if _key_file is not None and not _key_file.strip():
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]  # empty value: fall back to gcloud ADC
elif _key_file and not Path(_key_file).is_absolute() and (ROOT / _key_file).exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(ROOT / _key_file)

DATASET = "bigquery-public-data.thelook_ecommerce"


def _int(name: str, default: int) -> int:
    return int(os.getenv(name) or default)


@dataclass
class Settings:
    gcp_project: str | None = os.getenv("GCP_PROJECT_ID") or None
    # Defaults chosen for responsiveness on a free-tier key; see docs/TECHNICAL_EXPLANATION.md, section 3.
    model: str = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
    fallback_model: str = os.getenv("GEMINI_FALLBACK_MODEL") or "gemini-3.1-flash-lite"

    # Cost and loop bounds (requirement 5: self-correct without inflating costs).
    bq_max_bytes_billed: int = _int("BQ_MAX_BYTES_BILLED", 1024**3)  # per query, enforced by BigQuery
    # Per turn: once the queries of a turn have scanned this much, no further query is sent. The query
    # that reaches the cap still runs, so a turn scans less than this plus one query's cap.
    bq_max_bytes_per_turn: int = _int("BQ_MAX_BYTES_PER_TURN", 3 * 1024**3)
    bq_max_rows: int = _int("BQ_MAX_ROWS", 200)
    bq_timeout_s: int = _int("BQ_TIMEOUT_S", 60)
    max_queries_per_turn: int = _int("MAX_QUERIES_PER_TURN", 8)  # queries sent to BigQuery, failed ones included
    max_sql_failures: int = _int("MAX_SQL_FAILURES_PER_TURN", 3)
    # A report needs about eight steps (queries, one round to repair unsupported figures, the save and
    # the closing answer), so ten leaves room without letting a turn run on.
    max_model_calls: int = _int("MAX_MODEL_CALLS_PER_TURN", 10)
    llm_retries: int = _int("LLM_RETRIES", 2)  # per model, before falling back
    llm_max_wait_s: int = _int("LLM_MAX_WAIT_S", 20)  # longest single backoff we accept
    llm_timeout_s: int = _int("LLM_TIMEOUT_S", 60)
    max_history_messages: int = _int("MAX_HISTORY_MESSAGES", 40)
    max_result_chars: int = _int("MAX_RESULT_CHARS", 8000)

    persona_path: Path = ROOT / "config" / "persona.md"
    semantic_layer_path: Path = ROOT / "config" / "semantic_layer.md"
    users_path: Path = ROOT / "config" / "users.yaml"
    trios_path: Path = ROOT / "data" / "golden_trios.json"
    reports_db_path: Path = ROOT / "data" / "reports.db"
    trace_path: Path = ROOT / "logs" / "traces.jsonl"
    log_path: Path = ROOT / "logs" / "agent.log"


settings = Settings()


@dataclass(frozen=True)
class User:
    """An authenticated executive. `scope` maps brand to the authorized brand names.

    An empty scope is reserved for the CEO and means unrestricted. In production this comes from the identity
    provider and is enforced by BigQuery authorized views; here it drives sql_guard.
    """

    id: str
    name: str
    scope: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def scope_values(self) -> list[str]:
        """The names in the user's scope (brands). They are business names, never personal data."""
        return [value for values in self.scope.values() for value in values]

    def describe_scope(self) -> str:
        if not self.scope:
            return "all products"
        return "; ".join(f"{col} in {list(vals)}" for col, vals in self.scope.items())


def load_users(path: Path | None = None) -> dict[str, User]:
    raw = yaml.safe_load((path or settings.users_path).read_text())["users"]
    users = {}
    for user_id, spec in raw.items():
        scope = spec.get("scope") or {}
        unknown = set(scope) - SCOPE_KEYS
        if unknown:
            raise ValueError(
                f"users.yaml: user '{user_id}' has unsupported scope key(s) {sorted(unknown)}; use {sorted(SCOPE_KEYS)}"
            )
        if not scope and user_id != "admin":
            raise ValueError(f"users.yaml: user '{user_id}' needs a brand scope; only admin (CEO) may be unrestricted")
        if scope and (
            not isinstance(scope["brand"], list)
            or not scope["brand"]
            or any(not isinstance(value, str) or not value.strip() for value in scope["brand"])
        ):
            raise ValueError(f"users.yaml: user '{user_id}' needs a non-empty list of brand names")
        users[user_id] = User(
            id=user_id,
            name=spec.get("name", user_id),
            scope={column: tuple(values) for column, values in scope.items()},
        )
    return users
