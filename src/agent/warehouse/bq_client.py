"""BigQuery access with a free dry run, a cost cap, a row cap, a timeout, job labels and
error classification. It knows nothing about users or PII rules for queries:
sql_guard runs before it and pii.mask_dataframe after it.

Queries run without a default dataset, on purpose. A bare table name then reaches
no table: one that the guard failed to redirect to a view fails in BigQuery instead
of silently reading the raw table. Only a dotted path can name a real table or
view, and the only ones in a guarded query are in the guard's own views.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from google.api_core import exceptions as gexc
from google.auth import exceptions as auth_exc
from google.cloud import bigquery

from agent.config import DATASET, settings
from agent.guardrails.pii import PII_COLUMNS

log = logging.getLogger(__name__)


class SQLSyntaxError(Exception):
    """The query itself is wrong. The message goes back to the model for repair."""


class BQUnavailableError(Exception):
    """BigQuery is unreachable, throttled or access is denied. Not repairable by the model."""


class BQNotConfiguredError(BQUnavailableError):
    """This installation has no usable BigQuery credentials. A setup problem, not an outage:
    retrying will not help until someone configures access and restarts."""


@dataclass
class QueryOutcome:
    df: pd.DataFrame
    bytes_processed: int
    job_id: str | None
    truncated: bool
    duration_ms: int


_SYNTAX = (gexc.BadRequest, gexc.NotFound)
_UNAVAILABLE = (gexc.GoogleAPICallError, auth_exc.GoogleAuthError, TimeoutError, ConnectionError, OSError)


class BigQueryRunner:
    """A lean BigQuery client for executing SQL queries and returning DataFrame results."""

    def __init__(self, project_id: str | None = None, dataset_id: str = DATASET) -> None:
        self.project_id = project_id or settings.gcp_project
        self.dataset_id = dataset_id  # for schema lookups only; never a default dataset for queries
        self._client: bigquery.Client | None = None
        self._setup_error: str | None = None  # remembered: missing credentials do not fix themselves

    @property
    def client(self) -> bigquery.Client:
        # Created on first use so the CLI can start, and report nicely, without credentials.
        if self._setup_error:
            raise BQNotConfiguredError(self._setup_error)
        if self._client is None:
            try:
                self._client = bigquery.Client(project=self.project_id)
            except _UNAVAILABLE as e:
                if _is_setup_error(e):
                    # Looking for credentials can take seconds, so the answer is kept.
                    self._setup_error = _message(e)
                    raise BQNotConfiguredError(self._setup_error) from e
                raise BQUnavailableError(f"Could not initialise BigQuery client: {e}") from e
        return self._client

    def setup_problem(self) -> str | None:
        """Why BigQuery cannot be used on this installation, or None. Runs no query."""
        try:
            _ = self.client  # building the client is what finds the credentials
        except BQNotConfiguredError as e:
            return str(e)
        except BQUnavailableError:
            return None  # a transient failure is not a setup problem
        return None

    def dry_run(self, sql_query: str) -> int:
        """Validate the query and return the bytes it would scan. Costs nothing."""
        cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            return self.client.query(sql_query, job_config=cfg).total_bytes_processed or 0
        except _SYNTAX as e:
            raise SQLSyntaxError(_message(e)) from e
        except _UNAVAILABLE as e:
            raise BQUnavailableError(_message(e)) from e

    def run(self, sql_query: str, trace_id: str = "") -> QueryOutcome:
        """Dry-run, then execute with a cost cap and a row cap."""
        estimate = self.dry_run(sql_query)
        if estimate > settings.bq_max_bytes_billed:
            raise SQLSyntaxError(
                f"Query would scan {estimate / 1e9:.2f} GB, above the {settings.bq_max_bytes_billed / 1e9:.2f} GB limit. "
                "Select fewer columns or add filters."
            )
        cfg = bigquery.QueryJobConfig(
            maximum_bytes_billed=settings.bq_max_bytes_billed,
            job_timeout_ms=settings.bq_timeout_s * 1000,
            labels={"app": "analysis-agent", "trace_id": trace_id[:63].lower()}
            if trace_id
            else {"app": "analysis-agent"},
        )
        started = time.monotonic()
        try:
            job = self.client.query(sql_query, job_config=cfg)
            rows = job.result(max_results=settings.bq_max_rows + 1, timeout=settings.bq_timeout_s)
            df = rows.to_dataframe(create_bqstorage_client=False)
        except _SYNTAX as e:
            raise SQLSyntaxError(_message(e)) from e
        except _UNAVAILABLE as e:
            raise BQUnavailableError(_message(e)) from e
        truncated = len(df) > settings.bq_max_rows
        outcome = QueryOutcome(
            df=df.head(settings.bq_max_rows),
            bytes_processed=job.total_bytes_processed or 0,
            job_id=job.job_id,
            truncated=truncated,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        log.info(
            "BigQuery job %s: %d rows, %d bytes, %d ms",
            outcome.job_id,
            len(outcome.df),
            outcome.bytes_processed,
            outcome.duration_ms,
        )
        return outcome

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame."""
        return self.run(sql_query).df

    def get_table_schema(self, table_name: str, include_pii: bool = False) -> list[dict[str, Any]]:
        """Column information for a table. PII columns are hidden unless asked for."""
        try:
            table = self.client.get_table(f"{self.dataset_id}.{table_name}")
        except _SYNTAX as e:
            raise SQLSyntaxError(_message(e)) from e
        except _UNAVAILABLE as e:
            raise BQUnavailableError(_message(e)) from e
        return [
            {"name": f.name, "type": f.field_type, "mode": f.mode, "description": f.description or ""}
            for f in table.schema
            if include_pii or f.name.lower() not in PII_COLUMNS
        ]


def _is_setup_error(e: Exception) -> bool:
    """No credentials, a missing or unreadable key file, or no project to bill (not a network failure)."""
    if isinstance(e, auth_exc.DefaultCredentialsError):
        return True
    return isinstance(e, OSError) and not isinstance(e, (ConnectionError, TimeoutError))


def _message(e: Exception) -> str:
    return getattr(e, "message", None) or str(e)
