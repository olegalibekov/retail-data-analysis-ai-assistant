"""BigQueryRunner without a network: cost limits and how failures are classified."""

from types import SimpleNamespace

import pandas as pd
import pytest
from google.api_core import exceptions as gexc
from google.auth import exceptions as auth_exc

import agent.warehouse.bq_client as bq_module
from agent.config import settings
from agent.warehouse.bq_client import BigQueryRunner, BQNotConfiguredError, BQUnavailableError, SQLSyntaxError


class FakeClient:
    def __init__(self, df=None, scanned=1_000, error=None):
        self.df = pd.DataFrame({"n": [1]}) if df is None else df
        self.scanned, self.error, self.configs = scanned, error, []

    def query(self, sql, job_config=None):
        self.configs.append(job_config)
        if self.error and not job_config.dry_run:
            raise self.error
        result = lambda max_results=None, timeout=None: SimpleNamespace(  # noqa: E731
            to_dataframe=lambda create_bqstorage_client=False: self.df.head(max_results)
        )
        return SimpleNamespace(total_bytes_processed=self.scanned, job_id="job_42", result=result)


def runner(monkeypatch, make_client):
    built = []

    def factory(project=None):
        built.append(project)
        return make_client()

    monkeypatch.setattr(bq_module.bigquery, "Client", factory)
    return BigQueryRunner(project_id="my-project"), built


def raising(error):
    def make_client():
        raise error

    return make_client


@pytest.mark.parametrize(
    "error",
    [
        auth_exc.DefaultCredentialsError("Your default credentials were not found."),
        OSError("Project was not passed and could not be determined from the environment."),
        FileNotFoundError("File key.json was not found."),
    ],
)
def test_missing_credentials_are_a_setup_problem_and_remembered(monkeypatch, error):
    bq, built = runner(monkeypatch, raising(error))
    assert str(error) in bq.setup_problem()
    for _ in range(2):
        with pytest.raises(BQNotConfiguredError):
            bq.dry_run("SELECT 1")
    assert len(built) == 1  # the slow credential lookup is not repeated on every question


def test_network_failure_at_start_is_an_outage_not_a_setup_problem(monkeypatch):
    bq, built = runner(monkeypatch, raising(ConnectionError("network is down")))
    assert bq.setup_problem() is None
    with pytest.raises(BQUnavailableError) as error:
        bq.dry_run("SELECT 1")
    assert not isinstance(error.value, BQNotConfiguredError) and len(built) == 2  # tried again


def test_query_runs_with_dry_run_cost_cap_timeout_and_trace_label(monkeypatch):
    client = FakeClient(df=pd.DataFrame({"n": range(settings.bq_max_rows + 50)}), scanned=2_000_000)
    bq, _ = runner(monkeypatch, lambda: client)
    outcome = bq.run("SELECT n FROM t", trace_id="ABC123")

    assert len(outcome.df) == settings.bq_max_rows and outcome.truncated
    assert (outcome.bytes_processed, outcome.job_id) == (2_000_000, "job_42")
    dry, real = client.configs
    assert dry.dry_run and not real.dry_run
    assert real.maximum_bytes_billed == settings.bq_max_bytes_billed
    assert int(real.job_timeout_ms) == settings.bq_timeout_s * 1000
    assert real.labels == {"app": "analysis-agent", "trace_id": "abc123"}
    assert bq.execute_query("SELECT n FROM t").equals(outcome.df)  # the starter's original interface still works


def test_no_query_gets_a_default_dataset_so_a_bare_table_name_cannot_reach_a_real_table(monkeypatch):
    client = FakeClient()
    bq, _ = runner(monkeypatch, lambda: client)
    bq.run("SELECT n FROM t")
    dry, real = client.configs
    assert dry.default_dataset is None and real.default_dataset is None


def test_too_expensive_query_is_refused_before_it_runs(monkeypatch):
    client = FakeClient(scanned=settings.bq_max_bytes_billed + 1)
    bq, _ = runner(monkeypatch, lambda: client)
    with pytest.raises(SQLSyntaxError, match="above the"):
        bq.run("SELECT * FROM big")
    assert len(client.configs) == 1 and client.configs[0].dry_run  # only the free dry run happened


@pytest.mark.parametrize(
    "error, expected",
    [
        (gexc.BadRequest("Unrecognized name: revenu"), SQLSyntaxError),
        (gexc.NotFound("Table x was not found"), SQLSyntaxError),
        (gexc.ServiceUnavailable("backend error"), BQUnavailableError),
        (gexc.Forbidden("Access Denied"), BQUnavailableError),
        (gexc.TooManyRequests("quota"), BQUnavailableError),
        (TimeoutError("deadline"), BQUnavailableError),
    ],
)
def test_query_errors_are_split_into_repairable_and_not(monkeypatch, error, expected):
    bq, _ = runner(monkeypatch, lambda: FakeClient(error=error))
    with pytest.raises(expected) as raised:
        bq.run("SELECT 1")
    assert not isinstance(raised.value, BQNotConfiguredError)
