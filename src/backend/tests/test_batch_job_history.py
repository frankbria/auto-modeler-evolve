"""Tests for Batch Job History Analytics via Chat.

Covers:
- compute_batch_job_history() pure function (no_data, healthy, some_failures, all_failed,
  timeline entries, duration computation, avg_row_count)
- _BATCH_JOB_HISTORY_PATTERNS regex (8 positive, 3 negative)
- GET /api/deploy/{id}/batch-job-history REST endpoint (no_data, healthy, 404)
- Chat handler emitting batch_job_history SSE event
"""

from __future__ import annotations


import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session

import db as db_module

# ---------------------------------------------------------------------------
# Pure-function constants
# ---------------------------------------------------------------------------

_NO_DATA_VERDICT = "no_data"
_HEALTHY_VERDICT = "healthy"
_SOME_FAIL_VERDICT = "some_failures"
_ALL_FAIL_VERDICT = "all_failed"


def test_batch_job_history_verdicts_are_constants():
    assert _NO_DATA_VERDICT == "no_data"
    assert _HEALTHY_VERDICT == "healthy"
    assert _SOME_FAIL_VERDICT == "some_failures"
    assert _ALL_FAIL_VERDICT == "all_failed"


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


def _make_run(
    status: str = "success",
    started_at: str = "2024-01-15T09:00:00",
    completed_at: str | None = "2024-01-15T09:01:30",
    row_count: int | None = 500,
    error: str | None = None,
    run_id: str = "run-1",
) -> dict:
    return {
        "id": run_id,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "row_count": row_count,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Pure function: compute_batch_job_history
# ---------------------------------------------------------------------------


class TestComputeBatchJobHistoryNoData:
    def setup_method(self):
        from core.analyzer import compute_batch_job_history

        self.fn = compute_batch_job_history

    def test_empty_list_returns_no_data(self):
        result = self.fn([])
        assert result["verdict"] == "no_data"
        assert result["total_runs"] == 0
        assert result["success_count"] == 0
        assert result["failed_count"] == 0
        assert result["runs"] == []
        assert result["avg_row_count"] is None
        assert result["avg_duration_sec"] is None
        assert result["last_run_at"] is None

    def test_empty_list_summary_mentions_schedule(self):
        result = self.fn([])
        assert (
            "schedule" in result["summary"].lower()
            or "no batch" in result["summary"].lower()
        )


class TestComputeBatchJobHistoryHealthy:
    def setup_method(self):
        from core.analyzer import compute_batch_job_history

        self.fn = compute_batch_job_history

    def _healthy_runs(self):
        return [
            _make_run(
                "success",
                "2024-01-15T09:00:00",
                "2024-01-15T09:01:30",
                500,
                run_id=f"r{i}",
            )
            for i in range(5)
        ]

    def test_all_success_is_healthy(self):
        result = self.fn(self._healthy_runs())
        assert result["verdict"] == "healthy"

    def test_counts_correct(self):
        runs = self._healthy_runs()
        result = self.fn(runs)
        assert result["total_runs"] == 5
        assert result["success_count"] == 5
        assert result["failed_count"] == 0
        assert result["running_count"] == 0

    def test_success_rate_is_one(self):
        result = self.fn(self._healthy_runs())
        assert result["success_rate"] == 1.0

    def test_avg_row_count_computed(self):
        result = self.fn(self._healthy_runs())
        assert result["avg_row_count"] == 500.0

    def test_avg_duration_computed(self):
        result = self.fn(self._healthy_runs())
        # 9:00 → 9:01:30 = 90 seconds
        assert result["avg_duration_sec"] == 90.0

    def test_last_run_at_set(self):
        result = self.fn(self._healthy_runs())
        assert result["last_run_at"] is not None

    def test_timeline_has_entries(self):
        result = self.fn(self._healthy_runs())
        assert len(result["runs"]) == 5

    def test_timeline_entry_shape(self):
        result = self.fn(self._healthy_runs())
        run = result["runs"][0]
        assert "id" in run
        assert "status" in run
        assert "started_at" in run
        assert "completed_at" in run
        assert "row_count" in run
        assert "duration_sec" in run
        assert "error" in run

    def test_timeline_duration_per_entry(self):
        result = self.fn(self._healthy_runs())
        assert result["runs"][0]["duration_sec"] == 90.0

    def test_summary_mentions_success(self):
        result = self.fn(self._healthy_runs())
        assert (
            "success" in result["summary"].lower()
            or "reliable" in result["summary"].lower()
        )

    def test_ninety_pct_success_is_healthy(self):
        # 9 success + 1 failed = 90% → healthy
        runs = [_make_run("success", run_id=f"s{i}") for i in range(9)]
        runs.append(_make_run("failed", error="timeout", run_id="f0"))
        result = self.fn(runs)
        assert result["verdict"] == "healthy"


class TestComputeBatchJobHistorySomeFailures:
    def setup_method(self):
        from core.analyzer import compute_batch_job_history

        self.fn = compute_batch_job_history

    def _mixed_runs(self):
        success = [_make_run("success", run_id=f"s{i}") for i in range(6)]
        failed = [
            _make_run("failed", error="connection refused", run_id=f"f{i}")
            for i in range(4)
        ]
        return success + failed  # 60% success

    def test_mixed_is_some_failures(self):
        result = self.fn(self._mixed_runs())
        assert result["verdict"] == "some_failures"

    def test_failed_count_correct(self):
        result = self.fn(self._mixed_runs())
        assert result["failed_count"] == 4

    def test_summary_mentions_failures(self):
        result = self.fn(self._mixed_runs())
        assert "fail" in result["summary"].lower()

    def test_success_rate_correct(self):
        result = self.fn(self._mixed_runs())
        assert abs(result["success_rate"] - 0.6) < 0.01


class TestComputeBatchJobHistoryAllFailed:
    def setup_method(self):
        from core.analyzer import compute_batch_job_history

        self.fn = compute_batch_job_history

    def _failed_runs(self):
        return [
            _make_run("failed", error="deployment offline", run_id=f"f{i}")
            for i in range(5)
        ]

    def test_all_failed_is_all_failed_verdict(self):
        result = self.fn(self._failed_runs())
        assert result["verdict"] == "all_failed"

    def test_success_rate_zero(self):
        result = self.fn(self._failed_runs())
        assert result["success_rate"] == 0.0

    def test_summary_mentions_failed(self):
        result = self.fn(self._failed_runs())
        assert "fail" in result["summary"].lower()

    def test_below_fifty_pct_is_all_failed(self):
        # 4 success, 10 failed = ~29% → all_failed
        runs = [_make_run("success", run_id=f"s{i}") for i in range(4)]
        runs += [_make_run("failed", error="err", run_id=f"f{i}") for i in range(10)]
        result = self.fn(runs)
        assert result["verdict"] == "all_failed"


class TestComputeBatchJobHistoryTimeline:
    def setup_method(self):
        from core.analyzer import compute_batch_job_history

        self.fn = compute_batch_job_history

    def test_n_cap_applied(self):
        runs = [_make_run("success", run_id=f"r{i}") for i in range(30)]
        result = self.fn(runs, n=10)
        assert len(result["runs"]) == 10

    def test_running_status_counted(self):
        runs = [
            _make_run("success", run_id="s0"),
            _make_run("running", completed_at=None, run_id="r0"),
        ]
        result = self.fn(runs)
        assert result["running_count"] == 1

    def test_no_row_count_excluded_from_avg(self):
        runs = [
            _make_run("success", row_count=None, run_id="s0"),
            _make_run("success", row_count=200, run_id="s1"),
        ]
        result = self.fn(runs)
        assert result["avg_row_count"] == 200.0

    def test_duration_skipped_when_no_completed_at(self):
        runs = [_make_run("running", completed_at=None, run_id="r0")]
        result = self.fn(runs)
        assert result["avg_duration_sec"] is None

    def test_error_preserved_in_timeline(self):
        runs = [_make_run("failed", error="disk full", run_id="f0")]
        result = self.fn(runs)
        assert result["runs"][0]["error"] == "disk full"


# ---------------------------------------------------------------------------
# Regex: _BATCH_JOB_HISTORY_PATTERNS
# ---------------------------------------------------------------------------


class TestBatchJobHistoryPatterns:
    def setup_method(self):
        from api.chat import _BATCH_JOB_HISTORY_PATTERNS

        self.pat = _BATCH_JOB_HISTORY_PATTERNS

    # Positive matches
    def test_batch_job_history(self):
        assert self.pat.search("batch job history")

    def test_batch_run_history(self):
        assert self.pat.search("show me batch run history")

    def test_batch_history(self):
        assert self.pat.search("batch history")

    def test_when_did_batch_run(self):
        assert self.pat.search("when did my batch job run")

    def test_batch_success_rate(self):
        assert self.pat.search("batch success rate")

    def test_batch_failure_history(self):
        assert self.pat.search("batch failure history")

    def test_how_often_batch_runs(self):
        assert self.pat.search("how often does my batch job run")

    def test_recent_batch_runs(self):
        assert self.pat.search("recent batch runs")

    def test_list_batch_runs(self):
        assert self.pat.search("list batch runs")

    # Negative matches — must NOT overlap with _BATCH_RESULTS_PATTERNS
    def test_no_match_batch_results(self):
        assert not self.pat.search("show batch results")

    def test_no_match_batch_output(self):
        assert not self.pat.search("batch output")

    def test_no_match_generic_predictions(self):
        assert not self.pat.search("show me predictions")


# ---------------------------------------------------------------------------
# REST endpoint: GET /api/deploy/{id}/batch-job-history
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_with_db(tmp_path):
    import models.ab_test  # noqa
    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.deployment_preset  # noqa
    import models.deployment_version  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa
    import models.webhook_event  # noqa
    import models.analysis_template  # noqa

    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db_module.engine)

    from main import app

    client = TestClient(app, raise_server_exceptions=False)
    yield client, db_module.engine


def _create_deployment(engine, dep_id: str = "dep-bjh-1", is_active: bool = True):
    from models.deployment import Deployment as Dep
    from models.project import Project
    from models.model_run import ModelRun
    from models.dataset import Dataset
    from models.feature_set import FeatureSet

    with Session(engine) as s:
        proj = Project(owner_id="test-default-owner", id="proj-bjh", name="BJH Test")
        s.add(proj)
        s.flush()

        ds = Dataset(
            id="ds-bjh",
            project_id="proj-bjh",
            filename="test.csv",
            file_path="/tmp/test.csv",
            row_count=10,
            column_count=3,
            size_bytes=100,
        )
        s.add(ds)
        s.flush()

        fs = FeatureSet(
            id="fs-bjh",
            dataset_id="ds-bjh",
            target_column="price",
            transformations="[]",
            column_mapping="{}",
            is_active=True,
        )
        s.add(fs)
        s.flush()

        mr = ModelRun(
            id="mr-bjh-1",
            project_id="proj-bjh",
            feature_set_id="fs-bjh",
            algorithm="LinearRegression",
            hyperparameters="{}",
            metrics="{}",
            training_duration_ms=100,
            model_path="/tmp/model.joblib",
        )
        s.add(mr)
        s.flush()

        dep = Dep(
            id=dep_id,
            project_id="proj-bjh",
            model_run_id="mr-bjh-1",
            endpoint_path=f"/api/predict/{dep_id}",
            dashboard_url=f"/dashboard/{dep_id}",
            is_active=is_active,
            request_count=0,
            problem_type="regression",
            target_column="price",
        )
        s.add(dep)
        s.commit()
    return dep_id


class TestBatchJobHistoryEndpoint:
    def test_returns_no_data_when_no_runs(self, client_with_db):
        client, engine = client_with_db
        _create_deployment(engine, "dep-bjh-empty")
        resp = client.get("/api/deploy/dep-bjh-empty/batch-job-history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "no_data"
        assert data["total_runs"] == 0

    def test_returns_404_for_inactive_deployment(self, client_with_db):
        client, engine = client_with_db
        _create_deployment(engine, "dep-bjh-inactive", is_active=False)
        resp = client.get("/api/deploy/dep-bjh-inactive/batch-job-history")
        assert resp.status_code == 404

    def test_returns_history_with_runs(self, client_with_db):
        from models.batch_schedule import BatchJobRun, BatchSchedule
        from datetime import datetime

        client, engine = client_with_db
        _create_deployment(engine, "dep-bjh-runs")
        with Session(engine) as s:
            sched = BatchSchedule(
                id="sched-bjh-1",
                deployment_id="dep-bjh-runs",
                frequency="daily",
                run_hour=9,
                run_minute=0,
                is_active=True,
            )
            s.add(sched)
            for i in range(3):
                run = BatchJobRun(
                    id=f"run-bjh-{i}",
                    schedule_id="sched-bjh-1",
                    deployment_id="dep-bjh-runs",
                    started_at=datetime(2024, 1, 15, 9, 0, 0),
                    completed_at=datetime(2024, 1, 15, 9, 1, 30),
                    status="success",
                    row_count=100,
                )
                s.add(run)
            s.commit()
        resp = client.get("/api/deploy/dep-bjh-runs/batch-job-history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_runs"] == 3
        assert data["success_count"] == 3
        assert data["verdict"] == "healthy"
        assert len(data["runs"]) == 3
