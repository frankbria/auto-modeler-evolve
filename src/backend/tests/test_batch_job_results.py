"""Tests for Batch Job Results Analytics via Chat.

Covers:
- compute_batch_job_results() pure function (regression, classification,
  empty CSV, non-numeric regression, classification with confidence column)
- _BATCH_RESULTS_PATTERNS regex (8 positive, 3 negative)
- GET /api/deploy/{id}/batch-results REST endpoint (no jobs, missing file,
  regression result, classification result)
- Chat handler emitting batch_job_results SSE event
"""

from __future__ import annotations

import io
import json
import time
from unittest.mock import MagicMock, patch as mock_patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session

import db as db_module

# ---------------------------------------------------------------------------
# CSV fixtures
# ---------------------------------------------------------------------------

_REGRESSION_CSV = (
    b"feature1,feature2,price_prediction\n"
    b"1.0,2.0,100.0\n"
    b"2.0,3.0,150.0\n"
    b"3.0,4.0,200.0\n"
    b"4.0,5.0,250.0\n"
    b"5.0,6.0,300.0\n"
)

_CLASSIFICATION_CSV = (
    b"feature1,feature2,status_prediction\n"
    b"1.0,0.5,approved\n"
    b"2.0,0.8,denied\n"
    b"3.0,0.6,approved\n"
    b"4.0,0.3,approved\n"
    b"5.0,0.9,denied\n"
)

_CLASSIFICATION_CONFIDENCE_CSV = (
    b"feature1,status_prediction,prediction_confidence\n"
    b"1.0,approved,0.92\n"
    b"2.0,denied,0.78\n"
    b"3.0,approved,0.85\n"
    b"4.0,approved,0.95\n"
    b"5.0,denied,0.70\n"
)

_EMPTY_CSV = b"feature1,feature2\n"

_NON_NUMERIC_REGRESSION_CSV = (
    b"feature1,price_prediction\na,not_a_number\nb,also_not_a_number\n"
)


# ---------------------------------------------------------------------------
# Pure function: compute_batch_job_results
# ---------------------------------------------------------------------------


class TestComputeBatchJobResultsRegression:
    def _run(self, csv_bytes=_REGRESSION_CSV, target_column="price"):
        from core.analyzer import compute_batch_job_results

        return compute_batch_job_results(csv_bytes, "regression", target_column)

    def test_has_data_true(self):
        r = self._run()
        assert r["has_data"] is True

    def test_problem_type(self):
        r = self._run()
        assert r["problem_type"] == "regression"

    def test_total_rows(self):
        r = self._run()
        assert r["total_rows"] == 5

    def test_avg_prediction(self):
        r = self._run()
        assert abs(r["avg_prediction"] - 200.0) < 0.01

    def test_min_prediction(self):
        r = self._run()
        assert abs(r["min_prediction"] - 100.0) < 0.01

    def test_max_prediction(self):
        r = self._run()
        assert abs(r["max_prediction"] - 300.0) < 0.01

    def test_median_prediction(self):
        r = self._run()
        assert abs(r["median_prediction"] - 200.0) < 0.01

    def test_std_prediction_positive(self):
        r = self._run()
        assert r["std_prediction"] > 0

    def test_histogram_present(self):
        r = self._run()
        assert isinstance(r["histogram"], list)
        assert len(r["histogram"]) >= 3

    def test_histogram_bins_have_correct_keys(self):
        r = self._run()
        for bin_ in r["histogram"]:
            assert "bin_start" in bin_
            assert "bin_end" in bin_
            assert "count" in bin_

    def test_summary_contains_target(self):
        r = self._run()
        assert "price" in r["summary"]

    def test_auto_detect_prediction_column_by_target_prefix(self):
        csv = b"feature1,price_prediction\n1.0,500.0\n2.0,600.0\n3.0,700.0\n"
        r = self._run(csv_bytes=csv, target_column="price")
        assert r["has_data"] is True
        assert r["prediction_column"] == "price_prediction"

    def test_non_numeric_returns_no_data(self):
        from core.analyzer import compute_batch_job_results

        r = compute_batch_job_results(
            _NON_NUMERIC_REGRESSION_CSV, "regression", "price"
        )
        assert r["has_data"] is False


class TestComputeBatchJobResultsClassification:
    def _run(self, csv_bytes=_CLASSIFICATION_CSV, target_column="status"):
        from core.analyzer import compute_batch_job_results

        return compute_batch_job_results(csv_bytes, "classification", target_column)

    def test_has_data_true(self):
        r = self._run()
        assert r["has_data"] is True

    def test_problem_type(self):
        r = self._run()
        assert r["problem_type"] == "classification"

    def test_total_rows(self):
        r = self._run()
        assert r["total_rows"] == 5

    def test_class_distribution_present(self):
        r = self._run()
        assert isinstance(r["class_distribution"], list)
        assert len(r["class_distribution"]) == 2

    def test_class_distribution_entries_have_keys(self):
        r = self._run()
        for entry in r["class_distribution"]:
            assert "class_name" in entry
            assert "count" in entry
            assert "pct" in entry

    def test_top_class_is_approved(self):
        r = self._run()
        assert r["top_class"] == "approved"

    def test_top_pct_correct(self):
        r = self._run()
        assert abs(r["top_pct"] - 60.0) < 0.1

    def test_no_avg_confidence_without_confidence_col(self):
        r = self._run()
        assert r["avg_confidence"] is None

    def test_avg_confidence_detected(self):
        from core.analyzer import compute_batch_job_results

        r = compute_batch_job_results(
            _CLASSIFICATION_CONFIDENCE_CSV, "classification", "status"
        )
        assert r["avg_confidence"] is not None
        assert 70.0 < r["avg_confidence"] < 100.0

    def test_summary_contains_top_class(self):
        r = self._run()
        assert "approved" in r["summary"]

    def test_pct_sums_to_100(self):
        r = self._run()
        total = sum(e["pct"] for e in r["class_distribution"])
        assert abs(total - 100.0) < 0.5


class TestComputeBatchJobResultsEdgeCases:
    def test_empty_csv_returns_no_data(self):
        from core.analyzer import compute_batch_job_results

        r = compute_batch_job_results(_EMPTY_CSV, "regression", "price")
        assert r["has_data"] is False

    def test_malformed_bytes_returns_no_data(self):
        from core.analyzer import compute_batch_job_results

        r = compute_batch_job_results(b"\x00\xff\xfe", "regression", "price")
        assert r["has_data"] is False

    def test_fallback_to_last_column_when_no_match(self):
        csv = b"feat_a,feat_b,unknown_col\n1.0,2.0,42.0\n3.0,4.0,99.0\n"
        from core.analyzer import compute_batch_job_results

        r = compute_batch_job_results(csv, "regression", "price")
        assert r["has_data"] is True
        assert r["prediction_column"] == "unknown_col"


# ---------------------------------------------------------------------------
# _BATCH_RESULTS_PATTERNS regex
# ---------------------------------------------------------------------------


class TestBatchResultsPatterns:
    @pytest.fixture(autouse=True)
    def _import_pattern(self):
        from api.chat import _BATCH_RESULTS_PATTERNS

        self.pattern = _BATCH_RESULTS_PATTERNS

    @pytest.mark.parametrize(
        "phrase",
        [
            "show me the batch job results",
            "batch results",
            "latest batch results",
            "batch prediction results",
            "batch prediction summary",
            "how did the last batch job go",
            "view batch run results",
            "batch run analytics",
        ],
    )
    def test_positive_match(self, phrase):
        assert (
            self.pattern.search(phrase) is not None
        ), f"Expected match for: {phrase!r}"

    @pytest.mark.parametrize(
        "phrase",
        [
            "train a model",
            "show me the correlation matrix",
            "what is the accuracy?",
        ],
    )
    def test_negative_no_match(self, phrase):
        assert self.pattern.search(phrase) is None, f"Expected NO match for: {phrase!r}"


# ---------------------------------------------------------------------------
# REST endpoint + Chat integration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

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

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _make_deployment(
    deployment_id, problem_type="regression", target_column="price", project_id="proj1"
):
    from models.deployment import Deployment

    return Deployment(
        id=deployment_id,
        project_id=project_id,
        model_run_id="run1",
        name=f"Deploy {deployment_id}",
        status="active",
        endpoint_path=f"/predict/{deployment_id}",
        dashboard_url=f"/predict/{deployment_id}",
        problem_type=problem_type,
        target_column=target_column,
    )


def _make_batch_run(
    run_id, schedule_id, deployment_id, output_path="", status="success"
):
    from models.batch_schedule import BatchJobRun

    return BatchJobRun(
        id=run_id,
        schedule_id=schedule_id,
        deployment_id=deployment_id,
        status=status,
        output_path=output_path,
    )


def _make_schedule(schedule_id, deployment_id):
    from models.batch_schedule import BatchSchedule

    return BatchSchedule(
        id=schedule_id,
        deployment_id=deployment_id,
        cron_expression="0 9 * * *",
        status="active",
    )


# ---------------------------------------------------------------------------
# REST endpoint: GET /api/deploy/{id}/batch-results
# ---------------------------------------------------------------------------


class TestBatchResultsEndpoint:
    def test_unknown_deployment_returns_404(self, client):
        r = client.get("/api/deploy/nonexistent/batch-results")
        assert r.status_code == 404

    def test_no_jobs_returns_no_results(self, client):
        with Session(db_module.engine) as s:
            s.add(_make_deployment("dep1"))
            s.commit()
        r = client.get("/api/deploy/dep1/batch-results")
        assert r.status_code == 200
        assert r.json()["has_results"] is False

    def test_missing_output_file_returns_no_results(self, client):
        with Session(db_module.engine) as s:
            s.add(_make_deployment("dep2"))
            s.add(_make_schedule("sched2", "dep2"))
            s.commit()
            s.add(_make_batch_run("run2", "sched2", "dep2", "/nonexistent/path.csv"))
            s.commit()
        r = client.get("/api/deploy/dep2/batch-results")
        assert r.status_code == 200
        assert r.json()["has_results"] is False

    def test_regression_result_returned(self, client, tmp_path):
        csv_path = tmp_path / "batch_reg.csv"
        csv_path.write_bytes(_REGRESSION_CSV)
        with Session(db_module.engine) as s:
            s.add(_make_deployment("dep3", "regression", "price"))
            s.add(_make_schedule("sched3", "dep3"))
            s.commit()
            s.add(_make_batch_run("run3", "sched3", "dep3", str(csv_path)))
            s.commit()
        r = client.get("/api/deploy/dep3/batch-results")
        assert r.status_code == 200
        body = r.json()
        assert body["has_results"] is True
        assert body["problem_type"] == "regression"
        assert "avg_prediction" in body

    def test_classification_result_returned(self, client, tmp_path):
        csv_path = tmp_path / "batch_cls.csv"
        csv_path.write_bytes(_CLASSIFICATION_CSV)
        with Session(db_module.engine) as s:
            s.add(_make_deployment("dep4", "classification", "status"))
            s.add(_make_schedule("sched4", "dep4"))
            s.commit()
            s.add(_make_batch_run("run4", "sched4", "dep4", str(csv_path)))
            s.commit()
        r = client.get("/api/deploy/dep4/batch-results")
        assert r.status_code == 200
        body = r.json()
        assert body["has_results"] is True
        assert body["problem_type"] == "classification"
        assert "class_distribution" in body


# ---------------------------------------------------------------------------
# Chat handler integration: SSE event emission
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"feat1,feat2,target\n"
    b"1.0,0.5,10.0\n2.0,1.0,20.0\n3.0,1.5,30.0\n4.0,2.0,40.0\n5.0,2.5,50.0\n"
    b"6.0,3.0,60.0\n7.0,3.5,70.0\n8.0,4.0,80.0\n9.0,4.5,90.0\n10.0,5.0,100.0\n"
)


def _chat_events(client, project_id: str, message: str) -> list[dict]:
    with mock_patch("anthropic.Anthropic") as mock_cls:
        mock_c = MagicMock()
        mock_cls.return_value = mock_c
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Batch results loaded."])
        mock_c.messages.stream.return_value = mock_stream
        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": message, "project_id": project_id},
        )
    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _setup_deployed_project(client, tmp_path, problem_type="regression"):
    """Create project → upload → features → target → train → deploy."""
    proj = client.post("/api/projects", json={"name": "BatchResultsTest"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "target", "problem_type": problem_type},
    )

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    run_id = train_resp.json()["model_run_ids"][0]
    for _ in range(40):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert run["status"] == "done"

    dep = client.post(f"/api/deploy/{run_id}")
    assert dep.status_code in (200, 201), dep.text
    return project_id, dep.json()["id"]


class TestBatchResultsChatHandler:
    def test_no_batch_event_when_no_jobs(self, client, tmp_path):
        project_id, _ = _setup_deployed_project(client, tmp_path)
        events = _chat_events(client, project_id, "show me batch results")
        types = [e.get("type") for e in events]
        # No batch job run exists → handler should return has_results=False event
        if "batch_job_results" in types:
            evt = next(e for e in events if e.get("type") == "batch_job_results")
            assert evt["batch_job_results"]["has_results"] is False

    def test_batch_event_emitted_when_job_exists(self, client, tmp_path):
        project_id, deployment_id = _setup_deployed_project(client, tmp_path)

        csv_path = tmp_path / "batch_out.csv"
        csv_path.write_bytes(_REGRESSION_CSV)

        with Session(db_module.engine) as s:
            s.add(_make_schedule("sched_int", deployment_id))
            s.commit()
            s.add(_make_batch_run("run_int", "sched_int", deployment_id, str(csv_path)))
            s.commit()

        events = _chat_events(client, project_id, "show me batch results")
        types = [e.get("type") for e in events]
        assert "batch_job_results" in types
        evt = next(e for e in events if e.get("type") == "batch_job_results")
        assert evt["batch_job_results"]["has_results"] is True
