"""Tests for GET /api/models/{project_id}/history and _compute_trend."""

import io
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Shared CSV fixture
# ---------------------------------------------------------------------------

SAMPLE_CSV = b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-01,Widget B,South,850.00,8
2024-01-02,Widget A,East,2100.75,18
2024-01-02,Widget C,West,450.25,4
2024-01-03,Widget B,North,1650.00,15
2024-01-04,Widget A,South,980.00,9
2024-01-04,Widget C,North,1100.25,11
2024-01-05,Widget B,East,1750.00,16
2024-01-05,Widget A,West,2300.50,20
2024-01-06,Widget C,South,620.75,6
"""

CHURN_CSV = b"""age,tenure,spend,churned
25,12,200,0
40,3,150,1
35,24,300,0
28,6,100,1
50,36,400,0
32,9,180,1
45,18,250,0
29,4,90,1
55,48,500,0
38,15,220,0
"""


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.conversation  # noqa
    import models.model_run  # noqa
    import models.deployment  # noqa
    import models.prediction_log  # noqa
    import models.feedback_record  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _setup_trained_project(
    client, csv_bytes=SAMPLE_CSV, target="revenue", algorithm="linear_regression"
):
    """Helper: create project → upload → apply → set target → train one model."""
    proj = client.post("/api/projects", json={"name": "History Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": target})

    # Train synchronously (wait for completion) — endpoint returns 202 Accepted
    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": [algorithm]},
    )
    assert train_resp.status_code in (200, 202)

    # Poll until done (max 10s)
    for _ in range(40):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        if runs and runs[0]["status"] == "done":
            break
        time.sleep(0.25)

    return project_id, dataset_id


# ---------------------------------------------------------------------------
# Unit tests for _compute_trend
# ---------------------------------------------------------------------------


class TestComputeTrend:
    def test_improving_trend(self):
        from api.models import _compute_trend

        metrics = [0.5, 0.6, 0.7, 0.8, 0.85]
        trend, summary = _compute_trend(metrics, "R²")
        assert trend == "improving"
        assert "improved" in summary.lower()

    def test_declining_trend(self):
        from api.models import _compute_trend

        metrics = [0.85, 0.75, 0.65, 0.55, 0.45]
        trend, summary = _compute_trend(metrics, "R²")
        assert trend == "declining"
        assert "declined" in summary.lower()

    def test_stable_trend(self):
        from api.models import _compute_trend

        metrics = [0.80, 0.81, 0.80, 0.79, 0.80]
        trend, summary = _compute_trend(metrics, "R²")
        assert trend == "stable"
        assert "stable" in summary.lower()

    def test_insufficient_data_single_value(self):
        from api.models import _compute_trend

        metrics = [0.75]
        trend, summary = _compute_trend(metrics, "Accuracy")
        assert trend == "insufficient_data"
        assert "enough" in summary.lower()

    def test_insufficient_data_empty(self):
        from api.models import _compute_trend

        metrics = []
        trend, summary = _compute_trend(metrics, "Accuracy")
        assert trend == "insufficient_data"

    def test_exactly_two_points(self):
        from api.models import _compute_trend

        metrics = [0.60, 0.75]
        trend, summary = _compute_trend(metrics, "R²")
        assert trend in ("improving", "declining", "stable")

    def test_summary_contains_metric_label(self):
        from api.models import _compute_trend

        metrics = [0.5, 0.7, 0.9]
        _, summary = _compute_trend(metrics, "Accuracy")
        assert "Accuracy" in summary

    def test_summary_contains_run_count(self):
        from api.models import _compute_trend

        metrics = [0.5, 0.6, 0.7]
        _, summary = _compute_trend(metrics, "R²")
        assert "3" in summary


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestModelHistoryEndpoint:
    def test_404_for_missing_project(self, client):
        resp = client.get("/api/models/nonexistent-id/history")
        assert resp.status_code == 404

    def test_returns_empty_history_no_runs(self, client):
        """Project with no training runs should return empty runs list."""
        proj = client.post("/api/projects", json={"name": "Empty Project"})
        project_id = proj.json()["id"]

        # Upload + feature set but no training
        upload = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("d.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        )
        dataset_id = upload.json()["dataset_id"]
        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id}/target", json={"target_column": "revenue"}
        )

        resp = client.get(f"/api/models/{project_id}/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"] == []
        assert data["trend"] == "insufficient_data"
        assert data["best_metric"] is None
        assert data["latest_metric"] is None

    def test_history_shape_after_one_run(self, client):
        project_id, _ = _setup_trained_project(client)
        resp = client.get(f"/api/models/{project_id}/history")
        assert resp.status_code == 200
        data = resp.json()

        assert "project_id" in data
        assert "problem_type" in data
        assert "primary_metric" in data
        assert "primary_metric_label" in data
        assert "runs" in data
        assert "trend" in data
        assert "trend_summary" in data
        assert "best_metric" in data
        assert "latest_metric" in data

    def test_history_regression_primary_metric(self, client):
        project_id, _ = _setup_trained_project(client)
        resp = client.get(f"/api/models/{project_id}/history")
        data = resp.json()
        assert data["primary_metric"] == "r2"
        assert data["primary_metric_label"] == "R²"
        assert data["problem_type"] == "regression"

    def test_history_classification_primary_metric(self, client):
        project_id, _ = _setup_trained_project(
            client,
            csv_bytes=CHURN_CSV,
            target="churned",
            algorithm="logistic_regression",
        )
        resp = client.get(f"/api/models/{project_id}/history")
        data = resp.json()
        assert data["primary_metric"] == "accuracy"
        assert data["primary_metric_label"] == "Accuracy"
        assert data["problem_type"] == "classification"

    def test_runs_sorted_oldest_first(self, client):
        project_id, _ = _setup_trained_project(client)
        resp = client.get(f"/api/models/{project_id}/history")
        runs = resp.json()["runs"]
        if len(runs) >= 2:
            for i in range(len(runs) - 1):
                assert runs[i]["created_at"] <= runs[i + 1]["created_at"]

    def test_best_and_latest_metric_set_after_completed_run(self, client):
        project_id, _ = _setup_trained_project(client)
        resp = client.get(f"/api/models/{project_id}/history")
        data = resp.json()
        # With one completed run, best == latest
        assert data["best_metric"] is not None
        assert data["latest_metric"] is not None
        assert data["best_metric"] == data["latest_metric"]

    def test_run_fields_present(self, client):
        project_id, _ = _setup_trained_project(client)
        resp = client.get(f"/api/models/{project_id}/history")
        runs = resp.json()["runs"]
        assert len(runs) >= 1
        run = runs[0]
        assert "id" in run
        assert "algorithm" in run
        assert "status" in run
        assert "is_selected" in run
        assert "is_deployed" in run
        assert "metrics" in run
        assert "created_at" in run

    def test_trend_insufficient_with_one_run(self, client):
        """Single completed run → not enough data for trend direction."""
        project_id, _ = _setup_trained_project(client)
        resp = client.get(f"/api/models/{project_id}/history")
        data = resp.json()
        # One completed run → insufficient_data
        assert data["trend"] == "insufficient_data"

    def test_project_no_feature_set_returns_default_regression(self, client):
        """Project with dataset but no feature set → defaults to regression."""
        proj = client.post("/api/projects", json={"name": "No Feature Set"})
        project_id = proj.json()["id"]
        # Upload but no feature set
        client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("d.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        )
        resp = client.get(f"/api/models/{project_id}/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["problem_type"] == "regression"
        assert data["primary_metric"] == "r2"

    def test_project_no_dataset_returns_default(self, client):
        """Project with no dataset at all → still returns valid shape."""
        proj = client.post("/api/projects", json={"name": "No Data"})
        project_id = proj.json()["id"]
        resp = client.get(f"/api/models/{project_id}/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"] == []
        assert data["best_metric"] is None
