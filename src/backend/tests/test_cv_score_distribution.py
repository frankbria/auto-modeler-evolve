"""Tests for Cross-Validation Score Distribution Chat Integration.

Covers:
- _CV_SCORE_DIST_PATTERNS regex (positive + negative)
- Chat handler emitting cv_score_distribution SSE event when a completed run exists
- No event when no completed model runs exist
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
    b"East,160.0,16\nWest,210.0,21\n"
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
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


def _setup_with_trained_model(client):
    """Helper: create project → upload → features → target → train."""
    proj = client.post("/api/projects", json={"name": "CvDistTest"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "problem_type": "regression"},
    )

    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert train.status_code == 202
    run_id = train.json()["model_run_ids"][0]

    for _ in range(60):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    assert run and run["status"] == "done"
    return project_id


def _chat(client, project_id, message):
    with mock.patch("anthropic.Anthropic") as mock_ant:
        mock_client = mock.MagicMock()
        mock_ant.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["CV analysis shown."])
        mock_client.messages.stream.return_value = mock_stream

        resp = client.post(f"/api/chat/{project_id}", json={"message": message})
    return resp


def _parse_events(resp):
    return [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ") and line[6:].strip()
    ]


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestCvScoreDistPatterns:
    def setup_method(self):
        from api.chat import _CV_SCORE_DIST_PATTERNS

        self.pattern = _CV_SCORE_DIST_PATTERNS

    def test_how_consistent_is_my_model(self):
        assert self.pattern.search("how consistent is my model?")

    def test_cross_validation_scores(self):
        assert self.pattern.search("cross-validation scores")

    def test_show_fold_scores(self):
        assert self.pattern.search("show me the fold scores")

    def test_cv_variance(self):
        assert self.pattern.search("cv variance")

    def test_model_stability(self):
        assert self.pattern.search("model stability check")

    def test_is_my_model_stable(self):
        assert self.pattern.search("is my model stable?")

    def test_high_variance_in_cv(self):
        assert self.pattern.search("high variance in my cv folds")

    def test_fold_by_fold_performance(self):
        assert self.pattern.search("fold-by-fold performance")

    # Negative tests — should NOT trigger CV distribution
    def test_negative_train_model(self):
        assert not self.pattern.search("train a model to predict revenue")

    def test_negative_make_prediction(self):
        assert not self.pattern.search("make a prediction for region=East, units=100")


# ---------------------------------------------------------------------------
# Integration tests (chat SSE handler)
# ---------------------------------------------------------------------------


class TestCvScoreDistHandler:
    def test_no_runs_no_event(self, client):
        """No cv_score_distribution event when no completed model runs exist."""
        proj = client.post("/api/projects", json={"name": "NoRunsProj"})
        project_id = proj.json()["id"]

        resp = _chat(client, project_id, "how consistent is my model?")
        assert resp.status_code == 200
        events = _parse_events(resp)
        types = [e.get("type") for e in events]
        assert "cv_score_distribution" not in types

    def test_completed_run_emits_event(self, client):
        """cv_score_distribution event emitted when a completed run exists."""
        project_id = _setup_with_trained_model(client)

        resp = _chat(client, project_id, "how consistent is my model?")
        assert resp.status_code == 200
        events = _parse_events(resp)
        cv_events = [e for e in events if e.get("type") == "cv_score_distribution"]
        assert len(cv_events) == 1

    def test_event_has_required_fields(self, client):
        """cv_score_distribution event contains all expected fields."""
        project_id = _setup_with_trained_model(client)

        resp = _chat(client, project_id, "show me cross-validation scores")
        assert resp.status_code == 200
        events = _parse_events(resp)
        cv_events = [e for e in events if e.get("type") == "cv_score_distribution"]
        assert len(cv_events) == 1

        data = cv_events[0]["cv_score_distribution"]
        assert "algorithm" in data
        assert "algorithm_plain" in data
        assert "problem_type" in data
        assert "metric" in data
        assert "metric_plain" in data
        assert "scores" in data
        assert isinstance(data["scores"], list)
        assert len(data["scores"]) > 0
        assert "mean" in data
        assert "std" in data
        assert "ci_low" in data
        assert "ci_high" in data
        assert "n_splits" in data
        assert "consistency" in data
        assert data["consistency"] in ("stable", "moderate", "variable")
        assert "consistency_pct" in data
        assert "summary" in data
        assert data["problem_type"] == "regression"
        assert data["metric_plain"] == "R²"
