"""Tests for model health dashboard and guided retraining.

Covers:
- GET  /api/deploy/{deployment_id}/health  — unified health score
- POST /api/models/{project_id}/retrain    — smart retrain endpoint
- Chat _HEALTH_PATTERNS intent detection   — {type: health} SSE event
"""

from __future__ import annotations

import io
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

REGRESSION_CSV = b"""product,region,revenue,units
Widget A,North,1200.50,10
Widget B,South,850.00,8
Widget A,East,2100.75,18
Widget C,West,450.25,4
Widget B,North,1650.00,15
Widget A,South,980.00,9
Widget C,North,1100.25,11
Widget B,East,1750.00,16
Widget A,West,2300.50,20
Widget C,South,620.75,6
Widget A,North,1400.00,12
Widget B,West,900.00,9
Widget C,East,750.50,7
Widget A,South,1800.25,16
Widget B,North,2000.00,18
Widget A,East,950.75,8
Widget C,West,550.00,5
Widget B,South,1300.00,12
Widget A,North,2500.00,22
Widget C,East,800.00,8
"""

CLASSIFICATION_CSV = b"""feature_a,feature_b,label
1.2,3.4,cat
2.1,1.1,dog
3.3,2.2,cat
0.5,4.1,dog
2.8,0.9,cat
1.9,3.1,dog
3.0,2.5,cat
0.8,3.8,dog
2.3,1.5,cat
1.7,2.9,dog
2.5,1.8,cat
1.1,3.5,dog
3.2,2.0,cat
0.6,4.3,dog
2.7,1.2,cat
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa: F401
    import models.dataset  # noqa: F401
    import models.feature_set  # noqa: F401
    import models.conversation  # noqa: F401
    import models.model_run  # noqa: F401
    import models.deployment  # noqa: F401
    import models.prediction_log  # noqa: F401
    import models.feedback_record  # noqa: F401

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_mod

    data_mod.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_mod

    models_mod.MODELS_DIR = tmp_path / "models"

    import api.deploy as deploy_mod

    deploy_mod.DEPLOY_DIR = tmp_path / "deployments"

    from main import app

    with TestClient(app) as c:
        yield c


def _deploy_regression(client) -> tuple[str, str, str]:
    """Upload → apply → target → train → deploy. Returns (project_id, run_id, dep_id)."""
    pid = client.post("/api/projects", json={"name": "HealthTest"}).json()["id"]

    r = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(REGRESSION_CSV), "text/csv")},
        data={"project_id": pid},
    )
    did = r.json()["dataset_id"]

    fs = client.post(f"/api/features/{did}/apply", json={"transformations": []}).json()
    client.post(
        f"/api/features/{did}/target",
        json={"target_column": "revenue", "feature_set_id": fs["feature_set_id"]},
    )

    client.post(f"/api/models/{pid}/train", json={"algorithms": ["linear_regression"]})
    for _ in range(60):
        runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
        if all(r["status"] in ("done", "failed") for r in runs):
            break
        time.sleep(0.1)

    runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
    done_run = next(r for r in runs if r["status"] == "done")

    dep = client.post(f"/api/deploy/{done_run['id']}").json()
    return pid, done_run["id"], dep["id"]


def _deploy_classification(client) -> tuple[str, str, str]:
    """Upload → apply → target → train → deploy (classification). Returns (pid, run_id, dep_id)."""
    pid = client.post("/api/projects", json={"name": "HealthClsTest"}).json()["id"]

    r = client.post(
        "/api/data/upload",
        files={"file": ("cls.csv", io.BytesIO(CLASSIFICATION_CSV), "text/csv")},
        data={"project_id": pid},
    )
    did = r.json()["dataset_id"]

    fs = client.post(f"/api/features/{did}/apply", json={"transformations": []}).json()
    client.post(
        f"/api/features/{did}/target",
        json={
            "target_column": "label",
            "problem_type": "classification",
            "feature_set_id": fs["feature_set_id"],
        },
    )

    client.post(
        f"/api/models/{pid}/train", json={"algorithms": ["logistic_regression"]}
    )
    for _ in range(60):
        runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
        if all(r["status"] in ("done", "failed") for r in runs):
            break
        time.sleep(0.1)

    runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
    done_run = next(r for r in runs if r["status"] == "done")

    dep = client.post(f"/api/deploy/{done_run['id']}").json()
    return pid, done_run["id"], dep["id"]


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestModelHealth:
    def test_health_404_unknown_deployment(self, client):
        r = client.get("/api/deploy/nonexistent/health")
        assert r.status_code == 404

    def test_health_basic_fields(self, client):
        _, _, dep_id = _deploy_regression(client)
        r = client.get(f"/api/deploy/{dep_id}/health")
        assert r.status_code == 200
        body = r.json()
        assert "health_score" in body
        assert "status" in body
        assert "component_scores" in body
        assert "component_notes" in body
        assert "recommendations" in body
        assert "has_feedback_data" in body
        assert "has_drift_data" in body
        assert "model_age_days" in body

    def test_health_score_in_range(self, client):
        _, _, dep_id = _deploy_regression(client)
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert 0 <= body["health_score"] <= 100

    def test_health_status_is_valid(self, client):
        _, _, dep_id = _deploy_regression(client)
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert body["status"] in ("healthy", "warning", "critical")

    def test_health_fresh_model_age_score_high(self, client):
        """A freshly trained model should have age_score == 100."""
        _, _, dep_id = _deploy_regression(client)
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert body["component_scores"]["age"] == 100
        assert body["model_age_days"] == 0

    def test_health_no_feedback_returns_none_feedback_score(self, client):
        _, _, dep_id = _deploy_regression(client)
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert body["has_feedback_data"] is False
        assert body["component_scores"]["feedback"] is None

    def test_health_no_drift_returns_none_drift_score(self, client):
        _, _, dep_id = _deploy_regression(client)
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert body["has_drift_data"] is False
        assert body["component_scores"]["drift"] is None

    def test_health_recommendations_not_empty(self, client):
        _, _, dep_id = _deploy_regression(client)
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert len(body["recommendations"]) > 0

    def test_health_includes_algorithm(self, client):
        _, _, dep_id = _deploy_regression(client)
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert body["algorithm"] is not None

    def test_health_with_feedback_data(self, client):
        """After submitting feedback, has_feedback_data should be True."""
        _, _, dep_id = _deploy_regression(client)
        # Make a prediction first
        client.post(
            f"/api/predict/{dep_id}",
            json={"product": "Widget A", "region": "North", "units": 12},
        )
        # Submit feedback
        client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"actual_value": 1300.0},
        )
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert body["has_feedback_data"] is True
        assert body["component_scores"]["feedback"] is not None

    def test_health_with_classification(self, client):
        _, _, dep_id = _deploy_classification(client)
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert body["problem_type"] == "classification"
        assert 0 <= body["health_score"] <= 100

    def test_health_composite_score_age_only(self, client):
        """With no feedback or drift data, health_score == age_score."""
        _, _, dep_id = _deploy_regression(client)
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert not body["has_feedback_data"]
        assert not body["has_drift_data"]
        assert body["health_score"] == body["component_scores"]["age"]

    def test_health_deployment_id_in_response(self, client):
        _, _, dep_id = _deploy_regression(client)
        body = client.get(f"/api/deploy/{dep_id}/health").json()
        assert body["deployment_id"] == dep_id


# ---------------------------------------------------------------------------
# Retrain endpoint tests
# ---------------------------------------------------------------------------


class TestRetrainEndpoint:
    def test_retrain_404_unknown_project(self, client):
        r = client.post("/api/models/nonexistent/retrain")
        assert r.status_code == 404

    def test_retrain_400_no_dataset(self, client):
        pid = client.post("/api/projects", json={"name": "NoDataset"}).json()["id"]
        r = client.post(f"/api/models/{pid}/retrain")
        assert r.status_code == 400
        assert "dataset" in r.json()["detail"].lower()

    def test_retrain_400_no_completed_runs(self, client):
        """Retrain requires at least one completed run."""
        pid = client.post("/api/projects", json={"name": "NoRuns"}).json()["id"]
        r = client.post(
            "/api/data/upload",
            files={"file": ("d.csv", io.BytesIO(REGRESSION_CSV), "text/csv")},
            data={"project_id": pid},
        )
        did = r.json()["dataset_id"]
        fs = client.post(
            f"/api/features/{did}/apply", json={"transformations": []}
        ).json()
        client.post(
            f"/api/features/{did}/target",
            json={"target_column": "revenue", "feature_set_id": fs["feature_set_id"]},
        )
        r = client.post(f"/api/models/{pid}/retrain")
        assert r.status_code == 400
        assert "no completed" in r.json()["detail"].lower()

    def test_retrain_success_response_shape(self, client):
        pid, run_id, _ = _deploy_regression(client)
        r = client.post(f"/api/models/{pid}/retrain")
        assert r.status_code == 202
        body = r.json()
        assert "model_run_ids" in body
        assert "algorithms" in body
        assert body["status"] == "training_started"
        assert "source_run_id" in body
        assert body["project_id"] == pid

    def test_retrain_creates_new_run(self, client):
        """Retrain should add a new ModelRun on top of the existing one."""
        pid, _, _ = _deploy_regression(client)
        runs_before = client.get(f"/api/models/{pid}/runs").json()["runs"]
        client.post(f"/api/models/{pid}/retrain")
        runs_after = client.get(f"/api/models/{pid}/runs").json()["runs"]
        assert len(runs_after) > len(runs_before)

    def test_retrain_uses_same_algorithm(self, client):
        """The retrained run should use the same base algorithm as the original."""
        pid, run_id, _ = _deploy_regression(client)
        runs_before = client.get(f"/api/models/{pid}/runs").json()["runs"]
        original_algo = next(r["algorithm"] for r in runs_before if r["id"] == run_id)

        r = client.post(f"/api/models/{pid}/retrain")
        body = r.json()
        assert body["algorithms"][0] == original_algo

    def test_retrain_source_run_id_matches_original(self, client):
        pid, run_id, _ = _deploy_regression(client)
        r = client.post(f"/api/models/{pid}/retrain")
        body = r.json()
        # source_run_id should reference an existing run
        runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
        run_ids = {r["id"] for r in runs}
        assert body["source_run_id"] in run_ids

    def test_retrain_message_mentions_algorithm(self, client):
        pid, _, _ = _deploy_regression(client)
        body = client.post(f"/api/models/{pid}/retrain").json()
        assert body["algorithms"][0] in body["message"]

    def test_retrain_400_no_feature_set(self, client):
        """No active feature set should return a 400 with guidance."""
        pid = client.post("/api/projects", json={"name": "NoFS"}).json()["id"]
        client.post(
            "/api/data/upload",
            files={"file": ("d.csv", io.BytesIO(REGRESSION_CSV), "text/csv")},
            data={"project_id": pid},
        )
        # No apply/target — no feature set
        r = client.post(f"/api/models/{pid}/retrain")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Chat intent tests (health pattern)
# ---------------------------------------------------------------------------


def _mock_anthropic():
    mock_message = MagicMock()
    mock_message.__enter__ = lambda s: s
    mock_message.__exit__ = MagicMock(return_value=False)
    mock_message.text_stream = iter(["Model health is good."])
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_message
    return mock_client


class TestChatHealthIntent:
    def test_health_keywords_emit_health_event(self, client):
        """'model health' should trigger {type: health} SSE event."""
        pid, _, dep_id = _deploy_regression(client)

        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            r = client.post(
                f"/api/chat/{pid}",
                json={"message": "What is my model health?"},
            )

        assert r.status_code == 200
        chunks = [
            json.loads(line[6:])
            for line in r.text.splitlines()
            if line.startswith("data: ")
        ]
        event_types = {c["type"] for c in chunks}
        assert "health" in event_types

    def test_health_event_body_has_health_score(self, client):
        pid, _, dep_id = _deploy_regression(client)

        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            r = client.post(
                f"/api/chat/{pid}",
                json={"message": "Should I retrain my model?"},
            )

        chunks = [
            json.loads(line[6:])
            for line in r.text.splitlines()
            if line.startswith("data: ")
        ]
        health_event = next((c for c in chunks if c.get("type") == "health"), None)
        assert health_event is not None
        assert "health_score" in health_event["health"]
        assert "status" in health_event["health"]

    def test_irrelevant_message_no_health_event(self, client):
        """A generic data question shouldn't trigger health checking."""
        pid, _, _ = _deploy_regression(client)

        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            r = client.post(
                f"/api/chat/{pid}",
                json={"message": "What are the top selling products?"},
            )

        chunks = [
            json.loads(line[6:])
            for line in r.text.splitlines()
            if line.startswith("data: ")
        ]
        event_types = {c["type"] for c in chunks}
        assert "health" not in event_types

    def test_no_deployment_no_health_event(self, client):
        """Without a deployment, no health event should fire even with keywords."""
        pid = client.post("/api/projects", json={"name": "NoDep"}).json()["id"]

        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            r = client.post(
                f"/api/chat/{pid}",
                json={"message": "Should I retrain my model now?"},
            )

        chunks = [
            json.loads(line[6:])
            for line in r.text.splitlines()
            if line.startswith("data: ")
        ]
        event_types = {c["type"] for c in chunks}
        assert "health" not in event_types

    def test_update_model_triggers_health(self, client):
        pid, _, _ = _deploy_regression(client)

        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            r = client.post(
                f"/api/chat/{pid}",
                json={"message": "I need to update my model with new data"},
            )

        chunks = [
            json.loads(line[6:])
            for line in r.text.splitlines()
            if line.startswith("data: ")
        ]
        event_types = {c["type"] for c in chunks}
        assert "health" in event_types
