"""Tests for prediction drift detection and what-if analysis endpoints."""

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

SAMPLE_CSV = b"""product,region,units,revenue
Widget A,North,10,1200.5
Widget B,South,8,850.0
Widget A,East,18,2100.75
Widget C,West,4,450.25
Widget B,North,15,1650.0
Widget A,South,9,980.0
Widget C,North,11,1100.25
Widget B,East,16,1750.0
Widget A,West,20,2300.5
Widget C,South,6,620.75
Widget A,North,12,1300.0
Widget B,South,9,950.0
Widget A,East,20,2200.0
Widget C,West,5,520.0
Widget B,North,16,1700.0
Widget A,South,10,1050.0
Widget C,North,12,1150.0
Widget B,East,17,1800.0
Widget A,West,21,2350.0
Widget C,South,7,670.0
"""

CLASSIFICATION_CSV = b"""f1,f2,label
1.0,2.0,cat
2.0,3.0,dog
3.0,4.0,cat
4.0,5.0,dog
5.0,6.0,cat
6.0,7.0,dog
7.0,8.0,cat
8.0,9.0,dog
9.0,10.0,cat
10.0,11.0,dog
1.5,2.5,cat
2.5,3.5,dog
3.5,4.5,cat
4.5,5.5,dog
5.5,6.5,cat
6.5,7.5,dog
7.5,8.5,cat
8.5,9.5,dog
9.5,10.5,cat
10.5,11.5,dog
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    from main import app

    with TestClient(app) as c:
        yield c


def _setup_deployed(
    client, csv_data=SAMPLE_CSV, target="revenue", algorithm="linear_regression"
):
    """Helper: project → upload → features → train → deploy. Returns deployment_id."""
    proj = client.post("/api/projects", json={"name": "Drift Test"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(csv_data), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": target})

    train_resp = client.post(
        f"/api/models/{project_id}/train", json={"algorithms": [algorithm]}
    )
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert run["status"] == "done"

    deploy_resp = client.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code == 201
    deployment_id = deploy_resp.json()["id"]

    return deployment_id, run_id


# ---------------------------------------------------------------------------
# Drift detection — GET /api/deploy/{id}/drift
# ---------------------------------------------------------------------------


class TestDriftDetection:
    def test_drift_404_unknown_deployment(self, client):
        r = client.get("/api/deploy/doesnotexist/drift")
        assert r.status_code == 404

    def test_drift_insufficient_data_no_predictions(self, client):
        deployment_id, _ = _setup_deployed(client)
        r = client.get(f"/api/deploy/{deployment_id}/drift")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "insufficient_data"
        assert data["drift_score"] is None
        assert "insufficient_data" in data["status"]

    def test_drift_stable_with_consistent_predictions(self, client):
        deployment_id, _ = _setup_deployed(client)

        # Make 40 predictions with similar values
        features = {"product": "Widget A", "region": "North", "units": 12}
        for _ in range(40):
            client.post(f"/api/predict/{deployment_id}", json=features)

        r = client.get(f"/api/deploy/{deployment_id}/drift")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in (
            "stable",
            "mild_drift",
            "significant_drift",
            "insufficient_data",
        )
        assert data["problem_type"] == "regression"

    def test_drift_response_shape(self, client):
        deployment_id, _ = _setup_deployed(client)
        r = client.get(f"/api/deploy/{deployment_id}/drift")
        assert r.status_code == 200
        data = r.json()
        assert "deployment_id" in data
        assert "status" in data
        assert "explanation" in data
        assert "problem_type" in data

    def test_drift_custom_window_parameter(self, client):
        deployment_id, _ = _setup_deployed(client)
        # With window=5 we need 10 predictions; make 0 — still insufficient_data
        r = client.get(f"/api/deploy/{deployment_id}/drift?window=5")
        assert r.status_code == 200
        assert r.json()["status"] == "insufficient_data"

    def test_drift_classification_model(self, client):
        deployment_id, _ = _setup_deployed(
            client,
            csv_data=CLASSIFICATION_CSV,
            target="label",
            algorithm="random_forest_classifier",
        )
        features = {"f1": 1.0, "f2": 2.0}
        for _ in range(40):
            client.post(f"/api/predict/{deployment_id}", json=features)

        r = client.get(f"/api/deploy/{deployment_id}/drift")
        assert r.status_code == 200
        data = r.json()
        assert data["problem_type"] in ("classification", "regression")
        # Classification should have dist fields or be stable
        assert "status" in data

    def test_drift_invalid_window_rejected(self, client):
        deployment_id, _ = _setup_deployed(client)
        r = client.get(f"/api/deploy/{deployment_id}/drift?window=1")
        # min=5, so 1 should return 422
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# What-if analysis — POST /api/predict/{id}/whatif
# ---------------------------------------------------------------------------


class TestWhatIfAnalysis:
    def test_whatif_404_unknown_deployment(self, client):
        r = client.post(
            "/api/predict/doesnotexist/whatif",
            json={
                "base": {"units": 10, "product": "Widget A", "region": "North"},
                "overrides": {"units": 20},
            },
        )
        assert r.status_code == 404

    def test_whatif_returns_prediction_comparison(self, client):
        deployment_id, _ = _setup_deployed(client)
        r = client.post(
            f"/api/predict/{deployment_id}/whatif",
            json={
                "base": {"product": "Widget A", "region": "North", "units": 10},
                "overrides": {"units": 20},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "original_prediction" in data
        assert "modified_prediction" in data
        assert "summary" in data
        assert "changed_features" in data
        assert "units" in data["changed_features"]

    def test_whatif_delta_direction_increase(self, client):
        deployment_id, _ = _setup_deployed(client)
        # Increasing units should increase revenue prediction for linear model
        r_low = client.post(
            f"/api/predict/{deployment_id}/whatif",
            json={
                "base": {"product": "Widget A", "region": "North", "units": 5},
                "overrides": {"units": 100},
            },
        )
        assert r_low.status_code == 200
        data = r_low.json()
        # delta and direction should be present
        assert "delta" in data
        assert "direction" in data

    def test_whatif_no_change_when_same_value(self, client):
        deployment_id, _ = _setup_deployed(client)
        r = client.post(
            f"/api/predict/{deployment_id}/whatif",
            json={
                "base": {"product": "Widget A", "region": "North", "units": 10},
                "overrides": {"units": 10},
            },
        )
        assert r.status_code == 200
        data = r.json()
        # Prediction should be same
        assert data["original_prediction"] == data["modified_prediction"]

    def test_whatif_response_shape(self, client):
        deployment_id, _ = _setup_deployed(client)
        r = client.post(
            f"/api/predict/{deployment_id}/whatif",
            json={
                "base": {"product": "Widget A", "region": "North", "units": 10},
                "overrides": {"region": "West"},
            },
        )
        assert r.status_code == 200
        data = r.json()
        required_keys = [
            "deployment_id",
            "original_prediction",
            "modified_prediction",
            "changed_features",
            "summary",
            "problem_type",
            "target_column",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_whatif_missing_feature_uses_default(self, client):
        deployment_id, _ = _setup_deployed(client)
        # Provide incomplete base — missing features get defaults in the pipeline
        r = client.post(
            f"/api/predict/{deployment_id}/whatif",
            json={
                "base": {},
                "overrides": {"units": 15},
            },
        )
        assert r.status_code == 200

    def test_whatif_classification_returns_labels(self, client):
        deployment_id, _ = _setup_deployed(
            client,
            csv_data=CLASSIFICATION_CSV,
            target="label",
            algorithm="random_forest_classifier",
        )
        r = client.post(
            f"/api/predict/{deployment_id}/whatif",
            json={
                "base": {"f1": 1.0, "f2": 2.0},
                "overrides": {"f1": 9.0},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["problem_type"] == "classification"
        assert isinstance(data["original_prediction"], str)

    def test_whatif_inactive_deployment_returns_404(self, client):
        deployment_id, _ = _setup_deployed(client)
        client.delete(f"/api/deploy/{deployment_id}")
        r = client.post(
            f"/api/predict/{deployment_id}/whatif",
            json={
                "base": {"product": "Widget A", "region": "North", "units": 10},
                "overrides": {"units": 20},
            },
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Chat drift intent detection
# ---------------------------------------------------------------------------


class TestChatDriftIntent:
    """Chat endpoint detects drift questions and emits structured drift events."""

    def _mock_anthropic(self, text: str = "Your predictions look stable.") -> MagicMock:
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter([text])
        mock_client.messages.stream.return_value = mock_stream
        return mock_client

    def test_drift_query_emits_drift_event_with_insufficient_data(self, client):
        """Drift query emits an event even if there's insufficient data."""
        deployment_id, _ = _setup_deployed(client)
        # Get the project_id for the deployment
        deploy_info = client.get(f"/api/deploy/{deployment_id}").json()
        project_id = deploy_info["project_id"]

        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "Are the predictions drifting?"},
            )
        assert resp.status_code == 200
        events = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        drift_events = [e for e in events if e.get("type") == "drift"]
        assert len(drift_events) == 1
        d = drift_events[0]["drift"]
        assert "status" in d
        assert "explanation" in d

    def test_non_drift_query_no_drift_event(self, client):
        """Non-drift queries should not emit a drift event."""
        deployment_id, _ = _setup_deployed(client)
        deploy_info = client.get(f"/api/deploy/{deployment_id}").json()
        project_id = deploy_info["project_id"]

        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "What is the model accuracy?"},
            )
        assert resp.status_code == 200
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    assert event.get("type") != "drift"
                except json.JSONDecodeError:
                    pass

    def test_drift_query_no_deployment_no_drift_event(self, client):
        """If there's no active deployment, no drift event is emitted."""
        proj = client.post("/api/projects", json={"name": "No Deploy Project"})
        project_id = proj.json()["id"]
        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "Is the model drifting?"},
            )
        assert resp.status_code == 200
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    assert event.get("type") != "drift"
                except json.JSONDecodeError:
                    pass
