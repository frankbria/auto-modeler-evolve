"""Tests for Prediction Log Analytics Chat Card.

Covers:
- _ANALYTICS_PATTERNS regex (positive + negative)
- Chat handler emitting prediction_analytics_chat SSE event with enriched stats
- No event when no deployment exists
- Fields: by_day, 7d/30d totals, peak_day, class_counts, avg_prediction
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session

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


def _setup_deployed_model(client):
    """Create project → upload → features → target → train → deploy."""
    proj = client.post("/api/projects", json={"name": "AnalyticsTest"})
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

    deploy = client.post(f"/api/deploy/{run_id}")
    assert deploy.status_code == 201
    deployment_id = deploy.json()["id"]
    return project_id, dataset_id, deployment_id


def _chat(client, project_id, message):
    with mock.patch("anthropic.Anthropic") as mock_ant:
        mock_client = mock.MagicMock()
        mock_ant.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Analytics shown."])
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


class TestAnalyticsPatterns:
    def setup_method(self):
        from api.chat import _ANALYTICS_PATTERNS

        self.pattern = _ANALYTICS_PATTERNS

    def test_prediction_analytics(self):
        assert self.pattern.search("show prediction analytics")

    def test_how_many_predictions(self):
        assert self.pattern.search("how many predictions have been made?")

    def test_prediction_count(self):
        assert self.pattern.search("prediction count for my model")

    def test_usage_stats(self):
        assert self.pattern.search("usage stats for my deployment")

    def test_prediction_volume(self):
        assert self.pattern.search("prediction volume this week")

    def test_prediction_log(self):
        assert self.pattern.search("show me the prediction log")

    def test_show_analytic(self):
        assert self.pattern.search("show analytic dashboard")

    def test_prediction_usage(self):
        assert self.pattern.search("prediction usage report")

    # Negative — should NOT trigger
    def test_negative_train(self):
        assert not self.pattern.search("train a new model for me")

    def test_negative_filter(self):
        assert not self.pattern.search("filter to East region only")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAnalyticsChatHandler:
    def test_no_deployment_no_event(self, client):
        """No event when there is no deployment."""
        proj = client.post("/api/projects", json={"name": "NoDeploy"})
        project_id = proj.json()["id"]

        resp = _chat(client, project_id, "show prediction analytics")
        assert resp.status_code == 200
        events = _parse_events(resp)
        types = [e.get("type") for e in events]
        assert "prediction_analytics_chat" not in types

    def test_deployed_model_emits_event(self, client):
        """prediction_analytics_chat event emitted when deployment exists."""
        project_id, _, _ = _setup_deployed_model(client)

        resp = _chat(client, project_id, "how many predictions have been made?")
        assert resp.status_code == 200
        events = _parse_events(resp)
        a_events = [e for e in events if e.get("type") == "prediction_analytics_chat"]
        assert len(a_events) == 1

    def test_event_has_required_fields(self, client):
        """All required fields present in analytics event."""
        project_id, _, _ = _setup_deployed_model(client)

        resp = _chat(client, project_id, "prediction volume report")
        assert resp.status_code == 200
        events = _parse_events(resp)
        a_events = [e for e in events if e.get("type") == "prediction_analytics_chat"]
        assert len(a_events) == 1

        data = a_events[0]["prediction_analytics_chat"]
        assert "deployment_id" in data
        assert "total_predictions" in data
        assert "predictions_last_7_days" in data
        assert "predictions_last_30_days" in data
        assert "predictions_today" in data
        assert "predictions_by_day" in data
        assert "summary" in data
        assert isinstance(data["predictions_by_day"], list)
        assert len(data["predictions_by_day"]) == 14

    def test_by_day_has_date_and_count(self, client):
        """Each entry in predictions_by_day has date and count keys."""
        project_id, _, _ = _setup_deployed_model(client)

        resp = _chat(client, project_id, "usage stats for my model")
        assert resp.status_code == 200
        events = _parse_events(resp)
        a_events = [e for e in events if e.get("type") == "prediction_analytics_chat"]
        assert len(a_events) == 1

        by_day = a_events[0]["prediction_analytics_chat"]["predictions_by_day"]
        for entry in by_day:
            assert "date" in entry
            assert "count" in entry
            assert isinstance(entry["count"], int)

    def test_no_predictions_all_zeros(self, client):
        """With no PredictionLog rows, counts are zero and peak_day is None."""
        project_id, _, _ = _setup_deployed_model(client)

        resp = _chat(client, project_id, "show me the prediction log")
        assert resp.status_code == 200
        events = _parse_events(resp)
        a_events = [e for e in events if e.get("type") == "prediction_analytics_chat"]
        assert len(a_events) == 1

        data = a_events[0]["prediction_analytics_chat"]
        assert data["predictions_last_7_days"] == 0
        assert data["predictions_last_30_days"] == 0
        assert data["predictions_today"] == 0
        assert data["peak_day"] is None

    def test_peak_day_populated_after_predictions(self, client):
        """peak_day is populated when PredictionLog rows exist."""
        from models.prediction_log import PredictionLog

        project_id, _, deployment_id = _setup_deployed_model(client)

        with Session(db_module.engine) as session:
            today = datetime.now(UTC).replace(tzinfo=None)
            for i in range(3):
                session.add(
                    PredictionLog(
                        deployment_id=deployment_id,
                        input_features='{"units": 10}',
                        prediction="150.0",
                        prediction_numeric=150.0,
                        created_at=today,
                    )
                )
            session.commit()

        resp = _chat(client, project_id, "how many predictions have been made?")
        assert resp.status_code == 200
        events = _parse_events(resp)
        a_events = [e for e in events if e.get("type") == "prediction_analytics_chat"]
        assert len(a_events) == 1

        data = a_events[0]["prediction_analytics_chat"]
        assert data["predictions_today"] == 3
        assert data["peak_day"] is not None
        assert data["peak_day"]["count"] == 3
