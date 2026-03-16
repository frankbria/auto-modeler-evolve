"""Tests for Model Monitoring Alerts (GET /api/projects/{id}/alerts) and
chat intent detection for alerts / history / analytics.

Day 4 (10:00): Phase 8 Track B — proactive system-wide alerts + chat-triggered panels.
"""

import io
import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

REGRESSION_CSV = b"""age,income,score,target
25,50000,80,100.0
30,60000,85,120.0
35,70000,90,140.0
40,80000,75,160.0
45,90000,95,180.0
50,100000,70,200.0
55,55000,82,110.0
28,62000,88,125.0
33,73000,91,145.0
38,84000,78,165.0
43,95000,96,185.0
48,106000,72,205.0
53,57000,84,115.0
26,64000,87,128.0
31,75000,92,148.0
36,86000,76,168.0
41,97000,97,188.0
46,108000,71,208.0
51,59000,83,118.0
29,66000,89,131.0
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
    import models.feedback_record  # noqa

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


@pytest.fixture
def deployed_project(client):
    """Full pipeline: project → upload → features → train → select → deploy."""
    proj = client.post("/api/projects", json={"name": "Alerts Test"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(REGRESSION_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": "target"})

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    client.post(f"/api/models/{run_id}/select")
    deploy_resp = client.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code == 201

    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "deployment_id": deploy_resp.json()["id"],
    }


def _mock_anthropic():
    """Return a mock Anthropic client for chat tests."""
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(["ok"])
    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream
    return mock_client


# ---------------------------------------------------------------------------
# GET /api/projects/{id}/alerts — unit tests (no real deployment needed)
# ---------------------------------------------------------------------------


class TestAlertsEndpointBasic:
    """Basic endpoint contract tests."""

    def test_alerts_404_unknown_project(self, client):
        resp = client.get("/api/projects/nonexistent-id/alerts")
        assert resp.status_code == 404

    def test_alerts_empty_no_deployments(self, client):
        proj = client.post("/api/projects", json={"name": "Empty"}).json()
        resp = client.get(f"/api/projects/{proj['id']}/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alert_count"] == 0
        assert data["critical_count"] == 0
        assert data["warning_count"] == 0
        assert data["alerts"] == []

    def test_alerts_response_schema(self, deployed_project, client):
        project_id = deployed_project["project_id"]
        resp = client.get(f"/api/projects/{project_id}/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "project_id" in data
        assert "alert_count" in data
        assert "critical_count" in data
        assert "warning_count" in data
        assert "alerts" in data
        assert isinstance(data["alerts"], list)
        assert data["project_id"] == project_id

    def test_alerts_count_consistency(self, deployed_project, client):
        """critical_count + warning_count should equal alert_count."""
        project_id = deployed_project["project_id"]
        resp = client.get(f"/api/projects/{project_id}/alerts")
        data = resp.json()
        assert data["critical_count"] + data["warning_count"] == data["alert_count"]

    def test_no_predictions_alert_triggered(self, deployed_project, client):
        """A deployment with 0 predictions after >= 1 day should emit no_predictions alert."""
        from models.deployment import Deployment

        deployment_id = deployed_project["deployment_id"]
        project_id = deployed_project["project_id"]

        # Back-date the deployment so it's > 1 day old
        with Session(db_module.engine) as session:
            dep = session.get(Deployment, deployment_id)
            dep.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=2)
            dep.request_count = 0
            session.add(dep)
            session.commit()

        resp = client.get(f"/api/projects/{project_id}/alerts")
        data = resp.json()
        types = [a["type"] for a in data["alerts"]]
        assert "no_predictions" in types

    def test_no_predictions_alert_not_triggered_on_day0(self, deployed_project, client):
        """Fresh deployment (0 days old) should NOT emit no_predictions alert."""
        project_id = deployed_project["project_id"]
        resp = client.get(f"/api/projects/{project_id}/alerts")
        data = resp.json()
        types = [a["type"] for a in data["alerts"]]
        assert "no_predictions" not in types

    def test_stale_model_warning_after_61_days(self, deployed_project, client):
        """Model older than 60 days should emit stale_model warning."""
        from models.model_run import ModelRun

        run_id = deployed_project["run_id"]
        project_id = deployed_project["project_id"]

        with Session(db_module.engine) as session:
            run = session.get(ModelRun, run_id)
            run.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=61)
            session.add(run)
            session.commit()

        resp = client.get(f"/api/projects/{project_id}/alerts")
        data = resp.json()
        stale = [a for a in data["alerts"] if a["type"] == "stale_model"]
        assert len(stale) > 0
        assert stale[0]["severity"] == "warning"

    def test_stale_model_critical_after_91_days(self, deployed_project, client):
        """Model older than 90 days should emit stale_model critical."""
        from models.model_run import ModelRun

        run_id = deployed_project["run_id"]
        project_id = deployed_project["project_id"]

        with Session(db_module.engine) as session:
            run = session.get(ModelRun, run_id)
            run.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=95)
            session.add(run)
            session.commit()

        resp = client.get(f"/api/projects/{project_id}/alerts")
        data = resp.json()
        stale = [a for a in data["alerts"] if a["type"] == "stale_model"]
        assert any(a["severity"] == "critical" for a in stale)

    def test_alert_schema_fields(self, deployed_project, client):
        """Each alert dict must have all required fields."""
        from models.deployment import Deployment
        from models.model_run import ModelRun

        project_id = deployed_project["project_id"]
        run_id = deployed_project["run_id"]
        deployment_id = deployed_project["deployment_id"]

        # Force a stale model + no-predictions alert
        with Session(db_module.engine) as session:
            run = session.get(ModelRun, run_id)
            run.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=95)
            session.add(run)
            dep = session.get(Deployment, deployment_id)
            dep.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=2)
            dep.request_count = 0
            session.add(dep)
            session.commit()

        resp = client.get(f"/api/projects/{project_id}/alerts")
        for alert in resp.json()["alerts"]:
            assert "deployment_id" in alert
            assert "algorithm" in alert
            assert "severity" in alert
            assert alert["severity"] in ("critical", "warning")
            assert "type" in alert
            assert "message" in alert
            assert "recommendation" in alert

    def test_alerts_sorted_critical_first(self, deployed_project, client):
        """Critical alerts should appear before warning alerts."""
        from models.deployment import Deployment
        from models.model_run import ModelRun

        project_id = deployed_project["project_id"]
        run_id = deployed_project["run_id"]
        deployment_id = deployed_project["deployment_id"]

        # Force critical (stale >90d) + warning (no predictions >1d)
        with Session(db_module.engine) as session:
            run = session.get(ModelRun, run_id)
            run.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=95)
            session.add(run)
            dep = session.get(Deployment, deployment_id)
            dep.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=2)
            dep.request_count = 0
            session.add(dep)
            session.commit()

        resp = client.get(f"/api/projects/{project_id}/alerts")
        alerts = resp.json()["alerts"]
        if len(alerts) >= 2:
            # First critical must come before any warning
            severities = [a["severity"] for a in alerts]
            critical_idx = [i for i, s in enumerate(severities) if s == "critical"]
            warning_idx = [i for i, s in enumerate(severities) if s == "warning"]
            if critical_idx and warning_idx:
                assert max(critical_idx) < min(warning_idx)

    def test_inactive_deployments_excluded(self, deployed_project, client):
        """Inactive (undeployed) deployments should not appear in alerts."""

        project_id = deployed_project["project_id"]
        deployment_id = deployed_project["deployment_id"]

        # Undeploy
        client.delete(f"/api/deploy/{deployment_id}")

        resp = client.get(f"/api/projects/{project_id}/alerts")
        data = resp.json()
        assert data["alert_count"] == 0


# ---------------------------------------------------------------------------
# Chat intent — alerts / history / analytics patterns
# ---------------------------------------------------------------------------


class TestChatAlertsIntent:
    """Chat SSE emits {type: alerts} for monitoring-related keywords."""

    def test_alerts_keyword_triggers_event(self, deployed_project, client):
        project_id = deployed_project["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "any alerts for my models?"},
            )
        assert resp.status_code == 200
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        types = [c["type"] for c in chunks]
        assert "alerts" in types

    def test_monitor_keyword_triggers_event(self, deployed_project, client):
        project_id = deployed_project["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "monitor my deployment"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        assert "alerts" in [c["type"] for c in chunks]

    def test_alerts_event_schema(self, deployed_project, client):
        """alerts SSE event must include project_id and alert_count."""
        project_id = deployed_project["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "check my models for issues"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        alerts_events = [c for c in chunks if c.get("type") == "alerts"]
        assert len(alerts_events) == 1
        payload = alerts_events[0]["alerts"]
        assert "project_id" in payload
        assert "alert_count" in payload
        assert "alerts" in payload

    def test_irrelevant_message_no_alerts_event(self, deployed_project, client):
        project_id = deployed_project["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "what is the correlation between age and income?"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        assert "alerts" not in [c["type"] for c in chunks]


class TestChatHistoryIntent:
    """Chat SSE emits {type: history} for version-history keywords."""

    def test_show_history_triggers_event(self, deployed_project, client):
        project_id = deployed_project["project_id"]
        # Train a second run so there are 2+ completed runs
        dataset_id = deployed_project["dataset_id"]
        train_resp = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        run_id2 = train_resp.json()["model_run_ids"][0]
        for _ in range(30):
            runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
            run = next(r for r in runs if r["id"] == run_id2)
            if run["status"] in ("done", "failed"):
                break
            time.sleep(0.5)

        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "show model history"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        assert "history" in [c["type"] for c in chunks]

    def test_model_history_event_schema(self, deployed_project, client):
        project_id = deployed_project["project_id"]
        # Need 2+ completed runs for history event to fire
        train_resp = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        run_id2 = train_resp.json()["model_run_ids"][0]
        for _ in range(30):
            runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
            run = next(r for r in runs if r["id"] == run_id2)
            if run["status"] in ("done", "failed"):
                break
            time.sleep(0.5)

        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "how is my model improving over time?"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        hist = [c for c in chunks if c.get("type") == "history"]
        if hist:  # Only fires if 2+ completed runs exist
            assert "history" in hist[0]
            assert hist[0]["history"]["project_id"] == project_id

    def test_history_not_triggered_when_no_runs(self, client):
        """No history event when project has no completed model runs."""
        proj = client.post("/api/projects", json={"name": "Fresh"}).json()
        project_id = proj["id"]
        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "show model history"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        assert "history" not in [c["type"] for c in chunks]


class TestChatAnalyticsIntent:
    """Chat SSE emits {type: analytics} for prediction-analytics keywords."""

    def test_how_many_predictions_triggers_event(self, deployed_project, client):
        project_id = deployed_project["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "how many predictions have been made?"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        assert "analytics" in [c["type"] for c in chunks]

    def test_usage_stats_triggers_event(self, deployed_project, client):
        project_id = deployed_project["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "show me the usage stats"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        assert "analytics" in [c["type"] for c in chunks]

    def test_analytics_event_schema(self, deployed_project, client):
        project_id = deployed_project["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "prediction analytics please"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        analytics_events = [c for c in chunks if c.get("type") == "analytics"]
        assert len(analytics_events) == 1
        payload = analytics_events[0]["analytics"]
        assert "deployment_id" in payload
        assert "total_predictions" in payload

    def test_analytics_no_event_without_deployment(self, client):
        """No analytics event when project has no active deployment."""
        proj = client.post("/api/projects", json={"name": "NoDeployment"}).json()
        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{proj['id']}",
                json={"message": "how many predictions have been made?"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        assert "analytics" not in [c["type"] for c in chunks]

    def test_irrelevant_message_no_analytics_event(self, deployed_project, client):
        project_id = deployed_project["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=_mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "tell me about my data features"},
            )
        chunks = [
            json.loads(line[5:])
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        assert "analytics" not in [c["type"] for c in chunks]
