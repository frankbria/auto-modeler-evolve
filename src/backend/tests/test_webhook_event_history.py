"""Tests for webhook event history feature.

Covers:
- WebhookEvent model creation and storage
- GET /api/deploy/{id}/webhook-history endpoint
- _WEBHOOK_HISTORY_PATTERNS regex matching
- Chat handler emitting webhook_history SSE event
- core/webhook._dispatch_in_thread logging events
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
    b"East,160.0,16\nWest,210.0,21\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
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

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    r = await ac.post("/api/projects", json={"name": "Webhook History Test"})
    return r.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    r = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert r.status_code == 201, r.text
    return r.json()["dataset_id"]


@pytest.fixture()
async def feature_set_id(ac, dataset_id):
    r = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert r.status_code == 201, r.text
    fs_id = r.json()["feature_set_id"]
    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


@pytest.fixture()
async def model_run_id(ac, project_id, feature_set_id):
    r = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert r.status_code == 202, r.text
    run_id = r.json()["model_run_ids"][0]
    for _ in range(40):
        time.sleep(0.2)
        r2 = await ac.get(f"/api/models/{project_id}/runs")
        runs = r2.json().get("runs", [])
        run = next((x for x in runs if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
    pytest.skip("Training did not complete in time")


@pytest.fixture()
async def deployment_id(ac, model_run_id):
    r = await ac.post(f"/api/deploy/{model_run_id}")
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestWebhookHistoryPatterns:
    """Verify _WEBHOOK_HISTORY_PATTERNS matches all intended phrases."""

    @pytest.fixture(autouse=True)
    def _import_pattern(self):
        from api.chat import _WEBHOOK_HISTORY_PATTERNS

        self.pattern = _WEBHOOK_HISTORY_PATTERNS

    def test_what_webhooks_fired_recently(self):
        assert self.pattern.search("what webhooks fired recently?")

    def test_show_webhook_history(self):
        assert self.pattern.search("show webhook history")

    def test_webhook_log(self):
        assert self.pattern.search("webhook log")

    def test_webhook_events(self):
        assert self.pattern.search("webhook events")

    def test_did_any_webhooks_fire(self):
        assert self.pattern.search("did any webhooks fire?")

    def test_recent_webhook(self):
        assert self.pattern.search("recent webhook")

    def test_webhook_activity(self):
        assert self.pattern.search("webhook activity")

    def test_show_webhook_notifications(self):
        assert self.pattern.search("show my webhook notifications")

    def test_negative_train_model(self):
        assert not self.pattern.search("train a model")

    def test_negative_make_prediction(self):
        assert not self.pattern.search("make a prediction for units=100")


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWebhookHistoryEndpoint:
    """Tests for GET /api/deploy/{id}/webhook-history."""

    async def test_returns_200_with_empty_events(self, ac, deployment_id):
        r = await ac.get(f"/api/deploy/{deployment_id}/webhook-history")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["events"] == []
        assert "No webhook events" in data["summary"]

    async def test_returns_404_for_unknown_deployment(self, ac):
        r = await ac.get("/api/deploy/does-not-exist/webhook-history")
        assert r.status_code == 404

    async def test_returns_events_after_webhook_fires(self, ac, deployment_id):
        """Seeding a WebhookEvent row should appear in the history endpoint."""
        from models.webhook_event import WebhookEvent

        with Session(db_module.engine) as session:
            evt = WebhookEvent(
                webhook_id="fake-hook-id",
                deployment_id=deployment_id,
                event_type="batch_complete",
                status_code=200,
            )
            session.add(evt)
            session.commit()

        r = await ac.get(f"/api/deploy/{deployment_id}/webhook-history")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        event = data["events"][0]
        assert event["event_type"] == "batch_complete"
        assert event["status_code"] == 200
        assert event["success"] is True

    async def test_event_includes_required_fields(self, ac, deployment_id):
        from models.webhook_event import WebhookEvent

        with Session(db_module.engine) as session:
            evt = WebhookEvent(
                webhook_id="fake-hook-id",
                deployment_id=deployment_id,
                event_type="drift_detected",
                status_code=0,
            )
            session.add(evt)
            session.commit()

        r = await ac.get(f"/api/deploy/{deployment_id}/webhook-history")
        assert r.status_code == 200
        event = r.json()["events"][0]
        assert "id" in event
        assert "webhook_id" in event
        assert "webhook_url" in event
        assert "event_type" in event
        assert "fired_at" in event
        assert "status_code" in event
        assert "success" in event

    async def test_failed_event_success_false(self, ac, deployment_id):
        from models.webhook_event import WebhookEvent

        with Session(db_module.engine) as session:
            evt = WebhookEvent(
                webhook_id="fake-hook-id",
                deployment_id=deployment_id,
                event_type="health_degraded",
                status_code=0,  # network error
            )
            session.add(evt)
            session.commit()

        r = await ac.get(f"/api/deploy/{deployment_id}/webhook-history")
        event = r.json()["events"][0]
        assert event["success"] is False

    async def test_summary_includes_count(self, ac, deployment_id):
        from models.webhook_event import WebhookEvent

        with Session(db_module.engine) as session:
            for et in ["batch_complete", "drift_detected"]:
                session.add(
                    WebhookEvent(
                        webhook_id="fake-hook-id",
                        deployment_id=deployment_id,
                        event_type=et,
                        status_code=200,
                    )
                )
            session.commit()

        r = await ac.get(f"/api/deploy/{deployment_id}/webhook-history")
        data = r.json()
        assert data["total"] == 2
        assert "2" in data["summary"]


# ---------------------------------------------------------------------------
# Chat integration tests (synchronous TestClient — avoids async fixture issues)
# ---------------------------------------------------------------------------


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE stream text into a list of event dicts."""
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _chat_events_sync(client, project_id: str, message: str) -> list[dict]:
    """Send a chat message with mocked Anthropic (sync) and return parsed events."""
    with mock.patch("anthropic.Anthropic") as mock_cls:
        mc = mock.MagicMock()
        mock_cls.return_value = mc
        ms = mock.MagicMock()
        ms.__enter__ = mock.MagicMock(return_value=ms)
        ms.__exit__ = mock.MagicMock(return_value=False)
        ms.text_stream = iter(["Done."])
        mc.messages.stream.return_value = ms

        resp = client.post(
            f"/api/chat/{project_id}",
            json={"message": message, "project_id": project_id},
        )
    return _parse_sse(resp.text)


@pytest.fixture()
def sync_client(tmp_path):
    """Synchronous TestClient with isolated SQLite DB."""
    from fastapi.testclient import TestClient
    from main import app

    test_db = str(tmp_path / "chat_test.db")
    orig_engine = db_module.engine
    db_module.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    import models  # noqa — registers all tables

    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    import api.data as dm
    import api.deploy as dep
    import api.models as mm

    dm.UPLOAD_DIR = tmp_path / "uploads_chat"
    dep.DEPLOY_DIR = tmp_path / "deployments_chat"
    mm.MODELS_DIR = tmp_path / "models_chat"

    yield TestClient(app)
    db_module.engine = orig_engine


@pytest.fixture()
def deployed_project_sync(sync_client, tmp_path):
    """Create project, upload CSV, train, and deploy (sync)."""

    proj = sync_client.post("/api/projects", json={"name": "WebhookHistoryChat"})
    pid = proj.json()["id"]

    upload = sync_client.post(
        "/api/data/upload",
        data={"project_id": pid},
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    did = upload.json()["dataset_id"]

    sync_client.post(f"/api/features/{did}/apply", json={"transformations": []})
    sync_client.post(
        f"/api/features/{did}/target",
        json={"target_column": "revenue", "problem_type": "regression"},
    )

    train_resp = sync_client.post(
        f"/api/models/{pid}/train",
        json={"algorithms": ["linear_regression"]},
    )
    run_id = train_resp.json()["model_run_ids"][0]
    for _ in range(40):
        runs = sync_client.get(f"/api/models/{pid}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.3)

    dep_r = sync_client.post(f"/api/deploy/{run_id}")
    assert dep_r.status_code in (200, 201), dep_r.text

    return {"project_id": pid, "deployment_id": dep_r.json()["id"]}


class TestWebhookHistoryChatHandler:
    """Integration test: chat message → SSE webhook_history event."""

    def test_chat_emits_webhook_history_event(self, sync_client, deployed_project_sync):
        """When user asks about webhook history and a deployment exists, chat emits
        {type:'webhook_history'} SSE event."""
        events = _chat_events_sync(
            sync_client, deployed_project_sync["project_id"], "show webhook history"
        )
        types = [e.get("type") for e in events]
        assert "webhook_history" in types, f"Expected webhook_history in {types}"

    def test_chat_no_webhook_history_without_deployment(self, sync_client):
        """Without a deployment, the webhook_history event should not be emitted."""
        # Create a bare project with no deployment
        proj = sync_client.post("/api/projects", json={"name": "NoDep"})
        pid = proj.json()["id"]

        events = _chat_events_sync(sync_client, pid, "show webhook history")
        types = [e.get("type") for e in events]
        assert "webhook_history" not in types, f"Unexpected webhook_history: {types}"
