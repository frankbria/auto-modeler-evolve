"""Tests for cross-deployment webhook health summary via chat.

Covers:
- _WEBHOOK_HEALTH_PATTERNS regex matching (positive and negative)
- Chat handler emitting webhook_health_summary SSE event
- Correct aggregation of WebhookEvent rows across deployments
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

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

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    r = await ac.post("/api/projects", json={"name": "Webhook Health Test"})
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


class TestWebhookHealthPatterns:
    """Verify _WEBHOOK_HEALTH_PATTERNS matches all intended phrases."""

    @pytest.fixture(autouse=True)
    def _import_pattern(self):
        from api.chat import _WEBHOOK_HEALTH_PATTERNS

        self.pattern = _WEBHOOK_HEALTH_PATTERNS

    def test_are_my_webhooks_working(self):
        assert self.pattern.search("are my webhooks working?")

    def test_any_failed_webhooks(self):
        assert self.pattern.search("any failed webhooks?")

    def test_webhook_health(self):
        assert self.pattern.search("show webhook health")

    def test_webhook_status(self):
        assert self.pattern.search("webhook status")

    def test_webhook_errors(self):
        assert self.pattern.search("webhook errors")

    def test_check_webhook_health(self):
        assert self.pattern.search("check my webhook health")

    def test_webhook_failure_rate(self):
        assert self.pattern.search("webhook failure rate")

    def test_webhook_integration_status(self):
        assert self.pattern.search("webhook integration status")

    def test_negative_train_model(self):
        assert not self.pattern.search("train a new model")

    def test_negative_make_prediction(self):
        assert not self.pattern.search("make a prediction for units=100")


# ---------------------------------------------------------------------------
# Separation from history patterns
# ---------------------------------------------------------------------------


class TestWebhookHealthHistorySeparation:
    """Verify health patterns do not false-match on history phrases."""

    @pytest.fixture(autouse=True)
    def _import_patterns(self):
        from api.chat import _WEBHOOK_HEALTH_PATTERNS, _WEBHOOK_HISTORY_PATTERNS

        self.health = _WEBHOOK_HEALTH_PATTERNS
        self.history = _WEBHOOK_HISTORY_PATTERNS

    def test_history_phrase_hits_history_not_health(self):
        msg = "what webhooks fired recently?"
        assert self.history.search(msg)
        # health pattern must not match — so the mutual-exclusion guard works
        assert not self.health.search(msg)

    def test_webhook_log_hits_history(self):
        assert self.history.search("show webhook log")


# ---------------------------------------------------------------------------
# Integration: chat SSE emits webhook_health_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_webhook_config_returns_no_webhooks_status(
    ac, project_id, dataset_id
):
    """When no webhooks are configured, health summary returns no_webhooks status."""
    with mock.patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = mock.MagicMock()
        mock_anthropic.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["No webhooks configured."])
        mock_client.messages.stream.return_value = mock_stream

        events = []
        async with ac.stream(
            "POST",
            f"/api/chat/{project_id}",
            json={"message": "are my webhooks working?"},
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

    health_events = [e for e in events if e.get("type") == "webhook_health_summary"]
    assert len(health_events) == 1
    data = health_events[0]["webhook_health_summary"]
    assert data["overall_status"] == "no_webhooks"
    assert data["total_webhooks"] == 0


@pytest.mark.asyncio
async def test_health_event_has_required_fields(
    ac, project_id, dataset_id, deployment_id
):
    """webhook_health_summary event includes all required top-level fields."""
    with mock.patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = mock.MagicMock()
        mock_anthropic.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Webhook health checked."])
        mock_client.messages.stream.return_value = mock_stream

        events = []
        async with ac.stream(
            "POST",
            f"/api/chat/{project_id}",
            json={"message": "check webhook health"},
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

    health_events = [e for e in events if e.get("type") == "webhook_health_summary"]
    assert len(health_events) == 1
    data = health_events[0]["webhook_health_summary"]
    for field in ("overall_status", "total_webhooks", "total_events", "total_failed", "deployments", "summary"):
        assert field in data, f"Missing field: {field}"


@pytest.mark.asyncio
async def test_with_webhook_config_includes_deployment(
    ac, project_id, dataset_id, deployment_id
):
    """When a webhook is configured, deployment appears in the health summary."""
    # Register a webhook
    wh_resp = await ac.post(
        f"/api/deploy/{deployment_id}/webhooks",
        json={"url": "https://example.com/hook", "event_types": ["batch_complete"]},
    )
    assert wh_resp.status_code == 201, wh_resp.text

    with mock.patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = mock.MagicMock()
        mock_anthropic.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Webhook configured."])
        mock_client.messages.stream.return_value = mock_stream

        events = []
        async with ac.stream(
            "POST",
            f"/api/chat/{project_id}",
            json={"message": "are my webhooks working?"},
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

    health_events = [e for e in events if e.get("type") == "webhook_health_summary"]
    assert len(health_events) == 1
    data = health_events[0]["webhook_health_summary"]
    assert data["total_webhooks"] >= 1
    assert len(data["deployments"]) >= 1
    dep = data["deployments"][0]
    assert "webhooks" in dep
    assert len(dep["webhooks"]) >= 1
    wh_row = dep["webhooks"][0]
    assert wh_row["url"] == "https://example.com/hook"
    assert wh_row["status"] == "no_events"  # no events fired yet


@pytest.mark.asyncio
async def test_history_phrase_does_not_trigger_health(
    ac, project_id, dataset_id
):
    """A webhook history phrase should NOT emit webhook_health_summary."""
    with mock.patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = mock.MagicMock()
        mock_anthropic.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Webhook log shown."])
        mock_client.messages.stream.return_value = mock_stream

        events = []
        async with ac.stream(
            "POST",
            f"/api/chat/{project_id}",
            json={"message": "what webhooks fired recently?"},
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

    health_events = [e for e in events if e.get("type") == "webhook_health_summary"]
    assert len(health_events) == 0
