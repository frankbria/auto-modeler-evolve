"""Tests for webhook notification endpoints and dispatch logic.

Covers:
- POST /api/deploy/{id}/webhooks  — create webhook (returns secret once)
- GET  /api/deploy/{id}/webhooks  — list webhooks (secrets excluded)
- DELETE /api/deploy/{id}/webhooks/{wid}  — soft-delete
- POST /api/deploy/{id}/webhooks/{wid}/test  — send test payload
- create_webhook rejects invalid URL scheme
- create_webhook rejects unknown event_types
- create_webhook returns 404 for inactive/missing deployment
- dispatch_webhooks fires only matching event_types
- dispatch_webhooks is non-blocking and handles errors gracefully
- _sign_payload returns consistent HMAC-SHA256 digest
- WebhookConfig model auto-generates unique secret per row
"""

from __future__ import annotations

import hashlib
import hmac
import io
import unittest.mock as mock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine

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
    r = await ac.post("/api/projects", json={"name": "Webhook Test"})
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
    import time

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
# Webhook model unit tests
# ---------------------------------------------------------------------------


def test_webhook_config_generates_secret():
    """Each WebhookConfig row gets a unique random secret."""
    from models.webhook_config import WebhookConfig

    wh1 = WebhookConfig(deployment_id="a", url="http://x.com")
    wh2 = WebhookConfig(deployment_id="a", url="http://x.com")
    assert len(wh1.secret) == 64  # 32 bytes hex
    assert wh1.secret != wh2.secret


def test_sign_payload_deterministic():
    """_sign_payload returns the same digest for the same inputs."""
    from core.webhook import _sign_payload

    secret = "my-secret"
    body = b'{"event_type": "test"}'
    sig1 = _sign_payload(secret, body)
    sig2 = _sign_payload(secret, body)
    assert sig1 == sig2
    # Verify it matches stdlib hmac
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert sig1 == expected


def test_sign_payload_changes_with_content():
    from core.webhook import _sign_payload

    secret = "my-secret"
    sig1 = _sign_payload(secret, b"payload A")
    sig2 = _sign_payload(secret, b"payload B")
    assert sig1 != sig2


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_create_webhook_returns_secret_once(ac, deployment_id):
    r = await ac.post(
        f"/api/deploy/{deployment_id}/webhooks",
        json={"url": "https://example.com/hook", "event_types": ["batch_complete"]},
    )
    assert r.status_code == 201
    data = r.json()
    assert data["url"] == "https://example.com/hook"
    assert data["event_types"] == ["batch_complete"]
    assert "secret" in data
    assert len(data["secret"]) == 64  # 32-byte hex


@pytest.mark.anyio
async def test_list_webhooks_excludes_secret(ac, deployment_id):
    await ac.post(
        f"/api/deploy/{deployment_id}/webhooks",
        json={"url": "https://example.com/hook"},
    )
    r = await ac.get(f"/api/deploy/{deployment_id}/webhooks")
    assert r.status_code == 200
    hooks = r.json()
    assert len(hooks) >= 1
    assert "secret" not in hooks[0]


@pytest.mark.anyio
async def test_delete_webhook(ac, deployment_id):
    r = await ac.post(
        f"/api/deploy/{deployment_id}/webhooks",
        json={"url": "https://example.com/hook"},
    )
    wh_id = r.json()["id"]
    del_r = await ac.delete(f"/api/deploy/{deployment_id}/webhooks/{wh_id}")
    assert del_r.status_code == 204
    # Should no longer appear in list
    list_r = await ac.get(f"/api/deploy/{deployment_id}/webhooks")
    ids = [w["id"] for w in list_r.json()]
    assert wh_id not in ids


@pytest.mark.anyio
async def test_create_webhook_invalid_url(ac, deployment_id):
    r = await ac.post(
        f"/api/deploy/{deployment_id}/webhooks",
        json={"url": "ftp://bad-scheme.com"},
    )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_create_webhook_unknown_event_type(ac, deployment_id):
    r = await ac.post(
        f"/api/deploy/{deployment_id}/webhooks",
        json={"url": "https://ok.com", "event_types": ["unknown_event"]},
    )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_create_webhook_missing_deployment(ac):
    r = await ac.post(
        "/api/deploy/nonexistent-id/webhooks",
        json={"url": "https://ok.com"},
    )
    assert r.status_code == 404


@pytest.mark.anyio
async def test_list_webhooks_missing_deployment(ac):
    r = await ac.get("/api/deploy/nonexistent-id/webhooks")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_delete_webhook_wrong_deployment(ac, deployment_id):
    r = await ac.post(
        f"/api/deploy/{deployment_id}/webhooks",
        json={"url": "https://example.com/hook"},
    )
    wh_id = r.json()["id"]
    del_r = await ac.delete(f"/api/deploy/wrong-id/webhooks/{wh_id}")
    assert del_r.status_code == 404


@pytest.mark.anyio
async def test_test_webhook_returns_status(ac, deployment_id):
    """Test endpoint dispatches synchronously and returns status code."""
    r = await ac.post(
        f"/api/deploy/{deployment_id}/webhooks",
        json={"url": "https://httpbin.org/post"},
    )
    wh_id = r.json()["id"]

    # Mock _do_dispatch to avoid real HTTP
    with mock.patch("core.webhook._do_dispatch", return_value=200) as mock_dispatch:
        test_r = await ac.post(f"/api/deploy/{deployment_id}/webhooks/{wh_id}/test")
    assert test_r.status_code == 200
    data = test_r.json()
    assert data["status_code"] == 200
    assert data["success"] is True
    mock_dispatch.assert_called_once()


@pytest.mark.anyio
async def test_test_webhook_failure_not_error(ac, deployment_id):
    """Test returns 200 even when the target URL fails (status 0)."""
    r = await ac.post(
        f"/api/deploy/{deployment_id}/webhooks",
        json={"url": "https://example.com/hook"},
    )
    wh_id = r.json()["id"]

    with mock.patch("core.webhook._do_dispatch", return_value=0):
        test_r = await ac.post(f"/api/deploy/{deployment_id}/webhooks/{wh_id}/test")
    data = test_r.json()
    assert data["success"] is False
    assert data["status_code"] == 0


# ---------------------------------------------------------------------------
# dispatch_webhooks unit tests
# ---------------------------------------------------------------------------


def test_dispatch_webhooks_fires_matching_events(tmp_path):
    """dispatch_webhooks calls _do_dispatch only for matching event_types."""
    import models.webhook_config  # noqa — register table

    test_db = str(tmp_path / "dispatch_test.db")
    engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.engine = engine
    SQLModel.metadata.create_all(engine)

    from models.webhook_config import WebhookConfig

    with Session(engine) as session:
        wh = WebhookConfig(
            deployment_id="dep1",
            url="https://example.com/hook",
            event_types='["batch_complete"]',
        )
        session.add(wh)
        session.commit()

    from core.webhook import dispatch_webhooks

    with mock.patch("core.webhook._dispatch_in_thread") as mock_thread:
        dispatch_webhooks("dep1", "batch_complete", {"rows": 100})

    # Thread should have been started for the matching webhook
    mock_thread.assert_called_once()


def test_dispatch_webhooks_skips_non_matching_events(tmp_path):
    """dispatch_webhooks skips webhooks not subscribed to the fired event."""
    import models.webhook_config  # noqa

    test_db = str(tmp_path / "dispatch_skip_test.db")
    engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.engine = engine
    SQLModel.metadata.create_all(engine)

    from models.webhook_config import WebhookConfig

    with Session(engine) as session:
        wh = WebhookConfig(
            deployment_id="dep2",
            url="https://example.com/hook",
            event_types='["drift_detected"]',  # only drift
        )
        session.add(wh)
        session.commit()

    from core.webhook import dispatch_webhooks

    with mock.patch("core.webhook._dispatch_in_thread") as mock_thread:
        dispatch_webhooks("dep2", "batch_complete", {})  # different event

    mock_thread.assert_not_called()


def test_dispatch_webhooks_invalid_event_is_noop():
    """dispatch_webhooks silently ignores unknown event types."""
    from core.webhook import dispatch_webhooks

    with mock.patch("core.webhook._dispatch_in_thread") as mock_thread:
        dispatch_webhooks("dep3", "unknown_event", {})

    mock_thread.assert_not_called()


def test_dispatch_webhooks_handles_db_error_gracefully():
    """dispatch_webhooks never raises even when the DB is unavailable."""
    from core.webhook import dispatch_webhooks

    # Temporarily corrupt the engine
    original_engine = db_module.engine
    db_module.engine = None  # type: ignore
    try:
        dispatch_webhooks("dep4", "batch_complete", {})  # must not raise
    finally:
        db_module.engine = original_engine


def test_all_events_constant_has_three_entries():
    from core.webhook import (
        ALL_EVENTS,
        EVENT_BATCH_COMPLETE,
        EVENT_DRIFT_DETECTED,
        EVENT_HEALTH_DEGRADED,
    )

    assert EVENT_BATCH_COMPLETE in ALL_EVENTS
    assert EVENT_DRIFT_DETECTED in ALL_EVENTS
    assert EVENT_HEALTH_DEGRADED in ALL_EVENTS
    assert len(ALL_EVENTS) == 3
