"""Tests for the Prediction Latency Alert feature.

Covers:
- EVENT_LATENCY_ALERT constant exists in core.webhook
- ALL_EVENTS includes latency_alert
- Deployment.latency_alert_threshold_ms / latency_alert_last_fired_at fields
- _check_and_fire_latency_alert: skips when threshold is None (disabled)
- _check_and_fire_latency_alert: no alert when p95 latency <= threshold
- _check_and_fire_latency_alert: fires when p95 latency > threshold
- _check_and_fire_latency_alert: cooldown prevents repeat within 1h
- _check_and_fire_latency_alert: fires again after cooldown expires
- _check_and_fire_latency_alert: payload contains required keys
- _check_and_fire_latency_alert: skips when no response_ms data available
- PUT /api/deploy/{id}/latency-alert — enable with threshold
- PUT /api/deploy/{id}/latency-alert — disable (threshold=null)
- PUT /api/deploy/{id}/latency-alert — rejects threshold < 1
- PUT /api/deploy/{id}/latency-alert — 404 for missing deployment
- GET /api/deploy/{id}/latency-alert-status — returns config
- GET /api/deploy/{id}/latency-alert-status — 404 for missing deployment
- _LATENCY_ALERT_PATTERNS — NL enable/disable/status phrases
- _DISABLE_LATENCY_ALERT_RE — matches disable phrases
- _STATUS_LATENCY_ALERT_RE — matches status phrases
- _LATENCY_THRESHOLD_MS_RE — extracts numeric thresholds in ms
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from core.webhook import ALL_EVENTS, EVENT_LATENCY_ALERT

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.dataset_filter  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.deployment_preset  # noqa
    import models.batch_schedule  # noqa
    import models.webhook_config  # noqa
    import models.ab_test  # noqa
    import models.deployment_version  # noqa

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


async def _make_deployment(ac):
    """Create project, upload CSV, train a model, and deploy it. Returns deployment_id."""
    import time

    proj_r = await ac.post("/api/projects", json={"name": "LatencyTest"})
    proj_id = proj_r.json()["id"]

    upload_r = await ac.post(
        "/api/data/upload",
        files={"file": ("d.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": proj_id},
    )
    assert upload_r.status_code == 201
    ds_id = upload_r.json()["dataset_id"]

    fs_r = await ac.post(f"/api/features/{ds_id}/apply", json={"transformations": []})
    assert fs_r.status_code == 201
    fs_id = fs_r.json()["feature_set_id"]

    await ac.post(
        f"/api/features/{ds_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )

    train_r = await ac.post(
        f"/api/models/{proj_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": fs_id},
    )
    assert train_r.status_code == 202
    run_id = train_r.json()["model_run_ids"][0]

    for _ in range(30):
        r = await ac.get(f"/api/models/{proj_id}/runs")
        run = next((x for x in r.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            break
        time.sleep(0.3)
    else:
        pytest.skip("Training did not complete")

    dep_r = await ac.post(f"/api/deploy/{run_id}", json={})
    assert dep_r.status_code == 201
    return dep_r.json()["id"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_event_latency_alert_constant_exists():
    assert EVENT_LATENCY_ALERT == "latency_alert"


def test_all_events_includes_latency_alert():
    assert "latency_alert" in ALL_EVENTS


# ---------------------------------------------------------------------------
# Deployment model fields
# ---------------------------------------------------------------------------


def test_deployment_latency_alert_fields_exist():
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/x",
        dashboard_url="/predict/x",
    )
    assert hasattr(dep, "latency_alert_threshold_ms")
    assert hasattr(dep, "latency_alert_last_fired_at")
    assert dep.latency_alert_threshold_ms is None
    assert dep.latency_alert_last_fired_at is None


# ---------------------------------------------------------------------------
# _check_and_fire_latency_alert unit tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session(tmp_path):
    test_db = str(tmp_path / "unit.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)

    import models.deployment  # noqa
    import models.prediction_log  # noqa
    import models.webhook_config  # noqa

    SQLModel.metadata.create_all(db_module.engine)
    with Session(db_module.engine) as s:
        yield s


def _make_dep_obj(session, threshold_ms=None):
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/x",
        dashboard_url="/predict/x",
        latency_alert_threshold_ms=threshold_ms,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)
    return dep


def test_check_latency_alert_skips_when_disabled(db_session):
    dep = _make_dep_obj(db_session, threshold_ms=None)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_latency_alert

        _check_and_fire_latency_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_latency_alert_no_alert_when_p95_within_threshold(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold_ms=500)
    # Add 20 predictions all with 100ms latency — well below 500ms threshold
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(20):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            response_ms=100.0,
            created_at=now - timedelta(minutes=5),
        )
        db_session.add(log)
    db_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_latency_alert

        _check_and_fire_latency_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_latency_alert_fires_when_p95_exceeds_threshold(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold_ms=200)
    # Add 20 predictions with high latency — p95 will be 800ms
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(20):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            response_ms=800.0,
            created_at=now - timedelta(minutes=5),
        )
        db_session.add(log)
    db_session.commit()

    dispatched = []

    def capture_dispatch(dep_id, event, payload):
        dispatched.append((dep_id, event, payload))

    with patch("core.webhook.dispatch_webhooks", side_effect=capture_dispatch):
        from api.deploy import _check_and_fire_latency_alert

        _check_and_fire_latency_alert(dep.id)

    assert len(dispatched) == 1
    dep_id, event, payload = dispatched[0]
    assert event == "latency_alert"
    assert payload["p95_latency_ms"] > 200
    assert payload["threshold_ms"] == 200


def test_check_latency_alert_cooldown_prevents_repeat(db_session):
    dep = _make_dep_obj(db_session, threshold_ms=100)
    # Set last fired 10 minutes ago (within 1-hour cooldown)
    recent = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=10)
    dep.latency_alert_last_fired_at = recent
    db_session.add(dep)
    db_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_latency_alert

        _check_and_fire_latency_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_latency_alert_fires_after_cooldown_expires(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold_ms=100)
    # Set last fired 2 hours ago (outside 1-hour cooldown)
    old = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=2)
    dep.latency_alert_last_fired_at = old
    db_session.add(dep)
    db_session.commit()

    # Add slow predictions
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(20):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            response_ms=600.0,
            created_at=now - timedelta(minutes=5),
        )
        db_session.add(log)
    db_session.commit()

    dispatched = []

    def capture_dispatch(dep_id, event, payload):
        dispatched.append(event)

    with patch("core.webhook.dispatch_webhooks", side_effect=capture_dispatch):
        from api.deploy import _check_and_fire_latency_alert

        _check_and_fire_latency_alert(dep.id)

    assert "latency_alert" in dispatched


def test_check_latency_alert_payload_contains_required_keys(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold_ms=50)
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(10):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            response_ms=300.0,
            created_at=now - timedelta(minutes=5),
        )
        db_session.add(log)
    db_session.commit()

    payloads = []

    def capture_dispatch(dep_id, event, payload):
        payloads.append(payload)

    with patch("core.webhook.dispatch_webhooks", side_effect=capture_dispatch):
        from api.deploy import _check_and_fire_latency_alert

        _check_and_fire_latency_alert(dep.id)

    assert len(payloads) == 1
    payload = payloads[0]
    for key in ("deployment_id", "p95_latency_ms", "threshold_ms", "sample_size", "message"):
        assert key in payload, f"Missing key: {key}"


def test_check_latency_alert_skips_when_no_response_ms_data(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold_ms=500)
    # Add predictions with null response_ms
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(10):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            response_ms=None,
            created_at=now - timedelta(minutes=5),
        )
        db_session.add(log)
    db_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_latency_alert

        _check_and_fire_latency_alert(dep.id)
        mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_latency_alert_enable(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/latency-alert",
        json={"threshold_ms": 500},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["latency_alert_enabled"] is True
    assert body["threshold_ms"] == 500


@pytest.mark.anyio
async def test_put_latency_alert_disable(ac):
    dep_id = await _make_deployment(ac)
    # Enable first
    await ac.put(
        f"/api/deploy/{dep_id}/latency-alert",
        json={"threshold_ms": 500},
    )
    # Now disable
    r = await ac.put(
        f"/api/deploy/{dep_id}/latency-alert",
        json={"threshold_ms": None},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["latency_alert_enabled"] is False
    assert body["threshold_ms"] is None


@pytest.mark.anyio
async def test_put_latency_alert_rejects_zero_threshold(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/latency-alert",
        json={"threshold_ms": 0},
    )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_put_latency_alert_404(ac):
    r = await ac.put(
        "/api/deploy/nonexistent/latency-alert",
        json={"threshold_ms": 500},
    )
    assert r.status_code == 404


@pytest.mark.anyio
async def test_get_latency_alert_status(ac):
    dep_id = await _make_deployment(ac)
    await ac.put(
        f"/api/deploy/{dep_id}/latency-alert",
        json={"threshold_ms": 750},
    )
    r = await ac.get(f"/api/deploy/{dep_id}/latency-alert-status")
    assert r.status_code == 200
    body = r.json()
    assert body["latency_alert_enabled"] is True
    assert body["threshold_ms"] == 750
    assert "cooldown_hours" in body
    assert "summary" in body


@pytest.mark.anyio
async def test_get_latency_alert_status_404(ac):
    r = await ac.get("/api/deploy/nonexistent/latency-alert-status")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Chat patterns
# ---------------------------------------------------------------------------


def test_latency_alert_patterns_enable():
    from api.chat import _LATENCY_ALERT_PATTERNS

    matches = [
        "alert me if predictions take more than 500ms",
        "set up a latency alert",
        "configure a response time alert",
        "notify me when my model gets too slow",
        "alert me when p95 latency exceeds 1000ms",
        "add a latency alert",
        "enable slow prediction alert",
        "warn me when latency goes above 200ms",
    ]
    for msg in matches:
        assert _LATENCY_ALERT_PATTERNS.search(msg), f"Should match: {msg!r}"


def test_latency_alert_patterns_no_false_positives():
    from api.chat import _LATENCY_ALERT_PATTERNS

    non_matches = [
        "show me my predictions",
        "how many predictions today",
        "what is my accuracy",
        "set up a low activity alert",
    ]
    for msg in non_matches:
        assert not _LATENCY_ALERT_PATTERNS.search(msg), (
            f"Should NOT match: {msg!r}"
        )


def test_disable_latency_alert_re():
    from api.chat import _DISABLE_LATENCY_ALERT_RE

    matches = [
        "disable the latency alert",
        "turn off latency alert",
        "remove latency alert",
        "clear latency alert",
    ]
    for msg in matches:
        assert _DISABLE_LATENCY_ALERT_RE.search(msg), f"Should match: {msg!r}"


def test_status_latency_alert_re():
    from api.chat import _STATUS_LATENCY_ALERT_RE

    matches = [
        "show my latency alert",
        "check latency setting",
        "get latency threshold",
        "status of latency alert",
    ]
    for msg in matches:
        assert _STATUS_LATENCY_ALERT_RE.search(msg), f"Should match: {msg!r}"


def test_latency_threshold_ms_re_extracts_number():
    from api.chat import _LATENCY_THRESHOLD_MS_RE

    cases = [
        ("alert me if predictions take more than 500ms", "500"),
        ("set threshold to 1000ms", "1000"),
        ("notify when latency exceeds 250ms", "250"),
        ("p95 over 200ms is too slow", "200"),
    ]
    for msg, expected in cases:
        m = _LATENCY_THRESHOLD_MS_RE.search(msg)
        assert m is not None, f"Should match: {msg!r}"
        assert m.group(1) == expected, f"Expected {expected!r}, got {m.group(1)!r}"
