"""Tests for the Prediction High-Activity Burst Alert feature.

Covers:
- EVENT_HIGH_ACTIVITY_BURST constant exists in core.webhook
- ALL_EVENTS includes high_activity_burst
- Deployment.high_activity_threshold_per_hour / high_activity_burst_last_fired_at fields
- _check_and_fire_high_activity_burst: skips when threshold is None (disabled)
- _check_and_fire_high_activity_burst: no alert when hourly count <= threshold
- _check_and_fire_high_activity_burst: fires when hourly count > threshold
- _check_and_fire_high_activity_burst: cooldown prevents repeat within 1h
- _check_and_fire_high_activity_burst: fires again after cooldown expires
- _check_and_fire_high_activity_burst: payload contains required keys
- _check_and_fire_high_activity_burst: only counts logs within last 60 minutes
- PUT /api/deploy/{id}/high-activity-burst-alert — enable with threshold
- PUT /api/deploy/{id}/high-activity-burst-alert — disable (threshold=null)
- PUT /api/deploy/{id}/high-activity-burst-alert — rejects threshold < 1
- PUT /api/deploy/{id}/high-activity-burst-alert — 404 for missing deployment
- GET /api/deploy/{id}/high-activity-burst-alert-status — returns config
- GET /api/deploy/{id}/high-activity-burst-alert-status — 404 for missing deployment
- _HIGH_ACTIVITY_BURST_PATTERNS — NL enable/disable/status phrases
- _DISABLE_HIGH_ACTIVITY_BURST_RE — matches disable phrases
- _STATUS_HIGH_ACTIVITY_BURST_RE — matches status phrases
- _HIGH_ACTIVITY_THRESHOLD_RE — extracts numeric thresholds from messages
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from core.webhook import ALL_EVENTS, EVENT_HIGH_ACTIVITY_BURST
from tests.conftest import wait_for_run

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
    proj_r = await ac.post("/api/projects", json={"name": "BurstTest"})
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

    await wait_for_run(ac, proj_id, run_id)

    dep_r = await ac.post(f"/api/deploy/{run_id}", json={})
    assert dep_r.status_code == 201
    return dep_r.json()["id"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_event_high_activity_burst_constant_exists():
    assert EVENT_HIGH_ACTIVITY_BURST == "high_activity_burst"


def test_all_events_includes_high_activity_burst():
    assert "high_activity_burst" in ALL_EVENTS


# ---------------------------------------------------------------------------
# Deployment model fields
# ---------------------------------------------------------------------------


def test_deployment_high_activity_burst_fields_exist():
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/x",
        dashboard_url="/predict/x",
    )
    assert hasattr(dep, "high_activity_threshold_per_hour")
    assert hasattr(dep, "high_activity_burst_last_fired_at")
    assert dep.high_activity_threshold_per_hour is None
    assert dep.high_activity_burst_last_fired_at is None


# ---------------------------------------------------------------------------
# _check_and_fire_high_activity_burst unit tests
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


def _make_dep_obj(session, threshold=None):
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/x",
        dashboard_url="/predict/x",
        high_activity_threshold_per_hour=threshold,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)
    return dep


def test_check_high_activity_burst_skips_when_disabled(db_session):
    dep = _make_dep_obj(db_session, threshold=None)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_high_activity_burst

        _check_and_fire_high_activity_burst(dep.id)
        mock_dispatch.assert_not_called()


def test_check_high_activity_burst_no_alert_when_count_within_threshold(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold=20)
    # Add 10 predictions within the last hour — below threshold
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(10):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            created_at=now - timedelta(minutes=30),
        )
        db_session.add(log)
    db_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_high_activity_burst

        _check_and_fire_high_activity_burst(dep.id)
        mock_dispatch.assert_not_called()


def test_check_high_activity_burst_fires_when_count_exceeds_threshold(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold=5)
    # Add 10 predictions within the last hour — above threshold
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(10):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            created_at=now - timedelta(minutes=15),
        )
        db_session.add(log)
    db_session.commit()

    dispatched = []

    def capture_dispatch(dep_id, event, payload):
        dispatched.append((dep_id, event, payload))

    with patch("core.webhook.dispatch_webhooks", side_effect=capture_dispatch):
        from api.deploy import _check_and_fire_high_activity_burst

        _check_and_fire_high_activity_burst(dep.id)

    assert len(dispatched) == 1
    dep_id, event, payload = dispatched[0]
    assert event == "high_activity_burst"
    assert payload["hourly_prediction_count"] == 10
    assert payload["threshold_per_hour"] == 5


def test_check_high_activity_burst_cooldown_prevents_repeat(db_session):
    dep = _make_dep_obj(db_session, threshold=5)
    # Set last fired 10 minutes ago (within 1-hour cooldown)
    recent = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=10)
    dep.high_activity_burst_last_fired_at = recent
    db_session.add(dep)
    db_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_high_activity_burst

        _check_and_fire_high_activity_burst(dep.id)
        mock_dispatch.assert_not_called()


def test_check_high_activity_burst_fires_after_cooldown_expires(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold=3)
    # Set last fired 2 hours ago (outside 1-hour cooldown)
    old = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=2)
    dep.high_activity_burst_last_fired_at = old
    db_session.add(dep)
    db_session.commit()

    # Add 10 predictions in the last hour
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(10):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            created_at=now - timedelta(minutes=5),
        )
        db_session.add(log)
    db_session.commit()

    dispatched = []

    def capture_dispatch(dep_id, event, payload):
        dispatched.append(event)

    with patch("core.webhook.dispatch_webhooks", side_effect=capture_dispatch):
        from api.deploy import _check_and_fire_high_activity_burst

        _check_and_fire_high_activity_burst(dep.id)

    assert "high_activity_burst" in dispatched


def test_check_high_activity_burst_payload_contains_required_keys(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold=2)
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(5):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            created_at=now - timedelta(minutes=10),
        )
        db_session.add(log)
    db_session.commit()

    payloads = []

    def capture_dispatch(dep_id, event, payload):
        payloads.append(payload)

    with patch("core.webhook.dispatch_webhooks", side_effect=capture_dispatch):
        from api.deploy import _check_and_fire_high_activity_burst

        _check_and_fire_high_activity_burst(dep.id)

    assert len(payloads) == 1
    payload = payloads[0]
    for key in (
        "deployment_id",
        "hourly_prediction_count",
        "threshold_per_hour",
        "message",
    ):
        assert key in payload, f"Missing key: {key}"


def test_check_high_activity_burst_ignores_old_predictions(db_session):
    """Predictions older than 60 minutes should not count toward the hourly window."""
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold=3)
    now = datetime.now(UTC).replace(tzinfo=None)
    # Add 10 predictions that are 90 minutes old (outside 60-minute window)
    for _ in range(10):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            created_at=now - timedelta(minutes=90),
        )
        db_session.add(log)
    db_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_high_activity_burst

        _check_and_fire_high_activity_burst(dep.id)
        mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_high_activity_burst_alert_enable(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/high-activity-burst-alert",
        json={"threshold_per_hour": 100},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["high_activity_burst_enabled"] is True
    assert body["threshold_per_hour"] == 100


@pytest.mark.anyio
async def test_put_high_activity_burst_alert_disable(ac):
    dep_id = await _make_deployment(ac)
    # Enable first
    await ac.put(
        f"/api/deploy/{dep_id}/high-activity-burst-alert",
        json={"threshold_per_hour": 100},
    )
    # Now disable
    r = await ac.put(
        f"/api/deploy/{dep_id}/high-activity-burst-alert",
        json={"threshold_per_hour": None},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["high_activity_burst_enabled"] is False
    assert body["threshold_per_hour"] is None


@pytest.mark.anyio
async def test_put_high_activity_burst_alert_rejects_zero_threshold(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/high-activity-burst-alert",
        json={"threshold_per_hour": 0},
    )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_put_high_activity_burst_alert_404(ac):
    r = await ac.put(
        "/api/deploy/nonexistent/high-activity-burst-alert",
        json={"threshold_per_hour": 50},
    )
    assert r.status_code == 404


@pytest.mark.anyio
async def test_get_high_activity_burst_alert_status(ac):
    dep_id = await _make_deployment(ac)
    await ac.put(
        f"/api/deploy/{dep_id}/high-activity-burst-alert",
        json={"threshold_per_hour": 200},
    )
    r = await ac.get(f"/api/deploy/{dep_id}/high-activity-burst-alert-status")
    assert r.status_code == 200
    body = r.json()
    assert body["high_activity_burst_enabled"] is True
    assert body["threshold_per_hour"] == 200
    assert "cooldown_hours" in body
    assert "summary" in body


@pytest.mark.anyio
async def test_get_high_activity_burst_alert_status_404(ac):
    r = await ac.get("/api/deploy/nonexistent/high-activity-burst-alert-status")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Chat patterns
# ---------------------------------------------------------------------------


def test_high_activity_burst_patterns_enable():
    from api.chat import _HIGH_ACTIVITY_BURST_PATTERNS

    matches = [
        "alert me when my model gets more than 100 predictions per hour",
        "set up a high-activity alert",
        "configure a traffic burst alert",
        "notify me when my endpoint gets too many calls",
        "alert me when hourly predictions exceed 500",
        "add a burst alert",
        "enable high activity alert",
        "alert me when my model gets flooded",
    ]
    for msg in matches:
        assert _HIGH_ACTIVITY_BURST_PATTERNS.search(msg), f"Should match: {msg!r}"


def test_high_activity_burst_patterns_no_false_positives():
    from api.chat import _HIGH_ACTIVITY_BURST_PATTERNS

    non_matches = [
        "show me my predictions",
        "how many predictions today",
        "what is my accuracy",
    ]
    for msg in non_matches:
        assert not _HIGH_ACTIVITY_BURST_PATTERNS.search(msg), (
            f"Should NOT match: {msg!r}"
        )


def test_disable_high_activity_burst_re():
    from api.chat import _DISABLE_HIGH_ACTIVITY_BURST_RE

    matches = [
        "disable the burst alert",
        "turn off high-activity alert",
        "remove burst alert",
        "clear high activity alert",
    ]
    for msg in matches:
        assert _DISABLE_HIGH_ACTIVITY_BURST_RE.search(msg), f"Should match: {msg!r}"


def test_status_high_activity_burst_re():
    from api.chat import _STATUS_HIGH_ACTIVITY_BURST_RE

    matches = [
        "show my burst alert",
        "check high activity alert",
        "get burst setting",
        "status of burst alert",
    ]
    for msg in matches:
        assert _STATUS_HIGH_ACTIVITY_BURST_RE.search(msg), f"Should match: {msg!r}"


def test_high_activity_threshold_re_extracts_number():
    from api.chat import _HIGH_ACTIVITY_THRESHOLD_RE

    cases = [
        ("alert me if more than 100 predictions per hour", "100"),
        ("set maximum to 500 per hour", "500"),
        ("more than 50 calls hourly", "50"),
        ("above 200 requests an hour", "200"),
    ]
    for msg, expected in cases:
        m = _HIGH_ACTIVITY_THRESHOLD_RE.search(msg)
        assert m is not None, f"Should match: {msg!r}"
        assert m.group(1) == expected, f"Expected {expected!r}, got {m.group(1)!r}"
