"""Tests for the Prediction Low-Activity Alert feature.

Covers:
- EVENT_LOW_ACTIVITY constant exists in core.webhook
- ALL_EVENTS includes low_activity
- Deployment.low_activity_threshold_per_day / low_activity_alert_last_fired_at fields
- _check_and_fire_low_activity_alert: skips when threshold is None (disabled)
- _check_and_fire_low_activity_alert: no alert when daily count >= threshold
- _check_and_fire_low_activity_alert: fires when daily count < threshold
- _check_and_fire_low_activity_alert: cooldown prevents repeat within 24h
- _check_and_fire_low_activity_alert: fires again after cooldown expires
- _check_and_fire_low_activity_alert: payload contains required keys
- PUT /api/deploy/{id}/low-activity-alert — enable with threshold
- PUT /api/deploy/{id}/low-activity-alert — disable (threshold=null)
- PUT /api/deploy/{id}/low-activity-alert — rejects threshold < 1
- PUT /api/deploy/{id}/low-activity-alert — 404 for missing deployment
- GET /api/deploy/{id}/low-activity-alert-status — returns config
- GET /api/deploy/{id}/low-activity-alert-status — 404 for missing deployment
- _LOW_ACTIVITY_ALERT_PATTERNS — NL enable/disable/status phrases
- _DISABLE_LOW_ACTIVITY_ALERT_RE — matches disable phrases
- _STATUS_LOW_ACTIVITY_ALERT_RE — matches status phrases
- _LOW_ACTIVITY_THRESHOLD_RE — extracts numeric thresholds from messages
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from core.webhook import ALL_EVENTS, EVENT_LOW_ACTIVITY

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

    proj_r = await ac.post("/api/projects", json={"name": "LowActTest"})
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


def test_event_low_activity_constant_exists():
    assert EVENT_LOW_ACTIVITY == "low_activity"


def test_all_events_includes_low_activity():
    assert "low_activity" in ALL_EVENTS


# ---------------------------------------------------------------------------
# Deployment model fields
# ---------------------------------------------------------------------------


def test_deployment_low_activity_fields_exist():
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/x",
        dashboard_url="/predict/x",
    )
    assert hasattr(dep, "low_activity_threshold_per_day")
    assert hasattr(dep, "low_activity_alert_last_fired_at")
    assert dep.low_activity_threshold_per_day is None
    assert dep.low_activity_alert_last_fired_at is None


# ---------------------------------------------------------------------------
# _check_and_fire_low_activity_alert unit tests
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
        low_activity_threshold_per_day=threshold,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)
    return dep


def test_check_low_activity_skips_when_disabled(db_session):
    dep = _make_dep_obj(db_session, threshold=None)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_low_activity_alert

        _check_and_fire_low_activity_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_low_activity_no_alert_when_count_sufficient(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold=3)
    # Add 5 predictions today
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(5):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            created_at=now,
        )
        db_session.add(log)
    db_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_low_activity_alert

        _check_and_fire_low_activity_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_low_activity_fires_when_count_below_threshold(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, threshold=10)
    # Only 2 predictions today
    now = datetime.now(UTC).replace(tzinfo=None)
    for _ in range(2):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="1.0",
            created_at=now,
        )
        db_session.add(log)
    db_session.commit()

    dispatched = []

    def capture_dispatch(dep_id, event, payload):
        dispatched.append((dep_id, event, payload))

    with patch("core.webhook.dispatch_webhooks", side_effect=capture_dispatch):
        from api.deploy import _check_and_fire_low_activity_alert

        _check_and_fire_low_activity_alert(dep.id)

    assert len(dispatched) == 1
    dep_id, event, payload = dispatched[0]
    assert event == "low_activity"
    assert payload["daily_prediction_count"] == 2
    assert payload["threshold_per_day"] == 10


def test_check_low_activity_cooldown_prevents_repeat(db_session):

    dep = _make_dep_obj(db_session, threshold=10)
    # Set last fired 1 hour ago (within cooldown)
    recent = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    dep.low_activity_alert_last_fired_at = recent
    db_session.add(dep)
    db_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_low_activity_alert

        _check_and_fire_low_activity_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_low_activity_fires_after_cooldown_expires(db_session):

    dep = _make_dep_obj(db_session, threshold=10)
    # Set last fired 25 hours ago (outside cooldown)
    old = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=25)
    dep.low_activity_alert_last_fired_at = old
    db_session.add(dep)
    db_session.commit()

    # 0 predictions today
    dispatched = []

    def capture_dispatch(dep_id, event, payload):
        dispatched.append(event)

    with patch("core.webhook.dispatch_webhooks", side_effect=capture_dispatch):
        from api.deploy import _check_and_fire_low_activity_alert

        _check_and_fire_low_activity_alert(dep.id)

    assert "low_activity" in dispatched


def test_check_low_activity_payload_contains_required_keys(db_session):
    dep = _make_dep_obj(db_session, threshold=5)

    payloads = []

    def capture_dispatch(dep_id, event, payload):
        payloads.append(payload)

    with patch("core.webhook.dispatch_webhooks", side_effect=capture_dispatch):
        from api.deploy import _check_and_fire_low_activity_alert

        _check_and_fire_low_activity_alert(dep.id)

    assert len(payloads) == 1
    payload = payloads[0]
    for key in (
        "deployment_id",
        "daily_prediction_count",
        "threshold_per_day",
        "message",
    ):
        assert key in payload, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_low_activity_alert_enable(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/low-activity-alert",
        json={"threshold_per_day": 10},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["low_activity_alert_enabled"] is True
    assert body["threshold_per_day"] == 10


@pytest.mark.anyio
async def test_put_low_activity_alert_disable(ac):
    dep_id = await _make_deployment(ac)
    # Enable first
    await ac.put(
        f"/api/deploy/{dep_id}/low-activity-alert", json={"threshold_per_day": 10}
    )
    # Now disable
    r = await ac.put(
        f"/api/deploy/{dep_id}/low-activity-alert",
        json={"threshold_per_day": None},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["low_activity_alert_enabled"] is False
    assert body["threshold_per_day"] is None


@pytest.mark.anyio
async def test_put_low_activity_alert_rejects_zero_threshold(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/low-activity-alert",
        json={"threshold_per_day": 0},
    )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_put_low_activity_alert_404(ac):
    r = await ac.put(
        "/api/deploy/nonexistent/low-activity-alert",
        json={"threshold_per_day": 5},
    )
    assert r.status_code == 404


@pytest.mark.anyio
async def test_get_low_activity_alert_status(ac):
    dep_id = await _make_deployment(ac)
    await ac.put(
        f"/api/deploy/{dep_id}/low-activity-alert", json={"threshold_per_day": 20}
    )
    r = await ac.get(f"/api/deploy/{dep_id}/low-activity-alert-status")
    assert r.status_code == 200
    body = r.json()
    assert body["low_activity_alert_enabled"] is True
    assert body["threshold_per_day"] == 20
    assert "cooldown_hours" in body
    assert "summary" in body


@pytest.mark.anyio
async def test_get_low_activity_alert_status_404(ac):
    r = await ac.get("/api/deploy/nonexistent/low-activity-alert-status")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Chat patterns
# ---------------------------------------------------------------------------


def test_low_activity_alert_patterns_enable():
    from api.chat import _LOW_ACTIVITY_ALERT_PATTERNS

    matches = [
        "alert me if fewer than 10 predictions per day",
        "set up a low-activity alert",
        "configure a minimum prediction count alert",
        "notify me when my endpoint goes quiet",
        "alert me when daily predictions drop below 5",
        "add a low activity alert",
        "enable low-activity alert",
        "alert me when my model stops receiving enough calls",
    ]
    for msg in matches:
        assert _LOW_ACTIVITY_ALERT_PATTERNS.search(msg), f"Should match: {msg!r}"


def test_low_activity_alert_patterns_no_false_positives():
    from api.chat import _LOW_ACTIVITY_ALERT_PATTERNS

    non_matches = [
        "show me my predictions",
        "how many predictions today",
        "what is my accuracy",
    ]
    for msg in non_matches:
        assert not _LOW_ACTIVITY_ALERT_PATTERNS.search(msg), (
            f"Should NOT match: {msg!r}"
        )


def test_disable_low_activity_alert_re():
    from api.chat import _DISABLE_LOW_ACTIVITY_ALERT_RE

    matches = [
        "disable the low-activity alert",
        "turn off low activity alert",
        "remove low-activity alert",
        "clear low activity alert",
    ]
    for msg in matches:
        assert _DISABLE_LOW_ACTIVITY_ALERT_RE.search(msg), f"Should match: {msg!r}"


def test_status_low_activity_alert_re():
    from api.chat import _STATUS_LOW_ACTIVITY_ALERT_RE

    matches = [
        "show my low-activity alert",
        "check low activity alert",
        "get low activity setting",
        "status of low activity threshold",
    ]
    for msg in matches:
        assert _STATUS_LOW_ACTIVITY_ALERT_RE.search(msg), f"Should match: {msg!r}"


def test_low_activity_threshold_re_extracts_number():
    from api.chat import _LOW_ACTIVITY_THRESHOLD_RE

    cases = [
        ("alert me if fewer than 10 predictions per day", "10"),
        ("set minimum to 50 per day", "50"),
        ("fewer than 5 calls daily", "5"),
        ("below 100 requests a day", "100"),
    ]
    for msg, expected in cases:
        m = _LOW_ACTIVITY_THRESHOLD_RE.search(msg)
        assert m is not None, f"Should match: {msg!r}"
        assert m.group(1) == expected, f"Expected {expected!r}, got {m.group(1)!r}"
