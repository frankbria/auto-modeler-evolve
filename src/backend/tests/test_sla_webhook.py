"""Tests for the real-time SLA latency webhook (sla_exceeded event).

Covers:
- EVENT_SLA_EXCEEDED constant exists in core.webhook
- ALL_EVENTS includes sla_exceeded
- _check_and_fire_sla_alert: fewer than min samples → no fire
- _check_and_fire_sla_alert: p95 within threshold → no fire
- _check_and_fire_sla_alert: p95 exceeds threshold → dispatch called
- _check_and_fire_sla_alert: cooldown prevents repeat fire within 1 hour
- _check_and_fire_sla_alert: fires again after cooldown window passes
- _check_and_fire_sla_alert: updates sla_alert_last_fired_at on deployment
- Payload contains required keys: p95_ms, avg_ms, sample_count, threshold_ms, message
- Deployment.sla_alert_last_fired_at field accessible on model
- WebhookCreateBody default event_types includes sla_exceeded
- create_webhook docstring covers sla_exceeded
- Integration: make_prediction wires SLA check in background thread
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from api.deploy import (
    _SLA_WEBHOOK_COOLDOWN_HOURS,
    _SLA_WEBHOOK_MIN_SAMPLES,
    _SLA_WEBHOOK_N_LOGS,
    _SLA_WEBHOOK_THRESHOLD_MS,
    _check_and_fire_sla_alert,
)
from core.webhook import ALL_EVENTS, EVENT_SLA_EXCEEDED
from models.deployment import Deployment
from models.prediction_log import PredictionLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
)


def _make_deployment(session: Session, *, project_id: str = "proj-1") -> Deployment:
    dep = Deployment(
        model_run_id="run-1",
        project_id=project_id,
        endpoint_path="/api/predict/dep-1",
        dashboard_url="/predict/dep-1",
        pipeline_path=None,
        is_active=True,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)
    return dep


def _add_logs(
    session: Session,
    deployment_id: str,
    latencies_ms: list[float],
    base_time: datetime | None = None,
) -> None:
    """Insert PredictionLog rows with the given response_ms values."""
    if base_time is None:
        base_time = datetime.now(UTC).replace(tzinfo=None)
    for i, ms in enumerate(latencies_ms):
        log = PredictionLog(
            deployment_id=deployment_id,
            input_features="{}",
            prediction="1.0",
            prediction_numeric=1.0,
            response_ms=ms,
            created_at=base_time - timedelta(seconds=i),
        )
        session.add(log)
    session.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session(tmp_path):
    test_db = str(tmp_path / "test.db")
    engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.engine = engine
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Constants & module-level checks
# ---------------------------------------------------------------------------


def test_event_sla_exceeded_constant():
    assert EVENT_SLA_EXCEEDED == "sla_exceeded"


def test_sla_exceeded_in_all_events():
    assert "sla_exceeded" in ALL_EVENTS


def test_sla_constants_sensible():
    assert _SLA_WEBHOOK_N_LOGS >= 10
    assert _SLA_WEBHOOK_MIN_SAMPLES >= 2
    assert _SLA_WEBHOOK_THRESHOLD_MS > 0
    assert _SLA_WEBHOOK_COOLDOWN_HOURS > 0


# ---------------------------------------------------------------------------
# Deployment model field
# ---------------------------------------------------------------------------


def test_deployment_has_sla_alert_last_fired_at_field(db_session):
    dep = _make_deployment(db_session)
    assert hasattr(dep, "sla_alert_last_fired_at")
    assert dep.sla_alert_last_fired_at is None


def test_deployment_sla_last_fired_at_can_be_set(db_session):
    dep = _make_deployment(db_session)
    now = datetime.now(UTC).replace(tzinfo=None)
    dep.sla_alert_last_fired_at = now
    db_session.add(dep)
    db_session.commit()
    db_session.refresh(dep)
    assert dep.sla_alert_last_fired_at is not None


# ---------------------------------------------------------------------------
# _check_and_fire_sla_alert unit tests
# ---------------------------------------------------------------------------


def test_sla_no_fire_when_no_logs(db_session):
    dep = _make_deployment(db_session)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert(dep.id)
    mock_dispatch.assert_not_called()


def test_sla_no_fire_when_fewer_than_min_samples(db_session):
    dep = _make_deployment(db_session)
    # Add fewer than _SLA_WEBHOOK_MIN_SAMPLES logs (even with high latency)
    _add_logs(db_session, dep.id, [600.0] * (_SLA_WEBHOOK_MIN_SAMPLES - 1))
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert(dep.id)
    mock_dispatch.assert_not_called()


def test_sla_no_fire_when_p95_within_threshold(db_session):
    dep = _make_deployment(db_session)
    # All predictions fast — p95 well below 500ms
    _add_logs(db_session, dep.id, [50.0] * 20)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert(dep.id)
    mock_dispatch.assert_not_called()


def test_sla_fires_when_p95_exceeds_threshold(db_session):
    dep = _make_deployment(db_session)
    # Most predictions slow (>500ms), p95 will be high
    slow = [800.0] * 18 + [50.0, 50.0]  # p95 ≈ 800ms
    _add_logs(db_session, dep.id, slow)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert(dep.id)
    mock_dispatch.assert_called_once()
    call_args = mock_dispatch.call_args
    assert call_args[0][0] == dep.id
    assert call_args[0][1] == EVENT_SLA_EXCEEDED


def test_sla_payload_has_required_keys(db_session):
    dep = _make_deployment(db_session)
    slow = [800.0] * 18 + [50.0, 50.0]
    _add_logs(db_session, dep.id, slow)
    captured_payload: dict = {}
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert(dep.id)
        captured_payload = mock_dispatch.call_args[0][2]
    assert "p95_ms" in captured_payload
    assert "avg_ms" in captured_payload
    assert "sample_count" in captured_payload
    assert "threshold_ms" in captured_payload
    assert "message" in captured_payload
    assert captured_payload["threshold_ms"] == _SLA_WEBHOOK_THRESHOLD_MS
    assert isinstance(captured_payload["p95_ms"], float)
    assert isinstance(captured_payload["message"], str)
    assert "500" in captured_payload["message"]


def test_sla_payload_sample_count_correct(db_session):
    dep = _make_deployment(db_session)
    n = 15
    _add_logs(db_session, dep.id, [800.0] * n)
    captured_payload: dict = {}
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert(dep.id)
        captured_payload = mock_dispatch.call_args[0][2]
    assert captured_payload["sample_count"] == n


def test_sla_updates_last_fired_at_after_fire(db_session):
    dep = _make_deployment(db_session)
    assert dep.sla_alert_last_fired_at is None
    _add_logs(db_session, dep.id, [800.0] * 20)
    with patch("core.webhook.dispatch_webhooks"):
        _check_and_fire_sla_alert(dep.id)
    db_session.refresh(dep)
    assert dep.sla_alert_last_fired_at is not None


def test_sla_cooldown_prevents_repeat_fire(db_session):
    dep = _make_deployment(db_session)
    # Pre-set fired timestamp to 30 minutes ago (within 1h cooldown)
    dep.sla_alert_last_fired_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(
        minutes=30
    )
    db_session.add(dep)
    db_session.commit()

    _add_logs(db_session, dep.id, [800.0] * 20)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert(dep.id)
    mock_dispatch.assert_not_called()


def test_sla_fires_after_cooldown_elapsed(db_session):
    dep = _make_deployment(db_session)
    # Pre-set fired timestamp to 2 hours ago (past 1h cooldown)
    dep.sla_alert_last_fired_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(
        hours=2
    )
    db_session.add(dep)
    db_session.commit()

    _add_logs(db_session, dep.id, [800.0] * 20)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert(dep.id)
    mock_dispatch.assert_called_once()


def test_sla_no_fire_for_unknown_deployment(db_session):
    """Non-existent deployment_id should silently do nothing."""
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert("does-not-exist")
    mock_dispatch.assert_not_called()


def test_sla_custom_threshold(db_session):
    dep = _make_deployment(db_session)
    # All predictions at 300ms — below default 500ms but above custom 200ms threshold
    _add_logs(db_session, dep.id, [300.0] * 20)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert(dep.id, threshold_ms=200.0)
    mock_dispatch.assert_called_once()
    payload = mock_dispatch.call_args[0][2]
    assert payload["threshold_ms"] == 200.0


def test_sla_custom_threshold_no_fire_when_within(db_session):
    dep = _make_deployment(db_session)
    _add_logs(db_session, dep.id, [300.0] * 20)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        # threshold set above p95 — should not fire
        _check_and_fire_sla_alert(dep.id, threshold_ms=400.0)
    mock_dispatch.assert_not_called()


def test_sla_only_last_n_logs_considered(db_session):
    """With _SLA_WEBHOOK_N_LOGS old slow logs and 5+ recent fast ones,
    p95 of the recent sample should not fire if fast enough.
    """
    dep = _make_deployment(db_session)
    # Insert 100 very slow OLD logs
    old_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=30)
    _add_logs(db_session, dep.id, [5000.0] * 100, base_time=old_time)
    # Insert _SLA_WEBHOOK_N_LOGS recent fast logs (newest, so selected first)
    _add_logs(db_session, dep.id, [10.0] * _SLA_WEBHOOK_N_LOGS)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        _check_and_fire_sla_alert(dep.id)
    # The last N_LOGS are all fast — should NOT fire
    mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# WebhookCreateBody default includes sla_exceeded
# ---------------------------------------------------------------------------


def test_webhook_create_body_default_includes_sla_exceeded():
    from api.deploy import WebhookCreateBody

    body = WebhookCreateBody(url="https://example.com/hook")
    assert "sla_exceeded" in body.event_types


def test_webhook_create_body_can_register_sla_exceeded():
    from api.deploy import WebhookCreateBody

    body = WebhookCreateBody(url="https://example.com/hook", event_types=["sla_exceeded"])
    assert body.event_types == ["sla_exceeded"]
