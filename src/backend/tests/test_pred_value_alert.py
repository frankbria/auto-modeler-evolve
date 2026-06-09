"""Tests for the Prediction Value Trend Alert feature.

Covers:
- EVENT_PRED_VALUE_TREND_ALERT constant exists in core.webhook
- ALL_EVENTS includes pred_value_trend_alert
- Deployment fields: pred_value_alert_enabled, pred_value_alert_pct, pred_value_alert_last_fired_at
- _check_and_fire_pred_value_trend_alert: skips when disabled
- _check_and_fire_pred_value_trend_alert: skips when no prediction_numeric data
- _check_and_fire_pred_value_trend_alert: no alert when shift <= threshold
- _check_and_fire_pred_value_trend_alert: fires when shift > threshold
- _check_and_fire_pred_value_trend_alert: cooldown prevents repeat within 24h
- _check_and_fire_pred_value_trend_alert: fires again after cooldown expires
- _check_and_fire_pred_value_trend_alert: payload contains required keys
- _check_and_fire_pred_value_trend_alert: skips when too few samples (< 20)
- _check_and_fire_pred_value_trend_alert: falls back to confidence when no prediction_numeric
- PUT /api/deploy/{id}/pred-value-alert — enable with alert_pct
- PUT /api/deploy/{id}/pred-value-alert — disable (enabled=false)
- PUT /api/deploy/{id}/pred-value-alert — rejects invalid pct (> 100)
- PUT /api/deploy/{id}/pred-value-alert — 404 for missing deployment
- GET /api/deploy/{id}/pred-value-alert-status — returns config
- GET /api/deploy/{id}/pred-value-alert-status — 404 for missing deployment
- _PRED_VALUE_ALERT_PATTERNS — NL enable/disable/status phrases
- _DISABLE_PRED_VALUE_ALERT_RE — matches disable phrases
- _STATUS_PRED_VALUE_ALERT_RE — matches status phrases
- _PRED_VALUE_ALERT_PCT_RE — extracts numeric percentage thresholds
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from core.webhook import ALL_EVENTS, EVENT_PRED_VALUE_TREND_ALERT

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

    proj_r = await ac.post("/api/projects", json={"name": "PredValueTest"})
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


def test_event_pred_value_trend_alert_constant_exists():
    assert EVENT_PRED_VALUE_TREND_ALERT == "pred_value_trend_alert"


def test_all_events_includes_pred_value_trend_alert():
    assert "pred_value_trend_alert" in ALL_EVENTS


# ---------------------------------------------------------------------------
# Deployment model fields
# ---------------------------------------------------------------------------


def test_deployment_pred_value_alert_fields_exist():
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/x",
        dashboard_url="/predict/x",
    )
    assert hasattr(dep, "pred_value_alert_enabled")
    assert hasattr(dep, "pred_value_alert_pct")
    assert hasattr(dep, "pred_value_alert_last_fired_at")
    assert dep.pred_value_alert_enabled is False
    assert dep.pred_value_alert_pct is None
    assert dep.pred_value_alert_last_fired_at is None


# ---------------------------------------------------------------------------
# _check_and_fire_pred_value_trend_alert unit tests
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


def _make_dep_obj(session, enabled=False, alert_pct=None):
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/x",
        dashboard_url="/predict/x",
        pred_value_alert_enabled=enabled,
        pred_value_alert_pct=alert_pct,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)
    return dep


def _add_logs(session, deployment_id, values, offset_minutes=0):
    """Add PredictionLog rows with given prediction_numeric values."""
    from models.prediction_log import PredictionLog

    now = datetime.now(UTC).replace(tzinfo=None)
    for i, val in enumerate(values):
        log = PredictionLog(
            deployment_id=deployment_id,
            input_features="{}",
            prediction=str(val),
            prediction_numeric=val,
            created_at=now - timedelta(minutes=offset_minutes + i),
        )
        session.add(log)
    session.commit()


def test_check_pred_value_alert_skips_when_disabled(db_session):
    dep = _make_dep_obj(db_session, enabled=False, alert_pct=15.0)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_pred_value_trend_alert

        _check_and_fire_pred_value_trend_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_pred_value_alert_skips_when_no_data(db_session):
    dep = _make_dep_obj(db_session, enabled=True, alert_pct=15.0)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_pred_value_trend_alert

        _check_and_fire_pred_value_trend_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_pred_value_alert_skips_when_too_few_samples(db_session):
    dep = _make_dep_obj(db_session, enabled=True, alert_pct=5.0)
    # Add only 10 samples — below the 20-sample minimum
    _add_logs(db_session, dep.id, [100.0] * 10)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_pred_value_trend_alert

        _check_and_fire_pred_value_trend_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_pred_value_alert_no_alert_when_shift_within_threshold(db_session):
    dep = _make_dep_obj(db_session, enabled=True, alert_pct=20.0)
    # Early half: mean 100, recent half: mean 105 → 5% shift, below 20% threshold
    early = [100.0] * 50
    recent = [105.0] * 50
    _add_logs(db_session, dep.id, recent + early, offset_minutes=0)  # recent first (newest)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_pred_value_trend_alert

        _check_and_fire_pred_value_trend_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_pred_value_alert_fires_when_shift_exceeds_threshold(db_session):
    dep = _make_dep_obj(db_session, enabled=True, alert_pct=10.0)
    # Early half: mean 100, recent half: mean 130 → 30% shift, above 10% threshold
    early = [100.0] * 50
    recent = [130.0] * 50
    _add_logs(db_session, dep.id, recent + early, offset_minutes=0)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_pred_value_trend_alert

        _check_and_fire_pred_value_trend_alert(dep.id)
        mock_dispatch.assert_called_once()
        call_args = mock_dispatch.call_args
        assert call_args[0][1] == "pred_value_trend_alert"


def test_check_pred_value_alert_cooldown_prevents_repeat(db_session):
    dep = _make_dep_obj(db_session, enabled=True, alert_pct=10.0)
    early = [100.0] * 50
    recent = [130.0] * 50
    _add_logs(db_session, dep.id, recent + early)
    # Set last_fired_at to 1 hour ago — still within 24h cooldown
    dep.pred_value_alert_last_fired_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    db_session.add(dep)
    db_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_pred_value_trend_alert

        _check_and_fire_pred_value_trend_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_pred_value_alert_fires_after_cooldown_expires(db_session):
    dep = _make_dep_obj(db_session, enabled=True, alert_pct=10.0)
    early = [100.0] * 50
    recent = [130.0] * 50
    _add_logs(db_session, dep.id, recent + early)
    # Set last_fired_at to 25 hours ago — cooldown has expired
    dep.pred_value_alert_last_fired_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=25)
    db_session.add(dep)
    db_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_pred_value_trend_alert

        _check_and_fire_pred_value_trend_alert(dep.id)
        mock_dispatch.assert_called_once()


def test_check_pred_value_alert_payload_keys(db_session):
    dep = _make_dep_obj(db_session, enabled=True, alert_pct=10.0)
    early = [100.0] * 50
    recent = [130.0] * 50
    _add_logs(db_session, dep.id, recent + early)

    captured_payload = {}

    def capture(dep_id, event, payload):
        captured_payload.update(payload)

    with patch("core.webhook.dispatch_webhooks", side_effect=capture):
        from api.deploy import _check_and_fire_pred_value_trend_alert

        _check_and_fire_pred_value_trend_alert(dep.id)

    assert "deployment_id" in captured_payload
    assert "early_mean" in captured_payload
    assert "recent_mean" in captured_payload
    assert "change_pct" in captured_payload
    assert "threshold_pct" in captured_payload
    assert "sample_size" in captured_payload
    assert "message" in captured_payload


def test_check_pred_value_alert_falls_back_to_confidence(db_session):
    from models.prediction_log import PredictionLog

    dep = _make_dep_obj(db_session, enabled=True, alert_pct=10.0)
    # Add logs with confidence but no prediction_numeric
    now = datetime.now(UTC).replace(tzinfo=None)
    for i in range(50):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="cat_a",
            prediction_numeric=None,
            confidence=0.5,
            created_at=now - timedelta(minutes=i),
        )
        dep_session = db_session
        dep_session.add(log)
    for i in range(50):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features="{}",
            prediction="cat_a",
            prediction_numeric=None,
            confidence=0.9,  # big jump
            created_at=now - timedelta(minutes=50 + i),
        )
        dep_session.add(log)
    dep_session.commit()

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_pred_value_trend_alert

        _check_and_fire_pred_value_trend_alert(dep.id)
        mock_dispatch.assert_called_once()


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_pred_value_alert_enable(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/pred-value-alert",
        json={"enabled": True, "alert_pct": 15.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["pred_value_alert_enabled"] is True
    assert body["alert_pct"] == 15.0


@pytest.mark.anyio
async def test_put_pred_value_alert_disable(ac):
    dep_id = await _make_deployment(ac)
    await ac.put(
        f"/api/deploy/{dep_id}/pred-value-alert",
        json={"enabled": True, "alert_pct": 15.0},
    )
    r = await ac.put(
        f"/api/deploy/{dep_id}/pred-value-alert",
        json={"enabled": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["pred_value_alert_enabled"] is False
    assert body["alert_pct"] is None


@pytest.mark.anyio
async def test_put_pred_value_alert_rejects_invalid_pct(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/pred-value-alert",
        json={"enabled": True, "alert_pct": 150.0},
    )
    assert r.status_code == 400


@pytest.mark.anyio
async def test_put_pred_value_alert_404_missing_deployment(ac):
    r = await ac.put(
        "/api/deploy/nonexistent/pred-value-alert",
        json={"enabled": True, "alert_pct": 15.0},
    )
    assert r.status_code == 404


@pytest.mark.anyio
async def test_get_pred_value_alert_status(ac):
    dep_id = await _make_deployment(ac)
    await ac.put(
        f"/api/deploy/{dep_id}/pred-value-alert",
        json={"enabled": True, "alert_pct": 20.0},
    )
    r = await ac.get(f"/api/deploy/{dep_id}/pred-value-alert-status")
    assert r.status_code == 200
    body = r.json()
    assert body["pred_value_alert_enabled"] is True
    assert body["alert_pct"] == 20.0
    assert "cooldown_hours" in body
    assert "summary" in body


@pytest.mark.anyio
async def test_get_pred_value_alert_status_404_missing_deployment(ac):
    r = await ac.get("/api/deploy/nonexistent/pred-value-alert-status")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Chat regex pattern tests
# ---------------------------------------------------------------------------


def test_pred_value_alert_patterns_match_enable_phrases():
    from api.chat import _PRED_VALUE_ALERT_PATTERNS

    phrases = [
        "alert me when my average predictions drop by more than 15%",
        "set up a prediction value alert",
        "configure output trend alert",
        "notify me if my mean prediction shifts",
        "alert me if revenue prediction drops more than 20%",
        "prediction value trend alert",
        "enable a value trend alert",
    ]
    for phrase in phrases:
        assert _PRED_VALUE_ALERT_PATTERNS.search(phrase), f"No match: {phrase!r}"


def test_pred_value_alert_patterns_no_false_positives():
    from api.chat import _PRED_VALUE_ALERT_PATTERNS

    false_positives = [
        "show me feature importance",
        "how accurate is my model",
        "what is the latency threshold",
    ]
    for phrase in false_positives:
        assert not _PRED_VALUE_ALERT_PATTERNS.search(phrase), f"False positive: {phrase!r}"


def test_disable_pred_value_alert_re_matches():
    from api.chat import _DISABLE_PRED_VALUE_ALERT_RE

    phrases = [
        "disable the prediction value alert",
        "remove the output trend alert",
        "turn off value trend alert",
    ]
    for phrase in phrases:
        assert _DISABLE_PRED_VALUE_ALERT_RE.search(phrase), f"No match: {phrase!r}"


def test_status_pred_value_alert_re_matches():
    from api.chat import _STATUS_PRED_VALUE_ALERT_RE

    phrases = [
        "show my prediction value alert",
        "check my output trend alert setting",
        "status of prediction value alert",
    ]
    for phrase in phrases:
        assert _STATUS_PRED_VALUE_ALERT_RE.search(phrase), f"No match: {phrase!r}"


def test_pred_value_alert_pct_re_extracts_percentage():
    from api.chat import _PRED_VALUE_ALERT_PCT_RE

    assert _PRED_VALUE_ALERT_PCT_RE.search("alert me if predictions drop more than 15%").group(1) == "15"
    assert _PRED_VALUE_ALERT_PCT_RE.search("set threshold to 7.5%").group(1) == "7.5"
    assert _PRED_VALUE_ALERT_PCT_RE.search("no percentage here") is None
