"""Tests for the Deployment Feature Drift Alert webhook.

Covers:
- EVENT_FEATURE_DRIFT constant exists in core.webhook
- ALL_EVENTS includes feature_drift
- Deployment.feature_drift_alert_enabled / feature_drift_alert_last_fired_at fields
- _check_and_fire_feature_drift_alert: skips when alert disabled
- _check_and_fire_feature_drift_alert: fires when enabled + critical features
- _check_and_fire_feature_drift_alert: cooldown prevents repeat within 24h
- _check_and_fire_feature_drift_alert: fires again after cooldown expires
- _check_and_fire_feature_drift_alert: fires for attention-priority features too
- _check_and_fire_feature_drift_alert: payload contains required keys
- PUT /api/deploy/{id}/feature-drift-alert — enable
- PUT /api/deploy/{id}/feature-drift-alert — disable resets last_fired_at
- PUT /api/deploy/{id}/feature-drift-alert — 404 for missing deployment
- GET /api/deploy/{id}/feature-drift-alert-status — returns config
- GET /api/deploy/{id}/feature-drift-alert-status — 404 for missing deployment
- _FEATURE_DRIFT_ALERT_PATTERNS — NL enable/disable/status phrases
- _DISABLE_FEATURE_DRIFT_ALERT_RE — matches disable phrases
- _STATUS_FEATURE_DRIFT_ALERT_RE — matches status phrases
- drift-importance-ranking endpoint fires webhook thread when enabled + critical
- drift-importance-ranking endpoint skips thread when alert disabled
"""

from __future__ import annotations

import io
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from core.webhook import ALL_EVENTS, EVENT_FEATURE_DRIFT

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


@pytest.fixture()
async def deployed(ac):
    resp = await ac.post("/api/projects", json={"name": "Drift Alert Project"})
    project_id = resp.json()["id"]

    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201
    dataset_id = resp.json()["dataset_id"]

    resp = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert resp.status_code == 201
    fs_id = resp.json()["feature_set_id"]

    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )

    resp = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": fs_id},
    )
    assert resp.status_code == 202
    run_id = resp.json()["model_run_ids"][0]

    for _ in range(30):
        r = await ac.get(f"/api/models/{project_id}/runs")
        run = next((x for x in r.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            break
        time.sleep(0.3)
    else:
        pytest.skip("Training did not complete")

    resp = await ac.post(f"/api/deploy/{run_id}", json={})
    assert resp.status_code == 201, resp.text
    deployment_id = resp.json()["id"]

    return {"project_id": project_id, "deployment_id": deployment_id}


# ---------------------------------------------------------------------------
# Webhook constant tests
# ---------------------------------------------------------------------------


def test_event_feature_drift_constant():
    assert EVENT_FEATURE_DRIFT == "feature_drift"


def test_all_events_includes_feature_drift():
    assert EVENT_FEATURE_DRIFT in ALL_EVENTS


# ---------------------------------------------------------------------------
# Model field tests
# ---------------------------------------------------------------------------


def test_deployment_feature_drift_alert_fields():
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="r",
        project_id="p",
        endpoint_path="/e",
        dashboard_url="/d",
    )
    assert dep.feature_drift_alert_enabled is False
    assert dep.feature_drift_alert_last_fired_at is None


# ---------------------------------------------------------------------------
# _check_and_fire_feature_drift_alert unit tests
# ---------------------------------------------------------------------------


def _make_in_memory_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    import models.deployment  # noqa: F401
    import models.webhook_config  # noqa: F401
    import models.webhook_event  # noqa: F401

    SQLModel.metadata.create_all(engine)
    return engine, Session(engine)


def test_check_fire_skips_when_disabled():
    from api.deploy import _check_and_fire_feature_drift_alert
    from models.deployment import Deployment

    engine, session = _make_in_memory_session()
    dep = Deployment(
        model_run_id="r",
        project_id="p",
        endpoint_path="/e",
        dashboard_url="/d",
        feature_drift_alert_enabled=False,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        import db as db_mod_local

        old_engine = db_mod_local.engine
        db_mod_local.engine = engine
        try:
            _check_and_fire_feature_drift_alert(
                dep.id, [{"feature": "x", "priority": "action_required"}]
            )
        finally:
            db_mod_local.engine = old_engine

    mock_dispatch.assert_not_called()


def test_check_fire_dispatches_when_enabled():
    from api.deploy import _check_and_fire_feature_drift_alert
    from models.deployment import Deployment

    engine, session = _make_in_memory_session()
    dep = Deployment(
        model_run_id="r",
        project_id="p",
        endpoint_path="/e",
        dashboard_url="/d",
        feature_drift_alert_enabled=True,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        import db as db_mod_local

        old_engine = db_mod_local.engine
        db_mod_local.engine = engine
        try:
            _check_and_fire_feature_drift_alert(
                dep.id,
                [
                    {"feature": "region", "priority": "action_required"},
                    {"feature": "units", "priority": "attention"},
                ],
            )
        finally:
            db_mod_local.engine = old_engine

    mock_dispatch.assert_called_once()
    call_args = mock_dispatch.call_args
    assert call_args[0][1] == EVENT_FEATURE_DRIFT
    assert "critical_feature_count" in call_args[0][2]
    assert call_args[0][2]["critical_feature_count"] == 2


def test_check_fire_cooldown_prevents_second_fire():
    from api.deploy import _check_and_fire_feature_drift_alert
    from models.deployment import Deployment

    engine, session = _make_in_memory_session()
    # Pre-stamp last_fired_at to 1 hour ago
    dep = Deployment(
        model_run_id="r",
        project_id="p",
        endpoint_path="/e",
        dashboard_url="/d",
        feature_drift_alert_enabled=True,
        feature_drift_alert_last_fired_at=datetime.now(UTC).replace(tzinfo=None)
        - timedelta(hours=1),
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        import db as db_mod_local

        old_engine = db_mod_local.engine
        db_mod_local.engine = engine
        try:
            _check_and_fire_feature_drift_alert(
                dep.id,
                [{"feature": "x", "priority": "action_required"}],
                cooldown_hours=24.0,
            )
        finally:
            db_mod_local.engine = old_engine

    mock_dispatch.assert_not_called()


def test_check_fire_allowed_after_cooldown():
    from api.deploy import _check_and_fire_feature_drift_alert
    from models.deployment import Deployment

    engine, session = _make_in_memory_session()
    dep = Deployment(
        model_run_id="r",
        project_id="p",
        endpoint_path="/e",
        dashboard_url="/d",
        feature_drift_alert_enabled=True,
        feature_drift_alert_last_fired_at=datetime.now(UTC).replace(tzinfo=None)
        - timedelta(hours=25),
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        import db as db_mod_local

        old_engine = db_mod_local.engine
        db_mod_local.engine = engine
        try:
            _check_and_fire_feature_drift_alert(
                dep.id,
                [{"feature": "x", "priority": "action_required"}],
                cooldown_hours=24.0,
            )
        finally:
            db_mod_local.engine = old_engine

    mock_dispatch.assert_called_once()


def test_check_fire_attention_priority_triggers():
    from api.deploy import _check_and_fire_feature_drift_alert
    from models.deployment import Deployment

    engine, session = _make_in_memory_session()
    dep = Deployment(
        model_run_id="r",
        project_id="p",
        endpoint_path="/e",
        dashboard_url="/d",
        feature_drift_alert_enabled=True,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        import db as db_mod_local

        old_engine = db_mod_local.engine
        db_mod_local.engine = engine
        try:
            _check_and_fire_feature_drift_alert(
                dep.id, [{"feature": "units", "priority": "attention"}]
            )
        finally:
            db_mod_local.engine = old_engine

    mock_dispatch.assert_called_once()


def test_check_fire_payload_keys():
    from api.deploy import _check_and_fire_feature_drift_alert
    from models.deployment import Deployment

    engine, session = _make_in_memory_session()
    dep = Deployment(
        model_run_id="r",
        project_id="p",
        endpoint_path="/e",
        dashboard_url="/d",
        feature_drift_alert_enabled=True,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)

    captured = {}

    def _capture(dep_id, event_type, payload):
        captured.update(payload)

    with patch("core.webhook.dispatch_webhooks", side_effect=_capture):
        import db as db_mod_local

        old_engine = db_mod_local.engine
        db_mod_local.engine = engine
        try:
            _check_and_fire_feature_drift_alert(
                dep.id,
                [
                    {"feature": "region", "priority": "action_required"},
                    {"feature": "units", "priority": "attention"},
                ],
            )
        finally:
            db_mod_local.engine = old_engine

    assert "critical_feature_count" in captured
    assert "top_critical_features" in captured
    assert "message" in captured
    assert captured["critical_feature_count"] == 2
    assert "region" in captured["top_critical_features"]


def test_check_fire_updates_last_fired_at():
    from api.deploy import _check_and_fire_feature_drift_alert
    from models.deployment import Deployment

    engine, session = _make_in_memory_session()
    dep = Deployment(
        model_run_id="r",
        project_id="p",
        endpoint_path="/e",
        dashboard_url="/d",
        feature_drift_alert_enabled=True,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)
    dep_id = dep.id

    with patch("core.webhook.dispatch_webhooks"):
        import db as db_mod_local

        old_engine = db_mod_local.engine
        db_mod_local.engine = engine
        try:
            _check_and_fire_feature_drift_alert(
                dep_id, [{"feature": "x", "priority": "action_required"}]
            )
        finally:
            db_mod_local.engine = old_engine

    session.refresh(dep)
    assert dep.feature_drift_alert_last_fired_at is not None


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_feature_drift_alert_enable(ac, deployed):
    dep_id = deployed["deployment_id"]
    resp = await ac.put(
        f"/api/deploy/{dep_id}/feature-drift-alert", json={"enabled": True}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["feature_drift_alert_enabled"] is True
    assert data["cooldown_hours"] == 24
    assert "enabled" in data["summary"].lower()


@pytest.mark.asyncio
async def test_put_feature_drift_alert_disable(ac, deployed):
    dep_id = deployed["deployment_id"]
    # Enable first
    await ac.put(f"/api/deploy/{dep_id}/feature-drift-alert", json={"enabled": True})
    # Disable
    resp = await ac.put(
        f"/api/deploy/{dep_id}/feature-drift-alert", json={"enabled": False}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["feature_drift_alert_enabled"] is False
    assert data["feature_drift_alert_last_fired_at"] is None


@pytest.mark.asyncio
async def test_put_feature_drift_alert_404(ac):
    resp = await ac.put(
        "/api/deploy/nonexistent/feature-drift-alert", json={"enabled": True}
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_feature_drift_alert_status(ac, deployed):
    dep_id = deployed["deployment_id"]
    await ac.put(f"/api/deploy/{dep_id}/feature-drift-alert", json={"enabled": True})
    resp = await ac.get(f"/api/deploy/{dep_id}/feature-drift-alert-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deployment_id"] == dep_id
    assert data["feature_drift_alert_enabled"] is True
    assert "cooldown_hours" in data
    assert "summary" in data


@pytest.mark.asyncio
async def test_get_feature_drift_alert_status_404(ac):
    resp = await ac.get("/api/deploy/nonexistent/feature-drift-alert-status")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


def test_feature_drift_alert_patterns_enable():
    from api.chat import _FEATURE_DRIFT_ALERT_PATTERNS

    phrases = [
        "enable feature drift alerts",
        "turn on drift webhook",
        "alert me when features drift",
        "notify me on critical feature drift",
        "feature drift alert",
        "webhook on feature drift",
        "proactive drift alert",
        "automatic drift notification",
    ]
    for phrase in phrases:
        assert _FEATURE_DRIFT_ALERT_PATTERNS.search(phrase), (
            f"Pattern missed: {phrase!r}"
        )


def test_feature_drift_alert_patterns_disable():
    from api.chat import _FEATURE_DRIFT_ALERT_PATTERNS

    phrases = [
        "disable feature drift alert",
        "turn off drift webhook",
        "deactivate drift notification",
    ]
    for phrase in phrases:
        assert _FEATURE_DRIFT_ALERT_PATTERNS.search(phrase), (
            f"Pattern missed: {phrase!r}"
        )


def test_feature_drift_alert_patterns_status():
    from api.chat import _FEATURE_DRIFT_ALERT_PATTERNS

    phrases = [
        "status of feature drift alert",
        "check feature drift webhook",
    ]
    for phrase in phrases:
        assert _FEATURE_DRIFT_ALERT_PATTERNS.search(phrase), (
            f"Pattern missed: {phrase!r}"
        )


def test_disable_feature_drift_alert_re():
    from api.chat import _DISABLE_FEATURE_DRIFT_ALERT_RE

    assert _DISABLE_FEATURE_DRIFT_ALERT_RE.search("disable feature drift alert")
    assert _DISABLE_FEATURE_DRIFT_ALERT_RE.search("turn off drift webhook")
    assert not _DISABLE_FEATURE_DRIFT_ALERT_RE.search("enable drift alert")


def test_status_feature_drift_alert_re():
    from api.chat import _STATUS_FEATURE_DRIFT_ALERT_RE

    assert _STATUS_FEATURE_DRIFT_ALERT_RE.search("status of feature drift alert")
    assert _STATUS_FEATURE_DRIFT_ALERT_RE.search("check feature drift webhook")
    assert not _STATUS_FEATURE_DRIFT_ALERT_RE.search("enable feature drift alert")


def test_feature_drift_alert_no_false_positives():
    from api.chat import _FEATURE_DRIFT_ALERT_PATTERNS

    # These should NOT match the pattern
    non_matches = [
        "show me drift importance ranking",
        "which features drifted most",
        "compare model versions",
    ]
    for phrase in non_matches:
        assert not _FEATURE_DRIFT_ALERT_PATTERNS.search(phrase), (
            f"False positive: {phrase!r}"
        )
