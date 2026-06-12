"""Tests for the Input Distribution Drift Alert feature.

Covers:
- EVENT_INPUT_DIST_DRIFT constant exists in core.webhook
- ALL_EVENTS includes input_dist_drift
- Deployment fields: input_dist_drift_alert_enabled, input_dist_drift_severity_threshold,
  input_dist_drift_alert_last_fired_at
- _check_and_fire_input_dist_drift_alert: skips when disabled
- _check_and_fire_input_dist_drift_alert: skips when no pipeline_path
- _check_and_fire_input_dist_drift_alert: skips when too few samples
- _check_and_fire_input_dist_drift_alert: no alert when severity below threshold
- _check_and_fire_input_dist_drift_alert: fires when severity meets threshold (medium)
- _check_and_fire_input_dist_drift_alert: skips when high threshold but only medium severity
- _check_and_fire_input_dist_drift_alert: cooldown prevents repeat within 24h
- _check_and_fire_input_dist_drift_alert: fires again after cooldown expires
- _check_and_fire_input_dist_drift_alert: payload contains required keys
- PUT /api/deploy/{id}/input-dist-drift-alert — enable with medium threshold
- PUT /api/deploy/{id}/input-dist-drift-alert — enable with high threshold
- PUT /api/deploy/{id}/input-dist-drift-alert — disable (enabled=false)
- PUT /api/deploy/{id}/input-dist-drift-alert — invalid threshold defaults to medium
- PUT /api/deploy/{id}/input-dist-drift-alert — 404 for missing deployment
- GET /api/deploy/{id}/input-dist-drift-alert-status — returns config
- GET /api/deploy/{id}/input-dist-drift-alert-status — 404 for missing deployment
- _INPUT_DIST_DRIFT_ALERT_PATTERNS — NL enable/disable/status phrases
- _DISABLE_INPUT_DIST_DRIFT_ALERT_RE — matches disable phrases
- _STATUS_INPUT_DIST_DRIFT_ALERT_RE — matches status phrases
- _HIGH_THRESHOLD_INPUT_DIST_RE — matches high-severity request phrases
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from core.webhook import ALL_EVENTS, EVENT_INPUT_DIST_DRIFT

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
def db_session(tmp_path):
    test_db = str(tmp_path / "unit.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)

    import models.deployment  # noqa
    import models.prediction_log  # noqa
    import models.webhook_config  # noqa

    SQLModel.metadata.create_all(db_module.engine)
    with Session(db_module.engine) as s:
        yield s


async def _make_deployment(ac):
    """Create project, upload CSV, train a model, and deploy it. Returns deployment_id."""
    import time

    proj_r = await ac.post("/api/projects", json={"name": "InputDriftAlertTest"})
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


def _make_dep_obj(session, enabled=False, threshold="medium", pipeline_path=None):
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/x",
        dashboard_url="/predict/x",
        input_dist_drift_alert_enabled=enabled,
        input_dist_drift_severity_threshold=threshold,
        pipeline_path=pipeline_path,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)
    return dep


def _add_logs_with_inputs(session, deployment_id, inputs_list):
    from models.prediction_log import PredictionLog

    now = datetime.now(UTC).replace(tzinfo=None)
    for i, inp in enumerate(inputs_list):
        log = PredictionLog(
            deployment_id=deployment_id,
            input_features=json.dumps(inp),
            prediction="50.0",
            prediction_numeric=50.0,
            created_at=now - timedelta(minutes=i),
        )
        session.add(log)
    session.commit()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_event_input_dist_drift_constant_exists():
    assert EVENT_INPUT_DIST_DRIFT == "input_dist_drift"


def test_all_events_includes_input_dist_drift():
    assert "input_dist_drift" in ALL_EVENTS


# ---------------------------------------------------------------------------
# Deployment model fields
# ---------------------------------------------------------------------------


def test_deployment_input_dist_drift_fields_exist():
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/x",
        dashboard_url="/predict/x",
    )
    assert hasattr(dep, "input_dist_drift_alert_enabled")
    assert hasattr(dep, "input_dist_drift_severity_threshold")
    assert hasattr(dep, "input_dist_drift_alert_last_fired_at")
    assert dep.input_dist_drift_alert_enabled is False
    assert dep.input_dist_drift_severity_threshold == "medium"
    assert dep.input_dist_drift_alert_last_fired_at is None


# ---------------------------------------------------------------------------
# _check_and_fire_input_dist_drift_alert unit tests
# ---------------------------------------------------------------------------


def test_check_input_dist_drift_skips_when_disabled(db_session):
    dep = _make_dep_obj(db_session, enabled=False)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_input_dist_drift_alert

        _check_and_fire_input_dist_drift_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_input_dist_drift_skips_when_no_pipeline(db_session):
    dep = _make_dep_obj(db_session, enabled=True, pipeline_path=None)
    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_input_dist_drift_alert

        _check_and_fire_input_dist_drift_alert(dep.id)
        mock_dispatch.assert_not_called()


def test_check_input_dist_drift_skips_when_too_few_samples(db_session, tmp_path):
    pipeline_path = str(tmp_path / "fake.pkl")
    dep = _make_dep_obj(db_session, enabled=True, pipeline_path=pipeline_path)

    feature_ranges = {"units": (5.0, 35.0)}
    mock_pl = MagicMock()
    mock_pl.feature_ranges = feature_ranges

    _add_logs_with_inputs(db_session, dep.id, [{"units": 10.0}] * 5)

    with patch("api.deploy._lp_idd" if False else "core.deployer.load_pipeline", return_value=mock_pl):
        with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
            from api.deploy import _check_and_fire_input_dist_drift_alert

            _check_and_fire_input_dist_drift_alert(dep.id)
            mock_dispatch.assert_not_called()


def test_check_input_dist_drift_no_alert_when_severity_none(db_session, tmp_path):
    pipeline_path = str(tmp_path / "fake.pkl")
    dep = _make_dep_obj(db_session, enabled=True, threshold="medium", pipeline_path=pipeline_path)

    feature_ranges = {"units": (5.0, 35.0)}
    mock_pl = MagicMock()
    mock_pl.feature_ranges = feature_ranges

    # All inputs in range → severity "none"
    in_range_inputs = [{"units": 10.0}] * 30
    _add_logs_with_inputs(db_session, dep.id, in_range_inputs)

    mock_result = {
        "severity": "none",
        "severity_label": "None",
        "alert_count": 0,
        "sample_count": 30,
        "alerts": [],
        "has_alerts": False,
    }

    with patch("core.deployer.load_pipeline", return_value=mock_pl):
        with patch("core.analyzer.compute_covariate_drift_alert", return_value=mock_result):
            with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
                from api.deploy import _check_and_fire_input_dist_drift_alert

                _check_and_fire_input_dist_drift_alert(dep.id)
                mock_dispatch.assert_not_called()


def test_check_input_dist_drift_fires_on_medium_severity(db_session, tmp_path):
    pipeline_path = str(tmp_path / "fake.pkl")
    dep = _make_dep_obj(db_session, enabled=True, threshold="medium", pipeline_path=pipeline_path)

    feature_ranges = {"units": (5.0, 35.0)}
    mock_pl = MagicMock()
    mock_pl.feature_ranges = feature_ranges

    _add_logs_with_inputs(db_session, dep.id, [{"units": 10.0}] * 30)

    mock_result = {
        "severity": "medium",
        "severity_label": "Medium",
        "alert_count": 2,
        "sample_count": 30,
        "alerts": [{"feature": "units", "oor_rate": 0.20}],
        "has_alerts": True,
    }

    with patch("core.deployer.load_pipeline", return_value=mock_pl):
        with patch("core.analyzer.compute_covariate_drift_alert", return_value=mock_result):
            with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
                from api.deploy import _check_and_fire_input_dist_drift_alert

                _check_and_fire_input_dist_drift_alert(dep.id)
                mock_dispatch.assert_called_once()
                event_name = mock_dispatch.call_args[0][1]
                assert event_name == "input_dist_drift"


def test_check_input_dist_drift_skips_when_high_threshold_but_medium_severity(
    db_session, tmp_path
):
    pipeline_path = str(tmp_path / "fake.pkl")
    dep = _make_dep_obj(db_session, enabled=True, threshold="high", pipeline_path=pipeline_path)

    feature_ranges = {"units": (5.0, 35.0)}
    mock_pl = MagicMock()
    mock_pl.feature_ranges = feature_ranges

    _add_logs_with_inputs(db_session, dep.id, [{"units": 10.0}] * 30)

    mock_result = {
        "severity": "medium",
        "severity_label": "Medium",
        "alert_count": 2,
        "sample_count": 30,
        "alerts": [{"feature": "units", "oor_rate": 0.20}],
        "has_alerts": True,
    }

    with patch("core.deployer.load_pipeline", return_value=mock_pl):
        with patch("core.analyzer.compute_covariate_drift_alert", return_value=mock_result):
            with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
                from api.deploy import _check_and_fire_input_dist_drift_alert

                _check_and_fire_input_dist_drift_alert(dep.id)
                mock_dispatch.assert_not_called()


def test_check_input_dist_drift_cooldown_prevents_repeat(db_session, tmp_path):
    pipeline_path = str(tmp_path / "fake.pkl")
    dep = _make_dep_obj(db_session, enabled=True, pipeline_path=pipeline_path)
    dep.input_dist_drift_alert_last_fired_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    db_session.add(dep)
    db_session.commit()

    mock_pl = MagicMock()
    mock_pl.feature_ranges = {"units": (5.0, 35.0)}
    _add_logs_with_inputs(db_session, dep.id, [{"units": 10.0}] * 30)

    mock_result = {"severity": "medium", "alert_count": 1, "alerts": [], "sample_count": 30}

    with patch("core.deployer.load_pipeline", return_value=mock_pl):
        with patch("core.analyzer.compute_covariate_drift_alert", return_value=mock_result):
            with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
                from api.deploy import _check_and_fire_input_dist_drift_alert

                _check_and_fire_input_dist_drift_alert(dep.id)
                mock_dispatch.assert_not_called()


def test_check_input_dist_drift_fires_after_cooldown_expires(db_session, tmp_path):
    pipeline_path = str(tmp_path / "fake.pkl")
    dep = _make_dep_obj(db_session, enabled=True, pipeline_path=pipeline_path)
    dep.input_dist_drift_alert_last_fired_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=25)
    db_session.add(dep)
    db_session.commit()

    mock_pl = MagicMock()
    mock_pl.feature_ranges = {"units": (5.0, 35.0)}
    _add_logs_with_inputs(db_session, dep.id, [{"units": 10.0}] * 30)

    mock_result = {
        "severity": "medium",
        "severity_label": "Medium",
        "alert_count": 1,
        "sample_count": 30,
        "alerts": [{"feature": "units", "oor_rate": 0.20}],
        "has_alerts": True,
    }

    with patch("core.deployer.load_pipeline", return_value=mock_pl):
        with patch("core.analyzer.compute_covariate_drift_alert", return_value=mock_result):
            with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
                from api.deploy import _check_and_fire_input_dist_drift_alert

                _check_and_fire_input_dist_drift_alert(dep.id)
                mock_dispatch.assert_called_once()


def test_check_input_dist_drift_payload_keys(db_session, tmp_path):
    pipeline_path = str(tmp_path / "fake.pkl")
    dep = _make_dep_obj(db_session, enabled=True, pipeline_path=pipeline_path)

    mock_pl = MagicMock()
    mock_pl.feature_ranges = {"units": (5.0, 35.0), "region": None}
    _add_logs_with_inputs(db_session, dep.id, [{"units": 10.0}] * 30)

    mock_result = {
        "severity": "high",
        "severity_label": "High",
        "alert_count": 3,
        "sample_count": 30,
        "alerts": [{"feature": "units", "oor_rate": 0.35}, {"feature": "region", "oor_rate": 0.40}],
        "has_alerts": True,
    }

    captured = {}

    def capture(dep_id, event, payload):
        captured.update(payload)

    with patch("core.deployer.load_pipeline", return_value=mock_pl):
        with patch("core.analyzer.compute_covariate_drift_alert", return_value=mock_result):
            with patch("core.webhook.dispatch_webhooks", side_effect=capture):
                from api.deploy import _check_and_fire_input_dist_drift_alert

                _check_and_fire_input_dist_drift_alert(dep.id)

    assert "severity" in captured
    assert "severity_label" in captured
    assert "alert_count" in captured
    assert "sample_count" in captured
    assert "top_drifted_features" in captured
    assert "message" in captured


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_input_dist_drift_alert_enable_medium(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/input-dist-drift-alert",
        json={"enabled": True, "severity_threshold": "medium"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["input_dist_drift_alert_enabled"] is True
    assert body["input_dist_drift_severity_threshold"] == "medium"
    assert "cooldown_hours" in body
    assert "summary" in body


@pytest.mark.anyio
async def test_put_input_dist_drift_alert_enable_high(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/input-dist-drift-alert",
        json={"enabled": True, "severity_threshold": "high"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["input_dist_drift_alert_enabled"] is True
    assert body["input_dist_drift_severity_threshold"] == "high"


@pytest.mark.anyio
async def test_put_input_dist_drift_alert_disable(ac):
    dep_id = await _make_deployment(ac)
    await ac.put(
        f"/api/deploy/{dep_id}/input-dist-drift-alert",
        json={"enabled": True, "severity_threshold": "medium"},
    )
    r = await ac.put(
        f"/api/deploy/{dep_id}/input-dist-drift-alert",
        json={"enabled": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["input_dist_drift_alert_enabled"] is False
    assert body["input_dist_drift_alert_last_fired_at"] is None


@pytest.mark.anyio
async def test_put_input_dist_drift_alert_invalid_threshold_defaults_medium(ac):
    dep_id = await _make_deployment(ac)
    r = await ac.put(
        f"/api/deploy/{dep_id}/input-dist-drift-alert",
        json={"enabled": True, "severity_threshold": "extreme"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["input_dist_drift_severity_threshold"] == "medium"


@pytest.mark.anyio
async def test_put_input_dist_drift_alert_404_missing_deployment(ac):
    r = await ac.put(
        "/api/deploy/nonexistent/input-dist-drift-alert",
        json={"enabled": True},
    )
    assert r.status_code == 404


@pytest.mark.anyio
async def test_get_input_dist_drift_alert_status(ac):
    dep_id = await _make_deployment(ac)
    await ac.put(
        f"/api/deploy/{dep_id}/input-dist-drift-alert",
        json={"enabled": True, "severity_threshold": "high"},
    )
    r = await ac.get(f"/api/deploy/{dep_id}/input-dist-drift-alert-status")
    assert r.status_code == 200
    body = r.json()
    assert body["input_dist_drift_alert_enabled"] is True
    assert body["input_dist_drift_severity_threshold"] == "high"
    assert "cooldown_hours" in body
    assert "summary" in body


@pytest.mark.anyio
async def test_get_input_dist_drift_alert_status_404_missing_deployment(ac):
    r = await ac.get("/api/deploy/nonexistent/input-dist-drift-alert-status")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Chat regex pattern tests
# ---------------------------------------------------------------------------


def test_input_dist_drift_alert_patterns_match_enable_phrases():
    from api.chat import _INPUT_DIST_DRIFT_ALERT_PATTERNS

    phrases = [
        "enable input distribution drift alerts",
        "turn on distribution drift notifications",
        "alert me when live inputs diverge from training",
        "notify me when prediction inputs drift",
        "input distribution drift alert",
        "proactive distribution drift alerts",
        "alert me when live inputs change from training",
    ]
    for phrase in phrases:
        assert _INPUT_DIST_DRIFT_ALERT_PATTERNS.search(phrase), f"No match: {phrase!r}"


def test_input_dist_drift_alert_patterns_no_false_positives():
    from api.chat import _INPUT_DIST_DRIFT_ALERT_PATTERNS

    false_positives = [
        "show me feature importance",
        "what is the model accuracy",
        "configure a latency alert",
    ]
    for phrase in false_positives:
        assert not _INPUT_DIST_DRIFT_ALERT_PATTERNS.search(phrase), (
            f"False positive: {phrase!r}"
        )


def test_disable_input_dist_drift_alert_re_matches():
    from api.chat import _DISABLE_INPUT_DIST_DRIFT_ALERT_RE

    phrases = [
        "disable the input distribution drift alert",
        "turn off distribution drift notification",
        "deactivate input drift alarm",
    ]
    for phrase in phrases:
        assert _DISABLE_INPUT_DIST_DRIFT_ALERT_RE.search(phrase), f"No match: {phrase!r}"


def test_status_input_dist_drift_alert_re_matches():
    from api.chat import _STATUS_INPUT_DIST_DRIFT_ALERT_RE

    phrases = [
        "check my distribution drift alert",
        "status of input distribution drift alert",
        "status of distribution drift notification",
    ]
    for phrase in phrases:
        assert _STATUS_INPUT_DIST_DRIFT_ALERT_RE.search(phrase), f"No match: {phrase!r}"


def test_high_threshold_input_dist_re_matches():
    from api.chat import _HIGH_THRESHOLD_INPUT_DIST_RE

    assert _HIGH_THRESHOLD_INPUT_DIST_RE.search("set high severity threshold")
    assert _HIGH_THRESHOLD_INPUT_DIST_RE.search("use critical severity")
    assert _HIGH_THRESHOLD_INPUT_DIST_RE.search("extreme drift")
    assert not _HIGH_THRESHOLD_INPUT_DIST_RE.search("enable medium drift alert")
