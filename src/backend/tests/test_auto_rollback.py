"""Tests for the Accuracy-Triggered Auto-Rollback feature.

Covers:
- EVENT_ROLLBACK_TRIGGERED constant exists in core.webhook
- ALL_EVENTS includes rollback_triggered
- Deployment model fields: auto_rollback_enabled, auto_rollback_accuracy_threshold,
  auto_rollback_triggered_at
- _check_and_fire_accuracy_rollback: skips when disabled
- _check_and_fire_accuracy_rollback: skips when threshold is None
- _check_and_fire_accuracy_rollback: skips when fewer than min_feedback records
- _check_and_fire_accuracy_rollback: no rollback when accuracy >= threshold
- _check_and_fire_accuracy_rollback: triggers rollback when accuracy < threshold
- _check_and_fire_accuracy_rollback: cooldown prevents repeat within 24h
- _check_and_fire_accuracy_rollback: skips when fewer than 2 deployment versions
- _check_and_fire_accuracy_rollback: dispatches webhook with required keys
- PUT /api/deploy/{id}/auto-rollback — enable with threshold
- PUT /api/deploy/{id}/auto-rollback — disable (enabled=false)
- PUT /api/deploy/{id}/auto-rollback — rejects threshold out of range
- PUT /api/deploy/{id}/auto-rollback — rejects enabled=true without threshold
- PUT /api/deploy/{id}/auto-rollback — 404 for missing deployment
- GET /api/deploy/{id}/auto-rollback-status — returns config
- GET /api/deploy/{id}/auto-rollback-status — 404 for missing deployment
- _AUTO_ROLLBACK_PATTERNS — matches NL enable/disable/status phrases
- _DISABLE_AUTO_ROLLBACK_RE — matches disable phrases
- _STATUS_AUTO_ROLLBACK_RE — matches status phrases
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from core.webhook import ALL_EVENTS, EVENT_ROLLBACK_TRIGGERED
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
    import models.deployment_version  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa
    import models.webhook_event  # noqa
    import models.weekly_digest_config  # noqa

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
    proj_r = await ac.post("/api/projects", json={"name": "ARBTest"})
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


def test_event_rollback_triggered_constant():
    assert EVENT_ROLLBACK_TRIGGERED == "rollback_triggered"


def test_rollback_triggered_in_all_events():
    assert EVENT_ROLLBACK_TRIGGERED in ALL_EVENTS


def test_deployment_model_fields():
    from models.deployment import Deployment

    d = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/d1",
        dashboard_url="/predict/d1",
        auto_rollback_enabled=True,
        auto_rollback_accuracy_threshold=0.80,
        auto_rollback_triggered_at=None,
    )
    assert d.auto_rollback_enabled is True
    assert d.auto_rollback_accuracy_threshold == pytest.approx(0.80)
    assert d.auto_rollback_triggered_at is None


# ---------------------------------------------------------------------------
# _check_and_fire_accuracy_rollback unit tests
# ---------------------------------------------------------------------------


def _make_dep_with_rollback(session, enabled=True, threshold=0.80, last_triggered=None):
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="run-arb-001",
        project_id="proj-arb-001",
        endpoint_path="/api/predict/arb-dep",
        dashboard_url="/predict/arb-dep",
        auto_rollback_enabled=enabled,
        auto_rollback_accuracy_threshold=threshold if enabled else None,
        auto_rollback_triggered_at=last_triggered,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)
    return dep


def _make_deployment_version(session, dep_id, ver_num, is_current, pipeline_path):
    from models.deployment_version import DeploymentVersion

    v = DeploymentVersion(
        deployment_id=dep_id,
        version_number=ver_num,
        model_run_id="run-arb-001",
        algorithm="linear_regression",
        problem_type="regression",
        target_column="revenue",
        metrics="{}",
        pipeline_path=pipeline_path,
        is_current=is_current,
    )
    session.add(v)
    session.commit()
    session.refresh(v)
    return v


def _make_feedback(session, dep_id, is_correct_list):
    from models.feedback_record import FeedbackRecord

    for correct in is_correct_list:
        fb = FeedbackRecord(
            deployment_id=dep_id,
            prediction_log_id=None,
            actual_value=None,
            actual_label=None,
            is_correct=correct,
        )
        session.add(fb)
    session.commit()


@pytest.fixture()
def mem_session(tmp_path):
    test_db = str(tmp_path / "arb_test.db")
    engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.engine = engine

    import models.deployment  # noqa
    import models.deployment_version  # noqa
    import models.feedback_record  # noqa
    import models.prediction_log  # noqa
    import models.webhook_config  # noqa
    import models.webhook_event  # noqa
    import models.project  # noqa
    import models.dataset  # noqa
    import models.conversation  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.dataset_filter  # noqa
    import models.weekly_digest_config  # noqa

    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_rollback_skips_when_disabled(mem_session, tmp_path):
    dep = _make_dep_with_rollback(mem_session, enabled=False, threshold=None)
    _make_deployment_version(mem_session, dep.id, 1, True, str(tmp_path / "p.joblib"))
    _make_deployment_version(mem_session, dep.id, 2, False, str(tmp_path / "p2.joblib"))
    _make_feedback(mem_session, dep.id, [False] * 15)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_accuracy_rollback

        _check_and_fire_accuracy_rollback(dep.id)
        mock_dispatch.assert_not_called()


def test_rollback_skips_when_threshold_none(mem_session, tmp_path):
    dep = _make_dep_with_rollback(mem_session, enabled=True, threshold=None)
    dep.auto_rollback_accuracy_threshold = None
    mem_session.add(dep)
    mem_session.commit()
    _make_deployment_version(mem_session, dep.id, 1, True, str(tmp_path / "p.joblib"))
    _make_deployment_version(mem_session, dep.id, 2, False, str(tmp_path / "p2.joblib"))
    _make_feedback(mem_session, dep.id, [False] * 15)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_accuracy_rollback

        _check_and_fire_accuracy_rollback(dep.id)
        mock_dispatch.assert_not_called()


def test_rollback_skips_insufficient_feedback(mem_session, tmp_path):
    dep = _make_dep_with_rollback(mem_session, enabled=True, threshold=0.80)
    pl = str(tmp_path / "pipeline.joblib")
    Path(pl).touch()
    _make_deployment_version(mem_session, dep.id, 1, True, pl)
    _make_deployment_version(mem_session, dep.id, 2, False, pl)
    _make_feedback(mem_session, dep.id, [False] * 5)  # Only 5, needs 10

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_accuracy_rollback

        _check_and_fire_accuracy_rollback(dep.id)
        mock_dispatch.assert_not_called()


def test_rollback_skips_when_accuracy_above_threshold(mem_session, tmp_path):
    dep = _make_dep_with_rollback(mem_session, enabled=True, threshold=0.80)
    pl = str(tmp_path / "pipeline.joblib")
    Path(pl).touch()
    _make_deployment_version(mem_session, dep.id, 1, True, pl)
    _make_deployment_version(mem_session, dep.id, 2, False, pl)
    # 90% accuracy — above 80% threshold
    _make_feedback(
        mem_session, dep.id, [True] * 9 + [False] * 1 + [True] * 5 + [True] * 5
    )

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_accuracy_rollback

        _check_and_fire_accuracy_rollback(dep.id)
        mock_dispatch.assert_not_called()


def test_rollback_skips_when_only_one_version(mem_session, tmp_path):
    dep = _make_dep_with_rollback(mem_session, enabled=True, threshold=0.80)
    pl = str(tmp_path / "pipeline.joblib")
    Path(pl).touch()
    _make_deployment_version(mem_session, dep.id, 1, True, pl)
    # Only 1 version — can't roll back
    _make_feedback(mem_session, dep.id, [False] * 15)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_accuracy_rollback

        _check_and_fire_accuracy_rollback(dep.id)
        mock_dispatch.assert_not_called()


def test_rollback_fires_when_accuracy_below_threshold(mem_session, tmp_path):
    dep = _make_dep_with_rollback(mem_session, enabled=True, threshold=0.80)
    pl = str(tmp_path / "pipeline.joblib")
    Path(pl).touch()
    pl2 = str(tmp_path / "pipeline2.joblib")
    Path(pl2).touch()
    _make_deployment_version(mem_session, dep.id, 1, False, pl)
    _make_deployment_version(mem_session, dep.id, 2, True, pl2)
    dep.current_version_number = 2
    mem_session.add(dep)
    mem_session.commit()
    # 50% accuracy — below 80% threshold
    _make_feedback(mem_session, dep.id, [True] * 5 + [False] * 5 + [False] * 5)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_accuracy_rollback

        _check_and_fire_accuracy_rollback(dep.id)
        mock_dispatch.assert_called_once()


def test_rollback_cooldown_prevents_repeat(mem_session, tmp_path):
    recent = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    dep = _make_dep_with_rollback(
        mem_session, enabled=True, threshold=0.80, last_triggered=recent
    )
    pl = str(tmp_path / "pipeline.joblib")
    Path(pl).touch()
    _make_deployment_version(mem_session, dep.id, 1, False, pl)
    _make_deployment_version(mem_session, dep.id, 2, True, pl)
    dep.current_version_number = 2
    mem_session.add(dep)
    mem_session.commit()
    _make_feedback(mem_session, dep.id, [False] * 15)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_fire_accuracy_rollback

        _check_and_fire_accuracy_rollback(dep.id)
        mock_dispatch.assert_not_called()


def test_rollback_payload_keys(mem_session, tmp_path):
    dep = _make_dep_with_rollback(mem_session, enabled=True, threshold=0.80)
    pl = str(tmp_path / "pipeline.joblib")
    Path(pl).touch()
    _make_deployment_version(mem_session, dep.id, 1, False, pl)
    _make_deployment_version(mem_session, dep.id, 2, True, pl)
    dep.current_version_number = 2
    mem_session.add(dep)
    mem_session.commit()
    _make_feedback(mem_session, dep.id, [False] * 15)

    dispatched = {}

    def capture(dep_id, event_type, payload):
        dispatched["event_type"] = event_type
        dispatched["payload"] = payload

    with patch("core.webhook.dispatch_webhooks", side_effect=capture):
        from api.deploy import _check_and_fire_accuracy_rollback

        _check_and_fire_accuracy_rollback(dep.id)

    assert dispatched.get("event_type") == EVENT_ROLLBACK_TRIGGERED
    payload = dispatched.get("payload", {})
    assert "accuracy_pct" in payload
    assert "threshold_pct" in payload
    assert "n_feedback" in payload
    assert "rolled_back_to_version" in payload
    assert "new_version_number" in payload
    assert "message" in payload


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_auto_rollback_enable(ac):
    dep_id = await _make_deployment(ac)
    resp = await ac.put(
        f"/api/deploy/{dep_id}/auto-rollback",
        json={"enabled": True, "accuracy_threshold_pct": 80.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["auto_rollback_enabled"] is True
    assert data["accuracy_threshold_pct"] == pytest.approx(80.0)
    assert data["cooldown_hours"] == 24
    assert data["min_feedback_required"] == 10


@pytest.mark.anyio
async def test_put_auto_rollback_disable(ac):
    dep_id = await _make_deployment(ac)

    # Enable first
    await ac.put(
        f"/api/deploy/{dep_id}/auto-rollback",
        json={"enabled": True, "accuracy_threshold_pct": 75.0},
    )

    # Then disable
    resp = await ac.put(
        f"/api/deploy/{dep_id}/auto-rollback",
        json={"enabled": False, "accuracy_threshold_pct": None},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["auto_rollback_enabled"] is False
    assert data["accuracy_threshold_pct"] is None


@pytest.mark.anyio
async def test_put_auto_rollback_invalid_threshold(ac):
    dep_id = await _make_deployment(ac)
    resp = await ac.put(
        f"/api/deploy/{dep_id}/auto-rollback",
        json={"enabled": True, "accuracy_threshold_pct": 150.0},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_put_auto_rollback_404(ac):
    resp = await ac.put(
        "/api/deploy/nonexistent-dep/auto-rollback",
        json={"enabled": True, "accuracy_threshold_pct": 80.0},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_auto_rollback_status(ac):
    dep_id = await _make_deployment(ac)
    await ac.put(
        f"/api/deploy/{dep_id}/auto-rollback",
        json={"enabled": True, "accuracy_threshold_pct": 85.0},
    )
    resp = await ac.get(f"/api/deploy/{dep_id}/auto-rollback-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["auto_rollback_enabled"] is True
    assert data["accuracy_threshold_pct"] == pytest.approx(85.0)


@pytest.mark.anyio
async def test_get_auto_rollback_status_404(ac):
    resp = await ac.get("/api/deploy/nonexistent-dep/auto-rollback-status")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Regex / NL tests
# ---------------------------------------------------------------------------


def test_auto_rollback_patterns_match():
    from api.chat import _AUTO_ROLLBACK_PATTERNS

    phrases = [
        "auto-rollback",
        "auto rollback",
        "automatic rollback",
        "automatically revert",
        "roll back if accuracy drops",
        "rollback if performance degrades",
        "enable auto-rollback",
        "configure auto rollback",
        "accuracy threshold for rollback",
        "status of auto-rollback",
        "disable auto rollback",
    ]
    for phrase in phrases:
        assert _AUTO_ROLLBACK_PATTERNS.search(phrase), f"Should match: {phrase}"


def test_disable_auto_rollback_re_matches():
    from api.chat import _DISABLE_AUTO_ROLLBACK_RE

    phrases = [
        "disable auto-rollback",
        "turn off auto rollback",
        "deactivate autorollback",
        "remove auto-rollback",
    ]
    for phrase in phrases:
        assert _DISABLE_AUTO_ROLLBACK_RE.search(phrase), f"Should match: {phrase}"


def test_status_auto_rollback_re_matches():
    from api.chat import _STATUS_AUTO_ROLLBACK_RE

    phrases = [
        "status of auto-rollback",
        "check auto rollback",
        "show my auto rollback",
    ]
    for phrase in phrases:
        assert _STATUS_AUTO_ROLLBACK_RE.search(phrase), f"Should match: {phrase}"


def test_auto_rollback_threshold_re_extracts():
    from api.chat import _AUTO_ROLLBACK_THRESHOLD_RE

    msg = "roll back if accuracy drops below 80%"
    m = _AUTO_ROLLBACK_THRESHOLD_RE.search(msg)
    assert m is not None
    assert float(m.group(1)) == pytest.approx(80.0)


def test_auto_rollback_threshold_re_extracts_no_symbol():
    from api.chat import _AUTO_ROLLBACK_THRESHOLD_RE

    msg = "enable auto-rollback at 75"
    m = _AUTO_ROLLBACK_THRESHOLD_RE.search(msg)
    assert m is not None
    assert float(m.group(1)) == pytest.approx(75.0)
