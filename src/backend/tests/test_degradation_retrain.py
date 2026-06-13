"""Tests for the Degradation-Triggered Auto-Retrain feature.

Covers:
- EVENT_DEGRADATION_RETRAIN constant exists in core.webhook
- ALL_EVENTS includes degradation_retrain
- Deployment model fields: degradation_retrain_enabled,
  degradation_retrain_accuracy_threshold, degradation_retrain_last_triggered_at
- _check_and_trigger_degradation_retrain: skips when disabled
- _check_and_trigger_degradation_retrain: skips when threshold is None
- _check_and_trigger_degradation_retrain: skips when fewer than min_feedback records
- _check_and_trigger_degradation_retrain: no retrain when accuracy >= threshold
- _check_and_trigger_degradation_retrain: cooldown prevents repeat within 24h
- _check_and_trigger_degradation_retrain: dispatches webhook with required keys
  when accuracy < threshold and training started
- PUT /api/deploy/{id}/degradation-retrain — enable with threshold
- PUT /api/deploy/{id}/degradation-retrain — disable (enabled=false)
- PUT /api/deploy/{id}/degradation-retrain — rejects threshold out of range
- PUT /api/deploy/{id}/degradation-retrain — rejects enabled=true without threshold
- PUT /api/deploy/{id}/degradation-retrain — 404 for missing deployment
- GET /api/deploy/{id}/degradation-retrain-status — returns config
- GET /api/deploy/{id}/degradation-retrain-status — 404 for missing deployment
- _DEGRADATION_RETRAIN_PATTERNS — matches NL enable/disable/status phrases
- _DISABLE_DEGRADATION_RETRAIN_RE — matches disable phrases
- _STATUS_DEGRADATION_RETRAIN_RE — matches status phrases
- _DEGRADATION_RETRAIN_THRESHOLD_RE — extracts numeric threshold
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from core.webhook import ALL_EVENTS, EVENT_DEGRADATION_RETRAIN

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
    import time

    proj_r = await ac.post("/api/projects", json={"name": "DRTest"})
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


def test_event_degradation_retrain_constant():
    assert EVENT_DEGRADATION_RETRAIN == "degradation_retrain"


def test_degradation_retrain_in_all_events():
    assert EVENT_DEGRADATION_RETRAIN in ALL_EVENTS


def test_deployment_model_fields():
    from models.deployment import Deployment

    d = Deployment(
        model_run_id="r1",
        project_id="p1",
        endpoint_path="/api/predict/d1",
        dashboard_url="/predict/d1",
        degradation_retrain_enabled=True,
        degradation_retrain_accuracy_threshold=0.75,
        degradation_retrain_last_triggered_at=None,
    )
    assert d.degradation_retrain_enabled is True
    assert d.degradation_retrain_accuracy_threshold == pytest.approx(0.75)
    assert d.degradation_retrain_last_triggered_at is None


# ---------------------------------------------------------------------------
# _check_and_trigger_degradation_retrain unit tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem_session(tmp_path):
    test_db = str(tmp_path / "dr_test.db")
    engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.engine = engine

    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.deployment_version  # noqa
    import models.feedback_record  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.conversation  # noqa
    import models.dataset_filter  # noqa
    import models.webhook_config  # noqa
    import models.webhook_event  # noqa
    import models.weekly_digest_config  # noqa

    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _make_dep_with_dr(session, enabled=True, threshold=0.75, last_triggered=None):
    from models.deployment import Deployment

    dep = Deployment(
        model_run_id="run-dr-001",
        project_id="proj-dr-001",
        endpoint_path="/api/predict/dr-dep",
        dashboard_url="/predict/dr-dep",
        degradation_retrain_enabled=enabled,
        degradation_retrain_accuracy_threshold=threshold if enabled else None,
        degradation_retrain_last_triggered_at=last_triggered,
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)
    return dep


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


def test_dr_skips_when_disabled(mem_session):
    dep = _make_dep_with_dr(mem_session, enabled=False, threshold=None)
    _make_feedback(mem_session, dep.id, [False] * 15)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_trigger_degradation_retrain

        _check_and_trigger_degradation_retrain(dep.id)
        mock_dispatch.assert_not_called()


def test_dr_skips_when_threshold_none(mem_session):
    dep = _make_dep_with_dr(mem_session, enabled=True, threshold=0.75)
    dep.degradation_retrain_accuracy_threshold = None
    mem_session.add(dep)
    mem_session.commit()
    _make_feedback(mem_session, dep.id, [False] * 15)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_trigger_degradation_retrain

        _check_and_trigger_degradation_retrain(dep.id)
        mock_dispatch.assert_not_called()


def test_dr_skips_insufficient_feedback(mem_session):
    dep = _make_dep_with_dr(mem_session, enabled=True, threshold=0.75)
    _make_feedback(mem_session, dep.id, [False] * 5)  # Only 5, needs 10

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_trigger_degradation_retrain

        _check_and_trigger_degradation_retrain(dep.id)
        mock_dispatch.assert_not_called()


def test_dr_skips_when_accuracy_above_threshold(mem_session):
    dep = _make_dep_with_dr(mem_session, enabled=True, threshold=0.75)
    # 90% accuracy — above 75% threshold
    _make_feedback(mem_session, dep.id, [True] * 9 + [False] * 1 + [True] * 5)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_trigger_degradation_retrain

        _check_and_trigger_degradation_retrain(dep.id)
        mock_dispatch.assert_not_called()


def test_dr_cooldown_prevents_repeat(mem_session):
    recent = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1)
    dep = _make_dep_with_dr(
        mem_session, enabled=True, threshold=0.75, last_triggered=recent
    )
    _make_feedback(mem_session, dep.id, [False] * 15)

    with patch("core.webhook.dispatch_webhooks") as mock_dispatch:
        from api.deploy import _check_and_trigger_degradation_retrain

        _check_and_trigger_degradation_retrain(dep.id)
        mock_dispatch.assert_not_called()


def test_dr_fires_when_accuracy_below_threshold(mem_session, tmp_path):
    """When accuracy < threshold and model run context is available, webhook fires."""
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("revenue,units\n100,10\n200,20\n300,30\n400,40\n500,50\n")

    # Create Dataset
    ds = Dataset(
        project_id="proj-dr-001",
        filename="data.csv",
        file_path=str(csv_path),
        row_count=5,
        column_count=2,
        size_bytes=100,
    )
    mem_session.add(ds)
    mem_session.commit()
    mem_session.refresh(ds)

    # Create FeatureSet
    fs = FeatureSet(
        dataset_id=ds.id,
        target_column="revenue",
        problem_type="regression",
        transformations="[]",
        is_active=True,
    )
    mem_session.add(fs)
    mem_session.commit()
    mem_session.refresh(fs)

    # Create ModelRun
    run = ModelRun(
        project_id="proj-dr-001",
        feature_set_id=fs.id,
        algorithm="linear_regression",
        hyperparameters="{}",
        status="done",
    )
    mem_session.add(run)
    mem_session.commit()
    mem_session.refresh(run)

    dep = _make_dep_with_dr(mem_session, enabled=True, threshold=0.75)
    dep.model_run_id = run.id
    dep.algorithm = "linear_regression"
    mem_session.add(dep)
    mem_session.commit()

    # 40% accuracy — below 75% threshold
    _make_feedback(mem_session, dep.id, [True] * 4 + [False] * 6 + [False] * 5)

    dispatched = {}

    def capture(dep_id, event_type, payload):
        dispatched["event_type"] = event_type
        dispatched["payload"] = payload

    with (
        patch("core.webhook.dispatch_webhooks", side_effect=capture),
        patch("api.models._train_in_background"),
        patch("api.models._training_queues", {}),
        patch("api.models._training_counters", {}),
    ):
        from api.deploy import _check_and_trigger_degradation_retrain

        _check_and_trigger_degradation_retrain(dep.id)

    assert dispatched.get("event_type") == EVENT_DEGRADATION_RETRAIN
    payload = dispatched.get("payload", {})
    assert "accuracy_pct" in payload
    assert "threshold_pct" in payload
    assert "n_feedback" in payload
    assert "new_run_id" in payload
    assert "algorithm" in payload
    assert "message" in payload


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_degradation_retrain_enable(ac):
    dep_id = await _make_deployment(ac)
    resp = await ac.put(
        f"/api/deploy/{dep_id}/degradation-retrain",
        json={"enabled": True, "accuracy_threshold_pct": 75.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["degradation_retrain_enabled"] is True
    assert data["accuracy_threshold_pct"] == pytest.approx(75.0)
    assert data["cooldown_hours"] == 24
    assert data["min_feedback_required"] == 10


@pytest.mark.anyio
async def test_put_degradation_retrain_disable(ac):
    dep_id = await _make_deployment(ac)

    await ac.put(
        f"/api/deploy/{dep_id}/degradation-retrain",
        json={"enabled": True, "accuracy_threshold_pct": 80.0},
    )

    resp = await ac.put(
        f"/api/deploy/{dep_id}/degradation-retrain",
        json={"enabled": False, "accuracy_threshold_pct": None},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["degradation_retrain_enabled"] is False
    assert data["accuracy_threshold_pct"] is None


@pytest.mark.anyio
async def test_put_degradation_retrain_invalid_threshold(ac):
    dep_id = await _make_deployment(ac)
    resp = await ac.put(
        f"/api/deploy/{dep_id}/degradation-retrain",
        json={"enabled": True, "accuracy_threshold_pct": 150.0},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_put_degradation_retrain_no_threshold(ac):
    dep_id = await _make_deployment(ac)
    resp = await ac.put(
        f"/api/deploy/{dep_id}/degradation-retrain",
        json={"enabled": True, "accuracy_threshold_pct": None},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_put_degradation_retrain_404(ac):
    resp = await ac.put(
        "/api/deploy/nonexistent-dep/degradation-retrain",
        json={"enabled": True, "accuracy_threshold_pct": 75.0},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_degradation_retrain_status(ac):
    dep_id = await _make_deployment(ac)
    await ac.put(
        f"/api/deploy/{dep_id}/degradation-retrain",
        json={"enabled": True, "accuracy_threshold_pct": 80.0},
    )
    resp = await ac.get(f"/api/deploy/{dep_id}/degradation-retrain-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["degradation_retrain_enabled"] is True
    assert data["accuracy_threshold_pct"] == pytest.approx(80.0)


@pytest.mark.anyio
async def test_get_degradation_retrain_status_404(ac):
    resp = await ac.get("/api/deploy/nonexistent-dep/degradation-retrain-status")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Regex / NL tests
# ---------------------------------------------------------------------------


def test_degradation_retrain_patterns_match():
    from api.chat import _DEGRADATION_RETRAIN_PATTERNS

    phrases = [
        "enable auto-retraining when accuracy drops below 80%",
        "configure degradation-triggered retraining",
        "retrain automatically when performance degrades",
        "trigger retraining if accuracy falls below 75%",
        "automatic retraining on degradation",
        "retrain on degradation",
        "disable degradation retraining",
        "disable auto retraining",
        "status of degradation retraining",
        "degradation-triggered retrain",
    ]
    for phrase in phrases:
        assert _DEGRADATION_RETRAIN_PATTERNS.search(phrase), f"Should match: {phrase!r}"


def test_degradation_retrain_patterns_no_false_positives():
    from api.chat import _DEGRADATION_RETRAIN_PATTERNS

    phrases = [
        "show me my predictions",
        "what is my model accuracy",
        "train a new model",
    ]
    for phrase in phrases:
        assert not _DEGRADATION_RETRAIN_PATTERNS.search(
            phrase
        ), f"Should not match: {phrase!r}"


def test_disable_degradation_retrain_re_matches():
    from api.chat import _DISABLE_DEGRADATION_RETRAIN_RE

    phrases = [
        "disable degradation retraining",
        "turn off auto retraining",
        "deactivate degradation-triggered retrain",
        "stop auto retraining",
    ]
    for phrase in phrases:
        assert _DISABLE_DEGRADATION_RETRAIN_RE.search(
            phrase
        ), f"Should match: {phrase!r}"


def test_status_degradation_retrain_re_matches():
    from api.chat import _STATUS_DEGRADATION_RETRAIN_RE

    phrases = [
        "status of degradation retraining",
        "check degradation retrain",
        "show degradation retraining config",
    ]
    for phrase in phrases:
        assert _STATUS_DEGRADATION_RETRAIN_RE.search(
            phrase
        ), f"Should match: {phrase!r}"


def test_degradation_retrain_threshold_re_extracts():
    from api.chat import _DEGRADATION_RETRAIN_THRESHOLD_RE

    msg = "retrain automatically when accuracy drops below 75%"
    m = _DEGRADATION_RETRAIN_THRESHOLD_RE.search(msg)
    assert m is not None
    assert float(m.group(1)) == pytest.approx(75.0)


def test_degradation_retrain_threshold_re_no_symbol():
    from api.chat import _DEGRADATION_RETRAIN_THRESHOLD_RE

    msg = "trigger retraining if accuracy falls below 80"
    m = _DEGRADATION_RETRAIN_THRESHOLD_RE.search(msg)
    assert m is not None
    assert float(m.group(1)) == pytest.approx(80.0)
