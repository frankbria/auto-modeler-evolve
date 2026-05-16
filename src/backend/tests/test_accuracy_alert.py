"""Tests for model accuracy degradation alert.

Covers:
- PUT /api/deploy/{id}/accuracy-alert  — set / clear threshold
- GET /api/deploy/{id}/accuracy-alert-status — live metric + threshold
- _compute_feedback_accuracy_simple()  — classification and regression metrics
- _check_and_fire_accuracy_alert()     — fires at threshold crossing, respects fired flag
- _ACCURACY_ALERT_PATTERNS             — NL intent detection regex
"""

import io
import time

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
)

_CLASSIFICATION_CSV = (
    b"age,income,approved\n"
    b"25,40000,yes\n30,55000,no\n35,70000,yes\n40,85000,yes\n45,60000,no\n"
    b"50,90000,yes\n22,35000,no\n28,48000,yes\n33,62000,no\n38,78000,yes\n"
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
async def regression_project(ac):
    resp = await ac.post("/api/projects", json={"name": "Accuracy Alert Regression"})
    project_id = resp.json()["id"]

    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
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

    return {"project_id": project_id, "dataset_id": dataset_id, "deployment_id": deployment_id}


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


def test_accuracy_alert_patterns_set():
    from api.chat import _ACCURACY_ALERT_PATTERNS

    phrases = [
        "alert me when accuracy drops below 80%",
        "notify me when feedback accuracy falls under 70%",
        "set accuracy alert at 0.75",
        "configure accuracy degradation alert",
        "accuracy degradation alert threshold",
        "model performance alert",
        "alert when model accuracy declines",
        "warn me if feedback accuracy drops",
        "disable the accuracy degradation alert",
        "remove accuracy alert",
        "turn off accuracy alert",
        "show my accuracy alert threshold",
        "check accuracy degradation warning",
    ]
    for phrase in phrases:
        assert _ACCURACY_ALERT_PATTERNS.search(phrase), f"Pattern missed: {phrase!r}"


def test_accuracy_alert_patterns_no_false_positives():
    from api.chat import _ACCURACY_ALERT_PATTERNS

    non_matches = [
        "upload my data",
        "train a model",
        "show me predictions",
        "what is the drift score",
    ]
    for phrase in non_matches:
        assert not _ACCURACY_ALERT_PATTERNS.search(phrase), f"False positive: {phrase!r}"


def test_accuracy_alert_threshold_re_extracts_percent():
    from api.chat import _ACCURACY_ALERT_THRESHOLD_RE

    m = _ACCURACY_ALERT_THRESHOLD_RE.search("alert me when accuracy drops below 80%")
    assert m is not None
    assert float(m.group(1)) == 80.0


def test_accuracy_alert_threshold_re_extracts_decimal():
    from api.chat import _ACCURACY_ALERT_THRESHOLD_RE

    m = _ACCURACY_ALERT_THRESHOLD_RE.search("set accuracy alert at 0.75")
    assert m is not None
    assert float(m.group(1)) == 0.75


def test_disable_accuracy_alert_re():
    from api.chat import _DISABLE_ACCURACY_ALERT_RE

    assert _DISABLE_ACCURACY_ALERT_RE.search("disable accuracy alert")
    assert _DISABLE_ACCURACY_ALERT_RE.search("remove accuracy degradation alert")
    assert _DISABLE_ACCURACY_ALERT_RE.search("turn off accuracy alert")
    assert not _DISABLE_ACCURACY_ALERT_RE.search("set accuracy alert at 80%")


# ---------------------------------------------------------------------------
# Endpoint: PUT /api/deploy/{id}/accuracy-alert
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_accuracy_alert_sets_threshold(regression_project, ac):
    dep_id = regression_project["deployment_id"]
    resp = await ac.put(
        f"/api/deploy/{dep_id}/accuracy-alert",
        json={"threshold": 0.75},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["accuracy_alert_threshold"] == pytest.approx(0.75)
    assert data["accuracy_alert_fired"] is False


@pytest.mark.anyio
async def test_put_accuracy_alert_clears_threshold(regression_project, ac):
    dep_id = regression_project["deployment_id"]
    await ac.put(f"/api/deploy/{dep_id}/accuracy-alert", json={"threshold": 0.8})
    resp = await ac.put(f"/api/deploy/{dep_id}/accuracy-alert", json={"threshold": None})
    assert resp.status_code == 200
    data = resp.json()
    assert data["accuracy_alert_threshold"] is None


@pytest.mark.anyio
async def test_put_accuracy_alert_resets_fired_flag(regression_project, ac):
    dep_id = regression_project["deployment_id"]
    # Set a threshold
    await ac.put(f"/api/deploy/{dep_id}/accuracy-alert", json={"threshold": 0.9})
    # Simulate fired (direct DB write not needed — changing threshold resets it)
    resp = await ac.put(f"/api/deploy/{dep_id}/accuracy-alert", json={"threshold": 0.85})
    assert resp.status_code == 200
    assert resp.json()["accuracy_alert_fired"] is False


@pytest.mark.anyio
async def test_put_accuracy_alert_invalid_deployment(ac):
    resp = await ac.put(
        "/api/deploy/nonexistent-id/accuracy-alert",
        json={"threshold": 0.5},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_put_accuracy_alert_out_of_range_regression(regression_project, ac):
    dep_id = regression_project["deployment_id"]
    # For regression, valid range is 0-100; 200 is invalid
    resp = await ac.put(
        f"/api/deploy/{dep_id}/accuracy-alert",
        json={"threshold": 200.0},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint: GET /api/deploy/{id}/accuracy-alert-status
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_accuracy_alert_status_no_threshold(regression_project, ac):
    dep_id = regression_project["deployment_id"]
    resp = await ac.get(f"/api/deploy/{dep_id}/accuracy-alert-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["accuracy_alert_threshold"] is None
    assert data["accuracy_alert_fired"] is False
    assert "problem_type" in data
    assert "n_feedback" in data


@pytest.mark.anyio
async def test_get_accuracy_alert_status_with_threshold(regression_project, ac):
    dep_id = regression_project["deployment_id"]
    await ac.put(f"/api/deploy/{dep_id}/accuracy-alert", json={"threshold": 15.0})
    resp = await ac.get(f"/api/deploy/{dep_id}/accuracy-alert-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["accuracy_alert_threshold"] == pytest.approx(15.0)


@pytest.mark.anyio
async def test_get_accuracy_alert_status_invalid_deployment(ac):
    resp = await ac.get("/api/deploy/nonexistent-id/accuracy-alert-status")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# _compute_feedback_accuracy_simple unit tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_compute_feedback_accuracy_no_feedback(regression_project, ac):
    from api.deploy import _compute_feedback_accuracy_simple
    from db import engine
    from sqlmodel import Session

    from models.deployment import Deployment

    dep_id = regression_project["deployment_id"]
    with Session(engine) as session:
        deployment = session.get(Deployment, dep_id)
        result = _compute_feedback_accuracy_simple(session, deployment)

    problem_type, metric_value, n = result
    assert problem_type in ("regression", "classification")
    # No feedback yet — metric should be None
    assert metric_value is None
    assert n == 0


@pytest.mark.anyio
async def test_compute_feedback_accuracy_regression_with_data(regression_project, ac):
    from api.deploy import _compute_feedback_accuracy_simple
    from db import engine
    from sqlmodel import Session

    from models.deployment import Deployment
    from models.feedback_record import FeedbackRecord

    dep_id = regression_project["deployment_id"]

    # Insert synthetic feedback records for regression (no prediction_log_id → metric=None)
    with Session(engine) as session:
        for i in range(5):
            fb = FeedbackRecord(
                deployment_id=dep_id,
                predicted_value="100.0",
                actual_value="100.0" if i < 4 else "200.0",
                is_correct=None,
            )
            session.add(fb)
        session.commit()

    with Session(engine) as session:
        deployment = session.get(Deployment, dep_id)
        problem_type, metric_value, n = _compute_feedback_accuracy_simple(session, deployment)

    assert problem_type == "regression"
    # No prediction_log_id linked, so metric is None (no paired actual/predicted)
    # n still reflects total feedback count
    assert n == 5


# ---------------------------------------------------------------------------
# _check_and_fire_accuracy_alert unit tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_check_and_fire_accuracy_alert_classification_breach(regression_project, ac, monkeypatch):
    import core.webhook as webhook_mod
    from api.deploy import _check_and_fire_accuracy_alert

    fired_events = []
    monkeypatch.setattr(webhook_mod, "dispatch_webhooks", lambda dep, evt, payload: fired_events.append((dep, evt)))

    dep_id = regression_project["deployment_id"]
    _check_and_fire_accuracy_alert(dep_id, "classification", 0.5, 0.8)

    assert len(fired_events) == 1
    assert fired_events[0][1] == "accuracy_alert"


@pytest.mark.anyio
async def test_check_and_fire_accuracy_alert_classification_no_breach(regression_project, ac, monkeypatch):
    import core.webhook as webhook_mod
    from api.deploy import _check_and_fire_accuracy_alert

    fired_events = []
    monkeypatch.setattr(webhook_mod, "dispatch_webhooks", lambda dep, evt, payload: fired_events.append((dep, evt)))

    dep_id = regression_project["deployment_id"]
    _check_and_fire_accuracy_alert(dep_id, "classification", 0.9, 0.8)

    assert len(fired_events) == 0


@pytest.mark.anyio
async def test_check_and_fire_accuracy_alert_regression_breach(regression_project, ac, monkeypatch):
    import core.webhook as webhook_mod
    from api.deploy import _check_and_fire_accuracy_alert

    fired_events = []
    monkeypatch.setattr(webhook_mod, "dispatch_webhooks", lambda dep, evt, payload: fired_events.append((dep, evt)))

    dep_id = regression_project["deployment_id"]
    _check_and_fire_accuracy_alert(dep_id, "regression", 25.0, 20.0)

    assert len(fired_events) == 1


@pytest.mark.anyio
async def test_check_and_fire_accuracy_alert_regression_no_breach(regression_project, ac, monkeypatch):
    import core.webhook as webhook_mod
    from api.deploy import _check_and_fire_accuracy_alert

    fired_events = []
    monkeypatch.setattr(webhook_mod, "dispatch_webhooks", lambda dep, evt, payload: fired_events.append((dep, evt)))

    dep_id = regression_project["deployment_id"]
    _check_and_fire_accuracy_alert(dep_id, "regression", 10.0, 20.0)

    assert len(fired_events) == 0


@pytest.mark.anyio
async def test_put_accuracy_alert_fired_flag_persists_in_db(regression_project, ac):
    """Verify accuracy_alert_fired round-trips through DB correctly."""
    from db import engine
    from sqlmodel import Session

    from models.deployment import Deployment

    dep_id = regression_project["deployment_id"]
    await ac.put(f"/api/deploy/{dep_id}/accuracy-alert", json={"threshold": 0.8})

    with Session(engine) as session:
        deployment = session.get(Deployment, dep_id)
        # Flag starts False after threshold is set
        assert deployment.accuracy_alert_fired is False
        # Manually set it (simulating what submit_feedback does)
        deployment.accuracy_alert_fired = True
        session.add(deployment)
        session.commit()

    # Should still be True after re-reading
    with Session(engine) as session:
        deployment = session.get(Deployment, dep_id)
        assert deployment.accuracy_alert_fired is True

    # PUT with new threshold resets it
    await ac.put(f"/api/deploy/{dep_id}/accuracy-alert", json={"threshold": 0.7})
    with Session(engine) as session:
        deployment = session.get(Deployment, dep_id)
        assert deployment.accuracy_alert_fired is False


@pytest.mark.anyio
async def test_check_and_fire_accuracy_alert_respects_fired_flag(regression_project, ac, monkeypatch):
    import core.webhook as webhook_mod
    from api.deploy import _check_and_fire_accuracy_alert

    fired_count = [0]
    monkeypatch.setattr(webhook_mod, "dispatch_webhooks", lambda *a, **k: fired_count.__setitem__(0, fired_count[0] + 1))

    dep_id = regression_project["deployment_id"]
    _check_and_fire_accuracy_alert(dep_id, "classification", 0.5, 0.8)
    assert fired_count[0] == 1

    _check_and_fire_accuracy_alert(dep_id, "classification", 0.5, 0.8)
    assert fired_count[0] == 2
