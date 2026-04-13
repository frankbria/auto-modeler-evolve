"""Tests for quota alert threshold configuration.

Covers:
- PUT /api/deploy/{id}/quota-alert  — set / remove alert threshold
- GET /api/deploy/{id}/quota-status — includes quota_alert_threshold_pct field
- _check_and_fire_quota_alert()     — fires exactly at threshold crossing
- _QUOTA_ALERT_PATTERNS             — NL intent detection regex
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
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Quota Alert Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


@pytest.fixture()
async def feature_set_id(ac, dataset_id):
    resp = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert resp.status_code == 201, resp.text
    fs_id = resp.json()["feature_set_id"]
    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


@pytest.fixture()
async def trained_run_id(ac, project_id, feature_set_id):
    resp = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["model_run_ids"][0]
    for _ in range(30):
        r = await ac.get(f"/api/models/{project_id}/runs")
        runs = r.json().get("runs", [])
        run = next((x for x in runs if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.3)
    pytest.skip("Training did not complete")


@pytest.fixture()
async def deployment_id(ac, trained_run_id):
    resp = await ac.post(f"/api/deploy/{trained_run_id}", json={})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Tests — PUT /api/deploy/{id}/quota-alert
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_set_quota_alert_80pct(ac, deployment_id):
    resp = await ac.put(
        f"/api/deploy/{deployment_id}/quota-alert",
        json={"threshold_pct": 80},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["quota_alert_enabled"] is True
    assert data["quota_alert_threshold_pct"] == 80


@pytest.mark.anyio
async def test_set_quota_alert_90pct(ac, deployment_id):
    resp = await ac.put(
        f"/api/deploy/{deployment_id}/quota-alert",
        json={"threshold_pct": 90},
    )
    assert resp.status_code == 200
    assert resp.json()["quota_alert_threshold_pct"] == 90


@pytest.mark.anyio
async def test_remove_quota_alert_with_zero(ac, deployment_id):
    # Set it first
    await ac.put(f"/api/deploy/{deployment_id}/quota-alert", json={"threshold_pct": 80})
    # Then remove
    resp = await ac.put(
        f"/api/deploy/{deployment_id}/quota-alert",
        json={"threshold_pct": 0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["quota_alert_enabled"] is False
    assert data["quota_alert_threshold_pct"] is None


@pytest.mark.anyio
async def test_remove_quota_alert_with_null(ac, deployment_id):
    await ac.put(f"/api/deploy/{deployment_id}/quota-alert", json={"threshold_pct": 80})
    resp = await ac.put(
        f"/api/deploy/{deployment_id}/quota-alert",
        json={"threshold_pct": None},
    )
    assert resp.status_code == 200
    assert resp.json()["quota_alert_enabled"] is False


@pytest.mark.anyio
async def test_quota_alert_invalid_negative(ac, deployment_id):
    resp = await ac.put(
        f"/api/deploy/{deployment_id}/quota-alert",
        json={"threshold_pct": -5},
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_quota_alert_invalid_over_99(ac, deployment_id):
    resp = await ac.put(
        f"/api/deploy/{deployment_id}/quota-alert",
        json={"threshold_pct": 100},
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_quota_alert_not_found(ac):
    resp = await ac.put(
        "/api/deploy/nonexistent-id/quota-alert",
        json={"threshold_pct": 80},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests — GET /api/deploy/{id}/quota-status includes alert fields
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_quota_status_includes_alert_field_disabled(ac, deployment_id):
    resp = await ac.get(f"/api/deploy/{deployment_id}/quota-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "quota_alert_threshold_pct" in data
    assert "quota_alert_enabled" in data
    assert data["quota_alert_enabled"] is False
    assert data["quota_alert_threshold_pct"] is None


@pytest.mark.anyio
async def test_quota_status_reflects_set_alert(ac, deployment_id):
    await ac.put(f"/api/deploy/{deployment_id}/quota-alert", json={"threshold_pct": 75})
    resp = await ac.get(f"/api/deploy/{deployment_id}/quota-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["quota_alert_threshold_pct"] == 75
    assert data["quota_alert_enabled"] is True


# ---------------------------------------------------------------------------
# Tests — _check_and_fire_quota_alert() pure logic
# ---------------------------------------------------------------------------


def test_alert_fires_exactly_at_crossing():
    """Alert fires when used == ceil(quota * threshold / 100) — the crossing prediction."""
    import math
    from unittest.mock import patch

    from api.deploy import _check_and_fire_quota_alert

    quota = 100
    threshold = 80
    crossing_count = math.ceil(quota * threshold / 100)  # = 80

    with patch("api.deploy._check_and_fire_quota_alert.__module__") as _:
        pass  # ensure module loaded

    fired = []

    def fake_dispatch(dep_id, event_type, payload):
        fired.append((dep_id, event_type, payload))

    with patch("core.webhook.dispatch_webhooks", fake_dispatch):
        # One prediction below threshold — should NOT fire
        _check_and_fire_quota_alert("dep1", crossing_count - 1, quota, threshold)
        assert len(fired) == 0

        # Exactly at crossing — SHOULD fire
        _check_and_fire_quota_alert("dep1", crossing_count, quota, threshold)
        assert len(fired) == 1
        assert fired[0][1] == "quota_alert"

        # One above crossing — should NOT fire again
        _check_and_fire_quota_alert("dep1", crossing_count + 1, quota, threshold)
        assert len(fired) == 1  # still 1 — no second alert


def test_alert_does_not_fire_when_below_threshold():
    from unittest.mock import patch

    from api.deploy import _check_and_fire_quota_alert

    fired = []

    def fake_dispatch(dep_id, event_type, payload):
        fired.append(event_type)

    with patch("core.webhook.dispatch_webhooks", fake_dispatch):
        _check_and_fire_quota_alert("dep1", 10, 100, 80)  # 10% used, threshold 80%
        assert len(fired) == 0


def test_alert_payload_contains_expected_fields():
    import math
    from unittest.mock import patch

    from api.deploy import _check_and_fire_quota_alert

    quota = 200
    threshold = 75
    crossing = math.ceil(quota * threshold / 100)

    captured = []

    def fake_dispatch(dep_id, event_type, payload):
        captured.append(payload)

    with patch("core.webhook.dispatch_webhooks", fake_dispatch):
        _check_and_fire_quota_alert("dep-abc", crossing, quota, threshold)

    assert len(captured) == 1
    p = captured[0]
    assert p["deployment_id"] == "dep-abc"
    assert p["monthly_quota"] == quota
    assert p["threshold_pct"] == threshold
    assert p["used_this_month"] == crossing
    assert "message" in p


# ---------------------------------------------------------------------------
# Tests — _QUOTA_ALERT_PATTERNS regex
# ---------------------------------------------------------------------------


def test_quota_alert_patterns_positive_alert_me():
    from api.chat import _QUOTA_ALERT_PATTERNS

    assert _QUOTA_ALERT_PATTERNS.search("alert me when I hit 80% of my quota")


def test_quota_alert_patterns_positive_notify():
    from api.chat import _QUOTA_ALERT_PATTERNS

    assert _QUOTA_ALERT_PATTERNS.search("notify me when my quota is almost full")


def test_quota_alert_patterns_positive_set_at():
    from api.chat import _QUOTA_ALERT_PATTERNS

    assert _QUOTA_ALERT_PATTERNS.search("set a quota alert at 90%")


def test_quota_alert_patterns_positive_configure():
    from api.chat import _QUOTA_ALERT_PATTERNS

    assert _QUOTA_ALERT_PATTERNS.search("configure quota alert")


def test_quota_alert_patterns_positive_warn_at():
    from api.chat import _QUOTA_ALERT_PATTERNS

    assert _QUOTA_ALERT_PATTERNS.search("alert me at 85")


def test_quota_alert_patterns_positive_disable():
    from api.chat import _QUOTA_ALERT_PATTERNS

    assert _QUOTA_ALERT_PATTERNS.search("disable the quota alert")


def test_quota_alert_patterns_positive_threshold():
    from api.chat import _QUOTA_ALERT_PATTERNS

    assert _QUOTA_ALERT_PATTERNS.search("quota alert threshold")


def test_quota_alert_patterns_negative_model_accuracy():
    from api.chat import _QUOTA_ALERT_PATTERNS

    assert not _QUOTA_ALERT_PATTERNS.search("what is my model accuracy?")


def test_quota_alert_patterns_negative_train():
    from api.chat import _QUOTA_ALERT_PATTERNS

    assert not _QUOTA_ALERT_PATTERNS.search("train a new model on my data")
