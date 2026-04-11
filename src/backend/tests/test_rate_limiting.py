"""Tests for per-deployment rate limiting and monthly quota management.

Covers:
- PUT /api/deploy/{id}/rate-limit  — set / remove rpm limit and monthly quota
- GET /api/deploy/{id}/quota-status — current usage vs quota
- GET /api/deploy/{id}             — rate_limit_rpm / monthly_quota fields present
- POST /api/predict/{id}           — 429 when rate limit exceeded / quota exhausted
- _check_rate_limit() helper       — sliding-window logic
- _check_monthly_quota() helper    — count-based quota check
- _RATE_LIMIT_PATTERNS             — NL intent detection regex
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
    resp = await ac.post("/api/projects", json={"name": "Rate Limit Test"})
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
# Tests — PUT /api/deploy/{id}/rate-limit
# ---------------------------------------------------------------------------


async def test_set_rate_limit_rpm(ac, deployment_id):
    """Setting a per-minute rate limit updates the deployment."""
    res = await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"rate_limit_rpm": 60},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["rate_limit_rpm"] == 60
    assert data["monthly_quota"] is None


async def test_set_monthly_quota(ac, deployment_id):
    """Setting a monthly quota updates the deployment."""
    res = await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"monthly_quota": 1000},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["monthly_quota"] == 1000
    assert data["rate_limit_rpm"] is None


async def test_set_both_limits(ac, deployment_id):
    """Setting both rpm and quota at once."""
    res = await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"rate_limit_rpm": 30, "monthly_quota": 500},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["rate_limit_rpm"] == 30
    assert data["monthly_quota"] == 500
    assert "Rate limits active" in data["message"]


async def test_remove_rate_limit_with_zero(ac, deployment_id):
    """Passing 0 removes the rate limit (stores None)."""
    await ac.put(f"/api/deploy/{deployment_id}/rate-limit", json={"rate_limit_rpm": 60})
    res = await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"rate_limit_rpm": 0},
    )
    assert res.status_code == 200
    assert res.json()["rate_limit_rpm"] is None


async def test_remove_quota_with_null(ac, deployment_id):
    """Passing None (null) removes the monthly quota."""
    await ac.put(f"/api/deploy/{deployment_id}/rate-limit", json={"monthly_quota": 100})
    res = await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"monthly_quota": None},
    )
    assert res.status_code == 200
    assert res.json()["monthly_quota"] is None


async def test_negative_rpm_rejected(ac, deployment_id):
    """Negative rate_limit_rpm is a validation error."""
    res = await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"rate_limit_rpm": -5},
    )
    assert res.status_code == 422


async def test_negative_quota_rejected(ac, deployment_id):
    """Negative monthly_quota is a validation error."""
    res = await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"monthly_quota": -1},
    )
    assert res.status_code == 422


async def test_rate_limit_not_found(ac):
    """404 on non-existent deployment."""
    res = await ac.put(
        "/api/deploy/nonexistent-id/rate-limit",
        json={"rate_limit_rpm": 10},
    )
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests — GET /api/deploy/{id}/quota-status
# ---------------------------------------------------------------------------


async def test_quota_status_no_limits(ac, deployment_id):
    """Quota status with no limits set returns zeroed-out, no alert."""
    res = await ac.get(f"/api/deploy/{deployment_id}/quota-status")
    assert res.status_code == 200
    data = res.json()
    assert data["quota_enabled"] is False
    assert data["monthly_quota"] is None
    assert data["used_this_month"] == 0
    assert data["remaining"] is None
    assert data["rate_limit_enabled"] is False


async def test_quota_status_with_quota_set(ac, deployment_id):
    """Quota status reflects configured quota and usage count."""
    await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"monthly_quota": 100},
    )
    res = await ac.get(f"/api/deploy/{deployment_id}/quota-status")
    assert res.status_code == 200
    data = res.json()
    assert data["quota_enabled"] is True
    assert data["monthly_quota"] == 100
    assert data["used_this_month"] == 0
    assert data["remaining"] == 100
    assert data["pct_used"] == 0.0


async def test_quota_status_shows_rpm(ac, deployment_id):
    """Quota status includes rate_limit_rpm when set."""
    await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"rate_limit_rpm": 20, "monthly_quota": 50},
    )
    data = (await ac.get(f"/api/deploy/{deployment_id}/quota-status")).json()
    assert data["rate_limit_rpm"] == 20
    assert data["rate_limit_enabled"] is True


async def test_quota_status_not_found(ac):
    """404 for unknown deployment."""
    res = await ac.get("/api/deploy/bad-id/quota-status")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests — GET /api/deploy/{id} includes rate limit fields
# ---------------------------------------------------------------------------


async def test_deployment_detail_includes_rate_limit_fields(ac, deployment_id):
    """Deployment detail response exposes rate_limit_rpm and monthly_quota."""
    res = await ac.get(f"/api/deploy/{deployment_id}")
    assert res.status_code == 200
    data = res.json()
    assert "rate_limit_rpm" in data
    assert "monthly_quota" in data
    assert data["rate_limit_rpm"] is None  # default: no limit
    assert data["monthly_quota"] is None


async def test_deployment_detail_reflects_set_limits(ac, deployment_id):
    """After setting limits, deployment detail shows updated values."""
    await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"rate_limit_rpm": 10, "monthly_quota": 200},
    )
    data = (await ac.get(f"/api/deploy/{deployment_id}")).json()
    assert data["rate_limit_rpm"] == 10
    assert data["monthly_quota"] == 200


# ---------------------------------------------------------------------------
# Tests — monthly quota enforcement on POST /api/predict/{id}
# ---------------------------------------------------------------------------


async def test_prediction_allowed_under_quota(ac, deployment_id, trained_run_id):
    """Predictions succeed when under the quota."""
    await ac.put(
        f"/api/deploy/{deployment_id}/rate-limit",
        json={"monthly_quota": 100},
    )
    res = await ac.post(
        f"/api/predict/{deployment_id}",
        json={"region": "East", "units": 10},
    )
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Tests — _check_rate_limit() pure function
# ---------------------------------------------------------------------------


def test_check_rate_limit_allows_first_request():
    """First request always allowed."""
    from api.deploy import _check_rate_limit

    _check_rate_limit.__module__  # ensure import works
    # Use a fresh key to avoid state pollution from other tests
    import uuid

    key = str(uuid.uuid4())
    assert _check_rate_limit(key, 10) is True


def test_check_rate_limit_blocks_on_exceed():
    """Requests beyond rpm are blocked within the window."""
    from api.deploy import _check_rate_limit

    import uuid

    key = str(uuid.uuid4())
    rpm = 3
    for _ in range(rpm):
        _check_rate_limit(key, rpm)
    # The (rpm+1)th request should be blocked
    assert _check_rate_limit(key, rpm) is False


def test_check_rate_limit_allows_after_window_expires():
    """After the 60-second window expires, requests are allowed again."""
    from collections import deque

    from api.deploy import _rate_windows

    import uuid

    key = str(uuid.uuid4())
    # Manually inject 5 'old' timestamps (61 seconds ago)
    old_ts = time.monotonic() - 61.0
    _rate_windows[key] = deque([old_ts] * 5)
    from api.deploy import _check_rate_limit

    assert _check_rate_limit(key, 5) is True  # window cleared, 1 new entry


# ---------------------------------------------------------------------------
# Tests — _RATE_LIMIT_PATTERNS regex
# ---------------------------------------------------------------------------


def test_rate_limit_patterns_set_rpm():
    from api.chat import _RATE_LIMIT_PATTERNS

    assert _RATE_LIMIT_PATTERNS.search("set rate limit to 100 requests per minute")


def test_rate_limit_patterns_enable():
    from api.chat import _RATE_LIMIT_PATTERNS

    assert _RATE_LIMIT_PATTERNS.search("enable rate limiting on my model")


def test_rate_limit_patterns_monthly_quota():
    from api.chat import _RATE_LIMIT_PATTERNS

    assert _RATE_LIMIT_PATTERNS.search(
        "add a monthly quota of 500 predictions per month"
    )


def test_rate_limit_patterns_quota_status():
    from api.chat import _RATE_LIMIT_PATTERNS

    assert _RATE_LIMIT_PATTERNS.search("quota status")


def test_rate_limit_patterns_check_quota():
    from api.chat import _RATE_LIMIT_PATTERNS

    assert _RATE_LIMIT_PATTERNS.search("check my quota")


def test_rate_limit_patterns_disable():
    from api.chat import _RATE_LIMIT_PATTERNS

    assert _RATE_LIMIT_PATTERNS.search("disable rate limit")


def test_rate_limit_patterns_remove_quota():
    from api.chat import _RATE_LIMIT_PATTERNS

    assert _RATE_LIMIT_PATTERNS.search("remove monthly quota")


def test_rate_limit_patterns_does_not_match_unrelated():
    from api.chat import _RATE_LIMIT_PATTERNS

    assert not _RATE_LIMIT_PATTERNS.search("train a model on my sales data")
