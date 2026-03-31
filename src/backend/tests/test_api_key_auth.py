"""Tests for API key authentication on prediction endpoints.

Covers:
- POST /api/deploy/{id}/api-key  — generate / regenerate
- DELETE /api/deploy/{id}/api-key — disable protection
- GET  /api/deploy/{id}          — api_key_enabled field present
- POST /api/predict/{id}         — rejected without key, accepted with valid key
- POST /api/predict/{id}/batch   — same auth behaviour
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
    resp = await ac.post("/api/projects", json={"name": "API Key Test"})
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
# Tests — key generation
# ---------------------------------------------------------------------------


async def test_generate_api_key_returns_plaintext(ac, deployment_id):
    """POST /api/deploy/{id}/api-key returns a usable plaintext key."""
    res = await ac.post(f"/api/deploy/{deployment_id}/api-key")
    assert res.status_code == 201
    data = res.json()
    assert "api_key" in data
    assert len(data["api_key"]) > 20
    assert data["deployment_id"] == deployment_id
    assert "message" in data


async def test_fresh_deployment_has_no_key_protection(ac, deployment_id):
    """Freshly deployed model has api_key_enabled=False in response."""
    res = await ac.get(f"/api/deploy/{deployment_id}")
    assert res.json()["api_key_enabled"] is False


async def test_deployment_shows_key_enabled_after_generation(ac, deployment_id):
    """After key generation, deployment detail shows api_key_enabled=True."""
    await ac.post(f"/api/deploy/{deployment_id}/api-key")
    res = await ac.get(f"/api/deploy/{deployment_id}")
    assert res.json()["api_key_enabled"] is True


async def test_regenerate_produces_different_key(ac, deployment_id):
    """Calling key generation twice produces a different key each time."""
    first = (await ac.post(f"/api/deploy/{deployment_id}/api-key")).json()["api_key"]
    second = (await ac.post(f"/api/deploy/{deployment_id}/api-key")).json()["api_key"]
    assert first != second


# ---------------------------------------------------------------------------
# Tests — auth enforcement on predict endpoint
# ---------------------------------------------------------------------------


async def test_predict_open_deployment_no_key_needed(ac, deployment_id):
    """Unprotected deployment accepts requests without Authorization header."""
    res = await ac.post(
        f"/api/predict/{deployment_id}", json={"units": 25, "region": "East"}
    )
    assert res.status_code == 200
    assert "prediction" in res.json()


async def test_predict_blocked_no_header(ac, deployment_id):
    """Protected deployment rejects requests with no Authorization header."""
    await ac.post(f"/api/deploy/{deployment_id}/api-key")
    res = await ac.post(
        f"/api/predict/{deployment_id}", json={"units": 25, "region": "East"}
    )
    assert res.status_code == 401


async def test_predict_blocked_wrong_key(ac, deployment_id):
    """Protected deployment rejects requests with an incorrect key."""
    await ac.post(f"/api/deploy/{deployment_id}/api-key")
    res = await ac.post(
        f"/api/predict/{deployment_id}",
        json={"units": 25, "region": "East"},
        headers={"Authorization": "Bearer totally-wrong-key"},
    )
    assert res.status_code == 401


async def test_predict_succeeds_with_correct_key(ac, deployment_id):
    """Protected deployment accepts requests with the correct Bearer key."""
    key = (await ac.post(f"/api/deploy/{deployment_id}/api-key")).json()["api_key"]
    res = await ac.post(
        f"/api/predict/{deployment_id}",
        json={"units": 25, "region": "East"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert res.status_code == 200
    assert "prediction" in res.json()


async def test_predict_old_key_rejected_after_regenerate(ac, deployment_id):
    """Old key is rejected after regeneration; new key works."""
    old_key = (await ac.post(f"/api/deploy/{deployment_id}/api-key")).json()["api_key"]
    new_key = (await ac.post(f"/api/deploy/{deployment_id}/api-key")).json()["api_key"]

    res_old = await ac.post(
        f"/api/predict/{deployment_id}",
        json={"units": 25},
        headers={"Authorization": f"Bearer {old_key}"},
    )
    assert res_old.status_code == 401

    res_new = await ac.post(
        f"/api/predict/{deployment_id}",
        json={"units": 25},
        headers={"Authorization": f"Bearer {new_key}"},
    )
    assert res_new.status_code == 200


# ---------------------------------------------------------------------------
# Tests — disable protection
# ---------------------------------------------------------------------------


async def test_disable_api_key(ac, deployment_id):
    """DELETE /api/deploy/{id}/api-key restores open access."""
    await ac.post(f"/api/deploy/{deployment_id}/api-key")

    res = await ac.delete(f"/api/deploy/{deployment_id}/api-key")
    assert res.status_code == 204

    detail = await ac.get(f"/api/deploy/{deployment_id}")
    assert detail.json()["api_key_enabled"] is False

    # Prediction works again without key
    res = await ac.post(
        f"/api/predict/{deployment_id}", json={"units": 25, "region": "East"}
    )
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Tests — batch prediction auth
# ---------------------------------------------------------------------------


async def test_batch_blocked_without_key(ac, deployment_id):
    """Batch endpoint also enforces key protection."""
    await ac.post(f"/api/deploy/{deployment_id}/api-key")
    csv_bytes = b"units,region\n25,East\n30,West\n"
    res = await ac.post(
        f"/api/predict/{deployment_id}/batch",
        files={"file": ("batch.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert res.status_code == 401


async def test_batch_succeeds_with_correct_key(ac, deployment_id):
    """Batch endpoint accepts the correct key."""
    key = (await ac.post(f"/api/deploy/{deployment_id}/api-key")).json()["api_key"]
    csv_bytes = b"units,region\n25,East\n30,West\n"
    res = await ac.post(
        f"/api/predict/{deployment_id}/batch",
        files={"file": ("batch.csv", io.BytesIO(csv_bytes), "text/csv")},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Tests — 404 for missing deployments
# ---------------------------------------------------------------------------


async def test_generate_key_missing_deployment(ac):
    """Key generation on non-existent deployment returns 404."""
    res = await ac.post("/api/deploy/does-not-exist/api-key")
    assert res.status_code == 404


async def test_disable_key_missing_deployment(ac):
    """Key deletion on non-existent deployment returns 404."""
    res = await ac.delete("/api/deploy/does-not-exist/api-key")
    assert res.status_code == 404
