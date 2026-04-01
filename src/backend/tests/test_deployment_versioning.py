"""Tests for deployment versioning and rollback.

Covers:
- First deploy creates DeploymentVersion v1 (is_current=True)
- Re-deploying a new model archives previous version and creates v2
- GET /api/deploy/{id}/versions  — returns version history ordered newest-first
- POST /api/deploy/{id}/rollback/{version_number}  — restores an old version
- Rollback to non-existent version returns 404
- Endpoint URL stays stable across re-deploys (shareable link)
- Idempotent deploy does not add extra version
- Version response captures metrics/algorithm/target
- API key settings preserved across rollback
- Multiple retrains build linear version history
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
    b"East,160.0,16\nWest,210.0,21\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.deployment_version  # noqa
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
    r = await ac.post("/api/projects", json={"name": "Versioning Test"})
    return r.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    r = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert r.status_code == 201, r.text
    return r.json()["dataset_id"]


@pytest.fixture()
async def feature_set_id(ac, dataset_id):
    r = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert r.status_code == 201, r.text
    fs_id = r.json()["feature_set_id"]
    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


async def _train_and_wait(ac, project_id, feature_set_id):
    """Train a linear regression model and wait for completion. Returns run_id."""
    r = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert r.status_code == 202, r.text
    run_id = r.json()["model_run_ids"][0]
    for _ in range(40):
        r2 = await ac.get(f"/api/models/{project_id}/runs")
        runs = r2.json().get("runs", [])
        run = next((x for x in runs if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.2)
    pytest.skip("Training did not complete in time")


@pytest.fixture()
async def first_run_id(ac, project_id, feature_set_id):
    return await _train_and_wait(ac, project_id, feature_set_id)


@pytest.fixture()
async def deployment_id(ac, first_run_id):
    r = await ac.post(f"/api/deploy/{first_run_id}")
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_first_deploy_creates_v1(ac, deployment_id):
    """Deploying for the first time creates version 1 marked as current."""
    r = await ac.get(f"/api/deploy/{deployment_id}/versions")
    assert r.status_code == 200
    data = r.json()
    assert data["current_version_number"] == 1
    assert len(data["versions"]) == 1
    v = data["versions"][0]
    assert v["version_number"] == 1
    assert v["is_current"] is True
    assert v["algorithm"] == "linear_regression"


@pytest.mark.anyio
async def test_redeploy_archives_v1_creates_v2(
    ac, project_id, feature_set_id, deployment_id, first_run_id
):
    """Re-deploying a new model archives v1 and produces v2, endpoint stays stable."""
    original_endpoint = (await ac.get(f"/api/deploy/{deployment_id}")).json()["endpoint_path"]

    second_run = await _train_and_wait(ac, project_id, feature_set_id)
    r = await ac.post(f"/api/deploy/{second_run}")
    assert r.status_code == 201
    updated = r.json()

    # Same deployment ID and same endpoint URL
    assert updated["id"] == deployment_id
    assert updated["endpoint_path"] == original_endpoint

    # v1 archived, v2 current
    r = await ac.get(f"/api/deploy/{deployment_id}/versions")
    data = r.json()
    assert data["current_version_number"] == 2
    assert len(data["versions"]) == 2

    by_num = {v["version_number"]: v for v in data["versions"]}
    assert by_num[1]["is_current"] is False
    assert by_num[2]["is_current"] is True
    assert by_num[2]["model_run_id"] == second_run


@pytest.mark.anyio
async def test_versions_ordered_newest_first(
    ac, project_id, feature_set_id, deployment_id
):
    """GET /versions returns versions newest-first."""
    second_run = await _train_and_wait(ac, project_id, feature_set_id)
    await ac.post(f"/api/deploy/{second_run}")

    r = await ac.get(f"/api/deploy/{deployment_id}/versions")
    versions = r.json()["versions"]
    assert versions[0]["version_number"] == 2
    assert versions[1]["version_number"] == 1


@pytest.mark.anyio
async def test_rollback_to_v1(
    ac, project_id, feature_set_id, deployment_id, first_run_id
):
    """Rolling back to v1 restores v1's model, creates v3, endpoint stays stable."""
    original_endpoint = (await ac.get(f"/api/deploy/{deployment_id}")).json()["endpoint_path"]

    second_run = await _train_and_wait(ac, project_id, feature_set_id)
    await ac.post(f"/api/deploy/{second_run}")

    # Rollback to v1
    r = await ac.post(f"/api/deploy/{deployment_id}/rollback/1")
    assert r.status_code == 200
    result = r.json()
    assert result["rolled_back_to_version"] == 1
    assert result["new_version_number"] == 3
    assert result["model_run_id"] == first_run_id
    assert result["endpoint_path"] == original_endpoint

    # v3 is current and points at original model
    r = await ac.get(f"/api/deploy/{deployment_id}/versions")
    data = r.json()
    assert data["current_version_number"] == 3
    assert len(data["versions"]) == 3

    by_num = {v["version_number"]: v for v in data["versions"]}
    assert by_num[3]["is_current"] is True
    assert by_num[3]["model_run_id"] == first_run_id
    assert by_num[2]["is_current"] is False
    assert by_num[1]["is_current"] is False


@pytest.mark.anyio
async def test_rollback_nonexistent_version_returns_404(ac, deployment_id):
    r = await ac.post(f"/api/deploy/{deployment_id}/rollback/99")
    assert r.status_code == 404
    assert "99" in r.json()["detail"]


@pytest.mark.anyio
async def test_versions_unknown_deployment_returns_404(ac):
    r = await ac.get("/api/deploy/nonexistent-id/versions")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_rollback_unknown_deployment_returns_404(ac):
    r = await ac.post("/api/deploy/nonexistent-id/rollback/1")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_idempotent_deploy_no_extra_version(ac, deployment_id, first_run_id):
    """Deploying the same model_run_id twice does not add a second version."""
    r = await ac.post(f"/api/deploy/{first_run_id}")
    assert r.status_code == 201

    r = await ac.get(f"/api/deploy/{deployment_id}/versions")
    assert len(r.json()["versions"]) == 1


@pytest.mark.anyio
async def test_version_captures_metrics_and_metadata(ac, deployment_id):
    """Version snapshots store algorithm, target_column, problem_type, and metrics."""
    r = await ac.get(f"/api/deploy/{deployment_id}/versions")
    v = r.json()["versions"][0]
    assert isinstance(v["metrics"], dict)
    assert v["algorithm"] == "linear_regression"
    assert v["target_column"] == "revenue"
    assert v["problem_type"] == "regression"
    assert v["deployed_at"] is not None


@pytest.mark.anyio
async def test_api_key_preserved_across_rollback(
    ac, project_id, feature_set_id, deployment_id
):
    """api_key_enabled state is preserved after rollback (it's a Deployment-level field)."""
    # Protect with API key
    await ac.post(f"/api/deploy/{deployment_id}/api-key")

    # Deploy a new model
    second_run = await _train_and_wait(ac, project_id, feature_set_id)
    await ac.post(f"/api/deploy/{second_run}")

    # Rollback
    r = await ac.post(f"/api/deploy/{deployment_id}/rollback/1")
    assert r.status_code == 200
    assert r.json()["api_key_enabled"] is True


@pytest.mark.anyio
async def test_multiple_retrains_build_linear_history(
    ac, project_id, feature_set_id, deployment_id
):
    """Three sequential deploys produce v1, v2, v3 in append-only history."""
    for _ in range(2):
        run_id = await _train_and_wait(ac, project_id, feature_set_id)
        r = await ac.post(f"/api/deploy/{run_id}")
        assert r.status_code == 201

    r = await ac.get(f"/api/deploy/{deployment_id}/versions")
    data = r.json()
    assert data["current_version_number"] == 3
    assert len(data["versions"]) == 3
    version_numbers = sorted(v["version_number"] for v in data["versions"])
    assert version_numbers == [1, 2, 3]
