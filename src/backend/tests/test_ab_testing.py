"""Tests for champion-challenger A/B testing.

Covers:
- POST   /api/deploy/{id}/ab-test          — create test
- GET    /api/deploy/{id}/ab-test          — get status + metrics
- DELETE /api/deploy/{id}/ab-test          — end test (no promotion)
- POST   /api/deploy/{id}/ab-test/promote  — promote challenger
- Traffic routing in POST /api/predict/{id} when A/B test is active
- Validation errors: same deployment, invalid split pct, inactive deployments
"""

import io
import time
from unittest.mock import patch

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

    import models.ab_test  # noqa
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
    import models.webhook_config  # noqa

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
    resp = await ac.post("/api/projects", json={"name": "AB Test Project"})
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


async def _wait_for_run(ac, project_id, run_id):
    for _ in range(30):
        r = await ac.get(f"/api/models/{project_id}/runs")
        run = next((x for x in r.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.3)
    pytest.skip("Training did not complete")


@pytest.fixture()
async def champion_id(ac, project_id, feature_set_id):
    """Deploy a linear_regression model as the champion."""
    resp = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["model_run_ids"][0]
    await _wait_for_run(ac, project_id, run_id)
    r = await ac.post(f"/api/deploy/{run_id}", json={})
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.fixture()
async def project2_id(ac):
    """Second project for the challenger deployment."""
    resp = await ac.post("/api/projects", json={"name": "AB Challenger Project"})
    return resp.json()["id"]


@pytest.fixture()
async def challenger_id(ac, project2_id):
    """Deploy a model as the challenger on a second project."""
    # Upload a dataset for the second project
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales2.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project2_id},
    )
    assert resp.status_code == 201, resp.text
    ds2_id = resp.json()["dataset_id"]

    resp = await ac.post(
        f"/api/features/{ds2_id}/apply",
        json={"transformations": []},
    )
    assert resp.status_code == 201, resp.text
    fs_id = resp.json()["feature_set_id"]
    await ac.post(
        f"/api/features/{ds2_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    resp2 = await ac.post(
        f"/api/models/{project2_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": fs_id},
    )
    assert resp2.status_code == 202, resp2.text
    run_id = resp2.json()["model_run_ids"][0]
    await _wait_for_run(ac, project2_id, run_id)
    r = await ac.post(f"/api/deploy/{run_id}", json={})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------
# Tests — create A/B test
# ---------------------------------------------------------------------------


async def test_create_ab_test_returns_201(ac, champion_id, challenger_id):
    """POST /api/deploy/{id}/ab-test creates a test and returns 201."""
    res = await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id, "champion_split_pct": 80},
    )
    assert res.status_code == 201, res.text
    data = res.json()
    assert data["champion_id"] == champion_id
    assert data["challenger_id"] == challenger_id
    assert data["champion_split_pct"] == 80
    assert data["challenger_split_pct"] == 20
    assert data["is_active"] is True
    assert data["winner"] is None


async def test_create_ab_test_default_split(ac, champion_id, challenger_id):
    """Default champion_split_pct is 80."""
    res = await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id},
    )
    assert res.status_code == 201
    assert res.json()["champion_split_pct"] == 80


async def test_create_ab_test_includes_algorithm_names(ac, champion_id, challenger_id):
    """Response includes champion_algorithm and challenger_algorithm."""
    res = await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id},
    )
    data = res.json()
    assert data["champion_algorithm"] is not None
    assert data["challenger_algorithm"] is not None


async def test_create_ab_test_replaces_existing(ac, champion_id, challenger_id):
    """Creating a second A/B test deactivates the first."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id, "champion_split_pct": 80},
    )
    # Create a new test (same challenger, different split)
    res = await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id, "champion_split_pct": 70},
    )
    assert res.status_code == 201
    assert res.json()["champion_split_pct"] == 70


# ---------------------------------------------------------------------------
# Tests — validation errors
# ---------------------------------------------------------------------------


async def test_create_ab_test_same_deployment_rejected(ac, champion_id):
    """Using the same ID for champion and challenger returns 400."""
    res = await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": champion_id},
    )
    assert res.status_code == 400


async def test_create_ab_test_invalid_split_pct(ac, champion_id, challenger_id):
    """champion_split_pct=0 and =100 are rejected."""
    for bad_pct in [0, 100]:
        res = await ac.post(
            f"/api/deploy/{champion_id}/ab-test",
            json={"challenger_id": challenger_id, "champion_split_pct": bad_pct},
        )
        assert res.status_code == 400


async def test_create_ab_test_unknown_challenger_rejected(ac, champion_id):
    """Unknown challenger_id returns 404."""
    res = await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": "nonexistent-id"},
    )
    assert res.status_code == 404


async def test_create_ab_test_unknown_champion_rejected(ac, challenger_id):
    """Unknown champion deployment_id returns 404."""
    res = await ac.post(
        "/api/deploy/nonexistent-id/ab-test",
        json={"challenger_id": challenger_id},
    )
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests — GET A/B test
# ---------------------------------------------------------------------------


async def test_get_ab_test_returns_active_test(ac, champion_id, challenger_id):
    """GET /api/deploy/{id}/ab-test returns the active test."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id},
    )
    res = await ac.get(f"/api/deploy/{champion_id}/ab-test")
    assert res.status_code == 200
    data = res.json()
    assert data["is_active"] is True
    assert data["champion_id"] == champion_id


async def test_get_ab_test_no_test_returns_404(ac, champion_id):
    """GET returns 404 when no active test exists."""
    res = await ac.get(f"/api/deploy/{champion_id}/ab-test")
    assert res.status_code == 404


async def test_get_ab_test_includes_significance(ac, champion_id, challenger_id):
    """A/B test response includes significance block (even with no data)."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id},
    )
    res = await ac.get(f"/api/deploy/{champion_id}/ab-test")
    sig = res.json()["significance"]
    assert "significant" in sig
    assert "note" in sig


async def test_get_ab_test_includes_per_variant_metrics(ac, champion_id, challenger_id):
    """A/B test response includes champion_metrics and challenger_metrics."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id},
    )
    data = (await ac.get(f"/api/deploy/{champion_id}/ab-test")).json()
    assert "champion_metrics" in data
    assert "challenger_metrics" in data
    assert "request_count" in data["champion_metrics"]


# ---------------------------------------------------------------------------
# Tests — end A/B test
# ---------------------------------------------------------------------------


async def test_end_ab_test_returns_204(ac, champion_id, challenger_id):
    """DELETE /api/deploy/{id}/ab-test returns 204."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id},
    )
    res = await ac.delete(f"/api/deploy/{champion_id}/ab-test")
    assert res.status_code == 204


async def test_end_ab_test_deactivates_test(ac, champion_id, challenger_id):
    """After ending, GET returns 404."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id},
    )
    await ac.delete(f"/api/deploy/{champion_id}/ab-test")
    assert (await ac.get(f"/api/deploy/{champion_id}/ab-test")).status_code == 404


async def test_end_ab_test_no_test_returns_404(ac, champion_id):
    """Ending a non-existent test returns 404."""
    res = await ac.delete(f"/api/deploy/{champion_id}/ab-test")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests — promote challenger
# ---------------------------------------------------------------------------


async def test_promote_challenger_returns_200(ac, champion_id, challenger_id):
    """POST /api/deploy/{id}/ab-test/promote returns 200 with updated deployment."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id},
    )
    res = await ac.post(f"/api/deploy/{champion_id}/ab-test/promote")
    assert res.status_code == 200, res.text
    data = res.json()
    assert "deployment" in data
    assert "message" in data
    assert "Challenger promoted" in data["message"]


async def test_promote_challenger_ends_test(ac, champion_id, challenger_id):
    """After promoting, GET ab-test returns 404."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id},
    )
    await ac.post(f"/api/deploy/{champion_id}/ab-test/promote")
    assert (await ac.get(f"/api/deploy/{champion_id}/ab-test")).status_code == 404


async def test_promote_challenger_keeps_endpoint_url(ac, champion_id, challenger_id):
    """Promoting challenger keeps the champion's endpoint URL unchanged."""
    orig_endpoint = (await ac.get(f"/api/deploy/{champion_id}")).json()["endpoint_path"]
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id},
    )
    await ac.post(f"/api/deploy/{champion_id}/ab-test/promote")
    new_endpoint = (await ac.get(f"/api/deploy/{champion_id}")).json()["endpoint_path"]
    assert orig_endpoint == new_endpoint


async def test_promote_challenger_no_test_returns_404(ac, champion_id):
    """Promoting when no test is active returns 404."""
    res = await ac.post(f"/api/deploy/{champion_id}/ab-test/promote")
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# Tests — traffic routing (deterministic with mock)
# ---------------------------------------------------------------------------


async def test_ab_routing_routes_to_challenger(ac, champion_id, challenger_id):
    """When random.random() >= split_pct/100, prediction uses challenger model."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id, "champion_split_pct": 80},
    )
    # Force routing to challenger: random returns 0.9 >= 0.8
    with patch("api.deploy.random.random", return_value=0.9):
        res = await ac.post(
            f"/api/predict/{champion_id}",
            json={"region": "East", "units": 10},
        )
    assert res.status_code == 200
    assert res.json()["ab_variant"] == "challenger"


async def test_ab_routing_routes_to_champion(ac, champion_id, challenger_id):
    """When random.random() < split_pct/100, prediction uses champion model."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id, "champion_split_pct": 80},
    )
    # Force routing to champion: random returns 0.5 < 0.8
    with patch("api.deploy.random.random", return_value=0.5):
        res = await ac.post(
            f"/api/predict/{champion_id}",
            json={"region": "East", "units": 10},
        )
    assert res.status_code == 200
    assert res.json()["ab_variant"] == "champion"


async def test_no_ab_test_variant_is_none(ac, champion_id):
    """When no A/B test is active, ab_variant is None in response."""
    res = await ac.post(
        f"/api/predict/{champion_id}",
        json={"region": "East", "units": 10},
    )
    assert res.status_code == 200
    assert res.json()["ab_variant"] is None


async def test_ab_routing_records_variant_in_logs(ac, champion_id, challenger_id):
    """Predictions made during A/B test are tagged in prediction logs."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id, "champion_split_pct": 80},
    )
    with patch("api.deploy.random.random", return_value=0.5):
        await ac.post(
            f"/api/predict/{champion_id}",
            json={"region": "East", "units": 10},
        )
    # After prediction, champion_metrics should show request_count = 1
    ab_data = (await ac.get(f"/api/deploy/{champion_id}/ab-test")).json()
    assert ab_data["champion_metrics"]["request_count"] == 1
    assert ab_data["challenger_metrics"]["request_count"] == 0


async def test_ab_challenger_requests_tracked_separately(
    ac, champion_id, challenger_id
):
    """Challenger predictions are tracked separately from champion predictions."""
    await ac.post(
        f"/api/deploy/{champion_id}/ab-test",
        json={"challenger_id": challenger_id, "champion_split_pct": 80},
    )
    with patch("api.deploy.random.random", return_value=0.9):  # challenger
        await ac.post(
            f"/api/predict/{champion_id}",
            json={"region": "East", "units": 10},
        )
    ab_data = (await ac.get(f"/api/deploy/{champion_id}/ab-test")).json()
    assert ab_data["challenger_metrics"]["request_count"] == 1
    assert ab_data["champion_metrics"]["request_count"] == 0


# ---------------------------------------------------------------------------
# Tests — significance testing helper
# ---------------------------------------------------------------------------


def test_ab_significance_needs_minimum_samples():
    """_ab_significance returns significant=False with < 5 samples per variant."""
    from api.deploy import _ab_significance
    from unittest.mock import MagicMock

    # Mock session that returns fewer than 5 predictions per variant
    mock_session = MagicMock()
    mock_session.exec.return_value.all.return_value = []
    result = _ab_significance("fake-id", mock_session)
    assert result["significant"] is False
    assert "5" in result["note"]


def test_ab_variant_metrics_empty_logs():
    """_ab_variant_metrics returns zeros when no logs exist."""
    from api.deploy import _ab_variant_metrics
    from unittest.mock import MagicMock

    mock_session = MagicMock()
    mock_session.exec.return_value.all.return_value = []
    result = _ab_variant_metrics("fake-id", "champion", mock_session)
    assert result["request_count"] == 0
    assert result["avg_confidence"] is None
    assert result["p95_ms"] is None
    assert result["avg_prediction"] is None


def test_percentile_sorted_edge_cases():
    """_percentile_sorted handles edge cases correctly."""
    from api.deploy import _percentile_sorted

    assert _percentile_sorted([], 95) == 0.0
    assert _percentile_sorted([42.0], 95) == 42.0
    assert _percentile_sorted([10.0, 20.0], 50) == pytest.approx(15.0, abs=0.1)
    assert _percentile_sorted([1.0, 2.0, 3.0, 4.0, 5.0], 100) == pytest.approx(
        5.0, abs=0.1
    )
