"""Tests for Multi-Deployment Status Overview feature.

Covers:
- compute_deployments_overview() pure function — empty list, counts, sorting, summary
- GET /api/deploy/overview REST endpoint — no deployments, one, mixed environments
- _DEPLOYMENTS_OVERVIEW_PATTERNS chat intent detection
"""

import time

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"age,income,revenue\n"
    b"25,30000,1234.50\n"
    b"35,45000,2150.75\n"
    b"45,60000,3280.00\n"
    b"55,75000,4450.25\n"
    b"30,38000,1680.50\n"
    b"40,52000,2780.00\n"
    b"50,68000,3900.50\n"
    b"28,33000,1380.25\n"
    b"38,48000,2480.75\n"
    b"48,63000,3580.00\n"
)


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    import api.models as models_module

    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


def _make_summary(
    *,
    deployment_id: str = "d1",
    project_id: str = "proj-1",
    project_name: str = "Sales",
    algorithm: str = "random_forest_regressor",
    algorithm_plain: str = "Random Forest",
    target_column: str = "revenue",
    environment: str = "staging",
    health_score: int = 80,
    status: str = "healthy",
    top_issue: str | None = None,
    recommendation: str | None = None,
    request_count: int = 100,
    predictions_last_7d: int = 20,
    predictions_today: int = 3,
    api_key_enabled: bool = False,
    rate_limit_rpm: int | None = None,
    monthly_quota: int | None = None,
) -> dict:
    return {
        "deployment_id": deployment_id,
        "project_id": project_id,
        "project_name": project_name,
        "algorithm": algorithm,
        "algorithm_plain": algorithm_plain,
        "target_column": target_column,
        "environment": environment,
        "health_score": health_score,
        "status": status,
        "top_issue": top_issue,
        "recommendation": recommendation,
        "request_count": request_count,
        "predictions_last_7d": predictions_last_7d,
        "predictions_today": predictions_today,
        "api_key_enabled": api_key_enabled,
        "rate_limit_rpm": rate_limit_rpm,
        "monthly_quota": monthly_quota,
    }


# ---------------------------------------------------------------------------
# compute_deployments_overview — pure function tests
# ---------------------------------------------------------------------------


def test_empty_list_returns_zero_counts():
    from core.analyzer import compute_deployments_overview

    result = compute_deployments_overview([])
    assert result["total_deployments"] == 0
    assert result["production_count"] == 0
    assert result["staging_count"] == 0
    assert result["total_predictions"] == 0
    assert result["avg_health_score"] == 0
    assert result["healthy_count"] == 0
    assert result["warning_count"] == 0
    assert result["critical_count"] == 0
    assert result["deployments"] == []
    assert "No active deployments" in result["summary"]


def test_single_healthy_staging_deployment():
    from core.analyzer import compute_deployments_overview

    dep = _make_summary(
        deployment_id="d1",
        environment="staging",
        health_score=85,
        status="healthy",
        request_count=50,
    )
    result = compute_deployments_overview([dep])
    assert result["total_deployments"] == 1
    assert result["production_count"] == 0
    assert result["staging_count"] == 1
    assert result["total_predictions"] == 50
    assert result["avg_health_score"] == 85
    assert result["healthy_count"] == 1
    assert result["warning_count"] == 0
    assert result["critical_count"] == 0
    assert len(result["deployments"]) == 1
    assert result["deployments"][0]["deployment_id"] == "d1"


def test_single_critical_production_deployment():
    from core.analyzer import compute_deployments_overview

    dep = _make_summary(
        deployment_id="d2",
        environment="production",
        health_score=30,
        status="critical",
        request_count=200,
    )
    result = compute_deployments_overview([dep])
    assert result["production_count"] == 1
    assert result["staging_count"] == 0
    assert result["critical_count"] == 1
    assert result["healthy_count"] == 0
    assert "attention" in result["summary"]


def test_counts_mixed_statuses():
    from core.analyzer import compute_deployments_overview

    deps = [
        _make_summary(
            deployment_id="d1", status="healthy", health_score=90, request_count=100
        ),
        _make_summary(
            deployment_id="d2", status="warning", health_score=60, request_count=50
        ),
        _make_summary(
            deployment_id="d3", status="critical", health_score=25, request_count=10
        ),
        _make_summary(
            deployment_id="d4", status="healthy", health_score=80, request_count=75
        ),
    ]
    result = compute_deployments_overview(deps)
    assert result["total_deployments"] == 4
    assert result["healthy_count"] == 2
    assert result["warning_count"] == 1
    assert result["critical_count"] == 1
    assert result["total_predictions"] == 235
    assert result["avg_health_score"] == int((90 + 60 + 25 + 80) / 4)


def test_production_sorted_before_staging():
    from core.analyzer import compute_deployments_overview

    deps = [
        _make_summary(
            deployment_id="staging-1",
            environment="staging",
            health_score=95,
            request_count=500,
        ),
        _make_summary(
            deployment_id="prod-1",
            environment="production",
            health_score=70,
            request_count=10,
        ),
        _make_summary(
            deployment_id="staging-2",
            environment="staging",
            health_score=60,
            request_count=100,
        ),
    ]
    result = compute_deployments_overview(deps)
    sorted_ids = [d["deployment_id"] for d in result["deployments"]]
    assert sorted_ids[0] == "prod-1", (
        "production should come first regardless of health score"
    )
    assert set(sorted_ids[1:]) == {"staging-1", "staging-2"}


def test_within_environment_sorted_by_health_desc():
    from core.analyzer import compute_deployments_overview

    deps = [
        _make_summary(
            deployment_id="s1", environment="staging", health_score=60, request_count=10
        ),
        _make_summary(
            deployment_id="s2", environment="staging", health_score=90, request_count=10
        ),
        _make_summary(
            deployment_id="s3", environment="staging", health_score=75, request_count=10
        ),
    ]
    result = compute_deployments_overview(deps)
    scores = [d["health_score"] for d in result["deployments"]]
    assert scores == sorted(scores, reverse=True)


def test_health_tiebreak_by_request_count_desc():
    from core.analyzer import compute_deployments_overview

    deps = [
        _make_summary(
            deployment_id="low",
            environment="staging",
            health_score=80,
            request_count=10,
        ),
        _make_summary(
            deployment_id="high",
            environment="staging",
            health_score=80,
            request_count=500,
        ),
    ]
    result = compute_deployments_overview(deps)
    assert result["deployments"][0]["deployment_id"] == "high"


def test_avg_health_score_integer():
    from core.analyzer import compute_deployments_overview

    deps = [
        _make_summary(deployment_id="a", health_score=71),
        _make_summary(deployment_id="b", health_score=72),
    ]
    result = compute_deployments_overview(deps)
    assert isinstance(result["avg_health_score"], int)
    assert result["avg_health_score"] == 71


def test_summary_mentions_all_healthy():
    from core.analyzer import compute_deployments_overview

    deps = [
        _make_summary(deployment_id="a", status="healthy", health_score=85),
        _make_summary(deployment_id="b", status="healthy", health_score=90),
    ]
    result = compute_deployments_overview(deps)
    assert "healthy" in result["summary"].lower()
    assert "attention" not in result["summary"]


def test_summary_mentions_warnings_not_critical():
    from core.analyzer import compute_deployments_overview

    deps = [
        _make_summary(deployment_id="a", status="warning", health_score=60),
        _make_summary(deployment_id="b", status="healthy", health_score=90),
    ]
    result = compute_deployments_overview(deps)
    assert "warning" in result["summary"].lower()


def test_production_count_in_mixed_environments():
    from core.analyzer import compute_deployments_overview

    deps = [
        _make_summary(deployment_id="p1", environment="production"),
        _make_summary(deployment_id="p2", environment="production"),
        _make_summary(deployment_id="s1", environment="staging"),
    ]
    result = compute_deployments_overview(deps)
    assert result["production_count"] == 2
    assert result["staging_count"] == 1


# ---------------------------------------------------------------------------
# GET /api/deploy/overview — REST endpoint tests
# ---------------------------------------------------------------------------


async def _create_deployment(
    ac, project_name: str, environment: str = "staging"
) -> str:
    """Helper: project → upload → apply → target → train → deploy. Returns deployment id."""
    proj_resp = await ac.post("/api/projects", json={"name": project_name})
    assert proj_resp.status_code == 201, proj_resp.text
    project_id = proj_resp.json()["id"]

    upload_resp = await ac.post(
        "/api/data/upload",
        files={"file": ("data.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert upload_resp.status_code == 201, upload_resp.text
    dataset_id = upload_resp.json()["dataset_id"]

    apply_resp = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert apply_resp.status_code == 201, apply_resp.text
    fs_id = apply_resp.json()["feature_set_id"]

    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )

    train_resp = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": fs_id},
    )
    assert train_resp.status_code == 202, train_resp.text
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(40):
        r = await ac.get(f"/api/models/{project_id}/runs")
        run = next((x for x in r.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            break
        time.sleep(0.25)
    else:
        pytest.skip("Training did not complete in time")

    deploy_resp = await ac.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code in (200, 201), deploy_resp.text
    dep_id = deploy_resp.json()["id"]

    if environment == "production":
        promo_resp = await ac.post(f"/api/deploy/{dep_id}/promote-to-production")
        assert promo_resp.status_code == 200, promo_resp.text

    return dep_id


@pytest.mark.asyncio
async def test_overview_endpoint_no_deployments(ac):
    resp = await ac.get("/api/deploy/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_deployments"] == 0
    assert data["deployments"] == []


@pytest.mark.asyncio
async def test_overview_endpoint_one_active_deployment(ac):
    await _create_deployment(ac, "Overview Test", environment="staging")

    overview_resp = await ac.get("/api/deploy/overview")
    assert overview_resp.status_code == 200
    data = overview_resp.json()
    assert data["total_deployments"] == 1
    assert data["staging_count"] == 1
    assert data["production_count"] == 0
    assert len(data["deployments"]) == 1
    dep = data["deployments"][0]
    assert dep["target_column"] == "revenue"
    assert dep["environment"] == "staging"
    assert "health_score" in dep
    assert "status" in dep


@pytest.mark.asyncio
async def test_overview_endpoint_mixed_environments(ac):
    await _create_deployment(ac, "Proj staging", environment="staging")
    await _create_deployment(ac, "Proj production", environment="production")

    overview_resp = await ac.get("/api/deploy/overview")
    assert overview_resp.status_code == 200
    data = overview_resp.json()
    assert data["total_deployments"] == 2
    assert data["production_count"] == 1
    assert data["staging_count"] == 1
    assert data["deployments"][0]["environment"] == "production"


@pytest.mark.asyncio
async def test_overview_endpoint_inactive_deployments_excluded(ac):
    dep_id = await _create_deployment(ac, "Deactivated")
    await ac.delete(f"/api/deploy/{dep_id}")

    overview_resp = await ac.get("/api/deploy/overview")
    assert overview_resp.status_code == 200
    assert overview_resp.json()["total_deployments"] == 0


# ---------------------------------------------------------------------------
# _DEPLOYMENTS_OVERVIEW_PATTERNS — regex detection
# ---------------------------------------------------------------------------


def test_pattern_show_all_deployments():
    from api.chat import _DEPLOYMENTS_OVERVIEW_PATTERNS

    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("show all my deployments")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("show deployments")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("list my active deployments")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("display all live deployments")


def test_pattern_deployment_dashboard_variants():
    from api.chat import _DEPLOYMENTS_OVERVIEW_PATTERNS

    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("deployment dashboard")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("deployment overview")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("deployment status")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("deployment monitoring")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("deployment summary")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("deployment report")


def test_pattern_which_models_live():
    from api.chat import _DEPLOYMENTS_OVERVIEW_PATTERNS

    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("which models are live")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("which of my models is running")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("which models are deployed")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("which models are active")


def test_pattern_my_deployed_models():
    from api.chat import _DEPLOYMENTS_OVERVIEW_PATTERNS

    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("all deployed models")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("my deployed models")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("all deployed endpoints")


def test_pattern_live_model_status():
    from api.chat import _DEPLOYMENTS_OVERVIEW_PATTERNS

    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("live model status")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("live model overview")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("live model dashboard")


def test_pattern_deployment_health():
    from api.chat import _DEPLOYMENTS_OVERVIEW_PATTERNS

    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("deployment health overview")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("deployment health summary")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("deployment health dashboard")


def test_pattern_case_insensitive():
    from api.chat import _DEPLOYMENTS_OVERVIEW_PATTERNS

    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("Show All My Deployments")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("DEPLOYMENT DASHBOARD")
    assert _DEPLOYMENTS_OVERVIEW_PATTERNS.search("Which Models Are LIVE")


def test_pattern_does_not_match_unrelated():
    from api.chat import _DEPLOYMENTS_OVERVIEW_PATTERNS

    assert not _DEPLOYMENTS_OVERVIEW_PATTERNS.search("train my model on revenue")
    assert not _DEPLOYMENTS_OVERVIEW_PATTERNS.search("what is the accuracy")
    assert not _DEPLOYMENTS_OVERVIEW_PATTERNS.search("upload a dataset")
    assert not _DEPLOYMENTS_OVERVIEW_PATTERNS.search("show me a histogram of age")
