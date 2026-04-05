"""Tests for Proactive Model Health Summary feature.

Covers:
- compute_deployment_health_item() pure function — age/usage scoring
- compute_project_health_summary() pure function — aggregation and summary
- GET /api/projects/{project_id}/health-summary endpoint
- _HEALTH_SUMMARY_PATTERNS chat intent detection
"""

from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_deployment_dict(
    *,
    deployment_id: str = "dep-1",
    algorithm: str = "random_forest_regressor",
    target_column: str = "revenue",
    created_at: datetime,
    request_count: int = 100,
    last_predicted_at: datetime | None = None,
    environment: str = "staging",
) -> dict:
    return {
        "deployment_id": deployment_id,
        "algorithm": algorithm,
        "target_column": target_column,
        "created_at": created_at,
        "request_count": request_count,
        "last_predicted_at": last_predicted_at,
        "environment": environment,
    }


# ---------------------------------------------------------------------------
# compute_deployment_health_item — age scoring
# ---------------------------------------------------------------------------


def test_fresh_deployment_healthy():
    from core.analyzer import compute_deployment_health_item

    now = datetime(2026, 4, 4)
    created = now - timedelta(days=5)
    item = compute_deployment_health_item(
        deployment_id="d1",
        algorithm="linear_regression",
        target_column="sales",
        created_at=created,
        request_count=50,
        last_predicted_at=now - timedelta(days=1),
        environment="production",
        now=now,
    )
    assert item["status"] == "healthy"
    assert item["health_score"] >= 75
    assert item["algorithm_plain"] == "Linear Regression"
    assert item["target_column"] == "sales"


def test_stale_idle_deployment_critical():
    from core.analyzer import compute_deployment_health_item

    now = datetime(2026, 4, 4)
    created = now - timedelta(days=200)  # over 180 days old → age_score=20
    last_pred = now - timedelta(days=120)  # idle for 120 days → usage_score=40
    item = compute_deployment_health_item(
        deployment_id="d2",
        algorithm="random_forest_regressor",
        target_column="revenue",
        created_at=created,
        request_count=5,
        last_predicted_at=last_pred,
        environment="staging",
        now=now,
    )
    # health_score = 20*0.55 + 40*0.45 = 11 + 18 = 29 → critical
    assert item["status"] == "critical"
    assert item["health_score"] < 50
    assert item["top_issue"] is not None
    assert "days old" in item["top_issue"]
    assert item["recommendation"] is not None


def test_moderately_old_deployment_warning():
    from core.analyzer import compute_deployment_health_item

    now = datetime(2026, 4, 4)
    created = now - timedelta(days=55)  # 55 days old → age_score=80
    item = compute_deployment_health_item(
        deployment_id="d3",
        algorithm="gradient_boosting_regressor",
        target_column="churn",
        created_at=created,
        request_count=30,
        last_predicted_at=now - timedelta(days=2),
        environment="production",
        now=now,
    )
    # 55 days old: age_score=80, still used → should be warning or healthy
    assert item["status"] in ("healthy", "warning")
    assert item["health_score"] >= 50


def test_unused_deployment_top_issue():
    from core.analyzer import compute_deployment_health_item

    now = datetime(2026, 4, 4)
    created = now - timedelta(days=40)
    item = compute_deployment_health_item(
        deployment_id="d4",
        algorithm="logistic_regression",
        target_column="default",
        created_at=created,
        request_count=0,  # never used
        last_predicted_at=None,
        environment="staging",
        now=now,
    )
    assert item["top_issue"] is not None
    assert (
        "prediction" in item["top_issue"].lower() or "used" in item["top_issue"].lower()
    )


def test_idle_deployment_top_issue():
    from core.analyzer import compute_deployment_health_item

    now = datetime(2026, 4, 4)
    created = now - timedelta(days=10)
    item = compute_deployment_health_item(
        deployment_id="d5",
        algorithm="decision_tree_regressor",
        target_column="price",
        created_at=created,
        request_count=100,
        last_predicted_at=now - timedelta(days=45),  # idle for 45 days
        environment="production",
        now=now,
    )
    assert item["top_issue"] is not None
    assert "days" in item["top_issue"].lower()


def test_health_item_name_format():
    from core.analyzer import compute_deployment_health_item

    now = datetime(2026, 4, 4)
    item = compute_deployment_health_item(
        deployment_id="d6",
        algorithm="xgboost_classifier",
        target_column="churn",
        created_at=now - timedelta(days=2),
        request_count=500,
        last_predicted_at=now - timedelta(days=1),
        environment="production",
        now=now,
    )
    assert item["name"] == "XGBoost → churn"
    assert item["algorithm_plain"] == "XGBoost"


def test_unknown_algorithm_falls_back():
    from core.analyzer import compute_deployment_health_item

    now = datetime(2026, 4, 4)
    item = compute_deployment_health_item(
        deployment_id="d7",
        algorithm="some_custom_algo",
        target_column="y",
        created_at=now - timedelta(days=1),
        request_count=10,
        last_predicted_at=None,
        environment="staging",
        now=now,
    )
    # Falls back to the raw algorithm string
    assert item["algorithm_plain"] == "some_custom_algo"


# ---------------------------------------------------------------------------
# compute_project_health_summary — aggregation
# ---------------------------------------------------------------------------


def test_empty_deployments_summary():
    from core.analyzer import compute_project_health_summary

    result = compute_project_health_summary([])
    assert result["total"] == 0
    assert result["overall_status"] == "healthy"
    assert result["alerts"] == []
    assert "No active deployments" in result["summary"]


def test_all_healthy_summary():
    from core.analyzer import compute_project_health_summary

    now = datetime(2026, 4, 4)
    deps = [
        {
            "deployment_id": f"d{i}",
            "algorithm": "linear_regression",
            "target_column": "sales",
            "created_at": now - timedelta(days=3),
            "request_count": 100,
            "last_predicted_at": now - timedelta(days=1),
            "environment": "production",
        }
        for i in range(3)
    ]
    result = compute_project_health_summary(deps, now=now)
    assert result["total"] == 3
    assert result["overall_status"] == "healthy"
    assert len(result["alerts"]) == 0
    assert "healthy" in result["summary"]


def test_critical_deployment_escalates_status():
    from core.analyzer import compute_project_health_summary

    now = datetime(2026, 4, 4)
    deps = [
        {
            "deployment_id": "fresh",
            "algorithm": "linear_regression",
            "target_column": "sales",
            "created_at": now - timedelta(days=3),
            "request_count": 100,
            "last_predicted_at": now - timedelta(days=1),
            "environment": "production",
        },
        {
            "deployment_id": "stale",
            "algorithm": "random_forest_regressor",
            "target_column": "churn",
            "created_at": now - timedelta(days=200),  # critical
            "request_count": 10,
            "last_predicted_at": now - timedelta(days=10),
            "environment": "staging",
        },
    ]
    result = compute_project_health_summary(deps, now=now)
    assert result["overall_status"] == "critical"
    assert len(result["alerts"]) >= 1
    assert "attention" in result["summary"]


def test_summary_alerts_exclude_healthy():
    from core.analyzer import compute_project_health_summary

    now = datetime(2026, 4, 4)
    deps = [
        {
            "deployment_id": "ok",
            "algorithm": "linear_regression",
            "target_column": "sales",
            "created_at": now - timedelta(days=3),
            "request_count": 200,
            "last_predicted_at": now - timedelta(days=1),
            "environment": "production",
        },
        {
            "deployment_id": "stale",
            "algorithm": "random_forest_regressor",
            "target_column": "revenue",
            "created_at": now - timedelta(days=200),
            "request_count": 5,
            "last_predicted_at": now - timedelta(days=50),
            "environment": "staging",
        },
    ]
    result = compute_project_health_summary(deps, now=now)
    alert_ids = {a["deployment_id"] for a in result["alerts"]}
    assert "ok" not in alert_ids
    assert "stale" in alert_ids


# ---------------------------------------------------------------------------
# _HEALTH_SUMMARY_PATTERNS
# ---------------------------------------------------------------------------


def test_health_summary_patterns_match():
    from api.chat import _HEALTH_SUMMARY_PATTERNS

    assert _HEALTH_SUMMARY_PATTERNS.search("how are my models doing?")
    assert _HEALTH_SUMMARY_PATTERNS.search("any issues with my deployments?")
    assert _HEALTH_SUMMARY_PATTERNS.search("model health check")
    assert _HEALTH_SUMMARY_PATTERNS.search("check my models")
    assert _HEALTH_SUMMARY_PATTERNS.search("are my models still accurate?")
    assert _HEALTH_SUMMARY_PATTERNS.search("do I need to retrain?")
    assert _HEALTH_SUMMARY_PATTERNS.search("model drift")


def test_health_summary_patterns_no_false_positives():
    from api.chat import _HEALTH_SUMMARY_PATTERNS

    assert not _HEALTH_SUMMARY_PATTERNS.search("train a new model")
    assert not _HEALTH_SUMMARY_PATTERNS.search("show me my data")
    assert not _HEALTH_SUMMARY_PATTERNS.search("what is the accuracy?")


# ---------------------------------------------------------------------------
# GET /api/projects/{project_id}/health-summary — integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


async def _make_project_with_db(tmp_path, project_id: str):
    import db
    from models.project import Project
    from sqlmodel import create_engine

    test_db = str(tmp_path / f"{project_id}.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)
    with next(db.get_session()) as session:
        proj = Project(id=project_id, name="Health Summary Test")
        session.merge(proj)
        session.commit()
    return project_id


@pytest.mark.anyio
async def test_health_summary_no_deployments(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "hs_empty.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project_with_db(tmp_path, "hs-empty")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/projects/{project_id}/health-summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["overall_status"] == "healthy"
    assert data["alerts"] == []
    assert data["project_id"] == project_id


@pytest.mark.anyio
async def test_health_summary_not_found(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "hs_404.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/projects/no-such-project/health-summary")

    assert resp.status_code == 404


@pytest.mark.anyio
async def test_health_summary_with_fresh_deployment(tmp_path, set_test_env):
    from main import app
    import db
    from models.deployment import Deployment
    from models.model_run import ModelRun
    from sqlmodel import create_engine

    test_db = str(tmp_path / "hs_fresh.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project_with_db(tmp_path, "hs-fresh")

    now = datetime.now(UTC).replace(tzinfo=None)
    with next(db.get_session()) as session:
        run = ModelRun(
            id="run-hs-1",
            project_id=project_id,
            algorithm="linear_regression",
            status="done",
            is_deployed=True,
        )
        session.add(run)
        dep = Deployment(
            id="dep-hs-fresh",
            model_run_id="run-hs-1",
            project_id=project_id,
            endpoint_path="/api/predict/dep-hs-fresh",
            dashboard_url="/predict/dep-hs-fresh",
            algorithm="linear_regression",
            target_column="sales",
            created_at=now - timedelta(days=2),
            request_count=50,
            last_predicted_at=now - timedelta(days=1),
            environment="production",
        )
        session.add(dep)
        session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/projects/{project_id}/health-summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["overall_status"] == "healthy"
    assert len(data["all_items"]) == 1
    assert data["all_items"][0]["algorithm_plain"] == "Linear Regression"
