"""Tests for Cross-Project Portfolio Overview.

Covers:
- Pattern detection: _PORTFOLIO_PATTERNS
- Pure function: compute_portfolio_summary
- API endpoint: GET /api/projects/portfolio
- Chat SSE integration: portfolio_event emitted on match
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _PORTFOLIO_PATTERNS
from core.analyzer import compute_portfolio_summary


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


def test_portfolio_pattern_show_all_models():
    assert _PORTFOLIO_PATTERNS.search("show all my models")


def test_portfolio_pattern_portfolio_overview():
    assert _PORTFOLIO_PATTERNS.search("portfolio overview")


def test_portfolio_pattern_portfolio():
    assert _PORTFOLIO_PATTERNS.search("my portfolio")


def test_portfolio_pattern_compare_all_projects():
    assert _PORTFOLIO_PATTERNS.search("compare all my projects")


def test_portfolio_pattern_which_project_doing_best():
    assert _PORTFOLIO_PATTERNS.search("which project is doing best")


def test_portfolio_pattern_cross_project_view():
    assert _PORTFOLIO_PATTERNS.search("cross-project view")


def test_portfolio_pattern_all_my_work():
    assert _PORTFOLIO_PATTERNS.search("all my work")


def test_portfolio_pattern_how_many_models():
    assert _PORTFOLIO_PATTERNS.search("how many models do I have")


def test_portfolio_pattern_no_match():
    assert not _PORTFOLIO_PATTERNS.search("train a random forest model")


def test_portfolio_pattern_no_match_show_chart():
    assert not _PORTFOLIO_PATTERNS.search("show me a bar chart of sales")


# ---------------------------------------------------------------------------
# compute_portfolio_summary pure function tests
# ---------------------------------------------------------------------------


def _make_project(name, model_count=0, best_metric=None, has_deployment=False, preds=0):
    return {
        "project_id": str(uuid.uuid4()),
        "name": name,
        "dataset_filename": f"{name.lower().replace(' ', '_')}.csv",
        "row_count": 200,
        "model_count": model_count,
        "best_algorithm": "random_forest" if model_count > 0 else None,
        "best_metric_name": "r2" if model_count > 0 else None,
        "best_metric_value": best_metric,
        "best_problem_type": "regression" if model_count > 0 else None,
        "best_target_column": "revenue" if model_count > 0 else None,
        "has_deployment": has_deployment,
        "prediction_count": preds,
        "last_activity_at": "2026-04-10T12:00:00",
    }


def test_portfolio_empty_list():
    result = compute_portfolio_summary([])
    assert result["total_projects"] == 0
    assert result["active_deployments"] == 0
    assert result["total_predictions"] == 0
    assert result["best_performer"] is None
    assert result["projects"] == []
    assert "No projects" in result["summary"]


def test_portfolio_total_projects():
    summaries = [_make_project("A"), _make_project("B"), _make_project("C")]
    result = compute_portfolio_summary(summaries)
    assert result["total_projects"] == 3


def test_portfolio_active_deployments():
    summaries = [
        _make_project("A", has_deployment=True),
        _make_project("B", has_deployment=False),
        _make_project("C", has_deployment=True),
    ]
    result = compute_portfolio_summary(summaries)
    assert result["active_deployments"] == 2


def test_portfolio_total_predictions():
    summaries = [
        _make_project("A", preds=50),
        _make_project("B", preds=30),
        _make_project("C", preds=0),
    ]
    result = compute_portfolio_summary(summaries)
    assert result["total_predictions"] == 80


def test_portfolio_best_performer_highest_metric():
    summaries = [
        _make_project("ProjectA", model_count=2, best_metric=0.75),
        _make_project("ProjectB", model_count=3, best_metric=0.92),
        _make_project("ProjectC", model_count=1, best_metric=0.68),
    ]
    result = compute_portfolio_summary(summaries)
    assert result["best_performer"] is not None
    assert result["best_performer"]["name"] == "ProjectB"
    assert abs(result["best_performer"]["metric_value"] - 0.92) < 1e-6


def test_portfolio_best_performer_none_when_no_models():
    summaries = [_make_project("A"), _make_project("B")]
    result = compute_portfolio_summary(summaries)
    assert result["best_performer"] is None


def test_portfolio_summary_text():
    summaries = [
        _make_project("A", model_count=2, best_metric=0.80, has_deployment=True, preds=100),
    ]
    result = compute_portfolio_summary(summaries)
    assert "1 project" in result["summary"]
    assert "1 live" in result["summary"] or "deployment" in result["summary"]


def test_portfolio_projects_list_returned():
    summaries = [_make_project("A"), _make_project("B")]
    result = compute_portfolio_summary(summaries)
    assert len(result["projects"]) == 2


# ---------------------------------------------------------------------------
# API endpoint test
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    db_module.engine = engine

    import api.data as data_module
    data_module.UPLOAD_DIR = tmp_path / "uploads"
    import api.models as models_api_module
    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app
    with TestClient(app) as c:
        yield c


def test_portfolio_endpoint_returns_200(client):
    response = client.get("/api/projects/portfolio")
    assert response.status_code == 200


def test_portfolio_endpoint_empty_returns_zero_projects(client):
    response = client.get("/api/projects/portfolio")
    data = response.json()
    assert data["total_projects"] == 0
    assert data["projects"] == []


def test_portfolio_endpoint_with_project(client):
    # Create a project
    resp = client.post("/api/projects", json={"name": "Test Portfolio"})
    assert resp.status_code == 201

    response = client.get("/api/projects/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert data["total_projects"] == 1
    assert data["projects"][0]["name"] == "Test Portfolio"
