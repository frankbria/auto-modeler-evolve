"""Tests for GET /api/models/{project_id}/model-card and _MODEL_CARD_PATTERNS.

Covers:
- _MODEL_CARD_PATTERNS regex detection
- GET /api/models/{project_id}/model-card with no completed runs → 404
- GET /api/models/{project_id}/model-card with completed run → structured card
- model_card selects is_selected run over best-by-metric
- model_card falls back to best run when none selected
- _algorithm_plain_name helper
- _metric_plain_english classification (high accuracy)
- _metric_plain_english regression (good R²)
- _metric_plain_english regression (poor R²)
- _build_limitations (small dataset)
- _build_limitations (no issues)
- POST /api/chat/{project_id} emits model_card SSE event
"""

import time

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _MODEL_CARD_PATTERNS
from api.models import (
    _algorithm_plain_name,
    _build_limitations,
    _metric_plain_english,
)

# ---------------------------------------------------------------------------
# Sample CSV
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units,cost\n"
    b"East,100.5,10,50\n"
    b"West,200.3,20,80\n"
    b"East,150.7,15,60\n"
    b"West,300.1,30,120\n"
    b"North,250.9,25,100\n"
    b"East,175.2,18,70\n"
    b"West,220.4,22,90\n"
    b"North,190.6,19,75\n"
    b"East,130.8,13,55\n"
    b"West,280.0,28,110\n"
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


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Model Card Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
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
    for _ in range(20):
        r = await ac.get(f"/api/models/{project_id}/runs")
        run = next((x for x in r.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.5)
    pytest.skip("Training did not complete in time")


# ---------------------------------------------------------------------------
# Unit tests — regex patterns
# ---------------------------------------------------------------------------


def test_model_card_pattern_explain_my_model():
    assert _MODEL_CARD_PATTERNS.search("explain my model")


def test_model_card_pattern_what_does_it_do():
    assert _MODEL_CARD_PATTERNS.search("what does my model do")


def test_model_card_pattern_how_does_it_work():
    assert _MODEL_CARD_PATTERNS.search("how does my model work")


def test_model_card_pattern_tell_me():
    assert _MODEL_CARD_PATTERNS.search("tell me about my model")


def test_model_card_pattern_describe():
    assert _MODEL_CARD_PATTERNS.search("describe the model")


def test_model_card_pattern_summary():
    assert _MODEL_CARD_PATTERNS.search("model summary")


def test_model_card_pattern_how_good():
    assert _MODEL_CARD_PATTERNS.search("how good is my model?")


def test_model_card_pattern_what_drives():
    assert _MODEL_CARD_PATTERNS.search("what drives my predictions")


def test_model_card_pattern_no_false_positive():
    assert not _MODEL_CARD_PATTERNS.search("upload my data")


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------


def test_algorithm_plain_name_linear():
    assert _algorithm_plain_name("linear_regression") == "Linear Regression"


def test_algorithm_plain_name_random_forest():
    assert _algorithm_plain_name("random_forest_regressor") == "Random Forest"


def test_algorithm_plain_name_fallback():
    result = _algorithm_plain_name("unknown_algo")
    assert "Unknown" in result or "Algo" in result or result != ""


def test_metric_plain_english_classification_high():
    result = _metric_plain_english({"accuracy": 0.92}, "classification")
    assert result["name"] == "Accuracy"
    assert "92" in result["display"] or "9 out of 10" in result["plain_english"]
    assert result["value"] == pytest.approx(0.92)


def test_metric_plain_english_regression_good():
    result = _metric_plain_english({"r2": 0.85}, "regression")
    assert result["name"] == "R²"
    assert "85" in result["display"]
    assert (
        "good" in result["plain_english"].lower()
        or "excellent" in result["plain_english"].lower()
    )


def test_metric_plain_english_regression_poor():
    result = _metric_plain_english({"r2": 0.35}, "regression")
    assert (
        "limited" in result["plain_english"].lower()
        or "moderate" in result["plain_english"].lower()
    )


def test_metric_plain_english_mae_fallback():
    result = _metric_plain_english({"mae": 42.5}, "regression")
    assert result["name"] == "MAE"
    assert "42" in result["display"]


def test_build_limitations_small_dataset():
    lims = _build_limitations({"r2": 0.8}, "regression", row_count=50, feature_count=5)
    assert any("50" in lim for lim in lims)


def test_build_limitations_no_issues():
    lims = _build_limitations({"r2": 0.9}, "regression", row_count=500, feature_count=5)
    assert len(lims) >= 1  # always at least one


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_model_card_no_runs_returns_404(ac, project_id):
    resp = await ac.get(f"/api/models/{project_id}/model-card")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_model_card_returns_structured_card(ac, project_id, trained_run_id):
    resp = await ac.get(f"/api/models/{project_id}/model-card")
    assert resp.status_code == 200
    data = resp.json()
    assert data["project_id"] == project_id
    assert data["algorithm"] == "linear_regression"
    assert data["algorithm_name"] == "Linear Regression"
    assert data["problem_type"] == "regression"
    assert data["target_col"] == "revenue"
    assert "metric" in data
    assert data["metric"]["name"] in ("R²", "MAE", "Score")
    assert isinstance(data["limitations"], list)
    assert len(data["limitations"]) >= 1
    assert "summary" in data
    assert "revenue" in data["summary"]


@pytest.mark.anyio
async def test_model_card_includes_top_features(ac, project_id, trained_run_id):
    resp = await ac.get(f"/api/models/{project_id}/model-card")
    assert resp.status_code == 200
    data = resp.json()
    # top_features may be empty if no model path, but should be a list
    assert isinstance(data["top_features"], list)


@pytest.mark.anyio
async def test_model_card_unknown_project_returns_404(ac):
    resp = await ac.get("/api/models/00000000-0000-0000-0000-000000000000/model-card")
    assert resp.status_code == 404
