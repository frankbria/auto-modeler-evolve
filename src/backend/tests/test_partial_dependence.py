"""Tests for Partial Dependence Plot (PDP) via chat.

Covers:
- _PDP_PATTERNS detection (8 positive, 2 negative)
- _detect_pdp_feature() helper
- compute_partial_dependence() pure function (regression, binary-cls, multiclass, constant)
- GET /api/models/{run_id}/partial-dependence API endpoint
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression, LogisticRegression
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


def test_pattern_partial_dependence_for():
    from api.chat import _PDP_PATTERNS

    assert _PDP_PATTERNS.search("partial dependence for price")


def test_pattern_pdp_for():
    from api.chat import _PDP_PATTERNS

    assert _PDP_PATTERNS.search("PDP for units")


def test_pattern_marginal_effect():
    from api.chat import _PDP_PATTERNS

    assert _PDP_PATTERNS.search("marginal effect of discount on revenue")


def test_pattern_partial_dependence_bare():
    from api.chat import _PDP_PATTERNS

    assert _PDP_PATTERNS.search("show partial dependence")


def test_pattern_average_effect():
    from api.chat import _PDP_PATTERNS

    assert _PDP_PATTERNS.search("average effect of units on the prediction")


def test_pattern_population_level():
    from api.chat import _PDP_PATTERNS

    assert _PDP_PATTERNS.search("population-level effect of cost on churn")


def test_pattern_pdp_on():
    from api.chat import _PDP_PATTERNS

    assert _PDP_PATTERNS.search("pdp on revenue_usd")


def test_pattern_partial_dependence_plot_for():
    from api.chat import _PDP_PATTERNS

    assert _PDP_PATTERNS.search("partial dependence plot for quantity")


def test_pattern_sensitivity_not_matched():
    """Sensitivity analysis should NOT match the PDP pattern (different intent)."""
    from api.chat import _PDP_PATTERNS

    # sensitivity phrasing without "partial dependence" or "pdp" or "marginal effect"
    assert not _PDP_PATTERNS.search("sensitivity analysis on price")


def test_pattern_unrelated_not_matched():
    from api.chat import _PDP_PATTERNS

    assert not _PDP_PATTERNS.search("train a new model")


# ---------------------------------------------------------------------------
# _detect_pdp_feature() helper
# ---------------------------------------------------------------------------


def test_detect_feature_exact_match():
    from api.chat import _detect_pdp_feature

    assert _detect_pdp_feature("PDP for units", ["units", "price", "region"]) == "units"


def test_detect_feature_underscore_to_space():
    from api.chat import _detect_pdp_feature

    # "revenue usd" in message should match "revenue_usd" feature
    assert (
        _detect_pdp_feature(
            "partial dependence for revenue usd", ["revenue_usd", "units"]
        )
        == "revenue_usd"
    )


def test_detect_feature_no_match_returns_none():
    from api.chat import _detect_pdp_feature

    assert _detect_pdp_feature("partial dependence plot", ["units", "price"]) is None


def test_detect_feature_longest_match_wins():
    from api.chat import _detect_pdp_feature

    # "product_category" should beat "category"
    feat = _detect_pdp_feature(
        "PDP for product_category",
        ["category", "product_category", "units"],
    )
    assert feat == "product_category"


# ---------------------------------------------------------------------------
# compute_partial_dependence() pure function
# ---------------------------------------------------------------------------


def _make_regression_model():
    """Return a tiny fitted LinearRegression with known coefficients."""
    np.random.seed(0)
    X = np.column_stack(
        [
            np.linspace(1, 10, 50),  # feature 0 (units)
            np.random.randn(50),  # feature 1 (noise)
        ]
    )
    y = 2 * X[:, 0] + np.random.randn(50) * 0.1
    model = LinearRegression().fit(X, y)
    return model, X


def _make_binary_classifier():
    """Return a tiny fitted LogisticRegression (binary)."""
    np.random.seed(42)
    X = np.column_stack(
        [
            np.linspace(0, 1, 80),
            np.random.randn(80),
        ]
    )
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegression(max_iter=500).fit(X, y)
    return model, X


def _make_multiclass_classifier():
    """Return a tiny fitted LogisticRegression (3 classes)."""
    np.random.seed(7)
    X = np.column_stack(
        [
            np.linspace(0, 2, 90),
            np.random.randn(90),
        ]
    )
    y = (X[:, 0] * 1.5).astype(int).clip(0, 2)
    model = LogisticRegression(max_iter=500).fit(X, y)
    return model, X


def test_pdp_regression_returns_expected_keys():
    from core.explainer import compute_partial_dependence

    model, X = _make_regression_model()
    grid = np.linspace(1, 10, 10)
    result = compute_partial_dependence(model, X, 0, grid, "regression")

    assert "grid_values" in result
    assert "mean_predictions" in result
    assert "std_predictions" in result
    assert "summary" in result
    assert "n_training_rows" in result
    assert "problem_type" in result


def test_pdp_regression_grid_length_matches():
    from core.explainer import compute_partial_dependence

    model, X = _make_regression_model()
    grid = np.linspace(1, 10, 15)
    result = compute_partial_dependence(model, X, 0, grid, "regression")

    assert len(result["grid_values"]) == 15
    assert len(result["mean_predictions"]) == 15
    assert len(result["std_predictions"]) == 15


def test_pdp_regression_trend_is_positive():
    """LinearRegression with coef=2 on feature 0 → predictions increase."""
    from core.explainer import compute_partial_dependence

    model, X = _make_regression_model()
    grid = np.linspace(1, 10, 10)
    result = compute_partial_dependence(model, X, 0, grid, "regression")

    preds = result["mean_predictions"]
    assert preds[-1] > preds[0], "Predictions should increase as feature 0 increases"


def test_pdp_regression_n_training_rows():
    from core.explainer import compute_partial_dependence

    model, X = _make_regression_model()
    grid = np.linspace(1, 10, 5)
    result = compute_partial_dependence(model, X, 0, grid, "regression")

    assert result["n_training_rows"] == len(X)


def test_pdp_regression_summary_text():
    from core.explainer import compute_partial_dependence

    model, X = _make_regression_model()
    grid = np.linspace(1, 10, 5)
    result = compute_partial_dependence(model, X, 0, grid, "regression")

    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 10


def test_pdp_binary_classification_probabilities():
    """Binary classification PDP should return values in [0, 1]."""
    from core.explainer import compute_partial_dependence

    model, X = _make_binary_classifier()
    grid = np.linspace(0, 1, 10)
    result = compute_partial_dependence(model, X, 0, grid, "classification")

    for v in result["mean_predictions"]:
        assert 0.0 <= v <= 1.0


def test_pdp_binary_classification_no_class_curves():
    """Binary classification should NOT produce class_curves (only multiclass does)."""
    from core.explainer import compute_partial_dependence

    model, X = _make_binary_classifier()
    grid = np.linspace(0, 1, 5)
    result = compute_partial_dependence(model, X, 0, grid, "classification")

    # class_curves may be None or absent for binary
    assert result.get("class_curves") is None


def test_pdp_multiclass_class_curves():
    """Multiclass classification should produce class_curves dict."""
    from core.explainer import compute_partial_dependence

    model, X = _make_multiclass_classifier()
    grid = np.linspace(0, 2, 5)
    result = compute_partial_dependence(
        model, X, 0, grid, "classification", class_names=["A", "B", "C"]
    )

    assert result.get("class_curves") is not None
    assert "A" in result["class_curves"]
    assert "B" in result["class_curves"]
    assert "C" in result["class_curves"]
    # Each curve has same length as grid
    for cls_curve in result["class_curves"].values():
        assert len(cls_curve) == 5


def test_pdp_constant_feature_single_point():
    """A feature with p5 == p95 (constant) results in a single grid point."""
    from core.explainer import compute_partial_dependence

    model, X = _make_regression_model()
    # Constant grid (single value)
    grid = np.array([5.0])
    result = compute_partial_dependence(model, X, 0, grid, "regression")

    assert len(result["grid_values"]) == 1
    assert len(result["mean_predictions"]) == 1


def test_pdp_problem_type_preserved():
    from core.explainer import compute_partial_dependence

    model, X = _make_regression_model()
    grid = np.linspace(1, 10, 5)
    result = compute_partial_dependence(model, X, 0, grid, "regression")

    assert result["problem_type"] == "regression"


# ---------------------------------------------------------------------------
# API endpoint integration tests
# ---------------------------------------------------------------------------

_PDP_CSV = "units,price,target\n" + "\n".join(
    f"{i},{i * 0.5 + 1.0},{i * 2.0 + 0.1}" for i in range(1, 101)
)


@pytest.fixture(autouse=True)
def _pdp_test_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'pdp.db'}")
    monkeypatch.setattr(db_module, "engine", engine)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture()
def pdp_client():
    from main import app

    return TestClient(app)


@pytest.fixture()
def seeded_pdp_run(pdp_client, tmp_path):
    """Create project + dataset + trained model for PDP endpoint tests."""
    # Project
    proj = pdp_client.post("/api/projects", json={"name": "PDPTest"})
    project_id = proj.json()["id"]

    # Upload CSV
    upload = pdp_client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("pdp.csv", io.BytesIO(_PDP_CSV.encode()), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    # Apply features + set target
    pdp_client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    pdp_client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "target", "problem_type": "regression"},
    )

    # Train a real model so model_path exists on disk
    import joblib
    from sqlmodel import Session
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun

    # Get feature_set_id
    with Session(db_module.engine) as session:
        from sqlmodel import select
        from models.dataset import Dataset

        ds = session.get(Dataset, dataset_id)
        fs_stmt = select(FeatureSet).where(FeatureSet.dataset_id == dataset_id)
        fs = session.exec(fs_stmt).first()
        assert fs is not None

        import pandas as pd
        from core.trainer import prepare_features

        df = pd.read_csv(ds.file_path)
        feature_cols = ["units", "price"]
        X, y, _ = prepare_features(df, feature_cols, "target", "regression")

        model = LinearRegression().fit(X, y)
        model_path = str(tmp_path / "test_model.joblib")
        joblib.dump(model, model_path)

        run = ModelRun(
            project_id=project_id,
            feature_set_id=fs.id,
            algorithm="linear_regression",
            status="done",
            model_path=model_path,
            metrics=json.dumps({"r2": 0.99, "mae": 0.05, "rmse": 0.07}),
            is_selected=True,
        )
        session.add(run)
        session.commit()
        run_id = run.id

    return {"project_id": project_id, "dataset_id": dataset_id, "run_id": run_id}


def test_pdp_api_200(pdp_client, seeded_pdp_run):
    run_id = seeded_pdp_run["run_id"]
    resp = pdp_client.get(f"/api/models/{run_id}/partial-dependence?feature=units")
    assert resp.status_code == 200


def test_pdp_api_response_fields(pdp_client, seeded_pdp_run):
    run_id = seeded_pdp_run["run_id"]
    resp = pdp_client.get(f"/api/models/{run_id}/partial-dependence?feature=units")
    data = resp.json()

    assert "grid_values" in data
    assert "mean_predictions" in data
    assert "std_predictions" in data
    assert "summary" in data
    assert "feature" in data
    assert data["feature"] == "units"


def test_pdp_api_steps_respected(pdp_client, seeded_pdp_run):
    run_id = seeded_pdp_run["run_id"]
    resp = pdp_client.get(
        f"/api/models/{run_id}/partial-dependence?feature=units&steps=12"
    )
    data = resp.json()

    assert len(data["grid_values"]) == 12


def test_pdp_api_unknown_feature_400(pdp_client, seeded_pdp_run):
    run_id = seeded_pdp_run["run_id"]
    resp = pdp_client.get(
        f"/api/models/{run_id}/partial-dependence?feature=nonexistent_col"
    )
    assert resp.status_code == 400


def test_pdp_api_not_found_404(pdp_client):
    resp = pdp_client.get(
        "/api/models/00000000-0000-0000-0000-000000000000/partial-dependence?feature=units"
    )
    assert resp.status_code == 404
