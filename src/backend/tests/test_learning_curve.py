"""Tests for Learning Curve Analysis.

Covers:
- _LEARNING_CURVE_PATTERNS detection (8 positive, 2 negative)
- compute_learning_curve() pure function (all branches)
- GET /api/models/{project_id}/learning-curve API integration
"""

import io as _io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


def test_pattern_would_more_data_help():
    from api.chat import _LEARNING_CURVE_PATTERNS

    assert _LEARNING_CURVE_PATTERNS.search("Would more data help my model?")


def test_pattern_learning_curve():
    from api.chat import _LEARNING_CURVE_PATTERNS

    assert _LEARNING_CURVE_PATTERNS.search("show me the learning curve")


def test_pattern_how_much_data_do_i_need():
    from api.chat import _LEARNING_CURVE_PATTERNS

    assert _LEARNING_CURVE_PATTERNS.search(
        "how much data do I need to improve accuracy?"
    )


def test_pattern_do_i_have_enough_data():
    from api.chat import _LEARNING_CURVE_PATTERNS

    assert _LEARNING_CURVE_PATTERNS.search(
        "do I have enough data to train a good model?"
    )


def test_pattern_is_dataset_big_enough():
    from api.chat import _LEARNING_CURVE_PATTERNS

    assert _LEARNING_CURVE_PATTERNS.search("is my training dataset big enough?")


def test_pattern_would_collecting_more_help():
    from api.chat import _LEARNING_CURVE_PATTERNS

    assert _LEARNING_CURVE_PATTERNS.search(
        "would collecting more rows improve performance?"
    )


def test_pattern_model_converged():
    from api.chat import _LEARNING_CURVE_PATTERNS

    assert _LEARNING_CURVE_PATTERNS.search("has my model converged?")


def test_pattern_training_size_analysis():
    from api.chat import _LEARNING_CURVE_PATTERNS

    assert _LEARNING_CURVE_PATTERNS.search("data size analysis for this model")


def test_pattern_negative_unrelated():
    from api.chat import _LEARNING_CURVE_PATTERNS

    assert not _LEARNING_CURVE_PATTERNS.search(
        "what is the correlation between revenue and units?"
    )


def test_pattern_negative_goal_training():
    from api.chat import _LEARNING_CURVE_PATTERNS

    # Goal training intent should not match
    assert not _LEARNING_CURVE_PATTERNS.search("I need 85% accuracy for this model")


# ---------------------------------------------------------------------------
# compute_learning_curve() pure function tests
# ---------------------------------------------------------------------------


def _make_regression_df(n: int = 150):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"feat1": rng.standard_normal(n), "feat2": rng.standard_normal(n)})
    y = pd.Series(2.0 * X["feat1"] + 0.5 * X["feat2"] + rng.standard_normal(n) * 0.2)
    return X, y


def _make_classification_df(n: int = 150):
    rng = np.random.default_rng(43)
    X = pd.DataFrame({"feat1": rng.standard_normal(n), "feat2": rng.standard_normal(n)})
    y = pd.Series((X["feat1"] + X["feat2"] > 0).astype(int))
    return X, y


def test_lc_regression_returns_required_fields():
    from core.trainer import compute_learning_curve

    X, y = _make_regression_df()
    result = compute_learning_curve(X, y, "linear_regression", "regression")

    required = [
        "sizes_pct",
        "train_scores",
        "val_scores",
        "converged",
        "plateau_pct",
        "best_val_score",
        "metric_label",
        "metric_key",
        "n_total",
        "algorithm",
        "algorithm_name",
        "recommendation",
        "summary",
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_lc_regression_sizes_match_scores():
    from core.trainer import compute_learning_curve

    X, y = _make_regression_df()
    result = compute_learning_curve(X, y, "linear_regression", "regression", n_sizes=4)

    assert len(result["sizes_pct"]) == len(result["train_scores"])
    assert len(result["sizes_pct"]) == len(result["val_scores"])


def test_lc_regression_metric_label_r2():
    from core.trainer import compute_learning_curve

    X, y = _make_regression_df()
    result = compute_learning_curve(X, y, "linear_regression", "regression")

    assert result["metric_label"] == "R²"
    assert result["metric_key"] == "r2"


def test_lc_classification_metric_label_accuracy():
    from core.trainer import compute_learning_curve

    X, y = _make_classification_df()
    result = compute_learning_curve(X, y, "logistic_regression", "classification")

    assert result["metric_label"] == "accuracy"
    assert result["metric_key"] == "accuracy"


def test_lc_n_total_equals_input_rows():
    from core.trainer import compute_learning_curve

    X, y = _make_regression_df(n=120)
    result = compute_learning_curve(X, y, "linear_regression", "regression")

    assert result["n_total"] == 120


def test_lc_best_val_score_non_negative():
    from core.trainer import compute_learning_curve

    X, y = _make_regression_df()
    result = compute_learning_curve(X, y, "linear_regression", "regression")

    # For a strongly linear dataset with linear regression, R² should be high
    assert result["best_val_score"] > 0.5


def test_lc_converged_has_plateau_pct():
    from core.trainer import compute_learning_curve

    # Use perfectly learnable data — large dataset should converge
    rng = np.random.default_rng(99)
    n = 400
    X = pd.DataFrame({"f": rng.standard_normal(n)})
    y = pd.Series(3.0 * X["f"] + rng.standard_normal(n) * 0.05)
    result = compute_learning_curve(X, y, "linear_regression", "regression", n_sizes=5)

    # If converged, plateau_pct must be set
    if result["converged"]:
        assert result["plateau_pct"] is not None
    else:
        # Not converged is also acceptable — just ensure plateau_pct is None
        assert result["plateau_pct"] is None


def test_lc_not_converged_plateau_pct_none():
    from core.trainer import compute_learning_curve

    # Very noisy data is unlikely to converge on a small sample
    rng = np.random.default_rng(7)
    n = 60
    X = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    y = pd.Series(rng.standard_normal(n))  # pure noise
    result = compute_learning_curve(X, y, "linear_regression", "regression", n_sizes=3)

    # plateau_pct should be None when not converged
    if not result["converged"]:
        assert result["plateau_pct"] is None


def test_lc_summary_is_string():
    from core.trainer import compute_learning_curve

    X, y = _make_regression_df()
    result = compute_learning_curve(X, y, "linear_regression", "regression")

    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_lc_recommendation_is_string():
    from core.trainer import compute_learning_curve

    X, y = _make_regression_df()
    result = compute_learning_curve(X, y, "linear_regression", "regression")

    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"]) > 0


def test_lc_too_few_rows_raises():
    from core.trainer import compute_learning_curve

    X = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="rows"):
        compute_learning_curve(X, y, "linear_regression", "regression")


def test_lc_unknown_algorithm_fallback():
    from core.trainer import compute_learning_curve

    X, y = _make_regression_df()
    # Unknown algorithm should fall back to linear_regression without error
    result = compute_learning_curve(X, y, "nonexistent_algo", "regression")

    assert result["algorithm"] == "linear_regression"


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

# Minimal CSV with 150 rows for training
_LC_SAMPLE_CSV = "feat1,feat2,target\n" + "\n".join(
    f"{i * 0.01:.3f},{(150 - i) * 0.01:.3f},{i * 0.02:.3f}" for i in range(150)
)


@pytest.fixture(autouse=True)
def _lc_test_db(tmp_path, monkeypatch):
    """Isolated SQLite DB for each integration test."""
    engine = create_engine(f"sqlite:///{tmp_path / 'lc.db'}")
    monkeypatch.setattr(db_module, "engine", engine)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture()
def lc_client():
    from main import app

    return TestClient(app)


@pytest.fixture()
def seeded_lc_project(lc_client, tmp_path):
    """Create project, upload CSV, apply features, set target, and train synchronously."""
    # Project
    proj = lc_client.post("/api/projects", json={"name": "LCTest"})
    project_id = proj.json()["id"]

    # Upload dataset (150 rows)
    upload = lc_client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("lc.csv", _io.BytesIO(_LC_SAMPLE_CSV.encode()), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    # Apply features + set target
    lc_client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    lc_client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "target", "problem_type": "regression"},
    )

    # Insert a completed ModelRun directly (avoids background thread timing)
    import json as _json
    from sqlmodel import Session
    from models.model_run import ModelRun

    with Session(db_module.engine) as session:
        run = ModelRun(
            project_id=project_id,
            algorithm="linear_regression",
            status="done",
            metrics=_json.dumps({"r2": 0.88, "mae": 0.05, "rmse": 0.07}),
            is_selected=True,
        )
        session.add(run)
        session.commit()

    return {"project_id": project_id, "dataset_id": dataset_id}


def test_lc_api_returns_200(lc_client, seeded_lc_project):
    project_id = seeded_lc_project["project_id"]
    resp = lc_client.get(f"/api/models/{project_id}/learning-curve")
    assert resp.status_code == 200


def test_lc_api_required_fields(lc_client, seeded_lc_project):
    project_id = seeded_lc_project["project_id"]
    resp = lc_client.get(f"/api/models/{project_id}/learning-curve")
    data = resp.json()

    assert "sizes_pct" in data
    assert "train_scores" in data
    assert "val_scores" in data
    assert "converged" in data
    assert "summary" in data
    assert "recommendation" in data
    assert data["project_id"] == project_id


def test_lc_api_unknown_project(lc_client):
    resp = lc_client.get("/api/models/nonexistent_project/learning-curve")
    assert resp.status_code == 404
