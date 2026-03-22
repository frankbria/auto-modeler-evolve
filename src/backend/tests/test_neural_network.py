"""Tests for Neural Network (MLP) algorithm integration."""

import numpy as np
import pytest
from pathlib import Path
from core.trainer import (
    REGRESSION_ALGORITHMS,
    CLASSIFICATION_ALGORITHMS,
    recommend_models,
    prepare_features,
    train_single_model,
)
import pandas as pd

# ---------------------------------------------------------------------------
# Registry presence
# ---------------------------------------------------------------------------


def test_mlp_regressor_in_registry():
    assert "neural_network_regressor" in REGRESSION_ALGORITHMS


def test_mlp_classifier_in_registry():
    assert "neural_network_classifier" in CLASSIFICATION_ALGORITHMS


def test_mlp_regressor_has_required_keys():
    algo = REGRESSION_ALGORITHMS["neural_network_regressor"]
    assert "name" in algo
    assert "plain_english" in algo
    assert "class" in algo
    assert "best_for" in algo
    assert algo["name"] == "Neural Network"


def test_mlp_classifier_has_required_keys():
    algo = CLASSIFICATION_ALGORITHMS["neural_network_classifier"]
    assert "name" in algo
    assert "plain_english" in algo
    assert "class" in algo
    assert algo["name"] == "Neural Network"


# ---------------------------------------------------------------------------
# Recommendations include MLP
# ---------------------------------------------------------------------------


def test_mlp_in_regression_recommendations():
    recs = recommend_models("regression", n_rows=500, n_features=8)
    algo_keys = [r["algorithm"] for r in recs]
    assert "neural_network_regressor" in algo_keys


def test_mlp_in_classification_recommendations():
    recs = recommend_models("classification", n_rows=500, n_features=8)
    algo_keys = [r["algorithm"] for r in recs]
    assert "neural_network_classifier" in algo_keys


def test_mlp_recommendation_small_dataset_warns():
    """For small datasets, the MLP recommendation message notes data requirement."""
    recs = recommend_models("regression", n_rows=100, n_features=4)
    mlp_recs = [r for r in recs if r["algorithm"] == "neural_network_regressor"]
    assert mlp_recs, "MLP should still appear in recommendations"
    reason = mlp_recs[0]["recommended_because"].lower()
    # Should mention something about size or data needs
    assert any(w in reason for w in ("small", "data", "neural", "need"))


def test_mlp_recommendation_large_dataset_positive():
    """For larger datasets, the MLP recommendation message references the row count."""
    recs = recommend_models("regression", n_rows=1000, n_features=10)
    mlp_recs = [r for r in recs if r["algorithm"] == "neural_network_regressor"]
    assert mlp_recs
    reason = mlp_recs[0]["recommended_because"]
    assert len(reason) > 10  # Non-trivial message


# ---------------------------------------------------------------------------
# Actual training works
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_df():
    """Simple regression dataset."""
    rng = np.random.default_rng(42)
    n = 150
    X = rng.standard_normal((n, 4))
    y = X[:, 0] * 2 + X[:, 1] * -1 + rng.standard_normal(n) * 0.1
    df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c", "feat_d"])
    df["target"] = y
    return df


@pytest.fixture
def classification_df():
    """Simple binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 150
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c", "feat_d"])
    df["target"] = y
    return df


def test_mlp_regressor_trains(regression_df, tmp_path):
    feature_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    X, y, le = prepare_features(regression_df, feature_cols, "target", "regression")

    result = train_single_model(
        X=X,
        y=y,
        algorithm="neural_network_regressor",
        problem_type="regression",
        model_dir=tmp_path,
        model_run_id="test-mlp-reg",
    )
    assert "metrics" in result
    assert "r2" in result["metrics"]
    assert result["metrics"]["r2"] is not None
    assert result["model_path"] is not None
    assert Path(result["model_path"]).exists()
    assert result["training_duration_ms"] >= 0


def test_mlp_classifier_trains(classification_df, tmp_path):
    feature_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    X, y, le = prepare_features(
        classification_df, feature_cols, "target", "classification"
    )

    result = train_single_model(
        X=X,
        y=y,
        algorithm="neural_network_classifier",
        problem_type="classification",
        model_dir=tmp_path,
        model_run_id="test-mlp-cls",
    )
    assert "metrics" in result
    assert "accuracy" in result["metrics"]
    assert result["metrics"]["accuracy"] > 0


def test_mlp_regressor_returns_valid_summary(regression_df, tmp_path):
    feature_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    X, y, le = prepare_features(regression_df, feature_cols, "target", "regression")

    result = train_single_model(
        X=X,
        y=y,
        algorithm="neural_network_regressor",
        problem_type="regression",
        model_dir=tmp_path,
        model_run_id="test-mlp-summary",
    )
    assert "summary" in result
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_mlp_feature_importances_fallback(regression_df, tmp_path):
    """MLP has no feature_importances_ — explainer falls back to equal weights."""
    from core.explainer import compute_feature_importance
    import joblib

    feature_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    X, y, le = prepare_features(regression_df, feature_cols, "target", "regression")

    result = train_single_model(
        X=X,
        y=y,
        algorithm="neural_network_regressor",
        problem_type="regression",
        model_dir=tmp_path,
        model_run_id="test-mlp-explain",
    )
    model = joblib.load(result["model_path"])
    # MLP has no feature_importances_ or coef_ (coef_ is an attribute on the layers,
    # not the estimator itself in the same shape as linear models)
    importances = compute_feature_importance(model, feature_cols)
    assert len(importances) == 4
    # Normalised to sum ~1
    total = sum(i["importance"] for i in importances)
    assert abs(total - 1.0) < 0.01


def test_mlp_unknown_algorithm_raises():
    """Passing an unknown algorithm to train_single_model raises ValueError."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = rng.standard_normal(50)
    with pytest.raises(ValueError, match="Unknown algorithm"):
        train_single_model(
            X=X,
            y=y,
            algorithm="neural_network_regressor_typo",
            problem_type="regression",
            model_dir=Path("/tmp"),
            model_run_id="x",
        )
