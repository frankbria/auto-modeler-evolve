"""Tests for compute_overfitting_analysis pure function and related chat patterns."""

import numpy as np
import pandas as pd
import pytest

from api.chat import _OVERFITTING_PATTERNS
from core.trainer import compute_overfitting_analysis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_regression_data(
    n: int = 80, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.normal(5, 2, n),
            "feature3": rng.normal(2, 1, n),
        }
    )
    y = pd.Series(2 * X["feature1"] + 0.5 * X["feature2"] + rng.normal(0, 0.3, n))
    return X, y


def _make_classification_data(
    n: int = 80, seed: int = 7
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.normal(0, 1, n),
        }
    )
    y = pd.Series((X["feature1"] + X["feature2"] > 0).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# Pure function: structure
# ---------------------------------------------------------------------------


def test_regression_result_has_required_keys():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    required = {
        "train_score",
        "cv_mean",
        "cv_std",
        "cv_scores",
        "gap",
        "gap_pct",
        "metric_key",
        "metric_label",
        "verdict",
        "verdict_label",
        "n_rows",
        "n_features",
        "algorithm_plain",
        "recommendations",
        "summary",
    }
    assert required.issubset(result.keys())


def test_classification_result_has_required_keys():
    X, y = _make_classification_data()
    result = compute_overfitting_analysis(X, y, "logistic_regression", "classification")
    required = {
        "train_score",
        "cv_mean",
        "cv_std",
        "cv_scores",
        "gap",
        "gap_pct",
        "metric_key",
        "metric_label",
        "verdict",
        "verdict_label",
        "n_rows",
        "n_features",
        "algorithm_plain",
        "recommendations",
        "summary",
    }
    assert required.issubset(result.keys())


def test_regression_metric_key():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert result["metric_key"] == "r2"
    assert result["metric_label"] == "R²"


def test_classification_metric_key():
    X, y = _make_classification_data()
    result = compute_overfitting_analysis(X, y, "logistic_regression", "classification")
    assert result["metric_key"] == "accuracy"
    assert result["metric_label"] == "Accuracy"


def test_n_rows_matches_input():
    X, y = _make_regression_data(n=60)
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert result["n_rows"] == 60


def test_n_features_matches_input():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert result["n_features"] == 3


def test_train_score_in_valid_range():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert -1.0 <= result["train_score"] <= 1.0


def test_cv_mean_in_valid_range():
    X, y = _make_classification_data()
    result = compute_overfitting_analysis(X, y, "logistic_regression", "classification")
    assert 0.0 <= result["cv_mean"] <= 1.0


def test_cv_std_non_negative():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert result["cv_std"] >= 0.0


def test_cv_scores_is_list():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert isinstance(result["cv_scores"], list)


def test_gap_equals_train_minus_cv():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    expected_gap = round(result["train_score"] - result["cv_mean"], 4)
    assert abs(result["gap"] - expected_gap) < 1e-6


def test_gap_pct_non_negative():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert result["gap_pct"] >= 0.0


def test_verdict_is_valid():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert result["verdict"] in ("well_fit", "mild_overfit", "overfit", "underfit")


def test_verdict_label_is_string():
    X, y = _make_classification_data()
    result = compute_overfitting_analysis(X, y, "logistic_regression", "classification")
    assert isinstance(result["verdict_label"], str)
    assert len(result["verdict_label"]) > 0


def test_algorithm_plain_is_string():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert result["algorithm_plain"] == "Linear Regression"


def test_algorithm_plain_random_forest():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "random_forest_regressor", "regression")
    assert result["algorithm_plain"] == "Random Forest"


def test_recommendations_is_list():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) >= 1


def test_summary_is_non_empty_string():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_too_few_rows_raises_value_error():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 0, 1])
    with pytest.raises(ValueError, match="at least 20"):
        compute_overfitting_analysis(X, y, "logistic_regression", "classification")


def test_unknown_algorithm_falls_back_gracefully():
    X, y = _make_regression_data()
    result = compute_overfitting_analysis(X, y, "nonexistent_algo", "regression")
    # Should fall back to linear regression without raising
    assert result["verdict"] in ("well_fit", "mild_overfit", "overfit", "underfit")


def test_ensemble_algorithm_falls_back():
    X, y = _make_regression_data()
    # Ensemble algorithms should fall back to Random Forest
    result = compute_overfitting_analysis(X, y, "voting_regressor", "regression")
    assert result["algorithm_plain"] == "Random Forest"


def test_well_fit_model_has_positive_recommendation():
    X, y = _make_regression_data(n=200)
    result = compute_overfitting_analysis(X, y, "linear_regression", "regression")
    if result["verdict"] == "well_fit":
        assert any("generalizes" in r.lower() for r in result["recommendations"])


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "is my model overfitting?",
        "is my model underfitting?",
        "check for overfitting",
        "detect underfitting",
        "show overfitting analysis",
        "overfitting check",
        "underfitting analysis",
        "overfitting report",
        "generalization gap",
        "training gap",
        "is my model memorizing the training data?",
        "is my model memorizing training data",
        "train score vs test",
        "train score versus validation",
        "model fit analysis",
        "model fitting check",
    ],
)
def test_pattern_matches_phrase(phrase):
    assert _OVERFITTING_PATTERNS.search(phrase), f"Expected match for: {phrase!r}"


@pytest.mark.parametrize(
    "phrase",
    [
        "how do I improve my model?",
        "feature importance",
        "what columns are useful?",
        "show me the confusion matrix",
        "train my model",
        "deploy my model",
        "what can I predict?",
    ],
)
def test_pattern_no_false_positives(phrase):
    assert not _OVERFITTING_PATTERNS.search(phrase), f"Unexpected match for: {phrase!r}"


def test_pattern_is_case_insensitive():
    assert _OVERFITTING_PATTERNS.search("IS MY MODEL OVERFITTING?")
    assert _OVERFITTING_PATTERNS.search("Check For Overfitting")
    assert _OVERFITTING_PATTERNS.search("GENERALIZATION GAP")
