"""Tests for compute_data_quality_impact pure function and related chat patterns."""

import numpy as np
import pandas as pd
import pytest

from api.chat import _DATA_QUALITY_IMPACT_PATTERNS
from core.trainer import compute_data_quality_impact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_regression_data(
    n: int = 60, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.normal(5, 2, n),
        }
    )
    y = pd.Series(2 * X["feature1"] + 0.5 * X["feature2"] + rng.normal(0, 0.1, n))
    return X, y


def _make_classification_data(
    n: int = 60, seed: int = 7
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
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    required = {
        "n_total",
        "n_outliers",
        "outlier_pct",
        "baseline_score",
        "clean_score",
        "delta",
        "metric_key",
        "metric_label",
        "verdict",
        "improvement",
        "recommendation",
        "summary",
    }
    assert required.issubset(result.keys())


def test_classification_result_has_required_keys():
    X, y = _make_classification_data()
    result = compute_data_quality_impact(X, y, "logistic_regression", "classification")
    required = {
        "n_total",
        "n_outliers",
        "outlier_pct",
        "baseline_score",
        "clean_score",
        "delta",
        "metric_key",
        "metric_label",
        "verdict",
        "improvement",
        "recommendation",
        "summary",
    }
    assert required.issubset(result.keys())


def test_regression_metric_key():
    X, y = _make_regression_data()
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    assert result["metric_key"] == "r2"
    assert result["metric_label"] == "R²"


def test_classification_metric_key():
    X, y = _make_classification_data()
    result = compute_data_quality_impact(X, y, "logistic_regression", "classification")
    assert result["metric_key"] == "accuracy"
    assert result["metric_label"] == "Accuracy"


def test_n_total_matches_input():
    X, y = _make_regression_data(n=80)
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    assert result["n_total"] == 80


def test_outlier_count_reasonable():
    X, y = _make_regression_data(n=60)
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    # With 5% contamination on 60 rows, expect ~3 outliers
    assert 0 <= result["n_outliers"] <= result["n_total"]


def test_outlier_pct_matches_count():
    X, y = _make_regression_data()
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    expected_pct = round(result["n_outliers"] / result["n_total"] * 100, 1)
    assert abs(result["outlier_pct"] - expected_pct) < 0.05


def test_delta_equals_clean_minus_baseline():
    X, y = _make_regression_data()
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    expected_delta = round(result["clean_score"] - result["baseline_score"], 4)
    assert abs(result["delta"] - expected_delta) < 1e-5


def test_scores_are_bounded_regression():
    X, y = _make_regression_data()
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    # R² can be negative for very bad models but should be in a reasonable range
    assert -10.0 <= result["baseline_score"] <= 1.0
    assert -10.0 <= result["clean_score"] <= 1.0


def test_scores_are_bounded_classification():
    X, y = _make_classification_data()
    result = compute_data_quality_impact(X, y, "logistic_regression", "classification")
    assert 0.0 <= result["baseline_score"] <= 1.0
    assert 0.0 <= result["clean_score"] <= 1.0


def test_verdict_is_valid():
    X, y = _make_regression_data()
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    assert result["verdict"] in {"worthwhile", "marginal", "no_benefit", "harmful"}


def test_improvement_flag_consistent_with_delta():
    X, y = _make_regression_data()
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    if result["delta"] > 0.01:
        assert result["improvement"] is True
    elif result["delta"] < -0.01:
        assert result["improvement"] is False


def test_recommendation_is_non_empty_string():
    X, y = _make_regression_data()
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"]) > 10


def test_summary_contains_row_counts():
    X, y = _make_regression_data(n=60)
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    # Summary should mention total rows
    assert "60" in result["summary"]


def test_summary_contains_metric_label():
    X, y = _make_regression_data()
    result = compute_data_quality_impact(X, y, "linear_regression", "regression")
    assert "R²" in result["summary"]


# ---------------------------------------------------------------------------
# Pure function: edge cases
# ---------------------------------------------------------------------------


def test_too_few_rows_raises():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([1, 2, 3])
    with pytest.raises(ValueError, match="rows"):
        compute_data_quality_impact(X, y, "linear_regression", "regression")


def test_unknown_algorithm_falls_back():
    X, y = _make_regression_data()
    result = compute_data_quality_impact(
        X, y, "nonexistent_algorithm_xyz", "regression"
    )
    # Should not raise — should fall back to linear_regression
    assert result["n_total"] == len(X)


def test_ensemble_algorithm_falls_back():
    X, y = _make_classification_data()
    # Ensemble models need extra infrastructure; should fall back gracefully
    result = compute_data_quality_impact(X, y, "voting_classifier", "classification")
    assert result["verdict"] in {"worthwhile", "marginal", "no_benefit", "harmful"}


# ---------------------------------------------------------------------------
# Chat regex patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "would removing outliers improve my model?",
        "Would removing the outliers help my accuracy?",
        "data quality impact on model",
        "impact of outliers on my model",
        "impact of removing bad rows on training",
        "how would cleaning the data help my model?",
        "how would cleaning help my model?",
        "how would removing outliers affect my model?",
        "effect of removing bad rows on my training",
        "outlier removal impact on accuracy",
        "what if I removed the outliers from training?",
        "would cleaner training data improve my accuracy?",
        "What if we cleaned the anomalies from my dataset?",
    ],
)
def test_pattern_matches(message: str):
    assert _DATA_QUALITY_IMPACT_PATTERNS.search(message), f"Should match: {message!r}"


@pytest.mark.parametrize(
    "message",
    [
        "what are the outliers in my data?",
        "show me anomalies",
        "how accurate is my model?",
        "feature engineering impact",
        "train a new model",
        "compare my models",
        "deploy my model",
    ],
)
def test_pattern_false_positives(message: str):
    assert not _DATA_QUALITY_IMPACT_PATTERNS.search(message), (
        f"Should NOT match: {message!r}"
    )


def test_pattern_case_insensitive():
    assert _DATA_QUALITY_IMPACT_PATTERNS.search(
        "WOULD REMOVING OUTLIERS IMPROVE MY MODEL?"
    )
    assert _DATA_QUALITY_IMPACT_PATTERNS.search("DATA QUALITY IMPACT ON MODEL")
