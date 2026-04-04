"""Tests for core/advisor.py — Model Improvement Advisor."""

import pytest

from core.advisor import (
    compute_improvement_suggestions,
    _primary_metric,
    _check_weak_features,
    _check_ensemble,
    _check_date_features,
    _check_data_volume,
    _check_imbalance,
    _check_calibration,
    _check_tuning,
    _check_feature_count,
    _check_linear_vs_nonlinear,
)


# ---------------------------------------------------------------------------
# _primary_metric
# ---------------------------------------------------------------------------


def test_primary_metric_regression():
    metrics = {"r2": 0.75, "mae": 120.0}
    val, name = _primary_metric(metrics, "regression")
    assert val == 0.75
    assert name == "R²"


def test_primary_metric_classification_accuracy():
    metrics = {"accuracy": 0.88, "f1": 0.84}
    val, name = _primary_metric(metrics, "classification")
    assert val == 0.88
    assert name == "accuracy"


def test_primary_metric_classification_fallback_f1():
    metrics = {"f1": 0.72}
    val, name = _primary_metric(metrics, "classification")
    assert val == 0.72


def test_primary_metric_missing_returns_zero():
    val, _ = _primary_metric({}, "regression")
    assert val == 0.0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def test_check_weak_features_adds_suggestion():
    out: list = []
    _check_weak_features(out, n_weak=3)
    assert len(out) == 1
    s = out[0]
    assert s["category"] == "features"
    assert "3" in s["title"]
    assert s["difficulty"] == "easy"
    assert s["expected_impact"] == "moderate"


def test_check_weak_features_singular():
    out: list = []
    _check_weak_features(out, n_weak=1)
    assert "1 Weak Feature" in out[0]["title"]


def test_check_weak_features_zero_no_suggestion():
    out: list = []
    _check_weak_features(out, n_weak=0)
    assert len(out) == 0


def test_check_ensemble_adds_for_low_regression():
    out: list = []
    _check_ensemble(
        out, is_ensemble=False, primary_metric=0.60, problem_type="regression"
    )
    assert len(out) == 1
    assert out[0]["action"] == "train_ensemble"
    assert out[0]["expected_impact"] == "high"


def test_check_ensemble_skipped_for_already_ensemble():
    out: list = []
    _check_ensemble(
        out, is_ensemble=True, primary_metric=0.50, problem_type="regression"
    )
    assert len(out) == 0


def test_check_ensemble_skipped_when_high_r2():
    out: list = []
    _check_ensemble(
        out, is_ensemble=False, primary_metric=0.85, problem_type="regression"
    )
    assert len(out) == 0


def test_check_ensemble_classification_threshold():
    out: list = []
    _check_ensemble(
        out, is_ensemble=False, primary_metric=0.80, problem_type="classification"
    )
    # 0.80 < 0.85 threshold → should add
    assert len(out) == 1


def test_check_date_features_adds_when_unused():
    out: list = []
    _check_date_features(
        out, has_date_col=True, date_col_used=False, algorithm="random_forest_regressor"
    )
    assert len(out) == 1
    assert out[0]["action"] == "feature_engineering"
    assert out[0]["difficulty"] == "easy"


def test_check_date_features_skipped_when_already_used():
    out: list = []
    _check_date_features(
        out, has_date_col=True, date_col_used=True, algorithm="random_forest_regressor"
    )
    assert len(out) == 0


def test_check_date_features_skipped_when_no_date():
    out: list = []
    _check_date_features(
        out, has_date_col=False, date_col_used=False, algorithm="linear_regression"
    )
    assert len(out) == 0


def test_check_data_volume_adds_for_small_dataset():
    out: list = []
    _check_data_volume(out, n_rows=500)
    assert len(out) == 1
    s = out[0]
    assert s["action"] == "add_data"
    assert s["expected_impact"] == "high"
    assert s["difficulty"] == "hard"
    assert "500" in s["explanation"]


def test_check_data_volume_skipped_for_large_dataset():
    out: list = []
    _check_data_volume(out, n_rows=5000)
    assert len(out) == 0


def test_check_imbalance_adds_for_unhandled_classification():
    out: list = []
    _check_imbalance(
        out,
        class_is_imbalanced=True,
        imbalance_strategy=None,
        problem_type="classification",
    )
    assert len(out) == 1
    assert out[0]["action"] == "class_imbalance"
    assert out[0]["expected_impact"] == "high"


def test_check_imbalance_skipped_for_regression():
    out: list = []
    _check_imbalance(
        out,
        class_is_imbalanced=True,
        imbalance_strategy=None,
        problem_type="regression",
    )
    assert len(out) == 0


def test_check_imbalance_skipped_when_strategy_already_applied():
    out: list = []
    _check_imbalance(
        out,
        class_is_imbalanced=True,
        imbalance_strategy="smote",
        problem_type="classification",
    )
    assert len(out) == 0


def test_check_imbalance_skipped_when_not_imbalanced():
    out: list = []
    _check_imbalance(
        out,
        class_is_imbalanced=False,
        imbalance_strategy=None,
        problem_type="classification",
    )
    assert len(out) == 0


def test_check_calibration_adds_for_uncalibrated_classifier():
    out: list = []
    _check_calibration(
        out, is_calibrated=False, problem_type="classification", metrics={}
    )
    assert len(out) == 1
    assert out[0]["action"] == "calibration"


def test_check_calibration_skipped_for_regression():
    out: list = []
    _check_calibration(out, is_calibrated=False, problem_type="regression", metrics={})
    assert len(out) == 0


def test_check_calibration_skipped_if_already_calibrated():
    out: list = []
    _check_calibration(
        out, is_calibrated=True, problem_type="classification", metrics={}
    )
    assert len(out) == 0


def test_check_calibration_skipped_for_low_brier():
    out: list = []
    _check_calibration(
        out,
        is_calibrated=False,
        problem_type="classification",
        metrics={"brier_score": 0.05},
    )
    assert len(out) == 0


def test_check_tuning_adds_for_nonlinear_underperformer():
    out: list = []
    _check_tuning(
        out,
        algorithm="random_forest_regressor",
        is_ensemble=False,
        primary_metric=0.70,
        problem_type="regression",
    )
    assert len(out) == 1
    assert out[0]["action"] == "hyperparameter_tuning"
    assert out[0]["difficulty"] == "easy"


def test_check_tuning_skipped_for_linear_regression():
    out: list = []
    _check_tuning(
        out,
        algorithm="linear_regression",
        is_ensemble=False,
        primary_metric=0.60,
        problem_type="regression",
    )
    assert len(out) == 0


def test_check_tuning_skipped_for_high_performance():
    out: list = []
    _check_tuning(
        out,
        algorithm="random_forest_regressor",
        is_ensemble=False,
        primary_metric=0.90,
        problem_type="regression",
    )
    assert len(out) == 0


def test_check_tuning_skipped_for_ensemble():
    out: list = []
    _check_tuning(
        out,
        algorithm="voting_regressor",
        is_ensemble=True,
        primary_metric=0.65,
        problem_type="regression",
    )
    assert len(out) == 0


def test_check_feature_count_adds_for_low_features():
    out: list = []
    _check_feature_count(out, n_features=2)
    assert len(out) == 1
    assert out[0]["action"] == "add_features"
    assert out[0]["expected_impact"] == "high"


def test_check_feature_count_skipped_for_sufficient():
    out: list = []
    _check_feature_count(out, n_features=5)
    assert len(out) == 0


def test_check_linear_vs_nonlinear_adds_for_weak_linear():
    out: list = []
    _check_linear_vs_nonlinear(
        out,
        algorithm="linear_regression",
        primary_metric=0.50,
        problem_type="regression",
    )
    assert len(out) == 1
    assert out[0]["action"] == "train_nonlinear"
    assert out[0]["expected_impact"] == "high"


def test_check_linear_vs_nonlinear_skipped_for_nonlinear_algo():
    out: list = []
    _check_linear_vs_nonlinear(
        out,
        algorithm="random_forest_regressor",
        primary_metric=0.50,
        problem_type="regression",
    )
    assert len(out) == 0


def test_check_linear_vs_nonlinear_skipped_when_good():
    out: list = []
    _check_linear_vs_nonlinear(
        out,
        algorithm="linear_regression",
        primary_metric=0.75,
        problem_type="regression",
    )
    assert len(out) == 0


# ---------------------------------------------------------------------------
# Full function
# ---------------------------------------------------------------------------


def test_compute_improvement_suggestions_returns_structure():
    result = compute_improvement_suggestions(
        metrics={"r2": 0.65, "mae": 200.0},
        algorithm="random_forest_regressor",
        problem_type="regression",
        n_features=5,
        n_rows=800,
    )
    assert "suggestions" in result
    assert "summary" in result
    assert "n_suggestions" in result
    assert "primary_metric" in result
    assert "primary_metric_name" in result
    assert result["primary_metric_name"] == "R²"
    assert result["primary_metric"] == pytest.approx(0.65)


def test_compute_improvement_suggestions_sorted_by_impact():
    result = compute_improvement_suggestions(
        metrics={"r2": 0.55},
        algorithm="linear_regression",
        problem_type="regression",
        n_features=2,
        n_rows=300,
        has_date_col=True,
        date_col_used=False,
        n_weak_features=0,
        is_ensemble=False,
        is_calibrated=False,
    )
    suggestions = result["suggestions"]
    assert len(suggestions) > 0
    # Verify ranks are sequential
    for i, s in enumerate(suggestions):
        assert s["rank"] == i + 1
    # High impact suggestions should come first
    impacts = [s["expected_impact"] for s in suggestions]
    order = {"high": 0, "moderate": 1, "low": 2}
    for i in range(len(impacts) - 1):
        assert order[impacts[i]] <= order[impacts[i + 1]]


def test_compute_improvement_suggestions_empty_for_excellent_model():
    result = compute_improvement_suggestions(
        metrics={"r2": 0.95},
        algorithm="random_forest_regressor",
        problem_type="regression",
        n_features=10,
        n_rows=10000,
        has_date_col=False,
        date_col_used=False,
        n_weak_features=0,
        is_ensemble=False,
        is_calibrated=False,
    )
    # Should have few or no suggestions (tuning might still appear)
    suggestions = result["suggestions"]
    for s in suggestions:
        # No data volume suggestion for 10k rows
        assert s["action"] != "add_data"


def test_compute_improvement_suggestions_classification():
    result = compute_improvement_suggestions(
        metrics={"accuracy": 0.72, "f1": 0.68},
        algorithm="random_forest_classifier",
        problem_type="classification",
        n_features=6,
        n_rows=1500,
        class_is_imbalanced=True,
        imbalance_strategy=None,
        is_calibrated=False,
    )
    actions = [s["action"] for s in result["suggestions"]]
    assert "class_imbalance" in actions  # imbalance unhandled


def test_compute_improvement_summary_contains_metric():
    result = compute_improvement_suggestions(
        metrics={"r2": 0.72},
        algorithm="random_forest_regressor",
        problem_type="regression",
        n_features=5,
        n_rows=5000,
    )
    assert "72%" in result["summary"]


def test_compute_improvement_n_suggestions_matches_list():
    result = compute_improvement_suggestions(
        metrics={"r2": 0.60},
        algorithm="random_forest_regressor",
        problem_type="regression",
        n_features=2,
        n_rows=300,
        has_date_col=True,
    )
    assert result["n_suggestions"] == len(result["suggestions"])


def test_compute_improvement_all_fields_present_in_each_suggestion():
    result = compute_improvement_suggestions(
        metrics={"r2": 0.55},
        algorithm="linear_regression",
        problem_type="regression",
        n_features=3,
        n_rows=400,
    )
    required_keys = {
        "rank",
        "category",
        "title",
        "explanation",
        "action",
        "difficulty",
        "expected_impact",
    }
    for s in result["suggestions"]:
        assert required_keys.issubset(s.keys()), f"Missing keys in {s}"


def test_compute_improvement_weak_features_suggestion_appears():
    result = compute_improvement_suggestions(
        metrics={"r2": 0.70},
        algorithm="random_forest_regressor",
        problem_type="regression",
        n_features=5,
        n_rows=3000,
        n_weak_features=2,
    )
    actions = [s["action"] for s in result["suggestions"]]
    assert "feature_selection" in actions
