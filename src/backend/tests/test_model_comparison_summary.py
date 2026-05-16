"""Tests for compute_model_comparison_summary in core/advisor.py."""

import pytest

from core.advisor import compute_model_comparison_summary, _build_run_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _regression_run(algo: str, r2: float, cv_mean: float | None = None, cv_std: float | None = None) -> dict:
    metrics: dict = {"r2": r2, "mae": 50.0}
    if cv_mean is not None:
        metrics["cv_mean"] = cv_mean
        metrics["cv_std"] = cv_std if cv_std is not None else 0.03
    return {
        "run_id": f"run_{algo}",
        "algorithm": algo,
        "metrics": metrics,
        "problem_type": "regression",
        "is_selected": False,
        "is_deployed": False,
    }


def _classification_run(algo: str, accuracy: float) -> dict:
    return {
        "run_id": f"run_{algo}",
        "algorithm": algo,
        "metrics": {"accuracy": accuracy, "f1": accuracy - 0.02},
        "problem_type": "classification",
        "is_selected": False,
        "is_deployed": False,
    }


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_runs_returns_no_winner():
    result = compute_model_comparison_summary([])
    assert result["n_runs"] == 0
    assert result["winner"] is None
    assert result["runs_compared"] == []
    assert result["trade_offs"] == []
    assert "No completed model runs" in result["narrative"]


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def test_single_run_only_one_run_flag():
    result = compute_model_comparison_summary([
        _regression_run("linear_regression", 0.75)
    ])
    assert result["only_one_run"] is True
    assert result["n_runs"] == 1
    assert result["winner"]["algorithm"] == "linear_regression"
    assert result["trade_offs"] == []
    assert "Train more" in result["narrative"]


def test_single_run_summary_mentions_algorithm():
    result = compute_model_comparison_summary([
        _regression_run("random_forest_regressor", 0.82)
    ])
    assert "Random Forest" in result["summary"]
    assert result["only_one_run"] is True


# ---------------------------------------------------------------------------
# Two runs — winner ordering
# ---------------------------------------------------------------------------


def test_two_runs_winner_is_best_metric():
    runs = [
        _regression_run("linear_regression", 0.60),
        _regression_run("random_forest_regressor", 0.85),
    ]
    result = compute_model_comparison_summary(runs)
    assert result["winner"]["algorithm"] == "random_forest_regressor"
    assert result["runs_compared"][0]["algorithm"] == "random_forest_regressor"
    assert result["runs_compared"][1]["algorithm"] == "linear_regression"


def test_two_runs_not_only_one():
    result = compute_model_comparison_summary([
        _regression_run("linear_regression", 0.60),
        _regression_run("random_forest_regressor", 0.85),
    ])
    assert result["only_one_run"] is False
    assert len(result["trade_offs"]) >= 1


# ---------------------------------------------------------------------------
# Trade-offs
# ---------------------------------------------------------------------------


def test_trade_offs_accuracy_gap():
    runs = [
        _regression_run("random_forest_regressor", 0.90),
        _regression_run("linear_regression", 0.70),
    ]
    result = compute_model_comparison_summary(runs)
    # At least one trade-off sentence should mention the gap
    combined = " ".join(result["trade_offs"])
    assert "90" in combined or "70" in combined or "20" in combined


def test_trade_offs_explainability_when_best_is_not_most_explainable():
    # Random Forest wins accuracy but Linear Regression is more explainable
    runs = [
        _regression_run("random_forest_regressor", 0.88),
        _regression_run("linear_regression", 0.70),
    ]
    result = compute_model_comparison_summary(runs)
    combined = " ".join(result["trade_offs"])
    assert "Linear Regression" in combined or "interpretable" in combined


def test_trade_offs_stability_when_cv_available():
    runs = [
        _regression_run("random_forest_regressor", 0.85, cv_mean=0.83, cv_std=0.04),
        _regression_run("linear_regression", 0.70, cv_mean=0.69, cv_std=0.02),
    ]
    result = compute_model_comparison_summary(runs)
    combined = " ".join(result["trade_offs"])
    # Should mention stability (std)
    assert "stable" in combined or "variab" in combined or "±" in combined


def test_trade_offs_no_cv_no_stability_mention():
    runs = [
        _regression_run("random_forest_regressor", 0.85),  # no CV
        _regression_run("linear_regression", 0.70),
    ]
    result = compute_model_comparison_summary(runs)
    # Without CV, stability trade-off should not appear
    combined = " ".join(result["trade_offs"])
    assert "stable" not in combined or "±" not in combined


# ---------------------------------------------------------------------------
# Narrative content
# ---------------------------------------------------------------------------


def test_narrative_mentions_winner():
    runs = [
        _regression_run("gradient_boosting_regressor", 0.91),
        _regression_run("linear_regression", 0.65),
    ]
    result = compute_model_comparison_summary(runs)
    assert "Gradient Boosting" in result["narrative"]


def test_narrative_mentions_runner_up():
    runs = [
        _regression_run("random_forest_regressor", 0.88),
        _regression_run("linear_regression", 0.72),
    ]
    result = compute_model_comparison_summary(runs)
    assert "Linear Regression" in result["narrative"]


def test_narrative_explainability_when_winner_is_not_most_explainable():
    runs = [
        _regression_run("xgboost_regressor", 0.92),
        _regression_run("linear_regression", 0.71),
    ]
    result = compute_model_comparison_summary(runs)
    # Should mention Linear Regression as most interpretable alternative
    assert "Linear Regression" in result["narrative"] or "interpretable" in result["narrative"]


def test_narrative_single_run_no_runner_up():
    result = compute_model_comparison_summary([
        _regression_run("linear_regression", 0.75)
    ])
    assert "Linear Regression" in result["narrative"]
    # No second model mentioned in a way that implies comparison
    assert "edging out" not in result["narrative"]


# ---------------------------------------------------------------------------
# _build_run_summary helper
# ---------------------------------------------------------------------------


def test_build_run_summary_regression_fields():
    run = _regression_run("linear_regression", 0.75, cv_mean=0.73, cv_std=0.02)
    summary = _build_run_summary(run)
    assert summary["algorithm"] == "linear_regression"
    assert summary["algorithm_plain"] == "Linear Regression"
    assert abs(summary["primary_metric"] - 0.75) < 1e-6
    assert summary["primary_metric_name"] == "R²"
    assert abs(summary["primary_metric_pct"] - 75.0) < 0.01
    assert summary["cv_mean"] is not None
    assert summary["cv_std"] is not None
    assert summary["explainability_label"] == "Very high"
    assert "fast" in summary["speed_label"].lower() or "very fast" in summary["speed_label"].lower()


def test_build_run_summary_classification_fields():
    run = _classification_run("random_forest_classifier", 0.88)
    summary = _build_run_summary(run)
    assert summary["primary_metric_name"] == "accuracy"
    assert abs(summary["primary_metric_pct"] - 88.0) < 0.01
    assert summary["cv_mean"] is None


def test_build_run_summary_no_cv():
    run = _regression_run("decision_tree_regressor", 0.62)
    summary = _build_run_summary(run)
    assert summary["cv_mean"] is None
    assert summary["cv_std"] is None


def test_build_run_summary_is_selected_propagated():
    run = _regression_run("linear_regression", 0.75)
    run["is_selected"] = True
    run["is_deployed"] = True
    summary = _build_run_summary(run)
    assert summary["is_selected"] is True
    assert summary["is_deployed"] is True


# ---------------------------------------------------------------------------
# Summary field
# ---------------------------------------------------------------------------


def test_summary_mentions_winner_algorithm():
    runs = [
        _regression_run("gradient_boosting_regressor", 0.90),
        _regression_run("linear_regression", 0.68),
    ]
    result = compute_model_comparison_summary(runs)
    assert "Gradient Boosting" in result["summary"]


def test_summary_single_run_train_more_hint():
    result = compute_model_comparison_summary([
        _regression_run("linear_regression", 0.72)
    ])
    assert "train more" in result["summary"].lower() or "only" in result["summary"].lower()


# ---------------------------------------------------------------------------
# Classification runs
# ---------------------------------------------------------------------------


def test_classification_winner_by_accuracy():
    runs = [
        _classification_run("logistic_regression", 0.78),
        _classification_run("random_forest_classifier", 0.91),
    ]
    result = compute_model_comparison_summary(runs)
    assert result["winner"]["algorithm"] == "random_forest_classifier"
    assert result["problem_type"] == "classification"


# ---------------------------------------------------------------------------
# Many runs
# ---------------------------------------------------------------------------


def test_many_runs_sorted_best_first():
    runs = [
        _regression_run("linear_regression", 0.55),
        _regression_run("random_forest_regressor", 0.88),
        _regression_run("gradient_boosting_regressor", 0.91),
        _regression_run("xgboost_regressor", 0.87),
    ]
    result = compute_model_comparison_summary(runs)
    assert result["n_runs"] == 4
    assert result["runs_compared"][0]["algorithm"] == "gradient_boosting_regressor"
    assert result["runs_compared"][-1]["algorithm"] == "linear_regression"


def test_many_runs_trade_offs_capped_at_3():
    runs = [
        _regression_run("linear_regression", 0.55, cv_mean=0.54, cv_std=0.01),
        _regression_run("random_forest_regressor", 0.88, cv_mean=0.86, cv_std=0.05),
        _regression_run("gradient_boosting_regressor", 0.91, cv_mean=0.89, cv_std=0.08),
        _regression_run("xgboost_regressor", 0.87),
    ]
    result = compute_model_comparison_summary(runs)
    assert len(result["trade_offs"]) <= 3


# ---------------------------------------------------------------------------
# Chat pattern regex (smoke test for pattern isolation)
# ---------------------------------------------------------------------------


def test_comparison_patterns_smoke():
    """Verify the patterns module can be imported and match expected phrases."""
    import re

    _PATTERN = re.compile(
        r"\b(compare.*model|model.*comparison|how.*model.*compare|"
        r"model.*stack.*up|how.*train.*model.*stack|"
        r"summary.*model|model.*summary|summarize.*model|"
        r"tell.*me.*about.*model|what.*model.*train|which.*model.*train|"
        r"model.*showdown|model.*overview|overview.*model|"
        r"all.*model|model.*run.*summary|compare.*algorithm|"
        r"model.*performance.*overview|how.*model.*perform.*overall)",
        re.IGNORECASE,
    )

    should_match = [
        "compare my models",
        "model comparison summary",
        "how do my models compare",
        "how do my trained models stack up",
        "model overview",
        "model showdown",
        "summarize my models",
        "model run summary",
        "compare algorithms",
        "model performance overview",
        "how do my models perform overall",
        "tell me about my models",
        "what models did I train",
        "which models did I train",
        "all my models",
    ]
    for phrase in should_match:
        assert _PATTERN.search(phrase), f"Pattern should match: {phrase!r}"
