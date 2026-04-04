"""advisor.py

Model Improvement Advisor — analyses a trained model's metrics and context,
then generates ranked, plain-English improvement suggestions for business analysts.

Design:
- Pure function with no database dependencies (accepts pre-loaded dicts)
- Each suggestion is an independent check; if a check fails it is skipped
- Returns structured dict suitable for SSE + frontend card rendering
- Difficulty: "easy" = chat command or one click | "medium" = retrain | "hard" = new data/design
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_improvement_suggestions(
    metrics: dict[str, Any],
    algorithm: str,
    problem_type: str,
    n_features: int,
    n_rows: int,
    has_date_col: bool = False,
    date_col_used: bool = False,
    n_weak_features: int = 0,
    is_ensemble: bool = False,
    is_calibrated: bool = False,
    imbalance_strategy: str | None = None,
    class_is_imbalanced: bool = False,
) -> dict[str, Any]:
    """Return ranked improvement suggestions for the given trained model.

    Parameters
    ----------
    metrics:
        The model run metrics dict (r2/accuracy/f1/mae/rmse etc.).
    algorithm:
        Algorithm key, e.g. "random_forest_regressor".
    problem_type:
        "regression" or "classification".
    n_features:
        Number of features used during training.
    n_rows:
        Total number of training rows.
    has_date_col:
        True if the dataset contains at least one datetime column.
    date_col_used:
        True if a date column was already used for time-based splitting.
    n_weak_features:
        Number of features flagged as weak by feature-selection analysis.
    is_ensemble:
        True if the current algorithm is an ensemble (voting/stacking).
    is_calibrated:
        True if CalibratedClassifierCV was applied.
    imbalance_strategy:
        Strategy already applied ("class_weight"/"smote"/"threshold") or None.
    class_is_imbalanced:
        True if class imbalance was detected before training.
    """
    suggestions: list[dict[str, Any]] = []

    primary_metric, primary_metric_name = _primary_metric(metrics, problem_type)

    # Run all checks and collect suggestions --------------------------------
    _check_weak_features(suggestions, n_weak_features)
    _check_ensemble(suggestions, is_ensemble, primary_metric, problem_type)
    _check_date_features(suggestions, has_date_col, date_col_used, algorithm)
    _check_data_volume(suggestions, n_rows)
    _check_imbalance(suggestions, class_is_imbalanced, imbalance_strategy, problem_type)
    _check_calibration(suggestions, is_calibrated, problem_type, metrics)
    _check_tuning(suggestions, algorithm, is_ensemble, primary_metric, problem_type)
    _check_feature_count(suggestions, n_features)
    _check_linear_vs_nonlinear(suggestions, algorithm, primary_metric, problem_type)

    # Sort by impact priority (high > moderate > low), then by ease of action
    _IMPACT_ORDER = {"high": 0, "moderate": 1, "low": 2}
    _DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
    suggestions.sort(
        key=lambda s: (
            _IMPACT_ORDER.get(s["expected_impact"], 9),
            _DIFFICULTY_ORDER.get(s["difficulty"], 9),
        )
    )

    # Assign ranks after sorting
    for i, s in enumerate(suggestions):
        s["rank"] = i + 1

    summary = _build_summary(
        suggestions, primary_metric, primary_metric_name, problem_type
    )

    return {
        "algorithm": algorithm,
        "problem_type": problem_type,
        "primary_metric": primary_metric,
        "primary_metric_name": primary_metric_name,
        "suggestions": suggestions,
        "summary": summary,
        "n_suggestions": len(suggestions),
    }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_weak_features(out: list[dict], n_weak: int) -> None:
    if n_weak <= 0:
        return
    noun = "feature" if n_weak == 1 else "features"
    out.append(
        {
            "category": "features",
            "title": f"Remove {n_weak} Weak {noun.title()}",
            "explanation": (
                f"Feature selection found {n_weak} {noun} with near-zero importance. "
                "Removing them before retraining reduces noise and often improves "
                "prediction accuracy on new data."
            ),
            "action": "feature_selection",
            "difficulty": "easy",
            "expected_impact": "moderate",
        }
    )


def _check_ensemble(
    out: list[dict],
    is_ensemble: bool,
    primary_metric: float,
    problem_type: str,
) -> None:
    if is_ensemble:
        return  # already using ensemble
    threshold = 0.80 if problem_type == "regression" else 0.85
    if primary_metric >= threshold:
        return  # good enough; ensemble not top priority
    metric_name = "R²" if problem_type == "regression" else "accuracy"
    out.append(
        {
            "category": "algorithm",
            "title": "Try an Ensemble Model",
            "explanation": (
                f"Your {metric_name} is {round(primary_metric, 2):.2f}. "
                "Ensemble methods (Voting or Stacking) combine multiple models and "
                "often outperform any single algorithm — sometimes by 5–15%."
            ),
            "action": "train_ensemble",
            "difficulty": "medium",
            "expected_impact": "high",
        }
    )


def _check_date_features(
    out: list[dict],
    has_date_col: bool,
    date_col_used: bool,
    algorithm: str,
) -> None:
    if not has_date_col or date_col_used:
        return
    out.append(
        {
            "category": "features",
            "title": "Unlock Date-Based Features",
            "explanation": (
                "Your dataset has a date column that wasn't used for splitting or "
                "feature engineering. Decomposing it into month, day-of-week, and "
                "quarter can reveal seasonal patterns and significantly boost accuracy."
            ),
            "action": "feature_engineering",
            "difficulty": "easy",
            "expected_impact": "moderate",
        }
    )


def _check_data_volume(out: list[dict], n_rows: int) -> None:
    if n_rows >= 2000:
        return
    out.append(
        {
            "category": "data",
            "title": "Collect More Training Data",
            "explanation": (
                f"Your model trained on only {n_rows:,} rows. Most ML algorithms "
                "become significantly more reliable with 2,000+ rows — each new "
                "example teaches the model a pattern it might not have seen yet."
            ),
            "action": "add_data",
            "difficulty": "hard",
            "expected_impact": "high",
        }
    )


def _check_imbalance(
    out: list[dict],
    class_is_imbalanced: bool,
    imbalance_strategy: str | None,
    problem_type: str,
) -> None:
    if problem_type != "classification":
        return
    if not class_is_imbalanced or imbalance_strategy is not None:
        return
    out.append(
        {
            "category": "data",
            "title": "Handle Class Imbalance",
            "explanation": (
                "Your dataset has an imbalanced class distribution but no correction "
                "strategy was applied. Try class weighting or SMOTE oversampling — "
                "this often dramatically improves recall on the minority class."
            ),
            "action": "class_imbalance",
            "difficulty": "easy",
            "expected_impact": "high",
        }
    )


def _check_calibration(
    out: list[dict],
    is_calibrated: bool,
    problem_type: str,
    metrics: dict,
) -> None:
    if problem_type != "classification":
        return
    if is_calibrated:
        return
    brier = metrics.get("brier_score")
    if brier is not None and brier < 0.1:
        return  # already well-calibrated even without explicit calibration
    out.append(
        {
            "category": "reliability",
            "title": "Calibrate Confidence Scores",
            "explanation": (
                "Calibration ensures the model's confidence percentages are trustworthy "
                "(e.g., when it says 80% confident, it should be right ~80% of the time). "
                "Retrain with calibration enabled for more honest probability estimates."
            ),
            "action": "calibration",
            "difficulty": "medium",
            "expected_impact": "moderate",
        }
    )


def _check_tuning(
    out: list[dict],
    algorithm: str,
    is_ensemble: bool,
    primary_metric: float,
    problem_type: str,
) -> None:
    # Skip: linear models have few meaningful hyperparameters
    linear_algos = {"linear_regression", "logistic_regression", "ridge_regression"}
    if algorithm in linear_algos or is_ensemble:
        return
    threshold = 0.85 if problem_type == "regression" else 0.90
    if primary_metric >= threshold:
        return  # already high-performing; tuning offers diminishing returns
    out.append(
        {
            "category": "algorithm",
            "title": "Tune Hyperparameters",
            "explanation": (
                "AutoModeler trained this model with sensible defaults. "
                "Automated hyperparameter search (RandomizedSearchCV) can often "
                "squeeze out another 2–5% improvement by finding better settings."
            ),
            "action": "hyperparameter_tuning",
            "difficulty": "easy",
            "expected_impact": "moderate",
        }
    )


def _check_feature_count(out: list[dict], n_features: int) -> None:
    if n_features >= 3:
        return
    out.append(
        {
            "category": "features",
            "title": "Add More Predictive Features",
            "explanation": (
                f"Your model uses only {n_features} feature{'s' if n_features != 1 else ''}. "
                "More relevant columns generally help — consider adding domain-specific "
                "variables, ratio features (e.g., revenue ÷ units), or external data."
            ),
            "action": "add_features",
            "difficulty": "hard",
            "expected_impact": "high",
        }
    )


def _check_linear_vs_nonlinear(
    out: list[dict],
    algorithm: str,
    primary_metric: float,
    problem_type: str,
) -> None:
    linear_algos = {"linear_regression", "logistic_regression", "ridge_regression"}
    if algorithm not in linear_algos:
        return
    threshold = 0.65 if problem_type == "regression" else 0.75
    if primary_metric >= threshold:
        return
    out.append(
        {
            "category": "algorithm",
            "title": "Try a Nonlinear Model",
            "explanation": (
                "Linear models assume a straight-line relationship between features "
                "and the target. Your data may have more complex patterns — "
                "Random Forest or Gradient Boosting can capture these automatically."
            ),
            "action": "train_nonlinear",
            "difficulty": "medium",
            "expected_impact": "high",
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _primary_metric(metrics: dict[str, Any], problem_type: str) -> tuple[float, str]:
    """Return (value, name) for the most important metric."""
    if problem_type == "regression":
        return float(metrics.get("r2", 0.0)), "R²"
    # classification: prefer accuracy
    return float(metrics.get("accuracy", metrics.get("f1", 0.0))), "accuracy"


def _build_summary(
    suggestions: list[dict],
    primary_metric: float,
    metric_name: str,
    problem_type: str,
) -> str:
    """Return a 1–2 sentence plain-English summary."""
    if problem_type == "regression":
        pct = round(primary_metric * 100)
        quality = (
            "excellent"
            if pct >= 85
            else "good"
            if pct >= 70
            else "moderate"
            if pct >= 55
            else "limited"
        )
        base = f"Your model explains {pct}% of variation in the target — {quality} predictive power."
    else:
        pct = round(primary_metric * 100)
        quality = (
            "excellent"
            if pct >= 90
            else "good"
            if pct >= 80
            else "moderate"
            if pct >= 70
            else "limited"
        )
        base = f"Your model achieves {pct}% accuracy — {quality} performance."

    if not suggestions:
        return (
            base
            + " No obvious improvements detected — your model looks well-optimised."
        )

    top = suggestions[0]
    return (
        base
        + f" Top suggestion: {top['title'].lower()} — "
        + top["explanation"].split(".")[0].lower()
        + "."
    )
