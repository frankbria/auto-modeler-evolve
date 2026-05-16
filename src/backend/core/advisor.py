"""advisor.py

Model Advisor — three capabilities:
1. Improvement Advisor: analyses a trained model and returns ranked plain-English
   suggestions for making it better.
2. Model Selection Advisor: given multiple completed runs, scores each one against
   an analyst-chosen criteria (accuracy, explainability, stability, speed, balanced)
   and returns a ranked recommendation.
3. Model Comparison Summary: given multiple completed runs, generates a narrative
   comparison — which model won, key trade-offs, and a plain-English summary.

Design:
- Pure functions with no database dependencies (accepts pre-loaded dicts)
- Each check / scoring pass is independent
- Returns structured dicts suitable for SSE + frontend card rendering
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


# ===========================================================================
# Model Selection Advisor
# ===========================================================================

# Explainability rank: lower = more explainable (1 = most transparent)
_EXPLAINABILITY_RANK: dict[str, int] = {
    "linear_regression": 1,
    "ridge_regression": 2,
    "logistic_regression": 3,
    "decision_tree_classifier": 4,
    "decision_tree_regressor": 4,
    "knn_classifier": 5,
    "knn_regressor": 5,
    "random_forest_classifier": 6,
    "random_forest_regressor": 6,
    "gradient_boosting_classifier": 7,
    "gradient_boosting_regressor": 7,
    "xgboost_classifier": 8,
    "xgboost_regressor": 8,
    "lgbm_classifier": 8,
    "lgbm_regressor": 8,
    "mlp_classifier": 9,
    "mlp_regressor": 9,
    "voting_classifier": 10,
    "voting_regressor": 10,
    "stacking_classifier": 10,
    "stacking_regressor": 10,
}

# Speed rank: lower = faster (1 = fastest)
_SPEED_RANK: dict[str, int] = {
    "linear_regression": 1,
    "ridge_regression": 1,
    "logistic_regression": 2,
    "decision_tree_classifier": 3,
    "decision_tree_regressor": 3,
    "knn_classifier": 4,
    "knn_regressor": 4,
    "random_forest_classifier": 5,
    "random_forest_regressor": 5,
    "gradient_boosting_classifier": 6,
    "gradient_boosting_regressor": 6,
    "xgboost_classifier": 7,
    "xgboost_regressor": 7,
    "lgbm_classifier": 6,
    "lgbm_regressor": 6,
    "mlp_classifier": 8,
    "mlp_regressor": 8,
    "voting_classifier": 9,
    "voting_regressor": 9,
    "stacking_classifier": 10,
    "stacking_regressor": 10,
}

# Plain-English algorithm names
_ALGO_PLAIN: dict[str, str] = {
    "linear_regression": "Linear Regression",
    "ridge_regression": "Ridge Regression",
    "logistic_regression": "Logistic Regression",
    "decision_tree_classifier": "Decision Tree",
    "decision_tree_regressor": "Decision Tree",
    "knn_classifier": "K-Nearest Neighbours",
    "knn_regressor": "K-Nearest Neighbours",
    "random_forest_classifier": "Random Forest",
    "random_forest_regressor": "Random Forest",
    "gradient_boosting_classifier": "Gradient Boosting",
    "gradient_boosting_regressor": "Gradient Boosting",
    "xgboost_classifier": "XGBoost",
    "xgboost_regressor": "XGBoost",
    "lgbm_classifier": "LightGBM",
    "lgbm_regressor": "LightGBM",
    "mlp_classifier": "Neural Network (MLP)",
    "mlp_regressor": "Neural Network (MLP)",
    "voting_classifier": "Voting Ensemble",
    "voting_regressor": "Voting Ensemble",
    "stacking_classifier": "Stacking Ensemble",
    "stacking_regressor": "Stacking Ensemble",
}

# Explainability descriptions for the winner card
_EXPLAIN_WHY: dict[str, str] = {
    "linear_regression": (
        "Linear Regression is fully transparent — each prediction is a simple "
        "weighted sum of your inputs. You can explain it in one sentence to any stakeholder."
    ),
    "ridge_regression": (
        "Ridge Regression is highly transparent — like Linear Regression but robust to "
        "correlated features. Coefficients directly show each input's influence."
    ),
    "logistic_regression": (
        "Logistic Regression is easy to explain — it outputs a probability and its "
        "coefficients show exactly which features push predictions up or down."
    ),
    "decision_tree_classifier": (
        "Decision Trees can be drawn as a flowchart: 'If region = East AND revenue > 100, "
        "predict churn.' Even non-technical stakeholders can follow the logic."
    ),
    "decision_tree_regressor": (
        "Decision Trees can be drawn as a flowchart. Even non-technical stakeholders "
        "can follow the logic — great for internal audits and regulatory reviews."
    ),
    "knn_classifier": (
        "K-Nearest Neighbours makes predictions by finding similar historical cases. "
        "You can always show a stakeholder 'these are the most similar past examples.'"
    ),
    "knn_regressor": (
        "K-Nearest Neighbours predicts by averaging similar historical records — "
        "intuitive to explain as 'average of the N most similar cases.'"
    ),
    "random_forest_classifier": (
        "Random Forest combines many Decision Trees. You can say: 'we asked 100 experts "
        "and the majority voted for this prediction.' Less transparent than a single tree."
    ),
    "random_forest_regressor": (
        "Random Forest averages many Decision Trees. Relatively interpretable — "
        "you can show feature importance charts to explain what drives predictions."
    ),
    "gradient_boosting_classifier": (
        "Gradient Boosting builds trees in sequence, each correcting the last. "
        "Feature importance charts show what matters, but the internal logic is complex."
    ),
    "gradient_boosting_regressor": (
        "Gradient Boosting builds trees in sequence. Feature importance charts help "
        "explain what drives predictions, but the full logic is hard to expose simply."
    ),
    "xgboost_classifier": (
        "XGBoost is a highly optimised boosting method. Very accurate but its internal "
        "logic is complex — use SHAP explanations to justify individual predictions."
    ),
    "xgboost_regressor": (
        "XGBoost is a highly optimised boosting method. Very accurate but its internal "
        "logic is complex — use SHAP explanations to justify individual predictions."
    ),
    "lgbm_classifier": (
        "LightGBM is fast and accurate but complex internally. Feature importance and "
        "SHAP values can explain what the model learned, though not in simple terms."
    ),
    "lgbm_regressor": (
        "LightGBM is fast and accurate but complex internally. Feature importance and "
        "SHAP values can help explain its behaviour to technical audiences."
    ),
    "mlp_classifier": (
        "Neural Networks (MLP) are the least transparent option — they learn abstract "
        "representations with no simple rules to show. Use explanation tools for audits."
    ),
    "mlp_regressor": (
        "Neural Networks (MLP) are the least transparent option — they learn abstract "
        "representations with no simple rules to show. Use explanation tools for audits."
    ),
    "voting_classifier": (
        "A Voting Ensemble combines multiple models. You can say 'N out of M models "
        "agreed on this prediction', but each base model has its own complexity."
    ),
    "voting_regressor": (
        "A Voting Ensemble averages multiple models. Relatively intuitive at a high "
        "level ('all models agreed'), but the individual components are complex."
    ),
    "stacking_classifier": (
        "Stacking layers models on top of each other — the most complex architecture "
        "here. Maximises accuracy but requires SHAP or LIME for any explanation."
    ),
    "stacking_regressor": (
        "Stacking layers models on top of each other — the most complex architecture "
        "here. Maximises accuracy but requires SHAP or LIME for any explanation."
    ),
}

_CRITERIA_DESCRIPTIONS: dict[str, str] = {
    "accuracy": "Highest accuracy — the most predictively powerful model",
    "explainability": "Most explainable — easiest to explain to stakeholders",
    "stability": "Most stable — consistent across data splits, not just lucky",
    "speed": "Fastest — lowest latency for real-time or high-volume predictions",
    "balanced": "Best overall — balances accuracy, explainability, and stability",
}

_EXPLAINABILITY_LABEL: dict[int, str] = {
    1: "Very high",
    2: "Very high",
    3: "High",
    4: "High",
    5: "Medium",
    6: "Medium",
    7: "Low",
    8: "Low",
    9: "Very low",
    10: "Very low",
}

_SPEED_LABEL: dict[int, str] = {
    1: "Very fast",
    2: "Fast",
    3: "Fast",
    4: "Medium",
    5: "Medium",
    6: "Medium",
    7: "Slower",
    8: "Slower",
    9: "Slow",
    10: "Slow",
}

_MAX_EXPLAINABILITY_RANK = max(_EXPLAINABILITY_RANK.values())
_MAX_SPEED_RANK = max(_SPEED_RANK.values())


def compute_model_selection(
    runs: list[dict[str, Any]],
    criteria: str = "balanced",
) -> dict[str, Any]:
    """Score and rank completed model runs by the given analyst criteria.

    Parameters
    ----------
    runs:
        List of run dicts, each with keys:
        run_id, algorithm, metrics (dict), problem_type.
    criteria:
        One of "accuracy" | "explainability" | "stability" | "speed" | "balanced".

    Returns
    -------
    dict with winner, ranked_runs, criteria, summary, etc.
    """
    if not runs:
        return {
            "criteria": criteria,
            "criteria_description": _CRITERIA_DESCRIPTIONS.get(criteria, criteria),
            "winner": None,
            "ranked_runs": [],
            "summary": "No completed model runs to compare.",
            "n_runs": 0,
        }

    criteria = criteria if criteria in _CRITERIA_DESCRIPTIONS else "balanced"
    scored = [_score_run(r, criteria) for r in runs]
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Assign ranks (1-based)
    for i, s in enumerate(scored):
        s["rank"] = i + 1

    winner = scored[0]

    return {
        "criteria": criteria,
        "criteria_description": _CRITERIA_DESCRIPTIONS[criteria],
        "winner": winner,
        "ranked_runs": scored,
        "summary": _selection_summary(winner, criteria),
        "n_runs": len(scored),
    }


def _score_run(run: dict[str, Any], criteria: str) -> dict[str, Any]:
    """Return a scored + annotated version of a single run."""
    algo = run.get("algorithm", "")
    metrics = run.get("metrics") or {}
    problem_type = run.get("problem_type", "")

    primary_metric, metric_name = _primary_metric(metrics, problem_type)

    # Component scores (0.0–1.0, higher = better)
    acc_score = _accuracy_score(primary_metric)
    expl_score = _explainability_score(algo)
    stab_score = _stability_score(metrics)
    speed_score = _speed_score(algo)

    if criteria == "accuracy":
        score = acc_score
    elif criteria == "explainability":
        score = expl_score
    elif criteria == "stability":
        score = stab_score
    elif criteria == "speed":
        score = speed_score
    else:  # balanced
        score = 0.40 * acc_score + 0.30 * expl_score + 0.30 * stab_score

    algo_plain = _ALGO_PLAIN.get(algo, algo.replace("_", " ").title())
    explain_why = _EXPLAIN_WHY.get(
        algo, f"{algo_plain} is a capable model for this task."
    )

    return {
        "run_id": run.get("run_id", run.get("id", "")),
        "algorithm": algo,
        "algorithm_plain": algo_plain,
        "score": round(score, 4),
        "primary_metric": round(primary_metric, 4),
        "primary_metric_name": metric_name,
        "component_scores": {
            "accuracy": round(acc_score, 3),
            "explainability": round(expl_score, 3),
            "stability": round(stab_score, 3),
            "speed": round(speed_score, 3),
        },
        "why": explain_why,
        "is_selected": run.get("is_selected", False),
        "is_deployed": run.get("is_deployed", False),
    }


def _accuracy_score(primary_metric: float) -> float:
    """Normalise primary metric to 0-1 (already 0-1 for R² and accuracy)."""
    return max(0.0, min(1.0, float(primary_metric)))


def _explainability_score(algo: str) -> float:
    """Convert explainability rank to a 0-1 score (1 = most explainable)."""
    rank = _EXPLAINABILITY_RANK.get(algo, _MAX_EXPLAINABILITY_RANK)
    # Invert: rank 1 → score 1.0; rank _MAX → score close to 0
    return 1.0 - (rank - 1) / _MAX_EXPLAINABILITY_RANK


def _stability_score(metrics: dict) -> float:
    """Score based on cross-validation coefficient of variation (lower cv = more stable)."""
    cv_mean = metrics.get("cv_mean")
    cv_std = metrics.get("cv_std")
    if cv_mean is None or cv_std is None:
        return 0.5  # no CV data — neutral
    if float(cv_mean) <= 0:
        return 0.5
    # Coefficient of variation: lower → more stable → higher score
    cov = float(cv_std) / float(cv_mean)
    # Map: cov=0 → 1.0, cov=0.5+ → 0.0
    return max(0.0, 1.0 - cov * 2.0)


def _speed_score(algo: str) -> float:
    """Convert speed rank to a 0-1 score (1 = fastest)."""
    rank = _SPEED_RANK.get(algo, _MAX_SPEED_RANK)
    return 1.0 - (rank - 1) / _MAX_SPEED_RANK


def _selection_summary(winner: dict[str, Any], criteria: str) -> str:
    """Return a one-sentence plain-English summary of the recommendation."""
    name = winner["algorithm_plain"]
    metric_val = round(winner["primary_metric"] * 100)
    metric_name = winner["primary_metric_name"]
    score_pct = round(winner["score"] * 100)

    if criteria == "accuracy":
        return (
            f"{name} achieves the highest {metric_name} of {metric_val}% "
            f"— it is the most predictively powerful model in this comparison."
        )
    if criteria == "explainability":
        return (
            f"{name} is the most transparent algorithm here — "
            f"ideal for situations where you need to explain predictions to stakeholders."
        )
    if criteria == "stability":
        return (
            f"{name} is the most consistent performer across data splits, "
            f"with the lowest variance in cross-validation results."
        )
    if criteria == "speed":
        return (
            f"{name} is the fastest option — ideal for real-time APIs "
            f"or high-volume batch prediction scenarios."
        )
    # balanced
    return (
        f"{name} scores best overall ({score_pct}/100) — "
        f"a strong balance of accuracy ({metric_val}% {metric_name}), "
        f"explainability, and stability."
    )


# ---------------------------------------------------------------------------
# Model Comparison Summary
# ---------------------------------------------------------------------------


def compute_model_comparison_summary(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate a narrative comparison of all completed model runs.

    Unlike compute_model_selection (criteria-based ranking), this produces a
    prose summary that highlights key differences, trade-offs, and a winner
    recommendation without requiring the analyst to specify criteria.

    Parameters
    ----------
    runs:
        List of run dicts with keys: run_id, algorithm, metrics, problem_type,
        is_selected, is_deployed.

    Returns
    -------
    dict with n_runs, winner, runs_compared, trade_offs, narrative, summary,
    problem_type, only_one_run (bool).
    """
    if not runs:
        return {
            "n_runs": 0,
            "winner": None,
            "runs_compared": [],
            "trade_offs": [],
            "narrative": "No completed model runs to compare yet.",
            "summary": "Train some models first, then ask me to compare them.",
            "problem_type": "regression",
            "only_one_run": False,
        }

    problem_type = runs[0].get("problem_type", "regression")

    # Build per-run summary dicts sorted best → worst by primary metric
    runs_compared = [_build_run_summary(r) for r in runs]
    runs_compared.sort(key=lambda x: x["primary_metric"], reverse=True)

    winner = runs_compared[0]
    only_one_run = len(runs_compared) == 1

    trade_offs = [] if only_one_run else _build_trade_offs(runs_compared, problem_type)
    narrative = _build_comparison_narrative(
        winner, runs_compared, problem_type, only_one_run
    )
    summary = _build_comparison_summary(winner, runs_compared, problem_type)

    return {
        "n_runs": len(runs_compared),
        "winner": winner,
        "runs_compared": runs_compared,
        "trade_offs": trade_offs,
        "narrative": narrative,
        "summary": summary,
        "problem_type": problem_type,
        "only_one_run": only_one_run,
    }


def _build_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    """Return a display-ready summary dict for one run."""
    algo = run.get("algorithm", "")
    metrics = run.get("metrics") or {}
    problem_type = run.get("problem_type", "regression")

    primary_metric, metric_name = _primary_metric(metrics, problem_type)

    expl_rank = _EXPLAINABILITY_RANK.get(algo, _MAX_EXPLAINABILITY_RANK)
    speed_rank = _SPEED_RANK.get(algo, _MAX_SPEED_RANK)
    algo_plain = _ALGO_PLAIN.get(algo, algo.replace("_", " ").title())

    cv_mean = metrics.get("cv_mean")
    cv_std = metrics.get("cv_std")

    # Secondary metrics for table
    secondary: dict[str, Any] = {}
    if problem_type == "regression":
        if "mae" in metrics:
            secondary["mae"] = round(float(metrics["mae"]), 4)
        if "r2" in metrics:
            secondary["r2"] = round(float(metrics["r2"]), 4)
    else:
        if "f1" in metrics:
            secondary["f1"] = round(float(metrics["f1"]), 4)
        if "precision" in metrics:
            secondary["precision"] = round(float(metrics["precision"]), 4)
        if "recall" in metrics:
            secondary["recall"] = round(float(metrics["recall"]), 4)

    return {
        "run_id": run.get("run_id", run.get("id", "")),
        "algorithm": algo,
        "algorithm_plain": algo_plain,
        "primary_metric": round(primary_metric, 4),
        "primary_metric_name": metric_name,
        "primary_metric_pct": round(primary_metric * 100, 1),
        "cv_mean": round(float(cv_mean), 4) if cv_mean is not None else None,
        "cv_std": round(float(cv_std), 4) if cv_std is not None else None,
        "explainability_rank": expl_rank,
        "explainability_label": _EXPLAINABILITY_LABEL.get(expl_rank, "Medium"),
        "speed_rank": speed_rank,
        "speed_label": _SPEED_LABEL.get(speed_rank, "Medium"),
        "is_selected": run.get("is_selected", False),
        "is_deployed": run.get("is_deployed", False),
        "secondary": secondary,
    }


def _build_trade_offs(
    runs_compared: list[dict[str, Any]], problem_type: str
) -> list[str]:
    """Return up to 3 plain-English trade-off sentences comparing the top runs."""
    if len(runs_compared) < 2:
        return []

    best = runs_compared[0]
    second = runs_compared[1]
    trade_offs: list[str] = []

    # Accuracy gap between first and second
    gap = best["primary_metric_pct"] - second["primary_metric_pct"]
    metric_name = best["primary_metric_name"]
    if gap >= 0.5:
        trade_offs.append(
            f"{best['algorithm_plain']} leads on {metric_name} "
            f"({best['primary_metric_pct']:.1f}% vs "
            f"{second['primary_metric_pct']:.1f}% — a {gap:.1f}% gap)."
        )
    else:
        trade_offs.append(
            f"{best['algorithm_plain']} and {second['algorithm_plain']} "
            f"are neck-and-neck on {metric_name} "
            f"({best['primary_metric_pct']:.1f}% vs "
            f"{second['primary_metric_pct']:.1f}%)."
        )

    # Explainability trade-off: best accuracy vs most explainable
    most_explainable = min(runs_compared, key=lambda r: r["explainability_rank"])
    if most_explainable["algorithm"] != best["algorithm"]:
        expl_gap = best["primary_metric_pct"] - most_explainable["primary_metric_pct"]
        direction = "more accurate" if expl_gap > 0 else "similar in accuracy"
        trade_offs.append(
            f"{most_explainable['algorithm_plain']} is the most interpretable option "
            f"(easier to explain to stakeholders), "
            f"while {best['algorithm_plain']} is {direction} "
            f"({best['primary_metric_pct']:.1f}% vs "
            f"{most_explainable['primary_metric_pct']:.1f}%)."
        )

    # Stability trade-off: CV score variance (if available)
    runs_with_cv = [r for r in runs_compared if r["cv_mean"] is not None]
    if len(runs_with_cv) >= 2:
        most_stable = min(
            runs_with_cv, key=lambda r: (r["cv_std"] or 1.0)
        )
        least_stable = max(
            runs_with_cv, key=lambda r: (r["cv_std"] or 0.0)
        )
        if (
            most_stable["algorithm"] != least_stable["algorithm"]
            and (least_stable["cv_std"] or 0) - (most_stable["cv_std"] or 0) > 0.01
        ):
            trade_offs.append(
                f"{most_stable['algorithm_plain']} is the most stable "
                f"(CV std ±{most_stable['cv_std']:.2f}), "  # type: ignore[str-format]
                f"while {least_stable['algorithm_plain']} shows more variability "
                f"(±{least_stable['cv_std']:.2f})."  # type: ignore[str-format]
            )

    return trade_offs[:3]


def _build_comparison_narrative(
    winner: dict[str, Any],
    runs_compared: list[dict[str, Any]],
    problem_type: str,
    only_one_run: bool,
) -> str:
    """Return a 2–3 sentence plain-English narrative."""
    n = len(runs_compared)
    metric_name = winner["primary_metric_name"]
    metric_pct = winner["primary_metric_pct"]

    if only_one_run:
        expl = winner["explainability_label"].lower()
        return (
            f"You have trained one model: {winner['algorithm_plain']}, "
            f"which achieves {metric_pct:.1f}% {metric_name}. "
            f"Explainability is {expl} — "
            f"{'easy to explain to stakeholders' if expl in ('very high', 'high') else 'may need SHAP charts for stakeholder explanations'}. "
            f"Train more algorithms and ask me to compare them for a full picture."
        )

    runner_up = runs_compared[1] if n >= 2 else None
    gap = metric_pct - (runner_up["primary_metric_pct"] if runner_up else 0)

    intro = (
        f"You trained {n} models. "
        f"{winner['algorithm_plain']} comes out ahead "
        f"with {metric_pct:.1f}% {metric_name}"
    )
    if runner_up:
        intro += (
            f", edging out {runner_up['algorithm_plain']} "
            f"by {gap:.1f} percentage points"
        )
    intro += "."

    # Explainability note
    most_explainable = min(runs_compared, key=lambda r: r["explainability_rank"])
    if most_explainable["algorithm"] != winner["algorithm"]:
        expl_note = (
            f" If you need to explain predictions to your VP, "
            f"{most_explainable['algorithm_plain']} is the most interpretable option "
            f"({most_explainable['primary_metric_pct']:.1f}% {metric_name})."
        )
    else:
        expl_note = (
            f" {winner['algorithm_plain']} is also the most interpretable here, "
            f"so it's a strong choice for VP presentations."
        )

    return intro + expl_note


def _build_comparison_summary(
    winner: dict[str, Any],
    runs_compared: list[dict[str, Any]],
    problem_type: str,
) -> str:
    """Return a one-sentence takeaway."""
    n = len(runs_compared)
    metric_name = winner["primary_metric_name"]
    metric_pct = winner["primary_metric_pct"]
    if n == 1:
        return (
            f"{winner['algorithm_plain']} is your only trained model "
            f"({metric_pct:.1f}% {metric_name}) — train more to compare."
        )
    return (
        f"{winner['algorithm_plain']} wins across {n} models "
        f"({metric_pct:.1f}% {metric_name})."
    )
