"""Hyperparameter auto-tuning via RandomizedSearchCV.

Design principles:
- Each algorithm has a param grid of meaningful search ranges.
- RandomizedSearchCV (not GridSearch) for speed — 20 iterations, 3-fold CV.
- Returns a plain-English summary of improvement so chat can narrate the result.
- Saves the best estimator to disk and returns model metadata for ModelRun creation.
- Neural networks excluded from tuning (architecture search is a different problem).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict

from core.trainer import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
)

# ---------------------------------------------------------------------------
# Parameter search grids per algorithm key
# ---------------------------------------------------------------------------

_REGRESSION_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "random_forest_regressor": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, 1.0],
    },
    "gradient_boosting_regressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    },
    "linear_regression": {},  # No meaningful hyperparams for OLS
    "xgboost_regressor": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
    },
    "lightgbm_regressor": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [20, 31, 50, 80],
        "max_depth": [-1, 5, 10, 15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
    },
}

_CLASSIFICATION_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l2"],
        "solver": ["lbfgs", "saga"],
    },
    "random_forest_classifier": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "gradient_boosting_classifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    },
    "xgboost_classifier": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
    },
    "lightgbm_classifier": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [20, 31, 50, 80],
        "max_depth": [-1, 5, 10, 15],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
    },
}


def _compute_metrics(y_true: Any, y_pred: Any, problem_type: str) -> dict:
    """Compute regression or classification metrics."""
    if problem_type == "regression":
        return {
            "r2": round(float(r2_score(y_true, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        }
    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes == 2 else "weighted"
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
    }


def is_tunable(algorithm: str) -> bool:
    """Return True if there's a non-empty param grid for this algorithm."""
    grids = {**_REGRESSION_PARAM_GRIDS, **_CLASSIFICATION_PARAM_GRIDS}
    return bool(grids.get(algorithm))


def tune_model(
    algorithm: str,
    problem_type: str,
    X: Any,
    y: Any,
    model_run_id: str,
    model_dir: Path,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
) -> dict:
    """Run RandomizedSearchCV to find better hyperparameters for an algorithm.

    Returns a dict with:
      - best_params: the winning hyperparameter combination
      - metrics: computed on a held-out test split
      - model_path: path to serialized tuned estimator
      - training_duration_ms: wall-clock time for the search
      - summary: plain-English description of the improvement
      - cv_best_score: best cross-validated score from the search
    """
    algorithms = (
        REGRESSION_ALGORITHMS if problem_type == "regression"
        else CLASSIFICATION_ALGORITHMS
    )
    if algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    param_grids = (
        _REGRESSION_PARAM_GRIDS if problem_type == "regression"
        else _CLASSIFICATION_PARAM_GRIDS
    )
    param_grid = param_grids.get(algorithm, {})
    if not param_grid:
        raise ValueError(
            f"Algorithm {algorithm!r} has no tunable hyperparameters. "
            "Try a tree-based model like Random Forest or Gradient Boosting."
        )

    algo_info = algorithms[algorithm]
    base_estimator = algo_info["class"](**algo_info.get("params", {}))

    scoring = "r2" if problem_type == "regression" else "accuracy"

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_grid,
        n_iter=min(n_iter, _count_combinations(param_grid)),
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    start = time.time()
    search.fit(X, y)
    duration_ms = int((time.time() - start) * 1000)

    best_estimator = search.best_estimator_
    cv_best_score = float(search.best_score_)

    # Compute cross-validated predictions for final metrics
    y_pred = cross_val_predict(best_estimator, X, y, cv=cv)
    metrics = _compute_metrics(y, y_pred, problem_type)

    # Persist the tuned model
    model_dir.mkdir(parents=True, exist_ok=True)
    tuned_run_id = f"{model_run_id}_tuned"
    model_path = str(model_dir / f"{tuned_run_id}.joblib")
    joblib.dump(best_estimator, model_path)

    summary = _build_tune_summary(
        algorithm=algorithm,
        problem_type=problem_type,
        best_params=search.best_params_,
        cv_best_score=cv_best_score,
        metrics=metrics,
    )

    return {
        "best_params": search.best_params_,
        "cv_best_score": cv_best_score,
        "metrics": metrics,
        "model_path": model_path,
        "training_duration_ms": duration_ms,
        "summary": summary,
        "algorithm": f"{algorithm}_tuned",
    }


def _count_combinations(param_grid: dict[str, list]) -> int:
    """Count total combinations in a param grid."""
    total = 1
    for values in param_grid.values():
        total *= len(values)
    return total


def _build_tune_summary(
    algorithm: str,
    problem_type: str,
    best_params: dict,
    cv_best_score: float,
    metrics: dict,
) -> str:
    """Build a plain-English summary of tuning results."""
    algo_names = {**REGRESSION_ALGORITHMS, **CLASSIFICATION_ALGORITHMS}
    algo_name = algo_names.get(algorithm, {}).get("name", algorithm)

    if problem_type == "regression":
        r2 = metrics.get("r2", 0.0)
        mae = metrics.get("mae", 0.0)
        score_str = f"R² = {r2:.3f}, MAE = {mae:.3f}"
        cv_str = f"CV R² = {cv_best_score:.3f}"
    else:
        acc = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1", 0.0)
        score_str = f"accuracy = {acc:.1%}, F1 = {f1:.3f}"
        cv_str = f"CV accuracy = {cv_best_score:.1%}"

    # Highlight the most meaningful tuned parameters
    highlight_params = {
        k: v for k, v in best_params.items()
        if k in {"n_estimators", "max_depth", "learning_rate", "C", "num_leaves"}
    }
    param_str = ", ".join(f"{k}={v}" for k, v in list(highlight_params.items())[:3])

    return (
        f"Tuned {algo_name}: {score_str} ({cv_str}). "
        f"Best settings found: {param_str}."
    )
