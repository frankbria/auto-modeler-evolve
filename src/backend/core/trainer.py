"""Model training, recommendation, and comparison.

Design:
- Each model is wrapped in a sklearn Pipeline with ColumnTransformer preprocessing.
  This prevents train/test leakage (transformers fitted on training folds only) and
  produces a single serializable artifact for Phase 6 deployment.
- recommend_models uses dataset characteristics (size, feature count, problem type)
  to suggest 2-4 algorithms with plain-English explanations.
- compare_models ranks completed runs and generates a human-readable summary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelRecommendation:
    algorithm: str
    display_name: str
    description: str
    reason: str
    hyperparameters: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

_CLASSIFIERS: dict[str, dict] = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "description": (
            "A mathematical formula that finds the best decision boundary — "
            "fast, transparent, and great for explaining which features push "
            "predictions up or down."
        ),
        "estimator_cls": LogisticRegression,
        "default_params": {"max_iter": 1000, "random_state": 42},
        "needs_scaling": True,
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "description": (
            "A flowchart of yes/no questions — the most explainable model. "
            "You can trace exactly why it made any prediction."
        ),
        "estimator_cls": DecisionTreeClassifier,
        "default_params": {"max_depth": 6, "random_state": 42},
        "needs_scaling": False,
    },
    "random_forest": {
        "display_name": "Random Forest",
        "description": (
            "Like asking 100 experts and taking a vote. Robust, rarely "
            "overfits, and handles messy data well."
        ),
        "estimator_cls": RandomForestClassifier,
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "needs_scaling": False,
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting",
        "description": (
            "A team of learners, each fixing the mistakes of the previous. "
            "Usually the most accurate, but slower to train."
        ),
        "estimator_cls": GradientBoostingClassifier,
        "default_params": {"n_estimators": 100, "random_state": 42},
        "needs_scaling": False,
    },
}

_REGRESSORS: dict[str, dict] = {
    "linear_regression": {
        "display_name": "Linear Regression",
        "description": (
            "Fits a straight line through your data. Fast, interpretable, "
            "and ideal when relationships between features and target are "
            "roughly linear."
        ),
        "estimator_cls": LinearRegression,
        "default_params": {},
        "needs_scaling": True,
    },
    "ridge": {
        "display_name": "Ridge Regression",
        "description": (
            "Linear regression with regularization — prevents overfitting "
            "when features are correlated or numerous."
        ),
        "estimator_cls": Ridge,
        "default_params": {"alpha": 1.0},
        "needs_scaling": True,
    },
    "random_forest": {
        "display_name": "Random Forest",
        "description": (
            "Like asking 100 experts and taking a vote. Captures non-linear "
            "patterns automatically without feature scaling."
        ),
        "estimator_cls": RandomForestRegressor,
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "needs_scaling": False,
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting",
        "description": (
            "A team of learners, each fixing the previous one's mistakes. "
            "Usually the most accurate regressor for structured data."
        ),
        "estimator_cls": GradientBoostingRegressor,
        "default_params": {"n_estimators": 100, "random_state": 42},
        "needs_scaling": False,
    },
}


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def recommend_models(
    problem_type: str,
    n_rows: int,
    n_features: int,
) -> list[ModelRecommendation]:
    """Suggest 2-4 algorithms appropriate for this dataset.

    Selection logic based on dataset size:
    - Very small (<200 rows): simpler models only
    - Medium (200-2000): all models including gradient boosting
    - Large (>2000): prefer ensemble methods
    """
    catalog = _CLASSIFIERS if problem_type == "classification" else _REGRESSORS

    if problem_type == "classification":
        if n_rows < 200:
            selected = ["logistic_regression", "decision_tree", "random_forest"]
        elif n_rows < 2000:
            selected = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]
        else:
            selected = ["random_forest", "gradient_boosting", "logistic_regression"]
    else:
        if n_rows < 200:
            selected = ["linear_regression", "ridge", "random_forest"]
        elif n_rows < 2000:
            selected = ["linear_regression", "ridge", "random_forest", "gradient_boosting"]
        else:
            selected = ["random_forest", "gradient_boosting", "ridge"]

    return [
        ModelRecommendation(
            algorithm=algo,
            display_name=catalog[algo]["display_name"],
            description=catalog[algo]["description"],
            reason=_selection_reason(algo, n_rows, n_features),
            hyperparameters=catalog[algo]["default_params"],
        )
        for algo in selected
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    algorithm: str,
    hyperparameters: dict,
    model_dir: Path,
) -> dict[str, Any]:
    """Train one algorithm and return metrics + model path.

    Returns:
        dict with keys: algorithm, display_name, metrics,
        training_duration_ms, model_path
    """
    catalog = _CLASSIFIERS if problem_type == "classification" else _REGRESSORS
    if algorithm not in catalog:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    info = catalog[algorithm]
    X, y, feature_names = _prepare_xy(df, target_col, problem_type)

    n_rows = len(X)
    cv = min(5, max(2, n_rows // 2))

    merged_params = {**info["default_params"], **hyperparameters}
    pipeline = _build_pipeline(X, info["estimator_cls"], merged_params, info["needs_scaling"])

    t0 = time.time()
    metrics = _compute_cv_metrics(pipeline, X, y, problem_type, cv)
    elapsed_ms = int((time.time() - t0) * 1000)

    # Final fit on all data for the serialized artifact
    pipeline.fit(X, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_id = str(uuid4())
    model_path = str(model_dir / f"{model_id}.joblib")
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": feature_names,
            "target_col": target_col,
            "problem_type": problem_type,
            "algorithm": algorithm,
        },
        model_path,
    )

    return {
        "algorithm": algorithm,
        "display_name": info["display_name"],
        "metrics": metrics,
        "training_duration_ms": elapsed_ms,
        "model_path": model_path,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_models(model_runs: list[dict], problem_type: str) -> dict:
    """Rank model runs and produce a plain-English comparison summary.

    Args:
        model_runs: list of dicts with keys: id, algorithm, display_name, metrics
        problem_type: "classification" | "regression"
    """
    if not model_runs:
        return {"summary": "No models trained yet.", "best_model_id": None, "rankings": []}

    primary_metric = "accuracy" if problem_type == "classification" else "r2"

    ranked = sorted(
        model_runs,
        key=lambda r: (r.get("metrics") or {}).get(primary_metric, -999),
        reverse=True,
    )

    return {
        "summary": _comparison_summary(ranked, primary_metric, problem_type),
        "best_model_id": ranked[0].get("id"),
        "primary_metric": primary_metric,
        "rankings": [
            {
                "id": r.get("id"),
                "algorithm": r.get("algorithm"),
                "display_name": r.get("display_name"),
                "metrics": r.get("metrics"),
                "training_duration_ms": r.get("training_duration_ms"),
                "is_selected": r.get("is_selected", False),
                "rank": i + 1,
            }
            for i, r in enumerate(ranked)
        ],
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Split df into feature matrix X and target array y.

    - Drops all-null columns and the target from X
    - Encodes classification target to integers if needed
    """
    from sklearn.preprocessing import LabelEncoder

    # Feature columns: everything except target, drop all-null
    feature_cols = [
        c for c in df.columns
        if c != target_col and df[c].notna().any()
    ]
    X = df[feature_cols].copy()

    # Target
    y_series = df[target_col].copy()
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_series):
        le = LabelEncoder()
        y = le.fit_transform(y_series.fillna("MISSING").astype(str))
    else:
        y = y_series.fillna(y_series.median() if pd.api.types.is_numeric_dtype(y_series) else 0).values

    return X, y, feature_cols


def _build_pipeline(
    X: pd.DataFrame,
    estimator_cls: type,
    params: dict,
    needs_scaling: bool,
) -> Pipeline:
    """Build a preprocessing + estimator Pipeline.

    Numeric columns → median impute → optional StandardScaler
    Categorical columns → 'missing' impute → OrdinalEncoder
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    transformers = []
    if numeric_cols:
        num_steps: list = [("imputer", SimpleImputer(strategy="median"))]
        if needs_scaling:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), numeric_cols))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical_cols,
        ))

    if not transformers:
        preprocessor: Any = "passthrough"
    else:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator_cls(**params))])


def _compute_cv_metrics(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    problem_type: str,
    cv: int,
) -> dict[str, float]:
    """Run cross-validation and return averaged metrics."""
    if problem_type == "classification":
        scoring: dict = {
            "accuracy": "accuracy",
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
        }
        if len(np.unique(y)) == 2:
            scoring["roc_auc"] = "roc_auc"
    else:
        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }

    try:
        cv_results = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring,
            error_score=0.0,  # fill with 0 rather than raising on edge cases
        )
    except Exception:
        # Last resort: fit on all data, return zeros
        pipeline.fit(X, y)
        return {}

    metrics: dict[str, float] = {}
    for key, values in cv_results.items():
        if not key.startswith("test_"):
            continue
        metric_name = key[5:]
        val = float(np.mean(values))
        # sklearn stores MAE/RMSE as negatives
        if metric_name in ("mae", "rmse"):
            val = abs(val)
        metrics[metric_name] = round(val, 4)

    return metrics


def _selection_reason(algorithm: str, n_rows: int, n_features: int) -> str:
    reasons = {
        "logistic_regression": (
            f"Good baseline for this {n_rows}-row dataset. "
            "Fast to train and easy to explain."
        ),
        "decision_tree": (
            "Maximum explainability — you can follow the exact decision "
            "path for any prediction."
        ),
        "random_forest": (
            f"Excellent all-rounder for {n_features} features. "
            "Handles non-linear patterns and feature interactions automatically."
        ),
        "gradient_boosting": (
            "Typically the most accurate for structured data. "
            "Worth the extra training time when accuracy is the top priority."
        ),
        "linear_regression": (
            f"Good baseline for {n_rows} rows. "
            "Assumes linear relationships between features and target."
        ),
        "ridge": (
            "Regularized linear regression — more stable than plain linear "
            "regression when features are correlated."
        ),
    }
    return reasons.get(algorithm, "Recommended for this dataset size and type.")


def _comparison_summary(
    ranked: list[dict],
    primary_metric: str,
    problem_type: str,
) -> str:
    best = ranked[0]
    best_score = (best.get("metrics") or {}).get(primary_metric)

    if best_score is None:
        return f"{best['display_name']} completed training."

    if problem_type == "classification":
        score_str = f"{best_score:.1%} {primary_metric}"
    else:
        score_str = f"R²={best_score:.3f}"

    summary = f"{best['display_name']} leads with {score_str}."

    if len(ranked) > 1:
        second = ranked[1]
        second_score = (second.get("metrics") or {}).get(primary_metric, 0)
        if problem_type == "classification":
            diff = best_score - second_score
            if diff < 0.02:
                summary += (
                    f" {second['display_name']} is nearly as accurate "
                    f"({second_score:.1%}) — consider it if explainability matters more."
                )
            else:
                summary += f" {second['display_name']} is the runner-up at {second_score:.1%}."
        else:
            summary += f" {second['display_name']} is the runner-up (R²={second_score:.3f})."

    return summary
