"""Model training: recommendations, feature preparation, training, metrics.

Design principles:
- recommend_models uses heuristics (no LLM) for speed and determinism.
- train_single_model is synchronous and self-contained — easy to test.
- Metrics are returned with plain-English summaries for the UI.
- No mutation of input DataFrames.
- Models are serialized with joblib for later prediction.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

def _build_regression_algorithms() -> dict[str, dict]:
    algos: dict[str, dict] = {
        "linear_regression": {
            "name": "Linear Regression",
            "description": "Fits a straight-line relationship between features and target.",
            "plain_english": (
                "Like drawing the best-fit line through your data. Fast and easy to "
                "explain — great when relationships are roughly linear."
            ),
            "best_for": "Small datasets, linear relationships, maximum explainability",
            "class": LinearRegression,
            "params": {},
        },
        "random_forest_regressor": {
            "name": "Random Forest",
            "description": "Trains many decision trees and averages their predictions.",
            "plain_english": (
                "Like asking 100 independent experts and averaging their answers. "
                "Handles complex, non-linear patterns without overfitting."
            ),
            "best_for": "Most tabular datasets — a reliable default choice",
            "class": RandomForestRegressor,
            "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        },
        "gradient_boosting_regressor": {
            "name": "Gradient Boosting",
            "description": "Trains trees sequentially, each correcting the previous one's errors.",
            "plain_english": (
                "Like a student who keeps improving by focusing on past mistakes. "
                "Often the most accurate — but takes longer to train."
            ),
            "best_for": "When maximum accuracy matters; works well with larger datasets",
            "class": GradientBoostingRegressor,
            "params": {"n_estimators": 100, "random_state": 42},
        },
    }
    if _XGBOOST_AVAILABLE:
        algos["xgboost_regressor"] = {
            "name": "XGBoost",
            "description": "Extreme Gradient Boosting — a highly optimized gradient boosting library.",
            "plain_english": (
                "The algorithm behind many Kaggle competition winners. Extremely fast "
                "and accurate — handles missing values automatically."
            ),
            "best_for": "Structured tabular data where you need the best possible accuracy",
            "class": XGBRegressor,
            "params": {
                "n_estimators": 100, "random_state": 42,
                "verbosity": 0, "n_jobs": -1,
            },
        }
    if _LIGHTGBM_AVAILABLE:
        algos["lightgbm_regressor"] = {
            "name": "LightGBM",
            "description": "Microsoft's gradient boosting framework — fast on large datasets.",
            "plain_english": (
                "Like XGBoost but trains leaf-first instead of level-first, "
                "so it's often faster and uses less memory on big datasets."
            ),
            "best_for": "Large datasets where training speed matters as much as accuracy",
            "class": LGBMRegressor,
            "params": {
                "n_estimators": 100, "random_state": 42,
                "n_jobs": -1, "verbose": -1,
            },
        }
    algos["neural_network_regressor"] = {
        "name": "Neural Network",
        "description": "A multi-layer perceptron that learns non-linear patterns.",
        "plain_english": (
            "Inspired by the brain — layers of connected nodes learn complex, "
            "non-linear relationships. Can capture patterns that trees miss, "
            "but needs more data to shine."
        ),
        "best_for": "Medium-to-large datasets with complex, non-linear relationships",
        "class": MLPRegressor,
        "params": {
            "hidden_layer_sizes": (100, 50),
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
            "validation_fraction": 0.1,
        },
    }
    return algos


def _build_classification_algorithms() -> dict[str, dict]:
    algos: dict[str, dict] = {
        "logistic_regression": {
            "name": "Logistic Regression",
            "description": "Estimates probability of each class using a linear model.",
            "plain_english": (
                "Adds up the evidence for each category and picks the most likely one. "
                "Returns probability scores, which are useful for understanding confidence."
            ),
            "best_for": "Binary classification, probability scores, small datasets",
            "class": LogisticRegression,
            "params": {"max_iter": 1000, "random_state": 42},
        },
        "random_forest_classifier": {
            "name": "Random Forest",
            "description": "Trains many decision trees and takes a majority vote.",
            "plain_english": (
                "Like asking 100 independent experts to vote. "
                "Handles complex patterns and imbalanced classes well."
            ),
            "best_for": "Most tabular classification tasks — a reliable default",
            "class": RandomForestClassifier,
            "params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        },
        "gradient_boosting_classifier": {
            "name": "Gradient Boosting",
            "description": "Trains trees sequentially to correct previous mistakes.",
            "plain_english": (
                "Keeps improving by focusing on the rows it got wrong last time. "
                "Often the most accurate, especially with imbalanced data."
            ),
            "best_for": "When accuracy is top priority; imbalanced class distributions",
            "class": GradientBoostingClassifier,
            "params": {"n_estimators": 100, "random_state": 42},
        },
    }
    if _XGBOOST_AVAILABLE:
        algos["xgboost_classifier"] = {
            "name": "XGBoost",
            "description": "Extreme Gradient Boosting — a highly optimized gradient boosting library.",
            "plain_english": (
                "The algorithm behind many Kaggle competition winners. Fast, accurate, "
                "and handles missing values and imbalanced classes gracefully."
            ),
            "best_for": "Structured tabular data — typically outperforms sklearn's GradientBoosting",
            "class": XGBClassifier,
            "params": {
                "n_estimators": 100, "random_state": 42,
                "verbosity": 0, "n_jobs": -1,
                "eval_metric": "logloss",
            },
        }
    if _LIGHTGBM_AVAILABLE:
        algos["lightgbm_classifier"] = {
            "name": "LightGBM",
            "description": "Microsoft's gradient boosting framework — fast on large datasets.",
            "plain_english": (
                "Like XGBoost but optimized for speed. Trains leaf-first for "
                "better accuracy per second on large datasets."
            ),
            "best_for": "Large datasets, categorical features with many levels",
            "class": LGBMClassifier,
            "params": {
                "n_estimators": 100, "random_state": 42,
                "n_jobs": -1, "verbose": -1,
            },
        }
    algos["neural_network_classifier"] = {
        "name": "Neural Network",
        "description": "A multi-layer perceptron that learns non-linear patterns.",
        "plain_english": (
            "Inspired by the brain — layers of connected nodes learn complex, "
            "non-linear relationships between features. "
            "Returns probability scores for each class."
        ),
        "best_for": "Complex classification with many features and sufficient data",
        "class": MLPClassifier,
        "params": {
            "hidden_layer_sizes": (100, 50),
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
            "validation_fraction": 0.1,
        },
    }
    return algos


REGRESSION_ALGORITHMS: dict[str, dict] = _build_regression_algorithms()
CLASSIFICATION_ALGORITHMS: dict[str, dict] = _build_classification_algorithms()


# ---------------------------------------------------------------------------
# Model recommendations
# ---------------------------------------------------------------------------

def recommend_models(
    problem_type: str,
    n_rows: int,
    n_features: int,
) -> list[dict]:
    """Return 2–4 algorithm recommendations tailored to the dataset.

    Each recommendation includes: algorithm key, name, plain-English description,
    and a dataset-specific 'recommended_because' explanation.
    """
    algorithms = (
        REGRESSION_ALGORITHMS if problem_type == "regression"
        else CLASSIFICATION_ALGORITHMS
    )

    recommendations = []
    for key, info in algorithms.items():
        rec = {
            "algorithm": key,
            "name": info["name"],
            "description": info["description"],
            "plain_english": info["plain_english"],
            "best_for": info["best_for"],
            "recommended_because": _why_recommended(key, n_rows, n_features),
        }
        recommendations.append(rec)

    return recommendations


def _why_recommended(algorithm: str, n_rows: int, n_features: int) -> str:
    if "neural_network" in algorithm:
        if n_rows < 500:
            return (
                f"Neural networks need data to learn — {n_rows} rows is on the small "
                "side, so other algorithms may outperform it here."
            )
        return (
            f"With {n_rows} rows and {n_features} features, a neural network can "
            "capture non-linear patterns that tree models might miss."
        )
    if n_rows < 200:
        if "linear" in algorithm or "logistic" in algorithm:
            return (
                f"With only {n_rows} rows, simpler models tend to generalize better "
                "and are less likely to memorize noise in the training data."
            )
        if "xgboost" in algorithm or "lightgbm" in algorithm:
            return (
                f"Included for comparison — {n_rows} rows is small for boosting, "
                "but may still outperform simpler models on complex patterns."
            )
        return (
            f"Included for comparison — {n_rows} rows is on the small side, "
            "so a simpler model may actually outperform this one."
        )
    if n_rows < 2000:
        if "random_forest" in algorithm:
            return (
                f"A solid choice for {n_rows} rows — balances accuracy and "
                "training speed without overfitting."
            )
        if "xgboost" in algorithm:
            return (
                f"Often beats sklearn's GradientBoosting at {n_rows} rows — "
                "try it alongside Random Forest for a fair comparison."
            )
        if "lightgbm" in algorithm:
            return (
                f"Trains faster than GradientBoosting at {n_rows} rows and "
                "typically achieves similar or better accuracy."
            )
        if "gradient_boosting" in algorithm:
            return (
                f"May achieve higher accuracy than Random Forest at {n_rows} rows, "
                "though it trains a bit slower."
            )
        return f"Good baseline for {n_rows} rows — fast to train and easy to explain."
    # Large dataset (2000+ rows)
    if "xgboost" in algorithm:
        return (
            f"An excellent choice for {n_rows} rows — handles missing values "
            "natively and is often the most accurate algorithm on tabular data."
        )
    if "lightgbm" in algorithm:
        return (
            f"Designed for large datasets like yours ({n_rows} rows) — "
            "trains faster than XGBoost with comparable accuracy."
        )
    if "gradient_boosting" in algorithm:
        return (
            f"With {n_rows} rows, gradient boosting can learn complex patterns "
            "and often achieves high accuracy."
        )
    return f"Reliable and well-tested — scales well to {n_rows} rows."


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    problem_type: str,
) -> tuple[np.ndarray, np.ndarray, Optional[LabelEncoder]]:
    """Prepare feature matrix X and target vector y for sklearn.

    - Drops rows where the target is missing.
    - Fills numeric NAs with column median.
    - Label-encodes string features.
    - Label-encodes string targets (classification only).

    Returns (X, y, label_encoder_or_None).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    valid_features = [c for c in feature_cols if c in df.columns]
    if not valid_features:
        raise ValueError("No valid feature columns found in DataFrame.")

    # Drop rows with missing target
    df_clean = df[valid_features + [target_col]].dropna(subset=[target_col]).reset_index(drop=True)

    if len(df_clean) < 2:
        raise ValueError("Not enough non-null rows to train a model.")

    # Build X
    X_parts: list[np.ndarray] = []
    for col in valid_features:
        series = df_clean[col]
        if pd.api.types.is_numeric_dtype(series):
            filled = series.fillna(series.median())
            X_parts.append(filled.values.reshape(-1, 1))
        else:
            le = LabelEncoder()
            encoded = le.fit_transform(series.fillna("MISSING").astype(str))
            X_parts.append(encoded.reshape(-1, 1))

    X = np.hstack(X_parts).astype(float)

    # Build y
    y_series = df_clean[target_col]
    le_target: Optional[LabelEncoder] = None
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_series):
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_series.astype(str))
    else:
        y = y_series.values.astype(float) if problem_type == "regression" else y_series.values

    return X, y, le_target


# ---------------------------------------------------------------------------
# Train a single model
# ---------------------------------------------------------------------------

def train_single_model(
    X: np.ndarray,
    y: np.ndarray,
    algorithm: str,
    problem_type: str,
    model_dir: Path,
    model_run_id: str,
) -> dict:
    """Train one sklearn model, compute held-out metrics, and save to disk.

    Returns:
        {metrics, model_path, training_duration_ms, summary}
    """
    algorithms = (
        REGRESSION_ALGORITHMS if problem_type == "regression"
        else CLASSIFICATION_ALGORITHMS
    )
    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Valid choices: {sorted(algorithms)}"
        )

    info = algorithms[algorithm]
    model_class = info["class"]
    params = info["params"]

    # Train/test split — use 20% for test when we have enough rows
    n = len(X)
    if n >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # Too few rows — train/eval on same data (metrics will be optimistic)
        X_train = X_test = X
        y_train = y_test = y

    start = time.time()
    model = model_class(**params)
    model.fit(X_train, y_train)
    elapsed_ms = int((time.time() - start) * 1000)

    y_pred = model.predict(X_test)

    if problem_type == "regression":
        metrics = _regression_metrics(y_test, y_pred)
        summary = _regression_summary(metrics)
    else:
        metrics = _classification_metrics(y_test, y_pred)
        summary = _classification_summary(metrics)

    metrics["train_size"] = len(X_train)
    metrics["test_size"] = len(X_test)

    # Persist model
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_dir / f"{model_run_id}.joblib")
    joblib.dump(model, model_path)

    return {
        "metrics": metrics,
        "model_path": model_path,
        "training_duration_ms": elapsed_ms,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes == 2 else "weighted"
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "precision": round(
            float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4
        ),
    }


def _regression_summary(metrics: dict) -> str:
    r2 = metrics["r2"]
    mae = metrics["mae"]
    if r2 >= 0.9:
        quality = "excellent fit"
    elif r2 >= 0.7:
        quality = "good fit"
    elif r2 >= 0.5:
        quality = "moderate fit"
    else:
        quality = "weak fit"
    return (
        f"R² = {r2:.2f} ({quality} — 1.0 would be perfect). "
        f"On average, predictions are off by {mae:.2f} units (MAE)."
    )


def _classification_summary(metrics: dict) -> str:
    acc = metrics["accuracy"]
    f1 = metrics["f1"]
    pct = round(acc * 100, 1)
    return (
        f"{pct}% accuracy on the held-out test set. "
        f"F1 = {f1:.2f} (balances precision and recall; 1.0 is perfect)."
    )


# ---------------------------------------------------------------------------
# Best model selection
# ---------------------------------------------------------------------------

def pick_best_model(models: list[dict], problem_type: str) -> dict | None:
    """Return the model dict with the highest primary metric."""
    if not models:
        return None

    def score(m: dict) -> float:
        metrics = m.get("metrics") or {}
        if problem_type == "regression":
            return metrics.get("r2", -999.0)
        return metrics.get("f1", -999.0)

    best = max(models, key=score)
    metric_name = "R²" if problem_type == "regression" else "F1 score"
    return {
        "model_run_id": best["id"],
        "algorithm": best["algorithm"],
        "reason": (
            f"Highest {metric_name} among trained models "
            f"({score(best):.3f})."
        ),
    }
