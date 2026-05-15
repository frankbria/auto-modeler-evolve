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
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from core.validator import run_cross_validation

try:
    from imblearn.over_sampling import SMOTE as _SMOTE

    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

# Algorithms that support class_weight="balanced" constructor param
_CLASS_WEIGHT_PARAM_ALGOS = frozenset(
    {"logistic_regression", "random_forest_classifier", "lightgbm_classifier"}
)
# Algorithms that accept sample_weight in fit() but not class_weight param
_SAMPLE_WEIGHT_FIT_ALGOS = frozenset(
    {"gradient_boosting_classifier", "xgboost_classifier"}
)

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
                "n_estimators": 100,
                "random_state": 42,
                "verbosity": 0,
                "n_jobs": -1,
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
                "n_estimators": 100,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
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
    algos["voting_regressor"] = {
        "name": "Voting Ensemble",
        "description": "Averages predictions from Linear Regression, Random Forest, and Gradient Boosting.",
        "plain_english": (
            "Gets a second opinion from three different models and averages their answers. "
            "Like consulting multiple experts — the combined wisdom is often more accurate "
            "than any single model."
        ),
        "best_for": "When you want the best possible accuracy; often beats any single model",
        "class": None,  # built dynamically in _train_ensemble_model
        "params": {},
        "is_ensemble": True,
        "ensemble_type": "voting",
        "base_algorithms": [
            "linear_regression",
            "random_forest_regressor",
            "gradient_boosting_regressor",
        ],
    }
    algos["stacking_regressor"] = {
        "name": "Stacking Ensemble",
        "description": "Uses a meta-learner to weight Linear Regression, Random Forest, and Gradient Boosting.",
        "plain_english": (
            "Trains three models, then trains a fourth model to learn the optimal combination. "
            "More sophisticated than simple averaging — the meta-learner figures out which "
            "base model to trust most for different patterns in the data."
        ),
        "best_for": "Medium-to-large datasets where squeezing out the last bit of accuracy matters",
        "class": None,
        "params": {},
        "is_ensemble": True,
        "ensemble_type": "stacking",
        "base_algorithms": [
            "linear_regression",
            "random_forest_regressor",
            "gradient_boosting_regressor",
        ],
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
                "n_estimators": 100,
                "random_state": 42,
                "verbosity": 0,
                "n_jobs": -1,
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
                "n_estimators": 100,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
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
    algos["voting_classifier"] = {
        "name": "Voting Ensemble",
        "description": "Combines Logistic Regression, Random Forest, and Gradient Boosting through probability averaging.",
        "plain_english": (
            "Asks three different models for their best guess (as probabilities) and averages "
            "the votes. Like a panel of judges — 3 out of 3 agreeing is more reliable than "
            "a single expert. Uses soft voting for maximum accuracy."
        ),
        "best_for": "When you want the most reliable predictions; often the best overall accuracy",
        "class": None,
        "params": {},
        "is_ensemble": True,
        "ensemble_type": "voting",
        "base_algorithms": [
            "logistic_regression",
            "random_forest_classifier",
            "gradient_boosting_classifier",
        ],
    }
    algos["stacking_classifier"] = {
        "name": "Stacking Ensemble",
        "description": "Trains a meta-learner on top of Logistic Regression, Random Forest, and Gradient Boosting.",
        "plain_english": (
            "Trains three models, then trains a fourth model to learn the optimal way to "
            "combine their predictions. More nuanced than simple voting — the meta-learner "
            "learns which base model to trust for which patterns."
        ),
        "best_for": "Medium-to-large datasets; often the highest accuracy of all ensemble types",
        "class": None,
        "params": {},
        "is_ensemble": True,
        "ensemble_type": "stacking",
        "base_algorithms": [
            "logistic_regression",
            "random_forest_classifier",
            "gradient_boosting_classifier",
        ],
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
        REGRESSION_ALGORITHMS
        if problem_type == "regression"
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
    if "voting" in algorithm or "stacking" in algorithm:
        if n_rows < 200:
            return (
                f"With only {n_rows} rows, individual models may outperform ensembles — "
                "but worth trying alongside simpler algorithms for comparison."
            )
        return (
            f"With {n_rows} rows, combining multiple models often produces better "
            "predictions than any single model. A great default when accuracy matters most."
        )
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
# Class imbalance detection
# ---------------------------------------------------------------------------


def detect_class_imbalance(y: np.ndarray) -> dict:
    """Detect class imbalance in a classification target array.

    Minority class threshold: < 20% of total rows.

    Returns a dict with:
        is_imbalanced, class_distribution, minority_class, minority_ratio,
        recommended_strategy ("class_weight" | "smote" | "threshold" | "none"),
        explanation (plain-English string).
    """
    classes, counts = np.unique(y, return_counts=True)
    total = int(len(y))
    distribution = [
        {"class": str(cls), "count": int(cnt), "ratio": round(float(cnt) / total, 4)}
        for cls, cnt in zip(classes, counts)
    ]

    min_ratio = float(min(d["ratio"] for d in distribution))
    minority = min(distribution, key=lambda d: d["ratio"])
    is_imbalanced = min_ratio < 0.20

    if not is_imbalanced:
        return {
            "is_imbalanced": False,
            "class_distribution": distribution,
            "minority_class": None,
            "minority_ratio": round(min_ratio, 4),
            "recommended_strategy": "none",
            "explanation": (
                "Your target classes are roughly balanced — no special handling needed."
            ),
        }

    minority_pct = round(min_ratio * 100, 1)
    minority_class_str = minority["class"]
    n = total

    # Choose recommended strategy based on severity and dataset size
    if min_ratio < 0.05 and n >= 100:
        strategy = "smote"
        reason = (
            "Severe imbalance (under 5% minority) benefits most from SMOTE, which "
            "creates realistic synthetic minority examples to balance training data."
        )
    elif n < 50:
        strategy = "class_weight"
        reason = (
            "With a small dataset, class weighting is the safest approach — "
            "SMOTE would create too many synthetic samples relative to real data."
        )
    else:
        strategy = "class_weight"
        reason = (
            "Class weighting tells the model to pay more attention to the minority "
            "class during training, without creating synthetic data."
        )

    return {
        "is_imbalanced": True,
        "class_distribution": distribution,
        "minority_class": minority_class_str,
        "minority_ratio": round(min_ratio, 4),
        "recommended_strategy": strategy,
        "explanation": (
            f"Your data has a class imbalance: only {minority_pct}% of rows belong to "
            f"'{minority_class_str}'. Without correction, the model will be biased "
            f"toward predicting the majority class and may miss the cases that matter "
            f"most. Recommended strategy: {reason}"
        ),
    }


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
    df_clean = (
        df[valid_features + [target_col]]
        .dropna(subset=[target_col])
        .reset_index(drop=True)
    )

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
        y = (
            y_series.values.astype(float)
            if problem_type == "regression"
            else y_series.values
        )

    return X, y, le_target


# ---------------------------------------------------------------------------
# Ensemble helpers
# ---------------------------------------------------------------------------


def _build_ensemble_estimators(
    base_algorithm_keys: list[str],
    algorithms: dict[str, dict],
) -> list[tuple[str, object]]:
    """Build (name, estimator) tuples for VotingClassifier/Regressor or Stacking.

    Only includes algorithms that are present in the registry (skips optional ones
    like xgboost/lightgbm if not installed).
    """
    estimators = []
    for key in base_algorithm_keys:
        if key in algorithms and algorithms[key].get("class") is not None:
            info = algorithms[key]
            estimators.append((key, info["class"](**dict(info["params"]))))
    return estimators


def _ensemble_vote_explanation(
    base_names: list[str],
    estimators_fitted: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
    classes: list | None = None,
) -> dict:
    """Build per-base-model vote/prediction data for ensemble explainability.

    Returns a dict suitable for storing in metrics["ensemble_votes"].
    """
    votes: dict[str, object] = {}
    for name, est in zip(base_names, estimators_fitted):
        try:
            if problem_type == "regression":
                preds = est.predict(X_test)
                votes[name] = round(float(np.mean(preds)), 4)
            else:
                # Classification: record per-class vote counts
                preds = est.predict(X_test)
                from collections import Counter

                cnt = Counter(int(p) for p in preds)
                if classes is not None:
                    votes[name] = {str(classes[k]): v for k, v in cnt.items()}
                else:
                    votes[name] = {str(k): v for k, v in cnt.items()}
        except Exception:  # noqa: BLE001
            pass
    return votes


def _stacking_weight_explanation(
    base_names: list[str],
    final_estimator,
) -> dict[str, float]:
    """Extract normalized meta-learner weights from a fitted stacking model.

    Returns {base_name: weight_fraction} where fractions sum to ~1.
    For binary classification (coef_ shape [1, n_estimators]) and multiclass
    (shape [n_classes, n_estimators]) both handled.
    """
    if not hasattr(final_estimator, "coef_"):
        return {}

    coef = np.array(final_estimator.coef_)
    if coef.ndim == 2:
        # Take mean absolute coefficient across classes for multiclass
        magnitudes = np.abs(coef).mean(axis=0)
    else:
        magnitudes = np.abs(coef)

    # Keep only one value per base estimator (stacking may passthrough features)
    n_bases = len(base_names)
    magnitudes = magnitudes[:n_bases]

    total = float(magnitudes.sum())
    if total == 0:
        return {name: round(1.0 / n_bases, 4) for name in base_names}

    return {
        name: round(float(magnitudes[i]) / total, 4)
        for i, name in enumerate(base_names)
    }


def _train_ensemble_model(
    X: np.ndarray,
    y: np.ndarray,
    algorithm: str,
    problem_type: str,
    model_dir: "Path",
    model_run_id: str,
    split_strategy: str,
    date_col_used: Optional[str],
    info: dict,
) -> dict:
    """Build and train a VotingClassifier/Regressor or StackingClassifier/Regressor.

    Returns the same dict format as train_single_model.
    """
    algorithms = (
        REGRESSION_ALGORITHMS
        if problem_type == "regression"
        else CLASSIFICATION_ALGORITHMS
    )
    base_keys: list[str] = info["base_algorithms"]
    ensemble_type: str = info["ensemble_type"]

    estimators = _build_ensemble_estimators(base_keys, algorithms)
    if len(estimators) < 2:
        raise ValueError(
            "Ensemble requires at least 2 base algorithms. "
            "Install scikit-learn base algorithms or check the registry."
        )

    # Train/test split (same logic as train_single_model)
    n = len(X)
    if n >= 10:
        if split_strategy == "chronological":
            train_idx, test_idx = chronological_split(n)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
    else:
        X_train = X_test = X
        y_train = y_test = y

    # Build the ensemble model
    if ensemble_type == "voting":
        if problem_type == "regression":
            model = VotingRegressor(estimators=estimators)
        else:
            model = VotingClassifier(estimators=estimators, voting="soft")
    else:  # stacking
        if problem_type == "regression":
            model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=min(5, max(2, n // 4)),
            )
        else:
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                cv=min(5, max(2, n // 4)),
            )

    start = time.time()
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
    metrics["split_strategy"] = split_strategy
    if split_strategy == "chronological" and date_col_used:
        metrics["date_col_used"] = date_col_used
        metrics["split_explanation"] = (
            f"Used time-based splitting on '{date_col_used}' — training on older data "
            "and testing on more recent data."
        )

    # Ensemble explainability
    base_names = [e[0] for e in estimators]
    fitted_estimators = getattr(model, "estimators_", [])
    metrics["ensemble_type"] = ensemble_type

    if ensemble_type == "voting" and fitted_estimators:
        classes = getattr(model, "classes_", None)
        metrics["ensemble_votes"] = _ensemble_vote_explanation(
            base_names,
            fitted_estimators,
            X_test,
            y_test,
            problem_type,
            classes=list(classes) if classes is not None else None,
        )
        # Plain-English vote summary for classification
        if problem_type == "classification" and metrics["ensemble_votes"]:
            n_models = len(metrics["ensemble_votes"])
            y_test_pred = model.predict(X_test)
            from collections import Counter as _Counter

            top_class = _Counter(str(p) for p in y_test_pred).most_common(1)[0][0]
            agreeing = sum(
                1
                for v in metrics["ensemble_votes"].values()
                if isinstance(v, dict) and max(v, key=v.get, default="") == top_class  # type: ignore[arg-type]
            )
            metrics["ensemble_summary"] = (
                f"{agreeing} out of {n_models} models voted for '{top_class}' "
                f"(majority class on held-out test set)."
            )
    elif ensemble_type == "stacking" and hasattr(model, "final_estimator_"):
        metrics["stacking_weights"] = _stacking_weight_explanation(
            base_names, model.final_estimator_
        )
        if metrics["stacking_weights"]:
            top_base = max(
                metrics["stacking_weights"], key=metrics["stacking_weights"].get
            )  # type: ignore[arg-type]
            top_pct = round(metrics["stacking_weights"][top_base] * 100)
            metrics["ensemble_summary"] = (
                f"Meta-learner trusted '{top_base}' most "
                f"({top_pct}% of weight) when combining predictions."
            )

    # Plain-English addendum to summary
    n_bases = len(base_names)
    base_display = ", ".join(
        algorithms[k]["name"] for k in base_names if k in algorithms
    )
    if ensemble_type == "voting":
        suffix = f" Combines {n_bases} models ({base_display}) via soft voting."
    else:
        suffix = f" Stacks {n_bases} models ({base_display}) through a meta-learner."
    summary = summary + suffix

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
# Train a single model
# ---------------------------------------------------------------------------


_LARGE_DATASET_THRESHOLD = 50_000
_LARGE_DATASET_SAMPLE_SIZE = 20_000


def sample_large_dataset(
    df: pd.DataFrame,
    max_rows: int = _LARGE_DATASET_SAMPLE_SIZE,
    threshold: int = _LARGE_DATASET_THRESHOLD,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Sub-sample a large DataFrame to prevent OOM during training.

    Returns (sampled_df, sample_info) where sample_info has:
      - was_sampled: bool
      - original_rows: int
      - sample_rows: int
      - note: str  (plain-English message for the analyst)
    """
    original_rows = len(df)
    if original_rows <= threshold:
        return df, {
            "was_sampled": False,
            "original_rows": original_rows,
            "sample_rows": original_rows,
            "note": "",
        }

    sampled = df.sample(n=max_rows, random_state=random_state)
    note = (
        f"Trained on {max_rows:,} of {original_rows:,} rows (random sample — "
        "full dataset is too large for in-memory training)."
    )
    return sampled, {
        "was_sampled": True,
        "original_rows": original_rows,
        "sample_rows": max_rows,
        "note": note,
    }


def chronological_split(
    n_rows: int,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, test_indices) for a chronological split.

    Assumes the data is pre-sorted by date ascending.  The oldest
    (1 - test_size) fraction becomes training data; the newest test_size
    fraction becomes the held-out test set.
    """
    n_test = max(1, int(n_rows * test_size))
    n_train = n_rows - n_test
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_rows)
    return train_idx, test_idx


def train_single_model(
    X: np.ndarray,
    y: np.ndarray,
    algorithm: str,
    problem_type: str,
    model_dir: Path,
    model_run_id: str,
    imbalance_strategy: Optional[str] = None,
    split_strategy: str = "random",
    date_col_used: Optional[str] = None,
) -> dict:
    """Train one sklearn model, compute held-out metrics, and save to disk.

    imbalance_strategy: "class_weight" | "smote" | "threshold" | None
        Only applied for classification problems.
    split_strategy: "random" | "chronological"
        "chronological" assumes X/y are pre-sorted oldest-first; the last 20%
        of rows are used as the test set.  Only meaningful when the training
        DataFrame was sorted by a date column before prepare_features().
    date_col_used: the name of the date column used for sorting (metadata only).

    Returns:
        {metrics, model_path, training_duration_ms, summary}
    """
    algorithms = (
        REGRESSION_ALGORITHMS
        if problem_type == "regression"
        else CLASSIFICATION_ALGORITHMS
    )
    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Valid choices: {sorted(algorithms)}"
        )

    info = algorithms[algorithm]

    # Dispatch ensemble algorithms to the ensemble trainer
    if info.get("is_ensemble"):
        return _train_ensemble_model(
            X,
            y,
            algorithm,
            problem_type,
            model_dir,
            model_run_id,
            split_strategy,
            date_col_used,
            info,
        )

    model_class = info["class"]
    params = dict(info["params"])  # copy so we don't mutate the registry

    # Train/test split
    n = len(X)
    if n >= 10:
        if split_strategy == "chronological":
            train_idx, test_idx = chronological_split(n)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
    else:
        # Too few rows — train/eval on same data (metrics will be optimistic)
        X_train = X_test = X
        y_train = y_test = y

    # ---- Class imbalance strategy ----
    apply_threshold_tuning = False
    if problem_type == "classification" and imbalance_strategy:
        if imbalance_strategy == "smote" and _SMOTE_AVAILABLE and len(y_train) >= 6:
            smote = _SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        elif imbalance_strategy == "class_weight":
            if algorithm in _CLASS_WEIGHT_PARAM_ALGOS:
                params["class_weight"] = "balanced"
            # GBC and XGB support sample_weight in fit — handled below
        elif imbalance_strategy == "threshold":
            apply_threshold_tuning = True

    # Calibration is applied for classifiers unless threshold tuning or SMOTE is
    # selected (those strategies already manipulate probabilities / training distribution).
    # Also skip for sample_weight algorithms (GBC, XGB with class_weight) because
    # CalibratedClassifierCV can't pass sample_weight through its internal CV.
    _skip_calibration = (
        problem_type != "classification"
        or apply_threshold_tuning
        or imbalance_strategy == "smote"
        or (
            imbalance_strategy == "class_weight"
            and algorithm in _SAMPLE_WEIGHT_FIT_ALGOS
        )
        or len(X_train) < 30
    )

    start = time.time()
    model = model_class(**params)

    if (
        problem_type == "classification"
        and imbalance_strategy == "class_weight"
        and algorithm in _SAMPLE_WEIGHT_FIT_ALGOS
    ):
        sample_weights = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    # Apply CalibratedClassifierCV for well-calibrated predict_proba outputs.
    # Uses 3-fold CV internally so the training data is not double-used.
    model_to_save = model
    if not _skip_calibration:
        try:
            calibrated = CalibratedClassifierCV(
                model_class(**params), cv=3, method="sigmoid"
            )
            calibrated.fit(X_train, y_train)
            model_to_save = calibrated
        except Exception:  # noqa: BLE001
            pass  # Fall back to uncalibrated model

    elapsed_ms = int((time.time() - start) * 1000)

    if problem_type == "classification" and apply_threshold_tuning:
        y_pred, optimal_threshold = _tune_threshold(model, X_test, y_test)
    else:
        y_pred = model.predict(X_test)
        optimal_threshold = None

    if problem_type == "regression":
        metrics = _regression_metrics(y_test, y_pred)
        summary = _regression_summary(metrics)
    else:
        metrics = _classification_metrics(y_test, y_pred)
        summary = _classification_summary(metrics)
        if optimal_threshold is not None:
            metrics["optimal_threshold"] = round(float(optimal_threshold), 2)
        # Compute reliability diagram data from calibrated model
        if not _skip_calibration and model_to_save is not model:
            _add_calibration_metrics(metrics, model_to_save, X_test, y_test)

    if imbalance_strategy and imbalance_strategy != "none":
        metrics["imbalance_strategy"] = imbalance_strategy

    metrics["train_size"] = len(X_train)
    metrics["test_size"] = len(X_test)
    metrics["split_strategy"] = split_strategy
    if split_strategy == "chronological" and date_col_used:
        metrics["date_col_used"] = date_col_used
        metrics["split_explanation"] = (
            f"Used time-based splitting on '{date_col_used}' — training on older data "
            "and testing on more recent data gives a more honest picture of how the "
            "model will perform on future data."
        )

    # Persist model (calibrated when applicable)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_dir / f"{model_run_id}.joblib")
    joblib.dump(model_to_save, model_path)

    # Run cross-validation on the full dataset so the training panel can show
    # CV score ± std alongside the train/test split metrics.  Skip for tiny
    # datasets (CV needs at least 2 rows) and catch all errors so CV never
    # blocks a successful training result.
    if len(X) >= 10:
        try:
            unfitted = model_class(**params)
            cv_result = run_cross_validation(unfitted, X, y, problem_type)
            if cv_result["mean"] is not None:
                metrics["cv_mean"] = cv_result["mean"]
                metrics["cv_std"] = cv_result["std"]
                metrics["cv_n_splits"] = cv_result["n_splits"]
        except Exception:  # noqa: BLE001
            pass

    return {
        "metrics": metrics,
        "model_path": model_path,
        "training_duration_ms": elapsed_ms,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _add_calibration_metrics(
    metrics: dict,
    calibrated_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Compute reliability diagram data and Brier score; mutate metrics in-place.

    Only runs for binary classification (calibration_curve is most meaningful there).
    Multiclass results in a ``is_calibrated=True`` flag without curve data.
    """
    try:
        n_classes = len(np.unique(y_test))
        metrics["is_calibrated"] = True
        if n_classes == 2:
            y_prob = calibrated_model.predict_proba(X_test)[:, 1]
            brier = float(brier_score_loss(y_test, y_prob))
            fraction_pos, mean_pred = calibration_curve(
                y_test, y_prob, n_bins=10, strategy="uniform"
            )
            metrics["brier_score"] = round(brier, 4)
            metrics["calibration_curve"] = [
                {"predicted": round(float(mp), 3), "actual": round(float(fp), 3)}
                for mp, fp in zip(mean_pred, fraction_pos)
            ]
            # Calibration quality summary
            max_deviation = (
                float(np.max(np.abs(fraction_pos - mean_pred)))
                if len(fraction_pos) > 0
                else 1.0
            )
            if max_deviation < 0.05:
                cal_quality = "well-calibrated"
            elif max_deviation < 0.15:
                cal_quality = "reasonably calibrated"
            else:
                cal_quality = "moderately calibrated"
            metrics["calibration_note"] = (
                f"Model is {cal_quality} (Brier score: {brier:.3f}). "
                "Bars close to the diagonal line mean confidence scores are trustworthy."
            )
    except Exception:  # noqa: BLE001
        pass  # Never crash training over a calibration display issue


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
    threshold_note = ""
    if "optimal_threshold" in metrics:
        threshold_note = (
            f" Decision threshold tuned to {metrics['optimal_threshold']:.2f} "
            f"to maximise F1 on imbalanced data."
        )
    return (
        f"{pct}% accuracy on the held-out test set. "
        f"F1 = {f1:.2f} (balances precision and recall; 1.0 is perfect).{threshold_note}"
    )


def _tune_threshold(model, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """Sweep decision thresholds to maximise F1 score (binary classification).

    Returns (y_pred_at_best_threshold, best_threshold).
    Falls back to standard predict() if model lacks predict_proba or is multiclass.
    """
    if not hasattr(model, "predict_proba"):
        return model.predict(X_test), None

    n_classes = len(np.unique(y_test))
    if n_classes != 2:
        return model.predict(X_test), None

    probas = model.predict_proba(X_test)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.05, 0.96, 0.05):
        preds = (probas >= thresh).astype(int)
        score = float(f1_score(y_test, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    y_pred = (probas >= best_thresh).astype(int)
    return y_pred, best_thresh


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

# Parameter grids for RandomizedSearchCV.
# Algorithms with no meaningful hyperparameters are excluded (linear_regression).
_TUNING_GRIDS: dict[str, dict] = {
    "random_forest_regressor": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "random_forest_classifier": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting_regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },
    "gradient_boosting_classifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "max_iter": [500, 1000, 2000],
    },
    "xgboost_regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
    "xgboost_classifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
    "lightgbm_regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 50],
    },
    "lightgbm_classifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 50],
    },
}


def get_tuning_grid(algorithm: str) -> dict | None:
    """Return the hyperparameter search grid for an algorithm, or None if not tunable."""
    return _TUNING_GRIDS.get(algorithm)


def tune_model(
    X: np.ndarray,
    y: np.ndarray,
    algorithm: str,
    problem_type: str,
    model_dir: Path,
    new_model_run_id: str,
    n_iter: int = 10,
    cv: int = 3,
) -> dict:
    """Run RandomizedSearchCV to find better hyperparameters for the given algorithm.

    Uses the base model class from the algorithm registry with the tuning grid.
    Trains the best estimator on the full training split and saves it.

    Returns:
        {
            best_params: dict,
            tuned_cv_score: float,    # CV score of best params
            metrics: dict,            # Held-out metrics of retrained tuned model
            model_path: str,
            training_duration_ms: int,
            summary: str,
            tunable: bool,            # False if no grid defined for this algorithm
        }
    """
    algorithms = (
        REGRESSION_ALGORITHMS
        if problem_type == "regression"
        else CLASSIFICATION_ALGORITHMS
    )
    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Valid choices: {sorted(algorithms)}"
        )

    param_grid = _TUNING_GRIDS.get(algorithm)
    info = algorithms[algorithm]
    model_class = info["class"]
    default_params = dict(info["params"])

    # Strip params that RandomizedSearchCV will override
    if not param_grid:
        # Not tunable — just retrain with defaults and return "already optimal"
        return {
            "best_params": default_params,
            "tuned_cv_score": None,
            "metrics": None,
            "model_path": None,
            "training_duration_ms": 0,
            "summary": (
                f"{info['name']} has no hyperparameters to tune — "
                "it's already using the optimal settings."
            ),
            "tunable": False,
        }

    # Train/test split
    n = len(X)
    if n >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train = X_test = X
        y_train = y_test = y

    # Scoring metric
    scoring = "r2" if problem_type == "regression" else "f1_weighted"

    # Base model with fixed defaults that we don't want to tune (e.g. random_state, verbosity)
    fixed_params = {k: v for k, v in default_params.items() if k not in param_grid}
    base_model = model_class(**fixed_params)

    start = time.time()
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=min(n_iter, 10),
        scoring=scoring,
        cv=min(cv, 3),
        random_state=42,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    elapsed_ms = int((time.time() - start) * 1000)

    best_model = search.best_estimator_
    best_params = {k: v for k, v in search.best_params_.items()}
    tuned_cv_score = round(float(search.best_score_), 4)

    y_pred = best_model.predict(X_test)

    if problem_type == "regression":
        metrics = _regression_metrics(y_test, y_pred)
        summary = _regression_summary(metrics)
    else:
        metrics = _classification_metrics(y_test, y_pred)
        summary = _classification_summary(metrics)

    metrics["train_size"] = len(X_train)
    metrics["test_size"] = len(X_test)

    # Save tuned model
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(model_dir / f"{new_model_run_id}.joblib")
    import joblib as _jl

    _jl.dump(best_model, model_path)

    return {
        "best_params": best_params,
        "tuned_cv_score": tuned_cv_score,
        "metrics": metrics,
        "model_path": model_path,
        "training_duration_ms": elapsed_ms,
        "summary": summary,
        "tunable": True,
    }


# ---------------------------------------------------------------------------
# Feature selection automation
# ---------------------------------------------------------------------------


def identify_weak_features(
    model,
    feature_cols: list[str],
    threshold_percentile: float = 20.0,
) -> dict:
    """Identify features with near-zero importance after training.

    Supports:
    - Tree-based models (RandomForest, GBT, XGB, LGB): uses `feature_importances_`
    - Linear models (LinearRegression, LogisticRegression, Ridge): uses `|coef_|`
    - MLP and ensembles: returns `has_importances=False`

    Returns:
        {
            "feature_importances": [
                {"name": str, "importance": float | None, "rank": int, "is_weak": bool},
                ...
            ],  # sorted by importance descending
            "weak_features": [str, ...],  # bottom threshold_percentile% by importance
            "threshold": float | None,
            "method": "feature_importances" | "coefficients" | "not_available",
            "has_importances": bool,
            "n_weak": int,
            "explanation": str,
        }
    """
    importances: np.ndarray | None = None
    method = "not_available"
    n_features = len(feature_cols)

    # Unwrap CalibratedClassifierCV to access the base estimator
    if hasattr(model, "calibrated_classifiers_"):
        try:
            model = model.calibrated_classifiers_[0].estimator
        except (AttributeError, IndexError):
            pass

    if hasattr(model, "feature_importances_"):
        raw = np.array(model.feature_importances_)
        if len(raw) == n_features:
            importances = raw
            method = "feature_importances"
    elif hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        if coef.ndim == 1 and len(coef) == n_features:
            importances = np.abs(coef)
            method = "coefficients"
        elif coef.ndim == 2 and coef.shape[1] == n_features:
            # Multiclass: take max absolute coefficient per feature
            importances = np.max(np.abs(coef), axis=0)
            method = "coefficients"

    if importances is None:
        return {
            "feature_importances": [
                {"name": col, "importance": None, "rank": i + 1, "is_weak": False}
                for i, col in enumerate(feature_cols)
            ],
            "weak_features": [],
            "threshold": None,
            "method": "not_available",
            "has_importances": False,
            "n_weak": 0,
            "explanation": (
                "Feature importances are not available for this model type. "
                "Try Random Forest or Gradient Boosting for built-in feature selection."
            ),
        }

    # Normalize to sum=1 (relative importance)
    total = float(importances.sum())
    normalized = importances / total if total > 0 else importances.copy()

    # Rank features (rank 1 = most important)
    ranked_indices = np.argsort(normalized)[::-1]
    ranks = np.empty(n_features, dtype=int)
    for rank_pos, idx in enumerate(ranked_indices):
        ranks[idx] = rank_pos + 1

    # Threshold at bottom threshold_percentile (e.g., bottom 20%)
    threshold = float(np.percentile(normalized, threshold_percentile))

    feature_data = []
    weak_features = []
    for i, col in enumerate(feature_cols):
        imp = float(normalized[i])
        is_weak = imp <= threshold
        feature_data.append(
            {
                "name": col,
                "importance": round(imp, 6),
                "rank": int(ranks[i]),
                "is_weak": is_weak,
            }
        )
        if is_weak:
            weak_features.append(col)

    # Sort by importance descending for display
    feature_data.sort(key=lambda x: x["importance"], reverse=True)

    n_weak = len(weak_features)
    method_label = (
        "SHAP-equivalent importance"
        if method == "feature_importances"
        else "coefficient magnitude"
    )
    if n_weak == 0:
        explanation = "All features are contributing meaningfully to the model. No features need to be removed."
    elif n_weak == 1:
        explanation = (
            f"1 feature has near-zero {method_label} (bottom {int(threshold_percentile)}%). "
            "Removing it may reduce noise without hurting accuracy."
        )
    else:
        explanation = (
            f"{n_weak} features have near-zero {method_label} (bottom {int(threshold_percentile)}%). "
            "Removing them may reduce noise and improve predictions on new data. "
            "Try retraining without these features and compare the metrics."
        )

    return {
        "feature_importances": feature_data,
        "weak_features": weak_features,
        "threshold": round(threshold, 6),
        "method": method,
        "has_importances": True,
        "n_weak": n_weak,
        "explanation": explanation,
    }


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
        "reason": (f"Highest {metric_name} among trained models ({score(best):.3f})."),
    }


# ---------------------------------------------------------------------------
# Goal-driven training
# ---------------------------------------------------------------------------

# Algorithms tried in priority order (fast → accurate)
_GOAL_TRIAL_ALGORITHMS: dict[str, list[str]] = {
    "regression": [
        "linear_regression",
        "random_forest_regressor",
        "gradient_boosting_regressor",
    ],
    "classification": [
        "logistic_regression",
        "random_forest_classifier",
        "gradient_boosting_classifier",
    ],
}

# Maximum rows to use for goal-driven trial runs (keeps latency acceptable)
_GOAL_MAX_ROWS = 5_000


def _goal_score(metrics: dict, goal_metric: str) -> float:
    """Return the numeric value for *goal_metric* from a metrics dict."""
    key_map: dict[str, str] = {
        "r2": "r2",
        "accuracy": "accuracy",
        "f1": "f1",
        "f1_score": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    key = key_map.get(
        goal_metric.lower().replace("-", "_").replace(" ", "_"), goal_metric
    )
    val = metrics.get(key)
    return float(val) if val is not None else -999.0


def run_goal_driven_training(
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    goal_metric: str,
    goal_target: float,
    model_dir: Path,
    base_id: str,
) -> dict:
    """Try algorithms in priority order until the target metric is achieved.

    Samples the dataset to *_GOAL_MAX_ROWS* rows for speed (trial mode).
    If the goal is not achieved after all algorithms, tries hyperparameter
    tuning on the best-performing algorithm.

    Returns:
        {
            goal_metric: str,
            goal_target: float,
            achieved: bool,
            winner_algorithm: str,
            winner_algorithm_name: str,
            winner_score: float,
            trials: list[{algorithm, algorithm_name, score, achieved_goal}],
            tried_tuning: bool,
            summary: str,
        }
    """
    all_algos = (
        REGRESSION_ALGORITHMS
        if problem_type == "regression"
        else CLASSIFICATION_ALGORITHMS
    )
    algorithms_to_try = [
        a for a in _GOAL_TRIAL_ALGORITHMS.get(problem_type, []) if a in all_algos
    ]

    # Sub-sample for speed
    if len(X) > _GOAL_MAX_ROWS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=_GOAL_MAX_ROWS, replace=False)
        X_trial, y_trial = X[idx], y[idx]
    else:
        X_trial, y_trial = X, y

    trials: list[dict] = []
    best_score = -999.0
    best_algo: str | None = None
    achieved = False

    for algo_key in algorithms_to_try:
        trial_id = f"{base_id}_goal_{algo_key}"
        try:
            result = train_single_model(
                X_trial, y_trial, algo_key, problem_type, model_dir, trial_id
            )
            score = _goal_score(result["metrics"], goal_metric)
            algo_name = all_algos[algo_key]["name"]
            hit = score >= goal_target
            trials.append(
                {
                    "algorithm": algo_key,
                    "algorithm_name": algo_name,
                    "score": round(score, 4),
                    "achieved_goal": hit,
                }
            )
            if score > best_score:
                best_score = score
                best_algo = algo_key
            if hit:
                achieved = True
                break  # Stop early — goal met
        except Exception:  # noqa: BLE001
            pass  # Skip failed algorithms; continue trying others

    # If still not achieved, try tuning the best algorithm
    tried_tuning = False
    if not achieved and best_algo and get_tuning_grid(best_algo):
        tried_tuning = True
        try:
            tune_result = tune_model(
                X_trial,
                y_trial,
                best_algo,
                problem_type,
                model_dir,
                f"{base_id}_goal_tuned",
                n_iter=10,
                cv=3,
            )
            if tune_result.get("metrics"):
                tuned_score = _goal_score(tune_result["metrics"], goal_metric)
                algo_name = all_algos[best_algo]["name"]
                hit = tuned_score >= goal_target
                trials.append(
                    {
                        "algorithm": best_algo,
                        "algorithm_name": f"{algo_name} (tuned)",
                        "score": round(tuned_score, 4),
                        "achieved_goal": hit,
                    }
                )
                if tuned_score > best_score:
                    best_score = tuned_score
                if hit:
                    achieved = True
        except Exception:  # noqa: BLE001
            pass

    # Pick the winner — highest score across all trials
    winner = max(trials, key=lambda t: t["score"]) if trials else None
    winner_algo = winner["algorithm"] if winner else (best_algo or "")
    winner_name = winner["algorithm_name"] if winner else ""
    winner_score = winner["score"] if winner else round(best_score, 4)

    # Plain-English summary
    _metric_labels: dict[str, str] = {
        "r2": "R²",
        "accuracy": "accuracy",
        "f1": "F1 score",
        "precision": "precision",
        "recall": "recall",
    }
    metric_label = _metric_labels.get(goal_metric.lower(), goal_metric.upper())
    if goal_metric.lower() == "r2":
        target_str = f"{goal_target:.2f}"
        score_str = f"{winner_score:.3f}"
    else:
        target_str = f"{goal_target * 100:.0f}%"
        score_str = f"{winner_score * 100:.0f}%"

    n_tried = len(trials)
    algo_word = "algorithm" if n_tried == 1 else "algorithms"
    if achieved:
        summary = (
            f"Goal achieved! {winner_name} reached {metric_label} = {score_str} "
            f"(target: {target_str}). Tried {n_tried} {algo_word}."
        )
    else:
        gap = goal_target - winner_score
        hint = (
            " Consider adding more data or engineering better features."
            if gap > 0.10
            else " The gap is small — try uploading more data or enabling hyperparameter tuning."
        )
        summary = (
            f"Best result: {winner_name} reached {metric_label} = {score_str} "
            f"(target was {target_str}). Tried {n_tried} {algo_word}.{hint}"
        )

    return {
        "goal_metric": goal_metric,
        "goal_target": goal_target,
        "achieved": achieved,
        "winner_algorithm": winner_algo,
        "winner_algorithm_name": winner_name,
        "winner_score": winner_score,
        "trials": trials,
        "tried_tuning": tried_tuning,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Learning Curve Analysis
# ---------------------------------------------------------------------------

# Friendly labels for primary metrics
_METRIC_LABELS: dict[str, str] = {
    "r2": "R²",
    "accuracy": "accuracy",
    "f1": "F1 score",
    "precision": "precision",
    "recall": "recall",
}


def compute_learning_curve(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm: str,
    problem_type: str,
    n_sizes: int = 5,
    cv_folds: int = 3,
) -> dict:
    """Compute train/validation scores at increasing dataset sizes.

    Trains *algorithm* at n_sizes fractions of the training data (evenly spaced
    from min_fraction to 1.0) using cross-validation, and returns the learning
    curve data.  Convergence is detected when the validation score improvement
    over the last two steps is below 1% of the full-data score.

    Args:
        X: Feature matrix (already preprocessed — no NaNs).
        y: Target vector.
        algorithm: Key from the algorithm registry (e.g. "random_forest_regressor").
        problem_type: "regression" or "classification".
        n_sizes: Number of training-size checkpoints (default 5).
        cv_folds: Number of cross-validation folds (default 3).

    Returns:
        dict with keys:
            sizes_pct       – list[int]: training sizes as % of full data
            train_scores    – list[float]: mean train score at each size
            val_scores      – list[float]: mean CV val score at each size
            converged       – bool: True when adding more data shows < 1% gain
            plateau_pct     – int | None: size (%) where convergence first detected
            best_val_score  – float: highest validation score achieved
            metric_label    – str: human-friendly metric name
            metric_key      – str: internal metric identifier
            n_total         – int: total number of training rows
            algorithm       – str: algorithm key used
            algorithm_name  – str: human-friendly algorithm name
            recommendation  – str: plain-English advice
            summary         – str: one-sentence overview
    """
    from sklearn.model_selection import learning_curve as _lc

    min_rows = max(cv_folds * 5, 20)
    if len(X) < min_rows:
        raise ValueError(
            f"Dataset has only {len(X)} rows — need at least {min_rows} for "
            "learning curve analysis."
        )

    # Build algorithm registry for the right problem type
    all_algos = (
        _build_regression_algorithms()
        if problem_type == "regression"
        else _build_classification_algorithms()
    )

    # Fall back to a sensible default if algorithm not found
    if algorithm not in all_algos:
        algorithm = (
            "linear_regression"
            if problem_type == "regression"
            else "logistic_regression"
        )

    algo_info = all_algos[algorithm]
    estimator = algo_info["class"](**algo_info["params"])

    # Determine scoring metric
    metric_key = "r2" if problem_type == "regression" else "accuracy"
    scoring = "r2" if metric_key == "r2" else "accuracy"

    # Fractional training sizes — sklearn learning_curve accepts floats in (0, 1]
    n_total = len(X)
    # Smallest valid fraction: enough rows for each fold
    min_fraction = max(cv_folds * 4 / n_total, 0.15)
    fractions = np.unique(np.linspace(min_fraction, 1.0, n_sizes).round(3))
    # Keep only fractions that are <= 1.0 and != duplicate
    fractions = fractions[fractions <= 1.0]

    # LabelEncode target for classification
    y_fit = y
    if problem_type == "classification":
        le = LabelEncoder()
        y_fit = pd.Series(le.fit_transform(y), index=y.index)

    try:
        train_sizes_out, train_scores_raw, val_scores_raw = _lc(
            estimator,
            X,
            y_fit,
            train_sizes=fractions,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
        )
    except Exception as exc:
        raise ValueError(f"Learning curve computation failed: {exc}") from exc

    train_means = [round(float(s.mean()), 4) for s in train_scores_raw]
    val_means = [round(float(s.mean()), 4) for s in val_scores_raw]
    sizes_pct = [round(int(s) / n_total * 100) for s in train_sizes_out]

    # Convergence detection: val score gain over last 2 steps < 1% of full-data val score
    best_val = max(val_means) if val_means else 0.0
    converged = False
    plateau_pct = None
    if len(val_means) >= 3:
        for i in range(2, len(val_means)):
            gain = val_means[i] - val_means[i - 1]
            if best_val != 0 and abs(gain) < 0.01 * abs(best_val):
                converged = True
                plateau_pct = sizes_pct[i - 1]
                break

    metric_label = _METRIC_LABELS.get(metric_key, metric_key.upper())

    if converged:
        recommendation = (
            f"Your model appears to have converged around {plateau_pct}% of your data. "
            f"Collecting more data is unlikely to significantly improve {metric_label}. "
            "Focus on better features or a more powerful algorithm instead."
        )
    else:
        gain = (val_means[-1] - val_means[0]) if len(val_means) > 1 else 0
        if gain > 0.02:
            recommendation = (
                f"The validation {metric_label} is still climbing — more training data "
                "would likely improve your model's accuracy. Try collecting 2-3× more rows."
            )
        else:
            recommendation = (
                "The validation score is relatively flat. More data may not help much — "
                "consider engineering better features or trying a different algorithm."
            )

    # One-sentence summary
    full_val = val_means[-1] if val_means else 0.0
    score_str = f"{full_val:.3f}" if metric_key == "r2" else f"{full_val * 100:.0f}%"
    converge_note = (
        "Model has converged — more data won't help much."
        if converged
        else "More data would likely improve accuracy."
    )
    summary = (
        f"With {n_total} training rows, {metric_label} = {score_str}. {converge_note}"
    )

    return {
        "sizes_pct": sizes_pct,
        "train_scores": train_means,
        "val_scores": val_means,
        "converged": converged,
        "plateau_pct": plateau_pct,
        "best_val_score": round(best_val, 4),
        "metric_label": metric_label,
        "metric_key": metric_key,
        "n_total": n_total,
        "algorithm": algorithm,
        "algorithm_name": algo_info["name"],
        "recommendation": recommendation,
        "summary": summary,
    }
