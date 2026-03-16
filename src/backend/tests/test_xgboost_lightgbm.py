"""Tests for XGBoost and LightGBM algorithm integration in trainer.py.

These tests verify:
- XGBoost and LightGBM appear in algorithm registries
- Both train correctly for regression and classification
- feature_importances_ is accessible (needed by explainer.py)
- recommend_models returns the new algorithms with sensible explanations
- train_single_model handles the new algorithm keys without errors
"""

from __future__ import annotations

import numpy as np
import pytest

from core.trainer import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    _LIGHTGBM_AVAILABLE,
    _XGBOOST_AVAILABLE,
    recommend_models,
    train_single_model,
    prepare_features,
)
import pandas as pd
from pathlib import Path
import tempfile


# ---------------------------------------------------------------------------
# Registry presence
# ---------------------------------------------------------------------------


def test_xgboost_in_regression_registry():
    if not _XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    assert "xgboost_regressor" in REGRESSION_ALGORITHMS
    info = REGRESSION_ALGORITHMS["xgboost_regressor"]
    assert info["name"] == "XGBoost"
    assert "class" in info
    assert "plain_english" in info


def test_xgboost_in_classification_registry():
    if not _XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    assert "xgboost_classifier" in CLASSIFICATION_ALGORITHMS
    info = CLASSIFICATION_ALGORITHMS["xgboost_classifier"]
    assert info["name"] == "XGBoost"


def test_lightgbm_in_regression_registry():
    if not _LIGHTGBM_AVAILABLE:
        pytest.skip("lightgbm not installed")
    assert "lightgbm_regressor" in REGRESSION_ALGORITHMS
    info = REGRESSION_ALGORITHMS["lightgbm_regressor"]
    assert info["name"] == "LightGBM"


def test_lightgbm_in_classification_registry():
    if not _LIGHTGBM_AVAILABLE:
        pytest.skip("lightgbm not installed")
    assert "lightgbm_classifier" in CLASSIFICATION_ALGORITHMS
    info = CLASSIFICATION_ALGORITHMS["lightgbm_classifier"]
    assert info["name"] == "LightGBM"


# ---------------------------------------------------------------------------
# recommend_models includes new algorithms
# ---------------------------------------------------------------------------


def test_recommend_models_regression_includes_xgboost():
    if not _XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    recs = recommend_models("regression", n_rows=500, n_features=5)
    keys = [r["algorithm"] for r in recs]
    assert "xgboost_regressor" in keys


def test_recommend_models_regression_includes_lightgbm():
    if not _LIGHTGBM_AVAILABLE:
        pytest.skip("lightgbm not installed")
    recs = recommend_models("regression", n_rows=500, n_features=5)
    keys = [r["algorithm"] for r in recs]
    assert "lightgbm_regressor" in keys


def test_recommend_models_classification_includes_xgboost():
    if not _XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    recs = recommend_models("classification", n_rows=300, n_features=4)
    keys = [r["algorithm"] for r in recs]
    assert "xgboost_classifier" in keys


def test_recommend_models_large_dataset_xgboost_explanation():
    if not _XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    recs = recommend_models("regression", n_rows=5000, n_features=10)
    xgb_rec = next(r for r in recs if r["algorithm"] == "xgboost_regressor")
    # Should mention the row count in the recommendation
    assert "5000" in xgb_rec["recommended_because"]


def test_recommend_models_small_dataset_xgboost_explanation():
    if not _XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    recs = recommend_models("regression", n_rows=50, n_features=3)
    xgb_rec = next(r for r in recs if r["algorithm"] == "xgboost_regressor")
    # Should warn about small dataset
    assert (
        "50" in xgb_rec["recommended_because"]
        or "small" in xgb_rec["recommended_because"].lower()
    )


# ---------------------------------------------------------------------------
# Actual training
# ---------------------------------------------------------------------------


def _make_regression_data(n=200):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, 4))
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] * 0.5 + rng.standard_normal(n) * 0.1
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y
    return df


def _make_classification_data(n=200):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y
    return df


def test_train_xgboost_regressor():
    if not _XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    df = _make_regression_data()
    X, y, _ = prepare_features(df, ["a", "b", "c", "d"], "target", "regression")
    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_single_model(
            X, y, "xgboost_regressor", "regression", Path(tmpdir), "run_xgb_reg"
        )
        assert "metrics" in result
        assert result["metrics"]["r2"] > 0.8  # should fit well on synthetic linear data
        assert result["training_duration_ms"] >= 0
        assert Path(result["model_path"]).exists()


def test_train_xgboost_classifier():
    if not _XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    df = _make_classification_data()
    X, y, _ = prepare_features(df, ["a", "b", "c", "d"], "target", "classification")
    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_single_model(
            X, y, "xgboost_classifier", "classification", Path(tmpdir), "run_xgb_cls"
        )
        assert "metrics" in result
        assert result["metrics"]["accuracy"] > 0.8
        assert Path(result["model_path"]).exists()


def test_train_lightgbm_regressor():
    if not _LIGHTGBM_AVAILABLE:
        pytest.skip("lightgbm not installed")
    df = _make_regression_data()
    X, y, _ = prepare_features(df, ["a", "b", "c", "d"], "target", "regression")
    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_single_model(
            X, y, "lightgbm_regressor", "regression", Path(tmpdir), "run_lgbm_reg"
        )
        assert "metrics" in result
        assert result["metrics"]["r2"] > 0.8
        assert Path(result["model_path"]).exists()


def test_train_lightgbm_classifier():
    if not _LIGHTGBM_AVAILABLE:
        pytest.skip("lightgbm not installed")
    df = _make_classification_data()
    X, y, _ = prepare_features(df, ["a", "b", "c", "d"], "target", "classification")
    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_single_model(
            X, y, "lightgbm_classifier", "classification", Path(tmpdir), "run_lgbm_cls"
        )
        assert "metrics" in result
        assert result["metrics"]["accuracy"] > 0.8
        assert Path(result["model_path"]).exists()


# ---------------------------------------------------------------------------
# Feature importances accessible (needed by explainer.py)
# ---------------------------------------------------------------------------


def test_xgboost_feature_importances_accessible():
    """XGBoost models expose feature_importances_ — compatible with explainer.py."""
    if not _XGBOOST_AVAILABLE:
        pytest.skip("xgboost not installed")
    from xgboost import XGBRegressor

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = X[:, 0] + rng.standard_normal(100) * 0.1
    model = XGBRegressor(n_estimators=10, random_state=0, verbosity=0)
    model.fit(X, y)
    assert hasattr(model, "feature_importances_")
    fi = model.feature_importances_
    assert len(fi) == 3
    assert abs(sum(fi) - 1.0) < 0.01  # should sum to ~1


def test_lightgbm_feature_importances_accessible():
    """LightGBM models expose feature_importances_ — compatible with explainer.py."""
    if not _LIGHTGBM_AVAILABLE:
        pytest.skip("lightgbm not installed")
    from lightgbm import LGBMRegressor

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    y = X[:, 0] + rng.standard_normal(100) * 0.1
    model = LGBMRegressor(n_estimators=10, random_state=0, verbose=-1)
    model.fit(X, y)
    assert hasattr(model, "feature_importances_")
    fi = model.feature_importances_
    assert len(fi) == 3


# ---------------------------------------------------------------------------
# Invalid algorithm key still raises ValueError
# ---------------------------------------------------------------------------


def test_unknown_algorithm_raises():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    y = rng.standard_normal(50)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            train_single_model(
                X, y, "unknown_algo_xyz", "regression", Path(tmpdir), "run_bad"
            )
