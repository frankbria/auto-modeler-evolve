"""Tests for core/deployer.py — Phase 6 deployment and prediction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.deployer import (
    get_feature_schema,
    load_artifact,
    predict_batch,
    predict_single,
)
from core.trainer import train_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_df():
    rng = np.random.default_rng(42)
    n = 60
    x1 = rng.normal(50, 15, n)
    x2 = rng.choice(["A", "B", "C"], n)
    y = 2 * x1 + rng.normal(0, 5, n)
    return pd.DataFrame({"feature1": x1, "category": x2, "target": y})


@pytest.fixture
def classification_df():
    rng = np.random.default_rng(7)
    n = 60
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    labels = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"feature1": x1, "feature2": x2, "label": labels})


@pytest.fixture
def trained_regression_model(tmp_path, regression_df):
    result = train_model(
        df=regression_df,
        target_col="target",
        problem_type="regression",
        algorithm="linear_regression",
        hyperparameters={},
        model_dir=tmp_path,
    )
    return result["model_path"]


@pytest.fixture
def trained_classification_model(tmp_path, classification_df):
    result = train_model(
        df=classification_df,
        target_col="label",
        problem_type="classification",
        algorithm="logistic_regression",
        hyperparameters={},
        model_dir=tmp_path,
    )
    return result["model_path"]


# ---------------------------------------------------------------------------
# load_artifact
# ---------------------------------------------------------------------------

class TestLoadArtifact:
    def test_loads_valid_artifact(self, trained_regression_model):
        artifact = load_artifact(trained_regression_model)
        assert "pipeline" in artifact
        assert "feature_names" in artifact
        assert "target_col" in artifact
        assert "problem_type" in artifact
        assert "algorithm" in artifact

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifact(str(tmp_path / "nonexistent.joblib"))


# ---------------------------------------------------------------------------
# get_feature_schema
# ---------------------------------------------------------------------------

class TestGetFeatureSchema:
    def test_returns_list_of_descriptors(self, trained_regression_model):
        schema = get_feature_schema(trained_regression_model)
        assert isinstance(schema, list)
        assert len(schema) > 0

    def test_schema_has_required_keys(self, trained_regression_model):
        schema = get_feature_schema(trained_regression_model)
        for entry in schema:
            assert "name" in entry
            assert "dtype" in entry
            assert entry["dtype"] in ("numeric", "categorical")

    def test_numeric_and_categorical_detected(self, trained_regression_model):
        schema = get_feature_schema(trained_regression_model)
        dtypes = {e["name"]: e["dtype"] for e in schema}
        assert dtypes.get("feature1") == "numeric"
        assert dtypes.get("category") == "categorical"


# ---------------------------------------------------------------------------
# predict_single
# ---------------------------------------------------------------------------

class TestPredictSingle:
    def test_regression_returns_numeric_prediction(self, trained_regression_model):
        result = predict_single(trained_regression_model, {"feature1": 50, "category": "A"})
        assert isinstance(result["prediction"], float)
        assert result["probability"] is None
        assert "Predicted value" in result["interpretation"]

    def test_classification_returns_label(self, trained_classification_model):
        result = predict_single(trained_classification_model, {"feature1": 1.5, "feature2": 1.5})
        assert result["prediction"] in ("0", "1", 0, 1)
        assert result["probability"] is not None
        assert 0.0 <= result["probability"] <= 1.0

    def test_missing_features_handled_gracefully(self, trained_regression_model):
        # Missing features should not crash — pipeline handles NaN via imputer
        result = predict_single(trained_regression_model, {})
        assert "prediction" in result

    def test_result_has_target_column(self, trained_regression_model):
        result = predict_single(trained_regression_model, {"feature1": 50})
        assert result["target_column"] == "target"

    def test_result_has_algorithm(self, trained_regression_model):
        result = predict_single(trained_regression_model, {"feature1": 50})
        assert result["algorithm"] == "linear_regression"


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_batch_regression(self, trained_regression_model, regression_df):
        input_df = regression_df.drop(columns=["target"])
        result = predict_batch(trained_regression_model, input_df)
        assert len(result["predictions"]) == len(input_df)
        assert result["probabilities"] is None  # regression has no proba
        assert "predicted_target" in result["output_df"].columns

    def test_batch_classification(self, trained_classification_model, classification_df):
        input_df = classification_df.drop(columns=["label"])
        result = predict_batch(trained_classification_model, input_df)
        assert len(result["predictions"]) == len(input_df)
        assert result["probabilities"] is not None
        assert len(result["probabilities"]) == len(input_df)
        assert "confidence" in result["output_df"].columns

    def test_output_df_preserves_original_columns(self, trained_regression_model, regression_df):
        input_df = regression_df.drop(columns=["target"])
        result = predict_batch(trained_regression_model, input_df)
        for col in input_df.columns:
            assert col in result["output_df"].columns

    def test_row_count_matches(self, trained_regression_model, regression_df):
        input_df = regression_df.drop(columns=["target"])
        result = predict_batch(trained_regression_model, input_df)
        assert result["row_count"] == len(input_df)
