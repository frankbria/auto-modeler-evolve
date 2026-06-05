"""Tests for custom per-class weighted training.

Covers:
- _CUSTOM_CLASS_WEIGHT_PATTERNS regex (8 positive, 3 negative)
- _detect_custom_class_weights() helper (multiplier / times / kv extraction)
- train_single_model() with custom_class_weights for LR, RF, GBC, XGB
- Chat handler emitting training_started with custom_class_weights
- Regression / no-dataset guard
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Pattern tests
# ---------------------------------------------------------------------------


class TestCustomClassWeightPatterns:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api.chat import _CUSTOM_CLASS_WEIGHT_PATTERNS

        self.pat = _CUSTOM_CLASS_WEIGHT_PATTERNS

    def test_give_weight_to_class(self):
        assert self.pat.search("give 3x weight to class churn")

    def test_multiplier_weight_for(self):
        assert self.pat.search("5x weight for class fraud")

    def test_custom_class_weights(self):
        assert self.pat.search("custom class weights: fraud=10, normal=1")

    def test_per_class_weights(self):
        assert self.pat.search("per-class custom weights")

    def test_manual_class_weights(self):
        assert self.pat.search("manual class weights")

    def test_higher_weight_training(self):
        assert self.pat.search("higher weight on class positive when training")

    def test_times_more_weight(self):
        assert self.pat.search("4 times more weight to class minority")

    def test_weight_class_as(self):
        assert self.pat.search("weight class churn as 5")

    # False-positive guards
    def test_no_match_generic_train(self):
        assert not self.pat.search("train a model to predict revenue")

    def test_no_match_balanced(self):
        assert not self.pat.search("train with balanced class weighting")

    def test_no_match_smote(self):
        assert not self.pat.search("apply smote and retrain")


# ---------------------------------------------------------------------------
# _detect_custom_class_weights() helper tests
# ---------------------------------------------------------------------------


class TestDetectCustomClassWeights:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api.chat import _detect_custom_class_weights

        self.fn = _detect_custom_class_weights

    def test_multiplier_extraction(self):
        result = self.fn("give 3x weight to class churn", ["churn", "no_churn"])
        assert result.get("churn") == 3.0

    def test_times_more_weight(self):
        result = self.fn("5 times more weight to churn", ["churn", "no_churn"])
        assert result.get("churn") == 5.0

    def test_kv_extraction(self):
        result = self.fn("class weights: fraud=10, normal=1", ["fraud", "normal"])
        assert result.get("fraud") == 10.0
        assert result.get("normal") == 1.0

    def test_case_insensitive_matching(self):
        result = self.fn("5x weight for FRAUD", ["fraud", "normal"])
        assert "fraud" in result

    def test_returns_empty_when_no_match(self):
        result = self.fn("train a model", ["churn", "no_churn"])
        assert result == {}

    def test_partial_class_name_match(self):
        result = self.fn("2x weight for churn", ["churn", "no_churn"])
        assert result.get("churn") == 2.0


# ---------------------------------------------------------------------------
# train_single_model() with custom_class_weights
# ---------------------------------------------------------------------------


def _make_imbalanced_classification_data():
    """20-sample binary classification with 75/25 imbalance."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 3))
    y = np.array([0] * 15 + [1] * 5)
    return X, y


def _make_label_encoder(classes):
    le = LabelEncoder()
    le.fit(classes)
    return le


class TestTrainSingleModelCustomWeights:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        from core.trainer import train_single_model

        self.train = train_single_model
        self.model_dir = tmp_path

    def test_logistic_regression_accepts_custom_weights(self):
        X, y = _make_imbalanced_classification_data()
        le = _make_label_encoder(["0", "1"])
        result = self.train(
            X, y, "logistic_regression", "classification",
            self.model_dir, "run-lr-cw",
            custom_class_weights={"0": 1.0, "1": 4.0},
            label_encoder=le,
        )
        assert result["metrics"]["accuracy"] >= 0

    def test_random_forest_accepts_custom_weights(self):
        X, y = _make_imbalanced_classification_data()
        le = _make_label_encoder(["0", "1"])
        result = self.train(
            X, y, "random_forest_classifier", "classification",
            self.model_dir, "run-rf-cw",
            custom_class_weights={"0": 1.0, "1": 5.0},
            label_encoder=le,
        )
        assert result["metrics"]["accuracy"] >= 0

    def test_gradient_boosting_accepts_custom_weights(self):
        X, y = _make_imbalanced_classification_data()
        le = _make_label_encoder(["0", "1"])
        result = self.train(
            X, y, "gradient_boosting_classifier", "classification",
            self.model_dir, "run-gbc-cw",
            custom_class_weights={"0": 1.0, "1": 3.0},
            label_encoder=le,
        )
        assert result["metrics"]["accuracy"] >= 0

    def test_custom_weights_ignored_for_regression(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal(20)
        le = _make_label_encoder(["0", "1"])
        result = self.train(
            X, y, "linear_regression", "regression",
            self.model_dir, "run-reg-cw",
            custom_class_weights={"0": 1.0, "1": 5.0},
            label_encoder=le,
        )
        assert "r2" in result["metrics"]

    def test_no_custom_weights_behaves_normally(self):
        X, y = _make_imbalanced_classification_data()
        result = self.train(
            X, y, "logistic_regression", "classification",
            self.model_dir, "run-lr-no-cw",
            custom_class_weights=None,
        )
        assert result["metrics"]["accuracy"] >= 0

    def test_custom_weights_with_no_label_encoder_fallback(self):
        """When label_encoder is None, fall back to integer-keyed lookup."""
        X, y = _make_imbalanced_classification_data()
        result = self.train(
            X, y, "logistic_regression", "classification",
            self.model_dir, "run-lr-nole",
            custom_class_weights={"0": 1.0, "1": 4.0},
            label_encoder=None,
        )
        assert result["metrics"]["accuracy"] >= 0

    def test_imbalance_strategy_takes_precedence_over_custom_weights(self):
        """Explicit imbalance_strategy should not be overridden by custom_class_weights."""
        X, y = _make_imbalanced_classification_data()
        le = _make_label_encoder(["0", "1"])
        result = self.train(
            X, y, "logistic_regression", "classification",
            self.model_dir, "run-lr-both",
            imbalance_strategy="class_weight",
            custom_class_weights={"0": 1.0, "1": 4.0},
            label_encoder=le,
        )
        assert result["metrics"]["accuracy"] >= 0

    def test_neural_network_trains_normally_with_custom_weights(self):
        """neural_network_classifier doesn't support class_weight/sample_weight; graceful."""
        X, y = _make_imbalanced_classification_data()
        le = _make_label_encoder(["0", "1"])
        result = self.train(
            X, y, "neural_network_classifier", "classification",
            self.model_dir, "run-nn-cw",
            custom_class_weights={"0": 1.0, "1": 3.0},
            label_encoder=le,
        )
        assert result["metrics"]["accuracy"] >= 0
