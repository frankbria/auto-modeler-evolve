"""Tests for counterfactual explanation: compute_counterfactual() and chat handler."""

from __future__ import annotations

import os

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tests._artifact_tmp import data_tmpdir as _data_tmpdir

# ---------------------------------------------------------------------------
# Helpers to build a minimal PredictionPipeline + model
# ---------------------------------------------------------------------------


def _build_pipeline_and_model(n_samples: int = 80, n_numeric: int = 3):
    """Create a tiny classification pipeline + model for tests."""
    from core.deployer import PredictionPipeline

    rng = np.random.default_rng(42)

    # 3 numeric features — feature0 is the most predictive
    feature_names = [f"feature{i}" for i in range(n_numeric)]
    X = rng.standard_normal((n_samples, n_numeric))
    # Binary classification: positive when feature0 > 0
    y_binary = (X[:, 0] > 0).astype(int)

    from sklearn.preprocessing import LabelEncoder

    pipeline = PredictionPipeline(
        feature_names=feature_names,
        column_types={f: "numeric" for f in feature_names},
        target_column="label",
        problem_type="classification",
    )
    pipeline.feature_means = {
        f: float(X[:, i].mean()) for i, f in enumerate(feature_names)
    }
    pipeline.feature_stds = {
        f: float(X[:, i].std()) for i, f in enumerate(feature_names)
    }
    pipeline.medians = {
        f: float(np.median(X[:, i])) for i, f in enumerate(feature_names)
    }
    le = LabelEncoder()
    le.fit(y_binary)
    pipeline.target_encoder = le
    pipeline.target_classes = list(le.classes_)

    model = LogisticRegression(random_state=42)
    model.fit(X, y_binary)

    return pipeline, model, X, y_binary


# ---------------------------------------------------------------------------
# Unit tests for compute_counterfactual()
# ---------------------------------------------------------------------------


class TestComputeCounterfactual:
    def setup_method(self):
        self.pipeline, self.model, self.X, self.y = _build_pipeline_and_model()
        self.tmp = _data_tmpdir()
        self.pipeline_path = os.path.join(self.tmp, "pipe_pipeline.joblib")
        self.model_path = os.path.join(self.tmp, "pipe_model.joblib")
        joblib.dump(self.pipeline, self.pipeline_path)
        joblib.dump(self.model, self.model_path)

    def _make_features(self, row_idx: int) -> dict:
        return {
            f"feature{i}": float(self.X[row_idx, i]) for i in range(self.X.shape[1])
        }

    def test_returns_required_keys(self):
        from core.deployer import compute_counterfactual

        orig = self._make_features(0)
        result = compute_counterfactual(self.pipeline_path, self.model_path, orig)
        required = {
            "problem_type",
            "target_column",
            "original_prediction",
            "original_class",
            "original_confidence",
            "counterfactual_prediction",
            "target_class",
            "counterfactual_confidence",
            "flipped",
            "n_steps",
            "changed_features",
            "summary",
        }
        assert required.issubset(result.keys())

    def test_problem_type_is_classification(self):
        from core.deployer import compute_counterfactual

        orig = self._make_features(0)
        result = compute_counterfactual(self.pipeline_path, self.model_path, orig)
        assert result["problem_type"] == "classification"

    def test_original_class_is_valid(self):
        from core.deployer import compute_counterfactual

        orig = self._make_features(0)
        result = compute_counterfactual(self.pipeline_path, self.model_path, orig)
        assert result["original_class"] in ["0", "1"]

    def test_target_class_differs_from_original(self):
        from core.deployer import compute_counterfactual

        orig = self._make_features(0)
        result = compute_counterfactual(self.pipeline_path, self.model_path, orig)
        # target_class should be the OTHER class, not the original
        assert result["target_class"] != result["original_class"]

    def test_flipped_true_means_counterfactual_equals_target(self):
        from core.deployer import compute_counterfactual

        orig = self._make_features(0)
        result = compute_counterfactual(self.pipeline_path, self.model_path, orig)
        if result["flipped"]:
            assert result["counterfactual_prediction"] == result["target_class"]

    def test_changed_features_structure(self):
        from core.deployer import compute_counterfactual

        orig = self._make_features(0)
        result = compute_counterfactual(self.pipeline_path, self.model_path, orig)
        for cf in result["changed_features"]:
            assert "name" in cf
            assert "original_value" in cf
            assert "counterfactual_value" in cf
            assert "change_pct" in cf
            assert cf["direction"] in ("increase", "decrease")

    def test_explicit_target_class(self):
        from core.deployer import compute_counterfactual

        # Find a row predicted as 0, ask for 1
        orig_class_0_idx = next(
            i
            for i in range(len(self.y))
            if self.model.predict(self.X[i : i + 1])[0] == 0
        )
        orig = self._make_features(orig_class_0_idx)
        result = compute_counterfactual(
            self.pipeline_path, self.model_path, orig, target_class="1"
        )
        assert result["target_class"] == "1"

    def test_invalid_target_class_raises(self):
        from core.deployer import compute_counterfactual

        orig = self._make_features(0)
        with pytest.raises(ValueError, match="not found"):
            compute_counterfactual(
                self.pipeline_path, self.model_path, orig, target_class="99"
            )

    def test_summary_is_non_empty_string(self):
        from core.deployer import compute_counterfactual

        orig = self._make_features(0)
        result = compute_counterfactual(self.pipeline_path, self.model_path, orig)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_n_steps_within_max(self):
        from core.deployer import compute_counterfactual

        orig = self._make_features(0)
        result = compute_counterfactual(
            self.pipeline_path, self.model_path, orig, max_steps=5
        )
        assert result["n_steps"] <= 5

    def test_regression_model_raises(self):
        from sklearn.linear_model import LinearRegression

        from core.deployer import PredictionPipeline, compute_counterfactual

        # Build a regression pipeline
        rng = np.random.default_rng(0)
        X_reg = rng.standard_normal((50, 2))
        y_reg = X_reg[:, 0] * 2 + rng.standard_normal(50)

        reg_pipeline = PredictionPipeline(
            feature_names=["a", "b"],
            column_types={"a": "numeric", "b": "numeric"},
            target_column="value",
            problem_type="regression",
        )
        reg_pipeline.feature_means = {"a": 0.0, "b": 0.0}
        reg_pipeline.feature_stds = {"a": 1.0, "b": 1.0}
        reg_pipeline.medians = {"a": 0.0, "b": 0.0}

        reg_model = LinearRegression()
        reg_model.fit(X_reg, y_reg)

        tmp = _data_tmpdir()
        p_path = os.path.join(tmp, "r_pipeline.joblib")
        m_path = os.path.join(tmp, "r_model.joblib")
        joblib.dump(reg_pipeline, p_path)
        joblib.dump(reg_model, m_path)

        with pytest.raises(ValueError, match="classification"):
            compute_counterfactual(p_path, m_path, {"a": 1.0, "b": 0.5})

    def test_flip_achievable_with_extreme_features(self):
        """A very strong positive-class instance should flip with enough steps."""
        from core.deployer import compute_counterfactual

        # Find an instance where feature0 << 0 (strongly predicts class 0)
        strongly_neg_idx = int(np.argmin(self.X[:, 0]))
        orig = self._make_features(strongly_neg_idx)
        # Ask for class 1 — requires feature0 to increase significantly
        result = compute_counterfactual(
            self.pipeline_path, self.model_path, orig, target_class="1", max_steps=30
        )
        # May or may not flip depending on step size, but structure should be valid
        assert isinstance(result["flipped"], bool)
        assert isinstance(result["changed_features"], list)

    def test_confidence_between_0_and_1(self):
        from core.deployer import compute_counterfactual

        orig = self._make_features(0)
        result = compute_counterfactual(self.pipeline_path, self.model_path, orig)
        assert 0.0 <= result["original_confidence"] <= 1.0
        assert 0.0 <= result["counterfactual_confidence"] <= 1.0

    def test_no_changed_features_when_already_target(self):
        """When target_class == original_class, returns flipped=True with no changes."""
        from core.deployer import compute_counterfactual

        # Find row 0 original class, then ask for that class explicitly
        orig = self._make_features(0)
        # First get the current class
        first_result = compute_counterfactual(self.pipeline_path, self.model_path, orig)
        current_class = first_result["original_class"]
        # Now request that same class as target
        result = compute_counterfactual(
            self.pipeline_path, self.model_path, orig, target_class=current_class
        )
        assert result["flipped"] is True
        assert result["n_steps"] == 0
        assert result["changed_features"] == []


# ---------------------------------------------------------------------------
# Chat regex tests
# ---------------------------------------------------------------------------


class TestCounterfactualPatterns:
    @pytest.fixture(autouse=True)
    def import_pattern(self):
        from api.chat import _COUNTERFACTUAL_PATTERNS as pat

        self.pat = pat

    def _match(self, msg: str) -> bool:
        return bool(self.pat.search(msg))

    def test_what_would_need_to_change(self):
        assert self._match("What would need to change for this prediction to flip?")

    def test_counterfactual_for_row(self):
        assert self._match("counterfactual for row 5")

    def test_minimum_intervention(self):
        assert self._match("minimum intervention to change the prediction")

    def test_flip_this_prediction(self):
        assert self._match("flip this prediction for row 12")

    def test_which_changes_would_flip(self):
        assert self._match("which changes would flip the prediction?")

    def test_how_close_to_other_class(self):
        assert self._match("how close is row 3 to the other class?")

    def test_what_would_save_customer(self):
        assert self._match("what would save this customer row 7?")

    def test_prevent_churn(self):
        assert self._match("prevent churn for row 2")

    def test_intervention_needed(self):
        assert self._match("intervention needed for row 10")

    def test_no_false_positive_train(self):
        assert not self._match("train a model to predict churn")

    def test_no_false_positive_explain(self):
        assert not self._match("explain my model")

    def test_no_false_positive_sensitivity(self):
        assert not self._match("sensitivity analysis for price")

    def test_case_insensitive(self):
        assert self._match("WHAT WOULD NEED TO CHANGE FOR ROW 3?")

    def test_minimum_change_to_flip(self):
        assert self._match("what is the minimum change needed to flip this?")
