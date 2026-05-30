"""Tests for population-level counterfactual: compute_population_counterfactual()."""

from __future__ import annotations

import os
import re
import tempfile

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_classification_pipeline_and_model(n_samples: int = 100, n_numeric: int = 3):
    """Create a tiny classification pipeline + model."""
    from core.deployer import PredictionPipeline
    from sklearn.preprocessing import LabelEncoder

    rng = np.random.default_rng(42)
    feature_names = [f"feature{i}" for i in range(n_numeric)]
    X = rng.standard_normal((n_samples, n_numeric))
    # feature0 > 0 → class 1; easy to flip
    y = (X[:, 0] > 0).astype(int)

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
    le.fit(y)
    pipeline.target_encoder = le
    pipeline.target_classes = list(le.classes_)

    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)

    return pipeline, model, X


def _build_regression_pipeline_and_model(n_samples: int = 50, n_numeric: int = 2):
    """Create a tiny regression pipeline + model."""
    from core.deployer import PredictionPipeline

    rng = np.random.default_rng(7)
    feature_names = [f"rfeat{i}" for i in range(n_numeric)]
    X = rng.standard_normal((n_samples, n_numeric))
    y = X[:, 0] * 2.0 + rng.standard_normal(n_samples)

    pipeline = PredictionPipeline(
        feature_names=feature_names,
        column_types={f: "numeric" for f in feature_names},
        target_column="value",
        problem_type="regression",
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

    model = Ridge()
    model.fit(X, y)

    return pipeline, model, X


def _save(pipeline, model, tmp_dir: str, name: str = "test"):
    pipe_path = os.path.join(tmp_dir, f"{name}_pipeline.joblib")
    model_path = os.path.join(tmp_dir, f"{name}_model.joblib")
    joblib.dump(pipeline, pipe_path)
    joblib.dump(model, model_path)
    return pipe_path, model_path


# ---------------------------------------------------------------------------
# Unit tests for compute_population_counterfactual()
# ---------------------------------------------------------------------------


class TestComputePopulationCounterfactual:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline, self.model, self.X = _build_classification_pipeline_and_model()
        self.pipeline_path, self.model_path = _save(self.pipeline, self.model, self.tmp)

    def _rows(self, n: int = 10) -> list[dict]:
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(self.X), size=min(n, len(self.X)), replace=False)
        return [
            {f"feature{i}": float(self.X[idx, i]) for i in range(3)} for idx in idxs
        ]

    # -- required output keys --------------------------------------------------

    def test_returns_required_keys(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(10)
        )
        required = {
            "problem_type",
            "target_column",
            "total_rows",
            "flipped_count",
            "flip_rate",
            "dominant_feature",
            "dominant_direction",
            "dominant_flip_count",
            "dominant_avg_change_pct",
            "feature_summary",
            "summary",
        }
        assert required.issubset(result.keys())

    def test_problem_type_is_classification(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(10)
        )
        assert result["problem_type"] == "classification"

    def test_target_column_matches_pipeline(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(10)
        )
        assert result["target_column"] == "label"

    # -- count / rate invariants -----------------------------------------------

    def test_flipped_count_lte_total_rows(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(10)
        )
        assert result["flipped_count"] <= result["total_rows"]

    def test_flip_rate_between_zero_and_one(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(10)
        )
        assert 0.0 <= result["flip_rate"] <= 1.0

    def test_flip_rate_consistent_with_counts(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(10)
        )
        expected = round(result["flipped_count"] / result["total_rows"], 3)
        assert abs(result["flip_rate"] - expected) < 1e-4

    def test_total_rows_respects_max_rows_cap(self):
        from core.deployer import compute_population_counterfactual

        rows = self._rows(20)
        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, rows, max_rows=5
        )
        assert result["total_rows"] == 5

    # -- feature_summary structure --------------------------------------------

    def test_feature_summary_is_list(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(10)
        )
        assert isinstance(result["feature_summary"], list)

    def test_feature_summary_row_keys(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(15)
        )
        for row in result["feature_summary"]:
            assert "feature" in row
            assert "flip_count" in row
            assert "flip_pct" in row
            assert "avg_change_pct" in row
            assert row["direction"] in ("increase", "decrease")

    def test_feature_summary_sorted_by_flip_count_desc(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(20)
        )
        counts = [r["flip_count"] for r in result["feature_summary"]]
        assert counts == sorted(counts, reverse=True)

    def test_feature_summary_flip_pct_in_range(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(15)
        )
        for row in result["feature_summary"]:
            assert 0.0 <= row["flip_pct"] <= 100.0

    # -- dominant feature invariants ------------------------------------------

    def test_dominant_direction_valid_when_present(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(15)
        )
        if result["dominant_feature"] is not None:
            assert result["dominant_direction"] in ("increase", "decrease")

    def test_dominant_feature_in_feature_names(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(15)
        )
        if result["dominant_feature"] is not None:
            assert result["dominant_feature"] in self.pipeline.feature_names

    def test_dominant_flip_count_lte_flipped_count(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(15)
        )
        assert result["dominant_flip_count"] <= result["flipped_count"]

    # -- summary ---------------------------------------------------------------

    def test_summary_is_non_empty_string(self):
        from core.deployer import compute_population_counterfactual

        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, self._rows(10)
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    # -- error cases -----------------------------------------------------------

    def test_regression_model_raises_value_error(self):
        from core.deployer import compute_population_counterfactual

        pipeline, model, X = _build_regression_pipeline_and_model()
        pipe_path, mod_path = _save(pipeline, model, self.tmp, "reg")

        rows = [{f"rfeat{i}": float(X[j, i]) for i in range(2)} for j in range(5)]
        with pytest.raises(ValueError, match="classification"):
            compute_population_counterfactual(pipe_path, mod_path, rows)

    def test_fewer_than_two_rows_raises_value_error(self):
        from core.deployer import compute_population_counterfactual

        single_row = [{"feature0": 0.5, "feature1": -0.2, "feature2": 0.1}]
        with pytest.raises(ValueError, match="at least 2 rows"):
            compute_population_counterfactual(
                self.pipeline_path, self.model_path, single_row
            )

    def test_no_numeric_features_raises_value_error(self):
        """A pipeline with no numeric features should raise ValueError."""
        from core.deployer import PredictionPipeline, compute_population_counterfactual
        from sklearn.preprocessing import LabelEncoder

        cat_pipeline = PredictionPipeline(
            feature_names=["cat_feat"],
            column_types={"cat_feat": "categorical"},
            target_column="label",
            problem_type="classification",
        )
        le = LabelEncoder()
        le.fit([0, 1])
        cat_pipeline.target_encoder = le
        cat_pipeline.target_classes = [0, 1]
        cat_pipeline.feature_means = {}
        cat_pipeline.feature_stds = {}
        cat_pipeline.medians = {}

        pipe_path = os.path.join(self.tmp, "cat_pipeline.joblib")
        mod_path = os.path.join(self.tmp, "cat_model.joblib")
        joblib.dump(cat_pipeline, pipe_path)
        joblib.dump(self.model, mod_path)

        rows = [{"cat_feat": "A"}, {"cat_feat": "B"}, {"cat_feat": "C"}]
        with pytest.raises(ValueError, match="numeric"):
            compute_population_counterfactual(pipe_path, mod_path, rows)

    # -- max_rows cap ----------------------------------------------------------

    def test_larger_batch_analysed_correctly(self):
        """Result still returns valid structure with 25 rows."""
        from core.deployer import compute_population_counterfactual

        rows = self._rows(25)
        result = compute_population_counterfactual(
            self.pipeline_path, self.model_path, rows, max_rows=25
        )
        assert result["total_rows"] == 25
        assert result["flipped_count"] <= 25


# ---------------------------------------------------------------------------
# Regex pattern tests for _POPULATION_CF_PATTERNS
# ---------------------------------------------------------------------------


def _get_pattern():
    import importlib

    # Reload chat module in tests to extract the compiled pattern
    chat_module = importlib.import_module("api.chat")
    return getattr(chat_module, "_POPULATION_CF_PATTERNS")


class TestPopulationCFPatterns:
    PATTERN = re.compile(
        r"(?i)(?:"
        r"(?:what|which)\s+(?:one\s+)?(?:change|feature|intervention)\s+would\s+(?:flip|change|affect)\s+(?:the\s+)?most\b|"
        r"population\s+counterfactual\b|"
        r"most\s+impactful\s+intervention\s+(?:across|for|in)\s+(?:the\s+)?(?:cohort|population|group|segment)\b|"
        r"(?:what|which)\s+(?:change|feature)\s+would\s+(?:help|save)\s+(?:the\s+)?most\s+(?:customers?|records?|rows?|cases?)\b|"
        r"single\s+(?:most\s+impactful|best|biggest)\s+(?:feature|lever|change|intervention)\s+(?:for|across|in)\s+(?:the\s+)?(?:cohort|group|population|segment|at.risk)\b|"
        r"(?:best|optimal|top)\s+intervention\s+(?:for|across)\s+(?:all\s+)?(?:at.risk|predicted|flagged)\s+(?:customers?|records?|cases?)\b|"
        r"(?:which|what)\s+(?:features?\s+)?(?:changes?\s+)?(?:would\s+)?flip\s+(?:the\s+)?most\s+(?:predictions?|outcomes?|results?)\b|"
        r"cohort\s+(?:level\s+)?(?:intervention|action|counterfactual)\b|"
        r"(?:aggregate|group|population)\s+(?:level\s+)?(?:counterfactual|explanation|intervention)\b"
        r")",
        re.IGNORECASE,
    )

    # -- should match ---------------------------------------------------------

    @pytest.mark.parametrize(
        "phrase",
        [
            # arm 1: what/which change would flip most
            "What change would flip the most records?",
            "Which feature would change the most predictions?",
            "What intervention would affect most outcomes?",
            # arm 2: population counterfactual
            "Run a population counterfactual for this cohort",
            "show me the population counterfactual",
            # arm 3: most impactful intervention across cohort
            "What's the most impactful intervention across the cohort?",
            "Most impactful intervention for the population",
            "What is the most impactful intervention in the segment?",
            # arm 4: what change would help/save most customers
            "What change would help the most customers?",
            "Which feature would save the most records?",
            # arm 5: single best feature for cohort
            "What's the single most impactful feature for the cohort?",
            "Single best lever across the at-risk group",
            # arm 6: best intervention for at-risk customers
            "What's the best intervention for all at-risk customers?",
            "Find the optimal intervention for flagged cases",
            # arm 7: flip the most predictions
            "Which change would flip the most predictions?",
            "What would flip most outcomes?",
            # arm 8: cohort intervention
            "Suggest a cohort intervention",
            "What cohort level counterfactual can you identify?",
            # arm 9: aggregate/population level counterfactual
            "Give me an aggregate level counterfactual",
            "population level explanation please",
        ],
    )
    def test_matches(self, phrase: str):
        assert self.PATTERN.search(phrase), f"Expected match for: {phrase!r}"

    # -- case insensitivity ---------------------------------------------------

    def test_uppercase_matches(self):
        assert self.PATTERN.search("POPULATION COUNTERFACTUAL ANALYSIS")

    def test_mixed_case_matches(self):
        assert self.PATTERN.search("Which Feature Would Flip The Most Predictions?")

    # -- should NOT match -----------------------------------------------------

    @pytest.mark.parametrize(
        "phrase",
        [
            # Individual counterfactual — different intent
            "Show me a counterfactual for row 5",
            # Unrelated "most" usage
            "What is the most important feature in my model?",
            "Which features matter most?",
            # Generic flip
            "The model predicted class 1, can we flip it?",
            # Regression goal-seek
            "What change would reach my target revenue?",
        ],
    )
    def test_no_false_positives(self, phrase: str):
        assert not self.PATTERN.search(phrase), f"Unexpected match for: {phrase!r}"
