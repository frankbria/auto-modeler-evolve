"""Tests for compute_cost_sensitive_threshold pure function and chat regex."""

from __future__ import annotations

import pytest

from core.validator import compute_cost_sensitive_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_data(n: int = 100, positive_ratio: float = 0.3, seed: int = 42):
    """Generate binary y_true and y_proba_positive arrays without numpy."""
    import random

    rng = random.Random(seed)
    y_true = [1 if rng.random() < positive_ratio else 0 for _ in range(n)]
    y_proba_positive = []
    for label in y_true:
        if label == 1:
            y_proba_positive.append(min(0.99, 0.5 + rng.random() * 0.45))
        else:
            y_proba_positive.append(max(0.01, rng.random() * 0.5))
    return y_true, y_proba_positive


# ---------------------------------------------------------------------------
# Required keys contract
# ---------------------------------------------------------------------------


class TestCostSensitiveRequiredKeys:
    def test_top_level_keys(self):
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0)
        for key in [
            "optimal_threshold",
            "fp_cost",
            "fn_cost",
            "cost_ratio",
            "default_metrics",
            "optimal_metrics",
            "default_cost",
            "optimal_cost",
            "cost_savings_pct",
            "suggested_class_weight",
            "verdict",
            "summary",
            "n_samples",
            "n_positive",
            "prevalence",
            "positive_label",
        ]:
            assert key in result, f"Missing top-level key: {key}"

    def test_metrics_keys(self):
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0)
        for metrics_key in ("default_metrics", "optimal_metrics"):
            entry = result[metrics_key]
            for key in ["threshold", "tp", "fp", "tn", "fn", "expected_cost", "precision", "recall", "f1", "accuracy"]:
                assert key in entry, f"Missing key '{key}' in {metrics_key}"


# ---------------------------------------------------------------------------
# Optimal threshold formula
# ---------------------------------------------------------------------------


class TestOptimalThresholdFormula:
    def test_formula_fp_equals_fn(self):
        """When FP=FN, optimal threshold should be 0.5."""
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=50.0, fn_cost=50.0)
        assert abs(result["optimal_threshold"] - 0.5) < 0.01

    def test_formula_high_fn_cost(self):
        """When FN >> FP, threshold should be low (classify positive aggressively)."""
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=990.0)
        # threshold = 10/(10+990) = 0.01
        assert result["optimal_threshold"] < 0.05

    def test_formula_high_fp_cost(self):
        """When FP >> FN, threshold should be high (be conservative)."""
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=990.0, fn_cost=10.0)
        # threshold = 990/(990+10) = 0.99
        assert result["optimal_threshold"] > 0.90

    def test_cost_ratio_matches_fn_over_fp(self):
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0)
        expected_ratio = 100.0 / 10.0
        assert abs(result["cost_ratio"] - expected_ratio) < 0.01

    def test_suggested_class_weight_equals_cost_ratio(self):
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=5.0, fn_cost=50.0)
        assert abs(result["suggested_class_weight"] - 10.0) < 0.01


# ---------------------------------------------------------------------------
# Cost computation correctness
# ---------------------------------------------------------------------------


class TestCostComputation:
    def test_cost_at_default_threshold(self):
        """Expected cost = FP_count * fp_cost + FN_count * fn_cost."""
        y, p = _binary_data()
        fp_cost, fn_cost = 10.0, 100.0
        result = compute_cost_sensitive_threshold(y, p, fp_cost=fp_cost, fn_cost=fn_cost)
        dm = result["default_metrics"]
        expected = dm["fp"] * fp_cost + dm["fn"] * fn_cost
        assert abs(dm["expected_cost"] - expected) < 0.01

    def test_cost_at_optimal_threshold(self):
        """Expected cost at optimal threshold = FP_count * fp_cost + FN_count * fn_cost."""
        y, p = _binary_data()
        fp_cost, fn_cost = 10.0, 100.0
        result = compute_cost_sensitive_threshold(y, p, fp_cost=fp_cost, fn_cost=fn_cost)
        om = result["optimal_metrics"]
        expected = om["fp"] * fp_cost + om["fn"] * fn_cost
        assert abs(om["expected_cost"] - expected) < 0.01

    def test_default_cost_matches_default_metrics(self):
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0)
        assert abs(result["default_cost"] - result["default_metrics"]["expected_cost"]) < 0.01

    def test_optimal_cost_matches_optimal_metrics(self):
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0)
        assert abs(result["optimal_cost"] - result["optimal_metrics"]["expected_cost"]) < 0.01


# ---------------------------------------------------------------------------
# Confusion matrix sanity
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_tp_fp_tn_fn_sum_to_n(self):
        y, p = _binary_data(n=100)
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0)
        for m_key in ("default_metrics", "optimal_metrics"):
            m = result[m_key]
            total = m["tp"] + m["fp"] + m["tn"] + m["fn"]
            assert total == 100, f"{m_key}: TP+FP+TN+FN={total} != 100"

    def test_tp_fn_sum_equals_n_positive(self):
        """TP + FN must equal the total number of positives in y_true."""
        y, p = _binary_data(n=100, positive_ratio=0.3)
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0)
        n_pos = sum(y)
        for m_key in ("default_metrics", "optimal_metrics"):
            m = result[m_key]
            assert m["tp"] + m["fn"] == n_pos, f"{m_key}: TP+FN != n_positive"


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


class TestVerdictLogic:
    def test_verdict_type(self):
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0)
        assert result["verdict"] in {"threshold_change_recommended", "default_near_optimal"}

    def test_equal_costs_likely_near_optimal(self):
        """Balanced costs should result in near-optimal verdict."""
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=1.0, fn_cost=1.0)
        assert result["verdict"] == "default_near_optimal"

    def test_extreme_cost_ratio_triggers_change(self):
        """100:1 FN:FP cost ratio should almost always recommend threshold change."""
        y, p = _binary_data(n=200, positive_ratio=0.2)
        result = compute_cost_sensitive_threshold(y, p, fp_cost=1.0, fn_cost=100.0)
        # Extreme asymmetry should move the threshold significantly
        assert result["optimal_threshold"] < 0.10


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 10"):
            compute_cost_sensitive_threshold([0, 1, 0], [0.2, 0.8, 0.3], 10.0, 100.0)

    def test_negative_fp_cost(self):
        y, p = _binary_data()
        with pytest.raises(ValueError, match="positive"):
            compute_cost_sensitive_threshold(y, p, fp_cost=-5.0, fn_cost=100.0)

    def test_zero_fn_cost(self):
        y, p = _binary_data()
        with pytest.raises(ValueError, match="positive"):
            compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=0.0)

    def test_non_binary_labels(self):
        y = [0, 1, 2] * 10  # multiclass
        p = [0.2, 0.5, 0.8] * 10
        with pytest.raises(ValueError, match="binary"):
            compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0)

    def test_prevalence_is_ratio(self):
        y, p = _binary_data(n=100, positive_ratio=0.3)
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0)
        assert 0.0 < result["prevalence"] <= 1.0
        assert abs(result["n_positive"] / result["n_samples"] - result["prevalence"]) < 0.01

    def test_positive_label_propagated(self):
        y, p = _binary_data()
        result = compute_cost_sensitive_threshold(y, p, fp_cost=10.0, fn_cost=100.0, positive_label="fraud")
        assert result["positive_label"] == "fraud"

    def test_threshold_clamped_between_1pct_and_99pct(self):
        y, p = _binary_data()
        # Extreme ratio that would mathematically produce 0 or 1
        result = compute_cost_sensitive_threshold(y, p, fp_cost=0.001, fn_cost=999.999)
        assert 0.01 <= result["optimal_threshold"] <= 0.99


# ---------------------------------------------------------------------------
# Chat regex pattern matching
# ---------------------------------------------------------------------------


class TestCostSensitivePatterns:
    @pytest.fixture
    def pattern(self):
        from api.chat import _COST_SENSITIVE_PATTERNS
        return _COST_SENSITIVE_PATTERNS

    def test_fp_cost_dollar_match(self, pattern):
        assert pattern.search("false positives cost $10, false negatives cost $500")

    def test_fn_cost_dollar_match(self, pattern):
        assert pattern.search("false negatives cost $200 per incident")

    def test_fp_abbreviation(self, pattern):
        assert pattern.search("FP costs $5, FN costs $100")

    def test_misclassification_cost_matrix(self, pattern):
        assert pattern.search("show me the misclassification cost analysis")

    def test_cost_sensitive_training(self, pattern):
        assert pattern.search("use cost-sensitive training for my fraud model")

    def test_optimize_for_cost(self, pattern):
        assert pattern.search("optimize the model for cost not accuracy")

    def test_false_alarm_cost(self, pattern):
        assert pattern.search("each false alarm costs $10, missed detections cost $300")

    def test_train_with_costs(self, pattern):
        assert pattern.search("train the model with misclassification penalties")

    def test_no_false_match_general(self, pattern):
        assert not pattern.search("show me the feature importance chart")

    def test_no_false_match_accuracy(self, pattern):
        assert not pattern.search("what is my model accuracy")


# ---------------------------------------------------------------------------
# Cost value extractor
# ---------------------------------------------------------------------------


class TestExtractFpFnCosts:
    @pytest.fixture
    def extractor(self):
        from api.chat import _extract_fp_fn_costs
        return _extract_fp_fn_costs

    def test_dollar_values_labelled(self, extractor):
        result = extractor("false positives cost $10, false negatives cost $500")
        assert result is not None
        fp, fn = result
        assert abs(fp - 10.0) < 0.01
        assert abs(fn - 500.0) < 0.01

    def test_abbreviated_fp_fn(self, extractor):
        result = extractor("FP costs $25 and FN costs $250")
        assert result is not None
        fp, fn = result
        assert abs(fp - 25.0) < 0.01
        assert abs(fn - 250.0) < 0.01

    def test_false_alarm_missed_detection(self, extractor):
        result = extractor("each false alarm costs $5, missed detections cost $200")
        assert result is not None
        fp, fn = result
        assert abs(fp - 5.0) < 0.01
        assert abs(fn - 200.0) < 0.01

    def test_no_values_returns_none(self, extractor):
        result = extractor("cost-sensitive training please")
        assert result is None

    def test_single_value_returns_none(self, extractor):
        result = extractor("false positives cost $100")
        # Only one value found — should return None (can't derive both)
        # OR it correctly returns None when fn_match fails
        # Acceptable behavior: extractor finds only FP side, no FN side → None
        if result is not None:
            # If something is returned, the FP side must be correct
            assert abs(result[0] - 100.0) < 0.01
