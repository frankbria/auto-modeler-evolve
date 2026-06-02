"""Tests for compute_deployment_monitoring_digest() pure function, REST endpoint, and chat regex."""

from __future__ import annotations


from api.chat import _MONITORING_DIGEST_PATTERNS
from core.analyzer import compute_deployment_monitoring_digest


# ---------------------------------------------------------------------------
# 1. Pure function tests
# ---------------------------------------------------------------------------


class TestComputeDeploymentMonitoringDigest:
    def test_returns_required_keys(self):
        result = compute_deployment_monitoring_digest()
        for key in [
            "signals",
            "overall_health",
            "overall_score",
            "signals_total",
            "signals_firing",
            "priority_actions",
            "summary",
            "problem_type",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_no_signals_returns_healthy(self):
        result = compute_deployment_monitoring_digest(usage_7d=10)
        assert result["overall_health"] == "healthy"
        assert result["overall_score"] == 100
        assert result["signals_firing"] == 0

    def test_anomaly_no_anomalies_is_green(self):
        result = compute_deployment_monitoring_digest(
            anomaly_result={"verdict": "no_anomalies", "anomaly_rate": 0.0}
        )
        anom = next(
            s for s in result["signals"] if s["signal_key"] == "output_anomalies"
        )
        assert anom["severity"] == "green"
        assert anom["is_available"] is True

    def test_anomaly_few_anomalies_is_amber(self):
        result = compute_deployment_monitoring_digest(
            anomaly_result={"verdict": "few_anomalies", "anomaly_rate": 0.07}
        )
        anom = next(
            s for s in result["signals"] if s["signal_key"] == "output_anomalies"
        )
        assert anom["severity"] == "amber"
        assert "7%" in anom["finding"]

    def test_anomaly_many_anomalies_is_red(self):
        result = compute_deployment_monitoring_digest(
            anomaly_result={"verdict": "many_anomalies", "anomaly_rate": 0.15}
        )
        anom = next(
            s for s in result["signals"] if s["signal_key"] == "output_anomalies"
        )
        assert anom["severity"] == "red"

    def test_anomaly_no_data_is_gray(self):
        result = compute_deployment_monitoring_digest(
            anomaly_result={"verdict": "no_data"}
        )
        anom = next(
            s for s in result["signals"] if s["signal_key"] == "output_anomalies"
        )
        assert anom["severity"] == "gray"
        assert anom["is_available"] is False

    def test_value_trend_stable_is_green(self):
        result = compute_deployment_monitoring_digest(
            value_trend_result={"direction": "stable", "overall_change_pct": 0.5},
            problem_type="regression",
        )
        trend = next(s for s in result["signals"] if s["signal_key"] == "value_trend")
        assert trend["severity"] == "green"

    def test_value_trend_trending_up_is_amber(self):
        result = compute_deployment_monitoring_digest(
            value_trend_result={"direction": "trending_up", "overall_change_pct": 12.5},
            problem_type="regression",
        )
        trend = next(s for s in result["signals"] if s["signal_key"] == "value_trend")
        assert trend["severity"] == "amber"
        assert "Trending up" in trend["verdict_label"]

    def test_value_trend_not_included_for_classification(self):
        result = compute_deployment_monitoring_digest(
            value_trend_result={"direction": "stable", "overall_change_pct": 0.0},
            problem_type="classification",
        )
        keys = [s["signal_key"] for s in result["signals"]]
        assert "value_trend" not in keys

    def test_dist_shift_stable_is_green(self):
        result = compute_deployment_monitoring_digest(
            dist_shift_result={"verdict": "stable", "mean_shift_pct": 2.0}
        )
        ds = next(s for s in result["signals"] if s["signal_key"] == "dist_shift")
        assert ds["severity"] == "green"

    def test_dist_shift_significant_is_red(self):
        result = compute_deployment_monitoring_digest(
            dist_shift_result={"verdict": "significant_shift", "mean_shift_pct": 35.0}
        )
        ds = next(s for s in result["signals"] if s["signal_key"] == "dist_shift")
        assert ds["severity"] == "red"

    def test_readiness_stable_is_green(self):
        result = compute_deployment_monitoring_digest(
            readiness_result={"verdict": "stable", "score": 0, "recommendations": []}
        )
        rr = next(
            s for s in result["signals"] if s["signal_key"] == "retraining_readiness"
        )
        assert rr["severity"] == "green"

    def test_readiness_retrain_now_is_red(self):
        result = compute_deployment_monitoring_digest(
            readiness_result={
                "verdict": "retrain_now",
                "score": 85,
                "recommendations": ["Retrain immediately."],
            }
        )
        rr = next(
            s for s in result["signals"] if s["signal_key"] == "retraining_readiness"
        )
        assert rr["severity"] == "red"

    def test_usage_active_is_green(self):
        result = compute_deployment_monitoring_digest(usage_7d=50, usage_prev_7d=45)
        usage = next(
            s for s in result["signals"] if s["signal_key"] == "usage_activity"
        )
        assert usage["severity"] == "green"

    def test_usage_zero_is_amber(self):
        result = compute_deployment_monitoring_digest(usage_7d=0, usage_prev_7d=30)
        usage = next(
            s for s in result["signals"] if s["signal_key"] == "usage_activity"
        )
        assert usage["severity"] == "amber"

    def test_one_red_signal_gives_warning(self):
        result = compute_deployment_monitoring_digest(
            dist_shift_result={"verdict": "significant_shift", "mean_shift_pct": 35.0},
        )
        assert result["overall_health"] == "warning"

    def test_two_red_signals_gives_critical(self):
        result = compute_deployment_monitoring_digest(
            anomaly_result={"verdict": "many_anomalies", "anomaly_rate": 0.2},
            dist_shift_result={"verdict": "significant_shift", "mean_shift_pct": 40.0},
        )
        assert result["overall_health"] == "critical"

    def test_all_green_gives_healthy(self):
        result = compute_deployment_monitoring_digest(
            anomaly_result={"verdict": "no_anomalies", "anomaly_rate": 0.0},
            dist_shift_result={"verdict": "stable", "mean_shift_pct": 1.0},
            readiness_result={"verdict": "stable", "score": 0, "recommendations": []},
            usage_7d=20,
            usage_prev_7d=18,
        )
        assert result["overall_health"] == "healthy"
        assert result["signals_firing"] == 0

    def test_amber_signals_give_watching(self):
        result = compute_deployment_monitoring_digest(
            anomaly_result={"verdict": "few_anomalies", "anomaly_rate": 0.05},
        )
        assert result["overall_health"] == "watching"

    def test_score_decreases_with_red_signals(self):
        result_clean = compute_deployment_monitoring_digest()
        result_red = compute_deployment_monitoring_digest(
            dist_shift_result={"verdict": "significant_shift", "mean_shift_pct": 35.0},
        )
        assert result_red["overall_score"] < result_clean["overall_score"]

    def test_priority_actions_populated_for_retrain_now(self):
        result = compute_deployment_monitoring_digest(
            readiness_result={
                "verdict": "retrain_now",
                "score": 90,
                "recommendations": [
                    "Retrain your model now.",
                    "Check for distribution shift.",
                ],
            }
        )
        assert len(result["priority_actions"]) >= 1

    def test_priority_actions_capped_at_3(self):
        result = compute_deployment_monitoring_digest(
            anomaly_result={"verdict": "many_anomalies", "anomaly_rate": 0.2},
            dist_shift_result={"verdict": "significant_shift", "mean_shift_pct": 35.0},
            readiness_result={
                "verdict": "retrain_now",
                "score": 90,
                "recommendations": ["Retrain immediately.", "Investigate PSI."],
            },
            usage_7d=0,
        )
        assert len(result["priority_actions"]) <= 3

    def test_summary_is_string(self):
        result = compute_deployment_monitoring_digest()
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_each_signal_has_required_keys(self):
        result = compute_deployment_monitoring_digest(
            anomaly_result={"verdict": "no_anomalies", "anomaly_rate": 0.0},
            usage_7d=10,
        )
        for signal in result["signals"]:
            for key in [
                "signal_key",
                "signal_label",
                "verdict",
                "verdict_label",
                "severity",
                "finding",
                "icon",
                "is_available",
            ]:
                assert key in signal, f"Signal missing key: {key}"


# ---------------------------------------------------------------------------
# 2. Regex tests
# ---------------------------------------------------------------------------


class TestMonitoringDigestPatterns:
    def test_show_all_monitoring_signals(self):
        assert _MONITORING_DIGEST_PATTERNS.search("show all my monitoring signals")

    def test_monitoring_signal_digest(self):
        assert _MONITORING_DIGEST_PATTERNS.search("monitoring signal digest")

    def test_deployment_diagnostics(self):
        assert _MONITORING_DIGEST_PATTERNS.search("deployment diagnostics")

    def test_monitoring_overview(self):
        assert _MONITORING_DIGEST_PATTERNS.search("give me a full monitoring overview")

    def test_all_monitoring_signals(self):
        assert _MONITORING_DIGEST_PATTERNS.search("all my monitoring signals")

    def test_comprehensive_monitoring_check(self):
        assert _MONITORING_DIGEST_PATTERNS.search("comprehensive monitoring check")

    def test_what_are_my_monitoring_signals(self):
        assert _MONITORING_DIGEST_PATTERNS.search("what are all my monitoring signals")

    def test_monitoring_signals_at_a_glance(self):
        assert _MONITORING_DIGEST_PATTERNS.search("monitoring signals at a glance")

    def test_false_positive_retrain_recommendation(self):
        assert not _MONITORING_DIGEST_PATTERNS.search("should I retrain my model")

    def test_false_positive_prediction_anomalies(self):
        assert not _MONITORING_DIGEST_PATTERNS.search("show me prediction anomalies")
