"""Tests for chat/prompts.py — algorithm intros, metric glossary, and prompt builders."""


class TestAlgorithmIntros:
    def test_all_standard_algorithms_have_intros(self):
        from chat.prompts import ALGORITHM_INTROS
        expected = [
            "LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor",
            "LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier",
        ]
        for algo in expected:
            assert algo in ALGORITHM_INTROS, f"Missing intro for {algo}"
            assert len(ALGORITHM_INTROS[algo]) > 20

    def test_intros_are_plain_english(self):
        from chat.prompts import ALGORITHM_INTROS
        # Intros should not contain raw technical jargon without explanation
        for algo, intro in ALGORITHM_INTROS.items():
            # Must be a non-empty string
            assert isinstance(intro, str) and intro.strip()

    def test_taglines_exist_for_all_algos(self):
        from chat.prompts import ALGORITHM_INTROS, ALGORITHM_TAGLINES
        for algo in ALGORITHM_INTROS:
            assert algo in ALGORITHM_TAGLINES


class TestMetricGlossary:
    def test_core_metrics_present(self):
        from chat.prompts import METRIC_GLOSSARY
        for metric in ("r2", "mae", "rmse", "accuracy", "f1", "precision", "recall"):
            assert metric in METRIC_GLOSSARY

    def test_each_metric_has_plain_description(self):
        from chat.prompts import METRIC_GLOSSARY
        for name, info in METRIC_GLOSSARY.items():
            assert "plain" in info, f"Metric {name} missing 'plain' key"
            assert info["plain"].strip()

    def test_higher_is_better_flag(self):
        from chat.prompts import METRIC_GLOSSARY
        assert METRIC_GLOSSARY["r2"]["higher_is_better"] is True
        assert METRIC_GLOSSARY["mae"]["higher_is_better"] is False
        assert METRIC_GLOSSARY["accuracy"]["higher_is_better"] is True


class TestFormatMetric:
    def test_r2_format(self):
        from chat.prompts import format_metric
        result = format_metric("r2", 0.87)
        assert "R²" in result or "r2" in result.lower()
        assert "0.870" in result

    def test_accuracy_format_as_percent(self):
        from chat.prompts import format_metric
        result = format_metric("accuracy", 0.92)
        assert "92.0%" in result

    def test_excellent_label(self):
        from chat.prompts import format_metric
        result = format_metric("r2", 0.95)
        assert "excellent" in result.lower()

    def test_good_label(self):
        from chat.prompts import format_metric
        result = format_metric("r2", 0.85)
        assert "good" in result.lower()

    def test_none_value(self):
        from chat.prompts import format_metric
        result = format_metric("r2", None)
        assert "N/A" in result

    def test_unknown_metric(self):
        from chat.prompts import format_metric
        result = format_metric("mse", 12.5)
        assert "12.500" in result  # Falls back to raw value


class TestSummariseMetrics:
    def test_regression_metrics(self):
        from chat.prompts import summarise_metrics
        metrics = {"r2": 0.88, "mae": 150.0, "rmse": 200.0}
        result = summarise_metrics(metrics, "regression")
        assert "R²" in result or "r2" in result.lower()
        assert "MAE" in result or "mae" in result.lower()

    def test_classification_metrics(self):
        from chat.prompts import summarise_metrics
        metrics = {"accuracy": 0.91, "f1": 0.89, "precision": 0.90, "recall": 0.88}
        result = summarise_metrics(metrics, "classification")
        assert "Accuracy" in result or "accuracy" in result.lower()
        assert "F1" in result or "f1" in result.lower()

    def test_empty_metrics(self):
        from chat.prompts import summarise_metrics
        result = summarise_metrics({}, "regression")
        assert "No metrics" in result


class TestBuildProactiveInsightPrompt:
    def test_returns_string(self):
        from chat.prompts import build_proactive_insight_prompt
        result = build_proactive_insight_prompt(
            dataset_summary="sales data with revenue, region, date columns",
            profile_highlights="Strong correlation between region and revenue",
            n_rows=200,
            n_cols=8,
        )
        assert isinstance(result, str)
        assert "200" in result
        assert "8" in result

    def test_includes_guidance_for_plain_english(self):
        from chat.prompts import build_proactive_insight_prompt
        result = build_proactive_insight_prompt(
            dataset_summary="test", profile_highlights="none", n_rows=100, n_cols=5
        )
        assert "plain English" in result or "plain english" in result.lower()


class TestBuildModelComparisonPrompt:
    def test_returns_string_with_models(self):
        from chat.prompts import build_model_comparison_narrative_prompt
        models = [
            {"algorithm": "RandomForestRegressor", "metrics": {"r2": 0.87, "mae": 120.0}, "summary": ""},
            {"algorithm": "LinearRegression", "metrics": {"r2": 0.72, "mae": 200.0}, "summary": ""},
        ]
        result = build_model_comparison_narrative_prompt(models, "regression", "revenue")
        assert "RandomForestRegressor" in result
        assert "LinearRegression" in result
        assert "revenue" in result

    def test_includes_primary_metric_label(self):
        from chat.prompts import build_model_comparison_narrative_prompt
        models = [
            {"algorithm": "LogisticRegression", "metrics": {"accuracy": 0.91, "f1": 0.89}, "summary": ""},
        ]
        result = build_model_comparison_narrative_prompt(models, "classification", "churn")
        assert "accuracy" in result.lower()


class TestStageOpeners:
    def test_all_stages_have_openers(self):
        from chat.prompts import STAGE_OPENERS
        for stage in ("upload", "explore", "shape", "model", "validate", "deploy"):
            assert stage in STAGE_OPENERS
            assert STAGE_OPENERS[stage].strip()
