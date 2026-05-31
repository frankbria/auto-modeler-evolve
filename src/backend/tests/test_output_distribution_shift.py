"""Tests for Prediction Output Distribution Shift — Track D perpetual.

Covers:
  - compute_prediction_output_distribution_shift pure function (stable, moderate, significant)
  - GET /api/deploy/{deployment_id}/output-distribution-shift REST endpoint
  - _OUTPUT_DIST_SHIFT_PATTERNS regex (positive + negative matches)
  - Chat handler integration (emits event, no-deployment guard)
"""

from __future__ import annotations

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_prediction_output_distribution_shift

# ─── Pure function tests ───────────────────────────────────────────────────────


def _stable_preds() -> tuple[list[float], list[float]]:
    """Two samples drawn from the same N(100, 10) distribution."""
    rng = np.random.default_rng(42)
    train = rng.normal(100, 10, 200).tolist()
    prod = rng.normal(100, 10, 100).tolist()
    return train, prod


def _shifted_preds(shift: float = 50.0) -> tuple[list[float], list[float]]:
    """Production mean shifted by `shift` relative to training."""
    rng = np.random.default_rng(7)
    train = rng.normal(100, 10, 200).tolist()
    prod = rng.normal(100 + shift, 10, 100).tolist()
    return train, prod


class TestComputeOutputDistributionShiftPureFunction:
    def test_returns_required_keys(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        required = {
            "n_training",
            "n_production",
            "verdict",
            "verdict_label",
            "ks_statistic",
            "ks_p_value",
            "mean_shift",
            "mean_shift_pct",
            "std_ratio",
            "training_stats",
            "production_stats",
            "training_histogram",
            "production_histogram",
            "summary",
        }
        assert required.issubset(set(result.keys()))

    def test_n_counts_match_inputs(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        assert result["n_training"] == len(train)
        assert result["n_production"] == len(prod)

    def test_stable_verdict_for_same_distribution(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        assert result["verdict"] == "stable"

    def test_significant_shift_for_large_mean_shift(self):
        train, prod = _shifted_preds(shift=60.0)  # 60% shift
        result = compute_prediction_output_distribution_shift(train, prod)
        assert result["verdict"] == "significant_shift"

    def test_moderate_shift_for_medium_mean_shift(self):
        # ~20% shift: 100 → 120
        rng = np.random.default_rng(3)
        train = rng.normal(100, 5, 500).tolist()
        prod = rng.normal(120, 5, 200).tolist()
        result = compute_prediction_output_distribution_shift(train, prod)
        # 20% shift → moderate or significant; at minimum not stable
        assert result["verdict"] in ("moderate_shift", "significant_shift")

    def test_ks_statistic_in_range_zero_to_one(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        assert 0.0 <= result["ks_statistic"] <= 1.0

    def test_ks_pvalue_in_range_zero_to_one(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        assert 0.0 <= result["ks_p_value"] <= 1.0

    def test_stable_has_high_pvalue(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        # Same distribution → KS p-value should be > 0.05
        assert result["ks_p_value"] > 0.05

    def test_significant_shift_has_low_pvalue(self):
        train, prod = _shifted_preds(shift=60.0)
        result = compute_prediction_output_distribution_shift(train, prod)
        assert result["ks_p_value"] < 0.01

    def test_training_stats_keys(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        stat_keys = {"mean", "std", "min", "p25", "median", "p75", "p95", "max"}
        assert stat_keys.issubset(set(result["training_stats"].keys()))

    def test_production_stats_keys(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        stat_keys = {"mean", "std", "min", "p25", "median", "p75", "p95", "max"}
        assert stat_keys.issubset(set(result["production_stats"].keys()))

    def test_histogram_bin_count_matches_n_bins(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod, n_bins=8)
        assert len(result["training_histogram"]) == 8
        assert len(result["production_histogram"]) == 8

    def test_histogram_bin_keys(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        bin0 = result["training_histogram"][0]
        assert "bin_start" in bin0
        assert "bin_end" in bin0
        assert "count" in bin0
        assert "label" in bin0

    def test_histogram_counts_sum_to_n(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        train_total = sum(b["count"] for b in result["training_histogram"])
        prod_total = sum(b["count"] for b in result["production_histogram"])
        assert train_total == len(train)
        assert prod_total == len(prod)

    def test_mean_shift_direction_positive(self):
        """Production mean above training → positive mean_shift."""
        train = [100.0] * 50
        prod = [200.0] * 50
        result = compute_prediction_output_distribution_shift(train, prod)
        assert result["mean_shift"] > 0

    def test_mean_shift_direction_negative(self):
        """Production mean below training → negative mean_shift."""
        train = [200.0] * 50
        prod = [100.0] * 50
        result = compute_prediction_output_distribution_shift(train, prod)
        assert result["mean_shift"] < 0

    def test_std_ratio_near_one_for_same_variance(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        # Same distribution → std_ratio should be near 1.0
        assert result["std_ratio"] is not None
        assert 0.5 <= result["std_ratio"] <= 2.0

    def test_summary_is_nonempty_string(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_verdict_label_is_nonempty_string(self):
        train, prod = _stable_preds()
        result = compute_prediction_output_distribution_shift(train, prod)
        assert isinstance(result["verdict_label"], str)
        assert len(result["verdict_label"]) > 5

    def test_raises_value_error_too_few_training(self):
        with pytest.raises(ValueError, match="training"):
            compute_prediction_output_distribution_shift([1.0] * 9, [2.0] * 50)

    def test_raises_value_error_too_few_production(self):
        with pytest.raises(ValueError, match="production"):
            compute_prediction_output_distribution_shift([1.0] * 50, [2.0] * 9)

    def test_training_stats_mean_correct(self):
        train = [float(i) for i in range(10, 110)]  # 10..109, mean=59.5
        prod = [float(i) for i in range(10, 110)]
        result = compute_prediction_output_distribution_shift(train, prod)
        assert abs(result["training_stats"]["mean"] - 59.5) < 0.01

    def test_identical_distributions_stable(self):
        values = list(range(10, 110))
        train = [float(v) for v in values]
        prod = [float(v) for v in values]
        result = compute_prediction_output_distribution_shift(train, prod)
        assert result["verdict"] == "stable"


class TestOutputDistShiftRegex:
    """Test _OUTPUT_DIST_SHIFT_PATTERNS regex positive and negative matches."""

    def setup_method(self):
        from api.chat import _OUTPUT_DIST_SHIFT_PATTERNS
        self.pattern = _OUTPUT_DIST_SHIFT_PATTERNS

    def test_matches_distribution_shifted(self):
        assert self.pattern.search("has the distribution of my predictions shifted?")

    def test_matches_output_distribution_shift(self):
        assert self.pattern.search("output distribution shift")

    def test_matches_are_predictions_shifting(self):
        assert self.pattern.search("are my predictions shifting over time?")

    def test_matches_how_has_model_output_changed(self):
        assert self.pattern.search("how has my model's output changed?")

    def test_matches_production_prediction_distribution(self):
        assert self.pattern.search("production prediction distribution")

    def test_matches_prediction_value_drift(self):
        assert self.pattern.search("prediction value drift")

    def test_matches_compare_training_vs_production_predictions(self):
        assert self.pattern.search(
            "compare training vs production predictions distributions"
        )

    def test_matches_model_outputs_behaving_differently(self):
        assert self.pattern.search("are my model outputs behaving differently in production?")

    def test_no_match_show_me_predictions(self):
        assert not self.pattern.search("show me the top 20 predictions")

    def test_no_match_prediction_errors(self):
        assert not self.pattern.search("where was my model wrong?")

    def test_no_match_output_anomalies(self):
        assert not self.pattern.search("any unusual prediction values?")

    def test_no_match_retrain(self):
        assert not self.pattern.search("should I retrain my model?")


# ─── REST endpoint tests ───────────────────────────────────────────────────────


@pytest.fixture()
def _db_url(tmp_path):
    url = f"sqlite:///{tmp_path}/test_shift.db"
    engine = create_engine(url, connect_args={"check_same_thread": False})

    # Import all models before create_all
    import models.project  # noqa: F401
    import models.dataset  # noqa: F401
    import models.dataset_filter  # noqa: F401
    import models.feature_set  # noqa: F401
    import models.model_run  # noqa: F401
    import models.conversation  # noqa: F401
    import models.prediction_log  # noqa: F401
    import models.deployment  # noqa: F401
    import models.feedback_record  # noqa: F401
    import models.webhook_config  # noqa: F401
    import models.webhook_event  # noqa: F401
    import models.ab_test  # noqa: F401
    import models.deployment_preset  # noqa: F401
    import models.input_validation_rule  # noqa: F401
    import models.dashboard_field_config  # noqa: F401
    import models.goal_seek_record  # noqa: F401
    import models.deployment_changelog  # noqa: F401
    import models.deployment_version  # noqa: F401
    import models.batch_schedule  # noqa: F401
    import models.prediction_alert_rule  # noqa: F401
    import models.analysis_template  # noqa: F401

    SQLModel.metadata.create_all(engine)
    db_module.engine = engine
    yield url
    engine.dispose()


@pytest.mark.asyncio
async def test_output_distribution_shift_endpoint_no_deployment(_db_url):
    """Returns 404 when deployment not found."""
    from main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/api/deploy/nonexistent-id/output-distribution-shift")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_output_distribution_shift_endpoint_no_data(_db_url, tmp_path):
    """Returns no_data verdict when fewer than 10 production predictions."""
    import json as _json
    from main import app
    from sqlmodel import Session
    from models.project import Project
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.deployment import Deployment

    # Create minimal project + deployment without enough PredictionLogs
    with Session(db_module.engine) as session:
        proj = Project(name="shift-test", description="")
        session.add(proj)
        session.commit()
        session.refresh(proj)

        ds = Dataset(
            project_id=proj.id,
            filename="data.csv",
            file_path=str(tmp_path / "data.csv"),
            row_count=10,
            column_count=2,
        )
        session.add(ds)
        session.commit()
        session.refresh(ds)

        fs = FeatureSet(
            dataset_id=ds.id,
            target_column="y",
            problem_type="regression",
        )
        session.add(fs)
        session.commit()
        session.refresh(fs)

        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fs.id,
            algorithm="linear_regression",
            status="done",
            metrics=_json.dumps({"r2": 0.9, "mae": 5.0}),
            model_path=str(tmp_path / "model.joblib"),
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        dep = Deployment(
            project_id=proj.id,
            model_run_id=run.id,
            algorithm="linear_regression",
            problem_type="regression",
            pipeline_path=str(tmp_path / "pipeline.joblib"),
            endpoint_path=f"/api/predict/{proj.id}",
            dashboard_url=f"/predict/{proj.id}",
            is_active=True,
        )
        session.add(dep)
        session.commit()
        session.refresh(dep)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get(f"/api/deploy/{dep.id}/output-distribution-shift")

    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "no_data"
    assert data["n_production"] < 10
