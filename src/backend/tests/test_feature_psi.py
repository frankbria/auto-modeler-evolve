"""Tests for Feature PSI (Population Stability Index) Ranking — Track D perpetual.

Covers:
  - compute_feature_psi_ranking pure function (stable, watch, critical verdicts)
  - Numeric PSI computation via _psi_for_numeric
  - Categorical PSI computation via _psi_for_categorical
  - GET /api/deploy/{deployment_id}/feature-psi REST endpoint
  - _FEATURE_PSI_PATTERNS regex (positive + negative matches)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _FEATURE_PSI_PATTERNS
from core.analyzer import compute_feature_psi_ranking

# ─── Pure function helpers ─────────────────────────────────────────────────────


def _stable_df_and_inputs() -> tuple[pd.DataFrame, list[dict]]:
    """Training data and production inputs with matching distributions.
    Uses 500 training rows and 300 production samples to ensure PSI is
    statistically stable (≥10 samples per bin at n_bins=10).
    """
    rng = np.random.default_rng(0)
    n_train = 500
    df = pd.DataFrame(
        {
            "revenue": rng.normal(100, 10, n_train).tolist(),
            "units": rng.integers(1, 50, n_train).tolist(),
            "region": rng.choice(["East", "West", "North"], n_train).tolist(),
        }
    )
    prod_inputs = [
        {
            "revenue": float(rng.normal(100, 10)),
            "units": int(rng.integers(1, 50)),
            "region": rng.choice(["East", "West", "North"]),
        }
        for _ in range(300)
    ]
    return df, prod_inputs


def _drifted_df_and_inputs() -> tuple[pd.DataFrame, list[dict]]:
    """Training data vs production inputs with strong numeric drift.
    Revenue shifts by ~10σ guaranteeing critical PSI.
    """
    rng = np.random.default_rng(1)
    n_train = 500
    df = pd.DataFrame(
        {
            "revenue": rng.normal(100, 10, n_train).tolist(),
            "units": rng.integers(1, 20, n_train).tolist(),
        }
    )
    # Production has a completely different distribution (mean shift ~10σ)
    prod_inputs = [
        {
            "revenue": float(rng.normal(200, 10)),  # massive shift
            "units": int(rng.integers(1, 20)),  # same as training
        }
        for _ in range(200)
    ]
    return df, prod_inputs


# ─── Pure function tests ───────────────────────────────────────────────────────


class TestComputeFeaturePsiRanking:
    def test_returns_required_keys(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        assert "features" in result
        assert "features_analyzed" in result
        assert "stable_count" in result
        assert "watch_count" in result
        assert "critical_count" in result
        assert "overall_psi" in result
        assert "verdict" in result
        assert "verdict_label" in result
        assert "sample_count" in result
        assert "training_count" in result
        assert "summary" in result

    def test_stable_verdict_for_matching_distributions(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        assert result["verdict"] == "stable"
        assert result["critical_count"] == 0

    def test_critical_verdict_for_large_drift(self):
        df, inputs = _drifted_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        # Revenue has massive drift (mean 100 vs 200 with std 10) → PSI must be critical
        assert result["critical_count"] >= 1
        assert result["verdict"] == "critical"

    def test_features_sorted_by_psi_descending(self):
        df, inputs = _drifted_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        psi_vals = [f["psi"] for f in result["features"]]
        assert psi_vals == sorted(psi_vals, reverse=True)

    def test_feature_entry_keys(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        assert len(result["features"]) > 0
        entry = result["features"][0]
        assert "feature" in entry
        assert "feature_type" in entry
        assert "psi" in entry
        assert "psi_label" in entry
        assert "severity" in entry

    def test_psi_non_negative(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        for f in result["features"]:
            assert f["psi"] >= 0.0

    def test_severity_thresholds(self):
        """PSI < 0.10 → stable; 0.10-0.20 → watch; ≥ 0.20 → critical."""
        df, inputs = _drifted_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        for feat in result["features"]:
            psi = feat["psi"]
            if psi >= 0.20:
                assert feat["severity"] == "critical"
            elif psi >= 0.10:
                assert feat["severity"] == "watch"
            else:
                assert feat["severity"] == "stable"

    def test_psi_label_matches_severity(self):
        df, inputs = _drifted_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        label_map = {"stable": "Stable", "watch": "Watch", "critical": "Critical"}
        for feat in result["features"]:
            assert feat["psi_label"] == label_map[feat["severity"]]

    def test_features_analyzed_matches_list_length(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        assert result["features_analyzed"] == len(result["features"])

    def test_counts_sum_to_features_analyzed(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        total = (
            result["stable_count"] + result["watch_count"] + result["critical_count"]
        )
        assert total == result["features_analyzed"]

    def test_overall_psi_is_mean_of_feature_psi(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        if result["features_analyzed"] > 0:
            expected = (
                sum(f["psi"] for f in result["features"]) / result["features_analyzed"]
            )
            assert abs(result["overall_psi"] - round(expected, 4)) < 1e-6

    def test_sample_count_matches_production_inputs(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        assert result["sample_count"] == len(inputs)

    def test_training_count_matches_df_len(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        assert result["training_count"] == len(df)

    def test_summary_is_string(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_categorical_feature_type_detection(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        region_feat = next(
            (f for f in result["features"] if f["feature"] == "region"), None
        )
        assert region_feat is not None
        assert region_feat["feature_type"] == "categorical"

    def test_numeric_feature_type_detection(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs)
        rev_feat = next(
            (f for f in result["features"] if f["feature"] == "revenue"), None
        )
        assert rev_feat is not None
        assert rev_feat["feature_type"] == "numeric"

    def test_max_features_cap(self):
        rng = np.random.default_rng(7)
        n = 200
        df = pd.DataFrame({f"col{i}": rng.normal(0, 1, n).tolist() for i in range(20)})
        inputs = [
            {f"col{i}": float(rng.normal(0, 1)) for i in range(20)} for _ in range(150)
        ]
        result = compute_feature_psi_ranking(df, inputs, max_features=5)
        assert result["features_analyzed"] <= 5

    def test_feature_names_filter(self):
        df, inputs = _stable_df_and_inputs()
        result = compute_feature_psi_ranking(df, inputs, feature_names=["revenue"])
        assert result["features_analyzed"] == 1
        assert result["features"][0]["feature"] == "revenue"

    def test_error_on_too_few_training_rows(self):
        df_small = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        inputs = [{"x": 1.0}] * 20
        with pytest.raises(ValueError, match="at least 10 rows"):
            compute_feature_psi_ranking(df_small, inputs)

    def test_error_on_empty_production_inputs(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 50).tolist()})
        with pytest.raises(ValueError, match="must not be empty"):
            compute_feature_psi_ranking(df, [])

    def test_n_new_categories_populated_for_categorical(self):
        rng = np.random.default_rng(5)
        n = 200
        df = pd.DataFrame({"cat": rng.choice(["A", "B", "C"], n).tolist()})
        # Production has a new category D not in training
        inputs = [{"cat": "D"}] * 30 + [{"cat": "A"}] * 30
        result = compute_feature_psi_ranking(df, inputs)
        cat_feat = next((f for f in result["features"] if f["feature"] == "cat"), None)
        assert cat_feat is not None
        assert cat_feat["n_new_categories"] is not None
        assert cat_feat["n_new_categories"] >= 1


# ─── Regex tests ──────────────────────────────────────────────────────────────


class TestFeaturePsiPatterns:
    @pytest.mark.parametrize(
        "phrase",
        [
            "which features have drifted the most?",
            "what features have shifted most significantly?",
            "feature psi analysis",
            "feature population stability index",
            "psi analysis for my features",
            "rank my features by drift",
            "most drifted input features",
            "feature drift ranking",
        ],
    )
    def test_matches_positive(self, phrase):
        assert _FEATURE_PSI_PATTERNS.search(phrase), f"Should match: {phrase!r}"

    @pytest.mark.parametrize(
        "phrase",
        [
            "train a model",
            "show me correlation heatmap",
            "output distribution shift",
            "covariate drift alert",
        ],
    )
    def test_no_false_positives(self, phrase):
        assert not _FEATURE_PSI_PATTERNS.search(phrase), f"Should NOT match: {phrase!r}"


# ─── REST endpoint tests ──────────────────────────────────────────────────────


@pytest.fixture()
def _db_url(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_feat_psi.db",
        connect_args={"check_same_thread": False},
    )
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
    yield
    engine.dispose()


@pytest.mark.asyncio
async def test_feature_psi_404_missing_deployment(_db_url):
    """Returns 404 when deployment not found."""
    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/api/deploy/nonexistent-id/feature-psi")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_feature_psi_no_data_response(_db_url, tmp_path):
    """When a deployment exists but has no prediction logs, return no_data verdict."""
    from main import app
    from sqlmodel import Session
    from models.project import Project
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.deployment import Deployment

    with Session(db_module.engine) as session:
        proj = Project(owner_id="test-default-owner", name="psi-test", description="")
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
            target_column="revenue",
        )
        session.add(fs)
        session.commit()
        session.refresh(fs)

        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fs.id,
            algorithm="RandomForest",
            status="done",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        dep = Deployment(
            project_id=proj.id,
            model_run_id=run.id,
            endpoint_path=f"/api/predict/{run.id}",
            dashboard_url=f"/predict/{run.id}",
        )
        session.add(dep)
        session.commit()
        session.refresh(dep)
        dep_id = dep.id

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get(f"/api/deploy/{dep_id}/feature-psi")
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "no_data"
    assert data["sample_count"] == 0
