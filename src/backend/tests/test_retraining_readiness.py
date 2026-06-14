"""Tests for compute_retraining_readiness() pure function, REST endpoint, and chat regex."""

from __future__ import annotations

import json
import uuid

import joblib
import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.linear_model import LinearRegression
from sqlmodel import Session, SQLModel, create_engine

import db as db_module
from api.chat import _RETRAIN_READINESS_PATTERNS
from core.analyzer import compute_retraining_readiness

# ---------------------------------------------------------------------------
# 1. Pure function tests
# ---------------------------------------------------------------------------


class TestComputeRetrainingReadiness:
    def test_returns_required_keys(self):
        result = compute_retraining_readiness(age_days=30)
        for key in [
            "score",
            "verdict",
            "verdict_label",
            "signals",
            "signals_available",
            "signals_firing",
            "top_reason",
            "recommendations",
            "summary",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_no_signals_returns_stable_with_guidance(self):
        result = compute_retraining_readiness()
        assert result["score"] == 0
        assert result["verdict"] == "stable"
        assert result["signals_available"] == 0
        assert result["signals_firing"] == 0
        assert "No monitoring signals" in result["summary"]

    def test_fresh_model_is_stable(self):
        result = compute_retraining_readiness(age_days=5)
        assert result["score"] == 0
        assert result["verdict"] == "stable"
        assert result["signals_firing"] == 0

    def test_very_old_model_contributes_age_signal(self):
        result = compute_retraining_readiness(age_days=200)
        age_signal = next(s for s in result["signals"] if s["name"] == "model_age")
        assert age_signal["is_firing"]
        assert age_signal["score_contribution"] == 40

    def test_aging_model_contributes_partial_age_signal(self):
        result = compute_retraining_readiness(age_days=65)
        age_signal = next(s for s in result["signals"] if s["name"] == "model_age")
        assert age_signal["is_firing"]
        assert age_signal["score_contribution"] == 15

    def test_high_anomaly_rate_fires_signal(self):
        result = compute_retraining_readiness(anomaly_rate=0.20)
        anom_signal = next(s for s in result["signals"] if s["name"] == "anomaly_rate")
        assert anom_signal["is_firing"]
        assert anom_signal["score_contribution"] == 30

    def test_low_anomaly_rate_does_not_fire(self):
        result = compute_retraining_readiness(anomaly_rate=0.02)
        anom_signal = next(s for s in result["signals"] if s["name"] == "anomaly_rate")
        assert not anom_signal["is_firing"]
        assert anom_signal["score_contribution"] == 0

    def test_declining_confidence_fires_signal(self):
        result = compute_retraining_readiness(confidence_trend="declining")
        conf_signal = next(
            s for s in result["signals"] if s["name"] == "confidence_trend"
        )
        assert conf_signal["is_firing"]
        assert conf_signal["score_contribution"] == 20

    def test_stable_confidence_does_not_fire(self):
        result = compute_retraining_readiness(confidence_trend="stable")
        conf_signal = next(
            s for s in result["signals"] if s["name"] == "confidence_trend"
        )
        assert not conf_signal["is_firing"]
        assert conf_signal["score_contribution"] == 0

    def test_poor_feedback_fires_signal(self):
        result = compute_retraining_readiness(feedback_verdict="poor")
        fb_signal = next(
            s for s in result["signals"] if s["name"] == "feedback_accuracy"
        )
        assert fb_signal["is_firing"]
        assert fb_signal["score_contribution"] == 30

    def test_good_feedback_does_not_fire(self):
        result = compute_retraining_readiness(feedback_verdict="good")
        fb_signal = next(
            s for s in result["signals"] if s["name"] == "feedback_accuracy"
        )
        assert not fb_signal["is_firing"]
        assert fb_signal["score_contribution"] == 0

    def test_psi_critical_fires_signal(self):
        result = compute_retraining_readiness(psi_critical_count=2)
        psi_signal = next(s for s in result["signals"] if s["name"] == "psi_drift")
        assert psi_signal["is_firing"]
        assert psi_signal["score_contribution"] == 25

    def test_no_psi_critical_does_not_fire(self):
        result = compute_retraining_readiness(psi_critical_count=0)
        psi_signal = next(s for s in result["signals"] if s["name"] == "psi_drift")
        assert not psi_signal["is_firing"]

    def test_significant_output_shift_fires(self):
        result = compute_retraining_readiness(output_shift_verdict="significant_shift")
        shift_signal = next(s for s in result["signals"] if s["name"] == "output_shift")
        assert shift_signal["is_firing"]
        assert shift_signal["score_contribution"] == 25

    def test_stable_output_shift_does_not_fire(self):
        result = compute_retraining_readiness(output_shift_verdict="stable")
        shift_signal = next(s for s in result["signals"] if s["name"] == "output_shift")
        assert not shift_signal["is_firing"]

    def test_no_data_output_shift_excluded(self):
        result = compute_retraining_readiness(output_shift_verdict="no_data")
        names = [s["name"] for s in result["signals"]]
        assert "output_shift" not in names

    def test_signals_available_count(self):
        result = compute_retraining_readiness(
            age_days=30, anomaly_rate=0.10, confidence_trend="stable"
        )
        assert result["signals_available"] == 3

    def test_signals_firing_count(self):
        result = compute_retraining_readiness(
            age_days=100, anomaly_rate=0.20, confidence_trend="stable"
        )
        assert result["signals_firing"] == 2  # age + anomaly fire, stable conf doesn't

    def test_score_is_0_to_100(self):
        result = compute_retraining_readiness(
            age_days=200,
            anomaly_rate=0.20,
            confidence_trend="declining",
            feedback_verdict="poor",
        )
        assert 0 <= result["score"] <= 100

    def test_multiple_alarming_signals_give_high_score(self):
        result = compute_retraining_readiness(
            age_days=200,
            anomaly_rate=0.20,
            confidence_trend="declining",
            feedback_verdict="poor",
        )
        assert result["score"] >= 60

    def test_retrain_now_verdict_when_score_high(self):
        result = compute_retraining_readiness(
            age_days=200,
            anomaly_rate=0.20,
            confidence_trend="declining",
            feedback_verdict="poor",
            psi_critical_count=2,
        )
        assert result["verdict"] in ("retrain_now", "retrain_soon")

    def test_stable_verdict_for_all_green(self):
        result = compute_retraining_readiness(
            age_days=10,
            anomaly_rate=0.01,
            confidence_trend="stable",
            feedback_verdict="good",
        )
        assert result["verdict"] == "stable"

    def test_recommendations_not_empty(self):
        result = compute_retraining_readiness(age_days=100, anomaly_rate=0.20)
        assert len(result["recommendations"]) > 0

    def test_top_reason_is_none_when_nothing_fires(self):
        result = compute_retraining_readiness(age_days=10)
        assert result["top_reason"] is None

    def test_top_reason_is_string_when_signal_fires(self):
        result = compute_retraining_readiness(age_days=200)
        assert isinstance(result["top_reason"], str)

    def test_summary_is_string(self):
        result = compute_retraining_readiness(age_days=90)
        assert isinstance(result["summary"], str) and len(result["summary"]) > 0

    def test_verdict_label_matches_verdict(self):
        result = compute_retraining_readiness(age_days=10)
        assert result["verdict_label"] in [
            "No action needed",
            "Worth watching",
            "Retrain recommended",
            "Retrain urgently",
        ]

    def test_partial_signals_normalized(self):
        # With only age_days, score should still be 0-100
        result = compute_retraining_readiness(age_days=200)
        assert 0 <= result["score"] <= 100
        assert result["signals_available"] == 1


# ---------------------------------------------------------------------------
# 2. Chat regex tests
# ---------------------------------------------------------------------------


SHOULD_MATCH = [
    "should I retrain my model?",
    "Should I retrain my predictor?",
    "is it time to retrain?",
    "retraining readiness",
    "retrain recommendation",
    "retraining readiness check",
    "do I need to retrain?",
    "is my model degrading?",
    "model is getting worse",
    "comprehensive model health check",
    "model degradation score",
    "aggregate monitoring signals",
    "retraining assessment",
]

SHOULD_NOT_MATCH = [
    "show me the error distribution",
    "what is my model's accuracy?",
    "feature importance",
    "train a model to predict revenue",
    "how many predictions have been made?",
]


class TestRetrainReadinessPatterns:
    @pytest.mark.parametrize("msg", SHOULD_MATCH)
    def test_positive(self, msg):
        assert _RETRAIN_READINESS_PATTERNS.search(msg), f"Should match: {msg!r}"

    @pytest.mark.parametrize("msg", SHOULD_NOT_MATCH)
    def test_negative(self, msg):
        assert not _RETRAIN_READINESS_PATTERNS.search(msg), f"Should NOT match: {msg!r}"


# ---------------------------------------------------------------------------
# 3. REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)

    import models.conversation  # noqa: F401
    import models.dataset  # noqa: F401
    import models.deployment  # noqa: F401
    import models.feature_set  # noqa: F401
    import models.model_run  # noqa: F401
    import models.project  # noqa: F401

    SQLModel.metadata.create_all(db_module.engine)

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.anyio
async def test_retraining_readiness_404_unknown(ac):
    resp = await ac.get(
        "/api/deploy/00000000-0000-0000-0000-000000000000/retraining-readiness"
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_retraining_readiness_200_with_deployment(ac, tmp_path):
    """Full end-to-end: create project → dataset → feature set → model run → deployment → call endpoint."""
    from models.dataset import Dataset
    from models.deployment import Deployment
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project

    # minimal CSV
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1.0,2.0\n2.0,4.0\n3.0,6.0\n4.0,8.0\n5.0,10.0\n")

    # joblib model
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    pipeline_path = model_dir / "pipeline.joblib"
    model_path = model_dir / "model.joblib"
    from core.deployer import PredictionPipeline

    pp = PredictionPipeline(
        feature_names=["x"], feature_means={"x": 3.0}, column_types={"x": "numeric"}
    )
    joblib.dump(pp, pipeline_path)
    reg = LinearRegression().fit([[1], [2], [3], [4], [5]], [2, 4, 6, 8, 10])
    joblib.dump(reg, model_path)

    with Session(db_module.engine) as sess:
        proj = Project(owner_id="test-default-owner", name="Test")
        sess.add(proj)
        sess.flush()

        ds = Dataset(
            project_id=proj.id,
            filename="data.csv",
            file_path=str(csv_path),
            row_count=5,
            column_count=2,
        )
        sess.add(ds)
        sess.flush()

        fs = FeatureSet(
            dataset_id=ds.id,
            target_column="y",
            problem_type="regression",
            transformations="[]",
        )
        sess.add(fs)
        sess.flush()

        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fs.id,
            algorithm="linear_regression",
            status="done",
            metrics=json.dumps({"r2": 0.95, "mae": 0.1}),
            model_path=str(model_path),
            pipeline_path=str(pipeline_path),
        )
        sess.add(run)
        sess.flush()

        dep = Deployment(
            model_run_id=run.id,
            project_id=proj.id,
            endpoint_path=f"/api/predict/{uuid.uuid4()}",
            dashboard_url="/predict/test",
            is_active=True,
            problem_type="regression",
            target_column="y",
            algorithm=run.algorithm,
            pipeline_path=str(pipeline_path),
        )
        sess.add(dep)
        sess.commit()

        dep_id = dep.id

    resp = await ac.get(f"/api/deploy/{dep_id}/retraining-readiness")
    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.json()}"
    data = resp.json()
    assert "score" in data
    assert "verdict" in data
    assert "signals" in data
    assert data["deployment_id"] == dep_id
    assert 0 <= data["score"] <= 100
