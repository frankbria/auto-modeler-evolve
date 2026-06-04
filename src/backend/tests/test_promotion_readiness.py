"""Tests for the Model Promotion Readiness Check feature.

Covers:
- compute_promotion_readiness() pure function: all gates, verdicts, edge cases
- GET /api/models/{run_id}/promotion-readiness REST endpoint
- _PROMOTION_READINESS_PATTERNS regex: positive matches and false-positive guards
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.advisor import compute_promotion_readiness


@pytest.fixture()
def _db_url(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_promo_ready.db",
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
    import models.weekly_digest_config  # noqa: F401

    SQLModel.metadata.create_all(engine)
    db_module.engine = engine
    yield
    engine.dispose()


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestPromotionReadinessGates:
    """Test each gate individually."""

    BASE_METRICS_GOOD_REG = {
        "r2": 0.85,
        "train_r2": 0.87,
        "cv_mean": 0.84,
        "cv_std": 0.03,
        "brier_score": None,
    }

    BASE_METRICS_GOOD_CLS = {
        "accuracy": 0.90,
        "train_accuracy": 0.91,
        "cv_mean": 0.89,
        "cv_std": 0.03,
        "brier_score": 0.10,
    }

    def test_all_gates_pass_regression(self):
        result = compute_promotion_readiness(
            metrics=self.BASE_METRICS_GOOD_REG,
            algorithm="random_forest_regressor",
            problem_type="regression",
            n_rows=500,
            n_features=10,
        )
        assert result["overall_verdict"] == "ready"
        assert result["verdict_label"] == "Ready to Deploy"
        assert result["fail_count"] == 0
        assert result["warn_count"] == 0
        assert result["passed_count"] >= 4

    def test_all_gates_pass_classification(self):
        result = compute_promotion_readiness(
            metrics=self.BASE_METRICS_GOOD_CLS,
            algorithm="random_forest_classifier",
            problem_type="classification",
            n_rows=500,
            n_features=10,
        )
        assert result["overall_verdict"] == "ready"
        assert result["fail_count"] == 0

    def test_low_r2_fails_quality_gate(self):
        metrics = {**self.BASE_METRICS_GOOD_REG, "r2": 0.50, "train_r2": 0.52}
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=500,
            n_features=10,
        )
        assert result["overall_verdict"] == "not_ready"
        assert result["fail_count"] >= 1
        quality_gate = next(g for g in result["gates"] if g["gate"] == "model_quality")
        assert quality_gate["status"] == "fail"
        assert "R²" in quality_gate["detail"]

    def test_borderline_r2_warns_quality_gate(self):
        metrics = {
            **self.BASE_METRICS_GOOD_REG,
            "r2": 0.70,
            "train_r2": 0.72,
            "cv_mean": 0.69,
        }
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=500,
            n_features=10,
        )
        quality_gate = next(g for g in result["gates"] if g["gate"] == "model_quality")
        assert quality_gate["status"] == "warn"

    def test_low_accuracy_fails_quality_gate(self):
        metrics = {
            **self.BASE_METRICS_GOOD_CLS,
            "accuracy": 0.60,
            "train_accuracy": 0.62,
            "cv_mean": 0.59,
        }
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="logistic_regression",
            problem_type="classification",
            n_rows=500,
            n_features=10,
        )
        quality_gate = next(g for g in result["gates"] if g["gate"] == "model_quality")
        assert quality_gate["status"] == "fail"
        assert result["overall_verdict"] == "not_ready"

    def test_high_cv_std_fails_stability_gate(self):
        metrics = {**self.BASE_METRICS_GOOD_REG, "cv_std": 0.15}
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="random_forest_regressor",
            problem_type="regression",
            n_rows=500,
            n_features=10,
        )
        cv_gate = next(g for g in result["gates"] if g["gate"] == "cv_stability")
        assert cv_gate["status"] == "fail"

    def test_moderate_cv_std_warns_stability_gate(self):
        metrics = {**self.BASE_METRICS_GOOD_REG, "cv_std": 0.07}
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="random_forest_regressor",
            problem_type="regression",
            n_rows=500,
            n_features=10,
        )
        cv_gate = next(g for g in result["gates"] if g["gate"] == "cv_stability")
        assert cv_gate["status"] == "warn"

    def test_missing_cv_std_skips_stability_gate(self):
        metrics = {"r2": 0.85}
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=500,
            n_features=10,
        )
        gate_names = [g["gate"] for g in result["gates"]]
        assert "cv_stability" not in gate_names

    def test_large_train_cv_gap_fails_overfitting_gate(self):
        metrics = {**self.BASE_METRICS_GOOD_REG, "train_r2": 0.99, "cv_mean": 0.70}
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="random_forest_regressor",
            problem_type="regression",
            n_rows=500,
            n_features=10,
        )
        of_gate = next(g for g in result["gates"] if g["gate"] == "overfitting")
        assert of_gate["status"] == "fail"
        assert result["overall_verdict"] == "not_ready"

    def test_small_train_cv_gap_passes_overfitting_gate(self):
        metrics = {**self.BASE_METRICS_GOOD_REG, "train_r2": 0.87, "cv_mean": 0.84}
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="random_forest_regressor",
            problem_type="regression",
            n_rows=500,
            n_features=10,
        )
        of_gate = next(g for g in result["gates"] if g["gate"] == "overfitting")
        assert of_gate["status"] == "pass"

    def test_poor_calibration_fails_brier_gate(self):
        metrics = {**self.BASE_METRICS_GOOD_CLS, "brier_score": 0.30}
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="random_forest_classifier",
            problem_type="classification",
            n_rows=500,
            n_features=10,
        )
        cal_gate = next(g for g in result["gates"] if g["gate"] == "calibration")
        assert cal_gate["status"] == "fail"

    def test_good_calibration_passes_brier_gate(self):
        metrics = {**self.BASE_METRICS_GOOD_CLS, "brier_score": 0.08}
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="random_forest_classifier",
            problem_type="classification",
            n_rows=500,
            n_features=10,
        )
        cal_gate = next(g for g in result["gates"] if g["gate"] == "calibration")
        assert cal_gate["status"] == "pass"

    def test_no_brier_score_skips_calibration_gate_regression(self):
        result = compute_promotion_readiness(
            metrics=self.BASE_METRICS_GOOD_REG,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=500,
            n_features=10,
        )
        gate_names = [g["gate"] for g in result["gates"]]
        assert "calibration" not in gate_names

    def test_too_few_rows_fails_data_volume_gate(self):
        result = compute_promotion_readiness(
            metrics=self.BASE_METRICS_GOOD_REG,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=30,
            n_features=5,
        )
        vol_gate = next(g for g in result["gates"] if g["gate"] == "data_volume")
        assert vol_gate["status"] == "fail"

    def test_borderline_rows_warns_data_volume_gate(self):
        result = compute_promotion_readiness(
            metrics=self.BASE_METRICS_GOOD_REG,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=100,
            n_features=5,
        )
        vol_gate = next(g for g in result["gates"] if g["gate"] == "data_volume")
        assert vol_gate["status"] == "warn"

    def test_adequate_rows_passes_data_volume_gate(self):
        result = compute_promotion_readiness(
            metrics=self.BASE_METRICS_GOOD_REG,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=500,
            n_features=5,
        )
        vol_gate = next(g for g in result["gates"] if g["gate"] == "data_volume")
        assert vol_gate["status"] == "pass"

    def test_low_rpf_fails_sample_feature_ratio_gate(self):
        result = compute_promotion_readiness(
            metrics=self.BASE_METRICS_GOOD_REG,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=50,
            n_features=10,
        )
        rpf_gate = next(
            g for g in result["gates"] if g["gate"] == "sample_feature_ratio"
        )
        assert rpf_gate["status"] == "fail"

    def test_good_rpf_passes_sample_feature_ratio_gate(self):
        result = compute_promotion_readiness(
            metrics=self.BASE_METRICS_GOOD_REG,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=500,
            n_features=10,
        )
        rpf_gate = next(
            g for g in result["gates"] if g["gate"] == "sample_feature_ratio"
        )
        assert rpf_gate["status"] == "pass"


class TestPromotionReadinessVerdicts:
    """Test overall verdict logic."""

    def test_ready_with_warnings_when_warn_but_no_fail(self):
        metrics = {
            "r2": 0.82,
            "train_r2": 0.85,
            "cv_mean": 0.82,
            "cv_std": 0.08,  # warn
        }
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="random_forest_regressor",
            problem_type="regression",
            n_rows=300,  # pass
            n_features=10,
        )
        assert result["overall_verdict"] == "ready_with_warnings"
        assert result["verdict_label"] == "Deploy with Caution"
        assert result["warn_count"] >= 1
        assert result["fail_count"] == 0

    def test_not_ready_when_any_fail(self):
        metrics = {
            "r2": 0.40,  # fail
            "train_r2": 0.42,
            "cv_mean": 0.40,
            "cv_std": 0.02,
        }
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=300,
            n_features=10,
        )
        assert result["overall_verdict"] == "not_ready"
        assert result["fail_count"] >= 1

    def test_blocking_issues_list_contains_failed_gate_labels(self):
        metrics = {"r2": 0.30, "train_r2": 0.35, "cv_mean": 0.30}
        result = compute_promotion_readiness(
            metrics=metrics,
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=20,
            n_features=10,
        )
        assert len(result["blocking_issues"]) >= 1
        assert all(isinstance(b, str) for b in result["blocking_issues"])

    def test_result_has_required_keys(self):
        result = compute_promotion_readiness(
            metrics={"r2": 0.85, "cv_mean": 0.84, "cv_std": 0.03, "train_r2": 0.86},
            algorithm="random_forest_regressor",
            problem_type="regression",
            n_rows=300,
            n_features=10,
        )
        required = [
            "overall_verdict",
            "verdict_label",
            "verdict_color",
            "passed_count",
            "warn_count",
            "fail_count",
            "total_count",
            "gates",
            "blocking_issues",
            "warnings",
            "algorithm",
            "problem_type",
            "primary_metric",
            "primary_metric_name",
            "n_rows",
            "n_features",
            "summary",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_each_gate_has_required_fields(self):
        result = compute_promotion_readiness(
            metrics={"r2": 0.85, "cv_mean": 0.83, "cv_std": 0.04, "train_r2": 0.86},
            algorithm="random_forest_regressor",
            problem_type="regression",
            n_rows=300,
            n_features=10,
        )
        for gate in result["gates"]:
            assert "gate" in gate
            assert "label" in gate
            assert "status" in gate
            assert gate["status"] in ("pass", "warn", "fail")
            assert "detail" in gate
            assert "recommendation" in gate

    def test_summary_is_nonempty_string(self):
        result = compute_promotion_readiness(
            metrics={"r2": 0.85},
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=200,
            n_features=5,
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_n_rows_and_n_features_in_result(self):
        result = compute_promotion_readiness(
            metrics={"r2": 0.85},
            algorithm="linear_regression",
            problem_type="regression",
            n_rows=250,
            n_features=8,
        )
        assert result["n_rows"] == 250
        assert result["n_features"] == 8


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------

SAMPLE_CSV = b"""revenue,units,region
100,10,East
200,20,West
150,15,North
300,30,South
250,25,East
180,18,West
120,12,North
220,22,South
170,17,East
280,28,West
140,14,North
310,31,South
190,19,East
160,16,West
240,24,North
"""

SAMPLE_CSV_BIG = b"revenue,units,region\n" + b"".join(
    f"{50 + (i % 300)},{5 + (i % 30)},{'East' if i % 3 == 0 else 'West'}\n".encode()
    for i in range(300)
)


@pytest.mark.asyncio
async def test_promotion_readiness_404_unknown_run(_db_url):
    """Returns 404 for a nonexistent run ID."""
    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/api/models/nonexistent-id/promotion-readiness")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_promotion_readiness_400_pending_run(_db_url, tmp_path):
    """Returns 400 when model run is not done yet."""
    import db as db_module
    from main import app
    from sqlmodel import Session
    from models.project import Project
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun

    csv_path = tmp_path / "data.csv"
    csv_path.write_bytes(SAMPLE_CSV)

    with Session(db_module.engine) as session:
        proj = Project(name="promo-test-pending", description="")
        session.add(proj)
        session.commit()
        session.refresh(proj)

        ds = Dataset(
            project_id=proj.id,
            filename="data.csv",
            file_path=str(csv_path),
            row_count=15,
            column_count=3,
        )
        session.add(ds)
        session.commit()
        session.refresh(ds)

        fs = FeatureSet(
            dataset_id=ds.id,
            target_column="revenue",
            problem_type="regression",
            transformations="[]",
            is_active=True,
        )
        session.add(fs)
        session.commit()
        session.refresh(fs)

        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fs.id,
            algorithm="linear_regression",
            status="pending",
            metrics="{}",
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get(f"/api/models/{run_id}/promotion-readiness")
    assert resp.status_code == 400
    assert "not done" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_promotion_readiness_done_run(_db_url, tmp_path):
    """Returns verdict dict for a completed run."""
    import db as db_module
    from main import app
    from sqlmodel import Session
    from models.project import Project
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    import json as _json

    csv_path = tmp_path / "data.csv"
    csv_path.write_bytes(SAMPLE_CSV_BIG)

    metrics = _json.dumps(
        {
            "r2": 0.85,
            "train_r2": 0.86,
            "cv_mean": 0.84,
            "cv_std": 0.03,
        }
    )

    with Session(db_module.engine) as session:
        proj = Project(name="promo-done-test", description="")
        session.add(proj)
        session.commit()
        session.refresh(proj)

        ds = Dataset(
            project_id=proj.id,
            filename="data.csv",
            file_path=str(csv_path),
            row_count=300,
            column_count=3,
        )
        session.add(ds)
        session.commit()
        session.refresh(ds)

        fs = FeatureSet(
            dataset_id=ds.id,
            target_column="revenue",
            problem_type="regression",
            transformations="[]",
            is_active=True,
        )
        session.add(fs)
        session.commit()
        session.refresh(fs)

        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fs.id,
            algorithm="linear_regression",
            status="done",
            metrics=metrics,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get(f"/api/models/{run_id}/promotion-readiness")
    assert resp.status_code == 200
    data = resp.json()
    assert "overall_verdict" in data
    assert data["overall_verdict"] in ("ready", "ready_with_warnings", "not_ready")
    assert "gates" in data
    assert isinstance(data["gates"], list)
    assert len(data["gates"]) >= 2
    assert "run_id" in data
    assert data["run_id"] == run_id


# ---------------------------------------------------------------------------
# Chat regex tests
# ---------------------------------------------------------------------------


class TestPromotionReadinessPatterns:
    """Validate _PROMOTION_READINESS_PATTERNS triggers and false-positive guards."""

    @pytest.fixture(autouse=True)
    def pattern(self):
        from api.chat import _PROMOTION_READINESS_PATTERNS

        self.pat = _PROMOTION_READINESS_PATTERNS

    def _match(self, text):
        return bool(self.pat.search(text))

    # Positive matches
    def test_promotion_readiness(self):
        assert self._match("promotion readiness check")

    def test_ready_to_promote(self):
        assert self._match("Is my model ready to promote to production?")

    def test_promote_model(self):
        assert self._match("promote my model to production")

    def test_pre_deployment_checklist(self):
        assert self._match("run the pre-deployment checklist")

    def test_deployment_gate(self):
        assert self._match("deployment gate check")

    def test_deployment_checklist(self):
        assert self._match("deployment checklist for my model")

    def test_go_no_go(self):
        assert self._match("go/no-go decision")

    def test_production_readiness_check(self):
        assert self._match("production readiness check")

    def test_all_checks_pass(self):
        assert self._match("do all checks pass?")

    def test_run_a_readiness_check(self):
        assert self._match("run a readiness check")

    # False-positive guards
    def test_no_match_generic_ready(self):
        assert not self._match("is my model ready?")

    def test_no_match_deploy_only(self):
        assert not self._match("deploy my model now")

    def test_no_match_data_readiness(self):
        assert not self._match("is my data ready?")

    def test_no_match_retrain(self):
        assert not self._match("should I retrain my model?")
