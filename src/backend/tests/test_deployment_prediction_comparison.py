"""Tests for Deployment Prediction Distribution Comparison.

Covers:
- compute_deployment_prediction_comparison pure function (regression + classification)
- _DEPLOY_PRED_DIST_COMPARE_PATTERNS regex (positive + negative)
- REST GET /api/deploy/{id}/prediction-distribution-comparison endpoint
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
    b"East,160.0,16\nWest,210.0,21\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.ab_test  # noqa
    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.deployment_preset  # noqa
    import models.deployment_version  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.goal_seek_record  # noqa
    import models.model_run  # noqa
    import models.prediction_alert_rule  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa
    import models.input_validation_rule  # noqa
    import models.dashboard_field_config  # noqa
    import models.deployment_changelog  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    from main import app

    with mock.patch("anthropic.Anthropic"):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


def _reg_logs(values: list[float]) -> list[dict]:
    """Build regression prediction log dicts."""
    return [{"prediction": str(v), "prediction_numeric": v, "confidence": None} for v in values]


def _cls_logs(labels: list[str]) -> list[dict]:
    """Build classification prediction log dicts."""
    return [{"prediction": json.dumps(lbl), "prediction_numeric": None, "confidence": 0.8} for lbl in labels]


class TestComputeDeploymentPredictionComparison:
    from core.analyzer import compute_deployment_prediction_comparison as _fn

    def setup_method(self):
        from core.analyzer import compute_deployment_prediction_comparison
        self.fn = compute_deployment_prediction_comparison

    def test_regression_required_keys(self):
        result = self.fn(_reg_logs([1.0] * 10), _reg_logs([2.0] * 10), "regression")
        for key in ("verdict", "verdict_label", "problem_type", "n_baseline", "n_current", "summary", "mean_shift", "mean_shift_pct", "baseline_stats", "current_stats"):
            assert key in result, f"Missing key: {key}"

    def test_regression_n_counts(self):
        result = self.fn(_reg_logs([1.0] * 7), _reg_logs([2.0] * 9), "regression")
        assert result["n_baseline"] == 7
        assert result["n_current"] == 9

    def test_regression_current_higher(self):
        baseline = _reg_logs([100.0] * 10)
        current = _reg_logs([120.0] * 10)
        result = self.fn(baseline, current, "regression")
        assert result["verdict"] == "current_higher"
        assert result["mean_shift_pct"] == pytest.approx(20.0, abs=0.1)

    def test_regression_current_lower(self):
        baseline = _reg_logs([100.0] * 10)
        current = _reg_logs([80.0] * 10)
        result = self.fn(baseline, current, "regression")
        assert result["verdict"] == "current_lower"
        assert result["mean_shift_pct"] == pytest.approx(-20.0, abs=0.1)

    def test_regression_similar(self):
        baseline = _reg_logs([100.0] * 10)
        current = _reg_logs([102.0] * 10)  # 2% shift — within threshold
        result = self.fn(baseline, current, "regression")
        assert result["verdict"] == "similar"

    def test_regression_mean_shift_direction(self):
        result = self.fn(_reg_logs([50.0] * 10), _reg_logs([60.0] * 10), "regression")
        assert result["mean_shift"] == pytest.approx(10.0, abs=0.01)

    def test_regression_baseline_stats_keys(self):
        result = self.fn(_reg_logs([1.0] * 10), _reg_logs([2.0] * 10), "regression")
        for key in ("mean", "median", "std", "min", "max", "p25", "p75", "n"):
            assert key in result["baseline_stats"], f"baseline_stats missing key: {key}"

    def test_regression_current_stats_keys(self):
        result = self.fn(_reg_logs([1.0] * 10), _reg_logs([2.0] * 10), "regression")
        for key in ("mean", "median", "std", "min", "max", "p25", "p75", "n"):
            assert key in result["current_stats"], f"current_stats missing key: {key}"

    def test_regression_stats_n_matches_input(self):
        result = self.fn(_reg_logs([1.0] * 8), _reg_logs([2.0] * 6), "regression")
        assert result["baseline_stats"]["n"] == 8
        assert result["current_stats"]["n"] == 6

    def test_regression_stats_mean_correct(self):
        result = self.fn(_reg_logs([10.0] * 10), _reg_logs([20.0] * 10), "regression")
        assert result["baseline_stats"]["mean"] == pytest.approx(10.0, abs=0.01)
        assert result["current_stats"]["mean"] == pytest.approx(20.0, abs=0.01)

    def test_regression_problem_type_field(self):
        result = self.fn(_reg_logs([1.0] * 10), _reg_logs([2.0] * 10), "regression")
        assert result["problem_type"] == "regression"

    def test_regression_summary_non_empty(self):
        result = self.fn(_reg_logs([1.0] * 10), _reg_logs([2.0] * 10), "regression")
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_regression_raises_too_few_baseline(self):
        with pytest.raises(ValueError, match="at least 5 baseline"):
            self.fn(_reg_logs([1.0] * 3), _reg_logs([2.0] * 10), "regression")

    def test_regression_raises_too_few_current(self):
        with pytest.raises(ValueError, match="at least 5 current"):
            self.fn(_reg_logs([1.0] * 10), _reg_logs([2.0] * 2), "regression")

    def test_classification_required_keys(self):
        result = self.fn(_cls_logs(["yes"] * 10), _cls_logs(["yes"] * 10), "classification")
        for key in ("verdict", "verdict_label", "problem_type", "n_baseline", "n_current", "summary", "class_shifts", "n_classes"):
            assert key in result, f"Missing key: {key}"

    def test_classification_class_shifts_structure(self):
        result = self.fn(_cls_logs(["yes"] * 10), _cls_logs(["no"] * 10), "classification")
        for shift in result["class_shifts"]:
            for key in ("class_label", "baseline_pct", "current_pct", "shift_pct"):
                assert key in shift

    def test_classification_distribution_shifted(self):
        baseline = _cls_logs(["yes"] * 5 + ["no"] * 5)
        current = _cls_logs(["yes"] * 2 + ["no"] * 8)
        result = self.fn(baseline, current, "classification")
        assert result["verdict"] == "distribution_shifted"

    def test_classification_similar_distribution(self):
        baseline = _cls_logs(["yes"] * 5 + ["no"] * 5)
        current = _cls_logs(["yes"] * 5 + ["no"] * 5)
        result = self.fn(baseline, current, "classification")
        assert result["verdict"] == "similar"

    def test_classification_n_classes_correct(self):
        result = self.fn(_cls_logs(["a", "b", "c"] * 4), _cls_logs(["a", "b", "c"] * 4), "classification")
        assert result["n_classes"] == 3

    def test_classification_shifts_sorted_by_absolute(self):
        baseline = _cls_logs(["yes"] * 9 + ["no"])
        current = _cls_logs(["no"] * 9 + ["yes"])
        result = self.fn(baseline, current, "classification")
        shifts = result["class_shifts"]
        abs_shifts = [abs(s["shift_pct"]) for s in shifts]
        assert abs_shifts == sorted(abs_shifts, reverse=True)

    def test_classification_problem_type_field(self):
        result = self.fn(_cls_logs(["a"] * 10), _cls_logs(["b"] * 10), "classification")
        assert result["problem_type"] == "classification"

    def test_regression_no_class_shifts_key(self):
        result = self.fn(_reg_logs([1.0] * 10), _reg_logs([2.0] * 10), "regression")
        assert "class_shifts" not in result


# ---------------------------------------------------------------------------
# Regex tests
# ---------------------------------------------------------------------------


@pytest.fixture
def pattern():
    from api.chat import _DEPLOY_PRED_DIST_COMPARE_PATTERNS
    return _DEPLOY_PRED_DIST_COMPARE_PATTERNS


class TestDeployPredDistComparePatterns:
    def test_current_higher_lower(self, pattern):
        assert pattern.search("Is my new deployment predicting higher values?")

    def test_compare_deployment_prediction_distributions(self, pattern):
        assert pattern.search("compare my deployment prediction distributions")

    def test_deployment_prediction_comparison(self, pattern):
        assert pattern.search("show deployment prediction comparison")

    def test_how_different_are_predictions(self, pattern):
        assert pattern.search("how different are my deployment predictions")

    def test_retrained_model_predicting_differently(self, pattern):
        assert pattern.search("has my retrained model changed its predictions")

    def test_old_vs_new_deployment_predictions(self, pattern):
        assert pattern.search("old vs new deployment predictions")

    def test_deployment_version_prediction_comparison(self, pattern):
        assert pattern.search("deployment version prediction comparison")

    def test_production_prediction_distribution_across_versions(self, pattern):
        assert pattern.search("production prediction distribution comparison across versions")

    def test_false_positive_general_compare(self, pattern):
        # "compare my features" should NOT match
        assert not pattern.search("compare my features to see which ones are important")

    def test_false_positive_training_metrics(self, pattern):
        # "compare my model versions metrics" — no "prediction/distribution" word
        assert not pattern.search("did my accuracy improve after retraining")


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


class TestPredictionDistributionComparisonEndpoint:
    def test_unknown_deployment_returns_404(self, client):
        resp = client.get(
            "/api/deploy/unknown-id/prediction-distribution-comparison?vs=other-id"
        )
        assert resp.status_code == 404

    def _setup_and_deploy(self, client) -> tuple[str, str]:
        """Create project → upload → apply → target → train (poll) → deploy."""
        proj = client.post("/api/projects", json={"name": "DistCompare Test"})
        assert proj.status_code == 201, proj.text
        project_id = proj.json()["id"]

        upload = client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
            data={"project_id": project_id},
        )
        assert upload.status_code == 201, upload.text
        dataset_id = upload.json()["dataset_id"]

        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id}/target",
            json={"target_column": "revenue", "problem_type": "regression"},
        )

        train = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        assert train.status_code == 202, train.text
        run_id = train.json()["model_run_ids"][0]

        for _ in range(60):
            runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
            run = next((r for r in runs if r["id"] == run_id), None)
            if run and run["status"] in ("done", "failed"):
                break
            time.sleep(0.2)

        assert run and run["status"] == "done", f"Training did not finish: {run}"

        deploy = client.post(f"/api/deploy/{run_id}")
        assert deploy.status_code == 201, deploy.text
        return deploy.json()["id"], run_id

    def test_no_data_when_insufficient_predictions(self, client):
        """Both deployments exist but neither has prediction logs → no_data."""
        dep1_id, run_id = self._setup_and_deploy(client)

        # Second deploy of the same run
        deploy2 = client.post(f"/api/deploy/{run_id}")
        assert deploy2.status_code == 201
        dep2_id = deploy2.json()["id"]

        # Neither deployment has any predictions → no_data
        resp = client.get(
            f"/api/deploy/{dep1_id}/prediction-distribution-comparison?vs={dep2_id}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "no_data"

    def test_unknown_baseline_returns_404(self, client):
        """Current deployment exists, baseline doesn't → 404."""
        dep_id, _ = self._setup_and_deploy(client)

        resp = client.get(
            f"/api/deploy/{dep_id}/prediction-distribution-comparison?vs=non-existent"
        )
        assert resp.status_code == 404
