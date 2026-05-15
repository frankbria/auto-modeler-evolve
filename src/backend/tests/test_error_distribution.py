"""Tests for compute_error_distribution() and GET /api/models/{run_id}/error-distribution."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REGRESSION_CSV = b"""feature1,feature2,target
1.0,2.0,5.0
2.0,3.0,8.0
3.0,4.0,11.0
4.0,5.0,14.0
5.0,6.0,17.0
6.0,7.0,20.0
7.0,8.0,23.0
8.0,9.0,26.0
9.0,10.0,29.0
10.0,11.0,32.0
11.0,12.0,35.0
12.0,13.0,38.0
15.0,16.0,47.0
20.0,21.0,62.0
25.0,26.0,77.0
"""

CLASSIFICATION_CSV = b"""age,income,churned
25,45000,yes
45,80000,no
30,35000,yes
55,120000,no
28,42000,yes
60,95000,no
35,55000,no
40,70000,yes
22,30000,yes
50,100000,no
33,48000,no
48,88000,no
27,38000,yes
52,110000,no
31,41000,yes
"""


# ---------------------------------------------------------------------------
# Unit tests: compute_error_distribution()
# ---------------------------------------------------------------------------


class TestComputeErrorDistributionRegression:
    def setup_method(self):
        from core.validator import compute_error_distribution

        self.fn = compute_error_distribution

    def test_returns_bins_and_stats(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        y_pred = np.array([1.1, 2.2, 2.8, 3.9, 5.1] * 10)
        result = self.fn(y_true, y_pred, "regression")
        assert "bins" in result
        assert len(result["bins"]) >= 5
        assert "stats" in result
        assert "summary" in result
        assert result["problem_type"] == "regression"

    def test_stats_contain_required_keys(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 4)
        y_pred = y_true + 0.1
        result = self.fn(y_true, y_pred, "regression")
        stats = result["stats"]
        for key in [
            "mean",
            "std",
            "mae",
            "bias_label",
            "bias_pct",
            "within_1std_pct",
            "total",
        ]:
            assert key in stats, f"missing key: {key}"

    def test_unbiased_model(self):
        rng = np.random.default_rng(42)
        y_true = np.arange(1.0, 101.0)
        noise = rng.normal(0, 0.5, 100)
        y_pred = y_true + noise
        result = self.fn(y_true, y_pred, "regression")
        # Mean residual should be near 0 → unbiased
        assert result["stats"]["bias_label"] == "unbiased"

    def test_over_predicting_model(self):
        y_true = np.ones(50)
        y_pred = (
            np.ones(50) * 1.5
        )  # always predicts 0.5 too high → under-predicts (residual negative)
        result = self.fn(y_true, y_pred, "regression")
        # residual = actual - pred = 1 - 1.5 = -0.5 → mean < 0 → under-predicts
        assert result["stats"]["bias_label"] == "under-predicts"

    def test_bins_sum_to_total(self):
        y_true = np.arange(1.0, 51.0)
        y_pred = y_true + np.random.default_rng(1).normal(0, 2, 50)
        result = self.fn(y_true, y_pred, "regression")
        total_from_bins = sum(b["count"] for b in result["bins"])
        assert total_from_bins == len(y_true)

    def test_empty_array_returns_gracefully(self):
        result = self.fn(np.array([]), np.array([]), "regression")
        assert result["bins"] == []
        assert "No training data" in result["summary"]

    def test_n_bins_clamped_minimum(self):
        y_true = np.arange(1.0, 101.0)
        y_pred = y_true
        result = self.fn(y_true, y_pred, "regression", n_bins=2)
        assert len(result["bins"]) >= 5

    def test_n_bins_clamped_maximum(self):
        y_true = np.arange(1.0, 101.0)
        y_pred = y_true
        result = self.fn(y_true, y_pred, "regression", n_bins=50)
        assert len(result["bins"]) <= 30

    def test_summary_contains_mae(self):
        y_true = np.array([1.0, 2.0, 3.0] * 5)
        y_pred = y_true + 1.0
        result = self.fn(y_true, y_pred, "regression")
        assert "MAE" in result["summary"]

    def test_within_1std_pct_is_valid_percentage(self):
        y_true = np.arange(1.0, 101.0)
        y_pred = y_true + np.random.default_rng(99).normal(0, 1, 100)
        result = self.fn(y_true, y_pred, "regression")
        pct = result["stats"]["within_1std_pct"]
        assert 0 <= pct <= 100


class TestComputeErrorDistributionClassification:
    def setup_method(self):
        from core.validator import compute_error_distribution

        self.fn = compute_error_distribution

    def test_returns_class_breakdown(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        result = self.fn(y_true, y_pred, "classification")
        assert "class_breakdown" in result
        assert len(result["class_breakdown"]) == 2

    def test_class_breakdown_keys(self):
        y_true = np.array([0, 1, 2, 0, 1, 2] * 3)
        y_pred = y_true.copy()
        y_pred[0] = 1  # one error
        result = self.fn(y_true, y_pred, "classification")
        row = result["class_breakdown"][0]
        for key in ["class", "total", "wrong", "error_rate", "error_pct"]:
            assert key in row, f"missing key: {key}"

    def test_sorted_highest_error_first(self):
        # class 0: perfect; class 1: all wrong
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        result = self.fn(y_true, y_pred, "classification")
        breakdown = result["class_breakdown"]
        assert breakdown[0]["error_rate"] >= breakdown[-1]["error_rate"]

    def test_perfect_classifier(self):
        y_true = np.array([0, 1, 2] * 5)
        y_pred = y_true.copy()
        result = self.fn(y_true, y_pred, "classification")
        for row in result["class_breakdown"]:
            assert row["error_rate"] == 0.0

    def test_class_labels_decoded(self):
        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_pred = np.array([0.0, 0.0, 1.0, 1.0])
        result = self.fn(y_true, y_pred, "classification", target_classes=["no", "yes"])
        classes = {r["class"] for r in result["class_breakdown"]}
        assert "yes" in classes
        assert "no" in classes

    def test_summary_contains_accuracy(self):
        y_true = np.array([0, 0, 1, 1] * 5)
        y_pred = y_true.copy()
        result = self.fn(y_true, y_pred, "classification")
        assert "accurate" in result["summary"].lower() or "100%" in result["summary"]

    def test_stats_contain_required_keys(self):
        y_true = np.array([0, 1] * 10)
        y_pred = y_true
        result = self.fn(y_true, y_pred, "classification")
        stats = result["stats"]
        for key in [
            "total",
            "total_wrong",
            "overall_error_rate",
            "overall_accuracy",
            "n_classes",
        ]:
            assert key in stats, f"missing key: {key}"

    def test_empty_classification_returns_gracefully(self):
        result = self.fn(np.array([]), np.array([]), "classification")
        assert "No training data" in result["summary"]


# ---------------------------------------------------------------------------
# Integration tests: GET /api/models/{run_id}/error-distribution
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
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa
    import models.webhook_event  # noqa
    import models.analysis_template  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _train_regression_model(client):
    import io
    import time

    proj = client.post("/api/projects", json={"name": "ErrDistTest"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload_resp = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(REGRESSION_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "target", "problem_type": "regression"},
    )

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert train_resp.status_code == 202
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(60):
        runs = client.get(f"/api/models/{project_id}/runs").json().get("runs", [])
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    assert run and run["status"] == "done", "training did not complete"
    return run_id


def test_endpoint_returns_bins_and_stats(client):
    run_id = _train_regression_model(client)
    resp = client.get(f"/api/models/{run_id}/error-distribution")
    assert resp.status_code == 200
    data = resp.json()
    assert "bins" in data
    assert "stats" in data
    assert "summary" in data
    assert data["problem_type"] == "regression"


def test_endpoint_404_on_unknown_run(client):
    resp = client.get("/api/models/nonexistent-run-id/error-distribution")
    assert resp.status_code == 404


def test_endpoint_includes_algorithm_and_target(client):
    run_id = _train_regression_model(client)
    resp = client.get(f"/api/models/{run_id}/error-distribution")
    data = resp.json()
    assert "algorithm" in data
    assert "target_col" in data
    assert data["target_col"] == "target"


# ---------------------------------------------------------------------------
# Chat pattern tests: _ERROR_DIST_PATTERNS
# ---------------------------------------------------------------------------


class TestErrorDistPatterns:
    def setup_method(self):
        from api.chat import _ERROR_DIST_PATTERNS

        self.re = _ERROR_DIST_PATTERNS

    def _match(self, text: str) -> bool:
        return bool(self.re.search(text))

    def test_error_distribution(self):
        assert self._match("show me the error distribution")

    def test_residual_histogram(self):
        assert self._match("residual histogram")

    def test_residual_distribution(self):
        assert self._match("what's the residual distribution?")

    def test_how_errors_distributed(self):
        assert self._match("how are my errors distributed?")

    def test_histogram_of_errors(self):
        assert self._match("histogram of prediction errors")

    def test_where_model_struggles(self):
        assert self._match("where does my model struggle?")

    def test_where_model_fails(self):
        assert self._match("where does my model fail?")

    def test_per_class_error_rate(self):
        assert self._match("per class error rate")

    def test_class_error_breakdown(self):
        assert self._match("class error breakdown")

    def test_misclassification_rate_by_class(self):
        assert self._match("misclassification rate by class")

    def test_which_class_struggles(self):
        assert self._match("which classes does my model struggle with?")

    def test_no_match_generic_error_question(self):
        # "show errors" belongs to _PRED_ERROR_PATTERNS not _ERROR_DIST_PATTERNS
        assert not self._match("show the worst errors")
        assert not self._match("show mistakes")

    def test_no_match_unrelated(self):
        assert not self._match("train a model")
        assert not self._match("upload data")
