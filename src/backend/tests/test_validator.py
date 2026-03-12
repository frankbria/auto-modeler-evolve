"""Tests for core/validator.py and core/explainer.py — Phase 5 validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.trainer import train_model
from core.validator import validate_model, _assess_limitations, _build_ci_metrics
from core.explainer import compute_global_importance, explain_prediction


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_df():
    rng = np.random.default_rng(42)
    n = 80
    x1 = rng.normal(50, 15, n)
    x2 = rng.choice(["A", "B", "C"], n)
    x3 = rng.normal(10, 3, n)
    y = 2 * x1 + rng.normal(0, 5, n)
    return pd.DataFrame({"feature1": x1, "category": x2, "feature3": x3, "target": y})


@pytest.fixture
def classification_df():
    rng = np.random.default_rng(7)
    n = 80
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    labels = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"feature1": x1, "feature2": x2, "label": labels})


@pytest.fixture
def trained_regression_model(regression_df, tmp_path):
    result = train_model(
        df=regression_df,
        target_col="target",
        problem_type="regression",
        algorithm="ridge",
        hyperparameters={},
        model_dir=tmp_path,
    )
    return result["model_path"]


@pytest.fixture
def trained_classification_model(classification_df, tmp_path):
    result = train_model(
        df=classification_df,
        target_col="label",
        problem_type="classification",
        algorithm="random_forest",
        hyperparameters={},
        model_dir=tmp_path,
    )
    return result["model_path"]


# ---------------------------------------------------------------------------
# validate_model — regression
# ---------------------------------------------------------------------------

class TestValidateModelRegression:
    def test_returns_problem_type(self, regression_df, trained_regression_model):
        result = validate_model(trained_regression_model, regression_df)
        assert result["problem_type"] == "regression"

    def test_metrics_contain_r2(self, regression_df, trained_regression_model):
        result = validate_model(trained_regression_model, regression_df)
        assert "r2" in result["metrics"]

    def test_metrics_have_ci_fields(self, regression_df, trained_regression_model):
        result = validate_model(trained_regression_model, regression_df)
        for metric_info in result["metrics"].values():
            assert "mean" in metric_info
            assert "std" in metric_info
            assert "ci_low" in metric_info
            assert "ci_high" in metric_info
            assert "fold_scores" in metric_info

    def test_error_analysis_is_residuals(self, regression_df, trained_regression_model):
        result = validate_model(trained_regression_model, regression_df)
        ea = result["error_analysis"]
        assert ea["type"] == "residuals"
        assert "actual" in ea
        assert "predicted" in ea
        assert "residuals" in ea
        assert len(ea["actual"]) == len(ea["predicted"])

    def test_limitations_is_list_of_strings(self, regression_df, trained_regression_model):
        result = validate_model(trained_regression_model, regression_df)
        assert isinstance(result["limitations"], list)
        assert len(result["limitations"]) >= 1
        for s in result["limitations"]:
            assert isinstance(s, str)

    def test_n_rows_reported(self, regression_df, trained_regression_model):
        result = validate_model(trained_regression_model, regression_df)
        assert result["n_rows"] == len(regression_df)


# ---------------------------------------------------------------------------
# validate_model — classification
# ---------------------------------------------------------------------------

class TestValidateModelClassification:
    def test_returns_problem_type(self, classification_df, trained_classification_model):
        result = validate_model(trained_classification_model, classification_df)
        assert result["problem_type"] == "classification"

    def test_metrics_contain_accuracy(self, classification_df, trained_classification_model):
        result = validate_model(trained_classification_model, classification_df)
        assert "accuracy" in result["metrics"]

    def test_accuracy_in_range(self, classification_df, trained_classification_model):
        result = validate_model(trained_classification_model, classification_df)
        acc = result["metrics"]["accuracy"]["mean"]
        assert 0.0 <= acc <= 1.0

    def test_error_analysis_is_confusion_matrix(self, classification_df, trained_classification_model):
        result = validate_model(trained_classification_model, classification_df)
        ea = result["error_analysis"]
        assert ea["type"] == "confusion_matrix"
        assert "matrix" in ea
        assert "labels" in ea
        assert "test_accuracy" in ea

    def test_confusion_matrix_is_square(self, classification_df, trained_classification_model):
        result = validate_model(trained_classification_model, classification_df)
        matrix = result["error_analysis"]["matrix"]
        n = len(matrix)
        assert all(len(row) == n for row in matrix)


# ---------------------------------------------------------------------------
# _assess_limitations
# ---------------------------------------------------------------------------

class TestAssessLimitations:
    def test_small_dataset_triggers_warning(self):
        limitations = _assess_limitations({}, "regression", 50)
        combined = " ".join(limitations)
        assert "Small" in combined or "50" in combined

    def test_low_r2_triggers_warning(self):
        metrics = {"r2": {"mean": 0.3, "std": 0.05, "ci_low": 0.2, "ci_high": 0.4, "fold_scores": []}}
        limitations = _assess_limitations(metrics, "regression", 500)
        combined = " ".join(limitations)
        assert "R²" in combined or "explains" in combined

    def test_good_model_no_warnings(self):
        metrics = {
            "accuracy": {"mean": 0.92, "std": 0.03, "ci_low": 0.86, "ci_high": 0.98, "fold_scores": []}
        }
        limitations = _assess_limitations(metrics, "classification", 1000)
        # Should still return at least one (the default caveat)
        assert len(limitations) >= 1

    def test_high_accuracy_variance_triggers_warning(self):
        metrics = {
            "accuracy": {"mean": 0.85, "std": 0.15, "ci_low": 0.55, "ci_high": 1.0, "fold_scores": []}
        }
        limitations = _assess_limitations(metrics, "classification", 500)
        combined = " ".join(limitations)
        assert "fold" in combined.lower() or "variance" in combined.lower() or "varies" in combined.lower()


# ---------------------------------------------------------------------------
# compute_global_importance
# ---------------------------------------------------------------------------

class TestComputeGlobalImportance:
    def test_returns_features_list(self, regression_df, trained_regression_model):
        result = compute_global_importance(trained_regression_model, regression_df)
        assert "features" in result
        assert len(result["features"]) > 0

    def test_features_have_required_keys(self, regression_df, trained_regression_model):
        result = compute_global_importance(trained_regression_model, regression_df)
        for f in result["features"]:
            assert "column" in f
            assert "importance" in f
            assert "rank" in f

    def test_features_sorted_by_importance(self, regression_df, trained_regression_model):
        result = compute_global_importance(trained_regression_model, regression_df)
        importances = [f["importance"] for f in result["features"]]
        assert importances == sorted(importances, reverse=True)

    def test_rank_starts_at_1(self, regression_df, trained_regression_model):
        result = compute_global_importance(trained_regression_model, regression_df)
        assert result["features"][0]["rank"] == 1

    def test_summary_is_string(self, regression_df, trained_regression_model):
        result = compute_global_importance(trained_regression_model, regression_df)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_classification_works(self, classification_df, trained_classification_model):
        result = compute_global_importance(trained_classification_model, classification_df)
        assert result["problem_type"] == "classification"
        assert len(result["features"]) > 0


# ---------------------------------------------------------------------------
# explain_prediction
# ---------------------------------------------------------------------------

class TestExplainPrediction:
    def test_returns_prediction(self, regression_df, trained_regression_model):
        result = explain_prediction(trained_regression_model, regression_df, 0)
        assert "prediction" in result
        assert isinstance(result["prediction"], float)

    def test_returns_top_factors(self, regression_df, trained_regression_model):
        result = explain_prediction(trained_regression_model, regression_df, 0)
        assert "top_factors" in result
        assert len(result["top_factors"]) > 0

    def test_factors_sorted_by_abs_impact(self, regression_df, trained_regression_model):
        result = explain_prediction(trained_regression_model, regression_df, 5)
        impacts = [abs(f["impact"]) for f in result["top_factors"]]
        assert impacts == sorted(impacts, reverse=True)

    def test_direction_field_values(self, regression_df, trained_regression_model):
        result = explain_prediction(trained_regression_model, regression_df, 0)
        for f in result["top_factors"]:
            assert f["direction"] in ("positive", "negative", "neutral")

    def test_out_of_range_row_raises(self, regression_df, trained_regression_model):
        with pytest.raises(ValueError, match="out of range"):
            explain_prediction(trained_regression_model, regression_df, 9999)

    def test_classification_prediction_is_string(self, classification_df, trained_classification_model):
        result = explain_prediction(trained_classification_model, classification_df, 0)
        assert isinstance(result["prediction"], str)

    def test_probability_returned_for_classification(self, classification_df, trained_classification_model):
        result = explain_prediction(trained_classification_model, classification_df, 0)
        # random_forest has predict_proba
        if result["probability"] is not None:
            assert 0.0 <= result["probability"] <= 1.0

    def test_last_row_works(self, regression_df, trained_regression_model):
        last_idx = len(regression_df) - 1
        result = explain_prediction(trained_regression_model, regression_df, last_idx)
        assert result["row_index"] == last_idx


# ---------------------------------------------------------------------------
# API endpoint tests (integration)
# ---------------------------------------------------------------------------

class TestValidationAPI:
    def test_metrics_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Validate Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            train_resp = await client.post(
                f"/api/models/{project_id}/train",
                json={
                    "dataset_id": dataset_id,
                    "target_column": "revenue",
                    "algorithms": ["ridge"],
                },
            )
            model_run_id = train_resp.json()["runs"][0]["id"]

            resp = await client.get(f"/api/validate/{model_run_id}/metrics")
            assert resp.status_code == 200
            body = resp.json()
            assert "metrics" in body
            assert "error_analysis" in body
            assert "limitations" in body

        asyncio.get_event_loop().run_until_complete(run())

    def test_explain_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Explain Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            train_resp = await client.post(
                f"/api/models/{project_id}/train",
                json={
                    "dataset_id": dataset_id,
                    "target_column": "revenue",
                    "algorithms": ["random_forest"],
                },
            )
            model_run_id = train_resp.json()["runs"][0]["id"]

            resp = await client.get(f"/api/validate/{model_run_id}/explain")
            assert resp.status_code == 200
            body = resp.json()
            assert "features" in body
            assert "summary" in body

        asyncio.get_event_loop().run_until_complete(run())

    def test_explain_row_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Row Explain Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            train_resp = await client.post(
                f"/api/models/{project_id}/train",
                json={
                    "dataset_id": dataset_id,
                    "target_column": "revenue",
                    "algorithms": ["ridge"],
                },
            )
            model_run_id = train_resp.json()["runs"][0]["id"]

            resp = await client.get(f"/api/validate/{model_run_id}/explain/0")
            assert resp.status_code == 200
            body = resp.json()
            assert "prediction" in body
            assert "top_factors" in body
            assert body["row_index"] == 0

        asyncio.get_event_loop().run_until_complete(run())

    def test_invalid_model_run_returns_404(self, client, sample_csv_content):
        import asyncio

        async def run():
            resp = await client.get("/api/validate/nonexistent-id/metrics")
            assert resp.status_code == 404

        asyncio.get_event_loop().run_until_complete(run())
