"""Tests for core/trainer.py — Phase 4 model training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.trainer import (
    ModelRecommendation,
    compare_models,
    recommend_models,
    train_model,
)


# ---------------------------------------------------------------------------
# Fixtures
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


# ---------------------------------------------------------------------------
# recommend_models
# ---------------------------------------------------------------------------

class TestRecommendModels:
    def test_returns_list(self):
        recs = recommend_models("regression", 500, 10)
        assert isinstance(recs, list)
        assert len(recs) >= 2

    def test_all_recommendations_are_model_recommendations(self):
        recs = recommend_models("classification", 500, 5)
        for r in recs:
            assert isinstance(r, ModelRecommendation)
            assert r.algorithm
            assert r.display_name
            assert r.description
            assert r.reason

    def test_small_dataset_fewer_models(self):
        small = recommend_models("classification", 50, 5)
        large = recommend_models("classification", 5000, 5)
        # Both valid, but small should not include gradient boosting
        small_algos = [r.algorithm for r in small]
        assert "gradient_boosting" not in small_algos

    def test_classification_includes_logistic(self):
        recs = recommend_models("classification", 500, 5)
        algos = [r.algorithm for r in recs]
        assert "logistic_regression" in algos

    def test_regression_includes_linear(self):
        recs = recommend_models("regression", 500, 5)
        algos = [r.algorithm for r in recs]
        assert any(a in algos for a in ("linear_regression", "ridge"))

    def test_medium_dataset_has_4_models(self):
        recs = recommend_models("regression", 1000, 10)
        assert len(recs) == 4


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_regression_returns_metrics(self, regression_df, tmp_path):
        result = train_model(
            df=regression_df,
            target_col="target",
            problem_type="regression",
            algorithm="linear_regression",
            hyperparameters={},
            model_dir=tmp_path,
        )
        assert "metrics" in result
        assert "r2" in result["metrics"]
        assert "mae" in result["metrics"]

    def test_classification_returns_accuracy(self, classification_df, tmp_path):
        result = train_model(
            df=classification_df,
            target_col="label",
            problem_type="classification",
            algorithm="random_forest",
            hyperparameters={},
            model_dir=tmp_path,
        )
        assert "accuracy" in result["metrics"]
        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0

    def test_model_file_is_created(self, regression_df, tmp_path):
        result = train_model(
            df=regression_df,
            target_col="target",
            problem_type="regression",
            algorithm="ridge",
            hyperparameters={},
            model_dir=tmp_path,
        )
        assert Path(result["model_path"]).exists()

    def test_training_duration_recorded(self, regression_df, tmp_path):
        result = train_model(
            df=regression_df,
            target_col="target",
            problem_type="regression",
            algorithm="random_forest",
            hyperparameters={},
            model_dir=tmp_path,
        )
        assert result["training_duration_ms"] >= 0

    def test_categorical_features_handled(self, regression_df, tmp_path):
        """Dataset has a 'category' string column — must not raise."""
        result = train_model(
            df=regression_df,
            target_col="target",
            problem_type="regression",
            algorithm="random_forest",
            hyperparameters={},
            model_dir=tmp_path,
        )
        assert result["algorithm"] == "random_forest"

    def test_unknown_algorithm_raises(self, regression_df, tmp_path):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            train_model(
                df=regression_df,
                target_col="target",
                problem_type="regression",
                algorithm="magic_model",
                hyperparameters={},
                model_dir=tmp_path,
            )

    def test_all_classifier_algorithms(self, classification_df, tmp_path):
        for algo in ("logistic_regression", "decision_tree", "random_forest", "gradient_boosting"):
            result = train_model(
                df=classification_df,
                target_col="label",
                problem_type="classification",
                algorithm=algo,
                hyperparameters={},
                model_dir=tmp_path / algo,
            )
            assert result["algorithm"] == algo
            assert "accuracy" in result["metrics"]

    def test_all_regressor_algorithms(self, regression_df, tmp_path):
        for algo in ("linear_regression", "ridge", "random_forest", "gradient_boosting"):
            result = train_model(
                df=regression_df,
                target_col="target",
                problem_type="regression",
                algorithm=algo,
                hyperparameters={},
                model_dir=tmp_path / algo,
            )
            assert result["algorithm"] == algo
            assert "r2" in result["metrics"]

    def test_serialized_model_is_loadable(self, regression_df, tmp_path):
        import joblib
        result = train_model(
            df=regression_df,
            target_col="target",
            problem_type="regression",
            algorithm="ridge",
            hyperparameters={},
            model_dir=tmp_path,
        )
        artifact = joblib.load(result["model_path"])
        assert "pipeline" in artifact
        assert "feature_names" in artifact


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def _make_run(self, id, algo, display_name, metrics, is_selected=False):
        return {
            "id": id,
            "algorithm": algo,
            "display_name": display_name,
            "metrics": metrics,
            "training_duration_ms": 100,
            "is_selected": is_selected,
        }

    def test_empty_returns_no_rankings(self):
        result = compare_models([], "regression")
        assert result["rankings"] == []
        assert result["best_model_id"] is None

    def test_best_model_id_is_highest_scorer(self):
        runs = [
            self._make_run("1", "linear_regression", "Linear Regression", {"r2": 0.7, "mae": 10.0}),
            self._make_run("2", "random_forest", "Random Forest", {"r2": 0.9, "mae": 5.0}),
        ]
        result = compare_models(runs, "regression")
        assert result["best_model_id"] == "2"

    def test_rankings_ordered_by_primary_metric(self):
        runs = [
            self._make_run("1", "lr", "Linear", {"accuracy": 0.75}),
            self._make_run("2", "rf", "Random Forest", {"accuracy": 0.92}),
            self._make_run("3", "dt", "Decision Tree", {"accuracy": 0.80}),
        ]
        result = compare_models(runs, "classification")
        scores = [r["metrics"]["accuracy"] for r in result["rankings"]]
        assert scores == sorted(scores, reverse=True)

    def test_summary_mentions_best_model(self):
        runs = [
            self._make_run("1", "random_forest", "Random Forest", {"accuracy": 0.92}),
            self._make_run("2", "logistic_regression", "Logistic Regression", {"accuracy": 0.80}),
        ]
        result = compare_models(runs, "classification")
        assert "Random Forest" in result["summary"]

    def test_primary_metric_is_r2_for_regression(self):
        runs = [self._make_run("1", "rf", "RF", {"r2": 0.8, "mae": 5.0})]
        result = compare_models(runs, "regression")
        assert result["primary_metric"] == "r2"

    def test_primary_metric_is_accuracy_for_classification(self):
        runs = [self._make_run("1", "rf", "RF", {"accuracy": 0.9})]
        result = compare_models(runs, "classification")
        assert result["primary_metric"] == "accuracy"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------

class TestModelAPI:
    def test_recommend_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Recommend Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            resp = await client.get(
                f"/api/models/{project_id}/recommend",
                params={"dataset_id": dataset_id, "target_column": "revenue"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "recommendations" in body
            assert len(body["recommendations"]) >= 2
            assert body["problem_type"] in ("classification", "regression")

        asyncio.get_event_loop().run_until_complete(run())

    def test_train_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Train Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            resp = await client.post(
                f"/api/models/{project_id}/train",
                json={
                    "dataset_id": dataset_id,
                    "target_column": "revenue",
                    "algorithms": ["linear_regression", "random_forest"],
                },
            )
            assert resp.status_code == 201
            body = resp.json()
            assert "runs" in body
            assert len(body["runs"]) == 2
            for run in body["runs"]:
                assert run["status"] in ("done", "failed")
                assert run["algorithm"] in ("linear_regression", "random_forest")

        asyncio.get_event_loop().run_until_complete(run())

    def test_runs_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Runs Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            await client.post(
                f"/api/models/{project_id}/train",
                json={
                    "dataset_id": dataset_id,
                    "target_column": "revenue",
                    "algorithms": ["ridge"],
                },
            )

            resp = await client.get(f"/api/models/{project_id}/runs")
            assert resp.status_code == 200
            body = resp.json()
            assert "runs" in body
            assert len(body["runs"]) >= 1

        asyncio.get_event_loop().run_until_complete(run())

    def test_compare_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Compare Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            await client.post(
                f"/api/models/{project_id}/train",
                json={
                    "dataset_id": dataset_id,
                    "target_column": "revenue",
                    "algorithms": ["linear_regression", "random_forest"],
                },
            )

            resp = await client.get(f"/api/models/{project_id}/compare")
            assert resp.status_code == 200
            body = resp.json()
            assert "summary" in body
            assert "rankings" in body

        asyncio.get_event_loop().run_until_complete(run())

    def test_select_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Select Test"})
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
                    "algorithms": ["linear_regression"],
                },
            )
            model_run_id = train_resp.json()["runs"][0]["id"]

            select_resp = await client.post(f"/api/models/{model_run_id}/select")
            assert select_resp.status_code == 200
            body = select_resp.json()
            assert body["success"] is True

        asyncio.get_event_loop().run_until_complete(run())
