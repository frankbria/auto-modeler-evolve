"""Tests for core/trainer.py and the /api/models/* endpoints — Phase 4."""

import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine
import db as db_module


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV = b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-01,Widget B,South,850.00,8
2024-01-02,Widget A,East,2100.75,18
2024-01-02,Widget C,West,450.25,4
2024-01-03,Widget B,North,1650.00,15
2024-01-04,Widget A,South,980.00,9
2024-01-04,Widget C,North,1100.25,11
2024-01-05,Widget B,East,1750.00,16
2024-01-05,Widget A,West,2300.50,20
2024-01-06,Widget C,South,620.75,6
"""


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.conversation  # noqa
    import models.model_run  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def project_and_dataset(client):
    """Create a project, upload CSV, apply transforms, set target."""
    # Create project
    proj = client.post("/api/projects", json={"name": "Trainer Test Project"})
    assert proj.status_code == 201, proj.text
    project_id = proj.json()["id"]

    # Upload CSV
    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["dataset_id"]

    # Apply a minimal feature set (empty transforms just creates the FeatureSet)
    apply = client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert apply.status_code == 201, apply.text
    feature_set_id = apply.json()["feature_set_id"]

    # Set target column
    target = client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue"},
    )
    assert target.status_code == 200, target.text

    return project_id, dataset_id, feature_set_id


# ---------------------------------------------------------------------------
# Unit tests: recommend_models
# ---------------------------------------------------------------------------


class TestRecommendModels:
    def test_returns_recommendations_for_regression(self):
        from core.trainer import recommend_models

        recs = recommend_models("regression", 500, 10)
        assert isinstance(recs, list)
        assert len(recs) >= 2
        algos = [r["algorithm"] for r in recs]
        assert "random_forest_regressor" in algos

    def test_returns_recommendations_for_classification(self):
        from core.trainer import recommend_models

        recs = recommend_models("classification", 500, 10)
        algos = [r["algorithm"] for r in recs]
        assert "random_forest_classifier" in algos

    def test_each_has_required_fields(self):
        from core.trainer import recommend_models

        recs = recommend_models("regression", 1000, 8)
        for r in recs:
            assert "algorithm" in r
            assert "name" in r
            assert "plain_english" in r
            assert "recommended_because" in r

    def test_small_dataset_mentions_size(self):
        from core.trainer import recommend_models

        recs = recommend_models("regression", 50, 5)
        reasons = " ".join(r["recommended_because"] for r in recs)
        assert "50" in reasons


# ---------------------------------------------------------------------------
# Unit tests: prepare_features
# ---------------------------------------------------------------------------


class TestPrepareFeatures:
    @pytest.fixture
    def sales_df(self):
        return pd.DataFrame(
            {
                "product": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
                "region": ["N", "S", "E", "W", "N", "S", "E", "W", "N", "S"],
                "revenue": [
                    1200.0,
                    850.0,
                    2100.0,
                    450.0,
                    1650.0,
                    980.0,
                    1100.0,
                    1750.0,
                    2300.0,
                    620.0,
                ],
                "units": [10, 8, 18, 4, 15, 9, 11, 16, 20, 6],
            }
        )

    def test_basic_regression(self, sales_df):
        from core.trainer import prepare_features

        X, y, le = prepare_features(
            sales_df, ["product", "region", "units"], "revenue", "regression"
        )
        assert X.shape == (10, 3)
        assert y.shape == (10,)
        assert le is None

    def test_categorical_encoded_automatically(self, sales_df):
        from core.trainer import prepare_features

        X, y, le = prepare_features(sales_df, ["product"], "revenue", "regression")
        assert X.shape[1] == 1
        assert np.issubdtype(X.dtype, np.floating)

    def test_classification_string_target_encoded(self, sales_df):
        from core.trainer import prepare_features

        X, y, le = prepare_features(
            sales_df, ["revenue", "units"], "region", "classification"
        )
        assert le is not None
        assert np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.floating)

    def test_missing_target_column_raises(self, sales_df):
        from core.trainer import prepare_features

        with pytest.raises(ValueError, match="not found"):
            prepare_features(sales_df, ["units"], "nonexistent", "regression")

    def test_no_valid_features_raises(self, sales_df):
        from core.trainer import prepare_features

        with pytest.raises(ValueError, match="No valid feature columns"):
            prepare_features(sales_df, ["nonexistent"], "revenue", "regression")


# ---------------------------------------------------------------------------
# Unit tests: train_single_model
# ---------------------------------------------------------------------------


class TestTrainSingleModel:
    @pytest.fixture
    def xy_regression(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))
        y = 2 * X[:, 0] + X[:, 1] + rng.normal(0, 0.1, 50)
        return X, y

    @pytest.fixture
    def xy_classification(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_trains_linear_regression(self, tmp_path, xy_regression):
        from core.trainer import train_single_model

        X, y = xy_regression
        result = train_single_model(
            X, y, "linear_regression", "regression", tmp_path, "test_run_1"
        )
        assert "metrics" in result
        assert "r2" in result["metrics"]
        assert result["metrics"]["r2"] > 0.5  # should fit well on synthetic data
        assert "model_path" in result
        assert "summary" in result

    def test_trains_random_forest_regression(self, tmp_path, xy_regression):
        from core.trainer import train_single_model

        X, y = xy_regression
        result = train_single_model(
            X, y, "random_forest_regressor", "regression", tmp_path, "test_run_2"
        )
        assert result["metrics"]["r2"] > 0.0

    def test_trains_random_forest_classification(self, tmp_path, xy_classification):
        from core.trainer import train_single_model

        X, y = xy_classification
        result = train_single_model(
            X, y, "random_forest_classifier", "classification", tmp_path, "test_run_3"
        )
        assert "accuracy" in result["metrics"]
        assert result["metrics"]["accuracy"] >= 0.0

    def test_model_file_saved(self, tmp_path, xy_regression):
        from core.trainer import train_single_model
        import os

        X, y = xy_regression
        result = train_single_model(
            X, y, "linear_regression", "regression", tmp_path, "test_run_4"
        )
        assert os.path.exists(result["model_path"])

    def test_unknown_algorithm_raises(self, tmp_path, xy_regression):
        from core.trainer import train_single_model

        X, y = xy_regression
        with pytest.raises(ValueError, match="Unknown algorithm"):
            train_single_model(
                X, y, "definitely_not_real", "regression", tmp_path, "test_run_5"
            )

    def test_metrics_have_required_fields_regression(self, tmp_path, xy_regression):
        from core.trainer import train_single_model

        X, y = xy_regression
        result = train_single_model(
            X, y, "linear_regression", "regression", tmp_path, "test_run_6"
        )
        assert all(
            k in result["metrics"]
            for k in ("r2", "mae", "rmse", "train_size", "test_size")
        )

    def test_metrics_have_required_fields_classification(
        self, tmp_path, xy_classification
    ):
        from core.trainer import train_single_model

        X, y = xy_classification
        result = train_single_model(
            X, y, "logistic_regression", "classification", tmp_path, "test_run_7"
        )
        assert all(
            k in result["metrics"] for k in ("accuracy", "f1", "precision", "recall")
        )

    def test_summary_is_plain_english(self, tmp_path, xy_regression):
        from core.trainer import train_single_model

        X, y = xy_regression
        result = train_single_model(
            X, y, "linear_regression", "regression", tmp_path, "test_run_8"
        )
        assert "R²" in result["summary"]
        assert len(result["summary"]) > 20


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestModelAPI:
    def test_recommendations_endpoint(self, client, project_and_dataset):
        project_id, _, _ = project_and_dataset
        resp = client.get(f"/api/models/{project_id}/recommendations")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "recommendations" in body
        assert len(body["recommendations"]) >= 2
        assert body["problem_type"] == "regression"
        assert body["target_column"] == "revenue"

    def test_train_endpoint_returns_202(self, client, project_and_dataset):
        project_id, _, _ = project_and_dataset
        resp = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert "model_run_ids" in body
        assert len(body["model_run_ids"]) == 1

    def test_runs_endpoint_lists_runs(self, client, project_and_dataset):
        project_id, _, _ = project_and_dataset
        # Start training
        client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        resp = client.get(f"/api/models/{project_id}/runs")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "runs" in body
        assert len(body["runs"]) >= 1

    def test_training_completes(self, client, project_and_dataset):
        """Training should finish quickly for a small dataset."""
        project_id, _, _ = project_and_dataset
        train_resp = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        run_id = train_resp.json()["model_run_ids"][0]

        # Poll until done (max 10 seconds)
        for _ in range(20):
            runs_resp = client.get(f"/api/models/{project_id}/runs")
            runs = runs_resp.json()["runs"]
            run = next(r for r in runs if r["id"] == run_id)
            if run["status"] in ("done", "failed"):
                break
            time.sleep(0.5)

        assert run["status"] == "done", f"Training failed: {run.get('error_message')}"
        assert run["metrics"] is not None
        assert run["summary"] is not None

    def test_compare_endpoint(self, client, project_and_dataset):
        """Compare should work after at least one model finishes training."""
        project_id, _, _ = project_and_dataset
        client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        # Wait for training
        for _ in range(20):
            runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
            if all(r["status"] in ("done", "failed") for r in runs):
                break
            time.sleep(0.5)

        resp = client.get(f"/api/models/{project_id}/compare")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "models" in body
        assert "recommendation" in body

    def test_select_model(self, client, project_and_dataset):
        project_id, _, _ = project_and_dataset
        train_resp = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        run_id = train_resp.json()["model_run_ids"][0]

        # Wait for completion
        for _ in range(20):
            runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
            run = next(r for r in runs if r["id"] == run_id)
            if run["status"] in ("done", "failed"):
                break
            time.sleep(0.5)

        assert run["status"] == "done"

        select_resp = client.post(f"/api/models/{run_id}/select")
        assert select_resp.status_code == 200, select_resp.text
        assert select_resp.json()["is_selected"] is True

    def test_invalid_algorithm_rejected(self, client, project_and_dataset):
        project_id, _, _ = project_and_dataset
        resp = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["definitely_not_a_real_algorithm"]},
        )
        assert resp.status_code == 400

    def test_recommendations_404_for_unknown_project(self, client):
        resp = client.get("/api/models/nonexistent-project/recommendations")
        assert resp.status_code == 404

    def test_multiple_algorithms_train_in_parallel(self, client, project_and_dataset):
        project_id, _, _ = project_and_dataset
        resp = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression", "random_forest_regressor"]},
        )
        assert resp.status_code == 202
        assert len(resp.json()["model_run_ids"]) == 2
