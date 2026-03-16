"""Tests for core/deployer.py and /api/deploy/* + /api/predict/* endpoints — Phase 6."""

import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Shared sample data
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

CLASSIFICATION_CSV = b"""feature1,feature2,label
1.0,2.0,cat
2.0,3.0,dog
3.0,4.0,cat
4.0,5.0,dog
5.0,6.0,cat
6.0,7.0,dog
7.0,8.0,cat
8.0,9.0,dog
9.0,10.0,cat
10.0,11.0,dog
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    import models.deployment  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def trained_run(client):
    """Set up project → upload → features → train → wait for done."""
    proj = client.post("/api/projects", json={"name": "Deploy Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    apply = client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert apply.status_code == 201

    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue"},
    )

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert train_resp.status_code == 202
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    assert run["status"] == "done", f"Training failed: {run.get('error_message')}"
    return project_id, dataset_id, run_id


# ---------------------------------------------------------------------------
# Unit tests: PredictionPipeline
# ---------------------------------------------------------------------------


class TestPredictionPipeline:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "product": ["A", "B", "A", "C", "B", "A"],
                "region": ["N", "S", "E", "W", "N", "S"],
                "units": [10, 8, 18, 4, 15, 9],
                "revenue": [1200.0, 850.0, 2100.0, 450.0, 1650.0, 980.0],
            }
        )

    def test_build_pipeline_numeric(self, sample_df):
        from core.deployer import build_prediction_pipeline

        pipeline = build_prediction_pipeline(
            sample_df, ["units"], "revenue", "regression"
        )
        assert pipeline.column_types["units"] == "numeric"
        assert "units" in pipeline.medians

    def test_build_pipeline_categorical(self, sample_df):
        from core.deployer import build_prediction_pipeline

        pipeline = build_prediction_pipeline(
            sample_df, ["product"], "revenue", "regression"
        )
        assert pipeline.column_types["product"] == "categorical"
        assert "product" in pipeline.label_encoders

    def test_transform_returns_correct_shape(self, sample_df):
        from core.deployer import build_prediction_pipeline

        pipeline = build_prediction_pipeline(
            sample_df, ["units", "product"], "revenue", "regression"
        )
        X = pipeline.transform({"units": 12, "product": "A"})
        assert X.shape == (1, 2)

    def test_transform_handles_missing_feature(self, sample_df):
        from core.deployer import build_prediction_pipeline

        pipeline = build_prediction_pipeline(
            sample_df, ["units"], "revenue", "regression"
        )
        # Missing value filled with median
        X = pipeline.transform({})
        assert X.shape == (1, 1)
        assert not np.isnan(X[0, 0])

    def test_transform_handles_unseen_category(self, sample_df):
        from core.deployer import build_prediction_pipeline

        pipeline = build_prediction_pipeline(
            sample_df, ["product"], "revenue", "regression"
        )
        # Unseen category → 0 (no crash)
        X = pipeline.transform({"product": "UNSEEN_CATEGORY"})
        assert X.shape == (1, 1)
        assert X[0, 0] == 0.0

    def test_decode_regression_prediction(self, sample_df):
        from core.deployer import build_prediction_pipeline

        pipeline = build_prediction_pipeline(
            sample_df, ["units"], "revenue", "regression"
        )
        assert pipeline.decode_prediction(1234.567) == 1234.567

    def test_classification_with_string_target(self):
        from core.deployer import build_prediction_pipeline

        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "label": ["cat", "dog", "cat", "dog"],
            }
        )
        pipeline = build_prediction_pipeline(df, ["x"], "label", "classification")
        assert pipeline.target_encoder is not None
        decoded = pipeline.decode_prediction(0)
        assert decoded in ("cat", "dog")

    def test_save_and_load_pipeline(self, tmp_path, sample_df):
        from core.deployer import (
            build_prediction_pipeline,
            save_pipeline,
            load_pipeline,
        )

        pipeline = build_prediction_pipeline(
            sample_df, ["units", "product"], "revenue", "regression"
        )
        path = tmp_path / "pipeline.joblib"
        save_pipeline(pipeline, path)
        loaded = load_pipeline(path)
        X1 = pipeline.transform({"units": 10, "product": "A"})
        X2 = loaded.transform({"units": 10, "product": "A"})
        np.testing.assert_array_equal(X1, X2)

    def test_transform_df(self, sample_df):
        from core.deployer import build_prediction_pipeline

        pipeline = build_prediction_pipeline(
            sample_df, ["units", "product"], "revenue", "regression"
        )
        X = pipeline.transform_df(sample_df[["units", "product"]])
        assert X.shape == (len(sample_df), 2)


# ---------------------------------------------------------------------------
# Unit tests: predict_single / predict_batch with a real sklearn model
# ---------------------------------------------------------------------------


class TestPredictFunctions:
    @pytest.fixture
    def trained_pipeline_and_model(self, tmp_path):
        """Build a pipeline + train a model, save both."""
        from core.deployer import build_prediction_pipeline, save_pipeline
        from core.trainer import prepare_features, train_single_model

        df = pd.DataFrame(
            {
                "x1": list(range(20)),
                "x2": [float(i) * 0.5 for i in range(20)],
                "y": [float(i) * 2 + 1 for i in range(20)],
            }
        )
        pipeline = build_prediction_pipeline(df, ["x1", "x2"], "y", "regression")
        pipeline_path = tmp_path / "pl.joblib"
        save_pipeline(pipeline, pipeline_path)

        X, y, _ = prepare_features(df, ["x1", "x2"], "y", "regression")
        result = train_single_model(
            X, y, "linear_regression", "regression", tmp_path, "test_run"
        )
        return str(pipeline_path), result["model_path"]

    def test_predict_single_returns_numeric(self, trained_pipeline_and_model):
        from core.deployer import predict_single

        pl_path, m_path = trained_pipeline_and_model
        result = predict_single(pl_path, m_path, {"x1": 5, "x2": 2.5})
        assert "prediction" in result
        assert isinstance(result["prediction"], (int, float))

    def test_predict_single_reasonable_value(self, trained_pipeline_and_model):
        from core.deployer import predict_single

        pl_path, m_path = trained_pipeline_and_model
        result = predict_single(pl_path, m_path, {"x1": 10, "x2": 5.0})
        # y = 2*x1 + 1, so x1=10 → ~21
        assert 15 < result["prediction"] < 30

    def test_predict_batch_returns_csv(self, trained_pipeline_and_model):
        from core.deployer import predict_batch

        pl_path, m_path = trained_pipeline_and_model
        csv_bytes = b"x1,x2\n1,0.5\n2,1.0\n3,1.5\n"
        result = predict_batch(pl_path, m_path, csv_bytes)
        df = pd.read_csv(io.BytesIO(result))
        assert "prediction" in df.columns
        assert len(df) == 3

    def test_predict_batch_preserves_input_columns(self, trained_pipeline_and_model):
        from core.deployer import predict_batch

        pl_path, m_path = trained_pipeline_and_model
        csv_bytes = b"x1,x2\n1,0.5\n"
        result = predict_batch(pl_path, m_path, csv_bytes)
        df = pd.read_csv(io.BytesIO(result))
        assert "x1" in df.columns
        assert "x2" in df.columns
        assert "prediction" in df.columns

    def test_get_feature_schema(self, trained_pipeline_and_model):
        from core.deployer import get_feature_schema

        pl_path, _ = trained_pipeline_and_model
        schema = get_feature_schema(pl_path)
        assert isinstance(schema, list)
        names = [s["name"] for s in schema]
        assert "x1" in names
        assert "x2" in names
        for entry in schema:
            assert "type" in entry


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


class TestDeployAPI:
    def test_deploy_model_returns_201(self, client, trained_run):
        project_id, dataset_id, run_id = trained_run
        resp = client.post(f"/api/deploy/{run_id}")
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert "id" in body
        assert body["model_run_id"] == run_id
        assert body["is_active"] is True

    def test_deploy_returns_endpoint_and_dashboard_urls(self, client, trained_run):
        _, _, run_id = trained_run
        resp = client.post(f"/api/deploy/{run_id}")
        body = resp.json()
        dep_id = body["id"]
        assert body["endpoint_path"] == f"/api/predict/{dep_id}"
        assert body["dashboard_url"] == f"/predict/{dep_id}"

    def test_deploy_idempotent(self, client, trained_run):
        """Deploying twice returns the same deployment."""
        _, _, run_id = trained_run
        r1 = client.post(f"/api/deploy/{run_id}")
        r2 = client.post(f"/api/deploy/{run_id}")
        assert r1.json()["id"] == r2.json()["id"]

    def test_deploy_404_unknown_run(self, client):
        resp = client.post("/api/deploy/nonexistent-run")
        assert resp.status_code == 404

    def test_list_deployments(self, client, trained_run):
        _, _, run_id = trained_run
        client.post(f"/api/deploy/{run_id}")
        resp = client.get("/api/deployments")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert any(d["model_run_id"] == run_id for d in body)

    def test_get_deployment_detail(self, client, trained_run):
        _, _, run_id = trained_run
        dep_id = client.post(f"/api/deploy/{run_id}").json()["id"]
        resp = client.get(f"/api/deploy/{dep_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == dep_id
        assert "feature_schema" in body
        assert isinstance(body["feature_schema"], list)

    def test_undeploy(self, client, trained_run):
        _, _, run_id = trained_run
        dep_id = client.post(f"/api/deploy/{run_id}").json()["id"]
        resp = client.delete(f"/api/deploy/{dep_id}")
        assert resp.status_code == 204

        # Should no longer appear in list
        deployments = client.get("/api/deployments").json()
        assert not any(d["id"] == dep_id for d in deployments)

    def test_predict_single(self, client, trained_run):
        _, _, run_id = trained_run
        dep_id = client.post(f"/api/deploy/{run_id}").json()["id"]

        resp = client.post(
            f"/api/predict/{dep_id}",
            json={
                "product": "Widget A",
                "region": "North",
                "units": 10,
                "date": "2024-01-01",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "prediction" in body
        assert isinstance(body["prediction"], (int, float))

    def test_predict_increments_request_count(self, client, trained_run):
        _, _, run_id = trained_run
        dep_id = client.post(f"/api/deploy/{run_id}").json()["id"]

        client.post(
            f"/api/predict/{dep_id}",
            json={"units": 10},
        )
        client.post(
            f"/api/predict/{dep_id}",
            json={"units": 12},
        )
        detail = client.get(f"/api/deploy/{dep_id}").json()
        assert detail["request_count"] >= 2

    def test_predict_404_inactive_deployment(self, client, trained_run):
        _, _, run_id = trained_run
        dep_id = client.post(f"/api/deploy/{run_id}").json()["id"]
        client.delete(f"/api/deploy/{dep_id}")
        resp = client.post(f"/api/predict/{dep_id}", json={"units": 10})
        assert resp.status_code == 404

    def test_batch_prediction_returns_csv(self, client, trained_run):
        _, _, run_id = trained_run
        dep_id = client.post(f"/api/deploy/{run_id}").json()["id"]

        batch_csv = b"product,region,units,date\nWidget A,North,10,2024-01-01\nWidget B,South,8,2024-01-02\n"
        resp = client.post(
            f"/api/predict/{dep_id}/batch",
            files={"file": ("batch.csv", io.BytesIO(batch_csv), "text/csv")},
        )
        assert resp.status_code == 200, resp.text
        assert "text/csv" in resp.headers.get("content-type", "")

        result_df = pd.read_csv(io.BytesIO(resp.content))
        assert "prediction" in result_df.columns
        assert len(result_df) == 2
