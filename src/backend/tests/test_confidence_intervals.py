"""Tests for prediction confidence intervals.

Regression models return a 95% prediction interval (±1.96 × residual_std).
Classification models return a top-class confidence score (max predict_proba).
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

REGRESSION_CSV = b"""units,price,region,revenue
10,5.0,North,1200.0
8,4.5,South,850.0
18,6.0,East,2100.0
4,3.0,West,450.0
15,5.5,North,1650.0
9,4.8,South,980.0
11,5.1,North,1100.0
16,5.7,East,1750.0
20,6.2,West,2300.0
6,3.5,South,620.0
"""

CLASSIFICATION_CSV = b"""f1,f2,label
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
    import models.prediction_log  # noqa
    import models.feedback_record  # noqa

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


def _setup_and_deploy(
    client, csv_data: bytes, target: str, problem_type: str, algorithms: list[str]
):
    """Helper: create project → upload → features → train → deploy."""
    proj = client.post("/api/projects", json={"name": "CI Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(csv_data), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": target})

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": algorithms},
    )
    assert train_resp.status_code == 202
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(40):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.3)

    assert run["status"] == "done", f"Training failed: {run.get('error_message')}"

    deploy = client.post(f"/api/deploy/{run_id}")
    assert deploy.status_code == 201
    return deploy.json()["id"], run_id, project_id, dataset_id


# ---------------------------------------------------------------------------
# Unit tests: PredictionPipeline residual_std field
# ---------------------------------------------------------------------------


class TestPipelineResidualStd:
    def test_pipeline_has_residual_std_field(self):
        """PredictionPipeline should have a residual_std field defaulting to 0."""
        from core.deployer import PredictionPipeline

        p = PredictionPipeline(feature_names=[], column_types={})
        assert hasattr(p, "residual_std")
        assert p.residual_std == 0.0

    def test_build_pipeline_does_not_set_residual_std(self):
        """build_prediction_pipeline does not compute residuals (no model available)."""
        from core.deployer import build_prediction_pipeline

        df = pd.DataFrame(
            {
                "units": [10.0, 8.0, 18.0, 4.0, 15.0],
                "revenue": [1200.0, 850.0, 2100.0, 450.0, 1650.0],
            }
        )
        pipeline = build_prediction_pipeline(df, ["units"], "revenue", "regression")
        # residual_std is 0 because no model is available during pipeline build
        assert pipeline.residual_std == 0.0

    def test_pipeline_residual_std_can_be_set(self):
        """residual_std can be assigned manually (as done at deploy time)."""
        from core.deployer import PredictionPipeline

        p = PredictionPipeline(feature_names=[], column_types={})
        p.residual_std = 123.45
        assert p.residual_std == 123.45


# ---------------------------------------------------------------------------
# Unit tests: predict_single confidence interval
# ---------------------------------------------------------------------------


class TestPredictSingleConfidenceInterval:
    @pytest.fixture
    def deployed_pipeline(self, tmp_path):
        """Build a tiny pipeline + model and save to tmp dir."""
        from sklearn.linear_model import LinearRegression
        import joblib
        from core.deployer import build_prediction_pipeline, save_pipeline

        df = pd.DataFrame(
            {
                "units": [10.0, 8.0, 18.0, 4.0, 15.0, 9.0, 11.0, 16.0, 20.0, 6.0],
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
            }
        )

        pipeline = build_prediction_pipeline(df, ["units"], "revenue", "regression")

        model = LinearRegression()
        X = df[["units"]].values
        y = df["revenue"].values
        model.fit(X, y)

        # Simulate deploy: compute residual_std
        y_pred = model.predict(X)
        pipeline.residual_std = float(np.std(y - y_pred))

        pipeline_path = tmp_path / "pipeline.joblib"
        model_path = tmp_path / "model.joblib"
        save_pipeline(pipeline, pipeline_path)
        joblib.dump(model, model_path)

        return str(pipeline_path), str(model_path)

    @pytest.fixture
    def deployed_pipeline_no_std(self, tmp_path):
        """Pipeline without residual_std (residual_std = 0)."""
        from sklearn.linear_model import LinearRegression
        import joblib
        from core.deployer import build_prediction_pipeline, save_pipeline

        df = pd.DataFrame(
            {
                "units": [10.0, 8.0, 18.0, 4.0, 15.0],
                "revenue": [1200.0, 850.0, 2100.0, 450.0, 1650.0],
            }
        )

        pipeline = build_prediction_pipeline(df, ["units"], "revenue", "regression")
        pipeline.residual_std = 0.0  # explicitly no interval

        model = LinearRegression()
        model.fit(df[["units"]].values, df["revenue"].values)

        pipeline_path = tmp_path / "pipeline.joblib"
        model_path = tmp_path / "model.joblib"
        save_pipeline(pipeline, pipeline_path)
        joblib.dump(model, model_path)

        return str(pipeline_path), str(model_path)

    @pytest.fixture
    def deployed_cls_pipeline(self, tmp_path):
        """Classification pipeline."""
        from sklearn.linear_model import LogisticRegression
        import joblib
        from core.deployer import build_prediction_pipeline, save_pipeline

        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "label": ["cat", "dog", "cat", "dog", "cat", "dog"],
            }
        )

        pipeline = build_prediction_pipeline(
            df, ["f1", "f2"], "label", "classification"
        )

        model = LogisticRegression(random_state=42, max_iter=1000)
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(df["label"])
        model.fit(df[["f1", "f2"]].values, y)

        pipeline_path = tmp_path / "pipeline.joblib"
        model_path = tmp_path / "model.joblib"
        save_pipeline(pipeline, pipeline_path)
        joblib.dump(model, model_path)

        return str(pipeline_path), str(model_path)

    def test_confidence_interval_present_when_residual_std_set(self, deployed_pipeline):
        from core.deployer import predict_single

        pipeline_path, model_path = deployed_pipeline
        result = predict_single(pipeline_path, model_path, {"units": 12.0})
        assert "confidence_interval" in result
        ci = result["confidence_interval"]
        assert ci["lower"] < float(result["prediction"]) < ci["upper"]

    def test_confidence_interval_level_is_95(self, deployed_pipeline):
        from core.deployer import predict_single

        pipeline_path, model_path = deployed_pipeline
        result = predict_single(pipeline_path, model_path, {"units": 12.0})
        assert result["confidence_interval"]["level"] == 0.95

    def test_confidence_interval_width(self, deployed_pipeline):
        """Interval width should be ~2 × 1.96 × residual_std."""
        from core.deployer import predict_single, load_pipeline

        pipeline_path, model_path = deployed_pipeline
        pipeline = load_pipeline(pipeline_path)
        result = predict_single(pipeline_path, model_path, {"units": 12.0})
        ci = result["confidence_interval"]
        expected_half_width = 1.96 * pipeline.residual_std
        actual_half_width = (ci["upper"] - ci["lower"]) / 2
        assert abs(actual_half_width - expected_half_width) < 0.01

    def test_no_confidence_interval_when_residual_std_zero(
        self, deployed_pipeline_no_std
    ):
        from core.deployer import predict_single

        pipeline_path, model_path = deployed_pipeline_no_std
        result = predict_single(pipeline_path, model_path, {"units": 12.0})
        assert "confidence_interval" not in result

    def test_classification_returns_confidence(self, deployed_cls_pipeline):
        """Classification prediction includes 'confidence' = max proba."""
        from core.deployer import predict_single

        pipeline_path, model_path = deployed_cls_pipeline
        result = predict_single(pipeline_path, model_path, {"f1": 3.0, "f2": 4.0})
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classification_no_confidence_interval(self, deployed_cls_pipeline):
        """Classification predictions do not include a regression-style CI."""
        from core.deployer import predict_single

        pipeline_path, model_path = deployed_cls_pipeline
        result = predict_single(pipeline_path, model_path, {"f1": 3.0, "f2": 4.0})
        assert "confidence_interval" not in result

    def test_confidence_label_text(self, deployed_pipeline):
        from core.deployer import predict_single

        pipeline_path, model_path = deployed_pipeline
        result = predict_single(pipeline_path, model_path, {"units": 10.0})
        assert "label" in result["confidence_interval"]
        assert "95%" in result["confidence_interval"]["label"]


# ---------------------------------------------------------------------------
# Integration tests: deploy endpoint sets residual_std, predict returns CI
# ---------------------------------------------------------------------------


class TestConfidenceIntervalIntegration:
    def test_regression_predict_returns_confidence_interval(self, client):
        """End-to-end: deploy regression model → predict → confidence_interval in response."""
        deployment_id, *_ = _setup_and_deploy(
            client, REGRESSION_CSV, "revenue", "regression", ["linear_regression"]
        )
        resp = client.post(
            f"/api/predict/{deployment_id}",
            json={"units": 12.0, "price": 5.0, "region": "North"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "confidence_interval" in body
        ci = body["confidence_interval"]
        assert ci["lower"] < float(body["prediction"]) < ci["upper"]
        assert ci["level"] == 0.95

    def test_classification_predict_returns_confidence(self, client):
        """End-to-end: deploy classification model → predict → confidence in response."""
        deployment_id, *_ = _setup_and_deploy(
            client,
            CLASSIFICATION_CSV,
            "label",
            "classification",
            ["logistic_regression"],
        )
        resp = client.post(
            f"/api/predict/{deployment_id}",
            json={"f1": 3.0, "f2": 4.0},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "confidence" in body
        assert 0.0 <= body["confidence"] <= 1.0

    def test_confidence_interval_lower_less_than_upper(self, client):
        """Sanity: lower bound must be strictly less than upper bound."""
        deployment_id, *_ = _setup_and_deploy(
            client, REGRESSION_CSV, "revenue", "regression", ["linear_regression"]
        )
        resp = client.post(
            f"/api/predict/{deployment_id}",
            json={"units": 5.0, "price": 3.0, "region": "West"},
        )
        assert resp.status_code == 200
        ci = resp.json()["confidence_interval"]
        assert ci["lower"] < ci["upper"]

    def test_batch_csv_still_works_with_confidence(self, client):
        """Batch prediction should not be broken by the residual_std addition."""
        deployment_id, *_ = _setup_and_deploy(
            client, REGRESSION_CSV, "revenue", "regression", ["linear_regression"]
        )
        batch_csv = b"units,price,region\n10,5.0,North\n8,4.5,South\n"
        resp = client.post(
            f"/api/predict/{deployment_id}/batch",
            files={"file": ("batch.csv", io.BytesIO(batch_csv), "text/csv")},
        )
        assert resp.status_code == 200
        assert b"prediction" in resp.content
