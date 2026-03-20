"""Tests for POST /api/predict/{deployment_id}/explain — live prediction explanation."""

import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

REGRESSION_CSV = b"""age,income,revenue
25,30000,1234.50
35,45000,2150.75
45,60000,3280.00
55,75000,4450.25
30,38000,1680.50
40,52000,2780.00
50,68000,3900.50
28,33000,1380.25
38,48000,2480.75
48,63000,3580.00
32,42000,1920.50
42,57000,3050.25
52,72000,4150.00
60,85000,5300.50
36,46000,2250.75
"""

CLASSIFICATION_CSV = b"""x1,x2,label
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
# Fixture — test client with in-memory DB
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    SQLModel.metadata.create_all(db_module.engine)

    from main import app
    return TestClient(app)


def _full_pipeline(client, csv_bytes: bytes, target: str, problem_type: str):
    """Upload CSV → apply features → set target → train → deploy. Returns deployment dict."""
    # Create project
    proj_resp = client.post("/api/projects", json={"name": "Explain Test"})
    proj_id = proj_resp.json()["id"]

    # Upload CSV
    upload_resp = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"project_id": proj_id},
    )
    assert upload_resp.status_code in (200, 201), upload_resp.text
    dataset_id = upload_resp.json()["dataset_id"]

    # Apply features (no transformations)
    apply_resp = client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert apply_resp.status_code in (200, 201), apply_resp.text

    # Set target column
    target_resp = client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": target},
    )
    assert target_resp.status_code in (200, 201), target_resp.text

    # Train
    algo = "linear_regression" if problem_type == "regression" else "logistic_regression"
    train_resp = client.post(
        f"/api/models/{proj_id}/train",
        json={"algorithms": [algo]},
    )
    assert train_resp.status_code in (200, 202), train_resp.text
    run_id = train_resp.json()["model_run_ids"][0]

    # Wait for training to complete
    run = None
    for _ in range(60):
        runs_resp = client.get(f"/api/models/{proj_id}/runs")
        runs = runs_resp.json().get("runs", [])
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    assert run and run["status"] == "done", f"Training failed: {run}"

    # Deploy
    deploy_resp = client.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code in (200, 201), deploy_resp.text
    return deploy_resp.json(), problem_type


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_explain_endpoint_returns_contributions(client):
    """Explain endpoint returns contributions for all features."""
    deployment, _ = _full_pipeline(client, REGRESSION_CSV, "revenue", "regression")
    dep_id = deployment["id"]

    resp = client.post(
        f"/api/predict/{dep_id}/explain",
        json={"age": 40, "income": 55000},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "contributions" in data
    assert "summary" in data
    assert isinstance(data["contributions"], list)
    assert len(data["contributions"]) > 0


def test_explain_contributions_have_required_fields(client):
    """Each contribution entry has feature, value, mean_value, contribution, direction."""
    deployment, _ = _full_pipeline(client, REGRESSION_CSV, "revenue", "regression")
    dep_id = deployment["id"]

    resp = client.post(
        f"/api/predict/{dep_id}/explain",
        json={"age": 30, "income": 40000},
    )
    assert resp.status_code == 200
    contributions = resp.json()["contributions"]
    assert len(contributions) > 0

    for c in contributions:
        assert "feature" in c
        assert "value" in c
        assert "mean_value" in c
        assert "contribution" in c
        assert c["direction"] in ("positive", "negative")


def test_explain_contributions_sorted_by_abs_value(client):
    """Contributions are sorted by absolute value (highest first)."""
    deployment, _ = _full_pipeline(client, REGRESSION_CSV, "revenue", "regression")
    dep_id = deployment["id"]

    resp = client.post(
        f"/api/predict/{dep_id}/explain",
        json={"age": 50, "income": 70000},
    )
    assert resp.status_code == 200
    contributions = resp.json()["contributions"]
    abs_contribs = [abs(c["contribution"]) for c in contributions]
    assert abs_contribs == sorted(abs_contribs, reverse=True)


def test_explain_returns_summary_string(client):
    """Summary field is a non-empty string."""
    deployment, _ = _full_pipeline(client, REGRESSION_CSV, "revenue", "regression")
    dep_id = deployment["id"]

    resp = client.post(
        f"/api/predict/{dep_id}/explain",
        json={"age": 35, "income": 45000},
    )
    assert resp.status_code == 200
    summary = resp.json()["summary"]
    assert isinstance(summary, str)
    assert len(summary) > 10
    assert "revenue" in summary.lower() or "predict" in summary.lower()


def test_explain_returns_top_drivers(client):
    """top_drivers is a list of feature name strings."""
    deployment, _ = _full_pipeline(client, REGRESSION_CSV, "revenue", "regression")
    dep_id = deployment["id"]

    resp = client.post(
        f"/api/predict/{dep_id}/explain",
        json={"age": 25, "income": 30000},
    )
    assert resp.status_code == 200
    top_drivers = resp.json()["top_drivers"]
    assert isinstance(top_drivers, list)
    # All drivers must be feature names
    feature_names = resp.json()["contributions"]
    feat_names_set = {c["feature"] for c in feature_names}
    for d in top_drivers:
        assert d in feat_names_set


def test_explain_classification_model(client):
    """Explain endpoint works for classification models too."""
    deployment, _ = _full_pipeline(client, CLASSIFICATION_CSV, "label", "classification")
    dep_id = deployment["id"]

    resp = client.post(
        f"/api/predict/{dep_id}/explain",
        json={"x1": 3.0, "x2": 4.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "contributions" in data
    assert data["problem_type"] == "classification"


def test_explain_404_for_inactive_deployment(client):
    """Explain returns 404 for non-existent deployment."""
    resp = client.post("/api/predict/nonexistent-id/explain", json={"x": 1.0})
    assert resp.status_code == 404


def test_explain_includes_target_column(client):
    """Response includes target_column and problem_type."""
    deployment, _ = _full_pipeline(client, REGRESSION_CSV, "revenue", "regression")
    dep_id = deployment["id"]

    resp = client.post(
        f"/api/predict/{dep_id}/explain",
        json={"age": 40, "income": 50000},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_column"] == "revenue"
    assert data["problem_type"] == "regression"


def test_explain_prediction_matches_predict_endpoint(client):
    """Prediction value from /explain should match /predict for same inputs."""
    deployment, _ = _full_pipeline(client, REGRESSION_CSV, "revenue", "regression")
    dep_id = deployment["id"]
    inputs = {"age": 45, "income": 62000}

    pred_resp = client.post(f"/api/predict/{dep_id}", json=inputs)
    explain_resp = client.post(f"/api/predict/{dep_id}/explain", json=inputs)

    assert pred_resp.status_code == 200
    assert explain_resp.status_code == 200
    # Both should return a prediction for "score"
    assert explain_resp.json()["prediction"] == pred_resp.json()["prediction"]


def test_explain_deployer_feature_means_stored(tmp_path):
    """PredictionPipeline stores feature_means and feature_stds after build."""
    from core.deployer import build_prediction_pipeline

    df = pd.DataFrame({
        "age": [25, 35, 45, 55, 30],
        "income": [30000, 45000, 60000, 75000, 38000],
        "score": [60, 72, 85, 91, 65],
    })
    pipeline = build_prediction_pipeline(df, ["age", "income"], "score", "regression")

    assert "age" in pipeline.feature_means
    assert "income" in pipeline.feature_means
    assert "age" in pipeline.feature_stds
    assert "income" in pipeline.feature_stds

    # Means should be close to actual means
    assert abs(pipeline.feature_means["age"] - df["age"].mean()) < 0.01
    assert abs(pipeline.feature_stds["age"] - df["age"].std()) < 0.01


def test_explain_deployer_explain_prediction_function(tmp_path):
    """explain_prediction() function in deployer works end-to-end."""
    import joblib
    from core.deployer import build_prediction_pipeline, explain_prediction, save_pipeline
    from sklearn.linear_model import LinearRegression

    df = pd.DataFrame({
        "age": [25, 35, 45, 55, 30, 40, 50, 28, 38, 48],
        "income": [30000, 45000, 60000, 75000, 38000, 52000, 68000, 33000, 48000, 63000],
        "score": [60, 72, 85, 91, 65, 78, 88, 62, 75, 83],
    })
    pipeline = build_prediction_pipeline(df, ["age", "income"], "score", "regression")
    pipeline_path = tmp_path / "pipeline.pkl"
    save_pipeline(pipeline, pipeline_path)

    # Train a simple model
    X = pipeline.transform_df(df)
    y = df["score"].values
    model = LinearRegression()
    model.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    result = explain_prediction(str(pipeline_path), str(model_path), {"age": 40, "income": 55000})
    assert "prediction" in result
    assert "contributions" in result
    assert "summary" in result
    assert len(result["contributions"]) == 2  # age and income
    assert result["top_drivers"] == ["age", "income"] or set(result["top_drivers"]) == {"age", "income"}
