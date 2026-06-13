"""Tests for run_feature_sweep pure function, _FEATURE_SWEEP_PATTERNS, and REST endpoint."""

from __future__ import annotations

import json

import db as db_module
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

# ---------------------------------------------------------------------------
# Helpers — build tiny trained deployments
# ---------------------------------------------------------------------------


def _make_regression_deployment(tmp_path):
    """LinearRegression over two numeric features."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from core.deployer import build_prediction_pipeline, save_pipeline

    df = pd.DataFrame(
        {
            "units": np.arange(1, 21, dtype=float),
            "price": np.ones(20) * 10.0,
            "revenue": np.arange(1, 21, dtype=float) * 5.0,
        }
    )
    pipeline = build_prediction_pipeline(
        df, ["units", "price"], "revenue", "regression"
    )
    pipeline_path = str(tmp_path / "pipeline.joblib")
    save_pipeline(pipeline, pipeline_path)
    model = LinearRegression().fit(df[["units", "price"]].values, df["revenue"].values)
    model_path = str(tmp_path / "model.joblib")
    joblib.dump(model, model_path)
    return pipeline_path, model_path


def _make_classification_deployment(tmp_path):
    """LogisticRegression with categorical feature."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from core.deployer import build_prediction_pipeline, save_pipeline

    df = pd.DataFrame(
        {
            "score": list(range(1, 21)),
            "region": ["East"] * 10 + ["West"] * 10,
            "churn": ["no"] * 10 + ["yes"] * 10,
        }
    )
    pipeline = build_prediction_pipeline(
        df, ["score", "region"], "churn", "classification"
    )
    pipeline_path = str(tmp_path / "pipeline_cls.joblib")
    save_pipeline(pipeline, pipeline_path)
    X = pipeline.transform_df(df[["score", "region"]])
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(df["churn"])
    y = le.transform(df["churn"])
    model = LogisticRegression(random_state=0).fit(X, y)
    model_path = str(tmp_path / "model_cls.joblib")
    joblib.dump(model, model_path)
    return pipeline_path, model_path


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_pattern_which_feature_values_extreme():
    from api.chat import _FEATURE_SWEEP_PATTERNS

    assert _FEATURE_SWEEP_PATTERNS.search(
        "which feature values produce the most extreme predictions?"
    )


def test_pattern_what_inputs_maximize():
    from api.chat import _FEATURE_SWEEP_PATTERNS

    assert _FEATURE_SWEEP_PATTERNS.search("what inputs maximize the prediction")


def test_pattern_feature_sweep():
    from api.chat import _FEATURE_SWEEP_PATTERNS

    assert _FEATURE_SWEEP_PATTERNS.search("run a feature sweep")
    assert _FEATURE_SWEEP_PATTERNS.search("feature impact sweep analysis")


def test_pattern_what_combination_highest():
    from api.chat import _FEATURE_SWEEP_PATTERNS

    assert _FEATURE_SWEEP_PATTERNS.search(
        "what combination gives the highest prediction?"
    )


def test_pattern_which_inputs_biggest_impact():
    from api.chat import _FEATURE_SWEEP_PATTERNS

    assert _FEATURE_SWEEP_PATTERNS.search(
        "which inputs have the biggest prediction impact?"
    )


def test_pattern_most_impactful_feature_values():
    from api.chat import _FEATURE_SWEEP_PATTERNS

    assert _FEATURE_SWEEP_PATTERNS.search("what are the most impactful feature values?")


def test_pattern_extreme_prediction_analysis():
    from api.chat import _FEATURE_SWEEP_PATTERNS

    assert _FEATURE_SWEEP_PATTERNS.search("extreme prediction analysis")


def test_pattern_which_feature_moves_most():
    from api.chat import _FEATURE_SWEEP_PATTERNS

    assert _FEATURE_SWEEP_PATTERNS.search(
        "which feature moves the prediction the most?"
    )


def test_pattern_false_positive_unrelated():
    from api.chat import _FEATURE_SWEEP_PATTERNS

    assert not _FEATURE_SWEEP_PATTERNS.search("what is the average revenue?")
    assert not _FEATURE_SWEEP_PATTERNS.search("show me the confusion matrix")
    assert not _FEATURE_SWEEP_PATTERNS.search("which model should I use?")


def test_minimize_direction_re():
    from api.chat import _FEATURE_SWEEP_MINIMIZE_RE

    assert _FEATURE_SWEEP_MINIMIZE_RE.search("minimize the prediction")
    assert _FEATURE_SWEEP_MINIMIZE_RE.search("what gives the lowest prediction?")
    assert _FEATURE_SWEEP_MINIMIZE_RE.search("worst case prediction")
    assert not _FEATURE_SWEEP_MINIMIZE_RE.search("maximize revenue")
    assert not _FEATURE_SWEEP_MINIMIZE_RE.search("highest prediction")


# ---------------------------------------------------------------------------
# run_feature_sweep — pure function
# ---------------------------------------------------------------------------


def test_regression_maximize_required_fields(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_regression_deployment(tmp_path)
    result = run_feature_sweep(pipeline_path, model_path, direction="maximize")

    assert result["direction"] == "maximize"
    assert result["target_column"] == "revenue"
    assert result["problem_type"] == "regression"
    assert result["n_features"] > 0
    assert isinstance(result["features"], list)
    assert isinstance(result["optimal_config"], dict)
    assert "optimal_prediction" in result
    assert isinstance(result["summary"], str) and len(result["summary"]) > 0


def test_regression_minimize_direction(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_regression_deployment(tmp_path)
    result_max = run_feature_sweep(pipeline_path, model_path, direction="maximize")
    result_min = run_feature_sweep(pipeline_path, model_path, direction="minimize")

    assert result_max["direction"] == "maximize"
    assert result_min["direction"] == "minimize"
    # Minimize optimal prediction should be ≤ maximize optimal prediction
    assert float(result_min["optimal_prediction"]) <= float(
        result_max["optimal_prediction"]
    )


def test_features_sorted_by_delta_descending(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_regression_deployment(tmp_path)
    result = run_feature_sweep(pipeline_path, model_path)
    deltas = [f["delta"] for f in result["features"]]
    assert deltas == sorted(deltas, reverse=True)


def test_feature_rank_starts_at_one(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_regression_deployment(tmp_path)
    result = run_feature_sweep(pipeline_path, model_path)
    ranks = [f["rank"] for f in result["features"]]
    assert ranks[0] == 1
    assert ranks == list(range(1, len(ranks) + 1))


def test_optimal_config_contains_all_features(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_regression_deployment(tmp_path)
    result = run_feature_sweep(pipeline_path, model_path)
    for feat in result["features"]:
        assert feat["feature_name"] in result["optimal_config"]


def test_each_feature_has_required_fields(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_regression_deployment(tmp_path)
    result = run_feature_sweep(pipeline_path, model_path)
    for f in result["features"]:
        assert "feature_name" in f
        assert "feature_type" in f
        assert "best_value" in f
        assert "worst_value" in f
        assert "best_prediction" in f
        assert "worst_prediction" in f
        assert "delta" in f
        assert "rank" in f
        assert f["delta"] >= 0


def test_classification_uses_confidence(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_classification_deployment(tmp_path)
    result = run_feature_sweep(pipeline_path, model_path, direction="maximize")

    assert result["problem_type"] == "classification"
    assert result["n_features"] > 0
    # Confidence values are 0-1 fractions
    for f in result["features"]:
        assert 0.0 <= float(f["best_prediction"]) <= 1.0
        assert 0.0 <= float(f["worst_prediction"]) <= 1.0


def test_categorical_feature_included(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_classification_deployment(tmp_path)
    result = run_feature_sweep(pipeline_path, model_path)
    feat_types = {f["feature_name"]: f["feature_type"] for f in result["features"]}
    assert "region" in feat_types
    assert feat_types["region"] == "categorical"


def test_categorical_best_value_is_string(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_classification_deployment(tmp_path)
    result = run_feature_sweep(pipeline_path, model_path)
    region_feat = next(f for f in result["features"] if f["feature_name"] == "region")
    assert isinstance(region_feat["best_value"], str)
    assert isinstance(region_feat["worst_value"], str)


def test_n_steps_parameter(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_regression_deployment(tmp_path)
    result5 = run_feature_sweep(pipeline_path, model_path, n_steps=5)
    result20 = run_feature_sweep(pipeline_path, model_path, n_steps=20)
    # Both should produce valid output
    assert result5["n_features"] == result20["n_features"]


def test_units_has_larger_delta_than_constant_price(tmp_path):
    """'units' drives revenue linearly; 'price' is constant → units has larger delta."""
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_regression_deployment(tmp_path)
    result = run_feature_sweep(pipeline_path, model_path)
    feat_deltas = {f["feature_name"]: f["delta"] for f in result["features"]}
    # units varies 1-20 linearly → high delta; price is constant → delta ≈ 0
    assert feat_deltas.get("units", 0) >= feat_deltas.get("price", 0)


def test_summary_mentions_top_feature(tmp_path):
    from core.deployer import run_feature_sweep

    pipeline_path, model_path = _make_regression_deployment(tmp_path)
    result = run_feature_sweep(pipeline_path, model_path)
    if result["features"]:
        top_feat = result["features"][0]["feature_name"].replace("_", " ")
        assert top_feat in result["summary"]


# ---------------------------------------------------------------------------
# REST endpoint
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _sweep_test_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'sweep.db'}")
    monkeypatch.setattr(db_module, "engine", engine)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture()
def client():
    from main import app

    return TestClient(app)


def _create_real_deployment(tmp_path):
    """Return dep_id for a deployment with real pipeline + model files."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sqlmodel import Session

    from core.deployer import build_prediction_pipeline, save_pipeline
    from models.deployment import Deployment
    from models.model_run import ModelRun
    from models.project import Project

    df = pd.DataFrame(
        {
            "units": np.arange(1, 21, dtype=float),
            "price": np.ones(20) * 10.0,
            "revenue": np.arange(1, 21, dtype=float) * 5.0,
        }
    )
    pipeline = build_prediction_pipeline(
        df, ["units", "price"], "revenue", "regression"
    )
    pipeline_path = str(tmp_path / "sw_pipeline.joblib")
    save_pipeline(pipeline, pipeline_path)
    model = LinearRegression().fit(df[["units", "price"]].values, df["revenue"].values)
    model_path = str(tmp_path / "sw_model.joblib")
    joblib.dump(model, model_path)

    with Session(db_module.engine) as s:
        proj = Project(name=f"sweep-test-{id(db_module.engine)}")
        s.add(proj)
        s.commit()
        s.refresh(proj)
        run = ModelRun(
            project_id=proj.id,
            algorithm="linear_regression",
            status="done",
            model_path=model_path,
            metrics=json.dumps({"r2": 0.9}),
        )
        s.add(run)
        s.commit()
        s.refresh(run)
        dep = Deployment(
            model_run_id=run.id,
            project_id=proj.id,
            endpoint_path=f"/api/predict/{run.id}",
            dashboard_url=f"/predict/{run.id}",
            pipeline_path=pipeline_path,
            is_active=True,
            problem_type="regression",
        )
        s.add(dep)
        s.commit()
        return dep.id


def test_endpoint_404_unknown_deployment(client, tmp_path):
    resp = client.get("/api/deploy/nonexistent-sweep/feature-sweep")
    assert resp.status_code == 404


def test_endpoint_returns_required_fields(client, tmp_path):
    dep_id = _create_real_deployment(tmp_path)
    resp = client.get(f"/api/deploy/{dep_id}/feature-sweep")
    assert resp.status_code == 200
    data = resp.json()
    assert "direction" in data
    assert "features" in data
    assert "optimal_config" in data
    assert "optimal_prediction" in data
    assert "n_features" in data
    assert "summary" in data


def test_endpoint_default_direction_maximize(client, tmp_path):
    dep_id = _create_real_deployment(tmp_path)
    resp = client.get(f"/api/deploy/{dep_id}/feature-sweep")
    assert resp.status_code == 200
    assert resp.json()["direction"] == "maximize"


def test_endpoint_minimize_direction(client, tmp_path):
    dep_id = _create_real_deployment(tmp_path)
    resp = client.get(f"/api/deploy/{dep_id}/feature-sweep?direction=minimize")
    assert resp.status_code == 200
    assert resp.json()["direction"] == "minimize"


def test_endpoint_invalid_direction(client, tmp_path):
    dep_id = _create_real_deployment(tmp_path)
    resp = client.get(f"/api/deploy/{dep_id}/feature-sweep?direction=sideways")
    assert resp.status_code == 422


def test_endpoint_deployment_id_in_response(client, tmp_path):
    dep_id = _create_real_deployment(tmp_path)
    resp = client.get(f"/api/deploy/{dep_id}/feature-sweep")
    assert resp.status_code == 200
    assert resp.json().get("deployment_id") == dep_id
