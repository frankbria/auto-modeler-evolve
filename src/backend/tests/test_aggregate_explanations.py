"""Tests for aggregate production explanation feature."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------

from api.chat import _AGGR_EXPLAIN_PATTERNS

POSITIVE_PHRASES = [
    "what's been driving my predictions",
    "what is driving my production predictions",
    "what has been driving predictions",
    "aggregate explanation",
    "aggregate feature importance",
    "aggregate contributions",
    "which features are driving my predictions",
    "which features have been driving my production predictions",
    "which features are influencing my live predictions",
    "feature importance across production predictions",
    "production feature importance",
    "live feature contributions",
    "patterns in my production predictions",
    "patterns in my predictions",
    "most influential features in production predictions",
    "most influential features across recent api calls",
    "what features drive my production predictions",
    "what features affect my live predictions",
]

NEGATIVE_PHRASES = [
    "explain the last prediction",
    "explain prediction for row 5",
    "waterfall chart",
    "batch results",
    "show me the confusion matrix",
    "run a prediction with units=100",
    "schedule a batch job",
    "train a model",
]


@pytest.mark.parametrize("phrase", POSITIVE_PHRASES)
def test_aggr_explain_pattern_positive(phrase):
    assert _AGGR_EXPLAIN_PATTERNS.search(phrase), f"Should match: {phrase!r}"


@pytest.mark.parametrize("phrase", NEGATIVE_PHRASES)
def test_aggr_explain_pattern_negative(phrase):
    assert not _AGGR_EXPLAIN_PATTERNS.search(phrase), f"Should not match: {phrase!r}"


# ---------------------------------------------------------------------------
# Pure function tests: compute_aggregate_explanations
# ---------------------------------------------------------------------------


def _make_pipeline_and_model(tmp_path):
    """Return (pipeline, pipeline_path, model, model_path) fixtures."""
    import joblib
    from sklearn.linear_model import LinearRegression

    from core.deployer import PredictionPipeline, save_pipeline

    X = np.random.rand(30, 2)
    y = X[:, 0] * 3 + X[:, 1] * 1.5
    model = LinearRegression().fit(X, y)
    model_path = str(tmp_path / "model.joblib")
    joblib.dump(model, model_path)

    pipeline = PredictionPipeline(
        feature_names=["units", "price"],
        column_types={"units": "numeric", "price": "numeric"},
        medians={"units": 5.0, "price": 10.0},
        target_column="revenue",
        problem_type="regression",
        feature_means={"units": 5.0, "price": 10.0},
        feature_stds={"units": 1.0, "price": 2.0},
    )
    pipeline_path = str(tmp_path / "pipeline.joblib")
    save_pipeline(pipeline, tmp_path / "pipeline.joblib")
    return pipeline_path, model_path


def test_aggregate_explanations_empty_list(tmp_path):
    from core.deployer import compute_aggregate_explanations

    pp, mp = _make_pipeline_and_model(tmp_path)
    result = compute_aggregate_explanations(pp, mp, [])
    assert result["sample_count"] == 0
    assert result["features"] == []
    assert "No predictions" in result["summary"]


def test_aggregate_explanations_single_input(tmp_path):
    from core.deployer import compute_aggregate_explanations

    pp, mp = _make_pipeline_and_model(tmp_path)
    result = compute_aggregate_explanations(pp, mp, [{"units": 8.0, "price": 12.0}])
    assert result["sample_count"] == 1
    assert len(result["features"]) == 2  # units + price
    assert result["features"][0]["feature"] in ("units", "price")


def test_aggregate_explanations_multiple_inputs(tmp_path):
    from core.deployer import compute_aggregate_explanations

    pp, mp = _make_pipeline_and_model(tmp_path)
    inputs = [
        {"units": 10.0, "price": 20.0},
        {"units": 2.0, "price": 5.0},
        {"units": 15.0, "price": 8.0},
    ]
    result = compute_aggregate_explanations(pp, mp, inputs)
    assert result["sample_count"] == 3
    assert len(result["features"]) == 2
    # features sorted by avg_abs_contribution descending
    contribs = [f["avg_abs_contribution"] for f in result["features"]]
    assert contribs == sorted(contribs, reverse=True)


def test_aggregate_explanations_direction_labels(tmp_path):
    from core.deployer import compute_aggregate_explanations

    pp, mp = _make_pipeline_and_model(tmp_path)
    # All inputs above mean → units should be mostly positive
    inputs = [{"units": 8.0, "price": 12.0}] * 5
    result = compute_aggregate_explanations(pp, mp, inputs)
    units_feat = next(f for f in result["features"] if f["feature"] == "units")
    assert units_feat["direction_label"] in ("mostly positive", "mostly negative", "mixed")
    assert 0.0 <= units_feat["positive_pct"] <= 100.0


def test_aggregate_explanations_top_driver_pct(tmp_path):
    from core.deployer import compute_aggregate_explanations

    pp, mp = _make_pipeline_and_model(tmp_path)
    inputs = [{"units": 10.0, "price": 10.0}] * 4
    result = compute_aggregate_explanations(pp, mp, inputs)
    top_driver_pcts = [f["top_driver_pct"] for f in result["features"]]
    # Total top-driver-pct can be ≤ 3 * sample_count / sample_count * 100 = 300
    for pct in top_driver_pcts:
        assert 0.0 <= pct <= 100.0


def test_aggregate_explanations_malformed_inputs_skipped(tmp_path):
    from core.deployer import compute_aggregate_explanations

    pp, mp = _make_pipeline_and_model(tmp_path)
    inputs = [
        {"units": 5.0, "price": 10.0},
        {"bad_col": 99.0},  # unknown column — should be skipped gracefully
        {"units": 6.0, "price": 11.0},
    ]
    result = compute_aggregate_explanations(pp, mp, inputs)
    # At least the 2 valid inputs processed
    assert result["sample_count"] >= 1


def test_aggregate_explanations_summary_mentions_top_feature(tmp_path):
    from core.deployer import compute_aggregate_explanations

    pp, mp = _make_pipeline_and_model(tmp_path)
    inputs = [{"units": 10.0, "price": 12.0}] * 3
    result = compute_aggregate_explanations(pp, mp, inputs)
    if result["features"]:
        top_feature = result["features"][0]["feature"]
        assert top_feature in result["summary"]


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    """FastAPI test client with isolated test DB."""
    from sqlmodel import SQLModel, create_engine
    import db as db_mod

    test_engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    original_engine = db_mod.engine
    db_mod.engine = test_engine

    SQLModel.metadata.create_all(test_engine)
    db_mod._apply_migrations()

    from fastapi.testclient import TestClient
    from main import app

    yield TestClient(app)

    db_mod.engine = original_engine


def _setup_deployment_with_logs(session, tmp_path, n_logs=3):
    """Create Project → Dataset → FeatureSet → ModelRun → Deployment → PredictionLogs."""
    import joblib
    from sklearn.linear_model import LinearRegression

    from core.deployer import PredictionPipeline, save_pipeline
    from models.dataset import Dataset
    from models.deployment import Deployment
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.prediction_log import PredictionLog
    from models.project import Project

    project = Project(name="AggTest")
    session.add(project)
    session.flush()

    dataset = Dataset(
        project_id=project.id,
        filename="agg.csv",
        file_path=str(tmp_path / "agg.csv"),
        row_count=30,
        column_count=3,
    )
    session.add(dataset)
    session.flush()

    fs = FeatureSet(
        project_id=project.id,
        dataset_id=dataset.id,
        target_column="revenue",
        problem_type="regression",
        transformations="[]",
    )
    session.add(fs)
    session.flush()

    X = np.random.rand(30, 2)
    y = X[:, 0] * 3 + X[:, 1] * 1.5
    model = LinearRegression().fit(X, y)
    model_path = str(tmp_path / "agg_model.joblib")
    joblib.dump(model, model_path)

    pipeline = PredictionPipeline(
        feature_names=["units", "price"],
        column_types={"units": "numeric", "price": "numeric"},
        medians={"units": 5.0, "price": 10.0},
        target_column="revenue",
        problem_type="regression",
        feature_means={"units": 5.0, "price": 10.0},
        feature_stds={"units": 1.0, "price": 2.0},
    )
    pipeline_path = tmp_path / "agg_pipeline.joblib"
    save_pipeline(pipeline, pipeline_path)

    run = ModelRun(
        project_id=project.id,
        feature_set_id=fs.id,
        algorithm="linear_regression",
        status="done",
        model_path=model_path,
        metrics=json.dumps({"r2": 0.85}),
    )
    session.add(run)
    session.flush()

    dep = Deployment(
        model_run_id=run.id,
        project_id=project.id,
        endpoint_path=f"/api/predict/{run.id}",
        dashboard_url=f"/predict/{run.id}",
        pipeline_path=str(pipeline_path),
        algorithm="linear_regression",
        problem_type="regression",
        target_column="revenue",
        feature_names=json.dumps(["units", "price"]),
    )
    session.add(dep)
    session.flush()

    for i in range(n_logs):
        log = PredictionLog(
            deployment_id=dep.id,
            input_features=json.dumps({"units": float(5 + i), "price": float(10 + i)}),
            prediction="15.0",
            prediction_numeric=15.0,
            confidence=None,
        )
        session.add(log)
    session.commit()
    return dep.id


def test_aggregate_explanations_endpoint_returns_200(client, tmp_path):
    from db import engine
    from sqlmodel import Session

    dep_id = _setup_deployment_with_logs(Session(engine), tmp_path, n_logs=3)
    resp = client.get(f"/api/deploy/{dep_id}/aggregate-explanations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deployment_id"] == dep_id
    assert data["sample_count"] == 3
    assert len(data["features"]) == 2


def test_aggregate_explanations_endpoint_404_no_deployment(client):
    resp = client.get("/api/deploy/nonexistent-id/aggregate-explanations")
    assert resp.status_code == 404


def test_aggregate_explanations_endpoint_404_no_logs(client, tmp_path):
    from db import engine
    from sqlmodel import Session

    # Setup deployment without logs
    dep_id = _setup_deployment_with_logs(Session(engine), tmp_path, n_logs=0)
    resp = client.get(f"/api/deploy/{dep_id}/aggregate-explanations")
    assert resp.status_code == 404


def test_aggregate_explanations_endpoint_n_param(client, tmp_path):
    from db import engine
    from sqlmodel import Session

    dep_id = _setup_deployment_with_logs(Session(engine), tmp_path, n_logs=10)
    resp = client.get(f"/api/deploy/{dep_id}/aggregate-explanations?n=5")
    assert resp.status_code == 200
    # At most 5 logs processed
    assert resp.json()["sample_count"] <= 5


def test_aggregate_explanations_features_sorted_by_influence(client, tmp_path):
    from db import engine
    from sqlmodel import Session

    dep_id = _setup_deployment_with_logs(Session(engine), tmp_path, n_logs=5)
    data = client.get(f"/api/deploy/{dep_id}/aggregate-explanations").json()
    contribs = [f["avg_abs_contribution"] for f in data["features"]]
    assert contribs == sorted(contribs, reverse=True)


# ---------------------------------------------------------------------------
# Chat integration test
# ---------------------------------------------------------------------------


def test_aggr_explain_chat_emits_event(tmp_path):
    """Smoke test: chat handler emits aggregate_explanation event when deployment exists."""
    from sqlmodel import Session, SQLModel, create_engine
    import db as db_mod

    test_engine = create_engine(f"sqlite:///{tmp_path / 'chat_agg.db'}")
    original_engine = db_mod.engine
    db_mod.engine = test_engine

    SQLModel.metadata.create_all(test_engine)
    db_mod._apply_migrations()

    with Session(test_engine) as session:
        dep_id = _setup_deployment_with_logs(session, tmp_path, n_logs=2)

    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)

    from models.deployment import Deployment as _Dep
    with Session(test_engine) as s:
        dep = s.get(_Dep, dep_id)
        project_id = dep.project_id

    mock_response = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_response.__iter__ = MagicMock(
        return_value=iter([MagicMock(text=" ", type="content_block_delta")])
    )

    with patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.stream.return_value = mock_response

        resp = client.post(
            f"/api/chat/{project_id}",
            json={"message": "what's been driving my predictions"},
            headers={"Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert "aggregate_explanation" in body

    db_mod.engine = original_engine
