"""Tests for production prediction explanation via chat and REST endpoint."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------

from api.chat import _PROD_EXPLAIN_PATTERNS

POSITIVE_PHRASES = [
    "explain the last prediction",
    "explain the latest prediction",
    "explain the most recent prediction",
    "explain the most recent production prediction",
    "explain the last api call",
    "explain the latest request",
    "why did the model give that result",
    "why did the model predict that answer",
    "why did the model output that value",
    "what drove the last production prediction",
    "what caused the latest prediction",
    "what influenced the most recent prediction",
    "feature contributions for the last prediction",
    "feature contributions for the latest api call",
    "feature contribution for the most recent request",
    "explain production prediction",
    "explain live prediction",
    "interpret the last prediction",
    "interpret the most recent production prediction",
    "explain prediction from the log",
    "breakdown prediction from history",
]

NEGATIVE_PHRASES = [
    "explain prediction for row 5",
    "explain record 3",
    "local model explanation",
    "waterfall chart",
    "show me the confusion matrix",
    "batch results",
    "schedule a prediction",
    "run a prediction with units=100",
]


@pytest.mark.parametrize("phrase", POSITIVE_PHRASES)
def test_prod_explain_pattern_positive(phrase):
    assert _PROD_EXPLAIN_PATTERNS.search(phrase), f"Should match: {phrase!r}"


@pytest.mark.parametrize("phrase", NEGATIVE_PHRASES)
def test_prod_explain_pattern_negative(phrase):
    assert not _PROD_EXPLAIN_PATTERNS.search(phrase), f"Should not match: {phrase!r}"


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    """FastAPI test client with isolated test DB."""
    test_db = str(tmp_path / "test.db")
    from sqlmodel import SQLModel, create_engine as _ce
    import db as db_mod_inner

    test_engine = _ce(f"sqlite:///{test_db}")
    original_engine = db_mod_inner.engine
    db_mod_inner.engine = test_engine

    SQLModel.metadata.create_all(test_engine)
    db_mod_inner._apply_migrations()

    from fastapi.testclient import TestClient
    from main import app

    yield TestClient(app)

    db_mod_inner.engine = original_engine


def _setup_deployment(session, tmp_path):
    """Create minimal Deployment + ModelRun + PredictionLog fixtures."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LinearRegression

    from core.deployer import PredictionPipeline, save_pipeline
    from models.dataset import Dataset
    from models.deployment import Deployment
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.prediction_log import PredictionLog
    from models.project import Project

    # Project
    project = Project(name="TestProject")
    session.add(project)
    session.flush()

    # Dataset
    dataset = Dataset(
        project_id=project.id,
        filename="test.csv",
        file_path=str(tmp_path / "test.csv"),
        row_count=50,
        column_count=3,
    )
    session.add(dataset)
    session.flush()

    # Feature set
    fs = FeatureSet(
        project_id=project.id,
        dataset_id=dataset.id,
        target_column="revenue",
        problem_type="regression",
        transformations="[]",
    )
    session.add(fs)
    session.flush()

    # Trained model
    X = np.random.rand(50, 2)
    y = X[:, 0] * 3 + X[:, 1] * 1.5 + np.random.rand(50) * 0.1
    model = LinearRegression().fit(X, y)
    model_path = str(tmp_path / "model.joblib")
    joblib.dump(model, model_path)

    # Pipeline
    pipeline = PredictionPipeline(
        feature_names=["units", "price"],
        column_types={"units": "numeric", "price": "numeric"},
        medians={"units": 5.0, "price": 10.0},
        target_column="revenue",
        problem_type="regression",
        feature_means={"units": 5.0, "price": 10.0},
        feature_stds={"units": 1.0, "price": 2.0},
    )
    pipeline_path = tmp_path / "pipeline.joblib"
    save_pipeline(pipeline, pipeline_path)

    # ModelRun
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

    # Deployment
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

    # PredictionLog
    log = PredictionLog(
        deployment_id=dep.id,
        input_features=json.dumps({"units": 8.0, "price": 15.0}),
        prediction=json.dumps(40.5),
        prediction_numeric=40.5,
        confidence=None,
    )
    session.add(log)
    session.commit()

    return dep, run, log


def test_explain_most_recent_prediction(tmp_path):
    """REST endpoint returns feature contributions for the most recent PredictionLog."""
    from fastapi.testclient import TestClient
    from sqlmodel import Session, SQLModel, create_engine
    import db as db_mod

    test_db = str(tmp_path / "test.db")
    orig_engine = db_mod.engine
    db_mod.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(db_mod.engine)
    db_mod._apply_migrations()

    from main import app

    with Session(db_mod.engine) as session:
        dep, run, log = _setup_deployment(session, tmp_path)
        dep_id = dep.id
        log_id = log.id

    client = TestClient(app)
    resp = client.get(f"/api/deploy/{dep_id}/explain-prediction")
    assert resp.status_code == 200, (
        f"Expected 200 but got {resp.status_code}: {resp.text}"
    )
    data = resp.json()
    assert "contributions" in data
    assert isinstance(data["contributions"], list)
    assert len(data["contributions"]) == 2
    assert data["prediction_log_id"] == log_id
    assert data["target_column"] == "revenue"
    assert data["problem_type"] == "regression"
    assert data["algorithm"] == "linear_regression"
    assert "summary" in data

    db_mod.engine = orig_engine


def test_explain_by_prediction_id(tmp_path):
    """REST endpoint accepts an explicit prediction_id query param."""
    from fastapi.testclient import TestClient
    from sqlmodel import Session, SQLModel, create_engine
    import db as db_mod

    test_db = str(tmp_path / "test.db")
    orig_engine = db_mod.engine
    db_mod.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(db_mod.engine)
    db_mod._apply_migrations()

    from main import app

    with Session(db_mod.engine) as session:
        dep, run, log = _setup_deployment(session, tmp_path)
        dep_id = dep.id
        log_id = log.id

    client = TestClient(app)
    resp = client.get(f"/api/deploy/{dep_id}/explain-prediction?prediction_id={log_id}")
    assert resp.status_code == 200, (
        f"Expected 200 but got {resp.status_code}: {resp.text}"
    )
    data = resp.json()
    assert data["prediction_log_id"] == log_id

    db_mod.engine = orig_engine


def test_explain_no_predictions_404(tmp_path):
    """Returns 404 when no PredictionLog records exist for the deployment."""
    import json as _j

    import joblib
    import numpy as np
    from fastapi.testclient import TestClient
    from sklearn.linear_model import LinearRegression
    from sqlmodel import Session, SQLModel, create_engine

    import db as db_mod
    from core.deployer import PredictionPipeline, save_pipeline
    from models.dataset import Dataset
    from models.deployment import Deployment
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project

    test_db = str(tmp_path / "test.db")
    orig_engine = db_mod.engine
    db_mod.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(db_mod.engine)
    db_mod._apply_migrations()

    from main import app

    with Session(db_mod.engine) as session:
        project = Project(name="EmptyProject")
        session.add(project)
        session.flush()

        dataset = Dataset(
            project_id=project.id,
            filename="t.csv",
            file_path=str(tmp_path / "t.csv"),
            row_count=10,
            column_count=2,
        )
        session.add(dataset)
        session.flush()

        fs = FeatureSet(
            project_id=project.id,
            dataset_id=dataset.id,
            target_column="y",
            problem_type="regression",
            transformations="[]",
        )
        session.add(fs)
        session.flush()

        X = np.random.rand(10, 1)
        y = X.ravel()
        model = LinearRegression().fit(X, y)
        model_path = str(tmp_path / "model2.joblib")
        joblib.dump(model, model_path)

        pipeline = PredictionPipeline(
            feature_names=["x"],
            column_types={"x": "numeric"},
            medians={"x": 0.5},
            target_column="y",
            problem_type="regression",
        )
        pipeline_path = tmp_path / "pipeline2.joblib"
        save_pipeline(pipeline, pipeline_path)

        run = ModelRun(
            project_id=project.id,
            feature_set_id=fs.id,
            algorithm="linear_regression",
            status="done",
            model_path=model_path,
            metrics=_j.dumps({"r2": 0.9}),
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
            target_column="y",
            feature_names=_j.dumps(["x"]),
        )
        session.add(dep)
        session.commit()
        dep_id = dep.id

    client = TestClient(app)
    resp = client.get(f"/api/deploy/{dep_id}/explain-prediction")
    assert resp.status_code == 404

    db_mod.engine = orig_engine


def test_explain_inactive_deployment_404(tmp_path):
    """Returns 404 for an inactive or missing deployment."""
    from fastapi.testclient import TestClient
    from sqlmodel import SQLModel, create_engine

    import db as db_mod

    test_db = str(tmp_path / "test.db")
    orig_engine = db_mod.engine
    db_mod.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(db_mod.engine)
    db_mod._apply_migrations()

    from main import app

    client = TestClient(app)
    resp = client.get("/api/deploy/nonexistent-id/explain-prediction")
    assert resp.status_code == 404

    db_mod.engine = orig_engine


# ---------------------------------------------------------------------------
# Chat pattern integration tests (mocked Anthropic)
# ---------------------------------------------------------------------------


@pytest.fixture
def anthropic_mock():
    mock_client = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    mock_event = MagicMock()
    mock_event.type = "content_block_delta"
    mock_event.delta = MagicMock()
    mock_event.delta.type = "text_delta"
    mock_event.delta.text = "Great question!"

    mock_stream.__iter__ = MagicMock(return_value=iter([mock_event]))
    mock_client.messages.stream.return_value = mock_stream

    patcher = patch("api.chat.anthropic")
    mock_anthropic_module = patcher.start()
    mock_anthropic_module.Anthropic.return_value = mock_client
    yield mock_client
    patcher.stop()


def test_chat_emits_prod_explain_event(tmp_path, anthropic_mock):
    """Chat handler emits prod_prediction_explanation event when deployment + logs exist."""
    from fastapi.testclient import TestClient
    from sqlmodel import Session, SQLModel, create_engine
    import db as db_mod

    test_db = str(tmp_path / "test.db")
    orig_engine = db_mod.engine
    db_mod.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )

    from main import app

    SQLModel.metadata.create_all(db_mod.engine)
    db_mod._apply_migrations()

    with Session(db_mod.engine) as session:
        dep, run, log = _setup_deployment(session, tmp_path)
        project_id = dep.project_id

    client = TestClient(app)
    resp = client.post(
        f"/api/chat/{project_id}",
        json={"message": "explain the most recent production prediction"},
    )
    assert resp.status_code == 200, (
        f"Expected 200 but got {resp.status_code}: {resp.text}"
    )
    body = resp.text
    assert "prod_prediction_explanation" in body

    db_mod.engine = orig_engine


def test_chat_no_event_without_deployment(tmp_path, anthropic_mock):
    """Chat handler skips prod_explain when no deployment exists for the project."""
    from fastapi.testclient import TestClient
    from sqlmodel import Session, SQLModel, create_engine
    import db as db_mod
    from models.project import Project

    test_db = str(tmp_path / "test.db")
    orig_engine = db_mod.engine
    db_mod.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )

    from main import app

    SQLModel.metadata.create_all(db_mod.engine)
    db_mod._apply_migrations()

    with Session(db_mod.engine) as session:
        project = Project(name="NoDeploy")
        session.add(project)
        session.commit()
        project_id = project.id

    client = TestClient(app)
    resp = client.post(
        f"/api/chat/{project_id}",
        json={"message": "explain the last prediction"},
    )
    assert resp.status_code == 200, (
        f"Expected 200 but got {resp.status_code}: {resp.text}"
    )
    assert "prod_prediction_explanation" not in resp.text

    db_mod.engine = orig_engine
