"""Tests for goal-seek / reverse prediction feature.

Covers:
- _GOAL_SEEK_PATTERNS regex matching (6 tests)
- _extract_goal_seek_target helper (4 tests)
- run_goal_seek pure function (10 tests)
- POST /api/deploy/{id}/goal-seek endpoint (3 tests)
- Chat integration: SSE event emitted (2 tests)
"""
from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression, LogisticRegression

from api.chat import (
    _GOAL_SEEK_PATTERNS,
    _extract_goal_seek_target,
)
from core.deployer import PredictionPipeline, run_goal_seek
from db import get_session
from main import app
from models.dataset import Dataset
from models.deployment import Deployment
from models.feature_set import FeatureSet
from models.model_run import ModelRun
from models.project import Project


# ---------------------------------------------------------------------------
# Fixtures: build a minimal pipeline + model for testing
# ---------------------------------------------------------------------------


def _build_reg_pipeline(tmpdir: Path, n: int = 100) -> tuple[str, str]:
    """Return (pipeline_path, model_path) for a tiny regression model."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + rng.standard_normal(n) * 0.1

    model = LinearRegression()
    model.fit(X, y)

    pipeline = PredictionPipeline(
        feature_names=["units", "price", "discount"],
        column_types={"units": "numeric", "price": "numeric", "discount": "numeric"},
        problem_type="regression",
        target_column="revenue",
        feature_means={"units": float(X[:, 0].mean()), "price": float(X[:, 1].mean()), "discount": float(X[:, 2].mean())},
        feature_stds={"units": float(X[:, 0].std()), "price": float(X[:, 1].std()), "discount": float(X[:, 2].std())},
        feature_ranges={
            "units": {"p5": float(np.percentile(X[:, 0], 5)), "p95": float(np.percentile(X[:, 0], 95))},
            "price": {"p5": float(np.percentile(X[:, 1], 5)), "p95": float(np.percentile(X[:, 1], 95))},
            "discount": {"p5": float(np.percentile(X[:, 2], 5)), "p95": float(np.percentile(X[:, 2], 95))},
        },
    )
    pipeline_path = str(tmpdir / "pipeline.joblib")
    model_path = str(tmpdir / "model.joblib")
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(model, model_path)
    return pipeline_path, model_path


def _build_cls_pipeline(tmpdir: Path, n: int = 100) -> tuple[str, str]:
    """Return (pipeline_path, model_path) for a tiny binary classification model."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    pipeline = PredictionPipeline(
        feature_names=["age", "score"],
        column_types={"age": "numeric", "score": "numeric"},
        problem_type="classification",
        target_column="churn",
        target_classes=[0, 1],
        feature_means={"age": float(X[:, 0].mean()), "score": float(X[:, 1].mean())},
        feature_stds={"age": float(X[:, 0].std()), "score": float(X[:, 1].std())},
        feature_ranges={
            "age": {"p5": float(np.percentile(X[:, 0], 5)), "p95": float(np.percentile(X[:, 0], 95))},
            "score": {"p5": float(np.percentile(X[:, 1], 5)), "p95": float(np.percentile(X[:, 1], 95))},
        },
    )
    pipeline_path = str(tmpdir / "cls_pipeline.joblib")
    model_path = str(tmpdir / "cls_model.joblib")
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(model, model_path)
    return pipeline_path, model_path


# ---------------------------------------------------------------------------
# Pattern tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "goal seek",
        "what inputs would produce a revenue of $1M?",
        "what combination would achieve revenue 1000000?",
        "reverse prediction",
        "optimize my inputs to reach target",
        "what would it take to hit $500K",
        "how do I reach revenue of 2000",
        "find inputs to get revenue above 1000",
    ],
)
def test_goal_seek_patterns_match(message):
    assert _GOAL_SEEK_PATTERNS.search(message), f"Pattern should match: {message!r}"


@pytest.mark.parametrize(
    "message",
    [
        "train a model",
        "show me my data",
        "how is my model accuracy?",
    ],
)
def test_goal_seek_patterns_no_false_positives(message):
    assert not _GOAL_SEEK_PATTERNS.search(message), f"Pattern should NOT match: {message!r}"


# ---------------------------------------------------------------------------
# _extract_goal_seek_target tests
# ---------------------------------------------------------------------------


def test_extract_goal_seek_regression_number():
    target, fixed = _extract_goal_seek_target(
        "what inputs would give me revenue of 1000000?",
        "regression",
        "revenue",
        None,
    )
    assert target == 1000000.0
    assert fixed == {}


def test_extract_goal_seek_regression_millions():
    target, fixed = _extract_goal_seek_target(
        "what would it take to reach $2M?",
        "regression",
        "revenue",
        None,
    )
    assert target == 2_000_000.0


def test_extract_goal_seek_classification_class():
    target, fixed = _extract_goal_seek_target(
        "what inputs would predict churn = '1'?",
        "classification",
        "churn",
        [0, 1],
    )
    assert str(target) == "1"


def test_extract_goal_seek_classification_fallback_positive():
    target, fixed = _extract_goal_seek_target(
        "how do I get a positive prediction?",
        "classification",
        "outcome",
        ["no", "yes"],
    )
    assert str(target) == "yes"


# ---------------------------------------------------------------------------
# run_goal_seek pure function tests
# ---------------------------------------------------------------------------


def test_goal_seek_regression_returns_dict(tmp_path):
    pp, mp = _build_reg_pipeline(tmp_path)
    result = run_goal_seek(pp, mp, target_value=5.0, algorithm="linear_regression")

    assert "target_column" in result
    assert result["target_column"] == "revenue"
    assert result["problem_type"] == "regression"
    assert isinstance(result["achieved_value"], (int, float, str))
    assert isinstance(result["suggestions"], list)
    assert isinstance(result["achieved"], bool)
    assert "summary" in result


def test_goal_seek_regression_suggestions_structure(tmp_path):
    pp, mp = _build_reg_pipeline(tmp_path)
    result = run_goal_seek(pp, mp, target_value=3.0)

    for s in result["suggestions"]:
        assert "feature" in s
        assert "current_mean" in s
        assert "suggested_value" in s
        assert s["direction"] in ("increase", "decrease", "no_change")
        assert isinstance(s["change_pct"], float)


def test_goal_seek_regression_gap_pct_present(tmp_path):
    pp, mp = _build_reg_pipeline(tmp_path)
    result = run_goal_seek(pp, mp, target_value=100.0)  # very large target
    assert result["gap_pct"] is not None
    assert result["gap_pct"] >= 0


def test_goal_seek_classification_returns_dict(tmp_path):
    pp, mp = _build_cls_pipeline(tmp_path)
    result = run_goal_seek(pp, mp, target_value="1", algorithm="logistic_regression")

    assert result["problem_type"] == "classification"
    assert result["target_column"] == "churn"
    assert result["gap_pct"] is None
    assert isinstance(result["achieved"], bool)


def test_goal_seek_classification_summary_not_empty(tmp_path):
    pp, mp = _build_cls_pipeline(tmp_path)
    result = run_goal_seek(pp, mp, target_value="1")
    assert len(result["summary"]) > 0


def test_goal_seek_fixed_features_respected(tmp_path):
    pp, mp = _build_reg_pipeline(tmp_path)
    fixed = {"discount": 0.5}
    result = run_goal_seek(pp, mp, target_value=2.0, fixed_features=fixed)
    assert result["fixed_features"] == {"discount": 0.5}
    # discount should not appear in suggestions (it's fixed)
    feat_names = {s["feature"] for s in result["suggestions"]}
    assert "discount" not in feat_names


def test_goal_seek_suggestions_capped_at_8(tmp_path):
    """Even with many features, suggestions are capped at 8."""
    pp, mp = _build_reg_pipeline(tmp_path)
    result = run_goal_seek(pp, mp, target_value=10.0)
    assert len(result["suggestions"]) <= 8


def test_goal_seek_algorithm_plain_set(tmp_path):
    pp, mp = _build_reg_pipeline(tmp_path)
    result = run_goal_seek(pp, mp, target_value=1.0, algorithm="gradient_boosting_regressor")
    assert result["algorithm_plain"] == "Gradient Boosting"


def test_goal_seek_unknown_algorithm_plain(tmp_path):
    pp, mp = _build_reg_pipeline(tmp_path)
    result = run_goal_seek(pp, mp, target_value=1.0, algorithm="custom_algo")
    assert len(result["algorithm_plain"]) > 0  # falls back to title-cased name


def test_goal_seek_n_optimized_matches_free_features(tmp_path):
    pp, mp = _build_reg_pipeline(tmp_path)
    result = run_goal_seek(pp, mp, target_value=2.0)
    # No categorical features, no fixed — all 3 features should be optimized
    assert result["n_optimized"] == 3


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_with_gs(tmp_path) -> Generator[tuple[TestClient, str], None, None]:
    """TestClient wired with a real regression deployment for goal-seek tests."""
    pp, mp = _build_reg_pipeline(tmp_path)

    session_data: dict = {}

    def _override():
        from sqlmodel import Session, create_engine

        from models.project import Project
        from sqlmodel import SQLModel

        eng = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(eng)
        with Session(eng) as s:
            proj = Project(id="gs-proj-1", name="GSTest")
            s.add(proj)
            ds = Dataset(
                id="gs-ds-1",
                project_id="gs-proj-1",
                filename="test.csv",
                file_path=str(tmp_path / "test.csv"),
                row_count=100,
                column_count=3,
            )
            (tmp_path / "test.csv").write_text("units,price,discount,revenue\n1,2,3,4\n")
            s.add(ds)
            fs_obj = FeatureSet(
                id="gs-fs-1",
                project_id="gs-proj-1",
                dataset_id="gs-ds-1",
                target_column="revenue",
                problem_type="regression",
                feature_columns='["units","price","discount"]',
            )
            s.add(fs_obj)
            run_obj = ModelRun(
                id="gs-run-1",
                project_id="gs-proj-1",
                feature_set_id="gs-fs-1",
                algorithm="linear_regression",
                status="done",
                model_path=mp,
                metrics='{"r2": 0.95}',
                is_selected=True,
            )
            s.add(run_obj)
            s.commit()
            dep_obj = Deployment(
                id="gs-dep-1",
                project_id="gs-proj-1",
                model_run_id="gs-run-1",
                target_column="revenue",
                pipeline_path=pp,
                is_active=True,
                endpoint_path="/api/predict/gs-dep-1",
                dashboard_url="/predict/gs-dep-1",
            )
            s.add(dep_obj)
            s.commit()
            session_data["session"] = s
            yield s

    app.dependency_overrides[get_session] = _override
    client = TestClient(app)
    yield client, "gs-dep-1"
    app.dependency_overrides.clear()


def test_goal_seek_endpoint_returns_result(client_with_gs):
    client, dep_id = client_with_gs
    resp = client.post(
        f"/api/deploy/{dep_id}/goal-seek",
        json={"target_value": 2.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "target_column" in data
    assert "achieved_value" in data
    assert "suggestions" in data


def test_goal_seek_endpoint_not_found(client_with_gs):
    client, _ = client_with_gs
    resp = client.post(
        "/api/deploy/nonexistent-dep/goal-seek",
        json={"target_value": 1.0},
    )
    assert resp.status_code == 404


def test_goal_seek_endpoint_with_fixed_features(client_with_gs):
    client, dep_id = client_with_gs
    resp = client.post(
        f"/api/deploy/{dep_id}/goal-seek",
        json={"target_value": 1.5, "fixed_features": {"discount": 0.3}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fixed_features"] == {"discount": 0.3}


# ---------------------------------------------------------------------------
# Chat integration — SSE event
# ---------------------------------------------------------------------------


def _make_chat_client(tmp_path, db_path="sqlite:///:memory:"):
    """Build a TestClient with a real deployment for chat goal-seek tests."""
    pp, mp = _build_reg_pipeline(tmp_path)

    def _override():
        from sqlmodel import Session, create_engine

        from models.project import Project
        from sqlmodel import SQLModel

        eng = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(eng)
        with Session(eng) as s:
            proj = Project(id="chat-gs-proj", name="ChatGSTest")
            s.add(proj)
            ds = Dataset(
                id="chat-gs-ds",
                project_id="chat-gs-proj",
                filename="test.csv",
                file_path=str(tmp_path / "chat_test.csv"),
                row_count=100,
                column_count=3,
            )
            (tmp_path / "chat_test.csv").write_text("units,price,discount,revenue\n1,2,3,4\n")
            s.add(ds)
            fs_obj = FeatureSet(
                id="chat-gs-fs",
                project_id="chat-gs-proj",
                dataset_id="chat-gs-ds",
                target_column="revenue",
                problem_type="regression",
                feature_columns='["units","price","discount"]',
            )
            s.add(fs_obj)
            run_obj = ModelRun(
                id="chat-gs-run",
                project_id="chat-gs-proj",
                feature_set_id="chat-gs-fs",
                algorithm="linear_regression",
                status="done",
                model_path=mp,
                metrics='{"r2": 0.95}',
                is_selected=True,
            )
            s.add(run_obj)
            s.commit()
            dep_obj = Deployment(
                id="chat-gs-dep",
                project_id="chat-gs-proj",
                model_run_id="chat-gs-run",
                target_column="revenue",
                pipeline_path=pp,
                is_active=True,
                endpoint_path="/api/predict/chat-gs-dep",
                dashboard_url="/predict/chat-gs-dep",
            )
            s.add(dep_obj)
            s.commit()
            yield s

    app.dependency_overrides[get_session] = _override
    return TestClient(app), "chat-gs-proj"


@patch("anthropic.Anthropic")
def test_chat_goal_seek_emits_sse_event(mock_anthropic, tmp_path):
    """Chat returns a goal_seek SSE event for a goal-seek request."""
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(["The goal seek result is ready."])
    mock_client.messages.stream.return_value = mock_stream

    client, project_id = _make_chat_client(tmp_path)
    try:
        resp = client.post(
            f"/api/chat/{project_id}",
            json={"message": "what inputs would produce a revenue of 2.0?"},
        )
        assert resp.status_code == 200
        content = resp.text
        assert "goal_seek" in content
    finally:
        app.dependency_overrides.clear()


@patch("anthropic.Anthropic")
def test_chat_goal_seek_not_triggered_without_deployment(mock_anthropic, tmp_path):
    """Goal-seek event is NOT emitted when there is no active deployment."""
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(["No deployment."])
    mock_client.messages.stream.return_value = mock_stream

    def _no_dep():
        from sqlmodel import Session, create_engine

        from sqlmodel import SQLModel

        eng = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(eng)
        with Session(eng) as s:
            proj = Project(id="nodep-proj", name="NoDep")
            s.add(proj)
            ds = Dataset(
                id="nodep-ds",
                project_id="nodep-proj",
                filename="test.csv",
                file_path=str(tmp_path / "nodep.csv"),
                row_count=10,
                column_count=2,
            )
            (tmp_path / "nodep.csv").write_text("a,b\n1,2\n")
            s.add(ds)
            s.commit()
            yield s

    app.dependency_overrides[get_session] = _no_dep
    try:
        client_nd = TestClient(app)
        resp = client_nd.post(
            "/api/chat/nodep-proj",
            json={"message": "what inputs would produce a revenue of 1000?"},
        )
        assert resp.status_code == 200
        # goal_seek event should NOT be present
        assert '"type": "goal_seek"' not in resp.text
    finally:
        app.dependency_overrides.clear()
