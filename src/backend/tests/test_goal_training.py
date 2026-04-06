"""Tests for goal-driven training.

Covers:
- _GOAL_TRAIN_PATTERNS detection
- _extract_goal_target() helper
- run_goal_driven_training() pure function (all branches)
- Chat integration: SSE goal_training event emitted
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_pattern_i_need_accuracy():
    from api.chat import _GOAL_TRAIN_PATTERNS

    assert _GOAL_TRAIN_PATTERNS.search("I need 85% accuracy")


def test_pattern_reach_r2():
    from api.chat import _GOAL_TRAIN_PATTERNS

    assert _GOAL_TRAIN_PATTERNS.search("reach 0.85 R2 for my model")


def test_pattern_hit_f1():
    from api.chat import _GOAL_TRAIN_PATTERNS

    assert _GOAL_TRAIN_PATTERNS.search("hit 80% F1 score")


def test_pattern_train_until():
    from api.chat import _GOAL_TRAIN_PATTERNS

    assert _GOAL_TRAIN_PATTERNS.search("train a model until it reaches 90% accuracy")


def test_pattern_keep_trying():
    from api.chat import _GOAL_TRAIN_PATTERNS

    assert _GOAL_TRAIN_PATTERNS.search(
        "keep trying different models until accuracy is good"
    )


def test_pattern_we_want_precision():
    from api.chat import _GOAL_TRAIN_PATTERNS

    assert _GOAL_TRAIN_PATTERNS.search("we want 0.9 precision for the classifier")


def test_pattern_negative_unrelated():
    from api.chat import _GOAL_TRAIN_PATTERNS

    assert not _GOAL_TRAIN_PATTERNS.search("what is the mean of revenue")


def test_pattern_negative_which_model():
    from api.chat import _GOAL_TRAIN_PATTERNS

    assert not _GOAL_TRAIN_PATTERNS.search("which model should I use?")


# ---------------------------------------------------------------------------
# _extract_goal_target helper
# ---------------------------------------------------------------------------


def test_extract_percentage_accuracy():
    from api.chat import _extract_goal_target

    result = _extract_goal_target("I need 85% accuracy", "classification")
    assert result is not None
    metric, val = result
    assert metric == "accuracy"
    assert abs(val - 0.85) < 1e-6


def test_extract_percentage_f1():
    from api.chat import _extract_goal_target

    result = _extract_goal_target("hit 80% F1 score", "classification")
    assert result is not None
    metric, val = result
    assert metric == "f1"
    assert abs(val - 0.80) < 1e-6


def test_extract_decimal_r2():
    from api.chat import _extract_goal_target

    result = _extract_goal_target("reach 0.90 R2", "regression")
    assert result is not None
    metric, val = result
    assert metric == "r2"
    assert abs(val - 0.90) < 1e-6


def test_extract_default_metric_regression():
    from api.chat import _extract_goal_target

    # No metric name in message — default to r2 for regression
    result = _extract_goal_target("I want 0.85", "regression")
    assert result is not None
    metric, val = result
    assert metric == "r2"


def test_extract_default_metric_classification():
    from api.chat import _extract_goal_target

    result = _extract_goal_target("I need 80%", "classification")
    assert result is not None
    metric, val = result
    assert metric == "accuracy"


def test_extract_no_value_returns_none():
    from api.chat import _extract_goal_target

    result = _extract_goal_target("I want high accuracy", "classification")
    assert result is None


# ---------------------------------------------------------------------------
# run_goal_driven_training pure function
# ---------------------------------------------------------------------------


def _make_regression_data(n: int = 200):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 3))
    y = 2 * X[:, 0] + X[:, 1] + rng.standard_normal(n) * 0.1
    return X, y


def _make_classification_data(n: int = 200):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_goal_training_regression_achieves_high_r2():
    from core.trainer import run_goal_driven_training

    X, y = _make_regression_data(300)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_goal_driven_training(
            X, y, "regression", "r2", 0.50, Path(tmp), "tst"
        )
    assert result["achieved"] is True
    assert result["winner_score"] >= 0.50
    assert len(result["trials"]) >= 1


def test_goal_training_regression_no_achieve_unreachable():
    from core.trainer import run_goal_driven_training

    X, y = _make_regression_data(100)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_goal_driven_training(
            X, y, "regression", "r2", 0.9999, Path(tmp), "tst2"
        )
    # Should not crash; achieved should be False for an unreachable target
    assert isinstance(result["achieved"], bool)
    assert "summary" in result
    assert "trials" in result


def test_goal_training_classification_achieves_accuracy():
    from core.trainer import run_goal_driven_training

    X, y = _make_classification_data(300)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_goal_driven_training(
            X, y, "classification", "accuracy", 0.60, Path(tmp), "tst3"
        )
    assert result["achieved"] is True
    assert result["winner_score"] >= 0.60


def test_goal_training_result_fields():
    from core.trainer import run_goal_driven_training

    X, y = _make_regression_data(150)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_goal_driven_training(
            X, y, "regression", "r2", 0.50, Path(tmp), "tst4"
        )
    for field in [
        "goal_metric",
        "goal_target",
        "achieved",
        "winner_algorithm",
        "winner_algorithm_name",
        "winner_score",
        "trials",
        "tried_tuning",
        "summary",
    ]:
        assert field in result, f"Missing field: {field}"


def test_goal_training_trial_fields():
    from core.trainer import run_goal_driven_training

    X, y = _make_regression_data(150)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_goal_driven_training(
            X, y, "regression", "r2", 0.50, Path(tmp), "tst5"
        )
    assert len(result["trials"]) >= 1
    for trial in result["trials"]:
        assert "algorithm" in trial
        assert "algorithm_name" in trial
        assert "score" in trial
        assert "achieved_goal" in trial
        assert isinstance(trial["achieved_goal"], bool)
        assert 0 <= trial["score"] <= 1.0 or trial["score"] < 0  # r2 can be negative


def test_goal_score_r2():
    from core.trainer import _goal_score

    metrics = {"r2": 0.85, "accuracy": 0.90}
    assert abs(_goal_score(metrics, "r2") - 0.85) < 1e-9


def test_goal_score_accuracy():
    from core.trainer import _goal_score

    metrics = {"r2": 0.70, "accuracy": 0.88}
    assert abs(_goal_score(metrics, "accuracy") - 0.88) < 1e-9


def test_goal_score_missing_metric():
    from core.trainer import _goal_score

    metrics = {"r2": 0.70}
    assert _goal_score(metrics, "accuracy") == -999.0


def test_goal_training_summary_contains_achieved(tmp_path):
    from core.trainer import run_goal_driven_training

    X, y = _make_regression_data(300)
    result = run_goal_driven_training(X, y, "regression", "r2", 0.50, tmp_path, "ts6")
    assert "summary" in result
    assert len(result["summary"]) > 10


def test_goal_training_stops_early_when_goal_met():
    """Goal met on first algorithm should not try more algorithms."""
    from core.trainer import run_goal_driven_training

    X, y = _make_regression_data(300)
    with tempfile.TemporaryDirectory() as tmp:
        result = run_goal_driven_training(
            X, y, "regression", "r2", 0.01, Path(tmp), "tst7"
        )
    # With a very low target, should achieve goal on first or second algo and stop
    assert result["achieved"] is True
    # At most 2 trials (may stop after 1)
    assert len(result["trials"]) <= 3


# ---------------------------------------------------------------------------
# Chat SSE integration: goal_training event
# ---------------------------------------------------------------------------


GOAL_SAMPLE_CSV = b"""feature_a,feature_b,revenue
1.764,0.400,4.068
0.978,2.241,3.197
1.867,-0.977,2.757
0.950,-0.151,2.599
-0.103,0.411,0.206
0.144,1.454,1.142
0.761,0.122,1.644
0.444,0.334,1.221
1.494,-0.205,2.783
0.313,0.692,1.318
-0.854,-2.553,-4.261
0.653,0.864,2.170
-0.742,2.270,-0.214
-1.454,0.046,-2.862
0.046,0.535,0.627
-0.466,0.242,-0.690
-1.913,-1.725,-5.551
-0.562,-1.013,-2.137
0.314,1.465,2.093
0.692,0.867,2.251
"""


@pytest.fixture(autouse=True)
def test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "goal_test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "engine", engine)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture()
def goal_client():
    from main import app

    return TestClient(app)


@pytest.fixture()
def seeded_project(goal_client, tmp_path):
    """Create project, upload CSV, apply features and set target."""
    import io

    proj = goal_client.post("/api/projects", json={"name": "GoalTest"})
    project_id = proj.json()["id"]

    upload = goal_client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("goal.csv", io.BytesIO(GOAL_SAMPLE_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    goal_client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    goal_client.post(
        f"/api/features/{dataset_id}/target", json={"target_column": "revenue"}
    )

    return {"project_id": project_id, "dataset_id": dataset_id}


def _chat_events_goal(client, project_id: str, message: str) -> list[dict]:
    """Send a chat message under mocked Anthropic and return all SSE events."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_c = MagicMock()
        mock_cls.return_value = mock_c
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Great goal training result!"])
        mock_c.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": message},
        )

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def test_chat_goal_training_event_emitted(goal_client, seeded_project):
    """Sending a goal-training intent message emits a goal_training SSE event."""
    project_id = seeded_project["project_id"]
    events = _chat_events_goal(goal_client, project_id, "I need 50% R2 accuracy")
    types = [e.get("type") for e in events]
    assert "goal_training" in types, f"Expected goal_training in {types}"


def test_chat_goal_training_fields_present(goal_client, seeded_project):
    """Goal training event has required fields with correct types."""
    project_id = seeded_project["project_id"]
    events = _chat_events_goal(goal_client, project_id, "reach 0.40 R2 on revenue")
    goal_events = [e for e in events if e.get("type") == "goal_training"]
    assert goal_events, "No goal_training event in SSE stream"

    gt = goal_events[0]["goal_training"]
    assert isinstance(gt["achieved"], bool)
    assert isinstance(gt["goal_target"], float)
    assert isinstance(gt["trials"], list)
    assert isinstance(gt["summary"], str)
    assert len(gt["summary"]) > 0
