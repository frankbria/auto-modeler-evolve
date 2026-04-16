"""Tests for chat-initiated hyperparameter tuning.

Covers:
- _TUNE_PATTERNS regex detection (broad vocabulary)
- _EXPLICIT_TUNE_RE regex detection (unambiguous tuning intent)
- _EXPLICIT_TUNE_RE does NOT match generic improvement phrases
- POST /api/chat/{project_id} tune_chat SSE event (integration)
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import (
    _EXPLICIT_TUNE_RE,
    _TUNE_PATTERNS,
)

# ---------------------------------------------------------------------------
# Fixtures (same pattern as test_chat_training.py)
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units,cost\n"
    b"East,100,10,50\n"
    b"West,200,20,80\n"
    b"East,150,15,60\n"
    b"West,300,30,120\n"
    b"North,250,25,100\n"
    b"East,175,18,70\n"
    b"West,220,22,90\n"
    b"North,190,19,75\n"
    b"East,130,13,55\n"
    b"West,280,28,110\n"
)


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_module

    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Tuning Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


@pytest.fixture()
async def feature_set_id(ac, dataset_id):
    resp = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert resp.status_code == 201, resp.text
    fs_id = resp.json()["feature_set_id"]

    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


@pytest.fixture()
async def model_run_id(ac, project_id, dataset_id, feature_set_id):
    """Train a model so there is a completed run to tune."""
    resp = await ac.post(
        "/api/models/train",
        json={
            "project_id": project_id,
            "feature_set_id": feature_set_id,
            "algorithm": "random_forest",
            "problem_type": "regression",
        },
    )
    assert resp.status_code in (200, 201), resp.text
    data = resp.json()
    # Return the model run id from the train response
    return data.get("id") or data.get("model_run_id")


# ---------------------------------------------------------------------------
# Unit tests — _TUNE_PATTERNS (broad)
# ---------------------------------------------------------------------------


def test_tune_patterns_tune_model():
    assert _TUNE_PATTERNS.search("can you tune my model")


def test_tune_patterns_optimize():
    assert _TUNE_PATTERNS.search("optimize the model for better accuracy")


def test_tune_patterns_optimise_british():
    assert _TUNE_PATTERNS.search("optimise my model please")


def test_tune_patterns_hyperparameter():
    assert _TUNE_PATTERNS.search("run hyperparameter search")


def test_tune_patterns_grid_search():
    assert _TUNE_PATTERNS.search("do a grid search")


def test_tune_patterns_increase_accuracy():
    assert _TUNE_PATTERNS.search("increase accuracy of the model")


def test_tune_patterns_boost_performance():
    assert _TUNE_PATTERNS.search("boost performance")


def test_tune_patterns_improve_model():
    assert _TUNE_PATTERNS.search("how can I improve my model")


# ---------------------------------------------------------------------------
# Unit tests — _EXPLICIT_TUNE_RE (unambiguous — must fire inline job)
# ---------------------------------------------------------------------------


def test_explicit_tune_re_tune():
    assert _EXPLICIT_TUNE_RE.search("tune my model")


def test_explicit_tune_re_tuning():
    assert _EXPLICIT_TUNE_RE.search("start the tuning process")


def test_explicit_tune_re_optimize():
    assert _EXPLICIT_TUNE_RE.search("optimize hyperparameters")


def test_explicit_tune_re_hyperparameter():
    assert _EXPLICIT_TUNE_RE.search("run hyperparameter search")


def test_explicit_tune_re_go_ahead_and_tune():
    assert _EXPLICIT_TUNE_RE.search("go ahead and tune it")


def test_explicit_tune_re_run_the_tuning():
    assert _EXPLICIT_TUNE_RE.search("run the tuning now")


def test_explicit_tune_re_start_tuning():
    assert _EXPLICIT_TUNE_RE.search("start tuning")


def test_explicit_tune_re_best_params():
    assert _EXPLICIT_TUNE_RE.search("find the best params for this model")


def test_explicit_tune_re_best_settings():
    assert _EXPLICIT_TUNE_RE.search("find best settings")


# ---------------------------------------------------------------------------
# Unit tests — _EXPLICIT_TUNE_RE should NOT match generic improvement phrases
# ---------------------------------------------------------------------------


def test_explicit_tune_re_no_false_positive_improve():
    # "improve my model" without explicit tuning term — should not match
    assert not _EXPLICIT_TUNE_RE.search("how do I improve my model")


def test_explicit_tune_re_no_false_positive_better():
    assert not _EXPLICIT_TUNE_RE.search("make my model better")


def test_explicit_tune_re_no_false_positive_accuracy():
    assert not _EXPLICIT_TUNE_RE.search("increase accuracy")


# ---------------------------------------------------------------------------
# Integration test — tune_chat SSE event emitted when model run exists
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tune_chat_event_emitted(ac, project_id, dataset_id, feature_set_id, model_run_id):
    """Sending 'go ahead and tune it' with a trained model should emit tune_chat event."""
    resp = await ac.post(
        f"/api/chat/{project_id}",
        json={"message": "go ahead and tune it"},
        headers={"Accept": "text/event-stream"},
        timeout=120,
    )
    assert resp.status_code == 200

    # Collect all SSE events
    events: list[dict] = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            payload = line[6:]
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                pass

    event_types = {e.get("type") for e in events}
    assert "tune_chat" in event_types, f"Expected tune_chat event, got: {event_types}"

    tune_event = next(e for e in events if e.get("type") == "tune_chat")
    tc = tune_event["tune_chat"]
    assert tc["tunable"] is True
    assert tc["algorithm"] == "random_forest"
    assert "original_metrics" in tc
    assert "tuned_metrics" in tc
    assert "best_params" in tc
    assert "improved" in tc
    assert "summary" in tc
