"""Tests for chat-initiated hyperparameter tuning.

Covers:
- _TUNE_PATTERNS regex detection (broad vocabulary)
- _EXPLICIT_TUNE_RE regex detection (unambiguous tuning intent)
- _EXPLICIT_TUNE_RE does NOT match generic improvement phrases
- POST /api/chat/{project_id} tune_chat SSE event (integration)
"""

from __future__ import annotations

import io
import json
import unittest.mock as mock

import pytest
from sqlmodel import Session, SQLModel, create_engine

import db as db_module
from api.chat import (
    _EXPLICIT_TUNE_RE,
    _TUNE_PATTERNS,
)

# ---------------------------------------------------------------------------
# Fixtures (synchronous TestClient pattern — avoids background-thread engine issue)
# ---------------------------------------------------------------------------

# 100-row regression CSV — enough rows for 3-fold CV in RandomizedSearchCV
_SAMPLE_CSV = (
    b"units,price,cost,revenue\n"
    + b"10,5.0,3.0,50.0\n" * 34
    + b"20,8.0,5.0,160.0\n" * 33
    + b"15,6.0,4.0,90.0\n" * 33
)


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _chat_events(client, project_id: str, message: str) -> list[dict]:
    """Send a chat message with mocked Anthropic and return parsed SSE events."""
    with mock.patch("anthropic.Anthropic") as mock_cls:
        mc = mock.MagicMock()
        mock_cls.return_value = mc
        ms = mock.MagicMock()
        ms.__enter__ = mock.MagicMock(return_value=ms)
        ms.__exit__ = mock.MagicMock(return_value=False)
        ms.text_stream = iter(["Done."])
        mc.messages.stream.return_value = ms
        resp = client.post(
            f"/api/chat/{project_id}",
            json={"message": message, "project_id": project_id},
        )
    return _parse_sse(resp.text)


@pytest.fixture()
def sync_client(tmp_path):
    """Synchronous TestClient with isolated SQLite DB."""
    from fastapi.testclient import TestClient
    from main import app

    test_db = str(tmp_path / "tuning_chat_test.db")
    orig_engine = db_module.engine
    db_module.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    import models  # noqa — registers all tables

    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    import api.data as dm
    import api.models as mm

    dm.UPLOAD_DIR = tmp_path / "uploads"
    mm.MODELS_DIR = tmp_path / "models"

    yield TestClient(app)
    db_module.engine = orig_engine


def _setup_with_model_run(client, tmp_path) -> str:
    """Create project + dataset + feature_set, inject a done ModelRun. Returns project_id.

    Injects ModelRun directly to avoid the background-training thread / DB-engine
    mismatch. The chat handler calls tune_model() fresh on X/y — it doesn't load
    a saved model file — so no model on disk is needed.
    """
    from models.model_run import ModelRun

    proj = client.post("/api/projects", json={"name": "TuningChatTest"})
    pid = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": pid},
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    did = upload.json()["dataset_id"]

    fs_r = client.post(f"/api/features/{did}/apply", json={"transformations": []})
    assert fs_r.status_code == 201, fs_r.text
    fs_id = fs_r.json()["feature_set_id"]

    client.post(
        f"/api/features/{did}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )

    # Inject a completed ModelRun with a tunable algorithm
    # Must use the full registry key, not the display name
    with Session(db_module.engine) as session:
        run = ModelRun(
            project_id=pid,
            feature_set_id=fs_id,
            algorithm="random_forest_regressor",
            status="done",
            metrics=json.dumps({"r2": 0.72, "mae": 8.5, "rmse": 12.0}),
            summary="Random Forest Regressor: R² 0.720",
        )
        session.add(run)
        session.commit()

    return pid


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
# Integration tests — tune_chat SSE event
# ---------------------------------------------------------------------------


class TestTuneChatHandler:
    """Integration tests for the hyperparameter tuning chat handler."""

    def test_no_model_runs_no_event(self, sync_client):
        """Without any model runs the tune_chat event must not be emitted."""
        proj = sync_client.post("/api/projects", json={"name": "NoRuns"})
        pid = proj.json()["id"]
        events = _chat_events(sync_client, pid, "go ahead and tune it")
        assert not any(e.get("type") == "tune_chat" for e in events)

    def test_tune_chat_event_emitted(self, sync_client, tmp_path):
        """Explicit tuning phrase with a completed run emits tune_chat event."""
        pid = _setup_with_model_run(sync_client, tmp_path)
        events = _chat_events(sync_client, pid, "go ahead and tune it")
        tune_events = [e for e in events if e.get("type") == "tune_chat"]
        assert len(tune_events) == 1
        tc = tune_events[0]["tune_chat"]
        assert tc["tunable"] is True
        assert tc["algorithm"] == "random_forest_regressor"
        assert "original_metrics" in tc
        assert "tuned_metrics" in tc
        assert "best_params" in tc
        assert "improved" in tc
        assert "summary" in tc

    def test_generic_improve_phrase_no_tune_event(self, sync_client, tmp_path):
        """Generic 'improve my model' must not trigger inline tuning."""
        pid = _setup_with_model_run(sync_client, tmp_path)
        events = _chat_events(sync_client, pid, "how do I improve my model")
        assert not any(e.get("type") == "tune_chat" for e in events)
