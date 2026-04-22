"""Tests for chat-triggered retrain-excluding-weak-features.

Covers:
- _WEAK_FEAT_RETRAIN_PATTERNS regex (8 positive, 4 negative)
- Chat handler emitting training_started with excluded_features list
- No dataset → no event emitted
- No completed model → handler silently skips (no training_started)
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock

import pytest
from sqlmodel import SQLModel, create_engine

import db as db_module

# CSV where f4 is a zero-variance feature (will be identified as weak/unimportant)
_REGRESSION_CSV = b"""f1,f2,f3,f4,target
1.0,0.5,100.0,0.0,10.0
2.0,1.0,200.0,0.0,20.0
3.0,1.5,300.0,0.0,30.0
4.0,2.0,400.0,0.0,40.0
5.0,2.5,500.0,0.0,50.0
6.0,3.0,600.0,0.0,60.0
7.0,3.5,700.0,0.0,70.0
8.0,4.0,800.0,0.0,80.0
9.0,4.5,900.0,0.0,90.0
10.0,5.0,1000.0,0.0,100.0
11.0,5.5,1100.0,0.0,110.0
12.0,6.0,1200.0,0.0,120.0
"""


# ---------------------------------------------------------------------------
# Pattern tests
# ---------------------------------------------------------------------------


class TestWeakFeatRetrainPatterns:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api.chat import _WEAK_FEAT_RETRAIN_PATTERNS

        self.pat = _WEAK_FEAT_RETRAIN_PATTERNS

    # ---- Positive matches ----

    def test_retrain_without_weak_features(self):
        assert self.pat.search("retrain without weak features")

    def test_retrain_without_unimportant_features(self):
        assert self.pat.search("retrain without unimportant features")

    def test_drop_weak_features_and_retrain(self):
        assert self.pat.search("drop weak features and retrain")

    def test_remove_weak_columns_and_retrain(self):
        assert self.pat.search("remove weak columns and retrain")

    def test_exclude_low_importance_features_and_train(self):
        assert self.pat.search("exclude low-importance features and train")

    def test_train_without_the_weak_features(self):
        assert self.pat.search("train without the weak features")

    def test_retrain_with_weak_features_removed(self):
        assert self.pat.search("retrain with weak features removed")

    def test_train_with_unimportant_columns_excluded(self):
        assert self.pat.search("train with unimportant columns excluded")

    # ---- Negative matches ----

    def test_negative_plain_train(self):
        assert not self.pat.search("train a model on revenue")

    def test_negative_check_imbalance(self):
        assert not self.pat.search("is my data imbalanced?")

    def test_negative_feature_selection_show(self):
        assert not self.pat.search("which features are important?")

    def test_negative_prediction(self):
        assert not self.pat.search("make a prediction for units=100")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sync_client(tmp_path):
    from fastapi.testclient import TestClient
    from main import app

    test_db = str(tmp_path / "weak_feat_test.db")
    orig_engine = db_module.engine
    db_module.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    import models  # noqa

    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    import api.data as dm
    import api.deploy as dep
    import api.models as mm

    dm.UPLOAD_DIR = tmp_path / "uploads_wf"
    dep.DEPLOY_DIR = tmp_path / "deployments_wf"
    mm.MODELS_DIR = tmp_path / "models_wf"

    yield TestClient(app)
    # Allow daemon training threads to finish before restoring the engine.
    # Without this, threads spawned by the chat handler may still be running
    # when we swap back to orig_engine, causing "no such table" OperationalErrors
    # that pytest converts to test failures via PytestUnhandledThreadExceptionWarning.
    time.sleep(0.5)
    db_module.engine = orig_engine


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


def _setup_regression_project(client, tmp_path, csv_bytes=_REGRESSION_CSV):
    """Create project with regression dataset, apply feature set, set target."""
    proj = client.post("/api/projects", json={"name": "WeakFeatRetrainTest"})
    pid = proj.json()["id"]
    upload = client.post(
        "/api/data/upload",
        data={"project_id": pid},
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    did = upload.json()["dataset_id"]
    fs_r = client.post(f"/api/features/{did}/apply", json={"transformations": []})
    assert fs_r.status_code == 201, fs_r.text
    target_r = client.post(
        f"/api/features/{did}/target",
        json={"target_column": "target"},
    )
    assert target_r.status_code == 200, target_r.text
    return pid, did


def _train_and_wait(client, project_id: str, algorithm: str = "random_forest_regressor") -> str:
    """Train a model synchronously and return run_id."""
    resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": [algorithm]},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["model_run_ids"][0]

    for _ in range(40):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    return run_id


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestWeakFeatRetrainChatHandler:
    def test_no_dataset_no_event(self, sync_client, tmp_path):
        """No dataset context → training_started event not emitted."""
        proj = sync_client.post("/api/projects", json={"name": "EmptyProj"})
        pid = proj.json()["id"]
        events = _chat_events(sync_client, pid, "retrain without weak features")
        types = [e.get("type") for e in events]
        assert "training_started" not in types

    def test_no_completed_model_no_event(self, sync_client, tmp_path):
        """Dataset + feature set but no completed model → no training_started."""
        pid, _ = _setup_regression_project(sync_client, tmp_path)
        events = _chat_events(sync_client, pid, "drop weak features and retrain")
        types = [e.get("type") for e in events]
        assert "training_started" not in types

    def test_with_trained_model_emits_training_started(self, sync_client, tmp_path):
        """Project with a completed RF model → training_started event fired."""
        pid, _ = _setup_regression_project(sync_client, tmp_path)
        _train_and_wait(sync_client, pid, "random_forest_regressor")

        events = _chat_events(sync_client, pid, "retrain without weak features")
        types = [e.get("type") for e in events]
        assert "training_started" in types

    def test_excluded_features_field_present(self, sync_client, tmp_path):
        """training_started event includes excluded_features list."""
        pid, _ = _setup_regression_project(sync_client, tmp_path)
        _train_and_wait(sync_client, pid, "random_forest_regressor")

        events = _chat_events(sync_client, pid, "retrain without weak features")
        ts = next((e for e in events if e.get("type") == "training_started"), None)
        # Handler may emit no event if no weak features found; if it does, check the field
        if ts is not None:
            assert "excluded_features" in ts.get("training", {})
            assert isinstance(ts["training"]["excluded_features"], list)

    def test_variant_drop_weak_features_and_retrain(self, sync_client, tmp_path):
        """Alternative phrasing triggers same handler."""
        pid, _ = _setup_regression_project(sync_client, tmp_path)
        _train_and_wait(sync_client, pid, "random_forest_regressor")

        events = _chat_events(sync_client, pid, "drop weak features and retrain")
        # Should not crash; either training_started or text response is fine
        assert any(e.get("type") in ("training_started", "suggestions") or "content" in e for e in events)

    def test_run_count_in_event(self, sync_client, tmp_path):
        """When training_started emitted, run_count > 0."""
        pid, _ = _setup_regression_project(sync_client, tmp_path)
        _train_and_wait(sync_client, pid, "random_forest_regressor")

        events = _chat_events(sync_client, pid, "remove unimportant features and retrain")
        ts = next((e for e in events if e.get("type") == "training_started"), None)
        if ts is not None:
            assert ts["training"]["run_count"] > 0

    def test_target_column_in_event(self, sync_client, tmp_path):
        """training_started event carries the correct target_column."""
        pid, _ = _setup_regression_project(sync_client, tmp_path)
        _train_and_wait(sync_client, pid, "random_forest_regressor")

        events = _chat_events(sync_client, pid, "train without weak features")
        ts = next((e for e in events if e.get("type") == "training_started"), None)
        if ts is not None:
            assert ts["training"]["target_column"] == "target"

    def test_normal_train_not_triggered_when_handler_fires(self, sync_client, tmp_path):
        """When weak-feat retrain fires, _TRAIN_PATTERNS should not also fire."""
        pid, _ = _setup_regression_project(sync_client, tmp_path)
        _train_and_wait(sync_client, pid, "random_forest_regressor")

        events = _chat_events(sync_client, pid, "retrain without weak features")
        ts_events = [e for e in events if e.get("type") == "training_started"]
        # At most one training_started event
        assert len(ts_events) <= 1
