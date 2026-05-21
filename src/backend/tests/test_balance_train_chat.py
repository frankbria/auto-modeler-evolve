"""Tests for chat-triggered imbalance-corrected training.

Covers:
- _BALANCE_TRAIN_PATTERNS regex (8 positive, 4 negative)
- _detect_balance_strategy() helper (smote / threshold / class_weight)
- Chat handler emitting training_started with imbalance_strategy for classification
- Regression target → no training started event (imbalance n/a)
- No dataset → no event emitted
"""

from __future__ import annotations

import io
import json
import unittest.mock as mock

import pytest
from sqlmodel import SQLModel, create_engine

import db as db_module

# Small imbalanced classification CSV (20% positive class)
_IMBALANCED_CSV = (
    b"region,revenue,units,churn\n" + b"East,100,10,0\n" * 16 + b"West,200,20,1\n" * 4
)

# Regression CSV (continuous target)
_REGRESSION_CSV = (
    b"region,units,price,revenue\n"
    + b"East,10,5.0,50.0\n" * 10
    + b"West,20,8.0,160.0\n" * 10
)


# ---------------------------------------------------------------------------
# Pattern tests
# ---------------------------------------------------------------------------


class TestBalanceTrainPatterns:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api.chat import _BALANCE_TRAIN_PATTERNS

        self.pat = _BALANCE_TRAIN_PATTERNS

    def test_train_with_class_weighting(self):
        assert self.pat.search("train with class weighting")

    def test_train_with_class_weights(self):
        assert self.pat.search("train with class weights")

    def test_apply_smote_and_retrain(self):
        assert self.pat.search("apply SMOTE and retrain")

    def test_retrain_with_smote(self):
        assert self.pat.search("retrain with SMOTE")

    def test_use_oversampling_and_train(self):
        assert self.pat.search("use oversampling and train")

    def test_fix_imbalance_and_train(self):
        assert self.pat.search("fix the imbalance and train")

    def test_correct_imbalance_and_retrain(self):
        assert self.pat.search("correct class imbalance and retrain")

    def test_train_with_balanced_weights(self):
        assert self.pat.search("train with balanced weights")

    def test_negative_plain_train(self):
        assert not self.pat.search("train a random forest on revenue")

    def test_negative_check_imbalance(self):
        assert not self.pat.search("is my data imbalanced?")

    def test_negative_make_prediction(self):
        assert not self.pat.search("make a prediction for units=100")

    def test_negative_show_accuracy(self):
        assert not self.pat.search("show me model accuracy")


# ---------------------------------------------------------------------------
# _detect_balance_strategy helper tests
# ---------------------------------------------------------------------------


class TestDetectBalanceStrategy:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api.chat import _detect_balance_strategy

        self.detect = _detect_balance_strategy

    def test_smote_keyword_returns_smote(self):
        assert self.detect("apply SMOTE and retrain") == "smote"

    def test_oversample_returns_smote(self):
        assert self.detect("use oversampling to fix imbalance and train") == "smote"

    def test_threshold_keyword_returns_threshold(self):
        assert self.detect("train with threshold tuning") == "threshold"

    def test_class_weight_default(self):
        assert self.detect("train with class weighting") == "class_weight"

    def test_balanced_weights_default(self):
        assert self.detect("retrain with balanced weights") == "class_weight"

    def test_smote_case_insensitive(self):
        assert self.detect("Retrain with smote please") == "smote"


# ---------------------------------------------------------------------------
# Chat handler integration tests
# ---------------------------------------------------------------------------


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


@pytest.fixture()
def sync_client(tmp_path):
    from fastapi.testclient import TestClient
    from main import app

    test_db = str(tmp_path / "balance_train_test.db")
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

    dm.UPLOAD_DIR = tmp_path / "uploads_bt"
    dep.DEPLOY_DIR = tmp_path / "deployments_bt"
    mm.MODELS_DIR = tmp_path / "models_bt"

    yield TestClient(app)
    db_module.engine = orig_engine


def _setup_classification(client, tmp_path, csv_bytes=_IMBALANCED_CSV):
    """Create project with imbalanced classification dataset. Returns project_id."""
    proj = client.post("/api/projects", json={"name": "BalanceTrainTest"})
    pid = proj.json()["id"]
    upload = client.post(
        "/api/data/upload",
        data={"project_id": pid},
        files={"file": ("churn.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    did = upload.json()["dataset_id"]
    fs_r = client.post(f"/api/features/{did}/apply", json={"transformations": []})
    assert fs_r.status_code == 201, fs_r.text
    fs_id = fs_r.json()["feature_set_id"]
    client.post(
        f"/api/features/{did}/target",
        json={"target_column": "churn", "feature_set_id": fs_id},
    )
    return pid


def _setup_regression(client, tmp_path):
    """Create project with regression dataset. Returns project_id."""
    proj = client.post("/api/projects", json={"name": "RegBalanceTest"})
    pid = proj.json()["id"]
    upload = client.post(
        "/api/data/upload",
        data={"project_id": pid},
        files={"file": ("sales.csv", io.BytesIO(_REGRESSION_CSV), "text/csv")},
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
    return pid


class TestBalanceTrainChatHandler:
    def test_class_weight_emits_training_started(self, sync_client, tmp_path):
        """'train with class weighting' emits training_started with imbalance_strategy."""
        pid = _setup_classification(sync_client, tmp_path)
        events = _chat_events(sync_client, pid, "train with class weighting")
        started = [e for e in events if e.get("type") == "training_started"]
        assert (
            len(started) == 1
        ), f"Expected training_started, got: {[e.get('type') for e in events]}"
        payload = started[0]["training"]
        assert payload["imbalance_strategy"] == "class_weight"
        assert payload["problem_type"] == "classification"
        assert payload["status"] == "started"
        assert len(payload["algorithms"]) >= 1

    def test_smote_emits_correct_strategy(self, sync_client, tmp_path):
        """'apply SMOTE and retrain' emits training_started with imbalance_strategy=smote."""
        pid = _setup_classification(sync_client, tmp_path)
        events = _chat_events(sync_client, pid, "apply SMOTE and retrain the model")
        started = [e for e in events if e.get("type") == "training_started"]
        assert len(started) == 1
        assert started[0]["training"]["imbalance_strategy"] == "smote"

    def test_threshold_emits_correct_strategy(self, sync_client, tmp_path):
        """'train with threshold tuning' emits imbalance_strategy=threshold."""
        pid = _setup_classification(sync_client, tmp_path)
        events = _chat_events(sync_client, pid, "train with threshold tuning")
        started = [e for e in events if e.get("type") == "training_started"]
        assert len(started) == 1
        assert started[0]["training"]["imbalance_strategy"] == "threshold"

    def test_no_dataset_no_event(self, sync_client):
        """Without a dataset, no training_started event is emitted."""
        proj = sync_client.post("/api/projects", json={"name": "NoDataBalanceTrain"})
        pid = proj.json()["id"]
        events = _chat_events(sync_client, pid, "train with class weighting")
        assert not any(e.get("type") == "training_started" for e in events)

    def test_regression_target_no_training_started(self, sync_client, tmp_path):
        """For regression targets, balance-corrected training is not applicable — no training_started."""
        pid = _setup_regression(sync_client, tmp_path)
        events = _chat_events(sync_client, pid, "train with class weighting")
        started = [e for e in events if e.get("type") == "training_started"]
        assert (
            len(started) == 0
        ), "Should not start training for regression with imbalance correction"

    def test_fix_imbalance_variant_works(self, sync_client, tmp_path):
        """'fix the imbalance and train' phrase triggers balanced training."""
        pid = _setup_classification(sync_client, tmp_path)
        events = _chat_events(sync_client, pid, "fix the imbalance and train a model")
        started = [e for e in events if e.get("type") == "training_started"]
        assert len(started) == 1
        assert started[0]["training"]["imbalance_strategy"] is not None

    def test_training_started_has_run_count(self, sync_client, tmp_path):
        """training_started event has correct run_count field."""
        pid = _setup_classification(sync_client, tmp_path)
        events = _chat_events(sync_client, pid, "retrain with balanced weights")
        started = [e for e in events if e.get("type") == "training_started"]
        assert len(started) == 1
        assert started[0]["training"]["run_count"] >= 1

    def test_correct_imbalance_variant(self, sync_client, tmp_path):
        """'correct class imbalance and retrain' triggers balanced training."""
        pid = _setup_classification(sync_client, tmp_path)
        events = _chat_events(sync_client, pid, "correct class imbalance and retrain")
        started = [e for e in events if e.get("type") == "training_started"]
        assert len(started) == 1
