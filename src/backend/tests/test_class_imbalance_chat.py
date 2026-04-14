"""Tests for class imbalance detection via chat.

Covers:
- _CLASS_IMBALANCE_PATTERNS regex matching (10 positive, 2 negative)
- detect_class_imbalance pure function (imbalanced / balanced cases)
- Chat handler emitting class_imbalance_check SSE event for classification
- No dataset → no event emitted
"""

from __future__ import annotations

import io
import json
import unittest.mock as mock

import numpy as np
import pytest
from sqlmodel import SQLModel, create_engine

import db as db_module

# Small imbalanced CSV (10% positive class)
_IMBALANCED_CSV = (
    b"region,revenue,units,churn\n" + b"East,100,10,0\n" * 18 + b"West,200,20,1\n" * 2
)

# Small balanced CSV (50/50 classes)
_BALANCED_CSV = (
    b"region,revenue,units,churn\n" + b"East,100,10,0\n" * 10 + b"West,200,20,1\n" * 10
)


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestClassImbalancePatterns:
    """Verify _CLASS_IMBALANCE_PATTERNS matches all intended phrases."""

    @pytest.fixture(autouse=True)
    def _import_pattern(self):
        from api.chat import _CLASS_IMBALANCE_PATTERNS

        self.pattern = _CLASS_IMBALANCE_PATTERNS

    def test_class_imbalance_literal(self):
        assert self.pattern.search("my data has a class imbalance problem")

    def test_imbalanced_data(self):
        assert self.pattern.search("my dataset is imbalanced")

    def test_rare_class(self):
        assert self.pattern.search("I have a rare class — only 2% positive")

    def test_minority_class(self):
        assert self.pattern.search("my minority class is very small")

    def test_positive_class_rare(self):
        assert self.pattern.search("my positive class is rare")

    def test_balance_my_classes(self):
        assert self.pattern.search("how do I balance my classes?")

    def test_smote_keyword(self):
        assert self.pattern.search("should I use SMOTE for this?")

    def test_class_weight(self):
        assert self.pattern.search("should I apply class_weight to training?")

    def test_check_imbalance(self):
        assert self.pattern.search("check for class imbalance in my data")

    def test_is_data_balanced(self):
        assert self.pattern.search("is my data balanced or imbalanced?")

    def test_negative_train_model(self):
        assert not self.pattern.search("train a new random forest model")

    def test_negative_make_prediction(self):
        assert not self.pattern.search("make a prediction for units=100")


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestDetectClassImbalance:
    """Unit tests for detect_class_imbalance — no server needed."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from core.trainer import detect_class_imbalance

        self.detect = detect_class_imbalance

    def test_balanced_returns_not_imbalanced(self):
        y = np.array(["0"] * 50 + ["1"] * 50)
        result = self.detect(y)
        assert result["is_imbalanced"] is False
        assert result["recommended_strategy"] == "none"

    def test_imbalanced_10pct_returns_imbalanced(self):
        y = np.array(["0"] * 90 + ["1"] * 10)
        result = self.detect(y)
        assert result["is_imbalanced"] is True
        assert result["minority_ratio"] == pytest.approx(0.10, abs=0.01)
        assert result["minority_class"] == "1"

    def test_severe_imbalance_recommends_smote(self):
        # < 5% minority with n >= 100 → SMOTE
        y = np.array(["0"] * 96 + ["1"] * 4)
        result = self.detect(y)
        assert result["is_imbalanced"] is True
        assert result["recommended_strategy"] == "smote"

    def test_moderate_imbalance_recommends_class_weight(self):
        y = np.array(["0"] * 85 + ["1"] * 15)
        result = self.detect(y)
        assert result["is_imbalanced"] is True
        assert result["recommended_strategy"] == "class_weight"

    def test_class_distribution_has_all_classes(self):
        y = np.array(["A"] * 60 + ["B"] * 40)
        result = self.detect(y)
        classes = {d["class"] for d in result["class_distribution"]}
        assert classes == {"A", "B"}

    def test_explanation_is_nonempty_string(self):
        y = np.array(["0"] * 80 + ["1"] * 20)
        result = self.detect(y)
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0


# ---------------------------------------------------------------------------
# Chat handler integration tests (synchronous TestClient)
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
    """Send chat message with mocked Anthropic (sync) and return parsed SSE events."""
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

    test_db = str(tmp_path / "chat_test.db")
    orig_engine = db_module.engine
    db_module.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    import models  # noqa — registers all tables

    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    import api.data as dm
    import api.deploy as dep
    import api.models as mm

    dm.UPLOAD_DIR = tmp_path / "uploads_chat"
    dep.DEPLOY_DIR = tmp_path / "deployments_chat"
    mm.MODELS_DIR = tmp_path / "models_chat"

    yield TestClient(app)
    db_module.engine = orig_engine


def _setup_classification_sync(client, tmp_path, csv_bytes=None):
    """Create project, upload a classification CSV, set target. Returns project_id."""
    if csv_bytes is None:
        csv_bytes = _IMBALANCED_CSV
    proj = client.post("/api/projects", json={"name": "ImbalanceTest"})
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


class TestClassImbalanceChatHandler:
    """Integration tests for the class imbalance chat handler."""

    def test_imbalanced_classification_emits_event(self, sync_client, tmp_path):
        """With imbalanced classification data, SSE emits class_imbalance_check."""
        pid = _setup_classification_sync(sync_client, tmp_path, _IMBALANCED_CSV)
        events = _chat_events(sync_client, pid, "Is my data imbalanced?")
        imbalance_events = [
            e for e in events if e.get("type") == "class_imbalance_check"
        ]
        assert len(imbalance_events) == 1
        payload = imbalance_events[0]["class_imbalance_check"]
        assert payload["is_imbalanced"] is True
        assert payload["problem_type"] == "classification"

    def test_balanced_classification_emits_event_with_is_imbalanced_false(
        self, sync_client, tmp_path
    ):
        """With balanced data, SSE still emits the event but is_imbalanced=False."""
        pid = _setup_classification_sync(sync_client, tmp_path, _BALANCED_CSV)
        events = _chat_events(sync_client, pid, "check for class imbalance")
        imbalance_events = [
            e for e in events if e.get("type") == "class_imbalance_check"
        ]
        assert len(imbalance_events) == 1
        assert imbalance_events[0]["class_imbalance_check"]["is_imbalanced"] is False

    def test_no_dataset_no_event(self, sync_client):
        """Without a dataset, no class_imbalance_check event is emitted."""
        proj = sync_client.post("/api/projects", json={"name": "NoDataImbalance"})
        pid = proj.json()["id"]
        events = _chat_events(sync_client, pid, "is my data imbalanced?")
        assert not any(e.get("type") == "class_imbalance_check" for e in events)

    def test_imbalanced_event_has_required_fields(self, sync_client, tmp_path):
        """Event payload includes all required fields."""
        pid = _setup_classification_sync(sync_client, tmp_path, _IMBALANCED_CSV)
        events = _chat_events(sync_client, pid, "minority class is rare")
        payload = next(
            e["class_imbalance_check"]
            for e in events
            if e.get("type") == "class_imbalance_check"
        )
        for field in (
            "is_imbalanced",
            "class_distribution",
            "recommended_strategy",
            "explanation",
            "problem_type",
        ):
            assert field in payload, f"Missing field: {field}"
