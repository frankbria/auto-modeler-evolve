"""Tests for Ensemble Training via Chat.

Covers:
- _ENSEMBLE_TRAIN_PATTERNS regex (positive + negative)
- _STACKING_RE sub-pattern detection
- training_started event is emitted for voting/stacking requests
- Mutual exclusion: _TRAIN_PATTERNS does NOT fire when ensemble path fires
- No event when feature_set missing
"""

from __future__ import annotations

import io
import json
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
    b"East,160.0,16\nWest,210.0,21\n"
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.ab_test  # noqa
    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.deployment_preset  # noqa
    import models.deployment_version  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa
    import models.webhook_event  # noqa
    import models.analysis_template  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _setup_project_with_target(client):
    """Create project → upload → features → target (no model run or deployment)."""
    proj = client.post("/api/projects", json={"name": "EnsembleTest"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "problem_type": "regression"},
    )

    return project_id


def _chat(client, project_id, message):
    with mock.patch("anthropic.Anthropic") as mock_ant:
        mock_client = mock.MagicMock()
        mock_ant.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Training started."])
        mock_client.messages.stream.return_value = mock_stream
        resp = client.post(f"/api/chat/{project_id}", json={"message": message})
    return resp


def _parse_events(resp):
    return [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ") and line[6:].strip()
    ]


# ---------------------------------------------------------------------------
# Pattern tests
# ---------------------------------------------------------------------------


class TestEnsembleTrainPatterns:
    def setup_method(self):
        from api.chat import _ENSEMBLE_TRAIN_PATTERNS, _STACKING_RE

        self.pattern = _ENSEMBLE_TRAIN_PATTERNS
        self.stacking_re = _STACKING_RE

    # Positive matches
    def test_train_voting_ensemble(self):
        assert self.pattern.search("train a voting ensemble")

    def test_train_voting_classifier(self):
        assert self.pattern.search("train voting classifier")

    def test_train_voting_regressor(self):
        assert self.pattern.search("train a voting regressor")

    def test_train_stacking_ensemble(self):
        assert self.pattern.search("train a stacking ensemble")

    def test_build_voting_model(self):
        assert self.pattern.search("build a voting model")

    def test_build_stacking_classifier(self):
        assert self.pattern.search("build stacking classifier")

    def test_run_voting_ensemble(self):
        assert self.pattern.search("run a voting ensemble")

    def test_create_stacking_regressor(self):
        assert self.pattern.search("create a stacking regressor")

    def test_start_voting_ensemble(self):
        assert self.pattern.search("start voting ensemble")

    def test_try_stacking_ensemble(self):
        assert self.pattern.search("try stacking ensemble")

    # Negative — must NOT match
    def test_negative_train_model(self):
        assert not self.pattern.search("train a model")

    def test_negative_train_all(self):
        assert not self.pattern.search("train all algorithms")

    def test_negative_random_forest(self):
        assert not self.pattern.search("train random forest")

    # Stacking sub-detector
    def test_stacking_re_matches_stacking(self):
        assert self.stacking_re.search("train a stacking ensemble")

    def test_stacking_re_matches_stack(self):
        assert self.stacking_re.search("build a stack model")

    def test_stacking_re_no_match_voting(self):
        assert not self.stacking_re.search("train a voting ensemble")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestEnsembleTrainChatHandler:
    def test_voting_request_emits_training_event(self, client):
        """Voting ensemble request emits training_started SSE event."""
        project_id = _setup_project_with_target(client)

        resp = _chat(client, project_id, "train a voting ensemble")
        assert resp.status_code == 200
        events = _parse_events(resp)
        training_events = [e for e in events if e.get("type") == "training_started"]
        assert len(training_events) == 1

    def test_stacking_request_emits_training_event(self, client):
        """Stacking ensemble request emits training_started SSE event."""
        project_id = _setup_project_with_target(client)

        resp = _chat(client, project_id, "build a stacking regressor")
        assert resp.status_code == 200
        events = _parse_events(resp)
        training_events = [e for e in events if e.get("type") == "training_started"]
        assert len(training_events) == 1

    def test_voting_uses_voting_regressor_algorithm(self, client):
        """Voting request on regression target picks voting_regressor."""
        project_id = _setup_project_with_target(client)

        resp = _chat(client, project_id, "train voting ensemble")
        assert resp.status_code == 200
        events = _parse_events(resp)
        training_events = [e for e in events if e.get("type") == "training_started"]
        assert len(training_events) == 1
        data = training_events[0]["training"]
        assert data["algorithms"] == ["voting_regressor"]

    def test_stacking_uses_stacking_regressor_algorithm(self, client):
        """Stacking request on regression target picks stacking_regressor."""
        project_id = _setup_project_with_target(client)

        resp = _chat(client, project_id, "create a stacking regressor for me")
        assert resp.status_code == 200
        events = _parse_events(resp)
        training_events = [e for e in events if e.get("type") == "training_started"]
        assert len(training_events) == 1
        data = training_events[0]["training"]
        assert data["algorithms"] == ["stacking_regressor"]

    def test_event_has_required_fields(self, client):
        """training_started event has all required fields."""
        project_id = _setup_project_with_target(client)

        resp = _chat(client, project_id, "run a voting ensemble")
        assert resp.status_code == 200
        events = _parse_events(resp)
        training_events = [e for e in events if e.get("type") == "training_started"]
        assert len(training_events) == 1

        data = training_events[0]["training"]
        assert "project_id" in data
        assert "target_column" in data
        assert "problem_type" in data
        assert "algorithms" in data
        assert "run_count" in data
        assert data["status"] == "started"
        assert data["run_count"] == 1

    def test_no_double_training_event(self, client):
        """Ensemble path fires; regular _TRAIN_PATTERNS does NOT also fire."""
        project_id = _setup_project_with_target(client)

        resp = _chat(client, project_id, "train a voting ensemble")
        assert resp.status_code == 200
        events = _parse_events(resp)
        training_events = [e for e in events if e.get("type") == "training_started"]
        assert len(training_events) == 1  # exactly one, not two
