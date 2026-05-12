"""Tests for API Key Management via Chat.

Covers:
- _API_KEY_GENERATE_PATTERNS  — 7 NL variant groups
- _API_KEY_DISABLE_PATTERNS   — 4 NL variant groups
- _API_KEY_STATUS_PATTERNS    — 4 NL variant groups
- Chat integration: generate / regenerate / disable / status actions via SSE
"""

from __future__ import annotations

import io
import json
import time
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
# Helpers
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


def _chat_events(client: TestClient, project_id: str, message: str) -> list[dict]:
    """Send a chat message with mocked Anthropic and return parsed SSE events."""
    with mock.patch("anthropic.Anthropic") as mock_cls:
        mc = mock.MagicMock()
        mock_cls.return_value = mc
        ms = mock.MagicMock()
        ms.__enter__ = mock.MagicMock(return_value=ms)
        ms.__exit__ = mock.MagicMock(return_value=False)
        ms.text_stream = iter(["Done."])
        mc.messages.stream.return_value = ms

        resp = client.post(f"/api/chat/{project_id}", json={"message": message})
    return _parse_sse(resp.text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sync_client(tmp_path):
    from main import app

    test_db = str(tmp_path / "test.db")
    orig_engine = db_module.engine
    db_module.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )

    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    import api.data as dm
    import api.deploy as dep
    import api.models as mm

    dm.UPLOAD_DIR = tmp_path / "uploads"
    dep.DEPLOY_DIR = tmp_path / "deployments"
    mm.MODELS_DIR = tmp_path / "models"

    yield TestClient(app)
    db_module.engine = orig_engine


@pytest.fixture()
def deployed_project(sync_client, tmp_path):
    proj = sync_client.post("/api/projects", json={"name": "API Key Chat Test"})
    pid = proj.json()["id"]

    upload = sync_client.post(
        "/api/data/upload",
        data={"project_id": pid},
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    did = upload.json()["dataset_id"]

    sync_client.post(f"/api/features/{did}/apply", json={"transformations": []})
    sync_client.post(
        f"/api/features/{did}/target",
        json={"target_column": "revenue", "problem_type": "regression"},
    )

    train_resp = sync_client.post(
        f"/api/models/{pid}/train",
        json={"algorithms": ["linear_regression"]},
    )
    run_id = train_resp.json()["model_run_ids"][0]
    for _ in range(40):
        runs = sync_client.get(f"/api/models/{pid}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.3)

    dep_r = sync_client.post(f"/api/deploy/{run_id}")
    assert dep_r.status_code in (200, 201), dep_r.text

    return {"project_id": pid, "deployment_id": dep_r.json()["id"]}


# ---------------------------------------------------------------------------
# Pattern unit tests
# ---------------------------------------------------------------------------


class TestApiKeyGeneratePatterns:
    def setup_method(self):
        from api.chat import _API_KEY_GENERATE_PATTERNS

        self.pattern = _API_KEY_GENERATE_PATTERNS

    def test_generate_api_key(self):
        assert self.pattern.search("generate an API key for my deployment")

    def test_create_api_key(self):
        assert self.pattern.search("create an api key")

    def test_enable_api_key(self):
        assert self.pattern.search("enable API key protection")

    def test_protect_endpoint(self):
        assert self.pattern.search("protect my endpoint")

    def test_secure_prediction_api(self):
        assert self.pattern.search("secure my prediction API")

    def test_require_api_key(self):
        assert self.pattern.search("require API key authentication for my model")

    def test_regenerate_key(self):
        assert self.pattern.search("regenerate my API key")

    def test_make_private(self):
        assert self.pattern.search("make my endpoint private")

    def test_no_match_unrelated(self):
        assert not self.pattern.search("show me the correlation matrix")

    def test_no_match_remove(self):
        # Remove intent should NOT match generate
        assert not self.pattern.search("remove the api key protection")


class TestApiKeyDisablePatterns:
    def setup_method(self):
        from api.chat import _API_KEY_DISABLE_PATTERNS

        self.pattern = _API_KEY_DISABLE_PATTERNS

    def test_remove_api_key(self):
        assert self.pattern.search("remove the API key")

    def test_disable_api_key(self):
        assert self.pattern.search("disable API key protection")

    def test_make_endpoint_public(self):
        assert self.pattern.search("make my endpoint public")

    def test_revoke_key(self):
        assert self.pattern.search("revoke the api key")

    def test_no_match_generate(self):
        assert not self.pattern.search("generate an API key")

    def test_no_match_status(self):
        assert not self.pattern.search("is my endpoint protected")

    def test_no_match_unrelated(self):
        assert not self.pattern.search("train my model")


class TestApiKeyStatusPatterns:
    def setup_method(self):
        from api.chat import _API_KEY_STATUS_PATTERNS

        self.pattern = _API_KEY_STATUS_PATTERNS

    def test_show_api_key_status(self):
        assert self.pattern.search("show my API key status")

    def test_is_endpoint_protected(self):
        assert self.pattern.search("is my endpoint protected?")

    def test_do_i_have_api_key(self):
        assert self.pattern.search("do I have an API key on my model?")

    def test_check_api_key(self):
        assert self.pattern.search("check my api key")

    def test_api_key_enabled(self):
        assert self.pattern.search("api key status")

    def test_no_match_generate(self):
        assert not self.pattern.search("generate an API key")

    def test_no_match_unrelated(self):
        assert not self.pattern.search("cluster my customers")


# ---------------------------------------------------------------------------
# Chat integration tests
# ---------------------------------------------------------------------------


class TestApiKeyChatIntegration:
    def test_generate_api_key_via_chat(self, sync_client, deployed_project):
        events = _chat_events(
            sync_client,
            deployed_project["project_id"],
            "generate an API key for my deployment",
        )
        api_key_events = [e for e in events if e.get("type") == "api_key_result"]
        assert api_key_events, "Expected api_key_result SSE event"
        result = api_key_events[0]["api_key_result"]
        assert result["action"] in ("generated", "regenerated")
        assert result["is_protected"] is True
        assert "api_key" in result
        assert len(result["api_key"]) > 20

    def test_generate_returns_key_once(self, sync_client, deployed_project):
        events = _chat_events(
            sync_client,
            deployed_project["project_id"],
            "protect my endpoint with an API key",
        )
        api_key_events = [e for e in events if e.get("type") == "api_key_result"]
        assert api_key_events
        assert api_key_events[0]["api_key_result"].get("api_key")

    def test_regenerate_api_key_via_chat(self, sync_client, deployed_project):
        pid = deployed_project["project_id"]
        _chat_events(sync_client, pid, "generate an API key")
        events = _chat_events(sync_client, pid, "regenerate my API key")
        api_key_events = [e for e in events if e.get("type") == "api_key_result"]
        assert api_key_events
        result = api_key_events[0]["api_key_result"]
        assert result["action"] == "regenerated"
        assert result["is_protected"] is True

    def test_disable_api_key_via_chat(self, sync_client, deployed_project):
        pid = deployed_project["project_id"]
        _chat_events(sync_client, pid, "generate an API key")
        events = _chat_events(sync_client, pid, "remove the API key protection")
        api_key_events = [e for e in events if e.get("type") == "api_key_result"]
        assert api_key_events
        result = api_key_events[0]["api_key_result"]
        assert result["action"] == "disabled"
        assert result["is_protected"] is False
        assert "api_key" not in result

    def test_status_when_no_key(self, sync_client, deployed_project):
        events = _chat_events(
            sync_client, deployed_project["project_id"], "is my endpoint protected?"
        )
        api_key_events = [e for e in events if e.get("type") == "api_key_result"]
        assert api_key_events
        result = api_key_events[0]["api_key_result"]
        assert result["action"] == "status"
        assert result["is_protected"] is False

    def test_status_after_generate(self, sync_client, deployed_project):
        pid = deployed_project["project_id"]
        _chat_events(sync_client, pid, "generate an API key")
        events = _chat_events(sync_client, pid, "show my API key status")
        api_key_events = [e for e in events if e.get("type") == "api_key_result"]
        assert api_key_events
        result = api_key_events[0]["api_key_result"]
        assert result["action"] == "status"
        assert result["is_protected"] is True
        assert "api_key" not in result

    def test_no_event_without_deployment(self, sync_client):
        proj = sync_client.post("/api/projects", json={"name": "No Deploy"})
        pid = proj.json()["id"]
        events = _chat_events(sync_client, pid, "generate an API key for my model")
        api_key_events = [e for e in events if e.get("type") == "api_key_result"]
        assert not api_key_events
