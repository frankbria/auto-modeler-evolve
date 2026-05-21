"""Tests for Webhook Management via Chat (Day 61).

Covers:
- _WEBHOOK_CREATE_PATTERNS regex matching
- _WEBHOOK_LIST_CHAT_PATTERNS regex matching
- _WEBHOOK_REMOVE_CHAT_PATTERNS regex matching
- _WEBHOOK_TEST_CHAT_PATTERNS regex matching
- Chat handler emitting webhook_registered SSE event
- Chat handler emitting webhook_list_chat SSE event
- Chat handler emitting webhook_removed_chat SSE event
- Chat handler emitting webhook_test_chat SSE event
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock

import pytest
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
    b"East,160.0,16\nWest,210.0,21\n"
)


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE stream text into a list of event dicts."""
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _chat_events_sync(client, project_id: str, message: str) -> list[dict]:
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
            json={"message": message},
        )
    return _parse_sse(resp.text)


# ---------------------------------------------------------------------------
# Sync fixtures (follow test_webhook_event_history.py pattern)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sync_client(tmp_path):
    from fastapi.testclient import TestClient
    from main import app

    test_db = str(tmp_path / "chat_test.db")
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

    dm.UPLOAD_DIR = tmp_path / "uploads"
    dep.DEPLOY_DIR = tmp_path / "deployments"
    mm.MODELS_DIR = tmp_path / "models"

    yield TestClient(app)
    db_module.engine = orig_engine


@pytest.fixture()
def deployed_project(sync_client, tmp_path):
    """Create project, upload CSV, train, and deploy (sync)."""
    proj = sync_client.post("/api/projects", json={"name": "WebhookMgmtChat"})
    pid = proj.json()["id"]

    upload = sync_client.post(
        "/api/data/upload",
        data={"project_id": pid},
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
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
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestWebhookCreatePatterns:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api.chat import _WEBHOOK_CREATE_PATTERNS

        self.pattern = _WEBHOOK_CREATE_PATTERNS

    def test_register_webhook(self):
        assert self.pattern.search("register a webhook at https://example.com/hook")

    def test_add_webhook(self):
        assert self.pattern.search("add webhook https://myapp.io/callback")

    def test_create_webhook(self):
        assert self.pattern.search("create a webhook for drift events")

    def test_set_up_webhook(self):
        assert self.pattern.search("set up a webhook at https://hooks.example.com")

    def test_configure_webhook(self):
        assert self.pattern.search("configure webhook notifications")

    def test_enable_webhook(self):
        assert self.pattern.search("enable webhook at https://n8n.example.com/wh")

    def test_send_events_to_url(self):
        assert self.pattern.search("send events to https://example.com/hook")

    def test_notify_me_at_url(self):
        assert self.pattern.search("notify me at https://example.com/hook")

    def test_no_false_positive_list(self):
        assert not self.pattern.search("list my webhooks")

    def test_no_false_positive_remove(self):
        assert not self.pattern.search("remove the webhook")


class TestWebhookListChatPatterns:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api.chat import _WEBHOOK_LIST_CHAT_PATTERNS

        self.pattern = _WEBHOOK_LIST_CHAT_PATTERNS

    def test_list_webhooks(self):
        assert self.pattern.search("list my webhooks")

    def test_show_webhooks(self):
        assert self.pattern.search("show webhooks")

    def test_what_webhooks_are_configured(self):
        assert self.pattern.search("what webhooks are configured?")

    def test_active_webhooks(self):
        assert self.pattern.search("what are my active webhooks?")

    def test_registered_webhooks(self):
        assert self.pattern.search("show registered webhooks")

    def test_get_webhooks(self):
        assert self.pattern.search("get my webhooks")

    def test_view_webhooks(self):
        assert self.pattern.search("view webhooks")

    def test_no_false_positive_register(self):
        assert not self.pattern.search("register a webhook")


class TestWebhookRemoveChatPatterns:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api.chat import _WEBHOOK_REMOVE_CHAT_PATTERNS

        self.pattern = _WEBHOOK_REMOVE_CHAT_PATTERNS

    def test_remove_webhook(self):
        assert self.pattern.search("remove webhook https://example.com/hook")

    def test_delete_webhook(self):
        assert self.pattern.search("delete my webhook")

    def test_unregister_webhook(self):
        assert self.pattern.search("unregister webhook")

    def test_disable_webhook(self):
        assert self.pattern.search("disable the webhook")

    def test_cancel_webhook(self):
        assert self.pattern.search("cancel webhook notifications")

    def test_stop_webhook(self):
        assert self.pattern.search("stop my webhook")

    def test_unsubscribe_from_webhook(self):
        assert self.pattern.search("unsubscribe from webhook")

    def test_no_false_positive_list(self):
        assert not self.pattern.search("list webhooks")


class TestWebhookTestChatPatterns:
    @pytest.fixture(autouse=True)
    def _import(self):
        from api.chat import _WEBHOOK_TEST_CHAT_PATTERNS

        self.pattern = _WEBHOOK_TEST_CHAT_PATTERNS

    def test_test_webhook(self):
        assert self.pattern.search("test my webhook")

    def test_verify_webhook(self):
        assert self.pattern.search("verify webhook")

    def test_check_webhook(self):
        assert self.pattern.search("check webhook")

    def test_ping_webhook(self):
        assert self.pattern.search("ping my webhook")

    def test_validate_webhook(self):
        assert self.pattern.search("validate my webhook")

    def test_send_test_webhook(self):
        assert self.pattern.search("send a test webhook")

    def test_no_false_positive_list(self):
        assert not self.pattern.search("list webhooks")


# ---------------------------------------------------------------------------
# Integration: SSE events (sync client + deployed project)
# ---------------------------------------------------------------------------


class TestWebhookManagementChatSSE:
    def test_list_emits_event_no_webhooks(self, sync_client, deployed_project):
        """'show active webhooks' emits webhook_list_chat with total=0 when none registered."""
        events = _chat_events_sync(
            sync_client, deployed_project["project_id"], "show active webhooks"
        )
        types = [e.get("type") for e in events]
        assert "webhook_list_chat" in types, f"Expected webhook_list_chat in {types}"
        wl = next(e for e in events if e.get("type") == "webhook_list_chat")
        assert wl["webhook_list_chat"]["total"] == 0
        assert wl["webhook_list_chat"]["webhooks"] == []

    def test_register_emits_webhook_registered_event(
        self, sync_client, deployed_project
    ):
        """'register a webhook at URL' emits webhook_registered with url and secret."""
        events = _chat_events_sync(
            sync_client,
            deployed_project["project_id"],
            "register a webhook at https://httpbin.org/post for drift events",
        )
        types = [e.get("type") for e in events]
        assert "webhook_registered" in types, f"Expected webhook_registered in {types}"
        wc = next(e for e in events if e.get("type") == "webhook_registered")
        payload = wc["webhook_registered"]
        assert payload["url"] == "https://httpbin.org/post"
        assert "drift_detected" in payload["event_types"]
        assert payload["secret"]
        assert payload["deployment_id"] == deployed_project["deployment_id"]

    def test_remove_emits_event_no_webhooks(self, sync_client, deployed_project):
        """'remove my webhook' emits webhook_removed_chat with removed=[] when none exist."""
        events = _chat_events_sync(
            sync_client, deployed_project["project_id"], "remove my webhook"
        )
        types = [e.get("type") for e in events]
        assert (
            "webhook_removed_chat" in types
        ), f"Expected webhook_removed_chat in {types}"
        wr = next(e for e in events if e.get("type") == "webhook_removed_chat")
        assert wr["webhook_removed_chat"]["removed"] == []

    def test_test_emits_event_no_webhooks(self, sync_client, deployed_project):
        """'test my webhook' emits webhook_test_chat with url=null when none registered."""
        events = _chat_events_sync(
            sync_client, deployed_project["project_id"], "test my webhook"
        )
        types = [e.get("type") for e in events]
        assert "webhook_test_chat" in types, f"Expected webhook_test_chat in {types}"
        wt = next(e for e in events if e.get("type") == "webhook_test_chat")
        assert wt["webhook_test_chat"]["url"] is None

    def test_no_webhook_events_without_deployment(self, sync_client):
        """Without a deployment, no webhook management SSE events are emitted."""
        proj = sync_client.post("/api/projects", json={"name": "NoDep"})
        pid = proj.json()["id"]
        for msg in ["list my webhooks", "remove my webhook", "test my webhook"]:
            events = _chat_events_sync(sync_client, pid, msg)
            types = [e.get("type") for e in events]
            for et in (
                "webhook_list_chat",
                "webhook_registered",
                "webhook_removed_chat",
                "webhook_test_chat",
            ):
                assert (
                    et not in types
                ), f"Unexpected {et} for '{msg}' without deployment"
