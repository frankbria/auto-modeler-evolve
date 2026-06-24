"""Graceful error handling for the primary chat SSE stream (#18).

The chat streaming generator wraps `client.messages.stream(...)` in try/finally.
Before #18 there was no `except`, so a missing key / 429 / network error raised
mid-stream: the user saw a truncated stream with no error card and the `finally`
persisted an empty assistant message (corrupt history).

These tests force the stream to fail and assert:
- a structured `{"type": "error", ...}` SSE event is emitted,
- the friendly message is exception-specific,
- the missing-key guard short-circuits without calling the client,
- no empty/partial assistant message is persisted.
"""

from __future__ import annotations

import json
import unittest.mock as mock

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

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


def _anthropic_response(status: int) -> httpx.Response:
    return httpx.Response(
        status, request=httpx.Request("POST", "https://api.anthropic.com")
    )


def _auth_error() -> anthropic.AuthenticationError:
    return anthropic.AuthenticationError(
        "bad key", response=_anthropic_response(401), body=None
    )


def _rate_limit_error() -> anthropic.RateLimitError:
    return anthropic.RateLimitError(
        "slow down", response=_anthropic_response(429), body=None
    )


def _chat_events_raising(
    client: TestClient, project_id: str, message: str, exc: Exception
) -> list[dict]:
    """Send a chat message whose Anthropic stream raises *exc*; return SSE events."""
    with mock.patch("anthropic.Anthropic") as mock_cls:
        mc = mock.MagicMock()
        mock_cls.return_value = mc
        mc.messages.stream.side_effect = exc
        resp = client.post(f"/api/chat/{project_id}", json={"message": message})
    return _parse_sse(resp.text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sync_client(tmp_path, monkeypatch):
    from main import app

    # The stream's key guard reads the env; ensure a key is present by default so
    # the guard is not what short-circuits the error-path tests.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    test_db = str(tmp_path / "test.db")
    orig_engine = db_module.engine
    db_module.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    import api.data as dm

    dm.UPLOAD_DIR = tmp_path / "uploads"

    yield TestClient(app)
    db_module.engine = orig_engine


@pytest.fixture()
def project_id(sync_client):
    proj = sync_client.post("/api/projects", json={"name": "Chat Error Test"})
    assert proj.status_code in (200, 201), proj.text
    return proj.json()["id"]


def _conversation_messages(sync_client, project_id: str) -> list[dict]:
    """Read the persisted conversation history for a project."""
    from sqlmodel import Session, select

    from models.conversation import Conversation

    with Session(db_module.engine) as session:
        conv = session.exec(
            select(Conversation).where(Conversation.project_id == project_id)
        ).first()
        if conv is None or not conv.messages:
            return []
        return json.loads(conv.messages)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_auth_error_emits_error_event(sync_client, project_id):
    events = _chat_events_raising(sync_client, project_id, "hello", _auth_error())
    errors = [e for e in events if e.get("type") == "error"]
    assert (
        errors
    ), f"expected a type:error SSE event, got {[e.get('type') for e in events]}"
    assert "Authentication" in errors[0]["message"]


def test_rate_limit_error_has_specific_message(sync_client, project_id):
    events = _chat_events_raising(sync_client, project_id, "hello", _rate_limit_error())
    errors = [e for e in events if e.get("type") == "error"]
    assert errors, "expected a type:error SSE event"
    assert "busy" in errors[0]["message"].lower()


def test_missing_api_key_short_circuits_without_client(
    sync_client, project_id, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    with mock.patch("anthropic.Anthropic") as mock_cls:
        mc = mock.MagicMock()
        mock_cls.return_value = mc
        resp = sync_client.post(f"/api/chat/{project_id}", json={"message": "hi"})
        events = _parse_sse(resp.text)
        mc.messages.stream.assert_not_called()

    errors = [e for e in events if e.get("type") == "error"]
    assert errors, "expected an error event when no API key is configured"
    assert "configured" in errors[0]["message"].lower()


def test_no_empty_assistant_message_persisted_on_error(sync_client, project_id):
    _chat_events_raising(sync_client, project_id, "hello", _auth_error())
    messages = _conversation_messages(sync_client, project_id)
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert all(
        m.get("content", "").strip() for m in assistant_msgs
    ), f"an empty assistant message was persisted on error: {assistant_msgs}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
