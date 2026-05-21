"""Tests for the "What's Next?" workflow guidance card chat feature.

Pattern:  _WHAT_NEXT_PATTERNS
Handler:  what_next_event block in send_message()
SSE type: what_next
"""

from __future__ import annotations

import json
import pathlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _WHAT_NEXT_PATTERNS

# ---------------------------------------------------------------------------
# 1. Regex unit tests (no DB needed)
# ---------------------------------------------------------------------------

SHOULD_MATCH = [
    "what's my next step?",
    "What is my next step?",
    "what should I do next?",
    "what can I do now?",
    "guide me",
    "guide me through next steps",
    "help me get started",
    "help me understand next steps",
    "I'm not sure what to do",
    "I am confused what to do",
    "show me my options",
    "show me next steps",
    "show me what to do next",
    "where do I go from here?",
    "where do I start next?",
    "how do I get started?",
    "how do I proceed from here?",
    "what should I work on now?",
    "what should I work on next?",
    "what do I focus on next?",
]

SHOULD_NOT_MATCH = [
    "train a model",
    "deploy my model",
    "show me the correlations",
    "what is the mean of revenue?",
    "I want to predict sales",
    "upload a new file",
]


@pytest.mark.parametrize("phrase", SHOULD_MATCH)
def test_pattern_matches(phrase: str) -> None:
    assert _WHAT_NEXT_PATTERNS.search(phrase) is not None, f"Expected match: {phrase!r}"


@pytest.mark.parametrize("phrase", SHOULD_NOT_MATCH)
def test_pattern_no_false_positives(phrase: str) -> None:
    assert _WHAT_NEXT_PATTERNS.search(phrase) is None, f"Unexpected match: {phrase!r}"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    db_module.engine = engine

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse(raw: bytes) -> list[dict]:
    """Parse SSE response bytes into list of JSON event dicts."""
    events = []
    for line in raw.decode().splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _find_event(events: list[dict], event_type: str) -> dict | None:
    return next((e for e in events if e.get("type") == event_type), None)


def _create_project(client: TestClient) -> str:
    resp = client.post("/api/projects", json={"name": "WhatNext Project"})
    assert resp.status_code in (200, 201)
    return resp.json()["id"]


def _upload_dataset(client: TestClient, project_id: str) -> str:
    """Upload a minimal CSV and return dataset id."""
    sample = pathlib.Path(__file__).parent.parent / "data/sample/sample_sales.csv"
    if sample.exists():
        with open(sample, "rb") as f:
            resp = client.post(
                "/api/data/upload",
                files={"file": ("sales.csv", f, "text/csv")},
                data={"project_id": project_id},
            )
    else:
        csv_bytes = b"revenue,units,region\n100,10,North\n200,20,South\n150,15,East\n"
        resp = client.post(
            "/api/data/upload",
            files={"file": ("sales.csv", csv_bytes, "text/csv")},
            data={"project_id": project_id},
        )
    assert resp.status_code in (200, 201), f"Upload failed: {resp.text}"
    data = resp.json()
    return data.get("id") or data.get("dataset_id", "")


def _chat(client: TestClient, project_id: str, message: str) -> list[dict]:
    """Send a chat message (with mocked Anthropic) and return parsed SSE events."""
    mock_stream = iter(["Here is your workflow guidance."])
    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream
        with client.stream(
            "POST",
            f"/api/chat/{project_id}",
            json={"message": message, "conversation_id": None},
            headers={"accept": "text/event-stream"},
        ) as resp:
            events = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except Exception:
                        pass
    return events


# ---------------------------------------------------------------------------
# 2. Handler tests (stage detection)
# ---------------------------------------------------------------------------


def test_what_next_no_dataset_emits_upload_stage(client):
    """With no dataset loaded, guidance should be 'upload' stage."""
    pid = _create_project(client)
    events = _chat(client, pid, "what's my next step?")
    ev = _find_event(events, "what_next")
    assert ev is not None, "Expected what_next SSE event"
    data = ev["what_next"]
    assert data["stage"] == "upload"
    assert data["progress"] < 25
    assert len(data["steps"]) == 3


def test_what_next_with_dataset_emits_explore_stage(client):
    """With a dataset but no trained model, guidance should be 'explore' stage."""
    pid = _create_project(client)
    _upload_dataset(client, pid)
    events = _chat(client, pid, "what can I do now?")
    ev = _find_event(events, "what_next")
    assert ev is not None, "Expected what_next SSE event"
    data = ev["what_next"]
    assert data["stage"] == "explore"
    assert 20 <= data["progress"] <= 40
    assert len(data["steps"]) == 3
    assert data["summary"]


def test_what_next_stage_label_present(client):
    """stage_label should be a non-empty string."""
    pid = _create_project(client)
    events = _chat(client, pid, "show me my options")
    ev = _find_event(events, "what_next")
    assert ev is not None
    assert ev["what_next"]["stage_label"]


def test_what_next_steps_have_required_fields(client):
    """Each step must have icon, title, description, action."""
    pid = _create_project(client)
    events = _chat(client, pid, "guide me")
    ev = _find_event(events, "what_next")
    assert ev is not None
    for step in ev["what_next"]["steps"]:
        assert "icon" in step
        assert "title" in step
        assert "description" in step
        assert "action" in step


def test_what_next_no_event_for_unrelated_message(client):
    """A message that doesn't match _WHAT_NEXT_PATTERNS should not emit what_next."""
    pid = _create_project(client)
    events = _chat(client, pid, "show me the correlations between columns")
    ev = _find_event(events, "what_next")
    assert ev is None, "Unexpected what_next event for unrelated message"


def test_what_next_summary_is_string(client):
    """The summary field should be a non-empty string."""
    pid = _create_project(client)
    events = _chat(client, pid, "what should I do next?")
    ev = _find_event(events, "what_next")
    assert ev is not None
    summary = ev["what_next"]["summary"]
    assert isinstance(summary, str) and len(summary) > 0


def test_what_next_progress_in_valid_range(client):
    """progress must be in the range [0, 100]."""
    pid = _create_project(client)
    events = _chat(client, pid, "help me get started")
    ev = _find_event(events, "what_next")
    assert ev is not None
    progress = ev["what_next"]["progress"]
    assert 0 <= progress <= 100


def test_what_next_with_dataset_summary_is_detailed(client):
    """Explore-stage summary should be non-trivial text."""
    pid = _create_project(client)
    _upload_dataset(client, pid)
    events = _chat(client, pid, "where do I go from here?")
    ev = _find_event(events, "what_next")
    assert ev is not None
    data = ev["what_next"]
    assert data["stage"] == "explore"
    assert len(data["summary"]) > 10
