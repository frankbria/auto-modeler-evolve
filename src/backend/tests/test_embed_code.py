"""Tests for the embed code generator — REST endpoint and chat intent patterns.

Covers:
- _EMBED_CODE_PATTERNS regex — NL intent detection
- GET /api/deploy/{id}/embed-code
- Chat handler: embed_code SSE event emitted when deployment is active
"""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

SAMPLE_CSV = (
    b"product,region,units,revenue\n"
    b"Widget A,North,10,1200.5\n"
    b"Widget B,South,8,850.0\n"
    b"Widget A,East,18,2100.75\n"
    b"Widget C,West,4,450.25\n"
    b"Widget B,North,15,1650.0\n"
    b"Widget A,South,9,980.0\n"
    b"Widget C,North,11,1100.25\n"
    b"Widget B,East,16,1750.0\n"
    b"Widget A,West,20,2300.5\n"
    b"Widget C,South,6,620.75\n"
    b"Widget A,North,12,1300.0\n"
    b"Widget B,South,9,950.0\n"
    b"Widget A,East,20,2200.0\n"
    b"Widget C,West,5,520.0\n"
    b"Widget B,North,16,1700.0\n"
    b"Widget A,South,10,1050.0\n"
    b"Widget C,North,12,1150.0\n"
    b"Widget B,East,17,1800.0\n"
    b"Widget A,West,21,2350.0\n"
    b"Widget C,South,7,670.0\n"
)


# ---------------------------------------------------------------------------
# Pattern tests (pure regex — no DB needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "give me embed code for my dashboard",
        "how do I embed my prediction form",
        "iframe snippet for my dashboard",
        "embed this dashboard in our portal",
        "put this dashboard on our intranet",
        "SharePoint embed",
        "embed into our Notion page",
        "how can I embed on our company website",
        "confluence embed for the prediction form",
    ],
)
def test_embed_code_patterns_match(phrase: str) -> None:
    from api.chat import _EMBED_CODE_PATTERNS

    assert _EMBED_CODE_PATTERNS.search(phrase), f"Pattern did not match: {phrase!r}"


@pytest.mark.parametrize(
    "phrase",
    [
        "show my prediction history",
        "what is the dashboard title?",
        "lock region to North",
        "export model card",
    ],
)
def test_embed_code_patterns_no_false_positives(phrase: str) -> None:
    from api.chat import _EMBED_CODE_PATTERNS

    assert not _EMBED_CODE_PATTERNS.search(
        phrase
    ), f"Pattern falsely matched: {phrase!r}"


# ---------------------------------------------------------------------------
# REST endpoint fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models  # noqa — registers all tables via __init__

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    return TestClient(app)


def _build_deployed_project(client, project_name="Embed Code Test"):
    r = client.post("/api/projects", json={"name": project_name})
    project_id = r.json()["id"]

    r = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    dataset_id = r.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": "revenue"})

    train_r = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    run_id = train_r.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    if run["status"] != "done":
        pytest.skip("training did not complete")

    r = client.post(f"/api/deploy/{run_id}")
    deployment_id = r.json()["id"]
    return {"project_id": project_id, "deployment_id": deployment_id}


@pytest.fixture()
def deployed_project(client):
    return _build_deployed_project(client)


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


def test_embed_code_endpoint_returns_data(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.get(f"/api/deploy/{dep_id}/embed-code")
    assert r.status_code == 200
    data = r.json()
    assert data["deployment_id"] == dep_id
    assert "/predict/" in data["dashboard_url"]
    assert data["width"] == "100%"
    assert data["height"] == "700"
    assert "title" in data


def test_embed_code_endpoint_auto_title(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.get(f"/api/deploy/{dep_id}/embed-code")
    assert r.status_code == 200
    title = r.json()["title"]
    assert "Revenue" in title


def test_embed_code_endpoint_uses_custom_title(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    client.put(
        f"/api/deploy/{dep_id}/dashboard-metadata",
        params={"title": "Q2 Revenue Forecast"},
    )
    r = client.get(f"/api/deploy/{dep_id}/embed-code")
    assert r.status_code == 200
    assert r.json()["title"] == "Q2 Revenue Forecast"


def test_embed_code_endpoint_404_for_unknown(client):
    r = client.get("/api/deploy/no-such-deployment/embed-code")
    assert r.status_code == 404


def test_embed_code_dashboard_url_format(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.get(f"/api/deploy/{dep_id}/embed-code")
    url = r.json()["dashboard_url"]
    assert url == f"/predict/{dep_id}"


# ---------------------------------------------------------------------------
# Chat integration: embed_code SSE event
# ---------------------------------------------------------------------------


@pytest.fixture()
def chat_client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    return TestClient(app)


def _deploy_for_chat(client):
    try:
        d = _build_deployed_project(client, project_name="Embed Chat Test")
        return d["project_id"], d["deployment_id"]
    except pytest.skip.Exception:
        return None, None


def _chat(client, project_id, message):
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


def test_embed_code_chat_event_emitted(chat_client):
    from unittest.mock import patch

    project_id, dep_id = _deploy_for_chat(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["Here is your embed code for the prediction dashboard."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "give me embed code for my dashboard")

    embed_events = [e for e in events if e.get("type") == "embed_code"]
    assert embed_events, "No embed_code SSE event emitted"
    ec = embed_events[0]["embed_code"]
    assert "dashboard_url" in ec
    assert "/predict/" in ec["dashboard_url"]
    assert ec["width"] == "100%"
    assert ec["height"] == "700"
    assert "summary" in ec


def test_embed_code_no_event_without_deployment(chat_client):
    from unittest.mock import patch

    r = chat_client.post("/api/projects", json={"name": "No Deployment Project"})
    project_id = r.json()["id"]

    mock_stream = iter(["You need to deploy a model first."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "give me embed code")

    embed_events = [e for e in events if e.get("type") == "embed_code"]
    assert not embed_events, "embed_code event should not fire without a deployment"


def test_embed_code_iframe_variant(chat_client):
    from unittest.mock import patch

    project_id, dep_id = _deploy_for_chat(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["Here is the iframe snippet."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(
            chat_client,
            project_id,
            "can I get an iframe snippet for embedding my dashboard?",
        )

    embed_events = [e for e in events if e.get("type") == "embed_code"]
    assert embed_events, "No embed_code event for iframe phrase"


def test_embed_code_sharepoint_variant(chat_client):
    from unittest.mock import patch

    project_id, dep_id = _deploy_for_chat(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["Here is the SharePoint embed code."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(
            chat_client,
            project_id,
            "how do I add a SharePoint embed for this dashboard?",
        )

    embed_events = [e for e in events if e.get("type") == "embed_code"]
    assert embed_events, "No embed_code event for SharePoint phrase"
