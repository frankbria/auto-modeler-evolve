"""Tests for the share link feature — REST endpoint and chat intent patterns.

Covers:
- _SHARE_LINK_PATTERNS regex — NL intent detection
- GET /api/deploy/{id}/share-link
- Chat handler: share_link SSE event emitted when deployment is active
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
    b"Widget C,South,7,670.0\n"
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
        "generate a pre-filled link for units=100, region=North",
        "create a shareable link with these values: units=100",
        "copy this scenario as a link",
        "share this prediction as a link",
        "give me a scenario link",
        "make a pre-filled url with units: 50",
        "I want a shareable prediction URL",
        "link for this scenario",
        "build a pre-filled dashboard link",
    ],
)
def test_share_link_patterns_match(phrase: str) -> None:
    from api.chat import _SHARE_LINK_PATTERNS

    assert _SHARE_LINK_PATTERNS.search(phrase), f"Pattern did not match: {phrase!r}"


@pytest.mark.parametrize(
    "phrase",
    [
        "show my prediction history",
        "give me embed code for my dashboard",
        "what is the dashboard title?",
        "lock region to North",
    ],
)
def test_share_link_patterns_no_false_positives(phrase: str) -> None:
    from api.chat import _SHARE_LINK_PATTERNS

    assert not _SHARE_LINK_PATTERNS.search(
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


def _build_deployed_project(client, project_name="Share Link Test"):
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


def test_share_link_endpoint_returns_data(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.get(f"/api/deploy/{dep_id}/share-link")
    assert r.status_code == 200
    data = r.json()
    assert data["deployment_id"] == dep_id
    assert "/predict/" in data["dashboard_url"]
    assert "prefilled_url" in data
    assert "feature_values" in data
    assert "feature_count" in data
    assert "title" in data
    assert "summary" in data


def test_share_link_endpoint_no_features_returns_clean_url(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.get(f"/api/deploy/{dep_id}/share-link")
    assert r.status_code == 200
    data = r.json()
    assert data["feature_count"] == 0
    assert data["prefilled_url"] == data["dashboard_url"]
    assert "?" not in data["prefilled_url"]


def test_share_link_endpoint_with_feature_values(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    features = json.dumps({"units": "100", "region": "North"})
    r = client.get(f"/api/deploy/{dep_id}/share-link", params={"features": features})
    assert r.status_code == 200
    data = r.json()
    assert data["feature_count"] == 2
    assert "units=100" in data["prefilled_url"]
    assert "region=North" in data["prefilled_url"]
    assert data["feature_values"]["units"] == "100"
    assert data["feature_values"]["region"] == "North"


def test_share_link_endpoint_auto_title(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.get(f"/api/deploy/{dep_id}/share-link")
    assert r.status_code == 200
    title = r.json()["title"]
    assert "Revenue" in title


def test_share_link_endpoint_uses_custom_title(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    client.put(
        f"/api/deploy/{dep_id}/dashboard-metadata",
        params={"title": "Q2 Revenue Forecast"},
    )
    r = client.get(f"/api/deploy/{dep_id}/share-link")
    assert r.status_code == 200
    assert r.json()["title"] == "Q2 Revenue Forecast"


def test_share_link_endpoint_404_for_unknown(client):
    r = client.get("/api/deploy/no-such-deployment/share-link")
    assert r.status_code == 404


def test_share_link_dashboard_url_format(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.get(f"/api/deploy/{dep_id}/share-link")
    url = r.json()["dashboard_url"]
    assert url == f"/predict/{dep_id}"


# ---------------------------------------------------------------------------
# Chat integration: share_link SSE event
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
        d = _build_deployed_project(client, project_name="Share Link Chat Test")
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


def test_share_link_chat_event_emitted(chat_client):
    from unittest.mock import patch

    project_id, dep_id = _deploy_for_chat(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["Here is your pre-filled scenario link."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(
            chat_client, project_id, "create a shareable link for this scenario"
        )

    share_events = [e for e in events if e.get("type") == "share_link"]
    assert share_events, "No share_link SSE event emitted"
    sl = share_events[0]["share_link"]
    assert "dashboard_url" in sl
    assert "/predict/" in sl["dashboard_url"]
    assert "prefilled_url" in sl
    assert "summary" in sl


def test_share_link_no_event_without_deployment(chat_client):
    from unittest.mock import patch

    r = chat_client.post("/api/projects", json={"name": "No Deployment Project"})
    project_id = r.json()["id"]

    mock_stream = iter(["You need to deploy a model first."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "give me a scenario link")

    share_events = [e for e in events if e.get("type") == "share_link"]
    assert not share_events, "share_link event should not fire without a deployment"


def test_share_link_chat_bookmark_variant(chat_client):
    from unittest.mock import patch

    project_id, dep_id = _deploy_for_chat(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["Here is your bookmark link."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "bookmark this scenario")

    share_events = [e for e in events if e.get("type") == "share_link"]
    assert share_events, "No share_link event for bookmark phrase"


def test_share_link_chat_copy_variant(chat_client):
    from unittest.mock import patch

    project_id, dep_id = _deploy_for_chat(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["Here is your shareable link."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "share this prediction as a link")

    share_events = [e for e in events if e.get("type") == "share_link"]
    assert share_events, "No share_link event for 'share as link' phrase"
