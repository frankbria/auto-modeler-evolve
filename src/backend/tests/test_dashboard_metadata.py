"""Tests for prediction dashboard custom title and description via chat.

Covers:
- _DASHBOARD_META_PATTERNS regex — NL intent detection
- _DC_META_TITLE_RE — title extraction
- _DC_META_DESC_RE — description extraction
- _DC_META_CLEAR_RE — clear/remove detection
- _DC_META_STATUS_RE — status query detection
- GET /api/deploy/{id}/dashboard-metadata
- PUT /api/deploy/{id}/dashboard-metadata
- Chat handler: title_set, description_set, both_set, cleared, status actions
- Deployment model fields: dashboard_title, dashboard_description
"""

import io

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


def test_dashboard_meta_patterns_set_title():
    from api.chat import _DASHBOARD_META_PATTERNS

    assert _DASHBOARD_META_PATTERNS.search(
        "set the dashboard title to 'Q2 Revenue Forecast'"
    )
    assert _DASHBOARD_META_PATTERNS.search("change the prediction title to Sales Tool")
    assert _DASHBOARD_META_PATTERNS.search("update dashboard header to Revenue Forecast")
    assert _DASHBOARD_META_PATTERNS.search("give the dashboard name: My Tool")
    assert _DASHBOARD_META_PATTERNS.search("title this dashboard to Q2 Forecast")
    assert _DASHBOARD_META_PATTERNS.search("name this prediction tool Revenue Predictor")


def test_dashboard_meta_patterns_set_description():
    from api.chat import _DASHBOARD_META_PATTERNS

    assert _DASHBOARD_META_PATTERNS.search(
        "set dashboard description to Use this to forecast revenue"
    )
    assert _DASHBOARD_META_PATTERNS.search(
        "add a dashboard description: For internal finance team use only"
    )
    assert _DASHBOARD_META_PATTERNS.search(
        "update the prediction page description to: Forecasting tool"
    )


def test_dashboard_meta_patterns_clear():
    from api.chat import _DASHBOARD_META_PATTERNS

    assert _DASHBOARD_META_PATTERNS.search("clear the dashboard title")
    assert _DASHBOARD_META_PATTERNS.search("remove dashboard description")
    assert _DASHBOARD_META_PATTERNS.search("reset the dashboard header")


def test_dashboard_meta_patterns_status():
    from api.chat import _DASHBOARD_META_PATTERNS

    assert _DASHBOARD_META_PATTERNS.search("what's the dashboard title?")
    assert _DASHBOARD_META_PATTERNS.search("show the dashboard description")
    assert _DASHBOARD_META_PATTERNS.search("what is the dashboard name?")


def test_dashboard_meta_patterns_no_false_positives():
    from api.chat import _DASHBOARD_META_PATTERNS

    assert not _DASHBOARD_META_PATTERNS.search("train my model")
    assert not _DASHBOARD_META_PATTERNS.search("what is my accuracy?")
    assert not _DASHBOARD_META_PATTERNS.search("hide units from the dashboard")
    assert not _DASHBOARD_META_PATTERNS.search("set confidence threshold to 80%")


def test_dc_meta_title_re_extraction():
    from api.chat import _DC_META_TITLE_RE

    m = _DC_META_TITLE_RE.search("set the dashboard title to Q2 Revenue Forecast")
    assert m and "Q2 Revenue Forecast" in m.group(1)

    m = _DC_META_TITLE_RE.search("change dashboard title to 'Sales Tool'")
    assert m and "Sales Tool" in m.group(1)

    m = _DC_META_TITLE_RE.search("title this dashboard to Revenue Predictor")
    assert m and "Revenue Predictor" in m.group(1)


def test_dc_meta_desc_re_extraction():
    from api.chat import _DC_META_DESC_RE

    m = _DC_META_DESC_RE.search(
        "set dashboard description to Use this to forecast quarterly revenue"
    )
    assert m and "forecast quarterly revenue" in m.group(1)

    m = _DC_META_DESC_RE.search("add a dashboard description: For finance team only")
    assert m and "finance team only" in m.group(1)


def test_dc_meta_clear_re():
    from api.chat import _DC_META_CLEAR_RE

    assert _DC_META_CLEAR_RE.search("clear the dashboard title")
    assert _DC_META_CLEAR_RE.search("remove dashboard description")
    assert _DC_META_CLEAR_RE.search("reset the dashboard header")
    assert not _DC_META_CLEAR_RE.search("set dashboard title to Foo")


def test_dc_meta_status_re():
    from api.chat import _DC_META_STATUS_RE

    assert _DC_META_STATUS_RE.search("what's the dashboard title?")
    assert _DC_META_STATUS_RE.search("show the dashboard description")
    assert not _DC_META_STATUS_RE.search("set dashboard title to Foo")


# ---------------------------------------------------------------------------
# REST endpoint fixtures
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


def _build_deployed_project(client, project_name="Dashboard Metadata Test"):
    import time as _time

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
        _time.sleep(0.5)

    if run["status"] != "done":
        pytest.skip("training did not complete")

    r = client.post(f"/api/deploy/{run_id}")
    deployment_id = r.json()["id"]
    return {"project_id": project_id, "deployment_id": deployment_id}


@pytest.fixture()
def deployed_project(client):
    return _build_deployed_project(client)


# ---------------------------------------------------------------------------
# Deployment model field tests
# ---------------------------------------------------------------------------


def test_deployment_has_metadata_fields(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.get(f"/api/deploy/{dep_id}/dashboard-metadata")
    assert r.status_code == 200
    data = r.json()
    assert "dashboard_title" in data
    assert "dashboard_description" in data
    assert data["dashboard_title"] is None
    assert data["dashboard_description"] is None


def test_deployment_auto_title_fallback(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.get(f"/api/deploy/{dep_id}/dashboard-metadata")
    data = r.json()
    assert data["auto_title"] == "Revenue Predictor"


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


def test_get_dashboard_metadata_not_found(client):
    r = client.get("/api/deploy/no-such-id/dashboard-metadata")
    assert r.status_code == 404


def test_put_dashboard_metadata_set_title(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.put(
        f"/api/deploy/{dep_id}/dashboard-metadata",
        params={"title": "Q2 Revenue Forecast"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["dashboard_title"] == "Q2 Revenue Forecast"
    assert data["dashboard_description"] is None


def test_put_dashboard_metadata_set_description(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.put(
        f"/api/deploy/{dep_id}/dashboard-metadata",
        params={"description": "For finance team use only"},
    )
    assert r.status_code == 200
    assert r.json()["dashboard_description"] == "For finance team use only"


def test_put_dashboard_metadata_set_both(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.put(
        f"/api/deploy/{dep_id}/dashboard-metadata",
        params={
            "title": "Regional Forecast",
            "description": "Enter region and units to forecast revenue",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["dashboard_title"] == "Regional Forecast"
    assert data["dashboard_description"] == "Enter region and units to forecast revenue"


def test_put_dashboard_metadata_persists(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    client.put(
        f"/api/deploy/{dep_id}/dashboard-metadata",
        params={"title": "Persisted Title"},
    )
    r = client.get(f"/api/deploy/{dep_id}/dashboard-metadata")
    assert r.json()["dashboard_title"] == "Persisted Title"


def test_put_dashboard_metadata_clear(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    client.put(
        f"/api/deploy/{dep_id}/dashboard-metadata",
        params={"title": "Temp Title", "description": "Temp Desc"},
    )
    r = client.put(
        f"/api/deploy/{dep_id}/dashboard-metadata",
        params={"clear": True},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["dashboard_title"] is None
    assert data["dashboard_description"] is None


def test_put_dashboard_metadata_not_found(client):
    r = client.put(
        "/api/deploy/no-such-id/dashboard-metadata",
        params={"title": "Foo"},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Chat integration tests
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
        d = _build_deployed_project(client, project_name="Meta Chat Test")
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
                    import json

                    events.append(json.loads(line[6:]))
                except Exception:
                    pass
    return events


def test_chat_set_dashboard_title(chat_client):
    from unittest.mock import patch

    project_id, dep_id = _deploy_for_chat(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["Dashboard title set to Q2 Revenue Forecast."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(
            chat_client,
            project_id,
            "set the dashboard title to Q2 Revenue Forecast",
        )

    meta_events = [e for e in events if e.get("type") == "dashboard_metadata"]
    assert meta_events, "Expected a dashboard_metadata SSE event"
    ev = meta_events[0]["dashboard_metadata"]
    assert ev["action"] == "title_set"
    assert "Q2 Revenue Forecast" in (ev["dashboard_title"] or "")

    # Verify persisted
    r = chat_client.get(f"/api/deploy/{dep_id}/dashboard-metadata")
    assert "Q2 Revenue Forecast" in (r.json()["dashboard_title"] or "")


def test_chat_clear_dashboard_metadata(chat_client):
    from unittest.mock import patch

    project_id, dep_id = _deploy_for_chat(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    # Pre-set a title
    chat_client.put(
        f"/api/deploy/{dep_id}/dashboard-metadata",
        params={"title": "Remove This"},
    )

    mock_stream = iter(["Custom title cleared."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "clear the dashboard title")

    meta_events = [e for e in events if e.get("type") == "dashboard_metadata"]
    assert meta_events, "Expected a dashboard_metadata SSE event"
    assert meta_events[0]["dashboard_metadata"]["action"] == "cleared"


def test_chat_dashboard_status_shows_event(chat_client):
    from unittest.mock import patch

    project_id, dep_id = _deploy_for_chat(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["Current dashboard title is not set."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "what's the dashboard title?")

    meta_events = [e for e in events if e.get("type") == "dashboard_metadata"]
    assert meta_events, "Expected a dashboard_metadata SSE event"
    assert meta_events[0]["dashboard_metadata"]["action"] == "status"


def test_chat_metadata_no_deployment_no_event(chat_client):
    from unittest.mock import patch

    r = chat_client.post("/api/projects", json={"name": "No Deploy"})
    project_id = r.json()["id"]

    mock_stream = iter(["No dashboard available."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "set the dashboard title to My Tool")

    # No deployment → no metadata event
    meta_events = [e for e in events if e.get("type") == "dashboard_metadata"]
    assert not meta_events, "Should not emit metadata event with no deployment"
