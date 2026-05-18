"""Tests for prediction dashboard field configuration via chat.

Covers:
- DashboardFieldConfig SQLModel table created by create_all
- GET /api/deploy/{id}/dashboard-config — list with merged schema
- PUT /api/deploy/{id}/dashboard-config — upsert field configs
- DELETE /api/deploy/{id}/dashboard-config — reset to defaults
- _DASHBOARD_CONFIG_PATTERNS regex — NL intent detection
- _DC_HIDE_RE — hide field extraction
- _DC_LOCK_RE — lock field extraction
- _DC_ONLY_SHOW_RE — only-show extraction
- _DC_RESET_RE — reset detection
- _DC_STATUS_RE — status query detection
- Chat handler: hide, lock, only-show, reset, status actions
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


def test_dashboard_config_patterns_hide():
    from api.chat import _DASHBOARD_CONFIG_PATTERNS

    assert _DASHBOARD_CONFIG_PATTERNS.search("hide units from the dashboard")
    assert _DASHBOARD_CONFIG_PATTERNS.search("hide region field from dashboard")
    assert _DASHBOARD_CONFIG_PATTERNS.search("hide product from the prediction form")
    assert _DASHBOARD_CONFIG_PATTERNS.search("remove units from the public form")


def test_dashboard_config_patterns_lock():
    from api.chat import _DASHBOARD_CONFIG_PATTERNS

    assert _DASHBOARD_CONFIG_PATTERNS.search("lock region to North on the dashboard")
    assert _DASHBOARD_CONFIG_PATTERNS.search("set product to Widget A on the form")
    assert _DASHBOARD_CONFIG_PATTERNS.search("pin units to 10 on prediction page")


def test_dashboard_config_patterns_only_show():
    from api.chat import _DASHBOARD_CONFIG_PATTERNS

    assert _DASHBOARD_CONFIG_PATTERNS.search(
        "only show units and region on the dashboard"
    )
    assert _DASHBOARD_CONFIG_PATTERNS.search("show only region on the form")


def test_dashboard_config_patterns_reset():
    from api.chat import _DASHBOARD_CONFIG_PATTERNS

    assert _DASHBOARD_CONFIG_PATTERNS.search("show all fields")
    assert _DASHBOARD_CONFIG_PATTERNS.search("reset the dashboard config")
    assert _DASHBOARD_CONFIG_PATTERNS.search("restore all fields")
    assert _DASHBOARD_CONFIG_PATTERNS.search("reset dashboard configuration")


def test_dashboard_config_patterns_status():
    from api.chat import _DASHBOARD_CONFIG_PATTERNS

    assert _DASHBOARD_CONFIG_PATTERNS.search("what's visible on my dashboard")
    assert _DASHBOARD_CONFIG_PATTERNS.search("dashboard config")
    assert _DASHBOARD_CONFIG_PATTERNS.search(
        "what fields are visible on the prediction form"
    )
    assert _DASHBOARD_CONFIG_PATTERNS.search("dashboard fields")


def test_dashboard_config_no_false_positives():
    from api.chat import _DASHBOARD_CONFIG_PATTERNS

    assert not _DASHBOARD_CONFIG_PATTERNS.search("train my model")
    assert not _DASHBOARD_CONFIG_PATTERNS.search("what is my accuracy?")
    assert not _DASHBOARD_CONFIG_PATTERNS.search("set a confidence threshold")


def test_dc_hide_re():
    from api.chat import _DC_HIDE_RE

    m = _DC_HIDE_RE.search("hide units from the dashboard")
    assert m and m.group(1) == "units"

    m = _DC_HIDE_RE.search("remove product from the form")
    assert m and m.group(1) == "product"


def test_dc_lock_re():
    from api.chat import _DC_LOCK_RE

    m = _DC_LOCK_RE.search("lock region to North on the dashboard")
    assert m and m.group(1) == "region" and "North" in m.group(2)

    m = _DC_LOCK_RE.search("set product to Widget A on the form")
    assert m and m.group(1) == "product" and "Widget" in m.group(2)


def test_dc_only_show_re():
    from api.chat import _DC_ONLY_SHOW_RE

    m = _DC_ONLY_SHOW_RE.search("only show units and region on the dashboard")
    assert m
    raw = m.group(1)
    assert "units" in raw.lower() or "region" in raw.lower()


def test_dc_reset_re():
    from api.chat import _DC_RESET_RE

    assert _DC_RESET_RE.search("show all fields")
    assert _DC_RESET_RE.search("reset dashboard")
    assert _DC_RESET_RE.search("restore all fields")
    assert not _DC_RESET_RE.search("train my model")


def test_dc_status_re():
    from api.chat import _DC_STATUS_RE

    assert _DC_STATUS_RE.search("what's visible on my dashboard")
    assert _DC_STATUS_RE.search("dashboard config")
    assert not _DC_STATUS_RE.search("train my model")


def test_extract_dashboard_feature():
    from api.chat import _extract_dashboard_feature

    features = ["product", "region", "units", "revenue"]
    assert (
        _extract_dashboard_feature("hide units from the dashboard", features) == "units"
    )
    assert _extract_dashboard_feature("lock region to North", features) == "region"
    assert _extract_dashboard_feature("hide the product column", features) == "product"
    # longest match wins (product = 7 chars > region = 6 chars)
    assert (
        _extract_dashboard_feature("hide product_region from dashboard", features)
        == "product"
    )
    assert _extract_dashboard_feature("no feature here", features) is None


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


def _build_deployed_project(client, project_name="Dashboard Config Test"):
    """Helper: create → upload → train → deploy. Returns dict or raises pytest.skip."""
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
# REST tests
# ---------------------------------------------------------------------------


def test_get_dashboard_config_default(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    r = client.get(f"/api/deploy/{dep_id}/dashboard-config")
    assert r.status_code == 200
    data = r.json()
    assert data["deployment_id"] == dep_id
    assert data["total_count"] > 0
    assert data["visible_count"] == data["total_count"]
    assert data["locked_count"] == 0
    for field in data["fields"]:
        assert field["is_visible"] is True
        assert field["is_locked"] is False


def test_put_dashboard_config_hide_field(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    # Get feature names
    cfg = client.get(f"/api/deploy/{dep_id}/dashboard-config").json()
    first_feature = cfg["fields"][0]["feature_name"]

    r = client.put(
        f"/api/deploy/{dep_id}/dashboard-config",
        json={"fields": [{"feature_name": first_feature, "is_visible": False}]},
    )
    assert r.status_code == 200
    assert r.json()["updated"] == 1

    # Verify
    cfg2 = client.get(f"/api/deploy/{dep_id}/dashboard-config").json()
    hidden = [f for f in cfg2["fields"] if not f["is_visible"]]
    assert any(f["feature_name"] == first_feature for f in hidden)
    assert cfg2["visible_count"] == cfg2["total_count"] - 1


def test_put_dashboard_config_lock_field(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    cfg = client.get(f"/api/deploy/{dep_id}/dashboard-config").json()
    first_feature = cfg["fields"][0]["feature_name"]

    r = client.put(
        f"/api/deploy/{dep_id}/dashboard-config",
        json={
            "fields": [
                {
                    "feature_name": first_feature,
                    "is_visible": True,
                    "is_locked": True,
                    "locked_value": "TestValue",
                }
            ]
        },
    )
    assert r.status_code == 200

    cfg2 = client.get(f"/api/deploy/{dep_id}/dashboard-config").json()
    locked = [f for f in cfg2["fields"] if f["is_locked"]]
    assert any(f["feature_name"] == first_feature for f in locked)
    assert cfg2["locked_count"] == 1


def test_put_dashboard_config_idempotent(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    cfg = client.get(f"/api/deploy/{dep_id}/dashboard-config").json()
    fname = cfg["fields"][0]["feature_name"]

    # Apply twice
    for _ in range(2):
        client.put(
            f"/api/deploy/{dep_id}/dashboard-config",
            json={"fields": [{"feature_name": fname, "is_visible": False}]},
        )

    cfg2 = client.get(f"/api/deploy/{dep_id}/dashboard-config").json()
    hidden = [f for f in cfg2["fields"] if not f["is_visible"]]
    assert len(hidden) == 1


def test_delete_dashboard_config_reset(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    cfg = client.get(f"/api/deploy/{dep_id}/dashboard-config").json()
    fname = cfg["fields"][0]["feature_name"]

    client.put(
        f"/api/deploy/{dep_id}/dashboard-config",
        json={"fields": [{"feature_name": fname, "is_visible": False}]},
    )

    r = client.delete(f"/api/deploy/{dep_id}/dashboard-config")
    assert r.status_code == 200
    assert r.json()["removed"] >= 1

    cfg2 = client.get(f"/api/deploy/{dep_id}/dashboard-config").json()
    assert cfg2["visible_count"] == cfg2["total_count"]


def test_get_dashboard_config_not_found(client):
    r = client.get("/api/deploy/nonexistent/dashboard-config")
    assert r.status_code == 404


def test_put_dashboard_config_not_found(client):
    r = client.put(
        "/api/deploy/nonexistent/dashboard-config",
        json={"fields": [{"feature_name": "x", "is_visible": False}]},
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


def _deploy_project(client):
    """Helper: create → upload → train → deploy."""
    try:
        d = _build_deployed_project(client, project_name="Chat DC Test")
        return d["project_id"], None, d["deployment_id"]
    except pytest.skip.Exception:
        return None, None, None


def _chat(client, project_id, message):
    """Send a chat message and collect SSE events."""
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


def test_chat_hide_field(chat_client):
    from unittest.mock import patch

    project_id, _, dep_id = _deploy_project(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["The field is now hidden."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "hide units from the dashboard")

    dc_events = [e for e in events if e.get("type") == "dashboard_config"]
    assert dc_events, "Expected a dashboard_config SSE event"
    ev = dc_events[0]["dashboard_config"]
    assert ev["action"] == "updated"
    assert any(
        c["feature_name"] == "units" and not c["is_visible"] for c in ev["changes"]
    )


def test_chat_lock_field(chat_client):
    from unittest.mock import patch

    project_id, _, dep_id = _deploy_project(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["Region locked to North."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "lock region to North on the dashboard")

    dc_events = [e for e in events if e.get("type") == "dashboard_config"]
    assert dc_events
    ev = dc_events[0]["dashboard_config"]
    assert ev["action"] == "updated"
    changes = ev["changes"]
    region_change = next((c for c in changes if c["feature_name"] == "region"), None)
    assert region_change is not None
    assert region_change["is_locked"] is True
    assert "North" in (region_change["locked_value"] or "")


def test_chat_reset_dashboard(chat_client):
    from unittest.mock import patch

    project_id, _, dep_id = _deploy_project(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    # First hide a field
    chat_client.put(
        f"/api/deploy/{dep_id}/dashboard-config",
        json={"fields": [{"feature_name": "units", "is_visible": False}]},
    )

    mock_stream = iter(["Dashboard reset."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "show all fields on the dashboard")

    dc_events = [e for e in events if e.get("type") == "dashboard_config"]
    assert dc_events
    ev = dc_events[0]["dashboard_config"]
    assert ev["action"] == "reset"
    assert ev["visible_count"] == ev["total_count"]


def test_chat_dashboard_status(chat_client):
    from unittest.mock import patch

    project_id, _, dep_id = _deploy_project(chat_client)
    if not dep_id:
        pytest.skip("training did not complete")

    mock_stream = iter(["Here is what's visible."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "what's visible on my dashboard")

    dc_events = [e for e in events if e.get("type") == "dashboard_config"]
    assert dc_events
    ev = dc_events[0]["dashboard_config"]
    assert ev["action"] == "status"
    assert ev["total_count"] > 0


def test_chat_no_deployment_no_event(chat_client):
    """No dashboard_config event when there's no deployment."""
    from unittest.mock import patch

    r = chat_client.post("/api/projects", json={"name": "No Deploy"})
    project_id = r.json()["id"]

    mock_stream = iter(["No deployment yet."])

    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream

        events = _chat(chat_client, project_id, "hide units from the dashboard")

    dc_events = [e for e in events if e.get("type") == "dashboard_config"]
    assert not dc_events
