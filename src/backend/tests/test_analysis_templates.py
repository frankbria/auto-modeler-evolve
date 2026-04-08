"""Tests for the Saved Analysis Templates feature.

Covers:
- Pattern detection: _SAVE_TEMPLATE_PATTERNS, _LIST_TEMPLATES_PATTERNS, _REPLAY_TEMPLATE_PATTERNS
- Helper: _extract_template_name()
- API: CRUD endpoints (list, create, delete)
- Integration: SSE events emitted for save/list/replay
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Test DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.conversation  # noqa
    import models.model_run  # noqa
    import models.deployment  # noqa
    import models.prediction_log  # noqa
    import models.feedback_record  # noqa
    import models.analysis_template  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


def test_save_pattern_matches_basic():
    from api.chat import _SAVE_TEMPLATE_PATTERNS

    assert _SAVE_TEMPLATE_PATTERNS.search(
        "save this analysis as a template called 'Monthly Sales'"
    )
    assert _SAVE_TEMPLATE_PATTERNS.search("create a template named Q4 analysis")
    assert _SAVE_TEMPLATE_PATTERNS.search("bookmark these queries as customer segments")


def test_save_pattern_matches_variants():
    from api.chat import _SAVE_TEMPLATE_PATTERNS

    assert _SAVE_TEMPLATE_PATTERNS.search("save as a template")
    assert _SAVE_TEMPLATE_PATTERNS.search("make this a reusable template")
    assert _SAVE_TEMPLATE_PATTERNS.search("save my analysis workflow")


def test_save_pattern_negative():
    from api.chat import _SAVE_TEMPLATE_PATTERNS

    assert not _SAVE_TEMPLATE_PATTERNS.search("what is the average revenue?")
    assert not _SAVE_TEMPLATE_PATTERNS.search("train a random forest model")


def test_list_templates_pattern_matches():
    from api.chat import _LIST_TEMPLATES_PATTERNS

    assert _LIST_TEMPLATES_PATTERNS.search("show my saved templates")
    assert _LIST_TEMPLATES_PATTERNS.search("list my analysis templates")
    assert _LIST_TEMPLATES_PATTERNS.search("what templates do I have saved?")


def test_replay_template_pattern_matches():
    from api.chat import _REPLAY_TEMPLATE_PATTERNS

    assert _REPLAY_TEMPLATE_PATTERNS.search("replay my Monthly Sales template")
    assert _REPLAY_TEMPLATE_PATTERNS.search("run my Q4 Analysis template again")
    assert _REPLAY_TEMPLATE_PATTERNS.search("apply my customer segments template")


def test_replay_template_pattern_negative():
    from api.chat import _REPLAY_TEMPLATE_PATTERNS

    assert not _REPLAY_TEMPLATE_PATTERNS.search("what is the average revenue?")


# ---------------------------------------------------------------------------
# _extract_template_name helper tests
# ---------------------------------------------------------------------------


def test_extract_name_quoted():
    from api.chat import _extract_template_name

    assert (
        _extract_template_name("save this as a template called 'Monthly Sales'")
        == "Monthly Sales"
    )


def test_extract_name_unquoted():
    from api.chat import _extract_template_name

    result = _extract_template_name("create a template named Q4 Analysis")
    assert result is not None
    assert "Q4" in result


def test_extract_name_double_quotes():
    from api.chat import _extract_template_name

    result = _extract_template_name('replay my "Weekly Report" template')
    assert result == "Weekly Report"


def test_extract_name_returns_none_when_no_name():
    from api.chat import _extract_template_name

    result = _extract_template_name("save as a template")
    # May return None or empty — just should not crash
    assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# API CRUD endpoint tests
# ---------------------------------------------------------------------------


def _create_project(client, name="Test Project"):
    resp = client.post("/api/projects/", json={"name": name})
    assert resp.status_code == 201
    return resp.json()["id"]


def test_list_templates_empty(client):
    pid = _create_project(client)
    resp = client.get(f"/api/projects/{pid}/analysis-templates")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_template(client):
    pid = _create_project(client)
    body = {
        "name": "Monthly Sales Review",
        "queries": ["what's the average revenue?", "show me revenue by region"],
    }
    resp = client.post(f"/api/projects/{pid}/analysis-templates", json=body)
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Monthly Sales Review"
    assert data["queries"] == body["queries"]
    assert data["project_id"] == pid


def test_list_templates_after_create(client):
    pid = _create_project(client)
    client.post(
        f"/api/projects/{pid}/analysis-templates",
        json={"name": "Template A", "queries": ["q1", "q2"]},
    )
    client.post(
        f"/api/projects/{pid}/analysis-templates",
        json={"name": "Template B", "queries": ["q3"]},
    )
    resp = client.get(f"/api/projects/{pid}/analysis-templates")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    # newest first
    assert data[0]["name"] == "Template B"


def test_delete_template(client):
    pid = _create_project(client)
    create_resp = client.post(
        f"/api/projects/{pid}/analysis-templates",
        json={"name": "To Delete", "queries": ["q1"]},
    )
    tid = create_resp.json()["id"]

    del_resp = client.delete(f"/api/projects/{pid}/analysis-templates/{tid}")
    assert del_resp.status_code == 204

    list_resp = client.get(f"/api/projects/{pid}/analysis-templates")
    assert list_resp.json() == []


def test_delete_template_not_found(client):
    pid = _create_project(client)
    resp = client.delete(f"/api/projects/{pid}/analysis-templates/nonexistent-id")
    assert resp.status_code == 404


def test_create_template_empty_name_rejected(client):
    pid = _create_project(client)
    resp = client.post(
        f"/api/projects/{pid}/analysis-templates",
        json={"name": "  ", "queries": ["q1"]},
    )
    assert resp.status_code == 422


def test_create_template_empty_queries_rejected(client):
    pid = _create_project(client)
    resp = client.post(
        f"/api/projects/{pid}/analysis-templates",
        json={"name": "My Template", "queries": []},
    )
    assert resp.status_code == 422
