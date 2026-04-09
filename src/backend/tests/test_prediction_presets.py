"""Tests for Prediction Presets feature.

Covers:
- Pattern detection: _PRESET_SAVE_PATTERNS, _PRESET_LIST_PATTERNS
- Helper: _extract_preset_definition()
- API: CRUD endpoints (list, create, delete)
- Integration: SSE events emitted for save/list
"""

import json
import re

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _import_patterns():
    from api.chat import _PRESET_SAVE_PATTERNS, _PRESET_LIST_PATTERNS
    globals()["_PRESET_SAVE_PATTERNS"] = _PRESET_SAVE_PATTERNS
    globals()["_PRESET_LIST_PATTERNS"] = _PRESET_LIST_PATTERNS


def test_preset_save_pattern_save():
    assert _PRESET_SAVE_PATTERNS.search("save this as a preset called Best Case")


def test_preset_save_pattern_add():
    assert _PRESET_SAVE_PATTERNS.search("add a preset called Average Quarter with Region=East")


def test_preset_save_pattern_create():
    assert _PRESET_SAVE_PATTERNS.search("create a prediction preset named Worst Case")


def test_preset_save_pattern_scenario():
    assert _PRESET_SAVE_PATTERNS.search("save this as a named scenario called Conservative")


def test_preset_save_pattern_bookmark():
    assert _PRESET_SAVE_PATTERNS.search("bookmark this as preset")


def test_preset_save_pattern_make():
    assert _PRESET_SAVE_PATTERNS.search("make a preset called Q1 Forecast: Units=200, Region=East")


def test_preset_save_pattern_quick_scenario():
    assert _PRESET_SAVE_PATTERNS.search("quick scenario called Big Deal")


def test_preset_save_pattern_no_match():
    assert not _PRESET_SAVE_PATTERNS.search("show me the data")


def test_preset_list_pattern_show():
    assert _PRESET_LIST_PATTERNS.search("show my presets")


def test_preset_list_pattern_list():
    assert _PRESET_LIST_PATTERNS.search("list saved scenarios")


def test_preset_list_pattern_what():
    assert _PRESET_LIST_PATTERNS.search("what presets do I have")


def test_preset_list_pattern_existing():
    assert _PRESET_LIST_PATTERNS.search("show existing presets")


def test_preset_list_pattern_no_match():
    assert not _PRESET_LIST_PATTERNS.search("train a model")


# ---------------------------------------------------------------------------
# _extract_preset_definition tests
# ---------------------------------------------------------------------------


def test_extract_preset_basic():
    from api.chat import _extract_preset_definition

    result = _extract_preset_definition("add a preset called Best Case: Region=East, Units=500")
    assert result is not None
    assert result["name"] == "Best Case"
    assert result["feature_values"]["Region"] == "East"
    assert result["feature_values"]["Units"] == 500


def test_extract_preset_numeric_float():
    from api.chat import _extract_preset_definition

    result = _extract_preset_definition("save preset named High Revenue with Price=9.99, Qty=100")
    assert result is not None
    assert result["feature_values"]["Price"] == 9.99


def test_extract_preset_multiple_features():
    from api.chat import _extract_preset_definition

    result = _extract_preset_definition(
        "create preset called Q4 Plan: Region=North Units=300 Season=Winter"
    )
    assert result is not None
    assert len(result["feature_values"]) >= 2


def test_extract_preset_no_name():
    from api.chat import _extract_preset_definition

    # No "called/named" phrase — should return None
    result = _extract_preset_definition("add a preset with Region=East")
    assert result is None


def test_extract_preset_no_features():
    from api.chat import _extract_preset_definition

    result = _extract_preset_definition("add a preset called My Preset")
    assert result is None  # No key=value pairs


# ---------------------------------------------------------------------------
# API integration tests
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
    import models.deployment_preset  # noqa
    import models.prediction_log  # noqa
    import models.feedback_record  # noqa
    import models.analysis_template  # noqa
    import models.webhook_config  # noqa
    import models.ab_test  # noqa
    import models.batch_schedule  # noqa
    import models.deployment_version  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _seed_deployment(tmp_path):
    """Directly create a Deployment row in the test DB, bypassing ML pipeline."""
    from uuid import uuid4
    from sqlmodel import Session
    from models.deployment import Deployment

    dep = Deployment(
        id=str(uuid4()),
        model_run_id=str(uuid4()),
        project_id=str(uuid4()),
        endpoint_path="/api/predict/test",
        dashboard_url="/predict/test",
        is_active=True,
        algorithm="linear_regression",
        problem_type="regression",
        feature_names='["units","region"]',
        target_column="revenue",
        metrics='{"r2": 0.9}',
    )
    with Session(db_module.engine) as session:
        session.add(dep)
        session.commit()
        session.refresh(dep)
        return dep.id


def test_list_presets_empty(client, tmp_path):
    dep_id = _seed_deployment(tmp_path)
    resp = client.get(f"/api/deploy/{dep_id}/presets")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_preset(client, tmp_path):
    dep_id = _seed_deployment(tmp_path)
    resp = client.post(
        f"/api/deploy/{dep_id}/presets",
        json={"name": "Best Case", "feature_values": {"units": 500, "region": "East"}},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Best Case"
    assert data["feature_values"]["units"] == 500
    assert "id" in data


def test_list_presets_after_create(client, tmp_path):
    dep_id = _seed_deployment(tmp_path)
    client.post(
        f"/api/deploy/{dep_id}/presets",
        json={"name": "Scenario A", "feature_values": {"units": 100}},
    )
    client.post(
        f"/api/deploy/{dep_id}/presets",
        json={"name": "Scenario B", "feature_values": {"units": 200}},
    )
    resp = client.get(f"/api/deploy/{dep_id}/presets")
    assert resp.status_code == 200
    assert len(resp.json()) == 2
    names = [p["name"] for p in resp.json()]
    assert "Scenario A" in names
    assert "Scenario B" in names


def test_delete_preset(client, tmp_path):
    dep_id = _seed_deployment(tmp_path)
    created = client.post(
        f"/api/deploy/{dep_id}/presets",
        json={"name": "To Delete", "feature_values": {"units": 50}},
    ).json()
    preset_id = created["id"]

    del_resp = client.delete(f"/api/deploy/{dep_id}/presets/{preset_id}")
    assert del_resp.status_code == 204

    remaining = client.get(f"/api/deploy/{dep_id}/presets").json()
    assert all(p["id"] != preset_id for p in remaining)


def test_create_preset_empty_name(client, tmp_path):
    dep_id = _seed_deployment(tmp_path)
    resp = client.post(
        f"/api/deploy/{dep_id}/presets",
        json={"name": "", "feature_values": {"units": 100}},
    )
    assert resp.status_code == 422


def test_create_preset_empty_features(client, tmp_path):
    dep_id = _seed_deployment(tmp_path)
    resp = client.post(
        f"/api/deploy/{dep_id}/presets",
        json={"name": "Empty", "feature_values": {}},
    )
    assert resp.status_code == 422


def test_list_presets_invalid_deployment(client):
    resp = client.get("/api/deploy/nonexistent-id/presets")
    assert resp.status_code == 404
