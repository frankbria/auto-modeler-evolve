"""Tests for Saved Scenario Comparison (Track D perpetual).

Covers:
- _SAVE_SCENARIO_PATTERNS detection
- _VIEW_SCENARIOS_PATTERNS detection
- _DELETE_SCENARIO_PATTERNS detection
- _parse_save_scenario_request() helper
- compute_scenario_comparison() pure function
- REST GET /api/deploy/{id}/scenarios
- REST POST /api/deploy/{id}/scenarios
- REST DELETE /api/deploy/{id}/scenarios/{id}
- REST DELETE /api/deploy/{id}/scenarios (clear all)
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session


# ---------------------------------------------------------------------------
# Pattern detection: _SAVE_SCENARIO_PATTERNS
# ---------------------------------------------------------------------------


def test_save_pattern_as_name():
    from api.chat import _SAVE_SCENARIO_PATTERNS

    assert _SAVE_SCENARIO_PATTERNS.search("save this as Q2 Base")


def test_save_pattern_remember():
    from api.chat import _SAVE_SCENARIO_PATTERNS

    assert _SAVE_SCENARIO_PATTERNS.search("remember this scenario as High Risk")


def test_save_pattern_bookmark():
    from api.chat import _SAVE_SCENARIO_PATTERNS

    assert _SAVE_SCENARIO_PATTERNS.search("bookmark this prediction as Control")


def test_save_pattern_with_overrides():
    from api.chat import _SAVE_SCENARIO_PATTERNS

    assert _SAVE_SCENARIO_PATTERNS.search(
        "save discount=0.1 quantity=100 as Q2 Optimistic"
    )


def test_save_pattern_no_match_plain_save():
    from api.chat import _SAVE_SCENARIO_PATTERNS

    # "save" without "as <name>" should not match
    assert not _SAVE_SCENARIO_PATTERNS.search("save my work")


# ---------------------------------------------------------------------------
# Pattern detection: _VIEW_SCENARIOS_PATTERNS
# ---------------------------------------------------------------------------


def test_view_pattern_show_saved():
    from api.chat import _VIEW_SCENARIOS_PATTERNS

    assert _VIEW_SCENARIOS_PATTERNS.search("show my saved scenarios")


def test_view_pattern_compare():
    from api.chat import _VIEW_SCENARIOS_PATTERNS

    assert _VIEW_SCENARIOS_PATTERNS.search("compare my scenarios")


def test_view_pattern_scenario_comparison():
    from api.chat import _VIEW_SCENARIOS_PATTERNS

    assert _VIEW_SCENARIOS_PATTERNS.search("scenario comparison")


def test_view_pattern_library():
    from api.chat import _VIEW_SCENARIOS_PATTERNS

    assert _VIEW_SCENARIOS_PATTERNS.search("scenario library")


def test_view_pattern_no_match_unrelated():
    from api.chat import _VIEW_SCENARIOS_PATTERNS

    assert not _VIEW_SCENARIOS_PATTERNS.search("compare model accuracy")


# ---------------------------------------------------------------------------
# Pattern detection: _DELETE_SCENARIO_PATTERNS
# ---------------------------------------------------------------------------


def test_delete_pattern_delete_name():
    from api.chat import _DELETE_SCENARIO_PATTERNS

    assert _DELETE_SCENARIO_PATTERNS.search("delete scenario Base Case")


def test_delete_pattern_remove():
    from api.chat import _DELETE_SCENARIO_PATTERNS

    assert _DELETE_SCENARIO_PATTERNS.search("remove scenario Q2 Base")


def test_delete_pattern_clear_all():
    from api.chat import _DELETE_SCENARIO_PATTERNS

    assert _DELETE_SCENARIO_PATTERNS.search("clear all scenarios")


def test_delete_pattern_no_match_unrelated():
    from api.chat import _DELETE_SCENARIO_PATTERNS

    assert not _DELETE_SCENARIO_PATTERNS.search("delete my account")


# ---------------------------------------------------------------------------
# _parse_save_scenario_request helper
# ---------------------------------------------------------------------------


def test_parse_save_name_only():
    from api.chat import _parse_save_scenario_request

    name, overrides = _parse_save_scenario_request("save this as Base Case")
    assert name == "Base Case"
    assert overrides == {}


def test_parse_save_with_overrides():
    from api.chat import _parse_save_scenario_request

    name, overrides = _parse_save_scenario_request(
        "save discount=0.1 quantity=100 as Q2 Optimistic"
    )
    assert name == "Q2 Optimistic"
    assert overrides["discount"] == pytest.approx(0.1)
    assert overrides["quantity"] == 100


def test_parse_save_no_as_returns_none():
    from api.chat import _parse_save_scenario_request

    name, overrides = _parse_save_scenario_request("save this configuration")
    assert name is None


def test_parse_save_string_override():
    from api.chat import _parse_save_scenario_request

    name, overrides = _parse_save_scenario_request("save region=West as West Case")
    assert overrides["region"] == "West"


# ---------------------------------------------------------------------------
# compute_scenario_comparison pure function
# ---------------------------------------------------------------------------


def test_comparison_empty():
    from core.deployer import compute_scenario_comparison

    result = compute_scenario_comparison([], "regression")
    assert result["n_scenarios"] == 0
    assert result["best_scenario"] is None
    assert result["summary"] == "No saved scenarios yet."


def test_comparison_single_regression():
    from core.deployer import compute_scenario_comparison

    scenarios = [
        {
            "id": "1",
            "name": "Base",
            "inputs": {},
            "prediction_value": "100",
            "prediction_numeric": 100.0,
            "confidence": None,
        }
    ]
    result = compute_scenario_comparison(scenarios, "regression")
    assert result["n_scenarios"] == 1
    assert "Base" in result["summary"]


def test_comparison_regression_best_worst():
    from core.deployer import compute_scenario_comparison

    scenarios = [
        {
            "id": "1",
            "name": "Low",
            "inputs": {},
            "prediction_value": "50",
            "prediction_numeric": 50.0,
            "confidence": None,
        },
        {
            "id": "2",
            "name": "High",
            "inputs": {},
            "prediction_value": "200",
            "prediction_numeric": 200.0,
            "confidence": None,
        },
        {
            "id": "3",
            "name": "Mid",
            "inputs": {},
            "prediction_value": "120",
            "prediction_numeric": 120.0,
            "confidence": None,
        },
    ]
    result = compute_scenario_comparison(scenarios, "regression")
    assert result["best_scenario"]["name"] == "High"
    assert result["worst_scenario"]["name"] == "Low"
    assert result["spread"] == pytest.approx(150.0)


def test_comparison_classification_single_class():
    from core.deployer import compute_scenario_comparison

    scenarios = [
        {
            "id": "1",
            "name": "A",
            "inputs": {},
            "prediction_value": "churn",
            "prediction_numeric": None,
            "confidence": 0.8,
        },
        {
            "id": "2",
            "name": "B",
            "inputs": {},
            "prediction_value": "churn",
            "prediction_numeric": None,
            "confidence": 0.7,
        },
    ]
    result = compute_scenario_comparison(scenarios, "classification")
    assert "churn" in result["summary"]
    assert (
        result["best_scenario"] is None
    )  # no regression best/worst for classification


def test_comparison_classification_multi_class():
    from core.deployer import compute_scenario_comparison

    scenarios = [
        {
            "id": "1",
            "name": "A",
            "inputs": {},
            "prediction_value": "churn",
            "prediction_numeric": None,
            "confidence": 0.8,
        },
        {
            "id": "2",
            "name": "B",
            "inputs": {},
            "prediction_value": "no_churn",
            "prediction_numeric": None,
            "confidence": 0.9,
        },
    ]
    result = compute_scenario_comparison(scenarios, "classification")
    assert "2 classes" in result["summary"]


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_with_dep(tmp_path, monkeypatch):
    """Return (TestClient, deployment_id) using a temp-file DB."""
    import db as db_module
    from main import app
    import models  # noqa: F401
    import models.saved_scenario  # noqa: F401  # ensure table registered

    test_engine = create_engine(f"sqlite:///{tmp_path / 'sc_test.db'}")
    monkeypatch.setattr(db_module, "engine", test_engine)
    SQLModel.metadata.create_all(test_engine)

    from db import get_session

    def override_session():
        with Session(test_engine) as s:
            yield s

    app.dependency_overrides[get_session] = override_session

    with Session(test_engine) as s:
        from models.deployment import Deployment

        dep = Deployment(
            id="dep-sc-1",
            project_id="proj-1",
            model_run_id="run-1",
            endpoint_path="/api/predict/dep-sc-1",
            dashboard_url="/predict/dep-sc-1",
            is_active=True,
        )
        s.add(dep)
        s.commit()

    with TestClient(app) as client:
        yield client, "dep-sc-1"

    app.dependency_overrides.clear()


def test_list_scenarios_empty(client_with_dep):
    client, dep_id = client_with_dep
    resp = client.get(f"/api/deploy/{dep_id}/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_scenarios"] == 0
    assert data["scenarios"] == []


def test_save_and_list_scenario(client_with_dep):
    client, dep_id = client_with_dep
    payload = {
        "name": "Q2 Base",
        "inputs": {"discount": 0.1},
        "prediction_value": "100",
        "prediction_numeric": 100.0,
    }
    resp = client.post(f"/api/deploy/{dep_id}/scenarios", json=payload)
    assert resp.status_code == 200
    assert resp.json()["created"] is True

    list_resp = client.get(f"/api/deploy/{dep_id}/scenarios")
    assert list_resp.status_code == 200
    assert list_resp.json()["n_scenarios"] == 1
    assert list_resp.json()["scenarios"][0]["name"] == "Q2 Base"


def test_save_scenario_empty_name_rejected(client_with_dep):
    client, dep_id = client_with_dep
    resp = client.post(
        f"/api/deploy/{dep_id}/scenarios",
        json={"name": "   ", "inputs": {}},
    )
    assert resp.status_code == 400


def test_delete_scenario(client_with_dep):
    client, dep_id = client_with_dep
    save_resp = client.post(
        f"/api/deploy/{dep_id}/scenarios",
        json={
            "name": "Temp",
            "inputs": {},
            "prediction_value": "50",
            "prediction_numeric": 50.0,
        },
    )
    scenario_id = save_resp.json()["id"]

    del_resp = client.delete(f"/api/deploy/{dep_id}/scenarios/{scenario_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    list_resp = client.get(f"/api/deploy/{dep_id}/scenarios")
    assert list_resp.json()["n_scenarios"] == 0


def test_clear_all_scenarios(client_with_dep):
    client, dep_id = client_with_dep
    for i in range(3):
        client.post(
            f"/api/deploy/{dep_id}/scenarios",
            json={
                "name": f"S{i}",
                "inputs": {},
                "prediction_value": str(i * 10),
                "prediction_numeric": float(i * 10),
            },
        )

    clear_resp = client.delete(f"/api/deploy/{dep_id}/scenarios")
    assert clear_resp.status_code == 200
    assert clear_resp.json()["deleted"] == 3

    list_resp = client.get(f"/api/deploy/{dep_id}/scenarios")
    assert list_resp.json()["n_scenarios"] == 0


def test_delete_scenario_not_found(client_with_dep):
    client, dep_id = client_with_dep
    resp = client.delete(f"/api/deploy/{dep_id}/scenarios/nonexistent-id")
    assert resp.status_code == 404


def test_list_scenarios_unknown_deployment(client_with_dep):
    client, _ = client_with_dep
    resp = client.get("/api/deploy/no-such-dep/scenarios")
    assert resp.status_code == 404
