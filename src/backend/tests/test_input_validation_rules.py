"""Tests for prediction input validation rules.

Covers:
- validate_prediction_inputs() pure function
- _INPUT_VALIDATION_PATTERNS regex — NL intent detection
- _IV_RANGE_RE, _IV_BOUND_RE, _IV_ONE_OF_RE extraction regexes
- POST /api/deploy/{id}/input-validation-rules — create
- GET /api/deploy/{id}/input-validation-rules — list
- DELETE /api/deploy/{id}/input-validation-rules/{id} — delete
- make_prediction() rejects inputs that violate rules (HTTP 422)
- Chat handler — creates rule from natural-language message
"""

import io
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Shared CSV data for integration fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV = b"""product,region,units,revenue
Widget A,North,10,1200.5
Widget B,South,8,850.0
Widget A,East,18,2100.75
Widget C,West,4,450.25
Widget B,North,15,1650.0
Widget A,South,9,980.0
Widget C,North,11,1100.25
Widget B,East,16,1750.0
Widget A,West,20,2300.5
Widget C,South,6,620.75
Widget A,North,12,1300.0
Widget B,South,9,950.0
Widget A,East,20,2200.0
Widget C,West,5,520.0
Widget B,North,16,1700.0
Widget A,South,10,1050.0
Widget C,North,12,1150.0
Widget B,East,17,1800.0
Widget A,West,21,2350.0
Widget C,South,7,670.0
"""

# ---------------------------------------------------------------------------
# Pure function — validate_prediction_inputs
# ---------------------------------------------------------------------------


def test_validate_range_pass():
    from core.validator import validate_prediction_inputs

    rules = [{"feature_name": "age", "rule_type": "range", "min_val": 0.0, "max_val": 120.0, "allowed_values": None}]
    valid, violations = validate_prediction_inputs({"age": 35}, rules)
    assert valid
    assert violations == []


def test_validate_range_fail_below():
    from core.validator import validate_prediction_inputs

    rules = [{"feature_name": "age", "rule_type": "range", "min_val": 0.0, "max_val": 120.0, "allowed_values": None}]
    valid, violations = validate_prediction_inputs({"age": -5}, rules)
    assert not valid
    assert len(violations) == 1
    assert "age" in violations[0]["feature_name"]
    assert "-5" in violations[0]["message"] or "0" in violations[0]["message"]


def test_validate_range_fail_above():
    from core.validator import validate_prediction_inputs

    rules = [{"feature_name": "units", "rule_type": "range", "min_val": 1.0, "max_val": 10000.0, "allowed_values": None}]
    valid, violations = validate_prediction_inputs({"units": 99999}, rules)
    assert not valid
    assert "units" in violations[0]["feature_name"]


def test_validate_range_min_only():
    from core.validator import validate_prediction_inputs

    rules = [{"feature_name": "price", "rule_type": "range", "min_val": 0.01, "max_val": None, "allowed_values": None}]
    valid, violations = validate_prediction_inputs({"price": -1.0}, rules)
    assert not valid
    valid2, _ = validate_prediction_inputs({"price": 0.01}, rules)
    assert valid2


def test_validate_range_non_numeric():
    from core.validator import validate_prediction_inputs

    rules = [{"feature_name": "age", "rule_type": "range", "min_val": 0.0, "max_val": 120.0, "allowed_values": None}]
    valid, violations = validate_prediction_inputs({"age": "not-a-number"}, rules)
    assert not valid
    assert "must be a number" in violations[0]["message"]


def test_validate_one_of_pass():
    from core.validator import validate_prediction_inputs

    rules = [{"feature_name": "region", "rule_type": "one_of", "min_val": None, "max_val": None, "allowed_values": '["East","West","North","South"]'}]
    valid, violations = validate_prediction_inputs({"region": "East"}, rules)
    assert valid


def test_validate_one_of_fail():
    from core.validator import validate_prediction_inputs

    rules = [{"feature_name": "region", "rule_type": "one_of", "min_val": None, "max_val": None, "allowed_values": '["East","West","North","South"]'}]
    valid, violations = validate_prediction_inputs({"region": "Unknown"}, rules)
    assert not valid
    assert "must be one of" in violations[0]["message"]


def test_validate_not_null_pass():
    from core.validator import validate_prediction_inputs

    rules = [{"feature_name": "customer_id", "rule_type": "not_null", "min_val": None, "max_val": None, "allowed_values": None}]
    valid, violations = validate_prediction_inputs({"customer_id": "abc123"}, rules)
    assert valid


def test_validate_not_null_fail():
    from core.validator import validate_prediction_inputs

    rules = [{"feature_name": "customer_id", "rule_type": "not_null", "min_val": None, "max_val": None, "allowed_values": None}]
    valid, violations = validate_prediction_inputs({"customer_id": None}, rules)
    assert not valid
    assert "required" in violations[0]["message"].lower() or "not provided" in violations[0]["message"].lower()


def test_validate_multiple_rules_all_pass():
    from core.validator import validate_prediction_inputs

    rules = [
        {"feature_name": "age", "rule_type": "range", "min_val": 0.0, "max_val": 120.0, "allowed_values": None},
        {"feature_name": "region", "rule_type": "one_of", "min_val": None, "max_val": None, "allowed_values": '["East","West"]'},
    ]
    valid, violations = validate_prediction_inputs({"age": 30, "region": "East"}, rules)
    assert valid


def test_validate_multiple_rules_both_fail():
    from core.validator import validate_prediction_inputs

    rules = [
        {"feature_name": "age", "rule_type": "range", "min_val": 0.0, "max_val": 120.0, "allowed_values": None},
        {"feature_name": "region", "rule_type": "one_of", "min_val": None, "max_val": None, "allowed_values": '["East","West"]'},
    ]
    valid, violations = validate_prediction_inputs({"age": 999, "region": "Nowhere"}, rules)
    assert not valid
    assert len(violations) == 2


def test_validate_empty_rules():
    from core.validator import validate_prediction_inputs

    valid, violations = validate_prediction_inputs({"age": -99}, [])
    assert valid
    assert violations == []


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_pattern_add_validation_rule():
    from api.chat import _INPUT_VALIDATION_PATTERNS

    assert _INPUT_VALIDATION_PATTERNS.search("add a validation rule for age")


def test_pattern_validate_between():
    from api.chat import _INPUT_VALIDATION_PATTERNS

    assert _INPUT_VALIDATION_PATTERNS.search("validate that age is between 0 and 120")


def test_pattern_require_one_of():
    from api.chat import _INPUT_VALIDATION_PATTERNS

    assert _INPUT_VALIDATION_PATTERNS.search("require region to be one of East, West, South")


def test_pattern_reject_negative():
    from api.chat import _INPUT_VALIDATION_PATTERNS

    assert _INPUT_VALIDATION_PATTERNS.search("reject predictions where units is negative")


def test_pattern_list_rules():
    from api.chat import _INPUT_VALIDATION_PATTERNS

    assert _INPUT_VALIDATION_PATTERNS.search("show my validation rules")
    assert _INPUT_VALIDATION_PATTERNS.search("list my input validation rules")


def test_pattern_delete_rules():
    from api.chat import _INPUT_VALIDATION_PATTERNS

    assert _INPUT_VALIDATION_PATTERNS.search("remove the validation rules")
    assert _INPUT_VALIDATION_PATTERNS.search("delete validation rule")


def test_pattern_false_positive_rejection():
    from api.chat import _INPUT_VALIDATION_PATTERNS

    assert not _INPUT_VALIDATION_PATTERNS.search("show me a bar chart of revenue")
    assert not _INPUT_VALIDATION_PATTERNS.search("what is the average sales?")
    assert not _INPUT_VALIDATION_PATTERNS.search("train a model")


def test_range_regex_extraction():
    from api.chat import _IV_RANGE_RE

    m = _IV_RANGE_RE.search("validate that age is between 0 and 120")
    assert m is not None
    assert m.group("feature") == "age"
    assert float(m.group("lo")) == 0.0
    assert float(m.group("hi")) == 120.0


def test_range_regex_negative_lower():
    from api.chat import _IV_RANGE_RE

    m = _IV_RANGE_RE.search("units must be between -100 and 10000")
    assert m is not None
    assert float(m.group("lo")) == -100.0


def test_bound_regex_above():
    from api.chat import _IV_BOUND_RE

    m = _IV_BOUND_RE.search("price must be above 0")
    assert m is not None
    assert m.group("feature") == "price"
    assert m.group("op").lower() == "above"
    assert float(m.group("val")) == 0.0


def test_one_of_regex_extraction():
    from api.chat import _IV_ONE_OF_RE

    m = _IV_ONE_OF_RE.search("region must be one of East, West, North")
    assert m is not None
    assert m.group("feature") == "region"
    vals = [v.strip() for v in m.group("vals").split(",")]
    assert "East" in vals
    assert "West" in vals


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "engine", engine)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture()
def client():
    from main import app

    return TestClient(app)


@pytest.fixture()
def deployed_project(client, tmp_path, monkeypatch):
    """Creates project → uploads CSV → trains regression model → deploys."""
    import api.data as data_mod
    import api.deploy as deploy_mod
    import api.models as models_mod

    monkeypatch.setattr(data_mod, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(deploy_mod, "DEPLOY_DIR", tmp_path / "deployments")
    monkeypatch.setattr(models_mod, "MODELS_DIR", tmp_path / "models")

    proj = client.post("/api/projects", json={"name": "IV Test"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": "revenue"})

    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    run_id = train.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert run["status"] == "done"

    deploy_resp = client.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code == 201
    deployment_id = deploy_resp.json()["id"]

    return {"project_id": project_id, "dataset_id": dataset_id, "run_id": run_id, "deployment_id": deployment_id}


def test_create_range_rule(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    resp = client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "units", "rule_type": "range", "min_val": 1.0, "max_val": 10000.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["feature_name"] == "units"
    assert data["rule_type"] == "range"
    assert data["min_val"] == 1.0
    assert data["max_val"] == 10000.0
    assert "id" in data


def test_create_one_of_rule(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    resp = client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "region", "rule_type": "one_of", "allowed_values": ["East", "West", "North", "South"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["rule_type"] == "one_of"
    assert data["allowed_values"] == ["East", "West", "North", "South"]


def test_create_not_null_rule(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    resp = client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "product", "rule_type": "not_null"},
    )
    assert resp.status_code == 200
    assert resp.json()["rule_type"] == "not_null"


def test_create_rule_invalid_type(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    resp = client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "age", "rule_type": "bad_type"},
    )
    assert resp.status_code == 422


def test_create_range_rule_missing_bounds(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    resp = client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "age", "rule_type": "range"},
    )
    assert resp.status_code == 422


def test_create_one_of_rule_no_values(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    resp = client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "region", "rule_type": "one_of", "allowed_values": []},
    )
    assert resp.status_code == 422


def test_create_rule_nonexistent_deployment(client):
    resp = client.post(
        "/api/deploy/nonexistent-id/input-validation-rules",
        json={"feature_name": "age", "rule_type": "not_null"},
    )
    assert resp.status_code == 404


def test_list_rules_empty(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    resp = client.get(f"/api/deploy/{dep_id}/input-validation-rules")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["rules"] == []


def test_list_rules_after_creation(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "units", "rule_type": "range", "min_val": 1.0, "max_val": 1000.0},
    )
    client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "region", "rule_type": "not_null"},
    )
    resp = client.get(f"/api/deploy/{dep_id}/input-validation-rules")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    feature_names = {r["feature_name"] for r in data["rules"]}
    assert feature_names == {"units", "region"}


def test_delete_rule(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    create_resp = client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "price", "rule_type": "not_null"},
    )
    rule_id = create_resp.json()["id"]

    del_resp = client.delete(f"/api/deploy/{dep_id}/input-validation-rules/{rule_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    list_resp = client.get(f"/api/deploy/{dep_id}/input-validation-rules")
    assert list_resp.json()["count"] == 0


def test_delete_nonexistent_rule(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    resp = client.delete(f"/api/deploy/{dep_id}/input-validation-rules/no-such-rule")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# make_prediction() — validation enforcement
# ---------------------------------------------------------------------------


def test_prediction_passes_when_no_rules(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    resp = client.post(
        f"/api/predict/{dep_id}",
        json={"units": 10, "product": "Widget A", "region": "North"},
    )
    assert resp.status_code == 200


def test_prediction_blocked_by_range_rule(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "units", "rule_type": "range", "min_val": 1.0, "max_val": 100.0},
    )
    resp = client.post(
        f"/api/predict/{dep_id}",
        json={"units": -50, "product": "Widget A", "region": "North"},
    )
    assert resp.status_code == 422
    assert "Input validation failed" in resp.json()["detail"]
    assert "units" in resp.json()["detail"]


def test_prediction_blocked_by_one_of_rule(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "region", "rule_type": "one_of", "allowed_values": ["East", "West", "North", "South"]},
    )
    resp = client.post(
        f"/api/predict/{dep_id}",
        json={"units": 10, "product": "Widget A", "region": "Narnia"},
    )
    assert resp.status_code == 422
    assert "region" in resp.json()["detail"].lower()


def test_prediction_passes_with_valid_inputs(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "units", "rule_type": "range", "min_val": 1.0, "max_val": 10000.0},
    )
    resp = client.post(
        f"/api/predict/{dep_id}",
        json={"units": 15, "product": "Widget B", "region": "South"},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Chat handler integration — creates rule from NL message
# ---------------------------------------------------------------------------


def _chat_events(client, project_id: str, message: str) -> list[dict]:
    with patch("anthropic.Anthropic") as mock_cls:
        mock_c = MagicMock()
        mock_cls.return_value = mock_c
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["I've added the validation rule."])
        mock_c.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": message},
        )

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def test_chat_creates_range_rule(client, deployed_project):
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "validate that units is between 1 and 10000")
    iv_events = [e for e in events if e.get("type") == "input_validation_rule"]
    assert len(iv_events) == 1
    ev = iv_events[0]["input_validation_rule"]
    assert ev["action"] == "created"
    assert ev["rule_type"] == "range"
    assert ev["min_val"] == 1.0
    assert ev["max_val"] == 10000.0


def test_chat_lists_rules(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "units", "rule_type": "not_null"},
    )
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "show my validation rules")
    iv_events = [e for e in events if e.get("type") == "input_validation_rule"]
    assert len(iv_events) == 1
    ev = iv_events[0]["input_validation_rule"]
    assert ev["action"] == "list"
    assert ev["count"] >= 1


def test_chat_deletes_rules(client, deployed_project):
    dep_id = deployed_project["deployment_id"]
    client.post(
        f"/api/deploy/{dep_id}/input-validation-rules",
        json={"feature_name": "units", "rule_type": "not_null"},
    )
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "remove the validation rules")
    iv_events = [e for e in events if e.get("type") == "input_validation_rule"]
    assert len(iv_events) == 1
    ev = iv_events[0]["input_validation_rule"]
    assert ev["action"] == "deleted"
    assert ev["deleted_count"] >= 1
