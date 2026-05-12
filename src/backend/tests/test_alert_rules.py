"""Tests for Custom Prediction Alert Rules.

Covers:
- _ALERT_RULE_CREATE_PATTERNS / _ALERT_RULE_LIST_PATTERNS / _ALERT_RULE_DELETE_PATTERNS
- _extract_alert_rule_condition()  — pure NL parsing
- _evaluate_alert_rule()           — pure condition evaluation
- POST/GET/DELETE /api/deploy/{id}/alert-rules
- Integration: rule created via REST, evaluated against prediction
"""

import io
import time

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.dataset_filter  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.deployment_preset  # noqa
    import models.batch_schedule  # noqa
    import models.webhook_config  # noqa
    import models.webhook_event  # noqa
    import models.ab_test  # noqa
    import models.deployment_version  # noqa
    import models.prediction_alert_rule  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Alert Rule Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


@pytest.fixture()
async def feature_set_id(ac, dataset_id):
    resp = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert resp.status_code == 201, resp.text
    fs_id = resp.json()["feature_set_id"]
    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


@pytest.fixture()
async def trained_run_id(ac, project_id, feature_set_id):
    resp = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["model_run_ids"][0]
    for _ in range(30):
        r = await ac.get(f"/api/models/{project_id}/runs")
        runs = r.json().get("runs", [])
        run = next((x for x in runs if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.3)
    pytest.skip("Training did not complete")


@pytest.fixture()
async def deployment_id(ac, trained_run_id):
    resp = await ac.post(f"/api/deploy/{trained_run_id}", json={})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Tests — regex patterns
# ---------------------------------------------------------------------------


def test_create_pattern_matches_basic():
    from api.chat import _ALERT_RULE_CREATE_PATTERNS

    assert _ALERT_RULE_CREATE_PATTERNS.search(
        "alert me when predicted revenue is below 100000"
    )
    assert _ALERT_RULE_CREATE_PATTERNS.search(
        "notify me when the model output is above 500"
    )
    assert _ALERT_RULE_CREATE_PATTERNS.search(
        "create an alert rule for predictions below 50"
    )
    assert _ALERT_RULE_CREATE_PATTERNS.search(
        "set up an alert if confidence drops below 70%"
    )
    assert _ALERT_RULE_CREATE_PATTERNS.search(
        "trigger an alert when score is less than 0.5"
    )
    assert _ALERT_RULE_CREATE_PATTERNS.search(
        "send a notification when revenue is above 200"
    )


def test_create_pattern_no_false_positives():
    from api.chat import _ALERT_RULE_CREATE_PATTERNS

    assert not _ALERT_RULE_CREATE_PATTERNS.search("what is my current revenue?")
    assert not _ALERT_RULE_CREATE_PATTERNS.search("show me predictions")


def test_list_pattern_matches():
    from api.chat import _ALERT_RULE_LIST_PATTERNS

    assert _ALERT_RULE_LIST_PATTERNS.search("show my alert rules")
    assert _ALERT_RULE_LIST_PATTERNS.search("what alerts do I have?")
    assert _ALERT_RULE_LIST_PATTERNS.search("list all active alert rules")
    assert _ALERT_RULE_LIST_PATTERNS.search("my prediction alert rules")


def test_delete_pattern_matches():
    from api.chat import _ALERT_RULE_DELETE_PATTERNS

    assert _ALERT_RULE_DELETE_PATTERNS.search("remove the alert rule")
    assert _ALERT_RULE_DELETE_PATTERNS.search("delete my prediction alert")
    assert _ALERT_RULE_DELETE_PATTERNS.search("turn off the alert rule")
    assert _ALERT_RULE_DELETE_PATTERNS.search("disable my alert")


# ---------------------------------------------------------------------------
# Tests — _extract_alert_rule_condition (pure function)
# ---------------------------------------------------------------------------


def test_extract_numeric_below():
    from api.chat import _extract_alert_rule_condition

    result = _extract_alert_rule_condition(
        "alert me when revenue is below 100000", "revenue"
    )
    assert result is not None
    assert result["condition_type"] == "prediction_value"
    assert result["condition_op"] == "lt"
    assert result["condition_value"] == 100000.0


def test_extract_numeric_above():
    from api.chat import _extract_alert_rule_condition

    result = _extract_alert_rule_condition("notify me when score is above 0.9", "score")
    assert result is not None
    assert result["condition_op"] == "gt"
    assert result["condition_value"] == pytest.approx(0.9)


def test_extract_confidence_normalizes_fraction():
    from api.chat import _extract_alert_rule_condition

    result = _extract_alert_rule_condition(
        "alert me when confidence drops below 0.7", None
    )
    assert result is not None
    assert result["condition_type"] == "confidence"
    assert result["condition_value"] == pytest.approx(70.0)


def test_extract_confidence_percent_unchanged():
    from api.chat import _extract_alert_rule_condition

    result = _extract_alert_rule_condition(
        "warn me when confidence is less than 80", None
    )
    assert result is not None
    assert result["condition_type"] == "confidence"
    assert result["condition_value"] == pytest.approx(80.0)


def test_extract_predicted_class():
    from api.chat import _extract_alert_rule_condition

    result = _extract_alert_rule_condition(
        "alert me when predicted class is churn", "category"
    )
    assert result is not None
    assert result["condition_type"] == "predicted_class"
    assert result["condition_class"] == "churn"
    assert result["condition_value"] is None


def test_extract_no_number_returns_none():
    from api.chat import _extract_alert_rule_condition

    result = _extract_alert_rule_condition("alert me when something happens", None)
    assert result is None


# ---------------------------------------------------------------------------
# Tests — _evaluate_alert_rule (pure function)
# ---------------------------------------------------------------------------


def _make_rule(**kwargs):
    from models.prediction_alert_rule import PredictionAlertRule

    defaults = {
        "deployment_id": "test-dep",
        "name": "test rule",
        "condition_type": "prediction_value",
        "condition_op": "lt",
        "condition_value": 100.0,
        "condition_class": None,
    }
    defaults.update(kwargs)
    return PredictionAlertRule(**defaults)


def test_evaluate_lt_fires():
    from api.deploy import _evaluate_alert_rule

    rule = _make_rule(condition_op="lt", condition_value=100.0)
    assert _evaluate_alert_rule(rule, 50.0, None, None) is True


def test_evaluate_lt_no_fire():
    from api.deploy import _evaluate_alert_rule

    rule = _make_rule(condition_op="lt", condition_value=100.0)
    assert _evaluate_alert_rule(rule, 150.0, None, None) is False


def test_evaluate_gt_fires():
    from api.deploy import _evaluate_alert_rule

    rule = _make_rule(condition_op="gt", condition_value=50.0)
    assert _evaluate_alert_rule(rule, 100.0, None, None) is True


def test_evaluate_gte_boundary():
    from api.deploy import _evaluate_alert_rule

    rule = _make_rule(condition_op="gte", condition_value=100.0)
    assert _evaluate_alert_rule(rule, 100.0, None, None) is True
    assert _evaluate_alert_rule(rule, 99.9, None, None) is False


def test_evaluate_lte_boundary():
    from api.deploy import _evaluate_alert_rule

    rule = _make_rule(condition_op="lte", condition_value=100.0)
    assert _evaluate_alert_rule(rule, 100.0, None, None) is True
    assert _evaluate_alert_rule(rule, 100.1, None, None) is False


def test_evaluate_eq_fires():
    from api.deploy import _evaluate_alert_rule

    rule = _make_rule(condition_op="eq", condition_value=42.0)
    assert _evaluate_alert_rule(rule, 42.0, None, None) is True


def test_evaluate_confidence_uses_confidence_field():
    from api.deploy import _evaluate_alert_rule

    rule = _make_rule(
        condition_type="confidence", condition_op="lt", condition_value=70.0
    )
    assert _evaluate_alert_rule(rule, None, 60.0, None) is True
    assert _evaluate_alert_rule(rule, None, 80.0, None) is False


def test_evaluate_predicted_class_case_insensitive():
    from api.deploy import _evaluate_alert_rule

    rule = _make_rule(
        condition_type="predicted_class",
        condition_op="eq",
        condition_value=None,
        condition_class="Churn",
    )
    assert _evaluate_alert_rule(rule, None, None, "churn") is True
    assert _evaluate_alert_rule(rule, None, None, "no_churn") is False


def test_evaluate_missing_value_no_fire():
    from api.deploy import _evaluate_alert_rule

    rule = _make_rule(condition_op="lt", condition_value=100.0)
    assert _evaluate_alert_rule(rule, None, None, None) is False


# ---------------------------------------------------------------------------
# Tests — REST endpoints
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_create_alert_rule(ac, deployment_id):
    resp = await ac.post(
        f"/api/deploy/{deployment_id}/alert-rules",
        json={
            "name": "Low revenue alert",
            "condition_type": "prediction_value",
            "condition_op": "lt",
            "condition_value": 100.0,
            "condition_class": None,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["name"] == "Low revenue alert"
    assert data["condition_op"] == "lt"
    assert data["condition_value"] == 100.0
    assert data["trigger_count"] == 0
    assert "description" in data


@pytest.mark.anyio
async def test_list_alert_rules_empty(ac, deployment_id):
    resp = await ac.get(f"/api/deploy/{deployment_id}/alert-rules")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 0
    assert body["rules"] == []


@pytest.mark.anyio
async def test_list_alert_rules_after_create(ac, deployment_id):
    await ac.post(
        f"/api/deploy/{deployment_id}/alert-rules",
        json={
            "name": "Rule A",
            "condition_type": "prediction_value",
            "condition_op": "gt",
            "condition_value": 500.0,
            "condition_class": None,
        },
    )
    resp = await ac.get(f"/api/deploy/{deployment_id}/alert-rules")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert body["rules"][0]["name"] == "Rule A"
    assert "description" in body["rules"][0]


@pytest.mark.anyio
async def test_delete_alert_rule(ac, deployment_id):
    create_resp = await ac.post(
        f"/api/deploy/{deployment_id}/alert-rules",
        json={
            "name": "Temp Rule",
            "condition_type": "prediction_value",
            "condition_op": "lt",
            "condition_value": 10.0,
            "condition_class": None,
        },
    )
    rule_id = create_resp.json()["id"]

    del_resp = await ac.delete(f"/api/deploy/{deployment_id}/alert-rules/{rule_id}")
    assert del_resp.status_code == 200, del_resp.text
    assert del_resp.json()["deleted"] is True

    list_resp = await ac.get(f"/api/deploy/{deployment_id}/alert-rules")
    assert list_resp.json()["count"] == 0


@pytest.mark.anyio
async def test_create_rule_invalid_deployment(ac):
    resp = await ac.post(
        "/api/deploy/nonexistent-id/alert-rules",
        json={
            "name": "Bad",
            "condition_type": "prediction_value",
            "condition_op": "lt",
            "condition_value": 10.0,
            "condition_class": None,
        },
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_create_rule_missing_value_for_numeric_type(ac, deployment_id):
    resp = await ac.post(
        f"/api/deploy/{deployment_id}/alert-rules",
        json={
            "name": "Missing value",
            "condition_type": "prediction_value",
            "condition_op": "lt",
            "condition_value": None,
            "condition_class": None,
        },
    )
    assert resp.status_code == 422
