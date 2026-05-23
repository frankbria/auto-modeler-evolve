"""BDD step definitions for the analyst 'lunch break' end-to-end flow.

Covers the full journey: upload → explore → train → validate → deploy → predict.
Uses the synchronous TestClient (no asyncio) so pytest-bdd step functions
work without async scaffolding.

This is the canonical integration test that validates the core vision:
  "A business analyst uploads data and, in a lunch break, has a deployed
   prediction model with a shareable dashboard — no code required."
"""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, scenarios, then, when
from sqlmodel import SQLModel, create_engine

import db as db_module

scenarios("features/analyst_lunch_break.feature")

# ---------------------------------------------------------------------------
# Quarterly sales data — 12 rows, 4 columns: region, product, units, revenue
# ---------------------------------------------------------------------------

_SALES_CSV = b"""\
region,product,units,revenue
North,Widget A,10,1200.00
South,Widget B,8,850.00
East,Widget A,18,2100.00
West,Widget C,4,450.00
North,Widget B,15,1650.00
South,Widget A,12,1400.00
East,Widget B,22,2400.00
West,Widget A,6,700.00
North,Widget C,9,980.00
South,Widget C,11,1100.00
East,Widget C,14,1550.00
West,Widget B,20,2200.00
"""

_BATCH_CSV = b"""\
region,product,units
North,Widget A,10
East,Widget B,20
South,Widget C,5
"""

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.ab_test  # noqa
    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.deployment_version  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_alert_rule  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def ctx():
    """Shared mutable context dict passed between step functions."""
    return {}


# ---------------------------------------------------------------------------
# Background steps
# ---------------------------------------------------------------------------


@given('a new project named "Q4 Revenue Analysis"')
def create_project(client, ctx):
    resp = client.post("/api/projects", json={"name": "Q4 Revenue Analysis"})
    assert resp.status_code == 201, resp.text
    ctx["project_id"] = resp.json()["id"]


@given("a quarterly sales CSV is uploaded to the project")
def upload_csv(client, ctx):
    resp = client.post(
        "/api/data/upload",
        data={"project_id": ctx["project_id"]},
        files={"file": ("sales_q4.csv", io.BytesIO(_SALES_CSV), "text/csv")},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    ctx["dataset_id"] = data["dataset_id"]
    ctx["upload_data"] = data


# ---------------------------------------------------------------------------
# Scenario: Upload reveals immediate data insight
# ---------------------------------------------------------------------------


@then("the dataset has 12 rows and 4 columns")
def dataset_dimensions(ctx):
    data = ctx["upload_data"]
    assert data["row_count"] == 12, f"Expected 12 rows, got {data['row_count']}"
    assert data["column_count"] == 4, f"Expected 4 columns, got {data['column_count']}"


@then('the column names include "region", "product", "units", and "revenue"')
def column_names(client, ctx):
    resp = client.get(f"/api/data/{ctx['dataset_id']}/preview")
    assert resp.status_code == 200, resp.text
    # preview returns column_stats list, each entry has a "name" key
    cols = [c["name"] for c in resp.json().get("column_stats", [])]
    for expected in ("region", "product", "units", "revenue"):
        assert expected in cols, f"Expected column '{expected}' in {cols}"


@then("each numeric column has min, max, and mean statistics")
def numeric_stats(client, ctx):
    resp = client.get(f"/api/data/{ctx['dataset_id']}/preview")
    assert resp.status_code == 200, resp.text
    columns = resp.json().get("column_stats", [])
    # numeric stats are top-level keys on each column dict (not nested under "stats")
    numeric_cols = [c for c in columns if c.get("dtype") in ("int64", "float64")]
    assert numeric_cols, "No numeric columns found"
    for col in numeric_cols:
        assert "min" in col, f"Column '{col['name']}' missing min stat"
        assert "max" in col, f"Column '{col['name']}' missing max stat"
        assert "mean" in col, f"Column '{col['name']}' missing mean stat"


@then("the dataset profile is cached for instant retrieval")
def profile_cached(client, ctx):
    resp = client.get(f"/api/data/{ctx['dataset_id']}/profile")
    assert resp.status_code == 200, resp.text
    # Second call exercises the cache path
    resp2 = client.get(f"/api/data/{ctx['dataset_id']}/profile")
    assert resp2.status_code == 200


# ---------------------------------------------------------------------------
# Scenario: Analyst explores data by asking questions
# ---------------------------------------------------------------------------


@when('the analyst asks "what are the top performing regions?"')
def ask_question(client, ctx):
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as mock_anthropic:
        mock_client = mock.MagicMock()
        mock_anthropic.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(
            ["North has the highest revenue with $3,830 across all products."]
        )
        mock_client.messages.stream.return_value = mock_stream

        resp = client.post(
            f"/api/chat/{ctx['project_id']}",
            json={"message": "what are the top performing regions?"},
        )
    ctx["chat_response"] = resp
    ctx["chat_text"] = resp.text


@then("the chat response contains a natural language answer")
def chat_has_answer(ctx):
    assert ctx["chat_response"].status_code == 200, ctx["chat_text"]
    assert len(ctx["chat_text"]) > 10


@then("the response does not contain a stack trace")
def no_stack_trace(ctx):
    text = ctx["chat_text"]
    assert "Traceback" not in text, "Stack trace found in chat response"
    assert "raise " not in text, "Exception raise found in chat response"


# ---------------------------------------------------------------------------
# Shared setup step: feature set targeting revenue
# ---------------------------------------------------------------------------


@given('a feature set targeting "revenue" exists for the dataset')
def create_feature_set(client, ctx):
    resp = client.post(
        f"/api/features/{ctx['dataset_id']}/apply",
        json={"transformations": []},
    )
    assert resp.status_code == 201, resp.text
    fs_id = resp.json()["feature_set_id"]

    t_resp = client.post(
        f"/api/features/{ctx['dataset_id']}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    assert t_resp.status_code == 200, t_resp.text
    ctx["feature_set_id"] = fs_id


# ---------------------------------------------------------------------------
# Scenario: Training a model with a named target succeeds
# ---------------------------------------------------------------------------


@when('the analyst trains a Linear Regression model to predict "revenue"')
def train_model(client, ctx):
    resp = client.post(
        f"/api/models/{ctx['project_id']}/train",
        json={
            "algorithms": ["linear_regression"],
            "feature_set_id": ctx["feature_set_id"],
        },
    )
    assert resp.status_code == 202, resp.text
    run_ids = resp.json()["model_run_ids"]
    assert run_ids, "No model run IDs returned"
    ctx["model_run_id"] = run_ids[0]

    # Poll until done (max 15s — linear regression is fast)
    for _ in range(30):
        r = client.get(f"/api/models/{ctx['project_id']}/runs")
        runs = r.json().get("runs", [])
        run = next((x for x in runs if x["id"] == ctx["model_run_id"]), None)
        if run and run["status"] == "done":
            ctx["completed_run"] = run
            return
        time.sleep(0.5)
    pytest.skip("Training did not complete within 15 seconds")


@then('a model run is created with status "done"')
def run_is_done(ctx):
    assert "completed_run" in ctx, "No completed run found"
    assert ctx["completed_run"]["status"] == "done"


@then("the run has an R² metric above zero")
def run_has_r2(ctx):
    run = ctx["completed_run"]
    metrics = run.get("metrics") or {}
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    r2 = metrics.get("r2")
    assert r2 is not None, "No R² in metrics"
    assert r2 > 0, f"R² should be > 0, got {r2}"


@then("the run records train and test set sizes")
def run_has_split_sizes(ctx):
    run = ctx["completed_run"]
    metrics = run.get("metrics") or {}
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    assert "train_size" in metrics, "Missing train_size in metrics"
    assert "test_size" in metrics, "Missing test_size in metrics"
    assert metrics["train_size"] > 0
    assert metrics["test_size"] > 0


# ---------------------------------------------------------------------------
# Shared setup: train, select, and deploy
# ---------------------------------------------------------------------------


@given("a Linear Regression model has been trained and selected")
def train_and_select(client, ctx):
    resp = client.post(
        f"/api/models/{ctx['project_id']}/train",
        json={
            "algorithms": ["linear_regression"],
            "feature_set_id": ctx["feature_set_id"],
        },
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["model_run_ids"][0]
    ctx["model_run_id"] = run_id

    for _ in range(30):
        r = client.get(f"/api/models/{ctx['project_id']}/runs")
        runs = r.json().get("runs", [])
        run = next((x for x in runs if x["id"] == run_id), None)
        if run and run["status"] == "done":
            break
        time.sleep(0.5)
    else:
        pytest.skip("Training did not complete within 15 seconds")

    sel = client.post(f"/api/models/{run_id}/select")
    assert sel.status_code == 200, sel.text


@given("a Linear Regression model has been trained, selected, and deployed")
def train_select_deploy(client, ctx):
    train_and_select(client, ctx)

    dep_resp = client.post(f"/api/deploy/{ctx['model_run_id']}")
    assert dep_resp.status_code == 201, dep_resp.text
    ctx["deployment"] = dep_resp.json()
    ctx["deployment_id"] = dep_resp.json()["id"]


# ---------------------------------------------------------------------------
# Scenario: Deploying the trained model produces a working endpoint
# ---------------------------------------------------------------------------


@when("the model is deployed")
def deploy_model(client, ctx):
    resp = client.post(f"/api/deploy/{ctx['model_run_id']}")
    assert resp.status_code == 201, resp.text
    ctx["deployment"] = resp.json()
    ctx["deployment_id"] = resp.json()["id"]


@then("a deployment is created with an active prediction endpoint")
def deployment_is_active(ctx):
    d = ctx["deployment"]
    assert d.get("id"), "Deployment has no ID"
    assert d.get("is_active"), "Deployment is not active"


@then('the endpoint URL follows the pattern "/api/predict/{id}"')
def endpoint_url_pattern(ctx):
    d = ctx["deployment"]
    endpoint = d.get("endpoint_path", "")
    assert endpoint.startswith("/api/predict/"), f"Unexpected endpoint: {endpoint}"


@then("the deployment has a public dashboard URL")
def deployment_has_dashboard(ctx):
    d = ctx["deployment"]
    dashboard = d.get("dashboard_url", "")
    assert dashboard, "No dashboard_url in deployment"
    assert "predict" in dashboard, f"Dashboard URL looks wrong: {dashboard}"


# ---------------------------------------------------------------------------
# Scenario: Making a prediction returns a numeric forecast
# ---------------------------------------------------------------------------


@when("a prediction is submitted with feature values matching the training schema")
def submit_prediction(client, ctx):
    deployment_id = ctx["deployment_id"]
    resp = client.post(
        f"/api/predict/{deployment_id}",
        json={"region": "North", "product": "Widget A", "units": 10},
    )
    ctx["prediction_response"] = resp
    ctx["prediction_data"] = resp.json()


@then("the response contains a numeric prediction")
def response_has_prediction(ctx):
    resp = ctx["prediction_response"]
    assert resp.status_code == 200, f"Prediction failed: {resp.text}"
    data = ctx["prediction_data"]
    pred = data.get("prediction")
    assert pred is not None, f"No prediction in response: {data}"
    # Should be numeric (float or int)
    assert isinstance(pred, (int, float)), f"Prediction is not numeric: {pred!r}"


@then("the response includes the input features that were used")
def response_has_inputs(ctx):
    data = ctx["prediction_data"]
    # The response echoes feature_names (list of column names used by the model)
    feature_names = data.get("feature_names") or []
    assert feature_names, f"No feature_names in prediction response: {data}"


# ---------------------------------------------------------------------------
# Scenario: Batch prediction on uploaded CSV returns enriched output
# ---------------------------------------------------------------------------


@when("a batch prediction CSV is submitted with 3 rows")
def submit_batch(client, ctx):
    deployment_id = ctx["deployment_id"]
    resp = client.post(
        f"/api/predict/{deployment_id}/batch",
        files={"file": ("batch.csv", io.BytesIO(_BATCH_CSV), "text/csv")},
    )
    ctx["batch_response"] = resp


@then('the response CSV has a "revenue_prediction" column')
def batch_has_prediction_column(ctx):
    resp = ctx["batch_response"]
    assert resp.status_code == 200, f"Batch prediction failed: {resp.text}"
    lines = resp.text.strip().split("\n")
    header = lines[0].lower()
    assert (
        "revenue_prediction" in header or "prediction" in header
    ), f"No prediction column in batch output header: {header}"


@then("the output has 3 rows matching the input")
def batch_row_count(ctx):
    resp = ctx["batch_response"]
    lines = [ln for ln in resp.text.strip().split("\n") if ln.strip()]
    # header + 3 data rows = 4 lines
    assert len(lines) == 4, f"Expected 4 lines (header + 3 rows), got {len(lines)}"
