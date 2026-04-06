"""Tests for prediction sensitivity analysis via chat.

Covers:
- _SENSITIVITY_PATTERNS detection
- _detect_sensitivity_request() helper
- run_sensitivity_analysis() pure function (regression + classification)
- Chat integration: SSE sensitivity event emitted
"""

import io
import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_pattern_sensitivity_analysis_on():
    from api.chat import _SENSITIVITY_PATTERNS

    assert _SENSITIVITY_PATTERNS.search("sensitivity analysis on price")


def test_pattern_how_sensitive():
    from api.chat import _SENSITIVITY_PATTERNS

    assert _SENSITIVITY_PATTERNS.search("how sensitive is revenue to units?")


def test_pattern_sweep_from():
    from api.chat import _SENSITIVITY_PATTERNS

    assert _SENSITIVITY_PATTERNS.search("sweep price from 10 to 100")


def test_pattern_vary_from():
    from api.chat import _SENSITIVITY_PATTERNS

    assert _SENSITIVITY_PATTERNS.search("vary units from 50 to 300")


def test_pattern_run_sensitivity():
    from api.chat import _SENSITIVITY_PATTERNS

    assert _SENSITIVITY_PATTERNS.search("run a sensitivity analysis on quantity")


def test_pattern_how_does_change_as():
    from api.chat import _SENSITIVITY_PATTERNS

    assert _SENSITIVITY_PATTERNS.search(
        "how does the prediction change as units varies?"
    )


def test_pattern_effect_of_on():
    from api.chat import _SENSITIVITY_PATTERNS

    assert _SENSITIVITY_PATTERNS.search("show me the effect of price on the prediction")


def test_pattern_negative_unrelated():
    from api.chat import _SENSITIVITY_PATTERNS

    assert not _SENSITIVITY_PATTERNS.search("what is the average revenue?")
    assert not _SENSITIVITY_PATTERNS.search("which model should I use?")


# ---------------------------------------------------------------------------
# _detect_sensitivity_request — pure helper
# ---------------------------------------------------------------------------


def test_detect_explicit_range():
    from api.chat import _detect_sensitivity_request

    feats = ["units", "price", "region"]
    means = {"units": 100.0, "price": 50.0}
    result = _detect_sensitivity_request("sweep units from 50 to 200", feats, means)
    assert result is not None
    assert result["feature"] == "units"
    assert result["min_val"] == 50.0
    assert result["max_val"] == 200.0


def test_detect_default_range_from_mean():
    from api.chat import _detect_sensitivity_request

    feats = ["units", "price"]
    means = {"units": 100.0, "price": 50.0}
    result = _detect_sensitivity_request("sensitivity analysis on units", feats, means)
    assert result is not None
    assert result["feature"] == "units"
    # Default ±50% around mean=100 → min>=0, max=150
    assert result["min_val"] >= 0
    assert result["max_val"] == 150.0


def test_detect_step_count():
    from api.chat import _detect_sensitivity_request

    feats = ["price"]
    means = {"price": 50.0}
    result = _detect_sensitivity_request(
        "vary price from 10 to 100 in 20 steps", feats, means
    )
    assert result is not None
    assert result["n_steps"] == 20


def test_detect_default_step_count():
    from api.chat import _detect_sensitivity_request

    feats = ["price"]
    means = {"price": 50.0}
    result = _detect_sensitivity_request("sensitivity analysis on price", feats, means)
    assert result is not None
    assert result["n_steps"] == 10


def test_detect_feature_by_longest_match():
    """Longer column names are preferred over shorter ones in messages."""
    from api.chat import _detect_sensitivity_request

    feats = ["units", "unit_cost"]
    means = {"units": 10.0, "unit_cost": 5.0}
    result = _detect_sensitivity_request("vary unit_cost from 1 to 20", feats, means)
    assert result is not None
    assert result["feature"] == "unit_cost"


def test_detect_no_features_returns_none():
    from api.chat import _detect_sensitivity_request

    # Empty feature list
    result = _detect_sensitivity_request("sensitivity on price", [], {})
    assert result is None


# ---------------------------------------------------------------------------
# run_sensitivity_analysis — pure function
# ---------------------------------------------------------------------------


def _make_regression_deployment(tmp_path):
    """Build a tiny LinearRegression deployment and return pipeline/model paths."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from core.deployer import build_prediction_pipeline, save_pipeline

    df = pd.DataFrame(
        {
            "units": np.arange(1, 21, dtype=float),
            "price": np.ones(20) * 10.0,
            "revenue": np.arange(1, 21, dtype=float) * 100.0,
        }
    )
    feature_names = ["units", "price"]
    target_col = "revenue"

    pipeline = build_prediction_pipeline(df, feature_names, target_col, "regression")
    pipeline_path = str(tmp_path / "pipeline.joblib")
    save_pipeline(pipeline, pipeline_path)

    X = df[feature_names].values
    y = df[target_col].values
    model = LinearRegression().fit(X, y)
    model_path = str(tmp_path / "model.joblib")
    joblib.dump(model, model_path)

    return pipeline_path, model_path, dict(pipeline.feature_means)


def test_sensitivity_regression_returns_required_fields(tmp_path):
    from core.deployer import run_sensitivity_analysis

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    result = run_sensitivity_analysis(
        pipeline_path, model_path, "units", [5.0, 10.0, 15.0, 20.0], means
    )

    assert result["feature"] == "units"
    assert result["target_column"] == "revenue"
    assert result["problem_type"] == "regression"
    assert len(result["values"]) == 4
    assert len(result["predictions"]) == 4
    assert result["min_pred"] is not None
    assert result["max_pred"] is not None
    assert result["summary"]


def test_sensitivity_regression_predictions_increase_with_units(tmp_path):
    """Revenue = units × 100 — predictions should increase monotonically."""
    from core.deployer import run_sensitivity_analysis

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    sweep = list(np.linspace(1.0, 20.0, 10))
    result = run_sensitivity_analysis(pipeline_path, model_path, "units", sweep, means)

    preds = [float(p) for p in result["predictions"]]
    assert preds[-1] > preds[0], "Predictions should increase as units increase"


def test_sensitivity_regression_change_pct_positive(tmp_path):
    from core.deployer import run_sensitivity_analysis

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    result = run_sensitivity_analysis(
        pipeline_path, model_path, "units", [1.0, 10.0, 20.0], means
    )
    assert result["change_pct"] is not None
    assert result["change_pct"] > 0


def test_sensitivity_min_max_correct(tmp_path):
    from core.deployer import run_sensitivity_analysis

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    result = run_sensitivity_analysis(
        pipeline_path, model_path, "units", [5.0, 10.0, 15.0], means
    )
    preds = [float(p) for p in result["predictions"]]
    assert result["min_pred"] == pytest.approx(min(preds), abs=0.1)
    assert result["max_pred"] == pytest.approx(max(preds), abs=0.1)


def test_sensitivity_confidences_none_for_regression(tmp_path):
    from core.deployer import run_sensitivity_analysis

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    result = run_sensitivity_analysis(
        pipeline_path, model_path, "units", [5.0, 10.0], means
    )
    assert all(c is None for c in result["confidences"])


def test_sensitivity_unknown_feature_raises(tmp_path):
    from core.deployer import run_sensitivity_analysis

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    with pytest.raises(ValueError, match="not found in model"):
        run_sensitivity_analysis(
            pipeline_path, model_path, "nonexistent", [1.0, 2.0], means
        )


def test_sensitivity_summary_string(tmp_path):
    from core.deployer import run_sensitivity_analysis

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    result = run_sensitivity_analysis(
        pipeline_path, model_path, "units", list(np.linspace(1.0, 20.0, 5)), means
    )
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 10


# ---------------------------------------------------------------------------
# Chat integration — SSE sensitivity event
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
def deployed_project(client):
    """Creates project → uploads CSV → applies features → trains model → deploys."""
    proj = client.post("/api/projects", json={"name": "SensitivityTest"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": "revenue"})

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert run["status"] == "done"

    deploy_resp = client.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code == 201

    return {"project_id": project_id, "dataset_id": dataset_id, "run_id": run_id}


def _chat_events(client, project_id: str, message: str) -> list[dict]:
    """Send a chat message under mocked Anthropic and return all SSE events."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_c = MagicMock()
        mock_cls.return_value = mock_c
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(
            ["The sensitivity analysis shows how revenue changes."]
        )
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


def test_chat_sensitivity_event_emitted(client, deployed_project):
    """Sensitivity analysis SSE event is emitted when model deployed and message matches."""
    project_id = deployed_project["project_id"]
    events = _chat_events(
        client, project_id, "sensitivity analysis on units from 5 to 20"
    )
    types = [e.get("type") for e in events]
    assert "sensitivity" in types, f"Expected sensitivity in {types}"


def test_chat_sensitivity_event_required_fields(client, deployed_project):
    """The sensitivity event payload should have all required fields."""
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "run a sensitivity analysis on units")
    sens_events = [e for e in events if e.get("type") == "sensitivity"]
    assert sens_events, "No sensitivity event found"
    payload = sens_events[0]["sensitivity"]

    assert "feature" in payload
    assert "target_column" in payload
    assert "values" in payload
    assert "predictions" in payload
    assert "summary" in payload
    assert payload["target_column"] == "revenue"


def test_chat_sensitivity_has_multiple_data_points(client, deployed_project):
    """Sensitivity sweep should produce multiple prediction points."""
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "sweep units from 5 to 20")
    sens_events = [e for e in events if e.get("type") == "sensitivity"]
    if sens_events:
        payload = sens_events[0]["sensitivity"]
        assert len(payload["values"]) >= 3
        assert len(payload["predictions"]) == len(payload["values"])
