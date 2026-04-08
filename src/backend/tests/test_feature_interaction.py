"""Tests for feature interaction analysis (2-D heatmap via chat).

Covers:
- _INTERACTION_PATTERNS detection
- _detect_interaction_request() helper
- run_feature_interaction() pure function (regression + classification)
- Chat integration: SSE interaction event emitted
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


def test_pattern_interaction_between():
    from api.chat import _INTERACTION_PATTERNS

    assert _INTERACTION_PATTERNS.search("interaction between units and price")


def test_pattern_how_do_interact():
    from api.chat import _INTERACTION_PATTERNS

    assert _INTERACTION_PATTERNS.search("how do units and region interact?")


def test_pattern_joint_effect():
    from api.chat import _INTERACTION_PATTERNS

    assert _INTERACTION_PATTERNS.search("joint effect of units and region")


def test_pattern_2d_sensitivity():
    from api.chat import _INTERACTION_PATTERNS

    assert _INTERACTION_PATTERNS.search("2d sensitivity analysis")


def test_pattern_feature_interaction_plot():
    from api.chat import _INTERACTION_PATTERNS

    assert _INTERACTION_PATTERNS.search("show me a feature interaction plot")


def test_pattern_how_do_affect_prediction():
    from api.chat import _INTERACTION_PATTERNS

    assert _INTERACTION_PATTERNS.search(
        "how do units and price affect the prediction?"
    )


def test_pattern_combined_effect():
    from api.chat import _INTERACTION_PATTERNS

    assert _INTERACTION_PATTERNS.search("combined effect of region and category")


def test_pattern_interaction_heatmap():
    from api.chat import _INTERACTION_PATTERNS

    assert _INTERACTION_PATTERNS.search("show me an interaction heatmap")


def test_pattern_negative_unrelated():
    from api.chat import _INTERACTION_PATTERNS

    assert not _INTERACTION_PATTERNS.search("what is the average revenue?")
    assert not _INTERACTION_PATTERNS.search("how sensitive is revenue to units?")


# ---------------------------------------------------------------------------
# _detect_interaction_request — pure helper
# ---------------------------------------------------------------------------


def test_detect_two_features():
    from api.chat import _detect_interaction_request

    feats = ["units", "price", "region"]
    result = _detect_interaction_request("how do units and price interact?", feats)
    assert result is not None
    assert set(result.values()) == {"units", "price"}


def test_detect_longest_match_first():
    """Longer column names should be preferred over shorter overlapping names."""
    from api.chat import _detect_interaction_request

    feats = ["units", "unit_cost", "region"]
    result = _detect_interaction_request(
        "interaction between unit_cost and region", feats
    )
    assert result is not None
    assert "unit_cost" in result.values()
    assert "region" in result.values()


def test_detect_underscore_in_message():
    from api.chat import _detect_interaction_request

    feats = ["product_category", "sales_region"]
    result = _detect_interaction_request(
        "interaction between product_category and sales_region", feats
    )
    assert result is not None
    assert "product_category" in result.values()
    assert "sales_region" in result.values()


def test_detect_one_feature_returns_none():
    from api.chat import _detect_interaction_request

    feats = ["units", "price"]
    result = _detect_interaction_request("how does units interact?", feats)
    # Only one feature found → None
    assert result is None


def test_detect_empty_features_returns_none():
    from api.chat import _detect_interaction_request

    result = _detect_interaction_request("interaction between units and price", [])
    assert result is None


# ---------------------------------------------------------------------------
# run_feature_interaction — pure function tests
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
            "price": np.arange(10, 30, dtype=float),
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


def _make_classification_deployment(tmp_path):
    """Build a tiny classification deployment with a categorical feature."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from core.deployer import build_prediction_pipeline, save_pipeline

    df = pd.DataFrame(
        {
            "units": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            "region": ["North", "South"] * 10,
            "churn": ["no"] * 10 + ["yes"] * 10,
        }
    )
    feature_names = ["units", "region"]
    target_col = "churn"

    pipeline = build_prediction_pipeline(df, feature_names, target_col, "classification")
    pipeline_path = str(tmp_path / "pipeline_cls.joblib")
    save_pipeline(pipeline, pipeline_path)

    X = pipeline.transform_df(df)
    le = pipeline.target_encoder
    y = le.transform(df[target_col].values)
    model = LogisticRegression(max_iter=200).fit(X, y)
    model_path = str(tmp_path / "model_cls.joblib")
    joblib.dump(model, model_path)

    return pipeline_path, model_path, dict(pipeline.feature_means)


def test_interaction_regression_required_fields(tmp_path):
    from core.deployer import run_feature_interaction

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    result = run_feature_interaction(pipeline_path, model_path, "units", "price", means)

    assert result["feature1"] == "units"
    assert result["feature2"] == "price"
    assert result["target_column"] == "revenue"
    assert result["problem_type"] == "regression"
    assert isinstance(result["row_labels"], list)
    assert isinstance(result["col_labels"], list)
    assert isinstance(result["values"], list)
    assert result["summary"]


def test_interaction_regression_grid_dimensions(tmp_path):
    from core.deployer import run_feature_interaction

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    result = run_feature_interaction(
        pipeline_path, model_path, "units", "price", means, n_steps=5
    )

    assert len(result["row_labels"]) == 5
    assert len(result["col_labels"]) == 5
    assert len(result["values"]) == 5
    assert all(len(row) == 5 for row in result["values"])


def test_interaction_regression_min_max_present(tmp_path):
    from core.deployer import run_feature_interaction

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    result = run_feature_interaction(pipeline_path, model_path, "units", "price", means)

    assert result["min_val"] is not None
    assert result["max_val"] is not None
    assert result["min_val"] <= result["max_val"]


def test_interaction_regression_min_max_correct(tmp_path):
    from core.deployer import run_feature_interaction

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    result = run_feature_interaction(pipeline_path, model_path, "units", "price", means)

    flat = [v for row in result["values"] for v in row]
    numeric_flat = [float(v) for v in flat]
    assert abs(min(numeric_flat) - result["min_val"]) < 0.01
    assert abs(max(numeric_flat) - result["max_val"]) < 0.01


def test_interaction_regression_summary_mentions_features(tmp_path):
    from core.deployer import run_feature_interaction

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    result = run_feature_interaction(pipeline_path, model_path, "units", "price", means)

    assert "units" in result["summary"] or "price" in result["summary"]
    assert "revenue" in result["summary"]


def test_interaction_unknown_feature_raises(tmp_path):
    from core.deployer import run_feature_interaction

    pipeline_path, model_path, means = _make_regression_deployment(tmp_path)
    with pytest.raises(ValueError, match="not found in model"):
        run_feature_interaction(
            pipeline_path, model_path, "units", "nonexistent", means
        )


def test_interaction_classification_returns_class_labels(tmp_path):
    from core.deployer import run_feature_interaction

    pipeline_path, model_path, means = _make_classification_deployment(tmp_path)
    result = run_feature_interaction(pipeline_path, model_path, "units", "region", means)

    assert result["problem_type"] == "classification"
    # Values should be string class labels
    for row in result["values"]:
        for cell in row:
            assert isinstance(cell, str), f"Expected str class label, got {type(cell)}: {cell}"


def test_interaction_classification_min_max_none(tmp_path):
    from core.deployer import run_feature_interaction

    pipeline_path, model_path, means = _make_classification_deployment(tmp_path)
    result = run_feature_interaction(pipeline_path, model_path, "units", "region", means)

    # No numeric min/max for classification
    assert result["min_val"] is None
    assert result["max_val"] is None


# ---------------------------------------------------------------------------
# Chat integration: SSE event emitted
# ---------------------------------------------------------------------------

SAMPLE_CSV = b"""units,price,revenue
10,50,1200.5
8,40,850.0
18,90,2100.75
4,20,450.25
15,75,1650.0
9,45,980.0
11,55,1100.25
16,80,1750.0
20,100,2300.5
6,30,620.75
12,60,1300.0
9,45,950.0
20,100,2200.0
5,25,520.0
16,80,1700.0
10,50,1050.0
12,60,1150.0
17,85,1800.0
21,105,2350.0
7,35,670.0
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
    proj = client.post("/api/projects", json={"name": "InteractionTest"})
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
        mock_stream.text_stream = iter(["The interaction analysis is shown."])
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


def test_interaction_sse_event_emitted(client, deployed_project):
    """Sending an interaction message emits the interaction SSE event."""
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "interaction between units and price")
    types = [e.get("type") for e in events]
    assert "interaction" in types, f"Expected interaction in {types}"


def test_interaction_event_has_required_fields(client, deployed_project):
    """The emitted interaction event contains all required fields."""
    project_id = deployed_project["project_id"]
    events = _chat_events(
        client, project_id, "show feature interaction between units and price"
    )
    ia_events = [e for e in events if e.get("type") == "interaction"]
    assert ia_events, "No interaction event found"
    ia = ia_events[0]["interaction"]
    for field in (
        "feature1",
        "feature2",
        "target_column",
        "problem_type",
        "row_labels",
        "col_labels",
        "values",
        "summary",
    ):
        assert field in ia, f"Missing field: {field}"
    assert ia["target_column"] == "revenue"


def test_interaction_not_emitted_without_deployment(client):
    """No interaction event if project has no active deployment."""
    proj = client.post("/api/projects", json={"name": "NoDeploy"})
    project_id = proj.json()["id"]

    events = _chat_events(client, project_id, "interaction between units and price")
    ia_events = [e for e in events if e.get("type") == "interaction"]
    assert len(ia_events) == 0
