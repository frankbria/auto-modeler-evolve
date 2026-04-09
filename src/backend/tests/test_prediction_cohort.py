"""Tests for prediction cohort analysis ("who are the top predictions?").

Covers:
- _COHORT_PATTERNS detection (8 NL variants)
- compute_prediction_cohort() pure function (regression + classification)
- Chat integration: SSE prediction_cohort event emitted
"""

import io
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_pattern_who_are_top():
    from api.chat import _COHORT_PATTERNS

    assert _COHORT_PATTERNS.search("who are the top predictions?")


def test_pattern_what_do_they_have_in_common():
    from api.chat import _COHORT_PATTERNS

    assert _COHORT_PATTERNS.search("what do the top 20 predictions have in common?")


def test_pattern_profile_ranked():
    from api.chat import _COHORT_PATTERNS

    assert _COHORT_PATTERNS.search("profile the top ranked customers")


def test_pattern_characterize_at_risk():
    from api.chat import _COHORT_PATTERNS

    assert _COHORT_PATTERNS.search("characterize the at-risk records")


def test_pattern_common_traits():
    from api.chat import _COHORT_PATTERNS

    assert _COHORT_PATTERNS.search("common traits of the top predictions")


def test_pattern_tell_me_about_top():
    from api.chat import _COHORT_PATTERNS

    assert _COHORT_PATTERNS.search("tell me about the top 10 records")


def test_pattern_cohort_analysis():
    from api.chat import _COHORT_PATTERNS

    assert _COHORT_PATTERNS.search("cohort analysis of the ranked predictions")


def test_pattern_describe_at_risk():
    from api.chat import _COHORT_PATTERNS

    assert _COHORT_PATTERNS.search("describe the top predicted customers")


def test_pattern_negative_unrelated():
    from api.chat import _COHORT_PATTERNS

    assert not _COHORT_PATTERNS.search("what is the average revenue?")


# ---------------------------------------------------------------------------
# compute_prediction_cohort pure function
# ---------------------------------------------------------------------------


def _make_regression_setup():
    """Build a minimal regression pipeline, model, and multi-category dataframe."""
    from core.deployer import build_prediction_pipeline
    from sklearn.linear_model import LinearRegression

    df = pd.DataFrame(
        {
            "region": ["East", "West", "East", "North", "East", "West", "North", "East"]
            * 3,
            "units": [10, 20, 30, 40, 50, 15, 25, 35] * 3,
            "price": [5.0, 4.5, 4.0, 3.5, 3.0, 4.8, 4.2, 3.8] * 3,
            "revenue": [50, 90, 120, 140, 150, 72, 105, 133] * 3,
        }
    )
    pipeline = build_prediction_pipeline(
        df, ["units", "price"], "revenue", "regression"
    )
    X = pipeline.transform_df(df)
    y = df["revenue"].values
    model = LinearRegression().fit(X, y)
    return pipeline, model, df


def _make_classification_setup():
    """Build a minimal classification pipeline, model, and multi-category dataframe."""
    from core.deployer import build_prediction_pipeline
    from sklearn.linear_model import LogisticRegression

    df = pd.DataFrame(
        {
            "region": ["East", "West", "East", "North", "East", "West", "North", "East"]
            * 2,
            "age": [25, 35, 45, 55, 28, 38, 48, 58] * 2,
            "balance": [1000, 5000, 3000, 8000, 500, 6000, 2000, 9000] * 2,
            "churned": ["no", "no", "yes", "no", "yes", "no", "yes", "no"] * 2,
        }
    )
    pipeline = build_prediction_pipeline(
        df, ["age", "balance"], "churned", "classification"
    )
    X = pipeline.transform_df(df)
    y = pipeline.target_encoder.transform(df["churned"].values)
    model = LogisticRegression(max_iter=500).fit(X, y)
    return pipeline, model, df


def test_cohort_returns_required_fields():
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=5, direction="highest")

    required = {
        "target_column",
        "problem_type",
        "n",
        "direction",
        "total_scored",
        "categorical_profile",
        "numeric_profile",
        "characterization",
    }
    assert required.issubset(result.keys())


def test_cohort_n_matches_request():
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=5, direction="highest")

    assert result["n"] == 5
    assert result["direction"] == "highest"


def test_cohort_target_column():
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=5, direction="highest")

    assert result["target_column"] == "revenue"
    assert result["problem_type"] == "regression"


def test_cohort_total_scored():
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=5, direction="highest")

    assert result["total_scored"] == len(df)


def test_cohort_categorical_profile_populated():
    """Dataset has a 'region' column — categorical_profile should include it."""
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=6, direction="highest")

    cat_cols = [cp["column"] for cp in result["categorical_profile"]]
    assert "region" in cat_cols


def test_cohort_categorical_profile_has_percentages():
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=6, direction="highest")

    for cp in result["categorical_profile"]:
        assert "dominant" in cp
        assert "dominant_top_pct" in cp
        for cat in cp["categories"]:
            assert "value" in cat
            assert "top_pct" in cat
            assert "overall_pct" in cat
            assert "ratio" in cat


def test_cohort_numeric_profile_populated():
    """Dataset has numeric columns units + price — numeric_profile should include them."""
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=6, direction="highest")

    num_cols = [np["column"] for np in result["numeric_profile"]]
    # units and price should be in numeric_profile; revenue (target) should not
    assert "units" in num_cols or "price" in num_cols
    assert "revenue" not in num_cols


def test_cohort_numeric_profile_has_means():
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=6, direction="highest")

    for np_row in result["numeric_profile"]:
        assert "top_mean" in np_row
        assert "overall_mean" in np_row
        assert "direction" in np_row
        assert np_row["direction"] in ("higher", "lower", "similar")


def test_cohort_characterization_is_string():
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=5, direction="highest")

    assert isinstance(result["characterization"], str)
    assert len(result["characterization"]) > 10


def test_cohort_characterization_mentions_target():
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=5, direction="highest")

    assert "revenue" in result["characterization"]


def test_cohort_classification_returns_correct_type():
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_classification_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=5, direction="highest")

    assert result["problem_type"] == "classification"
    assert result["target_column"] == "churned"


def test_cohort_lowest_direction():
    from core.deployer import compute_prediction_cohort

    pipeline, model, df = _make_regression_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        pp = str(Path(tmpdir) / "pipeline.joblib")
        mp = str(Path(tmpdir) / "model.joblib")
        joblib.dump(pipeline, pp)
        joblib.dump(model, mp)

        result = compute_prediction_cohort(pp, mp, df, n=5, direction="lowest")

    assert result["direction"] == "lowest"
    assert "lowest" in result["characterization"]


# ---------------------------------------------------------------------------
# Chat API integration — SSE prediction_cohort event
# ---------------------------------------------------------------------------

SAMPLE_CSV = b"""region,units,revenue
East,10,1200.5
West,8,850.0
East,18,2100.75
North,4,450.25
East,15,1650.0
West,9,980.0
East,11,1100.25
North,16,1750.0
East,20,2300.5
West,6,620.75
East,12,1300.0
West,9,950.0
North,20,2200.0
East,5,520.0
West,16,1700.0
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
    """Creates project → uploads CSV → applies features → trains → deploys."""
    proj = client.post("/api/projects", json={"name": "CohortTest"})
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
        mock_stream.text_stream = iter(["Here is the cohort profile."])
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


def test_cohort_event_emitted(client, deployed_project):
    """Sending a cohort message emits the prediction_cohort SSE event."""
    project_id = deployed_project["project_id"]
    events = _chat_events(
        client,
        project_id,
        "who are the top predictions and what do they have in common?",
    )
    types = [e.get("type") for e in events]
    assert "prediction_cohort" in types, f"Expected prediction_cohort in {types}"


def test_cohort_event_has_required_fields(client, deployed_project):
    """The emitted prediction_cohort event contains all required fields."""
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "profile the top ranked records")
    cohort_events = [e for e in events if e.get("type") == "prediction_cohort"]
    assert cohort_events, "No prediction_cohort event found"
    data = cohort_events[0]["prediction_cohort"]
    for field in (
        "target_column",
        "characterization",
        "categorical_profile",
        "numeric_profile",
    ):
        assert field in data, f"Missing field: {field}"
    assert data["target_column"] == "revenue"


def test_cohort_not_emitted_without_deployment(client):
    """No prediction_cohort event if project has no active deployment."""
    proj = client.post("/api/projects", json={"name": "NoDeploy"})
    project_id = proj.json()["id"]

    events = _chat_events(client, project_id, "profile the top ranked customers")
    cohort_events = [e for e in events if e.get("type") == "prediction_cohort"]
    assert len(cohort_events) == 0
