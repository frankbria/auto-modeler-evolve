"""Tests for dataset ranking via model ("which customers are most likely to churn?").

Covers:
- _RANKED_PRED_PATTERNS detection
- _detect_ranked_pred_request() helper
- run_dataset_ranking() pure function (regression + classification)
- Chat integration: SSE ranked_predictions event emitted
"""

import io
import json
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_pattern_which_customers_most_likely():
    from api.chat import _RANKED_PRED_PATTERNS

    assert _RANKED_PRED_PATTERNS.search("which customers are most likely to churn")


def test_pattern_show_top_n():
    from api.chat import _RANKED_PRED_PATTERNS

    assert _RANKED_PRED_PATTERNS.search("show me the top 20 predictions")


def test_pattern_rank_by_prediction():
    from api.chat import _RANKED_PRED_PATTERNS

    assert _RANKED_PRED_PATTERNS.search("rank by predicted revenue")


def test_pattern_most_at_risk():
    from api.chat import _RANKED_PRED_PATTERNS

    assert _RANKED_PRED_PATTERNS.search("which records are most at risk?")


def test_pattern_best_opportunities():
    from api.chat import _RANKED_PRED_PATTERNS

    assert _RANKED_PRED_PATTERNS.search("show me the best 10 opportunities")


def test_pattern_apply_model_to_all_data():
    from api.chat import _RANKED_PRED_PATTERNS

    assert _RANKED_PRED_PATTERNS.search("apply the model to all my data")


def test_pattern_bottom():
    from api.chat import _RANKED_PRED_PATTERNS

    assert _RANKED_PRED_PATTERNS.search("bottom 5 by prediction")


def test_pattern_negative_unrelated():
    from api.chat import _RANKED_PRED_PATTERNS

    assert not _RANKED_PRED_PATTERNS.search("what is the average revenue?")


# ---------------------------------------------------------------------------
# _detect_ranked_pred_request helper
# ---------------------------------------------------------------------------


def test_detect_default_n_and_highest():
    from api.chat import _detect_ranked_pred_request

    req = _detect_ranked_pred_request("which customers are most likely to churn?")
    assert req["n"] == 20
    assert req["direction"] == "highest"


def test_detect_explicit_top_n():
    from api.chat import _detect_ranked_pred_request

    req = _detect_ranked_pred_request("show me the top 15 predictions")
    assert req["n"] == 15
    assert req["direction"] == "highest"


def test_detect_bottom_direction():
    from api.chat import _detect_ranked_pred_request

    req = _detect_ranked_pred_request("show me the bottom 10 by prediction")
    assert req["n"] == 10
    assert req["direction"] == "lowest"


def test_detect_lowest_keyword_direction():
    from api.chat import _detect_ranked_pred_request

    req = _detect_ranked_pred_request("find the lowest 5 predictions")
    assert req["direction"] == "lowest"


def test_detect_n_capped_at_100():
    from api.chat import _detect_ranked_pred_request

    req = _detect_ranked_pred_request("show me top 999 predictions")
    assert req["n"] == 100


# ---------------------------------------------------------------------------
# run_dataset_ranking pure function
# ---------------------------------------------------------------------------


def _make_regression_pipeline_and_model():
    """Build a minimal regression pipeline and model for testing."""
    from core.deployer import build_prediction_pipeline
    from sklearn.linear_model import LinearRegression

    df = pd.DataFrame(
        {
            "units": [10, 20, 30, 40, 50],
            "price": [5.0, 4.5, 4.0, 3.5, 3.0],
            "revenue": [50, 90, 120, 140, 150],
        }
    )
    pipeline = build_prediction_pipeline(df, ["units", "price"], "revenue", "regression")
    X = pipeline.transform_df(df)
    y = df["revenue"].values
    model = LinearRegression().fit(X, y)
    return pipeline, model, df


def _make_classification_pipeline_and_model():
    """Build a minimal classification pipeline and model for testing."""
    from core.deployer import build_prediction_pipeline
    from sklearn.linear_model import LogisticRegression

    df = pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 28, 38, 48, 58],
            "balance": [1000, 5000, 3000, 8000, 500, 6000, 2000, 9000],
            "churned": ["no", "no", "yes", "no", "yes", "no", "yes", "no"],
        }
    )
    pipeline = build_prediction_pipeline(df, ["age", "balance"], "churned", "classification")
    X = pipeline.transform_df(df)
    y = pipeline.target_encoder.transform(df["churned"].astype(str))
    model = LogisticRegression(max_iter=200).fit(X, y)
    return pipeline, model, df


def test_ranking_regression_required_fields(tmp_path):
    from core.deployer import run_dataset_ranking, save_pipeline
    import joblib

    pipeline, model, df = _make_regression_pipeline_and_model()
    pp = tmp_path / "pipe.joblib"
    mp = tmp_path / "model.joblib"
    save_pipeline(pipeline, pp)
    joblib.dump(model, mp)

    result = run_dataset_ranking(str(pp), str(mp), df)
    assert "problem_type" in result
    assert "target_column" in result
    assert "direction" in result
    assert "n" in result
    assert "total_scored" in result
    assert "rows" in result
    assert "summary" in result
    assert "class_names" in result


def test_ranking_regression_returns_n_rows(tmp_path):
    from core.deployer import run_dataset_ranking, save_pipeline
    import joblib

    pipeline, model, df = _make_regression_pipeline_and_model()
    pp = tmp_path / "pipe.joblib"
    mp = tmp_path / "model.joblib"
    save_pipeline(pipeline, pp)
    joblib.dump(model, mp)

    result = run_dataset_ranking(str(pp), str(mp), df, n=3)
    assert result["n"] == 3
    assert len(result["rows"]) == 3
    assert result["total_scored"] == len(df)


def test_ranking_regression_highest_direction(tmp_path):
    from core.deployer import run_dataset_ranking, save_pipeline
    import joblib

    pipeline, model, df = _make_regression_pipeline_and_model()
    pp = tmp_path / "pipe.joblib"
    mp = tmp_path / "model.joblib"
    save_pipeline(pipeline, pp)
    joblib.dump(model, mp)

    result = run_dataset_ranking(str(pp), str(mp), df, n=5, direction="highest")
    predictions = [r["prediction"] for r in result["rows"]]
    # Should be descending
    assert predictions == sorted(predictions, reverse=True)


def test_ranking_regression_lowest_direction(tmp_path):
    from core.deployer import run_dataset_ranking, save_pipeline
    import joblib

    pipeline, model, df = _make_regression_pipeline_and_model()
    pp = tmp_path / "pipe.joblib"
    mp = tmp_path / "model.joblib"
    save_pipeline(pipeline, pp)
    joblib.dump(model, mp)

    result = run_dataset_ranking(str(pp), str(mp), df, n=5, direction="lowest")
    predictions = [r["prediction"] for r in result["rows"]]
    # Should be ascending
    assert predictions == sorted(predictions)


def test_ranking_regression_rank_numbers(tmp_path):
    from core.deployer import run_dataset_ranking, save_pipeline
    import joblib

    pipeline, model, df = _make_regression_pipeline_and_model()
    pp = tmp_path / "pipe.joblib"
    mp = tmp_path / "model.joblib"
    save_pipeline(pipeline, pp)
    joblib.dump(model, mp)

    result = run_dataset_ranking(str(pp), str(mp), df, n=3)
    ranks = [r["rank"] for r in result["rows"]]
    assert ranks == [1, 2, 3]


def test_ranking_classification_returns_class_and_confidence(tmp_path):
    from core.deployer import run_dataset_ranking, save_pipeline
    import joblib

    pipeline, model, df = _make_classification_pipeline_and_model()
    pp = tmp_path / "pipe.joblib"
    mp = tmp_path / "model.joblib"
    save_pipeline(pipeline, pp)
    joblib.dump(model, mp)

    result = run_dataset_ranking(str(pp), str(mp), df, n=3)
    assert result["problem_type"] == "classification"
    assert result["class_names"] is not None
    row = result["rows"][0]
    assert "predicted_class" in row
    assert "confidence" in row
    assert "probabilities" in row
    assert 0.0 <= row["confidence"] <= 1.0


def test_ranking_summary_mentions_target(tmp_path):
    from core.deployer import run_dataset_ranking, save_pipeline
    import joblib

    pipeline, model, df = _make_regression_pipeline_and_model()
    pp = tmp_path / "pipe.joblib"
    mp = tmp_path / "model.joblib"
    save_pipeline(pipeline, pp)
    joblib.dump(model, mp)

    result = run_dataset_ranking(str(pp), str(mp), df)
    assert "revenue" in result["summary"]


def test_ranking_empty_df_raises(tmp_path):
    from core.deployer import run_dataset_ranking, save_pipeline
    import joblib

    pipeline, model, df = _make_regression_pipeline_and_model()
    pp = tmp_path / "pipe.joblib"
    mp = tmp_path / "model.joblib"
    save_pipeline(pipeline, pp)
    joblib.dump(model, mp)

    with pytest.raises(ValueError, match="empty"):
        run_dataset_ranking(str(pp), str(mp), pd.DataFrame())


# ---------------------------------------------------------------------------
# API integration tests
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
    proj = client.post("/api/projects", json={"name": "RankedPredTest"})
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
        mock_stream.text_stream = iter(["Here are the ranked predictions."])
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


def test_ranked_predictions_event_emitted(client, deployed_project):
    """Sending a ranking message emits the ranked_predictions SSE event."""
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "show me the top 5 predictions")
    types = [e.get("type") for e in events]
    assert "ranked_predictions" in types, f"Expected ranked_predictions in {types}"


def test_ranked_predictions_required_fields(client, deployed_project):
    """The emitted ranked_predictions event contains all required fields."""
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "rank by predicted revenue")
    rp_events = [e for e in events if e.get("type") == "ranked_predictions"]
    assert rp_events, "No ranked_predictions event found"
    rp = rp_events[0]["ranked_predictions"]
    for field in ("rows", "total_scored", "summary", "target_column", "direction"):
        assert field in rp, f"Missing field: {field}"
    assert len(rp["rows"]) > 0
    assert rp["rows"][0]["rank"] == 1
    assert rp["target_column"] == "revenue"


def test_ranked_predictions_not_emitted_without_deployment(client):
    """No ranked_predictions event if project has no active deployment."""
    proj = client.post("/api/projects", json={"name": "NoDeploy"})
    project_id = proj.json()["id"]

    events = _chat_events(client, project_id, "show me the top 10 predictions")
    rp_events = [e for e in events if e.get("type") == "ranked_predictions"]
    assert len(rp_events) == 0
