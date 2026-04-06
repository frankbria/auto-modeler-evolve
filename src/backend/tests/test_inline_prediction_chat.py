"""Tests for inline multi-feature prediction via chat.

Covers:
- _INLINE_PRED_PATTERNS detection
- _extract_multi_feature_prediction() helper
- Chat integration (SSE inline_prediction event)
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
# Pattern detection
# ---------------------------------------------------------------------------


def test_pattern_run_prediction():
    from api.chat import _INLINE_PRED_PATTERNS

    assert _INLINE_PRED_PATTERNS.search("run a prediction for Region=East, Units=100")


def test_pattern_make_prediction():
    from api.chat import _INLINE_PRED_PATTERNS

    assert _INLINE_PRED_PATTERNS.search("make a prediction with Units=200")


def test_pattern_give_me_prediction():
    from api.chat import _INLINE_PRED_PATTERNS

    assert _INLINE_PRED_PATTERNS.search("give me a prediction for these values")


def test_pattern_what_would_be():
    from api.chat import _INLINE_PRED_PATTERNS

    assert _INLINE_PRED_PATTERNS.search("what would my revenue be if Region=East")


def test_pattern_model_output():
    from api.chat import _INLINE_PRED_PATTERNS

    assert _INLINE_PRED_PATTERNS.search("what does the model predict for Units=100?")


def test_pattern_score_record():
    from api.chat import _INLINE_PRED_PATTERNS

    assert _INLINE_PRED_PATTERNS.search("score this record: Region=West, Units=50")


def test_pattern_run_model_on():
    from api.chat import _INLINE_PRED_PATTERNS

    assert _INLINE_PRED_PATTERNS.search("run the model on these inputs")


def test_pattern_negative_explore():
    """Generic exploration messages should NOT trigger inline prediction."""
    from api.chat import _INLINE_PRED_PATTERNS

    assert not _INLINE_PRED_PATTERNS.search("show me a summary of my data")
    assert not _INLINE_PRED_PATTERNS.search("what are the top selling regions?")
    assert not _INLINE_PRED_PATTERNS.search("how does revenue correlate with units?")


# ---------------------------------------------------------------------------
# _extract_multi_feature_prediction — pure function
# ---------------------------------------------------------------------------


def test_extract_explicit_equals():
    from api.chat import _extract_multi_feature_prediction

    feats = ["Region", "Units", "Category"]
    msg = "run a prediction for Region=East, Units=100, Category=Electronics"
    result = _extract_multi_feature_prediction(msg, feats)

    assert result["Region"] == "East"
    assert result["Units"] == 100.0
    assert result["Category"] == "Electronics"


def test_extract_case_insensitive():
    from api.chat import _extract_multi_feature_prediction

    feats = ["Revenue", "Units"]
    msg = "predict with revenue=500 and units=20"
    result = _extract_multi_feature_prediction(msg, feats)

    assert "Revenue" in result
    assert "Units" in result


def test_extract_underscore_variant():
    """Underscore column names are matched when the key uses underscores."""
    from api.chat import _extract_multi_feature_prediction

    feats = ["product_category"]
    msg = "make a prediction for product_category=Widget"
    result = _extract_multi_feature_prediction(msg, feats)

    assert "product_category" in result
    assert result["product_category"] == "Widget"


def test_extract_numeric_conversion():
    from api.chat import _extract_multi_feature_prediction

    feats = ["price", "quantity"]
    msg = "run a prediction for price=99.99 quantity=5"
    result = _extract_multi_feature_prediction(msg, feats)

    assert isinstance(result["price"], float)
    assert isinstance(result["quantity"], float)


def test_extract_unknown_features_excluded():
    """Features not in the model's feature list should not appear."""
    from api.chat import _extract_multi_feature_prediction

    feats = ["Region", "Units"]
    msg = "predict for Region=East, unknown_col=123"
    result = _extract_multi_feature_prediction(msg, feats)

    assert "Region" in result
    assert "unknown_col" not in result


def test_extract_empty_when_no_match():
    from api.chat import _extract_multi_feature_prediction

    feats = ["Revenue", "Units"]
    msg = "what can I predict with this data?"
    result = _extract_multi_feature_prediction(msg, feats)

    assert result == {}


# ---------------------------------------------------------------------------
# Chat integration — SSE inline_prediction event
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
    proj = client.post("/api/projects", json={"name": "InlinePredTest"})
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
        mock_stream.text_stream = iter(["The model predicts a result."])
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


def test_chat_inline_prediction_event_emitted(client, deployed_project):
    """When a deployed model exists and message matches inline prediction pattern,
    the SSE stream should contain an inline_prediction event."""
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "run a prediction for units=15")
    types = [e.get("type") for e in events]
    assert "inline_prediction" in types, f"Expected inline_prediction in {types}"


def test_chat_inline_prediction_result_fields(client, deployed_project):
    """The inline_prediction event should have required fields."""
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "make a prediction with units=20")
    pred_events = [e for e in events if e.get("type") == "inline_prediction"]
    assert pred_events, "No inline_prediction event"
    payload = pred_events[0]["inline_prediction"]

    assert "prediction" in payload
    assert "provided_features" in payload
    assert "summary" in payload
    assert "target_column" in payload
    assert payload["target_column"] == "revenue"


def test_chat_inline_prediction_provided_features_present(client, deployed_project):
    """Extracted feature values appear in provided_features."""
    project_id = deployed_project["project_id"]
    events = _chat_events(client, project_id, "calculate a prediction for units=30")
    pred_events = [e for e in events if e.get("type") == "inline_prediction"]
    assert pred_events
    provided = pred_events[0]["inline_prediction"]["provided_features"]
    assert len(provided) >= 1
