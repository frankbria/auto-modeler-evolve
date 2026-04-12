"""Tests for Calibration Check via chat.

Covers:
- _CALIBRATION_CHECK_PATTERNS detection (7 positive, 2 negative)
- Chat SSE event emitted when model has calibration data
- Chat SSE event contains required fields
- Brier score quality bucket ("excellent" when < 0.1)
- calibration_curve data forwarded in event
"""

from __future__ import annotations

import io
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

import db as db_module

# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


def test_pattern_how_well_calibrated():
    from api.chat import _CALIBRATION_CHECK_PATTERNS

    assert _CALIBRATION_CHECK_PATTERNS.search("how well-calibrated is my model?")


def test_pattern_calibrated_classifier():
    from api.chat import _CALIBRATION_CHECK_PATTERNS

    assert _CALIBRATION_CHECK_PATTERNS.search("how well calibrated is the classifier")


def test_pattern_confidence_scores_reliable():
    from api.chat import _CALIBRATION_CHECK_PATTERNS

    assert _CALIBRATION_CHECK_PATTERNS.search("are my confidence scores reliable?")


def test_pattern_reliability_diagram():
    from api.chat import _CALIBRATION_CHECK_PATTERNS

    assert _CALIBRATION_CHECK_PATTERNS.search("show me the reliability diagram")


def test_pattern_show_calibration():
    from api.chat import _CALIBRATION_CHECK_PATTERNS

    assert _CALIBRATION_CHECK_PATTERNS.search("show calibration for my model")


def test_pattern_brier_score():
    from api.chat import _CALIBRATION_CHECK_PATTERNS

    assert _CALIBRATION_CHECK_PATTERNS.search("what's the brier score?")


def test_pattern_calibration_check():
    from api.chat import _CALIBRATION_CHECK_PATTERNS

    assert _CALIBRATION_CHECK_PATTERNS.search("calibration check for the predictions")


def test_pattern_no_match_unrelated():
    from api.chat import _CALIBRATION_CHECK_PATTERNS

    assert not _CALIBRATION_CHECK_PATTERNS.search("train a new model please")


def test_pattern_no_match_general_accuracy():
    from api.chat import _CALIBRATION_CHECK_PATTERNS

    assert not _CALIBRATION_CHECK_PATTERNS.search("what is the model accuracy?")


# ---------------------------------------------------------------------------
# Sample CSV for integration tests
# ---------------------------------------------------------------------------

SAMPLE_CSV = (
    b"feat1,feat2,label\n"
    b"1.0,0.5,0\n"
    b"2.0,1.0,0\n"
    b"3.0,1.5,0\n"
    b"4.0,2.0,1\n"
    b"5.0,2.5,1\n"
    b"6.0,3.0,1\n"
    b"7.0,3.5,0\n"
    b"8.0,4.0,1\n"
    b"9.0,4.5,1\n"
    b"10.0,5.0,0\n"
    b"11.0,5.5,0\n"
    b"12.0,6.0,1\n"
    b"13.0,6.5,1\n"
    b"14.0,7.0,0\n"
    b"15.0,7.5,1\n"
    b"16.0,8.0,1\n"
    b"17.0,8.5,0\n"
    b"18.0,9.0,1\n"
    b"19.0,9.5,0\n"
    b"20.0,10.0,1\n"
)


@pytest.fixture()
def client(tmp_path):
    """TestClient backed by an isolated SQLite DB for this test."""
    import db as db_module_local

    from sqlmodel import SQLModel, create_engine as _ce

    from main import app

    test_db = str(tmp_path / "cal_test.db")
    orig_engine = db_module_local.engine
    db_module_local.engine = _ce(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(db_module_local.engine)
    # Run inline migrations so deployment/predictionlog columns exist
    db_module_local.create_db_and_tables()

    yield TestClient(app)

    db_module_local.engine = orig_engine


@pytest.fixture()
def calibrated_project(client, tmp_path):
    """Create project, upload CSV, train a model, then inject calibration metrics."""
    proj = client.post("/api/projects", json={"name": "CalCheckTest"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("cal.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "label", "problem_type": "binary_classification"},
    )

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["logistic_regression"]},
    )
    run_id = train_resp.json()["model_run_ids"][0]

    # Wait for training to complete
    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert run["status"] == "done"

    # Inject calibration metrics directly into the model run record
    fake_calibration = {
        "is_calibrated": True,
        "brier_score": 0.08,
        "calibration_note": "Injected for test.",
        "calibration_curve": [
            {"predicted": 0.1, "actual": 0.08},
            {"predicted": 0.5, "actual": 0.5},
            {"predicted": 0.9, "actual": 0.92},
        ],
    }
    from models.model_run import ModelRun

    with Session(db_module.engine) as session:
        run_obj = session.get(ModelRun, run_id)
        existing = json.loads(run_obj.metrics or "{}")
        existing.update(fake_calibration)
        run_obj.metrics = json.dumps(existing)
        run_obj.is_selected = True
        session.add(run_obj)
        session.commit()

    return {"project_id": project_id, "dataset_id": dataset_id, "run_id": run_id}


def _chat_events(client, project_id: str, message: str) -> list[dict]:
    """Send a chat message with mocked Anthropic and return all SSE events."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_c = MagicMock()
        mock_cls.return_value = mock_c
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["The calibration looks good."])
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


def test_calibration_check_event_emitted(client, calibrated_project):
    """Chat emits calibration_check SSE event when model has calibration data."""
    project_id = calibrated_project["project_id"]
    events = _chat_events(client, project_id, "how well-calibrated is my model?")
    types = [e.get("type") for e in events]
    assert "calibration_check" in types, f"Expected calibration_check in {types}"


def test_calibration_check_required_fields(client, calibrated_project):
    """calibration_check event contains all required fields."""
    project_id = calibrated_project["project_id"]
    events = _chat_events(client, project_id, "show calibration for my model")
    cal = next(
        e["calibration_check"] for e in events if e.get("type") == "calibration_check"
    )
    assert cal["is_calibrated"] is True
    assert "brier_score" in cal
    assert "calibration_curve" in cal
    assert "summary" in cal
    assert "calibration_quality" in cal
    assert "algorithm" in cal


def test_calibration_brier_score_quality(client, calibrated_project):
    """Brier score 0.08 → calibration_quality='excellent'."""
    project_id = calibrated_project["project_id"]
    events = _chat_events(client, project_id, "brier score for my model")
    cal = next(
        e["calibration_check"] for e in events if e.get("type") == "calibration_check"
    )
    assert cal["brier_score"] == pytest.approx(0.08)
    assert cal["calibration_quality"] == "excellent"


def test_calibration_curve_data_forwarded(client, calibrated_project):
    """calibration_curve from metrics is forwarded verbatim in the event."""
    project_id = calibrated_project["project_id"]
    events = _chat_events(client, project_id, "reliability diagram")
    cal = next(
        e["calibration_check"] for e in events if e.get("type") == "calibration_check"
    )
    assert len(cal["calibration_curve"]) == 3
    assert cal["calibration_curve"][1]["predicted"] == pytest.approx(0.5)
