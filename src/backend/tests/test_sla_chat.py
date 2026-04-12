"""Tests for SLA latency monitoring via chat.

Covers:
- _SLA_PATTERNS detection (8 positive, 2 negative)
- Chat SSE event emitted when deployment exists and predictions have latency data
- Chat SSE event contains required fields
- Alert flag set when p95 > 500ms
- Empty-predictions path returns event with sample_count=0
- latency_by_day grouping included in event
"""

from __future__ import annotations

import io
import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

import db as db_module

# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


def test_pattern_show_latency():
    from api.chat import _SLA_PATTERNS

    assert _SLA_PATTERNS.search("show me the prediction latency")


def test_pattern_check_latency():
    from api.chat import _SLA_PATTERNS

    assert _SLA_PATTERNS.search("check my latency stats")


def test_pattern_how_fast():
    from api.chat import _SLA_PATTERNS

    assert _SLA_PATTERNS.search("how fast is my model?")


def test_pattern_p95_latency():
    from api.chat import _SLA_PATTERNS

    assert _SLA_PATTERNS.search("what is the p95 latency?")


def test_pattern_sla_status():
    from api.chat import _SLA_PATTERNS

    assert _SLA_PATTERNS.search("SLA status for my deployment")


def test_pattern_response_time():
    from api.chat import _SLA_PATTERNS

    assert _SLA_PATTERNS.search("response time stats")


def test_pattern_api_speed():
    from api.chat import _SLA_PATTERNS

    assert _SLA_PATTERNS.search("API speed for my model")


def test_pattern_how_long_predictions():
    from api.chat import _SLA_PATTERNS

    assert _SLA_PATTERNS.search("how long do predictions take?")


def test_pattern_no_match_accuracy():
    from api.chat import _SLA_PATTERNS

    assert not _SLA_PATTERNS.search("what is the model accuracy?")


def test_pattern_no_match_train():
    from api.chat import _SLA_PATTERNS

    assert not _SLA_PATTERNS.search("train a new model with random forest")


# ---------------------------------------------------------------------------
# Sample CSV
# ---------------------------------------------------------------------------

SAMPLE_CSV = (
    b"feat1,feat2,target\n"
    b"1.0,0.5,10.0\n"
    b"2.0,1.0,20.0\n"
    b"3.0,1.5,30.0\n"
    b"4.0,2.0,40.0\n"
    b"5.0,2.5,50.0\n"
    b"6.0,3.0,60.0\n"
    b"7.0,3.5,70.0\n"
    b"8.0,4.0,80.0\n"
    b"9.0,4.5,90.0\n"
    b"10.0,5.0,100.0\n"
    b"11.0,5.5,110.0\n"
    b"12.0,6.0,120.0\n"
    b"13.0,6.5,130.0\n"
    b"14.0,7.0,140.0\n"
    b"15.0,7.5,150.0\n"
)


@pytest.fixture()
def client(tmp_path):
    """TestClient backed by an isolated SQLite DB for this test."""
    from sqlmodel import SQLModel, create_engine as _ce

    from main import app

    test_db = str(tmp_path / "sla_test.db")
    orig_engine = db_module.engine
    db_module.engine = _ce(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    yield TestClient(app)

    db_module.engine = orig_engine


@pytest.fixture()
def deployed_project(client, tmp_path):
    """Create project, upload CSV, train a model, and deploy it."""
    proj = client.post("/api/projects", json={"name": "SLATest"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sla.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "target", "problem_type": "regression"},
    )

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
    deployment_id = deploy_resp.json()["id"]

    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "deployment_id": deployment_id,
    }


def _inject_prediction_logs(deployment_id: str, latencies: list[float]) -> None:
    """Directly insert PredictionLog rows with given response_ms values."""
    from models.prediction_log import PredictionLog

    with Session(db_module.engine) as session:
        for i, ms in enumerate(latencies):
            log = PredictionLog(
                deployment_id=deployment_id,
                input_features=json.dumps({"feat1": float(i + 1)}),
                prediction=str(float(i * 10)),
                response_ms=ms,
                created_at=datetime.now(UTC).replace(tzinfo=None)
                - timedelta(days=i % 3),
            )
            session.add(log)
        session.commit()


def _chat_events(client, project_id: str, message: str) -> list[dict]:
    """Send a chat message with mocked Anthropic and return all SSE events."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_c = MagicMock()
        mock_cls.return_value = mock_c
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Latency looks good."])
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


def test_sla_event_emitted_with_data(client, deployed_project):
    """Chat emits sla_metrics SSE event when deployment has prediction logs."""
    dep_id = deployed_project["deployment_id"]
    _inject_prediction_logs(dep_id, [50.0, 80.0, 120.0, 200.0, 300.0])

    events = _chat_events(
        client, deployed_project["project_id"], "show me the prediction latency"
    )
    types = [e.get("type") for e in events]
    assert "sla_metrics" in types, f"Expected sla_metrics in {types}"


def test_sla_event_required_fields(client, deployed_project):
    """sla_metrics event contains all required fields."""
    dep_id = deployed_project["deployment_id"]
    _inject_prediction_logs(dep_id, [50.0, 80.0, 120.0])

    events = _chat_events(
        client, deployed_project["project_id"], "check my latency stats"
    )
    sla = next(e["sla_metrics"] for e in events if e.get("type") == "sla_metrics")
    assert "deployment_id" in sla
    assert "sample_count" in sla
    assert "p50_ms" in sla
    assert "p95_ms" in sla
    assert "p99_ms" in sla
    assert "avg_ms" in sla
    assert "alert" in sla
    assert "alert_message" in sla
    assert "latency_by_day" in sla
    assert "summary" in sla


def test_sla_event_no_predictions(client, deployed_project):
    """sla_metrics event with sample_count=0 when no prediction logs exist."""
    events = _chat_events(
        client, deployed_project["project_id"], "how fast is my model?"
    )
    sla = next(e["sla_metrics"] for e in events if e.get("type") == "sla_metrics")
    assert sla["sample_count"] == 0
    assert sla["p50_ms"] is None
    assert sla["alert"] is False


def test_sla_alert_when_p95_over_500ms(client, deployed_project):
    """alert=True when p95 latency exceeds 500ms."""
    dep_id = deployed_project["deployment_id"]
    # Inject latencies that push p95 above 500ms
    latencies = [50.0] * 18 + [600.0, 700.0]  # p95 will be ~600ms
    _inject_prediction_logs(dep_id, latencies)

    events = _chat_events(
        client, deployed_project["project_id"], "p95 latency for my API"
    )
    sla = next(e["sla_metrics"] for e in events if e.get("type") == "sla_metrics")
    assert sla["alert"] is True
    assert sla["alert_message"] is not None
    assert "500ms" in sla["alert_message"]


def test_sla_no_alert_when_p95_healthy(client, deployed_project):
    """alert=False when p95 latency is under 500ms."""
    dep_id = deployed_project["deployment_id"]
    _inject_prediction_logs(dep_id, [10.0, 20.0, 30.0, 40.0, 50.0])

    events = _chat_events(
        client, deployed_project["project_id"], "show me the prediction latency"
    )
    sla = next(e["sla_metrics"] for e in events if e.get("type") == "sla_metrics")
    assert sla["alert"] is False
    assert sla["alert_message"] is None
