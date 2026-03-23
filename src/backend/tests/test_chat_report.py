"""Tests for chat-triggered PDF report generation.

Covers:
- _REPORT_PATTERNS regex detects report-request phrases
- _REPORT_PATTERNS does not match unrelated phrases
- Chat SSE stream emits report_ready event when model is trained
- report_ready event includes model_run_id, algorithm, download_url
- Chat handles no completed runs gracefully (no report_ready event)
"""

import json
import time

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _REPORT_PATTERNS

# ---------------------------------------------------------------------------
# Sample CSV
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units,cost\n"
    b"East,100.5,10,50\n"
    b"West,200.3,20,80\n"
    b"East,150.7,15,60\n"
    b"West,300.1,30,120\n"
    b"North,250.9,25,100\n"
    b"East,175.2,18,70\n"
    b"West,220.4,22,90\n"
    b"North,190.6,19,75\n"
    b"East,130.8,13,55\n"
    b"West,280.0,28,110\n"
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
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    import api.models as models_module

    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Report Chat Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
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
    for _ in range(20):
        r = await ac.get(f"/api/models/{project_id}/runs")
        run = next((x for x in r.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.5)
    pytest.skip("Training did not complete in time")


# ---------------------------------------------------------------------------
# Unit tests — regex patterns
# ---------------------------------------------------------------------------


def test_report_pattern_generate_report():
    assert _REPORT_PATTERNS.search("generate a report")


def test_report_pattern_create_pdf():
    assert _REPORT_PATTERNS.search("create a PDF report")


def test_report_pattern_download_report():
    assert _REPORT_PATTERNS.search("download the model report")


def test_report_pattern_export_report():
    assert _REPORT_PATTERNS.search("export a report")


def test_report_pattern_give_me_report():
    assert _REPORT_PATTERNS.search("give me a PDF report")


def test_report_pattern_pdf_report_noun():
    assert _REPORT_PATTERNS.search("pdf report")


def test_report_pattern_share_report():
    assert _REPORT_PATTERNS.search("share a model report")


def test_report_pattern_print_report():
    assert _REPORT_PATTERNS.search("print the model report")


def test_report_pattern_make_report():
    assert _REPORT_PATTERNS.search("make a report")


def test_report_pattern_no_false_positive_upload():
    assert not _REPORT_PATTERNS.search("upload my data")


def test_report_pattern_no_false_positive_train():
    assert not _REPORT_PATTERNS.search("train a model")


def test_report_pattern_no_false_positive_deploy():
    assert not _REPORT_PATTERNS.search("deploy my model")


# ---------------------------------------------------------------------------
# Integration tests — SSE events
# ---------------------------------------------------------------------------


def _parse_sse_events(text: str) -> list[dict]:
    """Extract all JSON payloads from SSE stream text."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _mock_anthropic():
    """Return a context manager that mocks the Anthropic client."""
    import unittest.mock as mock

    mock_stream = mock.MagicMock()
    mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = mock.MagicMock(return_value=False)
    mock_stream.text_stream = iter(["Your PDF report is ready to download."])
    mock_anthropic = mock.patch("anthropic.Anthropic")
    mock_cls = mock_anthropic.start()
    mock_cls.return_value.messages.stream.return_value = mock_stream
    return mock_anthropic


@pytest.mark.anyio
async def test_chat_emits_report_ready_event(ac, project_id, trained_run_id):
    """Chat emits report_ready event when a trained model exists."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Your PDF report is ready!"])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "generate a report"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    report_events = [e for e in events if e.get("type") == "report_ready"]
    assert len(report_events) == 1
    ev = report_events[0]
    assert "report" in ev
    assert ev["report"]["model_run_id"] == trained_run_id
    assert ev["report"]["algorithm"] == "linear_regression"
    assert "/api/models/" in ev["report"]["download_url"]
    assert ev["report"]["download_url"].endswith("/report")
    assert ev["report"]["metric_name"] in ("r2", "accuracy")


@pytest.mark.anyio
async def test_chat_report_includes_metric_value(ac, project_id, trained_run_id):
    """report_ready event includes a numeric metric_value."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Downloading now."])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "download the model report"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    report_events = [e for e in events if e.get("type") == "report_ready"]
    assert len(report_events) == 1
    val = report_events[0]["report"]["metric_value"]
    assert val is None or isinstance(val, (int, float))


@pytest.mark.anyio
async def test_chat_no_report_event_without_model(ac, project_id):
    """Chat does NOT emit report_ready when no model runs exist."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["No model available yet."])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "generate a report"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    report_events = [e for e in events if e.get("type") == "report_ready"]
    assert len(report_events) == 0


@pytest.mark.anyio
async def test_chat_report_pdf_report_phrase(ac, project_id, trained_run_id):
    """'pdf report' phrase triggers report_ready event."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["PDF report ready."])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "pdf report"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    assert any(e.get("type") == "report_ready" for e in events)
