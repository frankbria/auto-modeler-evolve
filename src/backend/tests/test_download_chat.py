"""Tests for dataset download chat feature:
- _DOWNLOAD_PATTERNS in chat.py
- SSE data_export event emission via send_message
- GET /api/data/{id}/download REST endpoint
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100,10\n"
    b"West,200,20\n"
    b"East,150,15\n"
)


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
def project_and_dataset(client):
    proj_resp = client.post("/api/projects/", json={"name": "Download Test"})
    project_id = proj_resp.json()["id"]
    resp = client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code in (200, 201), resp.text
    return project_id, resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Unit tests — _DOWNLOAD_PATTERNS
# ---------------------------------------------------------------------------


def test_download_pattern_download_data():
    from api.chat import _DOWNLOAD_PATTERNS

    assert _DOWNLOAD_PATTERNS.search("download my data")


def test_download_pattern_download_dataset():
    from api.chat import _DOWNLOAD_PATTERNS

    assert _DOWNLOAD_PATTERNS.search("download the dataset")


def test_download_pattern_export_csv():
    from api.chat import _DOWNLOAD_PATTERNS

    assert _DOWNLOAD_PATTERNS.search("export data to CSV")


def test_download_pattern_export_data():
    from api.chat import _DOWNLOAD_PATTERNS

    assert _DOWNLOAD_PATTERNS.search("export my data")


def test_download_pattern_export_results():
    from api.chat import _DOWNLOAD_PATTERNS

    assert _DOWNLOAD_PATTERNS.search("export the results")


def test_download_pattern_save_data():
    from api.chat import _DOWNLOAD_PATTERNS

    assert _DOWNLOAD_PATTERNS.search("save the data as csv")


def test_download_pattern_filtered_data():
    from api.chat import _DOWNLOAD_PATTERNS

    assert _DOWNLOAD_PATTERNS.search("download my filtered data")


def test_download_pattern_give_me_csv():
    from api.chat import _DOWNLOAD_PATTERNS

    assert _DOWNLOAD_PATTERNS.search("give me the dataset as csv")


# Non-matches
def test_download_pattern_no_match_report():
    from api.chat import _DOWNLOAD_PATTERNS

    # PDF report download has its own pattern; plain "download" without dataset context
    assert not _DOWNLOAD_PATTERNS.search("show me the top customers")


def test_download_pattern_no_match_generic():
    from api.chat import _DOWNLOAD_PATTERNS

    assert not _DOWNLOAD_PATTERNS.search("what is the revenue trend")


# ---------------------------------------------------------------------------
# Chat SSE integration tests
# ---------------------------------------------------------------------------


def test_download_chat_emits_data_export_event(client, project_and_dataset):
    """Sending a download request via chat should emit a {type:'data_export'} event."""
    project_id, dataset_id = project_and_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Your data export is ready."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={
                "message": "download my data",
                "dataset_id": dataset_id,
            },
        )

    assert response.status_code == 200
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    event_types = []
    for line in lines:
        try:
            parsed = json.loads(line[6:])
            event_types.append(parsed.get("type"))
        except json.JSONDecodeError:
            pass

    assert "data_export" in event_types, f"Expected data_export event. Got: {event_types}"


def test_download_chat_event_has_correct_fields(client, project_and_dataset):
    """The emitted data_export event should contain the expected fields."""
    project_id, dataset_id = project_and_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here is your export."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={
                "message": "export my data",
                "dataset_id": dataset_id,
            },
        )

    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    export_event = None
    for line in lines:
        try:
            parsed = json.loads(line[6:])
            if parsed.get("type") == "data_export":
                export_event = parsed
                break
        except json.JSONDecodeError:
            pass

    assert export_event is not None, "No data_export event found"
    ex = export_event["data_export"]
    assert "dataset_id" in ex
    assert "filename" in ex
    assert "row_count" in ex
    assert "filtered" in ex
    assert "download_url" in ex
    assert ex["row_count"] == 3  # 3 rows in sample CSV
    assert ex["filtered"] is False
    assert "/api/data/" in ex["download_url"]
    assert "/download" in ex["download_url"]


# ---------------------------------------------------------------------------
# REST endpoint — GET /api/data/{id}/download
# ---------------------------------------------------------------------------


def test_download_endpoint_returns_csv(client, project_and_dataset):
    """GET /api/data/{id}/download should return a CSV file."""
    _project_id, dataset_id = project_and_dataset
    response = client.get(f"/api/data/{dataset_id}/download")
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    content = response.text
    # Should contain CSV headers
    assert "region" in content
    assert "revenue" in content
    assert "East" in content


def test_download_endpoint_404_on_missing_dataset(client):
    """GET /api/data/{id}/download should return 404 for unknown dataset."""
    response = client.get("/api/data/nonexistent-id-xyz/download")
    assert response.status_code == 404


def test_download_endpoint_has_content_disposition(client, project_and_dataset):
    """The download response should have a Content-Disposition attachment header."""
    _project_id, dataset_id = project_and_dataset
    response = client.get(f"/api/data/{dataset_id}/download")
    assert response.status_code == 200
    cd = response.headers.get("content-disposition", "")
    assert "attachment" in cd


def test_download_endpoint_filtered_data(client, project_and_dataset):
    """When an active filter is set, the download should return only filtered rows."""
    _project_id, dataset_id = project_and_dataset

    # Set a filter: region = East
    filter_resp = client.post(
        f"/api/data/{dataset_id}/set-filter",
        json={"conditions": [{"column": "region", "operator": "eq", "value": "East"}]},
    )
    assert filter_resp.status_code == 200

    response = client.get(f"/api/data/{dataset_id}/download")
    assert response.status_code == 200
    content = response.text
    lines = [ln for ln in content.strip().split("\n") if ln]
    # Header + 2 East rows
    assert len(lines) == 3
    assert "East" in content
    # West should be filtered out
    assert "West" not in content
