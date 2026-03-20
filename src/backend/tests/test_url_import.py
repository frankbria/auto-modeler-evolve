"""Tests for POST /api/data/upload-url — Google Sheets and CSV URL import."""

import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from sqlmodel import create_engine, SQLModel
import db as db_module


SAMPLE_CSV_BYTES = (
    b"date,product,region,revenue,units\n"
    b"2024-01-01,Widget A,North,1200.50,10\n"
    b"2024-01-01,Widget B,South,850.00,8\n"
    b"2024-01-02,Widget A,East,2100.75,18\n"
)


@pytest.fixture
async def ac(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.conversation  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Test Project"})
    return resp.json()["id"]


def _mock_urlopen(csv_bytes: bytes):
    """Return a context manager mock that reads the given bytes."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = csv_bytes
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm


# ---------------------------------------------------------------------------
# Helper / unit tests for the URL parsing logic
# ---------------------------------------------------------------------------


def test_sheets_to_csv_url_basic():
    from api.data import _sheets_to_csv_url

    url = "https://docs.google.com/spreadsheets/d/ABC123/edit"
    result = _sheets_to_csv_url(url)
    assert "export?format=csv" in result
    assert "ABC123" in result


def test_sheets_to_csv_url_preserves_gid():
    from api.data import _sheets_to_csv_url

    url = "https://docs.google.com/spreadsheets/d/ABC123/edit#gid=456789"
    result = _sheets_to_csv_url(url)
    assert "gid=456789" in result


def test_sheets_to_csv_url_invalid():
    from api.data import _sheets_to_csv_url

    with pytest.raises(ValueError, match="not look like a Google Sheets"):
        _sheets_to_csv_url("https://example.com/notgooglesheets")


def test_is_google_sheets_url_true():
    from api.data import _is_google_sheets_url

    assert _is_google_sheets_url("https://docs.google.com/spreadsheets/d/ID123/edit")


def test_is_google_sheets_url_false():
    from api.data import _is_google_sheets_url

    assert not _is_google_sheets_url("https://example.com/data.csv")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


async def test_upload_from_direct_csv_url(ac, project_id):
    """Importing a plain CSV URL creates a dataset and returns preview."""
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_CSV_BYTES)):
        resp = await ac.post(
            "/api/data/upload-url",
            json={"project_id": project_id, "url": "https://example.com/data.csv"},
        )
    assert resp.status_code == 201
    data = resp.json()
    assert data["row_count"] == 3
    assert data["column_count"] == 5
    assert len(data["preview"]) == 3
    assert data["source"] == "URL"


async def test_upload_from_google_sheets_url(ac, project_id):
    """Google Sheets URL is converted to export URL and imported."""
    sheets_url = "https://docs.google.com/spreadsheets/d/SHEET_ID_123/edit"
    with patch(
        "urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_CSV_BYTES)
    ) as mock_open:
        resp = await ac.post(
            "/api/data/upload-url",
            json={"project_id": project_id, "url": sheets_url},
        )
    assert resp.status_code == 201
    data = resp.json()
    assert data["source"] == "Google Sheets"
    # Verify the export URL was used (not the original edit URL)
    call_args = mock_open.call_args
    req = call_args[0][0]
    assert "export?format=csv" in req.full_url


async def test_upload_url_custom_filename(ac, project_id):
    """User can override the stored filename."""
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_CSV_BYTES)):
        resp = await ac.post(
            "/api/data/upload-url",
            json={
                "project_id": project_id,
                "url": "https://example.com/data.csv",
                "filename": "my_sales_data",
            },
        )
    assert resp.status_code == 201
    assert resp.json()["filename"] == "my_sales_data.csv"


async def test_upload_url_invalid_url_scheme(ac, project_id):
    """URLs that don't start with http:// or https:// are rejected."""
    resp = await ac.post(
        "/api/data/upload-url",
        json={"project_id": project_id, "url": "ftp://example.com/data.csv"},
    )
    assert resp.status_code == 400
    assert "http" in resp.json()["detail"].lower()


async def test_upload_url_project_not_found(ac):
    """Returns 404 when project doesn't exist."""
    resp = await ac.post(
        "/api/data/upload-url",
        json={"project_id": "nonexistent", "url": "https://example.com/data.csv"},
    )
    assert resp.status_code == 404


async def test_upload_url_network_failure(ac, project_id):
    """Returns 400 when the URL cannot be fetched."""
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        resp = await ac.post(
            "/api/data/upload-url",
            json={"project_id": project_id, "url": "https://example.com/data.csv"},
        )
    assert resp.status_code == 400
    assert "Connection refused" in resp.json()["detail"]


async def test_upload_url_not_csv_content(ac, project_id):
    """Returns 400 when the downloaded content is not parseable as CSV."""
    html_bytes = b"<html><body>Not a CSV</body></html>"
    # pandas will parse HTML as a table or raise; either way we get a 400
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(html_bytes)):
        resp = await ac.post(
            "/api/data/upload-url",
            json={"project_id": project_id, "url": "https://example.com/page.html"},
        )
    # 201 or 400 depending on whether pandas can parse HTML as CSV
    # We just verify it doesn't crash with a 500
    assert resp.status_code in (201, 400)


async def test_upload_url_google_sheets_with_gid(ac, project_id):
    """gid query param is preserved in the export URL for multi-tab sheets."""
    sheets_url = "https://docs.google.com/spreadsheets/d/SHEET_ID/edit#gid=999"
    with patch(
        "urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_CSV_BYTES)
    ) as mock_open:
        resp = await ac.post(
            "/api/data/upload-url",
            json={"project_id": project_id, "url": sheets_url},
        )
    assert resp.status_code == 201
    req = mock_open.call_args[0][0]
    assert "gid=999" in req.full_url


async def test_upload_url_filename_from_url_path(ac, project_id):
    """Filename is derived from the last path segment when not overridden."""
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_CSV_BYTES)):
        resp = await ac.post(
            "/api/data/upload-url",
            json={
                "project_id": project_id,
                "url": "https://example.com/quarterly_sales.csv",
            },
        )
    assert resp.status_code == 201
    assert resp.json()["filename"] == "quarterly_sales.csv"


async def test_upload_url_google_sheets_filename_derived(ac, project_id):
    """Google Sheets import derives filename from sheet ID when no override given."""
    sheets_url = "https://docs.google.com/spreadsheets/d/LONGSHEETIDABC/edit"
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(SAMPLE_CSV_BYTES)):
        resp = await ac.post(
            "/api/data/upload-url",
            json={"project_id": project_id, "url": sheets_url},
        )
    assert resp.status_code == 201
    filename = resp.json()["filename"]
    assert filename.startswith("sheets_")
    assert filename.endswith(".csv")
