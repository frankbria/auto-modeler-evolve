"""Tests for the SQLite database connector endpoints.

POST /api/data/upload-db   — upload a .db file, list tables
POST /api/data/extract-db  — extract table/query as Dataset
"""

import io
import sqlite3
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module


def _make_sqlite_db(tmp_path: Path) -> bytes:
    """Create a minimal SQLite database with one table and return its bytes."""
    db_path = tmp_path / "_test_input.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE sales (date TEXT, product TEXT, region TEXT, revenue REAL, units INTEGER)"
    )
    conn.executemany(
        "INSERT INTO sales VALUES (?, ?, ?, ?, ?)",
        [
            ("2024-01-01", "Widget A", "North", 1200.50, 10),
            ("2024-01-01", "Widget B", "South", 850.00, 8),
            ("2024-01-02", "Widget A", "East", 2100.75, 18),
            ("2024-01-02", "Widget C", "West", 450.25, 4),
        ],
    )
    conn.execute("CREATE TABLE inventory (sku TEXT, quantity INTEGER, location TEXT)")
    conn.executemany(
        "INSERT INTO inventory VALUES (?, ?, ?)",
        [("SKU-001", 100, "Warehouse A"), ("SKU-002", 50, "Warehouse B")],
    )
    conn.commit()
    conn.close()
    return db_path.read_bytes()


@pytest.fixture
async def ac(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    data_module._DB_UPLOADS_DIR = tmp_path / "db_uploads"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def project_id(ac):
    r = await ac.post("/api/projects", json={"name": "DB Connector Test"})
    assert r.status_code == 201
    return r.json()["id"]


@pytest.fixture
async def db_bytes(tmp_path):
    return _make_sqlite_db(tmp_path)


# ===========================================================================
# upload-db tests
# ===========================================================================


class TestUploadDb:
    @pytest.mark.asyncio
    async def test_upload_valid_sqlite_lists_tables(self, ac, project_id, db_bytes):
        """Uploading a valid SQLite file returns the list of tables."""
        r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": ("data.db", io.BytesIO(db_bytes), "application/octet-stream")
            },
        )
        assert r.status_code == 201
        data = r.json()
        assert "tables" in data
        assert set(data["tables"]) == {"sales", "inventory"}
        assert data["table_count"] == 2
        assert "db_path" in data

    @pytest.mark.asyncio
    async def test_upload_invalid_project_returns_404(self, ac, db_bytes):
        r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": "nonexistent"},
            files={
                "file": ("data.db", io.BytesIO(db_bytes), "application/octet-stream")
            },
        )
        assert r.status_code == 404
        assert "Project not found" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_wrong_extension_returns_400(self, ac, project_id):
        r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={"file": ("data.csv", io.BytesIO(b"a,b\n1,2"), "text/csv")},
        )
        assert r.status_code == 400
        assert "SQLite database files" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_corrupted_file_returns_400(self, ac, project_id):
        r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": (
                    "corrupt.db",
                    io.BytesIO(b"not a sqlite database"),
                    "application/octet-stream",
                )
            },
        )
        assert r.status_code == 400
        assert "valid SQLite database" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_empty_db_returns_400(self, ac, project_id, tmp_path):
        """SQLite file with no tables returns 400."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()
        r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": (
                    "empty.db",
                    io.BytesIO(db_path.read_bytes()),
                    "application/octet-stream",
                )
            },
        )
        assert r.status_code == 400
        assert "no tables" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_sqlite3_extension_accepted(
        self, ac, project_id, db_bytes, tmp_path
    ):
        """Files with .sqlite3 extension are accepted."""
        r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": (
                    "data.sqlite3",
                    io.BytesIO(db_bytes),
                    "application/octet-stream",
                )
            },
        )
        assert r.status_code == 201
        assert len(r.json()["tables"]) > 0


# ===========================================================================
# extract-db tests
# ===========================================================================


class TestExtractDb:
    @pytest.mark.asyncio
    async def test_extract_table_creates_dataset(self, ac, project_id, db_bytes):
        """Extracting a table creates a Dataset and returns preview."""
        # Upload first
        upload_r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": ("data.db", io.BytesIO(db_bytes), "application/octet-stream")
            },
        )
        db_path = upload_r.json()["db_path"]

        r = await ac.post(
            "/api/data/extract-db",
            json={"project_id": project_id, "db_path": db_path, "table_name": "sales"},
        )
        assert r.status_code == 201
        data = r.json()
        assert "dataset_id" in data
        assert data["row_count"] == 4
        assert data["column_count"] == 5
        assert data["source"] == "SQLite"
        assert data["table_name"] == "sales"
        assert len(data["preview"]) == 4

    @pytest.mark.asyncio
    async def test_extract_with_custom_query(self, ac, project_id, db_bytes):
        """Custom SELECT query filters rows correctly."""
        upload_r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": ("data.db", io.BytesIO(db_bytes), "application/octet-stream")
            },
        )
        db_path = upload_r.json()["db_path"]

        r = await ac.post(
            "/api/data/extract-db",
            json={
                "project_id": project_id,
                "db_path": db_path,
                "table_name": "sales",
                "query": "SELECT product, SUM(revenue) as total FROM sales GROUP BY product",
            },
        )
        assert r.status_code == 201
        data = r.json()
        # 3 products: Widget A, Widget B, Widget C
        assert data["row_count"] == 3
        assert data["column_count"] == 2

    @pytest.mark.asyncio
    async def test_extract_project_not_found(self, ac, db_bytes):
        r = await ac.post(
            "/api/data/extract-db",
            json={
                "project_id": "bad-id",
                "db_path": "/some/path.db",
                "table_name": "sales",
            },
        )
        assert r.status_code == 404
        assert "Project not found" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_extract_db_file_not_found(self, ac, project_id):
        r = await ac.post(
            "/api/data/extract-db",
            json={
                "project_id": project_id,
                "db_path": "/nonexistent/path.db",
                "table_name": "sales",
            },
        )
        assert r.status_code == 404
        assert "not found" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_extract_non_select_query_rejected(self, ac, project_id, db_bytes):
        """Non-SELECT queries (DROP, INSERT, etc.) are rejected."""
        upload_r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": ("data.db", io.BytesIO(db_bytes), "application/octet-stream")
            },
        )
        db_path = upload_r.json()["db_path"]

        r = await ac.post(
            "/api/data/extract-db",
            json={
                "project_id": project_id,
                "db_path": db_path,
                "table_name": "sales",
                "query": "DROP TABLE sales",
            },
        )
        assert r.status_code == 400
        assert "SELECT" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_extract_invalid_query_returns_400(self, ac, project_id, db_bytes):
        """Query referencing nonexistent column returns 400."""
        upload_r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": ("data.db", io.BytesIO(db_bytes), "application/octet-stream")
            },
        )
        db_path = upload_r.json()["db_path"]

        r = await ac.post(
            "/api/data/extract-db",
            json={
                "project_id": project_id,
                "db_path": db_path,
                "table_name": "sales",
                "query": "SELECT nonexistent_column FROM sales",
            },
        )
        assert r.status_code == 400
        assert "Query failed" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_extract_empty_result_returns_400(self, ac, project_id, db_bytes):
        """Query returning zero rows returns 400."""
        upload_r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": ("data.db", io.BytesIO(db_bytes), "application/octet-stream")
            },
        )
        db_path = upload_r.json()["db_path"]

        r = await ac.post(
            "/api/data/extract-db",
            json={
                "project_id": project_id,
                "db_path": db_path,
                "table_name": "sales",
                "query": "SELECT * FROM sales WHERE revenue > 999999",
            },
        )
        assert r.status_code == 400
        assert "no rows" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_extract_custom_filename(self, ac, project_id, db_bytes):
        """Custom save_as_filename is used for the Dataset file."""
        upload_r = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": ("data.db", io.BytesIO(db_bytes), "application/octet-stream")
            },
        )
        db_path = upload_r.json()["db_path"]

        r = await ac.post(
            "/api/data/extract-db",
            json={
                "project_id": project_id,
                "db_path": db_path,
                "table_name": "inventory",
                "save_as_filename": "warehouse_stock",
            },
        )
        assert r.status_code == 201
        data = r.json()
        assert "warehouse_stock" in data["filename"]
        assert data["row_count"] == 2
        assert data["column_count"] == 3
