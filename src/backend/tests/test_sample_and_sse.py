"""Tests for sample dataset loading and SSE training stream."""

import pytest
from httpx import AsyncClient, ASGITransport
from sqlmodel import create_engine, SQLModel

import db as db_module


@pytest.fixture
async def ac(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.conversation  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.deployment  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    data_module.UPLOAD_DIR = upload_dir

    # Point sample CSV at a real temp file for tests
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    sample_csv = sample_dir / "sample_sales.csv"
    sample_csv.write_text(
        "date,product,region,revenue,units\n"
        "2024-01-01,Widget A,North,1200.50,10\n"
        "2024-01-02,Widget B,South,850.00,8\n"
        "2024-01-03,Widget A,East,2100.75,18\n"
        "2024-01-04,Widget C,West,450.25,4\n"
        "2024-01-05,Widget B,North,1650.00,15\n"
    )
    data_module.SAMPLE_CSV = sample_csv

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Test Project"})
    assert resp.status_code in (200, 201)
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Sample dataset info
# ---------------------------------------------------------------------------


class TestSampleInfo:
    async def test_sample_info_returns_metadata(self, ac, tmp_path):
        resp = await ac.get("/api/data/sample/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "sample_sales.csv"
        assert data["row_count"] == 5
        assert data["column_count"] == 5
        assert "revenue" in data["columns"]
        assert "description" in data

    async def test_sample_info_lists_all_columns(self, ac, tmp_path):
        resp = await ac.get("/api/data/sample/info")
        cols = resp.json()["columns"]
        assert set(cols) == {"date", "product", "region", "revenue", "units"}


# ---------------------------------------------------------------------------
# Sample dataset load
# ---------------------------------------------------------------------------


class TestSampleLoad:
    async def test_load_sample_creates_dataset(self, ac, project_id):
        resp = await ac.post("/api/data/sample", json={"project_id": project_id})
        assert resp.status_code == 201
        data = resp.json()
        assert data["dataset_id"]
        assert data["filename"] == "sample_sales.csv"
        assert data["row_count"] == 5
        assert data["column_count"] == 5
        assert len(data["preview"]) > 0

    async def test_load_sample_idempotent(self, ac, project_id):
        """Loading twice returns the existing dataset without error."""
        r1 = await ac.post("/api/data/sample", json={"project_id": project_id})
        r2 = await ac.post("/api/data/sample", json={"project_id": project_id})
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["dataset_id"] == r2.json()["dataset_id"]
        assert r2.json()["already_existed"] is True

    async def test_load_sample_unknown_project_returns_404(self, ac):
        resp = await ac.post("/api/data/sample", json={"project_id": "nonexistent-id"})
        assert resp.status_code == 404

    async def test_load_sample_file_copied_to_uploads(self, ac, project_id, tmp_path):
        resp = await ac.post("/api/data/sample", json={"project_id": project_id})
        assert resp.status_code == 201
        upload_dir = tmp_path / "uploads" / project_id
        assert (upload_dir / "sample_sales.csv").exists()

    async def test_load_sample_returns_column_stats(self, ac, project_id):
        resp = await ac.post("/api/data/sample", json={"project_id": project_id})
        data = resp.json()
        assert "column_stats" in data
        assert len(data["column_stats"]) == 5


# ---------------------------------------------------------------------------
# SSE training stream
# ---------------------------------------------------------------------------


class TestTrainingStream:
    async def test_stream_returns_all_done_if_no_queue(self, ac, project_id):
        """When no training is in progress, stream immediately emits all_done."""
        import api.models as models_module

        models_module._training_queues.pop(project_id, None)

        resp = await ac.get(f"/api/models/{project_id}/training-stream")
        assert resp.status_code == 200
        assert b"all_done" in resp.content

    async def test_stream_delivers_events_from_queue(self, ac, project_id):
        """Events placed in the queue are delivered over SSE."""
        import queue as q_module
        import api.models as models_module

        test_queue = q_module.Queue()
        test_queue.put(
            {
                "type": "status",
                "run_id": "r1",
                "status": "training",
                "algorithm": "linear_regression",
            }
        )
        test_queue.put(None)  # sentinel

        models_module._training_queues[project_id] = test_queue

        resp = await ac.get(f"/api/models/{project_id}/training-stream")
        assert resp.status_code == 200
        body = resp.content.decode()

        assert "training" in body
        assert "all_done" in body

    async def test_stream_cleans_up_queue_after_sentinel(self, ac, project_id):
        """Queue is removed from dict after stream finishes."""
        import queue as q_module
        import api.models as models_module

        test_queue = q_module.Queue()
        test_queue.put(None)  # immediate sentinel
        models_module._training_queues[project_id] = test_queue

        await ac.get(f"/api/models/{project_id}/training-stream")

        assert project_id not in models_module._training_queues
