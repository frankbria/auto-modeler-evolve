"""Tests for project CRUD API."""
import pytest
from httpx import AsyncClient, ASGITransport
from sqlmodel import create_engine, SQLModel
import db as db_module


@pytest.fixture
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.conversation  # noqa
    SQLModel.metadata.create_all(db_module.engine)

    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_health_check(ac):
    resp = await ac.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_create_project(ac):
    resp = await ac.post("/api/projects", json={"name": "Q3 Sales", "description": "Q3 analysis"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Q3 Sales"
    assert "id" in data
    assert data["status"] == "exploring"


async def test_list_projects(ac):
    await ac.post("/api/projects", json={"name": "Project 1"})
    await ac.post("/api/projects", json={"name": "Project 2"})
    resp = await ac.get("/api/projects")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_get_project(ac):
    create_resp = await ac.post("/api/projects", json={"name": "My Project"})
    project_id = create_resp.json()["id"]
    resp = await ac.get(f"/api/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "My Project"


async def test_get_project_not_found(ac):
    resp = await ac.get("/api/projects/nonexistent-id")
    assert resp.status_code == 404


async def test_delete_project(ac):
    create_resp = await ac.post("/api/projects", json={"name": "Temp Project"})
    project_id = create_resp.json()["id"]
    del_resp = await ac.delete(f"/api/projects/{project_id}")
    assert del_resp.status_code == 204
    get_resp = await ac.get(f"/api/projects/{project_id}")
    assert get_resp.status_code == 404
