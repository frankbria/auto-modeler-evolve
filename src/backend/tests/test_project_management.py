"""Tests for Phase 7 project management features: rename, duplicate, quick stats."""

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
    import models.model_run  # noqa
    import models.feature_set  # noqa
    import models.deployment  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


async def test_rename_project(ac):
    resp = await ac.post("/api/projects", json={"name": "Original Name"})
    project_id = resp.json()["id"]

    patch_resp = await ac.patch(
        f"/api/projects/{project_id}", json={"name": "New Name"}
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["name"] == "New Name"

    # Verify get returns updated name
    get_resp = await ac.get(f"/api/projects/{project_id}")
    assert get_resp.json()["name"] == "New Name"


async def test_rename_project_not_found(ac):
    resp = await ac.patch("/api/projects/nonexistent", json={"name": "X"})
    assert resp.status_code == 404


async def test_rename_updates_description(ac):
    resp = await ac.post("/api/projects", json={"name": "My Project"})
    project_id = resp.json()["id"]

    patch_resp = await ac.patch(
        f"/api/projects/{project_id}",
        json={"name": "My Project", "description": "Updated desc"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["description"] == "Updated desc"


async def test_duplicate_project(ac):
    # Create original
    resp = await ac.post(
        "/api/projects", json={"name": "Original", "description": "test desc"}
    )
    project_id = resp.json()["id"]

    # Duplicate
    dup_resp = await ac.post(f"/api/projects/{project_id}/duplicate")
    assert dup_resp.status_code == 201
    dup = dup_resp.json()
    assert dup["name"] == "Original (copy)"
    assert dup["description"] == "test desc"
    assert dup["id"] != project_id
    assert dup["status"] == "exploring"


async def test_duplicate_project_not_found(ac):
    resp = await ac.post("/api/projects/nonexistent/duplicate")
    assert resp.status_code == 404


async def test_list_projects_with_stats(ac):
    resp = await ac.post("/api/projects", json={"name": "Stats Test"})
    assert resp.status_code == 201

    list_resp = await ac.get("/api/projects")
    assert list_resp.status_code == 200
    projects = list_resp.json()
    assert len(projects) >= 1
    project = next(p for p in projects if p["name"] == "Stats Test")

    # Stats fields should be present
    assert "model_count" in project
    assert "has_deployment" in project
    assert project["model_count"] == 0
    assert project["has_deployment"] is False
    # No dataset yet
    assert project["dataset_filename"] is None


async def test_partial_update_name_only(ac):
    """PATCH with only name should not touch description."""
    resp = await ac.post(
        "/api/projects", json={"name": "Project", "description": "Keep this"}
    )
    project_id = resp.json()["id"]

    patch_resp = await ac.patch(f"/api/projects/{project_id}", json={"name": "Renamed"})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["name"] == "Renamed"
    # Description should still be present (backend doesn't clear on None)
    get_resp = await ac.get(f"/api/projects/{project_id}")
    assert get_resp.json()["description"] == "Keep this"
