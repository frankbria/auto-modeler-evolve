"""Tests for auto-retrain feature.

Covers:
- _AUTO_RETRAIN_PATTERNS detection in chat.py
- GET/PUT /api/projects/{project_id}/auto-retrain endpoints
- trigger_auto_retrain() core function
- Upload endpoint response includes auto_retrain field
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel

# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_auto_retrain_pattern_enable():
    from api.chat import _AUTO_RETRAIN_PATTERNS

    assert _AUTO_RETRAIN_PATTERNS.search("enable auto-retrain for this project")
    assert _AUTO_RETRAIN_PATTERNS.search("turn on auto retrain")
    assert _AUTO_RETRAIN_PATTERNS.search("auto retrain when I upload new data")


def test_auto_retrain_pattern_disable():
    from api.chat import _AUTO_RETRAIN_PATTERNS

    assert _AUTO_RETRAIN_PATTERNS.search("disable auto-retrain")
    assert _AUTO_RETRAIN_PATTERNS.search("turn off auto retrain")


def test_auto_retrain_pattern_status():
    from api.chat import _AUTO_RETRAIN_PATTERNS

    assert _AUTO_RETRAIN_PATTERNS.search("what is the auto-retrain status?")
    assert _AUTO_RETRAIN_PATTERNS.search("is auto retrain enabled?")
    assert _AUTO_RETRAIN_PATTERNS.search("retrain automatically when upload")


def test_auto_retrain_pattern_fresh():
    from api.chat import _AUTO_RETRAIN_PATTERNS

    assert _AUTO_RETRAIN_PATTERNS.search("keep model fresh with new data")
    assert _AUTO_RETRAIN_PATTERNS.search("keep model current automatically")


def test_auto_retrain_pattern_no_match():
    from api.chat import _AUTO_RETRAIN_PATTERNS

    assert not _AUTO_RETRAIN_PATTERNS.search("show me a histogram")
    assert not _AUTO_RETRAIN_PATTERNS.search("what is the model accuracy?")
    assert not _AUTO_RETRAIN_PATTERNS.search("deploy the model")


# ---------------------------------------------------------------------------
# API fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


async def _make_project(tmp_path, project_id: str):
    import db
    from models.project import Project
    from sqlmodel import create_engine

    test_db = str(tmp_path / f"{project_id}.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)
    with next(db.get_session()) as session:
        proj = Project(id=project_id, name="Test Project")
        session.merge(proj)
        session.commit()
    return project_id


# ---------------------------------------------------------------------------
# GET /api/projects/{project_id}/auto-retrain
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_auto_retrain_default_disabled(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_get.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project(tmp_path, "ar-get-1")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/projects/{project_id}/auto-retrain")

    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False
    assert data["project_id"] == project_id


@pytest.mark.anyio
async def test_get_auto_retrain_not_found(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_404.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/projects/no-such-id/auto-retrain")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/projects/{project_id}/auto-retrain
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_auto_retrain_enable(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_put.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project(tmp_path, "ar-put-1")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.put(
            f"/api/projects/{project_id}/auto-retrain",
            json={"enabled": True},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert "enabled" in data["message"].lower()


@pytest.mark.anyio
async def test_put_auto_retrain_disable(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_put2.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project(tmp_path, "ar-put-2")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Enable first
        await client.put(
            f"/api/projects/{project_id}/auto-retrain",
            json={"enabled": True},
        )
        # Then disable
        resp = await client.put(
            f"/api/projects/{project_id}/auto-retrain",
            json={"enabled": False},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False


@pytest.mark.anyio
async def test_put_auto_retrain_persists(tmp_path, set_test_env):
    """Enable auto-retrain and verify GET returns updated state."""
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_persist.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project(tmp_path, "ar-persist-1")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.put(
            f"/api/projects/{project_id}/auto-retrain",
            json={"enabled": True},
        )
        resp = await client.get(f"/api/projects/{project_id}/auto-retrain")

    assert resp.status_code == 200
    assert resp.json()["enabled"] is True


@pytest.mark.anyio
async def test_put_auto_retrain_not_found(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_put404.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.put(
            "/api/projects/no-such-id/auto-retrain",
            json={"enabled": True},
        )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# trigger_auto_retrain() — skips gracefully when no selected model
# ---------------------------------------------------------------------------


def test_trigger_auto_retrain_no_selected_model(tmp_path):
    """trigger_auto_retrain returns None when no selected model exists."""
    import db
    from models.project import Project
    from sqlmodel import create_engine, SQLModel

    test_db = str(tmp_path / "ar_core.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    with next(db.get_session()) as session:
        proj = Project(id="ar-core-1", name="Core Test")
        session.add(proj)
        session.commit()

    from core.retrain import trigger_auto_retrain

    result = trigger_auto_retrain("ar-core-1", "some-dataset-id")
    assert result is None


def test_trigger_auto_retrain_nonexistent_project(tmp_path):
    """trigger_auto_retrain returns None (does not raise) for missing project."""
    import db
    from sqlmodel import create_engine, SQLModel

    test_db = str(tmp_path / "ar_core2.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    from core.retrain import trigger_auto_retrain

    result = trigger_auto_retrain("no-such-project", "no-such-dataset")
    assert result is None


# ---------------------------------------------------------------------------
# Upload endpoint includes auto_retrain field in response
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_upload_response_includes_auto_retrain_field(tmp_path, set_test_env):
    """Upload response always has 'auto_retrain' key (None when not triggered)."""
    import io
    from main import app
    import db
    from models.project import Project
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_upload.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    with next(db.get_session()) as session:
        proj = Project(id="ar-upload-1", name="Upload Test")
        session.add(proj)
        session.commit()

    csv_data = b"col_a,col_b\n1,2\n3,4\n5,6\n"

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/data/upload",
            data={"project_id": "ar-upload-1"},
            files={"file": ("test.csv", io.BytesIO(csv_data), "text/csv")},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert "auto_retrain" in data
