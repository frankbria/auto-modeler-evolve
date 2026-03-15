"""Test configuration and shared fixtures."""
import pytest
from httpx import AsyncClient, ASGITransport
from sqlmodel import create_engine, SQLModel


@pytest.fixture(autouse=True)
def set_test_env(tmp_path, monkeypatch):
    """Use temp directory for all file operations in tests."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture
async def client(tmp_path, set_test_env):
    """Async HTTP client that hits the FastAPI app directly."""
    import db
    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.conversation  # noqa
    import models.model_run  # noqa
    import models.deployment  # noqa
    import models.prediction_log  # noqa
    from main import app

    test_db = str(tmp_path / "test.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db.DATA_DIR = tmp_path
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_csv_content():
    """A small CSV representing sales data."""
    return b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-01,Widget B,South,850.00,8
2024-01-02,Widget A,East,2100.75,18
2024-01-02,Widget C,West,450.25,4
2024-01-03,Widget B,North,1650.00,15
"""
