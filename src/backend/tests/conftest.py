"""Test configuration and shared fixtures."""
import pytest
import tempfile
import os
from httpx import AsyncClient, ASGITransport
from sqlmodel import create_engine, Session, SQLModel

# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite:///./test_automodeler.db"


@pytest.fixture(autouse=True)
def set_test_env(tmp_path, monkeypatch):
    """Use temp directory for all file operations in tests."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    yield
    # Cleanup test DB if exists
    if os.path.exists("test_automodeler.db"):
        os.unlink("test_automodeler.db")


@pytest.fixture
async def client(set_test_env):
    """Async HTTP client that hits the FastAPI app directly."""
    # Import after env is set so db.py picks up the temp dir
    import db
    from main import app

    # Re-init DB with test path
    db.engine = create_engine(TEST_DATABASE_URL, echo=False)
    db.DATA_DIR = set_test_env if isinstance(set_test_env, str) else "./test_data"
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
