"""Tests for:
- POST /api/data/{id}/rename-column endpoint
- _HEATMAP_PATTERNS and _detect_rename_request in chat.py
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import (
    _HEATMAP_PATTERNS,
    _RENAME_PATTERNS,
    _detect_rename_request,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units,cost\n"
    b"East,100,10,50\n"
    b"West,200,20,80\n"
    b"East,150,15,60\n"
    b"West,300,30,120\n"
    b"North,250,25,100\n"
)


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def dataset_id(ac):
    proj_resp = await ac.post("/api/projects", json={"name": "Rename Test"})
    project_id = proj_resp.json()["id"]
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Unit tests — _detect_rename_request
# ---------------------------------------------------------------------------


def test_detect_rename_basic():
    cols = ["revenue_usd", "region", "units"]
    result = _detect_rename_request("rename revenue_usd to Revenue", cols)
    assert result is not None
    assert result["old_name"] == "revenue_usd"
    assert result["new_name"] == "Revenue"


def test_detect_rename_with_column_keyword():
    cols = ["rev_q1_adj", "cost"]
    result = _detect_rename_request("rename column rev_q1_adj to Q1Revenue", cols)
    assert result is not None
    assert result["old_name"] == "rev_q1_adj"
    assert result["new_name"] == "Q1Revenue"


def test_detect_rename_case_insensitive():
    cols = ["Revenue_USD", "units"]
    result = _detect_rename_request("rename revenue_usd to Sales", cols)
    # Should match case-insensitively against actual column name
    assert result is not None
    assert result["old_name"] == "Revenue_USD"


def test_detect_rename_change_name_of():
    cols = ["old_col", "region"]
    result = _detect_rename_request("change the name of old_col to new_col", cols)
    assert result is not None
    assert result["old_name"] == "old_col"
    assert result["new_name"] == "new_col"


def test_detect_rename_nonexistent_column():
    """Returns None if old_name doesn't match any column."""
    cols = ["revenue", "region"]
    result = _detect_rename_request("rename nonexistent to foo", cols)
    assert result is None


def test_detect_rename_no_match():
    """Returns None if message doesn't match the rename pattern."""
    cols = ["revenue", "region"]
    result = _detect_rename_request("show me the data", cols)
    assert result is None


# ---------------------------------------------------------------------------
# Unit tests — _HEATMAP_PATTERNS
# ---------------------------------------------------------------------------


def test_heatmap_pattern_correlation_matrix():
    assert _HEATMAP_PATTERNS.search("show me the correlation matrix")


def test_heatmap_pattern_heatmap():
    assert _HEATMAP_PATTERNS.search("show the heatmap")


def test_heatmap_pattern_heatmap_bare():
    assert _HEATMAP_PATTERNS.search("heatmap please")


def test_heatmap_pattern_how_related():
    assert _HEATMAP_PATTERNS.search("how are my columns related")


def test_heatmap_pattern_how_correlated():
    assert _HEATMAP_PATTERNS.search("how are my variables correlated")


def test_heatmap_pattern_show_correlations_between():
    assert _HEATMAP_PATTERNS.search("show me the correlations between all my columns")


def test_heatmap_pattern_full_correlation():
    assert _HEATMAP_PATTERNS.search("full correlation please")


def test_heatmap_pattern_no_match():
    assert not _HEATMAP_PATTERNS.search("what drives revenue")


def test_heatmap_pattern_rename_no_match():
    assert not _HEATMAP_PATTERNS.search("rename revenue to sales")


# ---------------------------------------------------------------------------
# Unit tests — _RENAME_PATTERNS
# ---------------------------------------------------------------------------


def test_rename_pattern_basic():
    assert _RENAME_PATTERNS.search("rename revenue_usd to Revenue")


def test_rename_pattern_with_column():
    assert _RENAME_PATTERNS.search("rename column rev_q1 to Q1 Revenue")


def test_rename_pattern_change_name():
    assert _RENAME_PATTERNS.search("change the column name of sales to total_sales")


def test_rename_pattern_no_match():
    assert not _RENAME_PATTERNS.search("show me the data")


# ---------------------------------------------------------------------------
# Endpoint tests — POST /api/data/{id}/rename-column
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_rename_column_success(ac, dataset_id):
    resp = await ac.post(
        f"/api/data/{dataset_id}/rename-column",
        json={"old_name": "revenue", "new_name": "Sales"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["old_name"] == "revenue"
    assert data["new_name"] == "Sales"
    assert data["column_count"] == 4
    assert data["row_count"] == 5


@pytest.mark.anyio
async def test_rename_column_persists(ac, dataset_id):
    """After rename, the new column appears in the dataset profile."""
    await ac.post(
        f"/api/data/{dataset_id}/rename-column",
        json={"old_name": "units", "new_name": "quantity"},
    )
    # Profile should now show 'quantity', not 'units'
    profile_resp = await ac.get(f"/api/data/{dataset_id}/profile")
    assert profile_resp.status_code == 200, profile_resp.text
    profile = profile_resp.json()
    col_names = [c["name"] for c in profile.get("columns", [])]
    assert "quantity" in col_names
    assert "units" not in col_names


@pytest.mark.anyio
async def test_rename_column_not_found(ac, dataset_id):
    resp = await ac.post(
        f"/api/data/{dataset_id}/rename-column",
        json={"old_name": "nonexistent", "new_name": "foo"},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_rename_column_conflict(ac, dataset_id):
    """Cannot rename to a name that already exists."""
    resp = await ac.post(
        f"/api/data/{dataset_id}/rename-column",
        json={"old_name": "revenue", "new_name": "units"},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_rename_column_invalid_name(ac, dataset_id):
    """Name with spaces or special chars is rejected."""
    resp = await ac.post(
        f"/api/data/{dataset_id}/rename-column",
        json={"old_name": "revenue", "new_name": "my column"},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_rename_column_empty_name(ac, dataset_id):
    resp = await ac.post(
        f"/api/data/{dataset_id}/rename-column",
        json={"old_name": "revenue", "new_name": "   "},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_rename_column_dataset_not_found(ac):
    resp = await ac.post(
        "/api/data/nonexistent-id/rename-column",
        json={"old_name": "revenue", "new_name": "Sales"},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_rename_column_same_name_ok(ac, dataset_id):
    """Renaming to the same name (no-op) should succeed."""
    resp = await ac.post(
        f"/api/data/{dataset_id}/rename-column",
        json={"old_name": "revenue", "new_name": "revenue"},
    )
    assert resp.status_code == 200
