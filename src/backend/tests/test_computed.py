"""Tests for computed column feature.

Covers:
- add_computed_column() unit tests (core/computed.py)
- preview_computed_column() unit tests
- POST /api/data/{dataset_id}/compute endpoint
- _detect_compute_request() helper in api/chat.py
"""

import io
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.computed import add_computed_column, preview_computed_column

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_CSV = (
    b"product,region,revenue,cost,units\n"
    b"Widget A,North,1200.0,800.0,10\n"
    b"Widget B,South,850.0,600.0,8\n"
    b"Widget A,East,2100.0,1400.0,18\n"
    b"Widget C,West,450.0,300.0,4\n"
    b"Widget B,North,1650.0,1100.0,15\n"
)


def make_df():
    return pd.read_csv(io.BytesIO(SAMPLE_CSV))


# ---------------------------------------------------------------------------
# Unit tests: add_computed_column()
# ---------------------------------------------------------------------------


class TestAddComputedColumn:
    def test_basic_division(self):
        df = make_df()
        result_df, info = add_computed_column(df, "margin", "revenue - cost")
        assert "margin" in result_df.columns
        assert info["column_name"] == "margin"
        assert info["action"] == "added"
        # Widget A North: 1200 - 800 = 400
        assert abs(result_df.loc[0, "margin"] - 400.0) < 0.01

    def test_ratio_expression(self):
        df = make_df()
        result_df, info = add_computed_column(df, "cost_ratio", "cost / revenue")
        assert "cost_ratio" in result_df.columns
        # Widget A North: 800/1200 ≈ 0.667
        assert abs(result_df.loc[0, "cost_ratio"] - 800 / 1200) < 0.001

    def test_original_columns_preserved(self):
        df = make_df()
        original_cols = list(df.columns)
        result_df, _ = add_computed_column(df, "new_col", "revenue * 2")
        for col in original_cols:
            assert col in result_df.columns

    def test_result_summary_mentions_column_name(self):
        df = make_df()
        _, info = add_computed_column(df, "profit", "revenue - cost")
        assert "profit" in info["summary"]

    def test_sample_values_returned(self):
        df = make_df()
        _, info = add_computed_column(df, "rev_per_unit", "revenue / units")
        assert isinstance(info["sample_values"], list)
        assert len(info["sample_values"]) > 0

    def test_invalid_column_name_raises(self):
        df = make_df()
        with pytest.raises(ValueError, match="Invalid column name"):
            add_computed_column(df, "123invalid", "revenue + cost")

    def test_invalid_column_name_with_space_raises(self):
        df = make_df()
        with pytest.raises(ValueError, match="Invalid column name"):
            add_computed_column(df, "my column", "revenue + cost")

    def test_invalid_expression_raises(self):
        df = make_df()
        with pytest.raises(ValueError, match="Invalid expression"):
            add_computed_column(df, "bad", "nonexistent_col + 1")

    def test_empty_expression_raises(self):
        df = make_df()
        with pytest.raises(ValueError, match="empty"):
            add_computed_column(df, "test", "   ")

    def test_update_existing_column(self):
        df = make_df()
        result_df, info = add_computed_column(df, "revenue", "revenue * 2")
        assert info["action"] == "updated"
        assert abs(result_df.loc[0, "revenue"] - 2400.0) < 0.01

    def test_dtype_returned(self):
        df = make_df()
        _, info = add_computed_column(df, "ratio", "revenue / cost")
        assert "dtype" in info
        assert info["dtype"].startswith("float")

    def test_row_count_preserved(self):
        df = make_df()
        result_df, info = add_computed_column(df, "total", "revenue + cost")
        assert len(result_df) == len(df)
        assert info["row_count"] == len(df)


# ---------------------------------------------------------------------------
# Unit tests: preview_computed_column()
# ---------------------------------------------------------------------------


class TestPreviewComputedColumn:
    def test_preview_returns_sample(self):
        df = make_df()
        preview = preview_computed_column(df, "margin", "revenue - cost")
        assert preview["column_name"] == "margin"
        assert len(preview["sample_values"]) <= 5

    def test_preview_does_not_modify_df(self):
        df = make_df()
        original_cols = list(df.columns)
        preview_computed_column(df, "margin", "revenue - cost")
        assert list(df.columns) == original_cols

    def test_preview_invalid_expression_raises(self):
        df = make_df()
        with pytest.raises(ValueError):
            preview_computed_column(df, "bad", "nonexistent + 1")


# ---------------------------------------------------------------------------
# API tests: POST /api/data/{dataset_id}/compute
# ---------------------------------------------------------------------------


@pytest.fixture
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


@pytest.fixture
async def dataset_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Compute Test"})
    project_id = resp.json()["id"]
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("data.csv", SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    return resp.json()["dataset_id"]


class TestComputeEndpoint:
    async def test_basic_compute(self, dataset_id, ac):
        resp = await ac.post(
            f"/api/data/{dataset_id}/compute",
            json={"name": "margin", "expression": "revenue - cost"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["compute_result"]["column_name"] == "margin"
        assert data["compute_result"]["action"] == "added"

    async def test_preview_returned(self, dataset_id, ac):
        resp = await ac.post(
            f"/api/data/{dataset_id}/compute",
            json={"name": "rev_per_unit", "expression": "revenue / units"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["preview"], list)
        assert len(data["preview"]) > 0

    async def test_updated_stats_returned(self, dataset_id, ac):
        resp = await ac.post(
            f"/api/data/{dataset_id}/compute",
            json={"name": "profit_pct", "expression": "(revenue - cost) / revenue"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Original had 5 columns; now should have 6
        assert data["updated_stats"]["column_count"] == 6

    async def test_404_unknown_dataset(self, ac):
        resp = await ac.post(
            "/api/data/00000000-0000-0000-0000-000000000000/compute",
            json={"name": "margin", "expression": "revenue - cost"},
        )
        assert resp.status_code == 404

    async def test_400_invalid_expression(self, dataset_id, ac):
        resp = await ac.post(
            f"/api/data/{dataset_id}/compute",
            json={"name": "bad", "expression": "nonexistent_col + 1"},
        )
        assert resp.status_code == 400

    async def test_400_invalid_column_name(self, dataset_id, ac):
        resp = await ac.post(
            f"/api/data/{dataset_id}/compute",
            json={"name": "123bad", "expression": "revenue + cost"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Unit tests: _detect_compute_request()
# ---------------------------------------------------------------------------


class TestDetectComputeRequest:
    def setup_method(self):
        from api.chat import _detect_compute_request

        self._detect = _detect_compute_request

    def test_detects_add_column_pattern(self):
        cols = ["revenue", "cost", "units"]
        result = self._detect("add a column called margin = revenue - cost", cols)
        assert result is not None
        assert result["name"] == "margin"
        assert "revenue" in result["expression"]

    def test_detects_create_field_pattern(self):
        cols = ["revenue", "cost"]
        result = self._detect("create profit as revenue - cost", cols)
        assert result is not None
        assert result["name"] == "profit"

    def test_detects_calculate_pattern(self):
        cols = ["revenue", "units"]
        result = self._detect("calculate revenue_per_unit as revenue / units", cols)
        assert result is not None
        assert result["name"] == "revenue_per_unit"

    def test_returns_none_no_column_match(self):
        cols = ["revenue", "cost"]
        result = self._detect("add a column called x = nonexistent + something", cols)
        assert result is None

    def test_returns_none_for_unrelated_message(self):
        cols = ["revenue", "cost"]
        result = self._detect("how many rows do I have?", cols)
        assert result is None
