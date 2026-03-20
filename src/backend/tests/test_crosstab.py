"""Tests for pivot table / cross-tabulation feature.

Covers:
- build_crosstab() unit tests (core/chart_builder.py)
- GET /api/data/{dataset_id}/crosstab endpoint
- _detect_crosstab_request() helper in api/chat.py
"""

import io
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.chart_builder import build_crosstab


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_CSV = (
    b"product,region,revenue,units\n"
    b"Widget A,North,1200.0,10\n"
    b"Widget B,South,850.0,8\n"
    b"Widget A,East,2100.0,18\n"
    b"Widget C,West,450.0,4\n"
    b"Widget B,North,1650.0,15\n"
    b"Widget A,South,980.0,9\n"
    b"Widget C,East,3200.0,28\n"
    b"Widget B,West,760.0,7\n"
    b"Widget A,North,500.0,5\n"
    b"Widget C,South,1100.0,11\n"
)


def make_df():
    return pd.read_csv(io.BytesIO(SAMPLE_CSV))


# ---------------------------------------------------------------------------
# Unit tests: build_crosstab()
# ---------------------------------------------------------------------------


class TestBuildCrosstab:
    def test_basic_sum(self):
        df = make_df()
        result = build_crosstab(
            df, row_col="product", col_col="region", value_col="revenue", agg_func="sum"
        )
        assert result["row_col"] == "product"
        assert result["col_col"] == "region"
        assert result["value_col"] == "revenue"
        assert result["agg_func"] == "sum"
        assert isinstance(result["col_headers"], list)
        assert len(result["col_headers"]) > 0
        assert isinstance(result["rows"], list)
        assert len(result["rows"]) == 3  # Widget A, B, C
        assert result["grand_total"] is not None
        assert result["grand_total"] > 0

    def test_row_totals_correct(self):
        df = make_df()
        result = build_crosstab(
            df, row_col="product", col_col="region", value_col="revenue", agg_func="sum"
        )
        # Each row_total should equal sum of its cells (ignoring None)
        for row in result["rows"]:
            cell_sum = sum(c for c in row["cells"] if c is not None)
            assert abs(row["row_total"] - cell_sum) < 0.01

    def test_count_mode(self):
        df = make_df()
        result = build_crosstab(
            df, row_col="product", col_col="region", value_col=None, agg_func="sum"
        )
        # In count mode (value_col=None) grand_total = total row count
        assert result["grand_total"] == len(df)

    def test_mean_aggregation(self):
        df = make_df()
        result = build_crosstab(
            df,
            row_col="product",
            col_col="region",
            value_col="revenue",
            agg_func="mean",
        )
        assert result["agg_func"] == "mean"
        # Widget A appears in North twice (1200 and 500 → mean 850)
        widget_a_row = next(r for r in result["rows"] if r["row_label"] == "Widget A")
        north_idx = result["col_headers"].index("North")
        assert abs(widget_a_row["cells"][north_idx] - 850.0) < 0.1

    def test_invalid_agg_raises(self):
        df = make_df()
        with pytest.raises(ValueError, match="Unsupported aggregation"):
            build_crosstab(
                df,
                row_col="product",
                col_col="region",
                value_col="revenue",
                agg_func="invalid",
            )

    def test_missing_row_col_raises(self):
        df = make_df()
        with pytest.raises(ValueError, match="Column\\(s\\) not found"):
            build_crosstab(
                df, row_col="nonexistent", col_col="region", value_col="revenue"
            )

    def test_missing_value_col_raises(self):
        df = make_df()
        with pytest.raises(ValueError, match="Value column"):
            build_crosstab(
                df, row_col="product", col_col="region", value_col="nonexistent"
            )

    def test_summary_text(self):
        df = make_df()
        result = build_crosstab(
            df, row_col="product", col_col="region", value_col="revenue"
        )
        assert "product" in result["summary"]
        assert "region" in result["summary"]

    def test_col_totals_length_matches_headers(self):
        df = make_df()
        result = build_crosstab(
            df, row_col="product", col_col="region", value_col="revenue"
        )
        assert len(result["col_totals"]) == len(result["col_headers"])

    def test_cells_length_matches_headers(self):
        df = make_df()
        result = build_crosstab(
            df, row_col="product", col_col="region", value_col="revenue"
        )
        for row in result["rows"]:
            assert len(row["cells"]) == len(result["col_headers"])

    def test_max_rows_cap(self):
        # Create a dataset with more than 3 unique row values
        df = make_df()
        result = build_crosstab(
            df, row_col="product", col_col="region", value_col="revenue", max_rows=2
        )
        assert len(result["rows"]) <= 2


# ---------------------------------------------------------------------------
# API tests: GET /api/data/{dataset_id}/crosstab
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
    resp = await ac.post("/api/projects", json={"name": "Crosstab Test"})
    project_id = resp.json()["id"]
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("data.csv", SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    return resp.json()["dataset_id"]


class TestCrosstabEndpoint:
    async def test_basic_sum(self, dataset_id, ac):
        resp = await ac.get(
            f"/api/data/{dataset_id}/crosstab",
            params={
                "rows": "product",
                "cols": "region",
                "values": "revenue",
                "agg": "sum",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["row_col"] == "product"
        assert data["col_col"] == "region"
        assert len(data["rows"]) == 3

    async def test_count_mode(self, dataset_id, ac):
        resp = await ac.get(
            f"/api/data/{dataset_id}/crosstab",
            params={"rows": "product", "cols": "region"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["grand_total"] == 10  # total rows in sample

    async def test_404_unknown_dataset(self, ac):
        resp = await ac.get(
            "/api/data/00000000-0000-0000-0000-000000000000/crosstab",
            params={"rows": "product", "cols": "region"},
        )
        assert resp.status_code == 404

    async def test_400_bad_column(self, dataset_id, ac):
        resp = await ac.get(
            f"/api/data/{dataset_id}/crosstab",
            params={"rows": "nonexistent", "cols": "region"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Unit tests: _detect_crosstab_request()
# ---------------------------------------------------------------------------


class TestDetectCrosstabRequest:
    def setup_method(self):
        from api.chat import _detect_crosstab_request

        self._detect = _detect_crosstab_request

    def test_detects_three_column_pattern(self):
        cols = ["revenue", "region", "product"]
        result = self._detect("break down revenue by region and product", cols)
        assert result is not None
        assert result["value_col"] == "revenue"
        assert result["row_col"] == "region"
        assert result["col_col"] == "product"

    def test_detects_two_column_pattern(self):
        cols = ["region", "product"]
        result = self._detect("show region by product", cols)
        assert result is not None
        assert result["row_col"] == "region"
        assert result["col_col"] == "product"
        assert result["value_col"] is None

    def test_returns_none_when_no_column_match(self):
        cols = ["revenue", "region"]
        result = self._detect("hello world", cols)
        assert result is None

    def test_case_insensitive_column_matching(self):
        cols = ["Revenue", "Region", "Product"]
        result = self._detect("break down revenue by region and product", cols)
        assert result is not None
        # Should resolve to original case
        assert result["value_col"] == "Revenue"
