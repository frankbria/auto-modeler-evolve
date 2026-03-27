"""Tests for top-N ranking: compute_top_n() + endpoint + chat patterns."""

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel

from core.analyzer import compute_top_n


# ---------------------------------------------------------------------------
# Test DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def sales_df():
    """Simple sales dataset for ranking tests."""
    return pd.DataFrame(
        {
            "customer": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"],
            "revenue": [5000.0, 1200.0, 8500.0, 300.0, 4200.0, 9100.0, 750.0, 6300.0],
            "units": [50, 12, 85, 3, 42, 91, 7, 63],
            "region": ["East", "West", "East", "North", "West", "East", "South", "North"],
        }
    )


@pytest.fixture
def nan_df():
    """Dataset with NaN values in the sort column."""
    return pd.DataFrame(
        {
            "product": ["A", "B", "C", "D", "E"],
            "sales": [100.0, None, 300.0, 50.0, None],
        }
    )


@pytest.fixture
def small_df():
    """Tiny dataset with fewer rows than n."""
    return pd.DataFrame({"name": ["X", "Y"], "score": [10.0, 20.0]})


# ---------------------------------------------------------------------------
# Unit tests: compute_top_n()
# ---------------------------------------------------------------------------


class TestComputeTopN:
    def test_top_n_desc(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=3)
        assert result["direction"] == "top"
        assert result["ascending"] is False
        assert result["n_returned"] == 3
        # Rows should be sorted descending
        revenues = [r["revenue"] for r in result["rows"]]
        assert revenues == sorted(revenues, reverse=True)
        # Rank 1 should be highest
        assert result["rows"][0]["_rank"] == 1
        assert result["rows"][0]["revenue"] == 9100.0

    def test_bottom_n_asc(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=3, ascending=True)
        assert result["direction"] == "bottom"
        assert result["ascending"] is True
        revenues = [r["revenue"] for r in result["rows"]]
        assert revenues == sorted(revenues)
        assert result["rows"][0]["revenue"] == 300.0

    def test_n_capped_at_50(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=100)
        # Dataset has 8 rows; n is capped at min(50, len(df))
        assert result["n_requested"] == 50
        assert result["n_returned"] == 8

    def test_n_at_least_1(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=0)
        assert result["n_returned"] == 1

    def test_unknown_column_returns_error(self, sales_df):
        result = compute_top_n(sales_df, "nonexistent")
        assert "error" in result
        assert "nonexistent" in result["error"]

    def test_non_numeric_column_returns_error(self, sales_df):
        result = compute_top_n(sales_df, "customer")
        assert "error" in result
        assert "not numeric" in result["error"]

    def test_nan_rows_excluded_from_ranking(self, nan_df):
        result = compute_top_n(nan_df, "sales", n=5)
        # Only 3 rows have valid sales values
        assert result["n_returned"] == 3
        for r in result["rows"]:
            assert r["sales"] is not None

    def test_fewer_rows_than_n(self, small_df):
        result = compute_top_n(small_df, "score", n=10)
        assert result["n_returned"] == 2
        assert result["total_rows"] == 2

    def test_display_cols_included(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=3, display_cols=["customer", "revenue"])
        for row in result["rows"]:
            assert "customer" in row
            assert "revenue" in row

    def test_sort_col_always_in_display(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=3, display_cols=["units"])
        for row in result["rows"]:
            assert "revenue" in row

    def test_rank_numbers_sequential(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=5)
        ranks = [r["_rank"] for r in result["rows"]]
        assert ranks == list(range(1, 6))

    def test_summary_contains_direction(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=3)
        assert "Top 3" in result["summary"]
        assert "revenue" in result["summary"].lower()

    def test_summary_bottom(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=3, ascending=True)
        assert "Bottom 3" in result["summary"]

    def test_total_rows_in_result(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=3)
        assert result["total_rows"] == 8

    def test_result_keys(self, sales_df):
        result = compute_top_n(sales_df, "revenue", n=5)
        for key in ["sort_col", "direction", "ascending", "n_requested", "n_returned",
                    "total_rows", "display_cols", "rows", "summary"]:
            assert key in result

    def test_cols_capped_at_8(self):
        """DataFrame with 12 columns — display should be capped at 8."""
        df = pd.DataFrame({f"col_{i}": [float(i)] for i in range(12)})
        result = compute_top_n(df, "col_0", n=1)
        assert len(result["display_cols"]) <= 8


# ---------------------------------------------------------------------------
# Pattern tests
# ---------------------------------------------------------------------------


class TestTopNPatterns:
    """Verify _TOPN_PATTERNS matches expected natural-language queries."""

    @pytest.fixture(autouse=True)
    def load_pattern(self):
        from api.chat import _TOPN_PATTERNS
        self.pattern = _TOPN_PATTERNS

    def test_top_10_customers(self):
        assert self.pattern.search("show me top 10 customers by revenue")

    def test_bottom_5_products(self):
        assert self.pattern.search("what are the bottom 5 products")

    def test_highest_3(self):
        assert self.pattern.search("show highest 3 orders")

    def test_lowest_n(self):
        assert self.pattern.search("lowest 5 performing regions")

    def test_best_accounts(self):
        assert self.pattern.search("who are my best accounts")

    def test_worst_customers(self):
        assert self.pattern.search("show me worst customers")

    def test_rank_by(self):
        assert self.pattern.search("rank by revenue")

    def test_list_top(self):
        assert self.pattern.search("list top products")

    def test_no_match_correlation(self):
        assert not self.pattern.search("what correlates with revenue")

    def test_no_match_forecast(self):
        assert not self.pattern.search("predict next 6 months")

    def test_show_me_top(self):
        assert self.pattern.search("show me top performers")

    def test_largest_orders(self):
        assert self.pattern.search("largest 10 orders")

    def test_smallest_accounts(self):
        assert self.pattern.search("smallest 3 accounts")


# ---------------------------------------------------------------------------
# _detect_topn_request helper tests
# ---------------------------------------------------------------------------


class TestDetectTopnRequest:
    @pytest.fixture(autouse=True)
    def load_fn(self):
        from api.chat import _detect_topn_request
        self.detect = _detect_topn_request

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "customer": ["a", "b", "c"],
            "revenue": [100.0, 200.0, 300.0],
            "units": [1.0, 2.0, 3.0],
        })

    def test_detects_column_name(self, df):
        result = self.detect("show me top 5 by revenue", df)
        assert result is not None
        assert result["sort_col"] == "revenue"

    def test_detects_n(self, df):
        result = self.detect("top 7 customers", df)
        assert result["n"] == 7

    def test_detects_n_word(self, df):
        result = self.detect("top ten customers", df)
        assert result["n"] == 10

    def test_default_n_10(self, df):
        result = self.detect("show top customers", df)
        assert result["n"] == 10

    def test_descending_for_top(self, df):
        result = self.detect("top 5 by revenue", df)
        assert result["ascending"] is False

    def test_ascending_for_bottom(self, df):
        result = self.detect("bottom 5 by revenue", df)
        assert result["ascending"] is True

    def test_ascending_for_worst(self, df):
        result = self.detect("worst 3 customers by units", df)
        assert result["ascending"] is True

    def test_fallback_col_when_no_match(self, df):
        result = self.detect("show top 5", df)
        assert result is not None
        assert result["sort_col"] in ["revenue", "units"]

    def test_returns_none_for_no_numeric_cols(self):
        df_text = pd.DataFrame({"name": ["a", "b"], "region": ["East", "West"]})
        result = self.detect("top 5", df_text)
        assert result is None


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------

_CSV_DATA = (
    "customer,revenue,region\n"
    "Alice,5000.0,East\n"
    "Bob,1200.0,West\n"
    "Carol,8500.0,East\n"
    "Dave,300.0,North\n"
    "Eve,4200.0,West\n"
)


@pytest.fixture
async def client_with_dataset(tmp_path, set_test_env):
    import db
    from main import app
    from sqlmodel import create_engine

    test_db = str(tmp_path / "test.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db.DATA_DIR = tmp_path
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        proj_resp = await ac.post("/api/projects", json={"name": "Test Project"})
        assert proj_resp.status_code == 201
        project_id = proj_resp.json()["id"]

        upload_resp = await ac.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", _CSV_DATA.encode(), "text/csv")},
        )
        assert upload_resp.status_code == 201
        dataset_id = upload_resp.json()["dataset_id"]

        yield ac, dataset_id


@pytest.mark.anyio
async def test_endpoint_top_n_default(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/top-n?col=revenue")
    assert resp.status_code == 200
    data = resp.json()
    assert data["direction"] == "top"
    assert data["n_returned"] == 5
    revenues = [r["revenue"] for r in data["rows"]]
    assert revenues == sorted(revenues, reverse=True)


@pytest.mark.anyio
async def test_endpoint_bottom_n(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/top-n?col=revenue&n=3&order=asc")
    assert resp.status_code == 200
    data = resp.json()
    assert data["direction"] == "bottom"
    assert data["n_returned"] == 3
    revenues = [r["revenue"] for r in data["rows"]]
    assert revenues == sorted(revenues)


@pytest.mark.anyio
async def test_endpoint_unknown_column(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/top-n?col=nonexistent")
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_endpoint_non_numeric_column(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/top-n?col=customer")
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_endpoint_unknown_dataset(client_with_dataset):
    ac, _ = client_with_dataset
    resp = await ac.get("/api/data/nonexistent-id/top-n?col=revenue")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_endpoint_n_clamped(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/top-n?col=revenue&n=200")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_requested"] == 50
