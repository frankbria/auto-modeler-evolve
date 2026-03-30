"""Tests for summary statistics and category value counts features."""

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel

from core.analyzer import compute_summary_stats, compute_value_counts


# ---------------------------------------------------------------------------
# Test DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_df():
    """DataFrame with both numeric and categorical columns."""
    return pd.DataFrame(
        {
            "region": [
                "East",
                "West",
                "East",
                "North",
                "West",
                "East",
                "South",
                "North",
            ],
            "product": ["A", "B", "A", "C", "B", "A", "C", "B"],
            "revenue": [5000.0, 1200.0, 8500.0, 300.0, 4200.0, 9100.0, 750.0, 6300.0],
            "units": [50, 12, 85, 3, 42, 91, 7, 63],
        }
    )


@pytest.fixture
def null_df():
    """DataFrame with some null values."""
    return pd.DataFrame(
        {
            "category": ["X", None, "Y", "X", None, "Z"],
            "value": [10.0, None, 30.0, 40.0, None, 60.0],
        }
    )


@pytest.fixture
def numeric_only_df():
    """DataFrame with only numeric columns."""
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


@pytest.fixture
def categorical_only_df():
    """DataFrame with only categorical columns."""
    return pd.DataFrame(
        {
            "color": ["red", "blue", "red", "green", "blue", "red"],
            "size": ["S", "M", "L", "S", "M", "L"],
        }
    )


# ---------------------------------------------------------------------------
# Unit tests: compute_summary_stats()
# ---------------------------------------------------------------------------


class TestComputeSummaryStats:
    def test_returns_required_keys(self, mixed_df):
        result = compute_summary_stats(mixed_df)
        for key in (
            "total_rows",
            "total_cols",
            "numeric_stats",
            "categorical_stats",
            "summary",
        ):
            assert key in result

    def test_total_counts(self, mixed_df):
        result = compute_summary_stats(mixed_df)
        assert result["total_rows"] == 8
        assert result["total_cols"] == 4

    def test_numeric_stats_count(self, mixed_df):
        result = compute_summary_stats(mixed_df)
        numeric_cols = [s["column"] for s in result["numeric_stats"]]
        assert "revenue" in numeric_cols
        assert "units" in numeric_cols
        assert len(result["numeric_stats"]) == 2

    def test_categorical_stats_count(self, mixed_df):
        result = compute_summary_stats(mixed_df)
        cat_cols = [s["column"] for s in result["categorical_stats"]]
        assert "region" in cat_cols
        assert "product" in cat_cols
        assert len(result["categorical_stats"]) == 2

    def test_numeric_stat_keys(self, mixed_df):
        result = compute_summary_stats(mixed_df)
        rev_stat = next(s for s in result["numeric_stats"] if s["column"] == "revenue")
        for key in (
            "count",
            "mean",
            "std",
            "min",
            "q25",
            "median",
            "q75",
            "max",
            "null_count",
        ):
            assert key in rev_stat

    def test_numeric_values_correct(self, mixed_df):
        result = compute_summary_stats(mixed_df)
        rev_stat = next(s for s in result["numeric_stats"] if s["column"] == "revenue")
        assert rev_stat["count"] == 8
        assert rev_stat["min"] == 300.0
        assert rev_stat["max"] == 9100.0
        assert rev_stat["null_count"] == 0

    def test_categorical_stat_keys(self, mixed_df):
        result = compute_summary_stats(mixed_df)
        reg_stat = next(
            s for s in result["categorical_stats"] if s["column"] == "region"
        )
        for key in ("count", "unique", "top", "freq", "null_count"):
            assert key in reg_stat

    def test_categorical_values_correct(self, mixed_df):
        result = compute_summary_stats(mixed_df)
        reg_stat = next(
            s for s in result["categorical_stats"] if s["column"] == "region"
        )
        assert reg_stat["count"] == 8
        assert reg_stat["unique"] == 4
        assert reg_stat["top"] == "East"  # most common (3 times)
        assert reg_stat["freq"] == 3
        assert reg_stat["null_count"] == 0

    def test_null_counts_tracked(self, null_df):
        result = compute_summary_stats(null_df)
        val_stat = next(s for s in result["numeric_stats"] if s["column"] == "value")
        cat_stat = next(
            s for s in result["categorical_stats"] if s["column"] == "category"
        )
        assert val_stat["null_count"] == 2
        assert cat_stat["null_count"] == 2

    def test_null_excluded_from_calculations(self, null_df):
        result = compute_summary_stats(null_df)
        val_stat = next(s for s in result["numeric_stats"] if s["column"] == "value")
        assert val_stat["count"] == 4  # 6 rows - 2 null

    def test_numeric_only_df(self, numeric_only_df):
        result = compute_summary_stats(numeric_only_df)
        assert len(result["numeric_stats"]) == 2
        assert len(result["categorical_stats"]) == 0

    def test_categorical_only_df(self, categorical_only_df):
        result = compute_summary_stats(categorical_only_df)
        assert len(result["numeric_stats"]) == 0
        assert len(result["categorical_stats"]) == 2

    def test_summary_string_contains_row_col_counts(self, mixed_df):
        result = compute_summary_stats(mixed_df)
        assert "8" in result["summary"]
        assert "4" in result["summary"]

    def test_nan_in_numeric_handled_gracefully(self):
        df = pd.DataFrame({"x": [float("inf"), float("nan"), 5.0]})
        result = compute_summary_stats(df)
        x_stat = result["numeric_stats"][0]
        # inf and nan are excluded from stats (dropna) or handled
        assert x_stat is not None

    def test_median_correct(self, mixed_df):
        result = compute_summary_stats(mixed_df)
        units_stat = next(s for s in result["numeric_stats"] if s["column"] == "units")
        # units: [50, 12, 85, 3, 42, 91, 7, 63] sorted: [3,7,12,42,50,63,85,91] → median=(42+50)/2=46
        assert units_stat["median"] == pytest.approx(46.0, abs=1.0)


# ---------------------------------------------------------------------------
# Unit tests: compute_value_counts()
# ---------------------------------------------------------------------------


class TestComputeValueCounts:
    def test_returns_required_keys(self, mixed_df):
        result = compute_value_counts(mixed_df, "region")
        for key in (
            "column",
            "total_rows",
            "non_null",
            "null_count",
            "unique_count",
            "rows",
            "has_more",
            "summary",
        ):
            assert key in result

    def test_column_set_correctly(self, mixed_df):
        result = compute_value_counts(mixed_df, "region")
        assert result["column"] == "region"

    def test_rows_sorted_by_frequency(self, mixed_df):
        result = compute_value_counts(mixed_df, "region")
        counts = [r["count"] for r in result["rows"]]
        assert counts == sorted(counts, reverse=True)

    def test_most_common_value_correct(self, mixed_df):
        result = compute_value_counts(mixed_df, "region")
        assert result["rows"][0]["value"] == "East"
        assert result["rows"][0]["count"] == 3

    def test_percentages_sum_to_100(self, mixed_df):
        result = compute_value_counts(mixed_df, "region")
        total_pct = sum(r["pct"] for r in result["rows"])
        assert abs(total_pct - 100.0) < 1.0  # allow float rounding

    def test_unique_count_correct(self, mixed_df):
        result = compute_value_counts(mixed_df, "region")
        assert result["unique_count"] == 4

    def test_null_count_tracked(self, null_df):
        result = compute_value_counts(null_df, "category")
        assert result["null_count"] == 2
        assert result["non_null"] == 4

    def test_n_cap_limits_rows(self, mixed_df):
        result = compute_value_counts(mixed_df, "region", n=2)
        assert len(result["rows"]) == 2
        assert result["has_more"] is True

    def test_n_cap_no_more(self, mixed_df):
        result = compute_value_counts(mixed_df, "region", n=20)
        assert result["has_more"] is False

    def test_invalid_column_raises(self, mixed_df):
        with pytest.raises(ValueError, match="not found"):
            compute_value_counts(mixed_df, "nonexistent")

    def test_summary_mentions_column(self, mixed_df):
        result = compute_value_counts(mixed_df, "region")
        assert "region" in result["summary"]

    def test_summary_mentions_top_value(self, mixed_df):
        result = compute_value_counts(mixed_df, "region")
        assert "East" in result["summary"]

    def test_total_rows_correct(self, mixed_df):
        result = compute_value_counts(mixed_df, "region")
        assert result["total_rows"] == 8

    def test_n_clamped_min(self, mixed_df):
        result = compute_value_counts(mixed_df, "region", n=0)
        assert len(result["rows"]) >= 1

    def test_n_clamped_max(self, mixed_df):
        result = compute_value_counts(mixed_df, "region", n=200)
        # n is clamped to 50
        assert len(result["rows"]) <= 50


# ---------------------------------------------------------------------------
# Pattern tests: _SUMMARY_STATS_PATTERNS
# ---------------------------------------------------------------------------


class TestSummaryStatsPatterns:
    @pytest.fixture
    def pattern(self):
        from api.chat import _SUMMARY_STATS_PATTERNS

        return _SUMMARY_STATS_PATTERNS

    @pytest.mark.parametrize(
        "phrase",
        [
            "summarize my data",
            "summarize my dataset",
            "describe my data",
            "statistical summary",
            "summary statistics",
            "descriptive statistics",
            "stats for all columns",
            "dataset statistics",
            "data overview",
            "give me the stats for my data",
            "show me statistics for all my columns",
        ],
    )
    def test_matches_trigger_phrases(self, pattern, phrase):
        assert pattern.search(phrase), f"Should match: '{phrase}'"

    @pytest.mark.parametrize(
        "phrase",
        [
            "describe the revenue column",
            "profile the region column",
            "most common values in region",
            "show me the missing values",
            "is my data ready?",
        ],
    )
    def test_does_not_match_unrelated(self, pattern, phrase):
        assert not pattern.search(phrase), f"Should NOT match: '{phrase}'"


# ---------------------------------------------------------------------------
# Pattern tests: _VALUE_COUNT_PATTERNS
# ---------------------------------------------------------------------------


class TestValueCountPatterns:
    @pytest.fixture
    def pattern(self):
        from api.chat import _VALUE_COUNT_PATTERNS

        return _VALUE_COUNT_PATTERNS

    @pytest.mark.parametrize(
        "phrase",
        [
            "most common values in region",
            "most common region values",
            "most frequent product",
            "frequency table for region",
            "frequency of region",
            "value counts for product",
            "value frequencies for product",
            "how often does each region appear",
            "how common is each region",
            "count of each product value",
            "top values in region",
            "top occurrences in product",
        ],
    )
    def test_matches_trigger_phrases(self, pattern, phrase):
        assert pattern.search(phrase), f"Should match: '{phrase}'"

    @pytest.mark.parametrize(
        "phrase",
        [
            "summarize my data",
            "describe the revenue column",
            "show me the missing values",
            "bar chart of revenue by region",
            "top 10 customers by revenue",
        ],
    )
    def test_does_not_match_unrelated(self, pattern, phrase):
        assert not pattern.search(phrase), f"Should NOT match: '{phrase}'"


# ---------------------------------------------------------------------------
# Pattern tests: _detect_value_counts_col()
# ---------------------------------------------------------------------------


class TestDetectValueCountsCol:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {"region": ["A"], "product_category": ["B"], "revenue": [1.0]}
        )

    def test_detects_exact_column_name(self, sample_df):
        from api.chat import _detect_value_counts_col

        result = _detect_value_counts_col("most common values in region", sample_df)
        assert result == "region"

    def test_detects_underscore_variant(self, sample_df):
        from api.chat import _detect_value_counts_col

        result = _detect_value_counts_col(
            "frequency table for product category", sample_df
        )
        assert result == "product_category"

    def test_longest_match_first(self, sample_df):
        from api.chat import _detect_value_counts_col

        # "product_category" is longer than "product" (which isn't in df anyway)
        result = _detect_value_counts_col(
            "most common product category values", sample_df
        )
        assert result == "product_category"

    def test_fallback_to_first_categorical(self, sample_df):
        from api.chat import _detect_value_counts_col

        result = _detect_value_counts_col("show me the top values", sample_df)
        # No column name mentioned — falls back to first categorical
        assert result in ("region", "product_category")

    def test_empty_df_returns_none(self):
        from api.chat import _detect_value_counts_col

        result = _detect_value_counts_col("most common values", pd.DataFrame())
        assert result is None


# ---------------------------------------------------------------------------
# Integration tests: REST endpoints
# ---------------------------------------------------------------------------

_CSV_DATA = (
    "region,product,revenue,units\n"
    "East,A,5000.0,50\n"
    "West,B,1200.0,12\n"
    "East,A,8500.0,85\n"
    "North,C,300.0,3\n"
    "West,B,4200.0,42\n"
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
        proj_resp = await ac.post("/api/projects", json={"name": "Test"})
        assert proj_resp.status_code == 201
        project_id = proj_resp.json()["id"]

        upload_resp = await ac.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("data.csv", _CSV_DATA.encode(), "text/csv")},
        )
        assert upload_resp.status_code in (200, 201)
        dataset_id = upload_resp.json()["dataset_id"]

        yield ac, dataset_id


@pytest.mark.anyio
async def test_summary_stats_endpoint_200(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/summary-stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "numeric_stats" in data
    assert "categorical_stats" in data
    assert data["total_rows"] == 5
    assert data["total_cols"] == 4


@pytest.mark.anyio
async def test_summary_stats_numeric_columns(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/summary-stats")
    assert resp.status_code == 200
    data = resp.json()
    numeric_cols = [s["column"] for s in data["numeric_stats"]]
    assert "revenue" in numeric_cols
    assert "units" in numeric_cols


@pytest.mark.anyio
async def test_summary_stats_categorical_columns(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/summary-stats")
    assert resp.status_code == 200
    data = resp.json()
    cat_cols = [s["column"] for s in data["categorical_stats"]]
    assert "region" in cat_cols
    assert "product" in cat_cols


@pytest.mark.anyio
async def test_summary_stats_404(client_with_dataset):
    ac, _ = client_with_dataset
    resp = await ac.get("/api/data/nonexistent-id/summary-stats")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_value_counts_endpoint_200(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/value-counts?col=region")
    assert resp.status_code == 200
    data = resp.json()
    assert data["column"] == "region"
    assert data["total_rows"] == 5
    assert len(data["rows"]) > 0


@pytest.mark.anyio
async def test_value_counts_sorted_by_frequency(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/value-counts?col=region")
    assert resp.status_code == 200
    counts = [r["count"] for r in resp.json()["rows"]]
    assert counts == sorted(counts, reverse=True)


@pytest.mark.anyio
async def test_value_counts_top_value_correct(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/value-counts?col=region")
    assert resp.status_code == 200
    # East appears twice (rows 1 and 3)
    assert resp.json()["rows"][0]["value"] == "East"


@pytest.mark.anyio
async def test_value_counts_n_parameter(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/value-counts?col=region&n=2")
    assert resp.status_code == 200
    assert len(resp.json()["rows"]) == 2


@pytest.mark.anyio
async def test_value_counts_unknown_column(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(f"/api/data/{dataset_id}/value-counts?col=nonexistent")
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_value_counts_404(client_with_dataset):
    ac, _ = client_with_dataset
    resp = await ac.get("/api/data/bad-id/value-counts?col=region")
    assert resp.status_code == 404
