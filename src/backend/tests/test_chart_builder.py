"""Tests for core/chart_builder.py covering previously-uncovered paths."""

import numpy as np
import pandas as pd
import pytest

from core.chart_builder import (
    build_bar_chart,
    build_boxplot,
    build_correlation_heatmap,
    build_histogram,
    build_line_chart,
    build_pie_chart,
    build_scatter_chart,
    chart_from_query_result,
)


# ---------------------------------------------------------------------------
# build_scatter_chart — with labels (lines 118-125)
# ---------------------------------------------------------------------------


class TestBuildScatterChart:
    def test_without_labels(self):
        result = build_scatter_chart([1, 2, 3], [4, 5, 6], "Test")
        assert result["chart_type"] == "scatter"
        assert all("label" not in p for p in result["data"])

    def test_with_labels(self):
        result = build_scatter_chart([1, 2], [3, 4], "Titled", labels=["A", "B"])
        assert result["data"][0]["label"] == "A"
        assert result["data"][1]["label"] == "B"

    def test_partial_labels(self):
        # labels list shorter than data — only first points get labels
        result = build_scatter_chart([1, 2, 3], [4, 5, 6], "T", labels=["only-one"])
        assert result["data"][0]["label"] == "only-one"
        assert "label" not in result["data"][1]


# ---------------------------------------------------------------------------
# build_pie_chart — Series input (line 143)
# ---------------------------------------------------------------------------


class TestBuildPieChart:
    def test_with_series(self):
        s = pd.Series({"A": 10, "B": 20, "C": 5})
        result = build_pie_chart(s, "Pie from Series")
        assert result["chart_type"] == "pie"
        # Sorted descending: B=20, A=10, C=5
        assert result["data"][0]["name"] == "B"
        assert result["data"][0]["value"] == 20

    def test_tail_rollup_into_other(self):
        data = {str(i): i for i in range(1, 16)}  # 15 items, limit=10
        result = build_pie_chart(data, "Rollup", limit=10)
        names = [d["name"] for d in result["data"]]
        assert "Other" in names
        assert len(result["data"]) == 11  # 10 top + Other

    def test_tail_rollup_no_other_when_zero(self):
        # If other_total == 0, "Other" should not be added
        data = {str(i): 0 for i in range(11)}
        result = build_pie_chart(data, "Zero tail", limit=10)
        names = [d["name"] for d in result["data"]]
        assert "Other" not in names


# ---------------------------------------------------------------------------
# chart_from_query_result — Series and various DataFrame shapes
# ---------------------------------------------------------------------------


class TestChartFromQueryResult:
    def test_series_input_becomes_bar(self):
        s = pd.Series({"North": 100, "South": 200})
        result = chart_from_query_result(s, "Sales by region")
        assert result is not None
        assert result["chart_type"] == "bar"

    def test_single_numeric_column_becomes_histogram(self):
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = chart_from_query_result(df, "Distribution")
        assert result is not None
        assert result["chart_type"] == "histogram"

    def test_single_non_numeric_column_returns_none(self):
        df = pd.DataFrame({"category": ["a", "b", "c"]})
        result = chart_from_query_result(df, "Labels")
        assert result is None

    def test_empty_dataframe_returns_none(self):
        df = pd.DataFrame({"x": [], "y": []})
        result = chart_from_query_result(df, "Empty")
        assert result is None

    def test_multi_column_line_chart_with_x_col(self):
        df = pd.DataFrame(
            {
                "month": ["Jan", "Feb", "Mar"],
                "revenue": [100, 200, 150],
                "cost": [50, 80, 60],
            }
        )
        result = chart_from_query_result(df, "Monthly trends", x_col="month")
        assert result is not None
        assert result["chart_type"] == "line"
        assert "revenue" in result["y_keys"]

    def test_two_column_numeric_y_bar(self):
        df = pd.DataFrame({"region": ["A", "B"], "sales": [300, 500]})
        result = chart_from_query_result(df, "Region sales")
        assert result is not None
        assert result["chart_type"] == "bar"


# ---------------------------------------------------------------------------
# _jsonify — NaN and numpy scalars (lines 232, 234)
# ---------------------------------------------------------------------------


class TestJsonify:
    """Test via build_bar_chart which calls _jsonify on values."""

    def test_nan_value_becomes_none(self):
        result = build_bar_chart({"A": float("nan")}, "NaN test")
        assert result["data"][0]["value"] is None

    def test_numpy_scalar_is_native_python(self):
        data = {"x": np.float64(3.14)}
        result = build_bar_chart(data, "Numpy scalar")
        val = result["data"][0]["value"]
        assert isinstance(val, float)
        assert not isinstance(val, np.floating)

    def test_numpy_int64_is_native_int(self):
        s = pd.Series([np.int64(10), np.int64(20)], index=["a", "b"])
        result = build_bar_chart(s, "Int64 test")
        for item in result["data"]:
            assert isinstance(item["value"], int)


# ---------------------------------------------------------------------------
# build_histogram edge cases
# ---------------------------------------------------------------------------


class TestBuildHistogram:
    def test_rounds_bins(self):
        result = build_histogram([1.12345, 2.23456], [5, 10], "Hist")
        assert result["data"][0]["bin"] == 1.1235  # rounded to 4dp
        assert result["data"][1]["bin"] == 2.2346


# ---------------------------------------------------------------------------
# build_line_chart edge cases
# ---------------------------------------------------------------------------


class TestBuildLineChart:
    def test_short_series_skips_missing(self):
        # y_series values shorter than x_values — missing indices are skipped
        result = build_line_chart(
            x_values=["A", "B", "C"],
            y_series={"s1": [1, 2]},  # only 2 values, 3 x points
            title="Short",
        )
        # Third data point should not have s1 key
        assert "s1" not in result["data"][2]


# ---------------------------------------------------------------------------
# chart_from_query_result — final fallback None path (line 226)
# ---------------------------------------------------------------------------


class TestChartFromQueryResultFallback:
    def test_multi_column_no_numeric_returns_none(self):
        # Multi-column with x_col but remaining columns are non-numeric → final None
        df = pd.DataFrame(
            {
                "month": ["Jan", "Feb"],
                "label1": ["x", "y"],
                "label2": ["a", "b"],
            }
        )
        result = chart_from_query_result(df, "All strings", x_col="month")
        assert result is None


# ---------------------------------------------------------------------------
# build_correlation_heatmap
# ---------------------------------------------------------------------------


class TestBuildCorrelationHeatmap:
    def _make_matrix(self, columns):
        """Create a synthetic corr matrix dict (same shape as _corr_matrix_dict output)."""
        import numpy as np

        n = len(columns)
        data = np.eye(n)
        rows = []
        for i, col in enumerate(columns):
            row = {"column": col}
            for j, other in enumerate(columns):
                row[other] = round(float(data[i, j]), 3)
            rows.append(row)
        return rows

    def test_basic_heatmap(self):
        cols = ["revenue", "units", "price"]
        matrix = self._make_matrix(cols)
        spec = build_correlation_heatmap(matrix, cols)
        assert spec["chart_type"] == "heatmap"
        assert spec["y_keys"] == cols
        assert spec["x_key"] == "row"
        assert len(spec["data"]) == len(cols)

    def test_diagonal_is_one(self):
        cols = ["a", "b"]
        matrix = self._make_matrix(cols)
        spec = build_correlation_heatmap(matrix, cols)
        for row in spec["data"]:
            assert row[row["row"]] == 1.0

    def test_custom_title(self):
        cols = ["x", "y"]
        matrix = self._make_matrix(cols)
        spec = build_correlation_heatmap(matrix, cols, title="My Heatmap")
        assert spec["title"] == "My Heatmap"

    def test_two_column_matrix(self):
        cols = ["a", "b"]
        matrix = [
            {"column": "a", "a": 1.0, "b": 0.75},
            {"column": "b", "a": 0.75, "b": 1.0},
        ]
        spec = build_correlation_heatmap(matrix, cols)
        row_a = next(r for r in spec["data"] if r["row"] == "a")
        assert row_a["b"] == 0.75


# ---------------------------------------------------------------------------
# build_boxplot — new chart type
# ---------------------------------------------------------------------------


class TestBuildBoxplot:
    def _make_df(self) -> pd.DataFrame:
        """Synthetic sales DataFrame: numeric 'sales', categorical 'region'."""
        return pd.DataFrame(
            {
                "sales": [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
                "region": ["North"] * 5 + ["South"] * 5,
            }
        )

    # --- chart_type and structure ---
    def test_chart_type_is_boxplot(self):
        df = self._make_df()
        spec = build_boxplot(df, value_col="sales")
        assert spec["chart_type"] == "boxplot"

    def test_x_key_is_group(self):
        df = self._make_df()
        spec = build_boxplot(df, value_col="sales")
        assert spec["x_key"] == "group"

    def test_y_keys_contain_quartiles(self):
        df = self._make_df()
        spec = build_boxplot(df, value_col="sales")
        for key in ("min", "q1", "median", "q3", "max"):
            assert key in spec["y_keys"]

    # --- single-column (no group) ---
    def test_single_column_returns_one_box(self):
        df = self._make_df()
        spec = build_boxplot(df, value_col="sales")
        assert len(spec["data"]) == 1
        assert spec["data"][0]["group"] == "sales"

    def test_single_column_stats_correct(self):
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        spec = build_boxplot(df, value_col="v")
        box = spec["data"][0]
        assert box["median"] == 3.0
        # pandas default linear interpolation: Q1=2.0, Q3=4.0 for [1,2,3,4,5]
        assert box["q1"] == pytest.approx(2.0, abs=0.01)
        assert box["q3"] == pytest.approx(4.0, abs=0.01)

    # --- grouped ---
    def test_grouped_returns_one_box_per_group(self):
        df = self._make_df()
        spec = build_boxplot(df, value_col="sales", group_col="region")
        assert len(spec["data"]) == 2
        groups = {r["group"] for r in spec["data"]}
        assert groups == {"North", "South"}

    def test_grouped_sorted_by_median_desc(self):
        df = self._make_df()
        spec = build_boxplot(df, value_col="sales", group_col="region")
        medians = [r["median"] for r in spec["data"]]
        assert medians == sorted(medians, reverse=True)

    def test_grouped_each_box_has_required_keys(self):
        df = self._make_df()
        spec = build_boxplot(df, value_col="sales", group_col="region")
        for box in spec["data"]:
            for key in ("group", "min", "q1", "median", "q3", "max", "mean", "count"):
                assert key in box, f"Missing key: {key}"

    def test_min_lte_q1_lte_median_lte_q3_lte_max(self):
        df = self._make_df()
        spec = build_boxplot(df, value_col="sales", group_col="region")
        for box in spec["data"]:
            assert box["min"] <= box["q1"] <= box["median"] <= box["q3"] <= box["max"]

    # --- limit ---
    def test_limit_caps_groups(self):
        df = pd.DataFrame(
            {
                "v": range(100),
                "g": [str(i) for i in range(100)],  # 100 distinct groups
            }
        )
        spec = build_boxplot(df, value_col="v", group_col="g", limit=5)
        assert len(spec["data"]) <= 5

    # --- title ---
    def test_auto_title_single_col(self):
        df = pd.DataFrame({"revenue": [1, 2, 3]})
        spec = build_boxplot(df, value_col="revenue")
        assert "revenue" in spec["title"].lower()

    def test_auto_title_grouped(self):
        df = self._make_df()
        spec = build_boxplot(df, value_col="sales", group_col="region")
        assert "sales" in spec["title"] and "region" in spec["title"]

    def test_explicit_title_respected(self):
        df = self._make_df()
        spec = build_boxplot(df, value_col="sales", title="My Custom Title")
        assert spec["title"] == "My Custom Title"

    # --- edge cases ---
    def test_empty_column_returns_empty_data(self):
        df = pd.DataFrame({"v": [float("nan"), float("nan")]})
        spec = build_boxplot(df, value_col="v")
        assert spec["data"] == []

    def test_nonexistent_group_col_falls_back_to_single(self):
        # group_col not in df → fall back to single-box behaviour
        df = pd.DataFrame({"v": [1, 2, 3]})
        spec = build_boxplot(df, value_col="v", group_col="missing")
        assert len(spec["data"]) == 1

    def test_whiskers_use_tukey_fence(self):
        # Add a clear outlier; whisker should not extend to it
        values = list(range(1, 11)) + [1000]  # 1000 is an outlier
        df = pd.DataFrame({"v": values})
        spec = build_boxplot(df, value_col="v")
        box = spec["data"][0]
        assert box["max"] < 1000  # whisker stops before outlier
