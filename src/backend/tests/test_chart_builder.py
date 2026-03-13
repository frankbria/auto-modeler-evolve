"""Tests for core/chart_builder.py covering previously-uncovered paths."""

import numpy as np
import pandas as pd
import pytest

from core.chart_builder import (
    build_bar_chart,
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
        df = pd.DataFrame({
            "month": ["Jan", "Feb", "Mar"],
            "revenue": [100, 200, 150],
            "cost": [50, 80, 60],
        })
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
        df = pd.DataFrame({
            "month": ["Jan", "Feb"],
            "label1": ["x", "y"],
            "label2": ["a", "b"],
        })
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
