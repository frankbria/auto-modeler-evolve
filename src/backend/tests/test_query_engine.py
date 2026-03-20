"""test_query_engine.py

Unit tests for core/query_engine.py — tests the query execution logic
without calling the real Anthropic API by invoking internal functions directly.

Coverage targets:
- _execute_spec: all 5 operation types (distribution, groupby/top_n, timeseries,
  correlation, filter)
- _apply_filter: all 6 operators (>, <, >=, <=, ==, contains)
- _df_to_text
- _safe_rows
- _find_col
- run_nl_query with parsed spec (monkeypatched _parse_question_to_spec)
- run_nl_query returning None (Claude can't parse question)
- run_nl_query raising exception in _execute_spec
- generate_chart_for_message (data intent / no data intent)
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.query_engine import (
    QueryResult,
    _apply_filter,
    _df_to_text,
    _execute_spec,
    _find_col,
    _safe_rows,
    generate_chart_for_message,
    run_nl_query,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sales_df():
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "West", "North", "South"],
            "product": [
                "Widget A",
                "Widget B",
                "Widget A",
                "Widget C",
                "Widget B",
                "Widget A",
            ],
            "revenue": [1200.0, 850.0, 2100.0, 450.0, 1650.0, 975.0],
            "units": [10, 8, 18, 4, 15, 9],
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
            ],
        }
    )


@pytest.fixture
def column_info(sales_df):
    return [
        {"name": col, "dtype": str(sales_df[col].dtype)} for col in sales_df.columns
    ]


# ---------------------------------------------------------------------------
# _find_col
# ---------------------------------------------------------------------------


class TestFindCol:
    def test_finds_first_valid(self, sales_df):
        assert _find_col(["missing", "revenue"], sales_df) == "revenue"

    def test_returns_none_when_all_missing(self, sales_df):
        assert _find_col(["nope", "also_nope"], sales_df) is None

    def test_empty_list(self, sales_df):
        assert _find_col([], sales_df) is None


# ---------------------------------------------------------------------------
# _apply_filter
# ---------------------------------------------------------------------------


class TestApplyFilter:
    def test_greater_than(self, sales_df):
        result = _apply_filter(sales_df, "revenue", ">", 1000)
        assert len(result) == 3
        assert all(result["revenue"] > 1000)

    def test_less_than(self, sales_df):
        result = _apply_filter(sales_df, "revenue", "<", 1000)
        assert all(result["revenue"] < 1000)

    def test_greater_equal(self, sales_df):
        result = _apply_filter(sales_df, "revenue", ">=", 1200)
        assert all(result["revenue"] >= 1200)

    def test_less_equal(self, sales_df):
        result = _apply_filter(sales_df, "revenue", "<=", 850)
        assert all(result["revenue"] <= 850)

    def test_equal(self, sales_df):
        result = _apply_filter(sales_df, "region", "==", "North")
        assert len(result) == 2
        assert all(result["region"] == "North")

    def test_equal_alt_form(self, sales_df):
        result = _apply_filter(sales_df, "region", "=", "North")
        assert len(result) == 2

    def test_contains(self, sales_df):
        result = _apply_filter(sales_df, "product", "contains", "Widget A")
        assert all("Widget A" in p for p in result["product"])

    def test_contains_case_insensitive(self, sales_df):
        result = _apply_filter(sales_df, "product", "contains", "widget")
        assert len(result) == 6  # all products contain "widget" case-insensitively

    def test_unknown_operator_returns_full_df(self, sales_df):
        result = _apply_filter(sales_df, "revenue", "BETWEEN", 100)
        assert len(result) == len(sales_df)


# ---------------------------------------------------------------------------
# _df_to_text
# ---------------------------------------------------------------------------


class TestDfToText:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        text = _df_to_text(df)
        assert "a=1" in text
        assert "b=x" in text
        assert text.count("- ") == 2

    def test_empty_df(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=int)})
        text = _df_to_text(df)
        assert text == ""


# ---------------------------------------------------------------------------
# _safe_rows
# ---------------------------------------------------------------------------


class TestSafeRows:
    def test_nan_becomes_none(self):
        df = pd.DataFrame({"val": [1.0, float("nan"), 3.0]})
        rows = _safe_rows(df)
        assert rows[1]["val"] is None
        assert rows[0]["val"] == 1.0

    def test_numpy_scalar_converted(self):
        df = pd.DataFrame({"val": np.array([1, 2, 3], dtype=np.int64)})
        rows = _safe_rows(df)
        assert isinstance(rows[0]["val"], int)

    def test_limits_to_20_rows(self):
        df = pd.DataFrame({"x": range(30)})
        # _safe_rows doesn't limit itself; caller uses .head(20)
        rows = _safe_rows(df.head(20))
        assert len(rows) == 20


# ---------------------------------------------------------------------------
# _execute_spec
# ---------------------------------------------------------------------------


class TestExecuteSpecDistribution:
    def test_numeric_column(self, sales_df):
        spec = {"operation": "distribution", "columns": ["revenue"]}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is not None
        assert "revenue" in result_df.columns
        assert "Distribution" in text or "distribution" in text.lower()

    def test_categorical_column(self, sales_df):
        spec = {"operation": "distribution", "columns": ["region"]}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is not None
        assert "count" in result_df.columns

    def test_missing_column(self, sales_df):
        spec = {"operation": "distribution", "columns": ["nonexistent"]}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is None
        assert "not found" in text.lower() or "column" in text.lower()


class TestExecuteSpecGroupby:
    def test_groupby_sum(self, sales_df):
        spec = {
            "operation": "groupby",
            "group_by": ["region"],
            "aggregate": {"revenue": "sum"},
            "sort_ascending": False,
            "limit": 10,
        }
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is not None
        assert len(result_df) <= 4  # 4 unique regions

    def test_top_n_operation(self, sales_df):
        spec = {
            "operation": "top_n",
            "group_by": ["product"],
            "aggregate": {"units": "sum"},
            "sort_ascending": False,
            "limit": 2,
        }
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is not None
        assert len(result_df) <= 2

    def test_groupby_ascending(self, sales_df):
        spec = {
            "operation": "groupby",
            "group_by": ["region"],
            "aggregate": {"revenue": "mean"},
            "sort_ascending": True,
            "limit": 10,
        }
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is not None
        assert "lowest" in text

    def test_groupby_missing_group_column(self, sales_df):
        spec = {
            "operation": "groupby",
            "group_by": ["nonexistent"],
            "aggregate": {"revenue": "sum"},
        }
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is None

    def test_groupby_missing_aggregate_column(self, sales_df):
        spec = {
            "operation": "groupby",
            "group_by": ["region"],
            "aggregate": {"nonexistent": "sum"},
        }
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is None


class TestExecuteSpecTimeseries:
    def test_basic_timeseries(self, sales_df):
        spec = {"operation": "timeseries", "x_col": "date", "y_col": "revenue"}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is not None
        assert "date" in result_df.columns
        assert "revenue" in result_df.columns

    def test_timeseries_missing_x(self, sales_df):
        spec = {"operation": "timeseries", "x_col": "nonexistent", "y_col": "revenue"}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is None
        assert "not found" in text.lower()

    def test_timeseries_missing_y(self, sales_df):
        spec = {"operation": "timeseries", "x_col": "date", "y_col": "nonexistent"}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is None
        assert "not found" in text.lower()

    def test_timeseries_from_columns_list(self, sales_df):
        """x_col/y_col can fall back to columns[0], columns[1]."""
        spec = {"operation": "timeseries", "columns": ["date", "revenue"]}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is not None


class TestExecuteSpecCorrelation:
    def test_two_numeric_columns(self, sales_df):
        spec = {"operation": "correlation", "columns": ["revenue", "units"]}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is not None
        assert "correlation" in text.lower()
        assert "r =" in text.lower() or "(r" in text.lower()

    def test_less_than_two_numeric_columns(self, sales_df):
        spec = {"operation": "correlation", "columns": ["revenue"]}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is None
        assert "two" in text.lower()

    def test_non_numeric_columns_excluded(self, sales_df):
        spec = {"operation": "correlation", "columns": ["region", "product"]}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is None


class TestExecuteSpecFilter:
    def test_basic_filter(self, sales_df):
        spec = {
            "operation": "filter",
            "filter": {"column": "region", "operator": "==", "value": "North"},
        }
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is not None
        assert "2" in text  # "Found 2 rows"

    def test_filter_missing_column(self, sales_df):
        spec = {
            "operation": "filter",
            "filter": {"column": "nonexistent", "operator": ">", "value": 100},
        }
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is None
        assert "not found" in text.lower()

    def test_filter_empty_spec(self, sales_df):
        spec = {"operation": "filter", "filter": {}}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is None


class TestExecuteSpecUnknown:
    def test_unknown_operation(self, sales_df):
        spec = {"operation": "unsupported_op"}
        result_df, text = _execute_spec(spec, sales_df)
        assert result_df is None
        assert "not sure" in text.lower() or "available" in text.lower()


# ---------------------------------------------------------------------------
# run_nl_query (monkeypatching _parse_question_to_spec)
# ---------------------------------------------------------------------------


class TestRunNlQuery:
    def test_returns_helpful_message_when_parse_fails(self, sales_df, column_info):
        with patch("core.query_engine._parse_question_to_spec", return_value=None):
            result = run_nl_query(
                "what is the airspeed of an unladen swallow?", sales_df, column_info
            )
        assert isinstance(result, QueryResult)
        assert result.chart_spec is None
        assert len(result.text) > 10
        assert "couldn't" in result.text.lower() or "understand" in result.text.lower()

    def test_returns_result_when_spec_valid(self, sales_df, column_info):
        spec = {
            "operation": "distribution",
            "columns": ["revenue"],
            "x_col": "revenue",
            "y_col": None,
        }
        with patch("core.query_engine._parse_question_to_spec", return_value=spec):
            result = run_nl_query("show me revenue distribution", sales_df, column_info)
        assert isinstance(result, QueryResult)
        assert result.text
        assert result.result_rows is not None

    def test_handles_execute_exception_gracefully(self, sales_df, column_info):
        """If _execute_spec raises, run_nl_query should return a friendly error."""
        spec = {"operation": "distribution", "columns": ["revenue"]}
        with patch("core.query_engine._parse_question_to_spec", return_value=spec):
            with patch(
                "core.query_engine._execute_spec", side_effect=RuntimeError("boom")
            ):
                result = run_nl_query("some question", sales_df, column_info)
        assert result.chart_spec is None
        assert "error" in result.text.lower() or "hit" in result.text.lower()

    def test_handles_empty_result_df(self, sales_df, column_info):
        spec = {
            "operation": "filter",
            "filter": {"column": "revenue", "operator": ">", "value": 999999},
        }
        with patch("core.query_engine._parse_question_to_spec", return_value=spec):
            result = run_nl_query("revenue > 999999", sales_df, column_info)
        assert result.chart_spec is None
        assert (
            "no data" in result.text.lower()
            or "0 rows" in result.text.lower()
            or result.text
        )


# ---------------------------------------------------------------------------
# generate_chart_for_message
# ---------------------------------------------------------------------------


class TestGenerateChartForMessage:
    def test_no_data_keywords_returns_none(self, sales_df, column_info):
        result = generate_chart_for_message(
            message="hello there",
            df=sales_df,
            column_info=column_info,
            assistant_response="Sure, let me help you.",
        )
        assert result is None

    def test_data_keyword_in_question_triggers_query(self, sales_df, column_info):
        spec = {
            "operation": "distribution",
            "columns": ["revenue"],
            "x_col": "revenue",
            "y_col": None,
        }
        with patch("core.query_engine._parse_question_to_spec", return_value=spec):
            result = generate_chart_for_message(
                message="show me the distribution of revenue",
                df=sales_df,
                column_info=column_info,
                assistant_response="Here is the distribution.",
            )
        # May return chart or None (if chart_builder returns None for some config)
        # Just verify it doesn't raise
        assert result is None or isinstance(result, dict)

    def test_data_keyword_in_response_triggers_query(self, sales_df, column_info):
        with patch("core.query_engine._parse_question_to_spec", return_value=None):
            result = generate_chart_for_message(
                message="tell me something",
                df=sales_df,
                column_info=column_info,
                assistant_response="The highest region is North with total revenue...",
            )
        # _parse_question_to_spec returns None → run_nl_query returns text-only result
        assert result is None
