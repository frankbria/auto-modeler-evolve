"""Tests for the Column Type Suggestion feature.

Backend:  compute_column_type_suggestions() pure function + chat SSE integration
SSE type: column_type_suggestions
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(columns: list[dict], row_count: int = 100) -> dict:
    """Build a minimal profile dict for testing."""
    return {
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "correlations": [],
    }


def _make_col(
    name: str,
    dtype: str,
    sample_values: list,
    unique_count: int = 10,
    null_pct: float = 0.0,
) -> dict:
    return {
        "name": name,
        "dtype": dtype,
        "sample_values": sample_values,
        "unique_count": unique_count,
        "null_pct": null_pct,
    }


# ---------------------------------------------------------------------------
# Pure-function tests for compute_column_type_suggestions()
# ---------------------------------------------------------------------------


class TestComputeColumnTypeSuggestions:
    def test_returns_dict_with_expected_keys(self):
        from core.analyzer import compute_column_type_suggestions

        result = compute_column_type_suggestions(_make_profile([]))
        assert "suggestions" in result
        assert "has_suggestions" in result
        assert "dataset_rows" in result
        assert "dataset_cols" in result

    def test_empty_profile_no_suggestions(self):
        from core.analyzer import compute_column_type_suggestions

        result = compute_column_type_suggestions(_make_profile([]))
        assert result["suggestions"] == []
        assert result["has_suggestions"] is False

    def test_numeric_looking_object_column_detected(self):
        """A column with dtype=object but numeric sample values should be flagged."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col(
            "price", "object", ["12.99", "8.50", "3.00", "15.00"], unique_count=50
        )
        result = compute_column_type_suggestions(_make_profile([col]))
        assert result["has_suggestions"] is True
        suggestion = result["suggestions"][0]
        assert suggestion["column"] == "price"
        assert suggestion["suggested_dtype"] == "numeric"
        assert suggestion["confidence"] == "high"

    def test_non_numeric_object_column_not_flagged(self):
        """A text column with real string values should not be flagged as numeric."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col(
            "region", "object", ["North", "South", "East", "West"], unique_count=4
        )
        result = compute_column_type_suggestions(_make_profile([col]))
        # region should not get a type suggestion
        cols_flagged = [s["column"] for s in result["suggestions"]]
        assert "region" not in cols_flagged

    def test_boolean_looking_object_column_detected(self):
        """A column with only true/false values stored as strings should be flagged."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col(
            "is_active", "object", ["true", "false", "true"], unique_count=2
        )
        result = compute_column_type_suggestions(_make_profile([col]))
        assert result["has_suggestions"] is True
        suggestion = result["suggestions"][0]
        assert suggestion["column"] == "is_active"
        assert suggestion["suggested_dtype"] == "boolean"

    def test_yes_no_column_detected_as_boolean(self):
        """Yes/No values should be flagged as boolean."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col("eligible", "object", ["yes", "no", "yes"], unique_count=2)
        result = compute_column_type_suggestions(_make_profile([col]))
        cols_suggested = {
            s["column"]: s["suggested_dtype"] for s in result["suggestions"]
        }
        assert cols_suggested.get("eligible") == "boolean"

    def test_date_column_detected(self):
        """An object column named 'order_date' with date-like values should be flagged."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col(
            "order_date",
            "object",
            ["2023-01-15", "2023-02-20", "2023-03-10"],
            unique_count=90,
        )
        result = compute_column_type_suggestions(_make_profile([col]))
        cols_suggested = {
            s["column"]: s["suggested_dtype"] for s in result["suggestions"]
        }
        assert cols_suggested.get("order_date") == "datetime"

    def test_float_whole_numbers_suggested_as_integer(self):
        """A float64 column where all values are whole numbers should be flagged as integer."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col("units", "float64", [1.0, 2.0, 3.0, 4.0], unique_count=20)
        result = compute_column_type_suggestions(_make_profile([col]))
        cols_suggested = {
            s["column"]: s["suggested_dtype"] for s in result["suggestions"]
        }
        assert cols_suggested.get("units") == "integer"

    def test_float_with_decimals_not_flagged_as_integer(self):
        """A float64 column with actual decimal values should not be flagged as integer."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col("price", "float64", [1.5, 2.7, 3.14, 4.99], unique_count=50)
        result = compute_column_type_suggestions(_make_profile([col]))
        cols_suggested = {
            s["column"]: s["suggested_dtype"] for s in result["suggestions"]
        }
        assert cols_suggested.get("price") != "integer"

    def test_id_column_not_flagged_as_integer(self):
        """An ID column stored as float64 (e.g. customer_id=1.0) should NOT be flagged."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col("customer_id", "float64", [1.0, 2.0, 3.0], unique_count=100)
        result = compute_column_type_suggestions(_make_profile([col]))
        cols_flagged = [s["column"] for s in result["suggestions"]]
        assert "customer_id" not in cols_flagged

    def test_high_null_column_skipped(self):
        """A column with >90% nulls provides unreliable samples and should be skipped."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col("sparse", "object", ["12.99"], unique_count=1, null_pct=95.0)
        result = compute_column_type_suggestions(_make_profile([col]))
        cols_flagged = [s["column"] for s in result["suggestions"]]
        assert "sparse" not in cols_flagged

    def test_suggestion_includes_sample_values(self):
        """Each suggestion should include sample_values from the column."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col("price", "object", ["12.99", "8.50", "3.00"], unique_count=50)
        result = compute_column_type_suggestions(_make_profile([col]))
        assert result["has_suggestions"] is True
        assert len(result["suggestions"][0]["sample_values"]) > 0

    def test_suggestion_includes_suggested_action(self):
        """Each suggestion must have a suggested_action string for the chat action button."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col("price", "object", ["12.99", "8.50", "3.00"], unique_count=50)
        result = compute_column_type_suggestions(_make_profile([col]))
        assert result["has_suggestions"] is True
        action = result["suggestions"][0]["suggested_action"]
        assert isinstance(action, str) and len(action) > 0

    def test_dataset_rows_and_cols_returned(self):
        """The result should include dataset_rows and dataset_cols counts."""
        from core.analyzer import compute_column_type_suggestions

        cols = [
            _make_col("a", "object", ["1", "2"]),
            _make_col("b", "float64", [1.0, 2.0]),
        ]
        result = compute_column_type_suggestions(_make_profile(cols, row_count=200))
        assert result["dataset_rows"] == 200
        assert result["dataset_cols"] == 2

    def test_numeric_with_comma_separator_detected(self):
        """Values like '1,234' should be parseable as numeric."""
        from core.analyzer import compute_column_type_suggestions

        col = _make_col(
            "revenue", "object", ["1,234", "5,678", "9,012"], unique_count=50
        )
        result = compute_column_type_suggestions(_make_profile([col]))
        cols_suggested = {
            s["column"]: s["suggested_dtype"] for s in result["suggestions"]
        }
        assert cols_suggested.get("revenue") == "numeric"


# ---------------------------------------------------------------------------
# Chat SSE integration tests
# ---------------------------------------------------------------------------


class TestColumnTypeSuggestionsChat:
    """Integration tests for the column_type_suggestions SSE event in send_message()."""

    def _make_client(self, monkeypatch):
        """Build a TestClient with minimal mocks for chat endpoint testing."""
        import sys
        import os

        for mod in list(sys.modules.keys()):
            if "automodeler" in mod or (
                mod.startswith(("api.", "core.", "models.", "chat.", "db"))
                and "site-packages" not in mod
            ):
                sys.modules.pop(mod, None)

        os.environ["DATABASE_URL"] = "sqlite://"

        from fastapi.testclient import TestClient
        from sqlmodel import SQLModel, create_engine

        test_engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}
        )

        import db as db_mod

        monkeypatch.setattr(db_mod, "engine", test_engine)

        # Create tables
        SQLModel.metadata.create_all(test_engine)
        db_mod._apply_migrations()

        from main import app

        client = TestClient(app, raise_server_exceptions=True)
        return client, test_engine

    def test_explicit_trigger_pattern_fires(self):
        """'check my column types' should match the regex pattern."""
        import sys

        for mod in list(sys.modules.keys()):
            if mod.startswith("api.chat") or mod == "api.chat":
                sys.modules.pop(mod, None)
        from api.chat import _COLUMN_TYPE_PATTERNS

        assert _COLUMN_TYPE_PATTERNS.search("check my column types")
        assert _COLUMN_TYPE_PATTERNS.search("are my data types correct?")
        assert _COLUMN_TYPE_PATTERNS.search("any type issues in my data?")
        assert _COLUMN_TYPE_PATTERNS.search("fix column types")
        assert _COLUMN_TYPE_PATTERNS.search("column types look wrong")

    def test_pattern_no_false_positives(self):
        """Normal chat messages should not match the column type pattern."""
        from api.chat import _COLUMN_TYPE_PATTERNS

        assert not _COLUMN_TYPE_PATTERNS.search("train a model on this data")
        assert not _COLUMN_TYPE_PATTERNS.search("show me a scatter plot")
        assert not _COLUMN_TYPE_PATTERNS.search("what is the average revenue")
