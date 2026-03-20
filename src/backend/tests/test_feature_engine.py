"""Tests for core/feature_engine.py — Phase 3 feature engineering."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from core.feature_engine import (
    apply_transformations,
    compute_feature_importance,
    detect_problem_type,
    suggest_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sales_df():
    """Small sales DataFrame representative of realistic business data."""
    return pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-08",
                "2024-01-15",
                "2024-01-22",
                "2024-01-29",
            ],
            "product": ["Widget A", "Widget B", "Widget A", "Widget C", "Widget B"],
            "region": ["North", "South", "East", "West", "North"],
            "revenue": [1200.50, 850.00, 2100.75, 450.25, 1650.00],
            "units": [10, 8, 18, 4, 15],
        }
    )


@pytest.fixture
def sales_column_stats(sales_df):
    """Column stats dict matching what analyzer.py produces."""
    from core.analyzer import analyze_dataframe

    result = analyze_dataframe(sales_df)
    return result["columns"]


@pytest.fixture
def skewed_df():
    """DataFrame with a heavily right-skewed numeric column."""
    rng = np.random.default_rng(42)
    values = np.concatenate([rng.exponential(scale=1, size=90), [100, 200, 500]])
    return pd.DataFrame({"amount": values})


@pytest.fixture
def categorical_df():
    """DataFrame with both low- and high-cardinality categoricals."""
    low = ["red", "green", "blue", "red", "green"] * 4
    high = [f"item_{i}" for i in range(20)]
    return pd.DataFrame(
        {
            "color": low,
            "sku": high,
            "price": np.random.default_rng(0).uniform(10, 100, 20),
        }
    )


# ---------------------------------------------------------------------------
# suggest_features
# ---------------------------------------------------------------------------


class TestSuggestFeatures:
    def test_date_column_yields_decompose(self, sales_df, sales_column_stats):
        suggestions = suggest_features(sales_df, sales_column_stats)
        date_sugg = [s for s in suggestions if s.transform_type == "date_decompose"]
        assert len(date_sugg) >= 1
        assert date_sugg[0].column == "date"
        assert "date_year" in date_sugg[0].preview_columns
        assert "date_month" in date_sugg[0].preview_columns

    def test_skewed_column_yields_log(self, skewed_df):
        from core.analyzer import analyze_dataframe

        stats = analyze_dataframe(skewed_df)["columns"]
        suggestions = suggest_features(skewed_df, stats)
        log_sugg = [s for s in suggestions if s.transform_type == "log_transform"]
        assert len(log_sugg) >= 1
        assert log_sugg[0].column == "amount"
        assert "amount_log" in log_sugg[0].preview_columns

    def test_low_cardinality_categorical_yields_one_hot(self, categorical_df):
        from core.analyzer import analyze_dataframe

        stats = analyze_dataframe(categorical_df)["columns"]
        suggestions = suggest_features(categorical_df, stats)
        oh_sugg = [s for s in suggestions if s.transform_type == "one_hot"]
        assert any(s.column == "color" for s in oh_sugg)

    def test_high_cardinality_categorical_yields_label_encode(self, categorical_df):
        from core.analyzer import analyze_dataframe

        stats = analyze_dataframe(categorical_df)["columns"]
        suggestions = suggest_features(categorical_df, stats)
        le_sugg = [s for s in suggestions if s.transform_type == "label_encode"]
        assert any(s.column == "sku" for s in le_sugg)

    def test_correlated_numeric_yields_interaction(self):
        rng = np.random.default_rng(1)
        x = rng.normal(100, 20, 50)
        df = pd.DataFrame({"price": x, "cost": x * 0.6 + rng.normal(0, 2, 50)})
        from core.analyzer import analyze_dataframe

        stats = analyze_dataframe(df)["columns"]
        suggestions = suggest_features(df, stats)
        inter_sugg = [s for s in suggestions if s.transform_type == "interaction"]
        assert len(inter_sugg) >= 1

    def test_returns_list(self, sales_df, sales_column_stats):
        result = suggest_features(sales_df, sales_column_stats)
        assert isinstance(result, list)

    def test_suggestions_are_json_serializable(self, sales_df, sales_column_stats):
        suggestions = suggest_features(sales_df, sales_column_stats)
        for s in suggestions:
            data = {
                "id": s.id,
                "column": s.column,
                "transform_type": s.transform_type,
                "title": s.title,
                "description": s.description,
                "preview_columns": s.preview_columns,
                "example_values": s.example_values,
            }
            json.dumps(data)  # must not raise


# ---------------------------------------------------------------------------
# apply_transformations
# ---------------------------------------------------------------------------


class TestApplyTransformations:
    def test_date_decompose(self, sales_df):
        transforms = [{"column": "date", "transform_type": "date_decompose"}]
        result, mapping = apply_transformations(sales_df, transforms)
        assert "date_year" in result.columns
        assert "date_month" in result.columns
        assert "date_dayofweek" in result.columns
        assert "date_is_weekend" in result.columns
        assert result["date_year"].iloc[0] == 2024
        assert result["date_month"].iloc[0] == 1

    def test_log_transform(self, sales_df):
        transforms = [{"column": "revenue", "transform_type": "log_transform"}]
        result, mapping = apply_transformations(sales_df, transforms)
        assert "revenue_log" in result.columns
        assert result["revenue_log"].iloc[0] == pytest.approx(math.log1p(1200.50))
        assert "revenue" in mapping
        assert "revenue_log" in mapping["revenue"]

    def test_bin_quartile(self, sales_df):
        transforms = [{"column": "revenue", "transform_type": "bin_quartile"}]
        result, mapping = apply_transformations(sales_df, transforms)
        assert "revenue_quartile" in result.columns
        assert mapping["revenue"] == ["revenue_quartile"]

    def test_one_hot(self, sales_df):
        transforms = [{"column": "region", "transform_type": "one_hot"}]
        result, mapping = apply_transformations(sales_df, transforms)
        new_cols = mapping.get("region", [])
        assert len(new_cols) >= 2
        for col in new_cols:
            assert col.startswith("region_")
            assert set(result[col].unique()).issubset({0, 1})

    def test_label_encode(self, sales_df):
        transforms = [{"column": "product", "transform_type": "label_encode"}]
        result, mapping = apply_transformations(sales_df, transforms)
        assert "product_encoded" in result.columns
        assert pd.api.types.is_integer_dtype(result["product_encoded"])
        assert result["product_encoded"].min() >= 0

    def test_interaction(self, sales_df):
        transforms = [{"column": "revenue__units", "transform_type": "interaction"}]
        result, mapping = apply_transformations(sales_df, transforms)
        assert "revenue_x_units" in result.columns
        expected = sales_df["revenue"] * sales_df["units"]
        pd.testing.assert_series_equal(
            result["revenue_x_units"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_original_df_not_mutated(self, sales_df):
        original_cols = sales_df.columns.tolist()
        transforms = [{"column": "date", "transform_type": "date_decompose"}]
        apply_transformations(sales_df, transforms)
        assert sales_df.columns.tolist() == original_cols

    def test_multiple_transforms(self, sales_df):
        transforms = [
            {"column": "date", "transform_type": "date_decompose"},
            {"column": "region", "transform_type": "one_hot"},
            {"column": "revenue", "transform_type": "log_transform"},
        ]
        result, mapping = apply_transformations(sales_df, transforms)
        assert "date_year" in result.columns
        assert "revenue_log" in result.columns
        assert "date" in mapping and "region" in mapping and "revenue" in mapping

    def test_invalid_transform_type_skipped(self, sales_df):
        transforms = [{"column": "revenue", "transform_type": "nonexistent_type"}]
        result, mapping = apply_transformations(sales_df, transforms)
        # Should not raise; nonexistent type is silently skipped
        assert result is not None


# ---------------------------------------------------------------------------
# detect_problem_type
# ---------------------------------------------------------------------------


class TestDetectProblemType:
    def test_numeric_high_cardinality_is_regression(self, sales_df):
        result = detect_problem_type(sales_df, "revenue")
        assert result["problem_type"] == "regression"

    def test_string_target_is_classification(self, sales_df):
        result = detect_problem_type(sales_df, "region")
        assert result["problem_type"] == "classification"
        assert isinstance(result["classes"], list)

    def test_low_cardinality_int_is_classification(self):
        df = pd.DataFrame({"rating": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]})
        result = detect_problem_type(df, "rating")
        assert result["problem_type"] == "classification"

    def test_boolean_target_is_classification(self):
        df = pd.DataFrame({"churned": [True, False, True, False, True]})
        result = detect_problem_type(df, "churned")
        assert result["problem_type"] == "classification"

    def test_missing_column_returns_none_type(self, sales_df):
        result = detect_problem_type(sales_df, "nonexistent")
        assert result["problem_type"] is None

    def test_result_has_reason(self, sales_df):
        result = detect_problem_type(sales_df, "revenue")
        assert "reason" in result
        assert len(result["reason"]) > 10


# ---------------------------------------------------------------------------
# compute_feature_importance
# ---------------------------------------------------------------------------


class TestComputeFeatureImportance:
    def test_returns_ranked_list(self, sales_df, sales_column_stats):
        importance = compute_feature_importance(
            sales_df, "revenue", "regression", sales_column_stats
        )
        assert isinstance(importance, list)
        assert len(importance) > 0
        assert importance[0]["rank"] == 1
        assert importance[0]["importance"] >= importance[-1]["importance"]

    def test_target_column_excluded(self, sales_df, sales_column_stats):
        importance = compute_feature_importance(
            sales_df, "revenue", "regression", sales_column_stats
        )
        columns = [r["column"] for r in importance]
        assert "revenue" not in columns

    def test_classification_target_works(self, sales_df, sales_column_stats):
        importance = compute_feature_importance(
            sales_df, "region", "classification", sales_column_stats
        )
        assert isinstance(importance, list)
        assert len(importance) > 0

    def test_each_entry_has_required_fields(self, sales_df, sales_column_stats):
        importance = compute_feature_importance(
            sales_df, "revenue", "regression", sales_column_stats
        )
        for entry in importance:
            assert "column" in entry
            assert "importance" in entry
            assert "importance_pct" in entry
            assert "rank" in entry
            assert "description" in entry

    def test_importance_pct_sums_to_100(self, sales_df, sales_column_stats):
        importance = compute_feature_importance(
            sales_df, "revenue", "regression", sales_column_stats
        )
        total_pct = sum(r["importance_pct"] for r in importance)
        assert abs(total_pct - 100.0) < 1.0  # within 1% due to rounding


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestFeatureAPI:
    def test_suggestions_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Feature Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            resp = await client.get(f"/api/features/{dataset_id}/suggestions")
            assert resp.status_code == 200
            body = resp.json()
            assert "suggestions" in body
            assert isinstance(body["suggestions"], list)

        asyncio.get_event_loop().run_until_complete(run())

    def test_apply_and_preview_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Apply Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            apply_resp = await client.post(
                f"/api/features/{dataset_id}/apply",
                json={
                    "transformations": [
                        {"column": "date", "transform_type": "date_decompose"},
                        {"column": "region", "transform_type": "one_hot"},
                    ]
                },
            )
            assert apply_resp.status_code == 201
            body = apply_resp.json()
            assert "feature_set_id" in body
            assert "new_columns" in body
            assert len(body["new_columns"]) > 0

            # Preview the feature set
            fs_id = body["feature_set_id"]
            preview_resp = await client.get(f"/api/features/{fs_id}/preview")
            assert preview_resp.status_code == 200
            preview = preview_resp.json()
            assert "date_year" in preview["columns"]

        asyncio.get_event_loop().run_until_complete(run())

    def test_target_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Target Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            resp = await client.post(
                f"/api/features/{dataset_id}/target",
                json={"target_column": "revenue"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["problem_type"] == "regression"
            assert "reason" in body

        asyncio.get_event_loop().run_until_complete(run())

    def test_importance_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj = await client.post("/api/projects", json={"name": "Importance Test"})
            project_id = proj.json()["id"]

            upload = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload.json()["dataset_id"]

            resp = await client.get(
                f"/api/features/{dataset_id}/importance",
                params={"target_column": "revenue"},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "features" in body
            assert body["problem_type"] == "regression"
            assert len(body["features"]) > 0

        asyncio.get_event_loop().run_until_complete(run())
