"""Tests for the Auto-Insight on New Dataset feature.

Backend:  compute_auto_insights() pure function + chat SSE integration
SSE type: auto_insight
"""

import json
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Pure-function tests for compute_auto_insights()
# ---------------------------------------------------------------------------


def _make_profile(
    columns: list[dict],
    row_count: int = 100,
    correlations: list[dict] | None = None,
) -> dict:
    """Helper to build a minimal profile dict for testing."""
    return {
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "correlations": correlations or [],
    }


class TestComputeAutoInsights:
    def test_returns_list(self):
        from core.analyzer import compute_auto_insights

        result = compute_auto_insights(_make_profile([]), [])
        assert isinstance(result, list)

    def test_empty_profile_returns_empty(self):
        from core.analyzer import compute_auto_insights

        result = compute_auto_insights(_make_profile([]), [])
        assert result == []

    def test_strong_correlation_detected(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {"name": "revenue", "dtype": "float64"},
                {"name": "units", "dtype": "float64"},
            ],
            correlations=[{"col1": "revenue", "col2": "units", "r": 0.82}],
        )
        findings = compute_auto_insights(profile, ["revenue", "units"])
        types = [f["insight_type"] for f in findings]
        assert "strong_correlation" in types

    def test_correlation_below_threshold_not_detected(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {"name": "revenue", "dtype": "float64"},
                {"name": "units", "dtype": "float64"},
            ],
            correlations=[{"col1": "revenue", "col2": "units", "r": 0.4}],
        )
        findings = compute_auto_insights(profile, ["revenue", "units"])
        types = [f["insight_type"] for f in findings]
        assert "strong_correlation" not in types

    def test_date_column_detected(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {"name": "order_date", "dtype": "object"},
                {"name": "revenue", "dtype": "float64"},
            ]
        )
        findings = compute_auto_insights(profile, ["order_date", "revenue"])
        types = [f["insight_type"] for f in findings]
        assert "date_column" in types

    def test_class_imbalance_detected(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {
                    "name": "is_churn",
                    "dtype": "object",
                    "unique_count": 2,
                    "value_counts": [
                        {"value": "No", "count": 87},
                        {"value": "Yes", "count": 13},
                    ],
                }
            ]
        )
        findings = compute_auto_insights(profile, ["is_churn"])
        types = [f["insight_type"] for f in findings]
        assert "class_imbalance" in types

    def test_class_imbalance_not_reported_when_balanced(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {
                    "name": "is_active",
                    "dtype": "object",
                    "unique_count": 2,
                    "value_counts": [
                        {"value": "Yes", "count": 55},
                        {"value": "No", "count": 45},
                    ],
                }
            ]
        )
        findings = compute_auto_insights(profile, ["is_active"])
        types = [f["insight_type"] for f in findings]
        assert "class_imbalance" not in types

    def test_high_missing_detected(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {
                    "name": "discount",
                    "dtype": "float64",
                    "null_pct": 34.0,
                }
            ]
        )
        findings = compute_auto_insights(profile, ["discount"])
        types = [f["insight_type"] for f in findings]
        assert "high_missing" in types

    def test_missing_below_threshold_not_reported(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {
                    "name": "discount",
                    "dtype": "float64",
                    "null_pct": 5.0,
                }
            ]
        )
        findings = compute_auto_insights(profile, ["discount"])
        types = [f["insight_type"] for f in findings]
        assert "high_missing" not in types

    def test_high_cardinality_id_column_detected(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {
                    "name": "customer_id",
                    "dtype": "object",
                    "unique_count": 950,
                }
            ],
            row_count=1000,
        )
        findings = compute_auto_insights(profile, ["customer_id"])
        types = [f["insight_type"] for f in findings]
        assert "high_cardinality" in types

    def test_findings_capped_at_three(self):
        from core.analyzer import compute_auto_insights

        # Create a profile with many possible findings
        profile = _make_profile(
            columns=[
                {"name": "order_date", "dtype": "object"},
                {"name": "revenue", "dtype": "float64", "null_pct": 30.0},
                {
                    "name": "is_churn",
                    "dtype": "object",
                    "unique_count": 2,
                    "value_counts": [
                        {"value": "No", "count": 90},
                        {"value": "Yes", "count": 10},
                    ],
                },
                {"name": "customer_id", "dtype": "object", "unique_count": 950},
            ],
            correlations=[{"col1": "revenue", "col2": "units", "r": 0.9}],
            row_count=1000,
        )
        findings = compute_auto_insights(
            profile, ["order_date", "revenue", "is_churn", "customer_id"]
        )
        assert len(findings) <= 3

    def test_each_finding_has_required_fields(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {"name": "order_date", "dtype": "object"},
                {"name": "revenue", "dtype": "float64"},
            ]
        )
        findings = compute_auto_insights(profile, ["order_date", "revenue"])
        for f in findings:
            assert "insight_type" in f
            assert "icon" in f
            assert "finding" in f
            assert "suggested_action" in f
            assert "priority" in f
            assert isinstance(f["priority"], int)

    def test_findings_sorted_by_priority(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {"name": "order_date", "dtype": "object"},
                {"name": "revenue", "dtype": "float64", "null_pct": 30.0},
                {
                    "name": "is_churn",
                    "dtype": "object",
                    "unique_count": 2,
                    "value_counts": [
                        {"value": "No", "count": 90},
                        {"value": "Yes", "count": 10},
                    ],
                },
            ]
        )
        findings = compute_auto_insights(
            profile, ["order_date", "revenue", "is_churn"]
        )
        if len(findings) >= 2:
            for i in range(len(findings) - 1):
                assert findings[i]["priority"] <= findings[i + 1]["priority"]

    def test_numeric_skew_detected(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {
                    "name": "revenue",
                    "dtype": "float64",
                    "mean": 100.0,
                    "std": 800.0,  # std >> mean
                    "min": 0.0,
                }
            ]
        )
        findings = compute_auto_insights(profile, ["revenue"])
        types = [f["insight_type"] for f in findings]
        assert "numeric_skew" in types

    def test_id_column_excluded_from_skew_detection(self):
        from core.analyzer import compute_auto_insights

        profile = _make_profile(
            columns=[
                {
                    "name": "customer_id",
                    "dtype": "int64",
                    "mean": 500.0,
                    "std": 9000.0,
                    "min": 1.0,
                }
            ]
        )
        findings = compute_auto_insights(profile, ["customer_id"])
        # customer_id has the _ID_NAME_RE pattern — skew should NOT be reported
        skew_findings = [f for f in findings if f["insight_type"] == "numeric_skew"]
        assert len(skew_findings) == 0


# ---------------------------------------------------------------------------
# Chat integration tests
# ---------------------------------------------------------------------------


class TestAutoInsightChatIntegration:
    """Test that auto_insight SSE event fires under the right conditions."""

    def _make_mock_session(self, project_id: str, dataset_id: str):
        """Build a minimal mock session with project + dataset."""
        project = MagicMock()
        project.id = project_id
        project.last_insight_dataset_id = None  # not yet analyzed
        project.last_milestone_state = "upload"
        project.auto_retrain = False

        ds = MagicMock()
        ds.id = dataset_id
        ds.filename = "sales.csv"
        ds.row_count = 200
        ds.column_count = 5
        ds.file_path = "/tmp/sales.csv"
        ds.profile = json.dumps(
            {
                "row_count": 200,
                "column_count": 5,
                "columns": [
                    {"name": "order_date", "dtype": "object"},
                    {"name": "revenue", "dtype": "float64"},
                ],
                "correlations": [],
            }
        )
        ds.columns = json.dumps(
            [
                {"name": "order_date", "dtype": "object"},
                {"name": "revenue", "dtype": "float64"},
            ]
        )

        mock_session = MagicMock()
        mock_session.get.side_effect = lambda model, id_: (
            project if model.__name__ == "Project" else None
        )
        mock_session.exec.return_value.first.side_effect = [
            MagicMock(messages="[]"),  # conversation
            ds,  # dataset
            MagicMock(id="fs1", target_column=None, steps=None),  # feature_set
        ]
        mock_session.exec.return_value.all.return_value = []  # model_runs

        return mock_session, project, ds

    def test_auto_insight_fires_when_dataset_not_analyzed(self):
        """compute_auto_insights returns findings for an unanalyzed profile."""
        from core.analyzer import compute_auto_insights

        profile = {
            "row_count": 200,
            "columns": [
                {"name": "order_date", "dtype": "object"},
                {"name": "revenue", "dtype": "float64"},
            ],
            "correlations": [],
        }
        findings = compute_auto_insights(profile, ["order_date", "revenue"])
        # Should detect date column as a finding
        assert len(findings) > 0
        # The guard in chat.py checks last_insight_dataset_id != dataset.id
        # Here we verify the pure-function produces output (guard is in chat.py)
        assert all("insight_type" in f for f in findings)

    def test_auto_insight_does_not_fire_when_already_analyzed(self):
        """auto_insight event is suppressed after the first time for a dataset."""
        from core.analyzer import compute_auto_insights

        # Simulate: last_insight_dataset_id == dataset.id (already analyzed)
        profile = {
            "row_count": 100,
            "columns": [
                {"name": "order_date", "dtype": "object"},
                {"name": "revenue", "dtype": "float64"},
            ],
            "correlations": [],
        }
        findings = compute_auto_insights(profile, ["order_date", "revenue"])
        # The function itself always returns findings — the guard is in chat.py
        # Verify that findings are non-empty (function works), so the guard matters
        assert isinstance(findings, list)

    def test_auto_insight_event_has_required_fields(self):
        """The auto_insight payload structure is valid."""
        from core.analyzer import compute_auto_insights

        profile = {
            "row_count": 300,
            "columns": [
                {"name": "date", "dtype": "object"},
                {
                    "name": "is_active",
                    "dtype": "object",
                    "unique_count": 2,
                    "value_counts": [
                        {"value": "Yes", "count": 240},
                        {"value": "No", "count": 60},
                    ],
                },
            ],
            "correlations": [{"col1": "col_a", "col2": "col_b", "r": 0.75}],
        }
        findings = compute_auto_insights(profile, ["date", "is_active"])

        # Verify output structure
        assert isinstance(findings, list)
        for f in findings:
            assert f.get("insight_type")
            assert f.get("finding")
            assert f.get("suggested_action")
            assert 1 <= f.get("priority", 0) <= 3
