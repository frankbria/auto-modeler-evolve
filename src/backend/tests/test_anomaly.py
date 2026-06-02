"""Tests for anomaly detection: core logic + API endpoint + chat pattern."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.anomaly import detect_anomalies

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_numeric_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "revenue": rng.normal(1000, 200, n),
            "quantity": rng.normal(50, 10, n),
            "discount": rng.uniform(0, 0.3, n),
        }
    )
    # Inject obvious anomalies at the end
    df.loc[n - 1, "revenue"] = 50000  # extreme value
    df.loc[n - 2, "quantity"] = -999  # impossible
    return df


# ---------------------------------------------------------------------------
# Core: detect_anomalies
# ---------------------------------------------------------------------------


class TestDetectAnomaliesCore:
    def test_returns_expected_keys(self):
        df = _make_numeric_df()
        result = detect_anomalies(df, features=["revenue", "quantity", "discount"])
        assert set(result.keys()) == {
            "anomaly_count",
            "total_rows",
            "contamination_used",
            "top_anomalies",
            "summary",
            "features_used",
        }

    def test_total_rows_matches_input(self):
        df = _make_numeric_df(80)
        result = detect_anomalies(df, features=["revenue", "quantity"])
        assert result["total_rows"] == 80

    def test_anomaly_count_positive_with_injected_outliers(self):
        df = _make_numeric_df(100)
        result = detect_anomalies(df, features=["revenue", "quantity", "discount"])
        assert result["anomaly_count"] >= 1

    def test_top_anomalies_length_capped_at_n_top(self):
        df = _make_numeric_df(200)
        result = detect_anomalies(df, features=["revenue", "quantity"], n_top=5)
        assert len(result["top_anomalies"]) <= 5

    def test_top_anomaly_record_keys(self):
        df = _make_numeric_df()
        result = detect_anomalies(df, features=["revenue"])
        rec = result["top_anomalies"][0]
        assert "row_index" in rec
        assert "anomaly_score" in rec
        assert "is_anomaly" in rec
        assert "values" in rec

    def test_anomaly_scores_between_0_and_100(self):
        df = _make_numeric_df()
        result = detect_anomalies(df, features=["revenue", "quantity"])
        for rec in result["top_anomalies"]:
            assert 0.0 <= rec["anomaly_score"] <= 100.0

    def test_features_used_only_numeric(self):
        df = _make_numeric_df(60)
        df["label"] = "cat"
        result = detect_anomalies(df, features=["revenue", "label"])
        assert "label" not in result["features_used"]
        assert "revenue" in result["features_used"]

    def test_nonexistent_columns_silently_dropped(self):
        df = _make_numeric_df(60)
        result = detect_anomalies(df, features=["revenue", "nonexistent_col"])
        assert "nonexistent_col" not in result["features_used"]

    def test_raises_if_no_numeric_features(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["NY", "LA"]})
        with pytest.raises(ValueError, match="No numeric features"):
            detect_anomalies(df, features=["name", "city"])

    def test_handles_nan_values_without_crash(self):
        df = _make_numeric_df(60)
        df.loc[5, "revenue"] = np.nan
        result = detect_anomalies(df, features=["revenue", "quantity"])
        assert result["total_rows"] == 60

    def test_all_nan_column_dropped(self):
        df = _make_numeric_df(60)
        df["all_nan"] = np.nan
        result = detect_anomalies(df, features=["revenue", "all_nan"])
        assert "all_nan" not in result["features_used"]

    def test_contamination_clamped_below_0_01(self):
        df = _make_numeric_df(100)
        result = detect_anomalies(df, features=["revenue"], contamination=0.0)
        assert result["contamination_used"] >= 0.01

    def test_contamination_clamped_above_0_5(self):
        df = _make_numeric_df(100)
        result = detect_anomalies(df, features=["revenue"], contamination=0.99)
        assert result["contamination_used"] <= 0.5

    def test_summary_contains_row_count(self):
        df = _make_numeric_df(50)
        result = detect_anomalies(df, features=["revenue"])
        assert "50" in result["summary"]

    def test_zero_anomalies_gives_no_anomalies_message(self):
        # Single constant column → no variance → IsolationForest marks nothing
        # (or very few). We just test that the code doesn't crash.
        df = pd.DataFrame({"x": [1.0] * 50})
        result = detect_anomalies(df, features=["x"], contamination=0.01)
        assert isinstance(result["summary"], str)


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_with_dataset(tmp_path):
    """Create a TestClient with a project + dataset pre-loaded."""
    import api.data as data_module
    from main import app

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    with patch.object(data_module, "UPLOAD_DIR", upload_dir):
        with TestClient(app) as c:
            # Create project
            proj = c.post("/api/projects", json={"name": "AnomalyTest"}).json()
            pid = proj["id"]

            # Upload CSV with obvious anomalies
            df = _make_numeric_df(80)
            csv_bytes = df.to_csv(index=False).encode()
            r = c.post(
                "/api/data/upload",
                data={"project_id": pid},
                files={"file": ("anomaly_test.csv", csv_bytes, "text/csv")},
            )
            did = r.json()["dataset_id"]
            yield c, pid, did


class TestAnomalyEndpoint:
    def test_200_with_valid_request(self, client_with_dataset):
        c, _, did = client_with_dataset
        r = c.post(
            f"/api/data/{did}/anomalies",
            json={"features": ["revenue", "quantity", "discount"]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["dataset_id"] == did
        assert "top_anomalies" in body
        assert body["total_rows"] == 80

    def test_404_unknown_dataset(self, client_with_dataset):
        c, _, _ = client_with_dataset
        r = c.post(
            "/api/data/nonexistent-id/anomalies",
            json={"features": ["revenue"]},
        )
        assert r.status_code == 404

    def test_400_empty_features_list(self, client_with_dataset):
        c, _, did = client_with_dataset
        r = c.post(f"/api/data/{did}/anomalies", json={"features": []})
        assert r.status_code == 400

    def test_400_no_numeric_features(self, client_with_dataset):
        c, pid, _ = client_with_dataset
        # Upload a text-only CSV
        df_text = pd.DataFrame(
            {"name": ["Alice", "Bob", "Carol"] * 20, "city": ["NY", "LA", "SF"] * 20}
        )
        csv_b = df_text.to_csv(index=False).encode()
        r2 = c.post(
            "/api/data/upload",
            data={"project_id": pid},
            files={"file": ("text_only.csv", csv_b, "text/csv")},
        )
        did2 = r2.json()["dataset_id"]
        r = c.post(f"/api/data/{did2}/anomalies", json={"features": ["name", "city"]})
        assert r.status_code == 400

    def test_top_anomalies_count_respects_n_top(self, client_with_dataset):
        c, _, did = client_with_dataset
        r = c.post(
            f"/api/data/{did}/anomalies",
            json={"features": ["revenue", "quantity"], "n_top": 3},
        )
        body = r.json()
        assert len(body["top_anomalies"]) <= 3


# ---------------------------------------------------------------------------
# Chat pattern detection
# ---------------------------------------------------------------------------


class TestAnomalyPatternDetection:
    def test_anomaly_keyword_matches(self):
        from api.chat import _ANOMALY_PATTERNS

        phrases = [
            "find anomalies in my data",
            "are there any unusual records?",
            "detect outliers",
            "suspicious rows",
            "anything odd in the data?",
            "show me anomalies",
            "identify anomalous points",
        ]
        for phrase in phrases:
            assert _ANOMALY_PATTERNS.search(phrase), f"Expected match for: {phrase!r}"

    def test_irrelevant_message_does_not_match(self):
        from api.chat import _ANOMALY_PATTERNS

        phrases = [
            "train a model",
            "what is the average revenue?",
            "show a correlation heatmap",
            "deploy the model",
        ]
        for phrase in phrases:
            assert not _ANOMALY_PATTERNS.search(
                phrase
            ), f"Unexpected match for: {phrase!r}"
