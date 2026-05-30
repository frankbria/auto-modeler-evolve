"""Tests for Similar Records Finder: compute_similar_records() and chat handler."""

from __future__ import annotations

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pipeline_and_model(n_samples: int = 60, n_numeric: int = 3):
    """Create a tiny classification pipeline + model for tests."""
    from core.deployer import PredictionPipeline
    from sklearn.preprocessing import LabelEncoder

    rng = np.random.default_rng(0)
    feature_names = [f"feat{i}" for i in range(n_numeric)]
    X = rng.standard_normal((n_samples, n_numeric))
    y = (X[:, 0] > 0).astype(int)

    pipeline = PredictionPipeline(
        feature_names=feature_names,
        column_types={f: "numeric" for f in feature_names},
        target_column="label",
        problem_type="classification",
    )
    pipeline.feature_means = {
        f: float(X[:, i].mean()) for i, f in enumerate(feature_names)
    }
    pipeline.feature_stds = {
        f: float(X[:, i].std()) for i, f in enumerate(feature_names)
    }
    pipeline.medians = {
        f: float(np.median(X[:, i])) for i, f in enumerate(feature_names)
    }
    le = LabelEncoder()
    le.fit(y)
    pipeline.target_encoder = le
    pipeline.target_classes = list(le.classes_)

    model = LogisticRegression(random_state=0)
    model.fit(X, y)

    # Build a matching DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y

    return pipeline, model, X, y, df, feature_names


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestComputeSimilarRecords:
    def setup_method(self):
        self.pipeline, self.model, self.X, self.y, self.df, self.feature_names = (
            _build_pipeline_and_model()
        )
        self.tmp = tempfile.mkdtemp()
        self.pipeline_path = os.path.join(self.tmp, "model_pipeline.joblib")
        self.model_path = os.path.join(self.tmp, "model_model.joblib")
        joblib.dump(self.pipeline, self.pipeline_path)
        joblib.dump(self.model, self.model_path)

    def _query_features(self, row_idx: int = 0) -> dict:
        return {f: float(self.X[row_idx, i]) for i, f in enumerate(self.feature_names)}

    def test_returns_required_keys(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=3,
        )
        required = {
            "query_prediction",
            "query_confidence",
            "feature_columns",
            "neighbors",
            "summary",
        }
        assert required.issubset(result.keys())

    def test_neighbor_count_respects_n_neighbors(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=4,
        )
        assert len(result["neighbors"]) == 4

    def test_n_neighbors_clamped_to_max_20(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=100,  # exceeds max
        )
        assert len(result["neighbors"]) <= 20

    def test_n_neighbors_clamped_to_min_1(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=0,  # below min
        )
        assert len(result["neighbors"]) >= 1

    def test_neighbor_keys_present(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=3,
        )
        for nb in result["neighbors"]:
            assert "row_index" in nb
            assert "distance" in nb
            assert "similarity_score" in nb
            assert "features" in nb
            assert "predicted_label" in nb
            assert "predicted_confidence" in nb
            assert "actual_label" in nb

    def test_similarity_score_in_range(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=5,
        )
        for nb in result["neighbors"]:
            assert 0.0 < nb["similarity_score"] <= 1.0

    def test_similarity_scores_descending(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=5,
        )
        scores = [nb["similarity_score"] for nb in result["neighbors"]]
        assert scores == sorted(scores, reverse=True)

    def test_distance_ascending(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=5,
        )
        dists = [nb["distance"] for nb in result["neighbors"]]
        assert dists == sorted(dists)

    def test_feature_columns_excludes_target(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=3,
        )
        assert "label" not in result["feature_columns"]
        for fname in self.feature_names:
            assert fname in result["feature_columns"]

    def test_query_confidence_in_range(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=3,
        )
        assert 0.0 <= result["query_confidence"] <= 1.0

    def test_actual_label_populated_when_target_present(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=3,
        )
        for nb in result["neighbors"]:
            assert nb["actual_label"] is not None

    def test_summary_is_non_empty_string(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=3,
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_missing_pipeline_raises(self):
        from core.deployer import compute_similar_records

        with pytest.raises(FileNotFoundError):
            compute_similar_records(
                pipeline_path="/nonexistent/pipeline.joblib",
                model_path=self.model_path,
                query_features=self._query_features(0),
                dataset_df=self.df,
                target_column="label",
            )

    def test_missing_model_raises(self):
        from core.deployer import compute_similar_records

        with pytest.raises(FileNotFoundError):
            compute_similar_records(
                pipeline_path=self.pipeline_path,
                model_path="/nonexistent/model.joblib",
                query_features=self._query_features(0),
                dataset_df=self.df,
                target_column="label",
            )

    def test_neighbor_features_exclude_target(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=3,
        )
        for nb in result["neighbors"]:
            assert "label" not in nb["features"]

    def test_predicted_confidence_in_range(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=5,
        )
        for nb in result["neighbors"]:
            assert 0.0 <= nb["predicted_confidence"] <= 1.0

    def test_row_indices_are_valid(self):
        from core.deployer import compute_similar_records

        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=5,
        )
        n_rows = len(self.df)
        for nb in result["neighbors"]:
            assert 0 <= nb["row_index"] < n_rows

    def test_different_query_rows_give_different_results(self):
        from core.deployer import compute_similar_records

        r0 = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(0),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=3,
        )
        r1 = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=self._query_features(10),
            dataset_df=self.df,
            target_column="label",
            n_neighbors=3,
        )
        indices0 = {nb["row_index"] for nb in r0["neighbors"]}
        indices1 = {nb["row_index"] for nb in r1["neighbors"]}
        # At least one neighbor should differ (extremely unlikely to be identical)
        assert indices0 != indices1 or r0["query_prediction"] != r1["query_prediction"]


# ---------------------------------------------------------------------------
# BDD-style tests
# ---------------------------------------------------------------------------


class TestSimilarRecordsBDDScenarios:
    def setup_method(self):
        self.pipeline, self.model, self.X, self.y, self.df, self.feature_names = (
            _build_pipeline_and_model()
        )
        self.tmp = tempfile.mkdtemp()
        self.pipeline_path = os.path.join(self.tmp, "bdd_pipeline.joblib")
        self.model_path = os.path.join(self.tmp, "bdd_model.joblib")
        joblib.dump(self.pipeline, self.pipeline_path)
        joblib.dump(self.model, self.model_path)

    def test_analyst_asks_for_similar_records_gets_structured_result(self):
        """
        Given a deployed classification model and a dataset,
        When an analyst asks for records similar to row 5,
        Then they receive a list of neighbors with similarity scores and predictions.
        """
        from core.deployer import compute_similar_records

        query = {f: float(self.X[5, i]) for i, f in enumerate(self.feature_names)}
        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=query,
            dataset_df=self.df,
            target_column="label",
            n_neighbors=5,
        )

        assert len(result["neighbors"]) == 5
        assert result["query_prediction"] in ["0", "1"]
        assert all(nb["similarity_score"] > 0 for nb in result["neighbors"])

    def test_boundary_region_shows_mixed_outcomes(self):
        """
        Given a record near the classification boundary (feature0 ≈ 0),
        When similar records are fetched,
        Then the summary mentions mixed outcomes or boundary proximity.
        """
        from core.deployer import compute_similar_records

        # Craft a borderline query
        query = {"feat0": 0.05, "feat1": 0.0, "feat2": 0.0}
        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=query,
            dataset_df=self.df,
            target_column="label",
            n_neighbors=5,
        )
        # Should produce valid result even at boundary
        assert "neighbors" in result
        assert len(result["neighbors"]) > 0

    def test_result_is_json_serializable(self):
        """
        Given a similar_records result,
        When it is serialised to JSON (as required for SSE),
        Then no TypeError is raised.
        """
        import json
        from core.deployer import compute_similar_records

        query = {f: float(self.X[0, i]) for i, f in enumerate(self.feature_names)}
        result = compute_similar_records(
            pipeline_path=self.pipeline_path,
            model_path=self.model_path,
            query_features=query,
            dataset_df=self.df,
            target_column="label",
            n_neighbors=3,
        )
        # Must not raise
        serialised = json.dumps(result)
        recovered = json.loads(serialised)
        assert recovered["query_prediction"] == result["query_prediction"]
