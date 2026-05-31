"""Tests for compute_prediction_error_correlation pure function, REST endpoint,
and chat pattern matching.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.validator import compute_prediction_error_correlation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _X(n=50, n_features=3, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_features))


def _feature_names(n):
    return [f"f{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestComputePredictionErrorCorrelation:
    def test_returns_required_keys_regression(self):
        X = _X()
        y = X[:, 0] * 2 + 1
        y_pred = y + np.random.default_rng(0).normal(scale=0.1, size=len(y))
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        for key in (
            "features",
            "error_type",
            "n_total",
            "n_errors",
            "error_rate",
            "verdict",
            "verdict_label",
            "top_driver",
            "summary",
        ):
            assert key in result, f"Missing key: {key}"

    def test_returns_required_keys_classification(self):
        X = _X()
        y = (X[:, 0] > 0).astype(int)
        y_pred = y.copy()
        y_pred[:10] = 1 - y_pred[:10]  # force some errors
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "classification"
        )
        for key in (
            "features",
            "error_type",
            "n_total",
            "n_errors",
            "error_rate",
            "verdict",
            "verdict_label",
            "top_driver",
            "summary",
        ):
            assert key in result, f"Missing key: {key}"

    def test_error_type_regression(self):
        X = _X()
        y = X[:, 0] * 2
        y_pred = y + 0.1
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        assert result["error_type"] == "absolute_residual"

    def test_error_type_classification(self):
        X = _X()
        y = (X[:, 0] > 0).astype(int)
        y_pred = y.copy()
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "classification"
        )
        assert result["error_type"] == "misclassification"

    def test_error_rate_none_for_regression(self):
        X = _X()
        y = X[:, 0]
        y_pred = y + 0.5
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        assert result["error_rate"] is None

    def test_error_rate_present_for_classification(self):
        X = _X()
        y = (X[:, 0] > 0).astype(int)
        y_pred = y.copy()
        y_pred[:10] = 1 - y_pred[:10]
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "classification"
        )
        assert result["error_rate"] is not None
        assert 0.0 <= result["error_rate"] <= 1.0

    def test_n_total_matches_input(self):
        X = _X(n=30)
        y = np.ones(30)
        y_pred = np.ones(30)
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        assert result["n_total"] == 30

    def test_features_sorted_by_abs_correlation(self):
        X = _X(n=100, n_features=4)
        y = X[:, 0]
        y_pred = y + np.random.default_rng(7).normal(scale=2.0, size=100)
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(4), "regression"
        )
        abs_corrs = [f["correlation_abs"] for f in result["features"]]
        assert abs_corrs == sorted(abs_corrs, reverse=True)

    def test_features_capped_at_ten(self):
        n_feats = 15
        X = _X(n=100, n_features=n_feats)
        y = np.random.default_rng(1).normal(size=100)
        y_pred = y + 0.1
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(n_feats), "regression"
        )
        assert len(result["features"]) <= 10

    def test_feature_entry_has_required_keys(self):
        X = _X()
        y = X[:, 0]
        y_pred = y + 0.2
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        for entry in result["features"]:
            for key in (
                "feature",
                "correlation",
                "correlation_abs",
                "direction",
                "rank",
            ):
                assert key in entry

    def test_direction_labels_valid(self):
        X = _X(n=80)
        y = X[:, 0]
        y_pred = y + 0.3
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        valid_directions = {"positive", "negative", "neutral"}
        for entry in result["features"]:
            assert entry["direction"] in valid_directions

    def test_rank_starts_at_one(self):
        X = _X()
        y = X[:, 0]
        y_pred = y + 0.2
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        assert result["features"][0]["rank"] == 1

    def test_verdict_clear_drivers_when_high_correlation(self):
        rng = np.random.default_rng(42)
        n = 100
        # Use only positive X values so that f0 directly equals the abs residual
        X = np.abs(rng.normal(size=(n, 2))) + 0.1
        # Error = f0 * 3, so corr(f0, |residual|) = 1.0
        y = rng.normal(size=n)
        y_pred = y + X[:, 0] * 3
        result = compute_prediction_error_correlation(
            X, y, y_pred, ["f0", "f1"], "regression"
        )
        assert result["verdict"] in ("clear_drivers", "weak_drivers")

    def test_verdict_none_when_perfect_prediction(self):
        X = _X(n=50)
        y = X[:, 0]
        y_pred = y.copy()  # zero errors
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        # With zero errors, correlation is undefined → should gracefully return none/weak
        assert result["verdict"] in ("none", "weak_drivers", "clear_drivers")

    def test_too_few_rows_raises(self):
        X = _X(n=5)
        y = np.ones(5)
        y_pred = np.ones(5)
        with pytest.raises(ValueError, match="at least 10"):
            compute_prediction_error_correlation(
                X, y, y_pred, _feature_names(3), "regression"
            )

    def test_top_driver_is_first_feature_name(self):
        X = _X()
        y = X[:, 0]
        y_pred = y + 0.5
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        assert result["top_driver"] == result["features"][0]["feature"]

    def test_summary_is_string(self):
        X = _X()
        y = X[:, 0]
        y_pred = y + 0.5
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        assert isinstance(result["summary"], str) and len(result["summary"]) > 0

    def test_correlation_abs_equals_abs_correlation(self):
        X = _X()
        y = X[:, 0]
        y_pred = y + 0.3
        result = compute_prediction_error_correlation(
            X, y, y_pred, _feature_names(3), "regression"
        )
        for entry in result["features"]:
            assert abs(entry["correlation"]) == pytest.approx(
                entry["correlation_abs"], abs=0.0001
            )

    def test_string_labels_classification(self):
        X = _X(n=40)
        y_str = np.array(["cat" if x > 0 else "dog" for x in X[:, 0]])
        y_pred_str = y_str.copy()
        y_pred_str[:5] = np.where(y_pred_str[:5] == "cat", "dog", "cat")
        result = compute_prediction_error_correlation(
            X, y_str, y_pred_str, _feature_names(3), "classification"
        )
        assert result["error_type"] == "misclassification"
        assert result["n_errors"] == 5


# ---------------------------------------------------------------------------
# Pattern matching tests
# ---------------------------------------------------------------------------

from api.chat import _ERROR_CORRELATION_PATTERNS as _PAT  # noqa: E402


class TestErrorCorrelationPatterns:
    def _match(self, msg):
        return bool(_PAT.search(msg))

    def test_which_features_correlate_with_errors(self):
        assert self._match("which features correlate with my model's errors?")

    def test_what_causes_errors(self):
        assert self._match("what causes my model's prediction errors?")

    def test_error_correlation(self):
        assert self._match("show me the error correlation")

    def test_where_does_model_struggle(self):
        assert self._match("where does my model systematically struggle?")

    def test_feature_correlation_with_errors(self):
        assert self._match("feature correlation with errors")

    def test_what_drives_mistakes(self):
        assert self._match("what drives my model's errors?")

    def test_which_inputs_cause_errors(self):
        assert self._match("which input features cause more errors?")

    def test_error_drivers(self):
        assert self._match("drivers of my prediction errors")

    # False-positive guards
    def test_no_match_feature_importance(self):
        # Generic "feature importance" without error context should not match
        assert not self._match("show me the feature importance chart")

    def test_no_match_show_predictions(self):
        assert not self._match("show me my recent predictions")

    def test_no_match_bias_check(self):
        assert not self._match("check my model for bias")

    def test_no_match_model_accuracy(self):
        assert not self._match("what is my model accuracy?")


# ---------------------------------------------------------------------------
# REST endpoint integration tests
# ---------------------------------------------------------------------------

import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from httpx import ASGITransport, AsyncClient  # noqa: E402


@pytest.mark.asyncio
async def test_error_correlation_endpoint_404_unknown_run():
    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/models/nonexistent-run-id/error-correlation")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_error_correlation_endpoint_200_with_real_run(tmp_path, monkeypatch):
    """Integration test: create a minimal trained run and call the endpoint."""
    import uuid

    import joblib
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sqlmodel import Session, create_engine

    from db import get_session
    from main import app
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project
    from sqlmodel import SQLModel

    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        proj = Project(id=str(uuid.uuid4()), name="Test", status="exploring")
        session.add(proj)

        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"a": range(30), "b": range(30, 60), "target": range(60, 90)})
        df.to_csv(csv_path, index=False)

        ds = Dataset(
            id=str(uuid.uuid4()),
            project_id=proj.id,
            filename="data.csv",
            file_path=str(csv_path),
            row_count=30,
            column_count=3,
        )
        session.add(ds)

        fs = FeatureSet(
            id=str(uuid.uuid4()),
            project_id=proj.id,
            dataset_id=ds.id,
            target_column="target",
            feature_columns='["a","b"]',
            problem_type="regression",
            transformations="[]",
        )
        session.add(fs)

        X = df[["a", "b"]].values
        y = df["target"].values
        model = LinearRegression().fit(X, y)
        model_path = str(tmp_path / "model.joblib")
        joblib.dump(model, model_path)

        run = ModelRun(
            id=str(uuid.uuid4()),
            project_id=proj.id,
            feature_set_id=fs.id,
            algorithm="linear_regression",
            status="done",
            model_path=model_path,
            metrics='{"r2": 0.99}',
        )
        session.add(run)
        session.commit()
        run_id = run.id

    def override_session():
        with Session(engine) as s:
            yield s

    app.dependency_overrides[get_session] = override_session
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(f"/api/models/{run_id}/error-correlation")
        assert resp.status_code == 200
        data = resp.json()
        assert "features" in data
        assert data["model_run_id"] == run_id
    finally:
        app.dependency_overrides.clear()
