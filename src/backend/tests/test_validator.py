"""Tests for core/validator.py, core/explainer.py, and /api/validate/* endpoints."""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV = b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-01,Widget B,South,850.00,8
2024-01-02,Widget A,East,2100.75,18
2024-01-02,Widget C,West,450.25,4
2024-01-03,Widget B,North,1650.00,15
2024-01-04,Widget A,South,980.00,9
2024-01-04,Widget C,North,1100.25,11
2024-01-05,Widget B,East,1750.00,16
2024-01-05,Widget A,West,2300.50,20
2024-01-06,Widget C,South,620.75,6
2024-01-07,Widget A,North,1450.00,13
2024-01-07,Widget B,West,1900.00,17
2024-01-08,Widget C,East,780.00,7
2024-01-08,Widget A,South,2050.00,19
2024-01-09,Widget B,North,1300.00,12
"""


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.conversation  # noqa
    import models.model_run  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module
    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def trained_model_run(client):
    """Create project → upload → apply features → train → wait for done."""
    # Create project
    proj = client.post("/api/projects", json={"name": "Validation Test"})
    assert proj.status_code == 201, proj.text
    project_id = proj.json()["id"]

    # Upload CSV
    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["dataset_id"]

    # Apply feature set
    apply = client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert apply.status_code == 201, apply.text

    # Set target
    target = client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue"},
    )
    assert target.status_code == 200, target.text

    # Train
    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["random_forest_regressor"]},
    )
    assert train.status_code == 202, train.text
    run_id = train.json()["model_run_ids"][0]

    # Poll until done
    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    assert run["status"] == "done", f"Training failed: {run.get('error_message')}"
    return project_id, dataset_id, run_id


# ---------------------------------------------------------------------------
# Unit tests: validator.py
# ---------------------------------------------------------------------------


class TestRunCrossValidation:
    def _make_xy(self, n=50):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (n, 3))
        y = 2 * X[:, 0] + rng.normal(0, 0.1, n)
        return X, y

    def test_returns_expected_keys(self):
        from core.validator import run_cross_validation
        from sklearn.linear_model import LinearRegression

        X, y = self._make_xy()
        result = run_cross_validation(LinearRegression(), X, y, "regression")
        assert all(k in result for k in ("mean", "std", "scores", "ci_low", "ci_high", "summary"))

    def test_scores_list_length_matches_n_splits(self):
        from core.validator import run_cross_validation
        from sklearn.linear_model import LinearRegression

        X, y = self._make_xy()
        result = run_cross_validation(LinearRegression(), X, y, "regression", n_splits=3)
        assert len(result["scores"]) == 3

    def test_r2_positive_on_linear_data(self):
        from core.validator import run_cross_validation
        from sklearn.linear_model import LinearRegression

        X, y = self._make_xy(n=100)
        result = run_cross_validation(LinearRegression(), X, y, "regression")
        assert result["mean"] > 0.5

    def test_classification_uses_f1_weighted(self):
        from core.validator import run_cross_validation
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(7)
        X = rng.normal(0, 1, (60, 3))
        y = (X[:, 0] > 0).astype(int)
        result = run_cross_validation(RandomForestClassifier(random_state=42), X, y, "classification")
        assert result["metric"] == "f1_weighted"
        assert 0.0 <= result["mean"] <= 1.0

    def test_too_small_dataset_returns_gracefully(self):
        from core.validator import run_cross_validation
        from sklearn.linear_model import LinearRegression

        X = np.array([[1.0]])
        y = np.array([2.0])
        result = run_cross_validation(LinearRegression(), X, y, "regression")
        assert result["mean"] is None


class TestComputeConfusionMatrix:
    def test_binary_classification(self):
        from core.validator import compute_confusion_matrix

        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        result = compute_confusion_matrix(y_true, y_pred)
        assert "matrix" in result
        assert "labels" in result
        assert "accuracy" in result
        assert "summary" in result
        assert 0 <= result["accuracy"] <= 1

    def test_matrix_shape_matches_classes(self):
        from core.validator import compute_confusion_matrix

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])
        result = compute_confusion_matrix(y_true, y_pred)
        assert len(result["matrix"]) == 3
        assert len(result["matrix"][0]) == 3

    def test_perfect_prediction(self):
        from core.validator import compute_confusion_matrix

        y = np.array([0, 1, 0, 1])
        result = compute_confusion_matrix(y, y)
        assert result["accuracy"] == 1.0


class TestComputeResiduals:
    def test_returns_expected_structure(self):
        from core.validator import compute_residuals

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        y_pred = y_true + np.random.default_rng(42).normal(0, 0.1, len(y_true))
        result = compute_residuals(y_true, y_pred)
        assert all(k in result for k in ("scatter", "mae", "bias", "std", "summary"))

    def test_scatter_capped_at_200(self):
        from core.validator import compute_residuals

        y = np.linspace(0, 1, 300)
        result = compute_residuals(y, y * 1.01)
        assert len(result["scatter"]) <= 200

    def test_mae_near_zero_for_perfect_predictions(self):
        from core.validator import compute_residuals

        y = np.array([1.0, 2.0, 3.0])
        result = compute_residuals(y, y)
        assert result["mae"] == 0.0


class TestAssessConfidenceLimitations:
    def test_high_confidence_for_good_model(self):
        from core.validator import assess_confidence_limitations

        result = assess_confidence_limitations(
            metrics={"r2": 0.92, "mae": 0.05},
            problem_type="regression",
            n_rows=500,
            n_features=10,
            cv_std=0.02,
        )
        assert result["overall_confidence"] == "high"

    def test_warns_on_small_dataset(self):
        from core.validator import assess_confidence_limitations

        result = assess_confidence_limitations(
            metrics={"r2": 0.9},
            problem_type="regression",
            n_rows=30,
            n_features=3,
            cv_std=0.04,
        )
        assert any("30 rows" in lim for lim in result["limitations"])

    def test_warns_on_high_cv_variance(self):
        from core.validator import assess_confidence_limitations

        result = assess_confidence_limitations(
            metrics={"r2": 0.8},
            problem_type="regression",
            n_rows=200,
            n_features=5,
            cv_std=0.15,
        )
        assert any("variance" in lim.lower() for lim in result["limitations"])

    def test_no_limitations_message_on_clean_model(self):
        from core.validator import assess_confidence_limitations

        result = assess_confidence_limitations(
            metrics={"accuracy": 0.95, "f1": 0.94},
            problem_type="classification",
            n_rows=1000,
            n_features=5,
            cv_std=0.02,
        )
        assert any("no major" in lim.lower() for lim in result["limitations"])


# ---------------------------------------------------------------------------
# Unit tests: explainer.py
# ---------------------------------------------------------------------------


class TestComputeFeatureImportance:
    def _rf_model(self):
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))
        y = 2 * X[:, 0] + X[:, 1] + rng.normal(0, 0.1, 50)
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        return model

    def test_returns_list_of_dicts(self):
        from core.explainer import compute_feature_importance

        model = self._rf_model()
        result = compute_feature_importance(model, ["a", "b", "c"])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_each_entry_has_required_keys(self):
        from core.explainer import compute_feature_importance

        model = self._rf_model()
        result = compute_feature_importance(model, ["a", "b", "c"])
        for item in result:
            assert "feature" in item
            assert "importance" in item
            assert "rank" in item

    def test_importances_sum_to_one(self):
        from core.explainer import compute_feature_importance

        model = self._rf_model()
        result = compute_feature_importance(model, ["a", "b", "c"])
        total = sum(item["importance"] for item in result)
        assert abs(total - 1.0) < 1e-4

    def test_sorted_descending(self):
        from core.explainer import compute_feature_importance

        model = self._rf_model()
        result = compute_feature_importance(model, ["a", "b", "c"])
        imps = [item["importance"] for item in result]
        assert imps == sorted(imps, reverse=True)

    def test_linear_model_uses_coefficients(self):
        from sklearn.linear_model import LinearRegression
        from core.explainer import compute_feature_importance

        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 2))
        y = 3 * X[:, 0] + rng.normal(0, 0.01, 50)
        model = LinearRegression().fit(X, y)
        result = compute_feature_importance(model, ["strong", "weak"])
        # "strong" should rank first because its coef is ~3 vs ~0
        assert result[0]["feature"] == "strong"


class TestExplainSinglePrediction:
    def _setup(self):
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 3))
        y = 2 * X[:, 0] + X[:, 1] + rng.normal(0, 0.1, 50)
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_returns_expected_keys(self):
        from core.explainer import explain_single_prediction

        model, X, _ = self._setup()
        result = explain_single_prediction(model, X[0], X, ["a", "b", "c"], "regression")
        assert "prediction" in result
        assert "contributions" in result
        assert "summary" in result

    def test_contributions_length_matches_features(self):
        from core.explainer import explain_single_prediction

        model, X, _ = self._setup()
        result = explain_single_prediction(model, X[0], X, ["a", "b", "c"], "regression")
        assert len(result["contributions"]) == 3

    def test_contribution_has_direction(self):
        from core.explainer import explain_single_prediction

        model, X, _ = self._setup()
        result = explain_single_prediction(model, X[0], X, ["a", "b", "c"], "regression")
        for c in result["contributions"]:
            assert c["direction"] in ("positive", "negative")


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


class TestValidationAPI:
    def test_metrics_endpoint_returns_200(self, client, trained_model_run):
        _, _, run_id = trained_model_run
        resp = client.get(f"/api/validate/{run_id}/metrics")
        assert resp.status_code == 200, resp.text

    def test_metrics_has_cross_validation(self, client, trained_model_run):
        _, _, run_id = trained_model_run
        body = client.get(f"/api/validate/{run_id}/metrics").json()
        assert "cross_validation" in body
        cv = body["cross_validation"]
        assert cv["mean"] is not None
        assert isinstance(cv["scores"], list)

    def test_metrics_has_error_analysis(self, client, trained_model_run):
        _, _, run_id = trained_model_run
        body = client.get(f"/api/validate/{run_id}/metrics").json()
        assert "error_analysis" in body
        # Regression → residuals
        assert body["error_analysis"]["type"] == "residuals"
        assert "scatter" in body["error_analysis"]

    def test_metrics_has_confidence(self, client, trained_model_run):
        _, _, run_id = trained_model_run
        body = client.get(f"/api/validate/{run_id}/metrics").json()
        assert "confidence" in body
        assert body["confidence"]["overall_confidence"] in ("high", "medium", "low")

    def test_explain_endpoint_returns_200(self, client, trained_model_run):
        _, _, run_id = trained_model_run
        resp = client.get(f"/api/validate/{run_id}/explain")
        assert resp.status_code == 200, resp.text

    def test_explain_has_feature_importance(self, client, trained_model_run):
        _, _, run_id = trained_model_run
        body = client.get(f"/api/validate/{run_id}/explain").json()
        assert "feature_importance" in body
        fi = body["feature_importance"]
        assert len(fi) > 0
        assert all("feature" in item for item in fi)
        assert all("importance" in item for item in fi)

    def test_explain_row_endpoint_returns_200(self, client, trained_model_run):
        _, _, run_id = trained_model_run
        resp = client.get(f"/api/validate/{run_id}/explain/0")
        assert resp.status_code == 200, resp.text

    def test_explain_row_has_contributions(self, client, trained_model_run):
        _, _, run_id = trained_model_run
        body = client.get(f"/api/validate/{run_id}/explain/0").json()
        assert "contributions" in body
        assert "prediction" in body
        assert "summary" in body

    def test_explain_row_out_of_range_returns_400(self, client, trained_model_run):
        _, _, run_id = trained_model_run
        resp = client.get(f"/api/validate/{run_id}/explain/99999")
        assert resp.status_code == 400

    def test_unknown_run_id_returns_404(self, client):
        resp = client.get("/api/validate/nonexistent-run/metrics")
        assert resp.status_code == 404
