"""Tests for compute_prediction_errors() and GET /api/models/{run_id}/prediction-errors."""

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

SAMPLE_CLASSIFICATION_CSV = b"""age,income,job_type,churned
25,45000,manual,no
45,80000,professional,no
30,35000,manual,yes
55,120000,management,no
28,42000,manual,yes
60,95000,professional,no
35,55000,manual,no
40,70000,professional,yes
22,30000,manual,yes
50,100000,management,no
33,48000,manual,no
48,88000,professional,no
27,38000,manual,yes
52,105000,management,no
38,62000,professional,no
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
    import models.prediction_log  # noqa
    import models.dataset_filter  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def trained_regression_run(client):
    """Create project → upload → apply features → train linear_regression."""
    proj = client.post("/api/projects", json={"name": "Error Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": "revenue"})

    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert train.status_code == 202
    run_id = train.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert run["status"] == "done", run.get("error_message")
    return project_id, dataset_id, run_id


@pytest.fixture
def trained_classification_run(client):
    """Create project → upload classification CSV → train logistic_regression."""
    proj = client.post("/api/projects", json={"name": "Class Error Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={
            "file": ("churn.csv", io.BytesIO(SAMPLE_CLASSIFICATION_CSV), "text/csv")
        },
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": "churned"})

    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["logistic_regression"]},
    )
    assert train.status_code == 202
    run_id = train.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert run["status"] == "done", run.get("error_message")
    return project_id, dataset_id, run_id


# ---------------------------------------------------------------------------
# Unit tests: compute_prediction_errors() — regression
# ---------------------------------------------------------------------------


class TestComputePredictionErrorsRegression:
    def _make_data(self, n=20, seed=42):
        rng = np.random.default_rng(seed)
        y_true = rng.uniform(100, 2000, n)
        y_pred = y_true + rng.normal(0, 150, n)
        return y_true, y_pred

    def test_returns_expected_keys(self):
        from core.validator import compute_prediction_errors

        y_true, y_pred = self._make_data()
        result = compute_prediction_errors(y_true, y_pred, "regression", n=5)
        for key in ("errors", "total_errors", "error_rate", "summary", "problem_type"):
            assert key in result

    def test_n_capped_at_request(self):
        from core.validator import compute_prediction_errors

        y_true, y_pred = self._make_data(n=20)
        result = compute_prediction_errors(y_true, y_pred, "regression", n=5)
        assert len(result["errors"]) == 5

    def test_errors_sorted_descending_by_abs_error(self):
        from core.validator import compute_prediction_errors

        y_true, y_pred = self._make_data(n=20)
        result = compute_prediction_errors(y_true, y_pred, "regression", n=10)
        abs_errors = [e["abs_error"] for e in result["errors"]]
        assert abs_errors == sorted(abs_errors, reverse=True)

    def test_rank_is_1_based(self):
        from core.validator import compute_prediction_errors

        y_true, y_pred = self._make_data(n=10)
        result = compute_prediction_errors(y_true, y_pred, "regression", n=3)
        assert result["errors"][0]["rank"] == 1
        assert result["errors"][-1]["rank"] == 3

    def test_n_clamped_below_1(self):
        from core.validator import compute_prediction_errors

        y_true, y_pred = self._make_data(n=10)
        result = compute_prediction_errors(y_true, y_pred, "regression", n=0)
        assert len(result["errors"]) == 1

    def test_n_clamped_above_50(self):
        from core.validator import compute_prediction_errors

        y_true, y_pred = self._make_data(n=60)
        result = compute_prediction_errors(y_true, y_pred, "regression", n=100)
        assert len(result["errors"]) == 50

    def test_feature_rows_attached(self):
        from core.validator import compute_prediction_errors

        y_true, y_pred = self._make_data(n=10)
        rows = [{"region": "North", "units": i} for i in range(10)]
        result = compute_prediction_errors(
            y_true, y_pred, "regression", n=3, feature_rows=rows
        )
        for err in result["errors"]:
            assert "features" in err
            assert "region" in err["features"]

    def test_empty_input_returns_empty(self):
        from core.validator import compute_prediction_errors

        result = compute_prediction_errors(
            np.array([]), np.array([]), "regression", n=5
        )
        assert result["errors"] == []
        assert result["total_errors"] == 0

    def test_summary_contains_mae_reference(self):
        from core.validator import compute_prediction_errors

        y_true, y_pred = self._make_data(n=20)
        result = compute_prediction_errors(y_true, y_pred, "regression", n=5)
        assert "MAE" in result["summary"] or "training rows" in result["summary"]

    def test_error_field_is_signed(self):
        from core.validator import compute_prediction_errors

        # y_pred overestimates → errors should be negative (actual - predicted < 0)
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([150.0, 250.0, 350.0])
        result = compute_prediction_errors(y_true, y_pred, "regression", n=3)
        for err in result["errors"]:
            assert err["error"] < 0  # under-prediction


# ---------------------------------------------------------------------------
# Unit tests: compute_prediction_errors() — classification
# ---------------------------------------------------------------------------


class TestComputePredictionErrorsClassification:
    def test_wrong_predictions_only(self):
        from core.validator import compute_prediction_errors

        y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # rows 0 and 3 wrong
        result = compute_prediction_errors(y_true, y_pred, "classification", n=10)
        assert len(result["errors"]) == 2
        for err in result["errors"]:
            assert err["actual"] != err["predicted"]

    def test_total_errors_count_correct(self):
        from core.validator import compute_prediction_errors

        y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        result = compute_prediction_errors(y_true, y_pred, "classification", n=10)
        assert result["total_errors"] == 2

    def test_error_rate_computed_correctly(self):
        from core.validator import compute_prediction_errors

        y_true = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        result = compute_prediction_errors(y_true, y_pred, "classification", n=10)
        assert result["error_rate"] == pytest.approx(0.4, abs=0.01)

    def test_target_classes_decoded(self):
        from core.validator import compute_prediction_errors

        y_true = np.array([0.0, 1.0])
        y_pred = np.array([1.0, 0.0])
        result = compute_prediction_errors(
            y_true, y_pred, "classification", n=5, target_classes=["no", "yes"]
        )
        for err in result["errors"]:
            assert err["actual"] in ("no", "yes")
            assert err["predicted"] in ("no", "yes")

    def test_no_errors_empty_list(self):
        from core.validator import compute_prediction_errors

        y_true = np.array([0.0, 1.0, 0.0])
        y_pred = np.array([0.0, 1.0, 0.0])
        result = compute_prediction_errors(y_true, y_pred, "classification", n=10)
        assert result["errors"] == []
        assert result["total_errors"] == 0


# ---------------------------------------------------------------------------
# Pattern tests: _PRED_ERROR_PATTERNS
# ---------------------------------------------------------------------------


class TestPredErrorPatterns:
    MATCH_CASES = [
        "where was my model wrong?",
        "show me the prediction errors",
        "which rows did my model get wrong?",
        "biggest prediction errors",
        "where did my model fail?",
        "show me the worst predictions",
        "show me model mistakes",
        "largest errors",
    ]
    NO_MATCH_CASES = [
        "what is my model accuracy?",
        "show me the correlation heatmap",
        "train a new model",
        "deploy my model",
    ]

    def test_matches(self):
        from api.chat import _PRED_ERROR_PATTERNS

        for msg in self.MATCH_CASES:
            assert _PRED_ERROR_PATTERNS.search(msg), f"Should match: {msg!r}"

    def test_no_matches(self):
        from api.chat import _PRED_ERROR_PATTERNS

        for msg in self.NO_MATCH_CASES:
            assert not _PRED_ERROR_PATTERNS.search(msg), f"Should not match: {msg!r}"


# ---------------------------------------------------------------------------
# Endpoint tests: GET /api/models/{run_id}/prediction-errors
# ---------------------------------------------------------------------------


class TestPredictionErrorsEndpoint:
    def test_200_regression_returns_errors(self, client, trained_regression_run):
        _, _, run_id = trained_regression_run
        resp = client.get(f"/api/models/{run_id}/prediction-errors?n=5")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "errors" in data
        assert "summary" in data
        assert data["problem_type"] == "regression"
        assert data["algorithm"] == "linear_regression"
        assert len(data["errors"]) <= 5

    def test_n_param_respected(self, client, trained_regression_run):
        _, _, run_id = trained_regression_run
        resp = client.get(f"/api/models/{run_id}/prediction-errors?n=3")
        assert resp.status_code == 200
        assert len(resp.json()["errors"]) <= 3

    def test_404_unknown_run(self, client):
        resp = client.get("/api/models/nonexistent-run-id/prediction-errors")
        assert resp.status_code == 404

    def test_200_classification(self, client, trained_classification_run):
        _, _, run_id = trained_classification_run
        resp = client.get(f"/api/models/{run_id}/prediction-errors?n=10")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["problem_type"] == "classification"
        assert "errors" in data
        assert "total_errors" in data
        assert "error_rate" in data

    def test_errors_have_required_fields(self, client, trained_regression_run):
        _, _, run_id = trained_regression_run
        resp = client.get(f"/api/models/{run_id}/prediction-errors?n=5")
        assert resp.status_code == 200
        for err in resp.json()["errors"]:
            assert "actual" in err
            assert "predicted" in err
            assert "error" in err
            assert "rank" in err

    def test_default_n_is_10(self, client, trained_regression_run):
        _, _, run_id = trained_regression_run
        resp = client.get(f"/api/models/{run_id}/prediction-errors")
        assert resp.status_code == 200
        assert resp.json()["n_requested"] == 10

    def test_response_includes_target_col(self, client, trained_regression_run):
        _, _, run_id = trained_regression_run
        resp = client.get(f"/api/models/{run_id}/prediction-errors")
        assert resp.status_code == 200
        assert resp.json()["target_col"] == "revenue"
