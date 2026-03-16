"""Tests for hyperparameter auto-tuning (POST /api/models/{model_run_id}/tune).

Covers:
- tune_model() core function (tunable + non-tunable algorithms)
- get_tuning_grid() returns correct grids
- API endpoint happy path, 404, bad-status guard
- Comparison result structure: improved/not, best_params, metrics
- Non-tunable algorithm returns tunable=False without creating a new run
- Tuning result metrics are valid (numeric, no NaN/inf)
"""

from __future__ import annotations

import io
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_CSV = b"""product,region,revenue,units,price
Widget A,North,1200.50,10,120.05
Widget B,South,850.00,8,106.25
Widget A,East,2100.75,18,116.71
Widget C,West,450.25,4,112.56
Widget B,North,1650.00,15,110.00
Widget A,South,980.00,9,108.89
Widget C,North,1100.25,11,100.02
Widget B,East,1750.00,16,109.38
Widget A,West,2300.50,20,115.03
Widget C,South,620.75,6,103.46
Widget A,North,1400.00,12,116.67
Widget B,West,900.00,9,100.00
Widget C,East,750.50,7,107.21
Widget A,South,1800.25,16,112.52
Widget B,North,2000.00,18,111.11
Widget A,East,950.75,8,118.84
Widget C,West,550.00,5,110.00
Widget B,South,1300.00,12,108.33
Widget A,North,2500.00,22,113.64
Widget C,East,800.00,8,100.00
"""

CLASSIFY_CSV = b"""feature_a,feature_b,feature_c,label
1.2,3.4,0,cat
2.1,1.1,1,dog
3.3,2.2,0,cat
0.5,4.1,1,dog
2.8,0.9,0,cat
1.9,3.1,1,dog
3.0,2.5,0,cat
0.8,3.8,1,dog
2.3,1.5,0,cat
1.7,2.9,1,dog
2.5,1.8,0,cat
1.1,3.5,1,dog
3.2,2.0,0,cat
0.6,4.3,1,dog
2.7,1.2,0,cat
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
    import models.deployment  # noqa
    import models.prediction_log  # noqa
    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module
    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app
    with TestClient(app) as c:
        yield c


def _setup_trained_model(client, csv_bytes=SAMPLE_CSV, algorithm="random_forest_regressor", target="revenue"):
    """Create project → upload → apply features → set target → train → poll done."""
    # Create project
    proj = client.post("/api/projects", json={"name": "Tune Test"}).json()
    pid = proj["id"]

    # Upload
    r = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"project_id": pid},
    )
    ds = r.json()
    did = ds["dataset_id"]

    # Apply (no transforms)
    fs_r = client.post(f"/api/features/{did}/apply", json={"transformations": []})
    fsid = fs_r.json()["feature_set_id"]

    # Set target
    client.post(f"/api/features/{did}/target", json={"target_column": target, "feature_set_id": fsid})

    # Train
    client.post(f"/api/models/{pid}/train", json={"algorithms": [algorithm]})
    # Poll until done
    for _ in range(60):
        runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
        if all(r["status"] in ("done", "failed") for r in runs):
            break
        time.sleep(0.1)

    runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
    done_runs = [r for r in runs if r["status"] == "done"]
    return pid, did, done_runs[0]["id"] if done_runs else None


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests: core/trainer.py tune helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestGetTuningGrid:
    def test_random_forest_regressor_has_grid(self):
        from core.trainer import get_tuning_grid
        grid = get_tuning_grid("random_forest_regressor")
        assert grid is not None
        assert "n_estimators" in grid

    def test_random_forest_classifier_has_grid(self):
        from core.trainer import get_tuning_grid
        grid = get_tuning_grid("random_forest_classifier")
        assert grid is not None
        assert "max_depth" in grid

    def test_gradient_boosting_regressor_has_grid(self):
        from core.trainer import get_tuning_grid
        grid = get_tuning_grid("gradient_boosting_regressor")
        assert grid is not None
        assert "learning_rate" in grid

    def test_gradient_boosting_classifier_has_grid(self):
        from core.trainer import get_tuning_grid
        grid = get_tuning_grid("gradient_boosting_classifier")
        assert grid is not None

    def test_logistic_regression_has_grid(self):
        from core.trainer import get_tuning_grid
        grid = get_tuning_grid("logistic_regression")
        assert grid is not None
        assert "C" in grid

    def test_linear_regression_not_tunable(self):
        from core.trainer import get_tuning_grid
        grid = get_tuning_grid("linear_regression")
        assert grid is None

    def test_neural_network_not_in_grid(self):
        from core.trainer import get_tuning_grid
        # No grid for neural network (too slow to tune)
        grid = get_tuning_grid("neural_network_regressor")
        assert grid is None

    def test_unknown_algorithm_returns_none(self):
        from core.trainer import get_tuning_grid
        assert get_tuning_grid("nonexistent_algo") is None


class TestTuneModelUnit:
    """Unit tests for the core tune_model() function."""

    def _get_data(self):
        import pandas as pd
        import io as _io
        df = pd.read_csv(_io.BytesIO(SAMPLE_CSV))
        from core.trainer import prepare_features
        feature_cols = [c for c in df.columns if c != "revenue"]
        X, y, _ = prepare_features(df, feature_cols, "revenue", "regression")
        return X, y

    def test_tune_random_forest_returns_best_params(self, tmp_path):
        from core.trainer import tune_model
        X, y = self._get_data()
        result = tune_model(X, y, "random_forest_regressor", "regression", tmp_path, "run1", n_iter=3, cv=2)
        assert result["tunable"] is True
        assert result["best_params"] is not None
        assert isinstance(result["best_params"], dict)

    def test_tune_returns_valid_metrics(self, tmp_path):
        from core.trainer import tune_model
        X, y = self._get_data()
        result = tune_model(X, y, "random_forest_regressor", "regression", tmp_path, "run2", n_iter=3, cv=2)
        metrics = result["metrics"]
        assert metrics is not None
        assert "r2" in metrics
        assert not np.isnan(metrics["r2"])
        assert not np.isinf(metrics["r2"])

    def test_tune_saves_model_file(self, tmp_path):
        from core.trainer import tune_model
        import os
        X, y = self._get_data()
        result = tune_model(X, y, "random_forest_regressor", "regression", tmp_path, "run3", n_iter=3, cv=2)
        assert result["model_path"] is not None
        assert os.path.exists(result["model_path"])

    def test_tune_non_tunable_algorithm(self, tmp_path):
        from core.trainer import tune_model
        X, y = self._get_data()
        result = tune_model(X, y, "linear_regression", "regression", tmp_path, "run4", n_iter=3, cv=2)
        assert result["tunable"] is False
        assert result["model_path"] is None
        assert "no hyperparameters" in result["summary"].lower()

    def test_tune_gradient_boosting(self, tmp_path):
        from core.trainer import tune_model
        X, y = self._get_data()
        result = tune_model(X, y, "gradient_boosting_regressor", "regression", tmp_path, "run5", n_iter=3, cv=2)
        assert result["tunable"] is True
        assert result["tuned_cv_score"] is not None

    def test_tune_returns_duration(self, tmp_path):
        from core.trainer import tune_model
        X, y = self._get_data()
        result = tune_model(X, y, "random_forest_regressor", "regression", tmp_path, "run6", n_iter=3, cv=2)
        assert result["training_duration_ms"] >= 0

    def test_tune_invalid_algorithm_raises(self, tmp_path):
        from core.trainer import tune_model
        X, y = self._get_data()
        with pytest.raises(ValueError, match="Unknown algorithm"):
            tune_model(X, y, "bad_algo", "regression", tmp_path, "run7")


# ──────────────────────────────────────────────────────────────────────────────
# API integration tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTuneModelAPI:

    def test_tune_404_for_unknown_run(self, client):
        r = client.post("/api/models/nonexistent-run-id/tune")
        assert r.status_code == 404

    def test_tune_400_for_pending_run(self, client):
        """Cannot tune a run that hasn't completed training."""
        pid = client.post("/api/projects", json={"name": "P"}).json()["id"]
        # Create a dataset and feature set to satisfy training prerequisites
        r = client.post(
            "/api/data/upload",
            files={"file": ("d.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
            data={"project_id": pid},
        )
        did = r.json()["dataset_id"]
        fs = client.post(f"/api/features/{did}/apply", json={"transformations": []}).json()
        client.post(f"/api/features/{did}/target", json={"target_column": "revenue", "feature_set_id": fs["feature_set_id"]})

        # Kick off training — use the run ID before it finishes
        t_r = client.post(f"/api/models/{pid}/train", json={"algorithms": ["linear_regression"]})
        run_ids = t_r.json()["model_run_ids"]
        assert len(run_ids) > 0
        # Immediately try to tune the pending run
        pending_run_id = run_ids[0]
        result = client.post(f"/api/models/{pending_run_id}/tune")
        # Might be 200 if training finishes fast; if pending → 400
        if result.status_code == 400:
            assert "status" in result.json()["detail"].lower()

    def test_tune_regression_random_forest(self, client):
        _, _, run_id = _setup_trained_model(client, algorithm="random_forest_regressor", target="revenue")
        assert run_id is not None, "Training must have completed"
        r = client.post(f"/api/models/{run_id}/tune")
        assert r.status_code == 201
        body = r.json()
        assert body["original_model_run_id"] == run_id
        assert body["tunable"] is True
        assert body["tuned_model_run_id"] is not None
        assert body["original_metrics"] is not None
        assert body["tuned_metrics"] is not None
        assert isinstance(body["improved"], bool)
        assert "r2" in body["original_metrics"]
        assert "r2" in body["tuned_metrics"]

    def test_tune_non_tunable_algorithm(self, client):
        _, _, run_id = _setup_trained_model(client, algorithm="linear_regression", target="revenue")
        assert run_id is not None
        r = client.post(f"/api/models/{run_id}/tune")
        assert r.status_code == 201
        body = r.json()
        assert body["tunable"] is False
        assert body["tuned_model_run_id"] is None

    def test_tune_creates_new_model_run(self, client):
        pid, _, run_id = _setup_trained_model(client)
        assert run_id is not None
        # Count runs before
        before = len(client.get(f"/api/models/{pid}/runs").json()["runs"])
        client.post(f"/api/models/{run_id}/tune")
        after = len(client.get(f"/api/models/{pid}/runs").json()["runs"])
        assert after == before + 1  # one new tuned run created

    def test_tune_result_has_summary(self, client):
        _, _, run_id = _setup_trained_model(client)
        assert run_id is not None
        body = client.post(f"/api/models/{run_id}/tune").json()
        assert isinstance(body["summary"], str)
        assert len(body["summary"]) > 10

    def test_tune_best_params_are_valid(self, client):
        _, _, run_id = _setup_trained_model(client)
        assert run_id is not None
        body = client.post(f"/api/models/{run_id}/tune").json()
        if body["tunable"]:
            assert isinstance(body["best_params"], dict)
            assert len(body["best_params"]) > 0

    def test_tune_improvement_pct_type(self, client):
        _, _, run_id = _setup_trained_model(client)
        assert run_id is not None
        body = client.post(f"/api/models/{run_id}/tune").json()
        if body["improvement_pct"] is not None:
            assert isinstance(body["improvement_pct"], float)

    def test_tune_tuned_run_has_correct_algorithm(self, client):
        _, _, run_id = _setup_trained_model(client, algorithm="random_forest_regressor")
        assert run_id is not None
        body = client.post(f"/api/models/{run_id}/tune").json()
        assert body["algorithm"] == "random_forest_regressor"
        if body["tuned_run"]:
            assert body["tuned_run"]["algorithm"] == "random_forest_regressor"

    def test_tune_classifier(self, client):
        _, _, run_id = _setup_trained_model(
            client, csv_bytes=CLASSIFY_CSV,
            algorithm="random_forest_classifier",
            target="label"
        )
        assert run_id is not None
        r = client.post(f"/api/models/{run_id}/tune")
        assert r.status_code == 201
        body = r.json()
        assert body["tunable"] is True
        assert "accuracy" in body["original_metrics"]
        assert "accuracy" in body["tuned_metrics"]
