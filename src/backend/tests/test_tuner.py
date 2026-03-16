"""Tests for hyperparameter auto-tuning: core/tuner.py and POST /api/models/{id}/tune."""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import api.models as models_module
import db as db_module


# ---------------------------------------------------------------------------
# Shared CSV fixture — enough rows for RandomizedSearchCV with cv=3
# ---------------------------------------------------------------------------

def _make_csv(n_rows: int = 60) -> bytes:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_rows):
        rows.append(
            f"2024-01-{(i % 28) + 1:02d},"
            f"{'AB'[i % 2]},"
            f"{rng.integers(1, 5)},"
            f"{rng.uniform(100, 2000):.2f}"
        )
    header = "date,category,region_id,revenue"
    return (header + "\n" + "\n".join(rows) + "\n").encode()


SAMPLE_CSV = _make_csv()


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.conversation  # noqa
    import models.deployment  # noqa
    import models.prediction_log  # noqa
    SQLModel.metadata.create_all(db_module.engine)

    models_module.MODELS_DIR = tmp_path / "models"

    from main import app
    return TestClient(app)


def _setup_trained_model(client, tmp_path, algorithm="random_forest_regressor"):
    """Helper: upload CSV, apply features, set target, train, return model_run_id."""
    # Create project
    r = client.post("/api/projects", json={"name": "Tune Test"})
    project_id = r.json()["id"]

    # Upload CSV
    r = client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert r.status_code in (200, 201)
    dataset_id = r.json()["dataset_id"]

    # Apply features (empty = creates FeatureSet) — returns 201
    r = client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    assert r.status_code in (200, 201)
    feature_set_id = r.json()["feature_set_id"]

    # Set target
    r = client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "problem_type": "regression"},
    )
    assert r.status_code == 200

    # Train
    r = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": [algorithm]},
    )
    assert r.status_code == 202
    model_run_id = r.json()["model_run_ids"][0]

    # Poll until done (max 60s for RandomForest on 60 rows)
    import time
    for _ in range(120):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == model_run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    return project_id, model_run_id


# ---------------------------------------------------------------------------
# Unit tests: core/tuner.py
# ---------------------------------------------------------------------------

class TestTunerUnit:
    def test_is_tunable_random_forest(self):
        from core.tuner import is_tunable
        assert is_tunable("random_forest_regressor") is True
        assert is_tunable("random_forest_classifier") is True

    def test_is_tunable_gradient_boosting(self):
        from core.tuner import is_tunable
        assert is_tunable("gradient_boosting_regressor") is True
        assert is_tunable("gradient_boosting_classifier") is True

    def test_is_tunable_linear_regression_false(self):
        from core.tuner import is_tunable
        # Linear regression has no meaningful hyperparameters
        assert is_tunable("linear_regression") is False

    def test_is_tunable_neural_network_false(self):
        from core.tuner import is_tunable
        assert is_tunable("neural_network_regressor") is False
        assert is_tunable("neural_network_classifier") is False

    def test_is_tunable_unknown_algorithm_false(self):
        from core.tuner import is_tunable
        assert is_tunable("nonexistent_algo") is False

    def test_tune_model_regression(self, tmp_path):
        from core.tuner import tune_model
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 3))
        y = X[:, 0] * 2 + rng.standard_normal(80) * 0.1

        result = tune_model(
            algorithm="random_forest_regressor",
            problem_type="regression",
            X=X,
            y=y,
            model_run_id="test-run-001",
            model_dir=tmp_path / "models",
            n_iter=5,
            cv=2,
        )

        assert "best_params" in result
        assert "metrics" in result
        assert "r2" in result["metrics"]
        assert "model_path" in result
        assert Path(result["model_path"]).exists()
        assert result["training_duration_ms"] > 0
        assert "summary" in result
        assert "Random Forest" in result["summary"]
        assert result["algorithm"] == "random_forest_regressor_tuned"

    def test_tune_model_classification(self, tmp_path):
        from core.tuner import tune_model
        rng = np.random.default_rng(1)
        X = rng.standard_normal((80, 3))
        y = (X[:, 0] > 0).astype(int)

        result = tune_model(
            algorithm="random_forest_classifier",
            problem_type="classification",
            X=X,
            y=y,
            model_run_id="test-run-002",
            model_dir=tmp_path / "models",
            n_iter=5,
            cv=2,
        )

        assert "accuracy" in result["metrics"]
        assert result["algorithm"] == "random_forest_classifier_tuned"

    def test_tune_model_unknown_algorithm_raises(self, tmp_path):
        from core.tuner import tune_model
        with pytest.raises(ValueError, match="Unknown algorithm"):
            tune_model(
                algorithm="bad_algo",
                problem_type="regression",
                X=np.zeros((10, 2)),
                y=np.zeros(10),
                model_run_id="x",
                model_dir=tmp_path,
            )

    def test_tune_model_untuneable_raises(self, tmp_path):
        from core.tuner import tune_model
        with pytest.raises(ValueError, match="no tunable hyperparameters"):
            tune_model(
                algorithm="linear_regression",
                problem_type="regression",
                X=np.zeros((10, 2)),
                y=np.zeros(10),
                model_run_id="x",
                model_dir=tmp_path,
            )

    def test_tune_model_best_params_keys(self, tmp_path):
        from core.tuner import tune_model, _REGRESSION_PARAM_GRIDS
        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 2))
        y = X[:, 0] + rng.standard_normal(60) * 0.5

        result = tune_model(
            algorithm="random_forest_regressor",
            problem_type="regression",
            X=X,
            y=y,
            model_run_id="test-run-003",
            model_dir=tmp_path / "models",
            n_iter=3,
            cv=2,
        )

        expected_keys = set(_REGRESSION_PARAM_GRIDS["random_forest_regressor"].keys())
        assert set(result["best_params"].keys()).issubset(expected_keys)

    def test_tune_model_gradient_boosting(self, tmp_path):
        from core.tuner import tune_model
        rng = np.random.default_rng(3)
        X = rng.standard_normal((60, 2))
        y = X[:, 0] * 3 + rng.standard_normal(60) * 0.5

        result = tune_model(
            algorithm="gradient_boosting_regressor",
            problem_type="regression",
            X=X,
            y=y,
            model_run_id="test-run-004",
            model_dir=tmp_path / "models",
            n_iter=3,
            cv=2,
        )

        assert result["algorithm"] == "gradient_boosting_regressor_tuned"
        assert "r2" in result["metrics"]

    def test_tune_summary_regression_content(self, tmp_path):
        from core.tuner import tune_model
        rng = np.random.default_rng(4)
        X = rng.standard_normal((60, 2))
        y = X[:, 0] * 2

        result = tune_model(
            algorithm="random_forest_regressor",
            problem_type="regression",
            X=X,
            y=y,
            model_run_id="test-run-005",
            model_dir=tmp_path / "models",
            n_iter=3,
            cv=2,
        )

        # Summary should mention the algorithm name and key metrics
        assert "R²" in result["summary"] or "r2" in result["summary"].lower()
        assert "CV" in result["summary"]


# ---------------------------------------------------------------------------
# API integration tests: POST /api/models/{model_run_id}/tune
# ---------------------------------------------------------------------------

class TestTuneEndpoint:
    def test_tune_not_found(self, client, tmp_path):
        r = client.post("/api/models/nonexistent-id/tune")
        assert r.status_code == 404

    def test_tune_untrained_model(self, client, tmp_path):
        """Tuning a 'pending' model run should return 400."""
        from sqlmodel import Session
        from models.model_run import ModelRun

        r = client.post("/api/projects", json={"name": "P"})
        project_id = r.json()["id"]

        with Session(db_module.engine) as session:
            run = ModelRun(
                project_id=project_id,
                feature_set_id=None,
                algorithm="random_forest_regressor",
                hyperparameters="{}",
                status="pending",
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            run_id = run.id

        r = client.post(f"/api/models/{run_id}/tune")
        assert r.status_code == 400
        assert "status" in r.json()["detail"]

    def test_tune_untuneable_algorithm(self, client, tmp_path):
        """Tuning linear_regression returns 201 with tunable=False (graceful, no error)."""
        # Need a real feature_set_id — go through full upload/apply/target workflow
        r = client.post("/api/projects", json={"name": "P"})
        project_id = r.json()["id"]

        r = client.post(
            "/api/data/upload",
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
            data={"project_id": project_id},
        )
        dataset_id = r.json()["dataset_id"]

        r = client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        assert r.status_code in (200, 201)

        client.post(
            f"/api/features/{dataset_id}/target",
            json={"target_column": "revenue", "problem_type": "regression"},
        )

        # Train linear_regression
        r = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        assert r.status_code == 202
        run_id = r.json()["model_run_ids"][0]

        # Poll until done
        import time
        for _ in range(60):
            runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
            run = next((r for r in runs if r["id"] == run_id), None)
            if run and run["status"] in ("done", "failed"):
                break
            time.sleep(0.5)

        # Tune — expects graceful 201 with tunable=False
        r = client.post(f"/api/models/{run_id}/tune")
        assert r.status_code == 201
        body = r.json()
        assert body["tunable"] is False
        assert body["tuned_model_run_id"] is None

    def test_tune_full_workflow(self, client, tmp_path):
        """Full integration: train RF, then tune, verify synchronous tuned result."""
        project_id, model_run_id = _setup_trained_model(
            client, tmp_path, algorithm="random_forest_regressor"
        )

        r = client.post(f"/api/models/{model_run_id}/tune")
        assert r.status_code == 201
        body = r.json()
        assert body["tunable"] is True
        assert "tuned_model_run_id" in body
        assert body["tuned_model_run_id"] is not None
        tuned_run_id = body["tuned_model_run_id"]

        # The tuned run should already be done (synchronous endpoint)
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        tuned = next((r for r in runs if r["id"] == tuned_run_id), None)

        assert tuned is not None
        assert tuned["status"] == "done"
        assert tuned["algorithm"] == "random_forest_regressor"
        assert tuned["metrics"] is not None
        assert "r2" in tuned["metrics"]

    def test_tune_creates_new_run_not_modifying_original(self, client, tmp_path):
        """Original run must remain unchanged after tuning."""
        project_id, model_run_id = _setup_trained_model(
            client, tmp_path, algorithm="random_forest_regressor"
        )

        # Capture original metrics before tuning
        runs_before = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        original = next(r for r in runs_before if r["id"] == model_run_id)
        original_metrics = original["metrics"]

        client.post(f"/api/models/{model_run_id}/tune")

        # Original run should be unchanged
        runs_after = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        original_after = next(r for r in runs_after if r["id"] == model_run_id)
        assert original_after["metrics"] == original_metrics
        assert original_after["algorithm"] == "random_forest_regressor"


# ---------------------------------------------------------------------------
# Chat intent detection tests
# ---------------------------------------------------------------------------

class TestChatTuneIntent:
    def _mock_anthropic(self):
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["I can tune your model."])
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream
        return mock_client

    def _setup_trained_project(self, client, tmp_path):
        from sqlmodel import Session
        from models.model_run import ModelRun
        from models.feature_set import FeatureSet

        r = client.post("/api/projects", json={"name": "Chat Tune Test"})
        project_id = r.json()["id"]

        r = client.post(
            "/api/data/upload",
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
            data={"project_id": project_id},
        )
        dataset_id = r.json()["dataset_id"]

        r = client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        feature_set_id = r.json()["feature_set_id"]

        client.post(
            f"/api/features/{dataset_id}/target",
            json={"target_column": "revenue", "problem_type": "regression"},
        )

        # Create a done ModelRun directly in DB (avoids background thread race)
        with Session(db_module.engine) as session:
            run = ModelRun(
                project_id=project_id,
                feature_set_id=feature_set_id,
                algorithm="random_forest_regressor",
                hyperparameters="{}",
                status="done",
                metrics=json.dumps({"r2": 0.85, "mae": 50.0, "rmse": 70.0}),
                summary="R² = 0.85 (good fit). On average, predictions are off by 50.00 units.",
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            model_run_id = run.id

        return project_id, model_run_id

    def test_tune_keyword_triggers_tune_event(self, client, tmp_path):
        project_id, _ = self._setup_trained_project(client, tmp_path)

        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            r = client.post(
                f"/api/chat/{project_id}",
                json={"message": "Can you tune my model to improve accuracy?"},
            )

        assert r.status_code == 200
        body = r.text
        assert '"type": "tune"' in body

    def test_optimize_keyword_triggers_tune_event(self, client, tmp_path):
        project_id, _ = self._setup_trained_project(client, tmp_path)

        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            r = client.post(
                f"/api/chat/{project_id}",
                json={"message": "Can you optimize the model performance?"},
            )

        assert r.status_code == 200
        assert '"type": "tune"' in r.text

    def test_improve_keyword_triggers_tune_event(self, client, tmp_path):
        project_id, _ = self._setup_trained_project(client, tmp_path)

        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            r = client.post(
                f"/api/chat/{project_id}",
                json={"message": "How can I improve my model?"},
            )

        assert r.status_code == 200
        assert '"type": "tune"' in r.text

    def test_no_tune_event_without_keyword(self, client, tmp_path):
        project_id, _ = self._setup_trained_project(client, tmp_path)

        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            r = client.post(
                f"/api/chat/{project_id}",
                json={"message": "What is the accuracy of my model?"},
            )

        assert r.status_code == 200
        assert '"type": "tune"' not in r.text

    def test_tune_event_includes_model_run_id(self, client, tmp_path):
        project_id, model_run_id = self._setup_trained_project(client, tmp_path)

        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            r = client.post(
                f"/api/chat/{project_id}",
                json={"message": "Can you tune this model?"},
            )

        assert '"type": "tune"' in r.text
        # Parse the tune event from SSE stream
        for line in r.text.splitlines():
            if line.startswith("data:") and '"type": "tune"' in line:
                event = json.loads(line[5:].strip())
                assert "tune" in event
                assert "model_run_id" in event["tune"]
                assert "algorithm" in event["tune"]
                break
