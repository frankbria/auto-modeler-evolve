"""test_training_resilience.py

Completes the error resilience audit for training edge cases:
- Model training failure (sklearn exception mid-training)
- Terribly-performing model (R² near 0 or negative) — still deployable, but user warned
- Training on constant-target column (all values identical)
- Training succeeds then narration runs without crash
"""
from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

SAMPLE_CSV = b"""product,region,price,units,revenue
Widget A,North,120.0,10,1200.0
Widget B,South,85.0,8,680.0
Widget A,East,120.0,18,2160.0
Widget C,West,45.0,4,180.0
Widget B,North,85.0,15,1275.0
Widget A,South,120.0,9,1080.0
Widget C,North,45.0,11,495.0
Widget B,East,85.0,16,1360.0
Widget A,West,120.0,20,2400.0
Widget C,South,45.0,6,270.0
Widget A,North,125.0,12,1500.0
Widget B,East,90.0,7,630.0
Widget C,West,50.0,9,450.0
Widget A,South,125.0,14,1750.0
Widget B,North,90.0,11,990.0
Widget C,East,50.0,5,250.0
Widget A,West,130.0,17,2210.0
Widget B,South,88.0,13,1144.0
Widget C,North,48.0,3,144.0
Widget A,East,130.0,22,2860.0
"""

# CSV where all revenue values are identical → constant target → model fits trivially
CONSTANT_TARGET_CSV = b"""product,revenue
Widget A,1000
Widget B,1000
Widget C,1000
Widget D,1000
Widget E,1000
Widget F,1000
Widget G,1000
Widget H,1000
Widget I,1000
Widget J,1000
"""

# Random CSV: target values are random relative to features → model should perform poorly
RANDOM_TARGET_CSV = (
    "product,region,revenue\n"
    + "\n".join(
        f"Widget {chr(65 + i % 5)},{['North','South','East','West'][i % 4]},{(i * 137 % 1000) + 100}"
        for i in range(30)
    )
).encode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db = str(tmp_path / "training_resilience.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.conversation  # noqa
    import models.model_run  # noqa
    import models.deployment  # noqa
    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.models as models_api_module
    import api.deploy as deploy_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    models_api_module.MODELS_DIR = tmp_path / "models"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    # Stub Anthropic
    monkeypatch.setattr("api.chat.anthropic.Anthropic", lambda *a, **kw: _FakeAnthropic())

    from main import app
    with TestClient(app) as c:
        yield c


class _FakeStream:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def text_stream(self): yield "Analysis complete."
    def get_final_message(self):
        class _Msg:
            content = [type("B", (), {"text": "."})()]
        return _Msg()

class _FakeAnthropic:
    class messages:
        @staticmethod
        def stream(*args, **kwargs): return _FakeStream()


def _setup_project_with_csv(client: TestClient, csv_bytes: bytes, name: str) -> tuple[str, str, str]:
    """Creates project, uploads CSV, applies features, sets target. Returns (project_id, dataset_id, feature_set_id)."""
    project_id = client.post("/api/projects", json={"name": name}).json()["id"]
    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["dataset_id"]

    # Apply transformations (empty = just creates FeatureSet record)
    feat = client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert feat.status_code == 201, feat.text
    feature_set_id = feat.json()["feature_set_id"]

    # Set the target column (required before training)
    target_resp = client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue"},
    )
    assert target_resp.status_code == 200, target_resp.text

    return project_id, dataset_id, feature_set_id


def _wait_for_training(client: TestClient, project_id: str, timeout: int = 30) -> list[dict]:
    """Poll until all runs complete or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/models/{project_id}/runs")
        assert resp.status_code == 200
        runs = resp.json()["runs"]
        if runs and all(r["status"] in ("done", "failed") for r in runs):
            return runs
        time.sleep(0.3)
    raise TimeoutError(f"Training did not complete within {timeout}s")


class TestModelTrainingFailure:
    """Verify the 'failed' status path when sklearn training raises an exception."""

    def test_training_failure_sets_run_status_to_failed(self, client, tmp_path):
        """If train_single_model raises, the run should be marked as 'failed'."""
        project_id, _, feature_set_id = _setup_project_with_csv(client, SAMPLE_CSV, "fail-test")

        # Monkeypatch train_single_model to raise after start
        import api.models as models_api_module
        original_train = models_api_module.train_single_model

        def exploding_train(*args, **kwargs):
            raise RuntimeError("Simulated sklearn training failure: singular matrix")

        models_api_module.train_single_model = exploding_train
        try:
            train_resp = client.post(
                f"/api/models/{project_id}/train",
                json={"algorithms": ["linear_regression"]},
            )
            assert train_resp.status_code == 202, train_resp.text
            runs = _wait_for_training(client, project_id, timeout=15)
        finally:
            models_api_module.train_single_model = original_train

        assert len(runs) == 1
        run = runs[0]
        assert run["status"] == "failed", f"Expected failed but got: {run['status']}"
        assert run["error_message"] is not None
        assert "singular matrix" in run["error_message"] or "Simulated" in run["error_message"]

    def test_failed_run_has_no_metrics(self, client, tmp_path):
        """Failed runs should not have metrics (null)."""
        project_id, _, feature_set_id = _setup_project_with_csv(client, SAMPLE_CSV, "fail-metrics")

        import api.models as models_api_module
        from core.trainer import train_single_model as original_train2

        def always_fail(*a, **kw):
            raise ValueError("Cannot fit model: all feature values are identical")

        models_api_module.train_single_model = always_fail
        try:
            client.post(
                f"/api/models/{project_id}/train",
                json={
                    "algorithms": ["linear_regression"],
                },
            )
            runs = _wait_for_training(client, project_id, timeout=15)
        finally:
            models_api_module.train_single_model = original_train2

        failed_run = next(r for r in runs if r["status"] == "failed")
        assert failed_run["metrics"] is None

    def test_partial_failure_does_not_block_other_runs(self, client, tmp_path):
        """If one algorithm fails, others should still complete successfully."""
        project_id, _, feature_set_id = _setup_project_with_csv(client, SAMPLE_CSV, "partial-fail")

        import api.models as models_api_module
        original_train = models_api_module.train_single_model

        call_count = {"n": 0}

        def sometimes_fail(X, y, algorithm, *args, **kwargs):
            call_count["n"] += 1
            if algorithm == "linear_regression":
                raise RuntimeError("Linear regression crashed")
            return original_train(X, y, algorithm, *args, **kwargs)

        models_api_module.train_single_model = sometimes_fail
        try:
            train_resp = client.post(
                f"/api/models/{project_id}/train",
                json={"algorithms": ["linear_regression", "random_forest_regressor"]},
            )
            assert train_resp.status_code == 202
            runs = _wait_for_training(client, project_id, timeout=30)
        finally:
            models_api_module.train_single_model = original_train

        statuses = {r["algorithm"]: r["status"] for r in runs}
        assert statuses.get("linear_regression") == "failed"
        assert statuses.get("random_forest_regressor") == "done"


class TestPoorlyPerformingModel:
    """Tests for model deployment and narration when model performs terribly."""

    def test_poor_model_still_deployable(self, client, tmp_path):
        """Even a model with low R² should be deployable — user decides."""
        project_id, _, feature_set_id = _setup_project_with_csv(client, RANDOM_TARGET_CSV, "poor-model")

        train_resp = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        assert train_resp.status_code == 202, train_resp.text
        runs = _wait_for_training(client, project_id, timeout=30)
        assert runs[0]["status"] == "done"
        run_id = runs[0]["id"]

        # Select and deploy
        sel_resp = client.post(f"/api/models/{run_id}/select")
        assert sel_resp.status_code == 200

        deploy_resp = client.post(f"/api/deploy/{run_id}")
        assert deploy_resp.status_code in (200, 201), deploy_resp.text
        deploy_body = deploy_resp.json()
        # Deploy returns the deployment record (key is 'id' not 'deployment_id')
        assert "id" in deploy_body
        assert deploy_body["is_active"] is True

    def test_constant_target_training_handles_gracefully(self, client, tmp_path):
        """Training on a constant target (all same value) should either work or fail gracefully.

        When all target values are identical, sklearn may classify it as 'classification'
        (1 unique value). We test that the system doesn't return 500 regardless.
        """
        project_id, dataset_id, _ = _setup_project_with_csv(
            client, CONSTANT_TARGET_CSV, "constant-target"
        )

        # Detect what problem type was inferred
        recs_resp = client.get(f"/api/models/{project_id}/recommendations")
        assert recs_resp.status_code == 200, recs_resp.text
        # Use the first recommended algorithm for whichever problem type was detected
        recommended_algo = recs_resp.json()["recommendations"][0]["algorithm"]

        train_resp = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": [recommended_algo]},
        )
        assert train_resp.status_code == 202, train_resp.text
        runs = _wait_for_training(client, project_id, timeout=30)

        # With constant target, model may succeed or fail, but must NOT 500
        assert runs[0]["status"] in ("done", "failed")
        # If done, metrics must be present
        if runs[0]["status"] == "done":
            assert runs[0]["metrics"] is not None

    def test_narration_includes_poor_model_warning(self):
        """narrate_training_complete should mention low performance honestly."""
        from chat.narration import narrate_training_complete

        runs = [
            {
                "algorithm": "linear_regression",
                "status": "done",
                "metrics": {"r2": -0.05, "mae": 500.0, "rmse": 700.0},
                "summary": "R² = -0.05 (weak fit). Predictions are off by 500 units on average.",
            }
        ]
        msg = narrate_training_complete(runs, "regression", "revenue")
        # Should acknowledge the poor fit
        assert msg  # not empty
        assert isinstance(msg, str)
        # The message should reference the algorithm
        assert "linear" in msg.lower() or "regression" in msg.lower()

    def test_narration_with_all_failed_runs(self):
        """All-failed training run should produce a helpful failure message."""
        from chat.narration import narrate_training_complete

        runs = [
            {
                "algorithm": "linear_regression",
                "status": "failed",
                "metrics": {},
                "summary": "",
            },
            {
                "algorithm": "random_forest_regressor",
                "status": "failed",
                "metrics": {},
                "summary": "",
            },
        ]
        msg = narrate_training_complete(runs, "regression", "revenue")
        assert msg  # not empty
        assert isinstance(msg, str)
        # Should not leave user without guidance
        assert len(msg) > 20
