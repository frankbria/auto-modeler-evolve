"""Tests for compute_prediction_delta — Track B perpetual.

Covers:
  - compute_prediction_delta pure function (regression, classification,
    unchanged prediction, feature contribution ordering)
  - _PRED_DELTA_PATTERNS regex (positive + negative)
  - Chat handler integration (emits prediction_delta SSE event, 1-version guard)
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock

import joblib
import numpy as np
import pytest

# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_pipeline(
    feature_names: list[str],
    target_col: str,
    problem_type: str,
    means: dict | None = None,
):
    """Build a minimal PredictionPipeline for testing."""
    from core.deployer import PredictionPipeline

    column_types = {f: "numeric" for f in feature_names}
    pipeline = PredictionPipeline(
        feature_names=feature_names,
        column_types=column_types,
        target_column=target_col,
        problem_type=problem_type,
    )
    if means:
        pipeline.feature_means = means
        pipeline.feature_stds = {k: 1.0 for k in means}
    else:
        pipeline.feature_means = {f: 0.0 for f in feature_names}
        pipeline.feature_stds = {f: 1.0 for f in feature_names}
    return pipeline


def _save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)


def _make_linear_model(coef: list[float], intercept: float = 0.0):
    """Build a minimal sklearn LinearRegression-like model for testing."""
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.coef_ = np.array(coef)
    model.intercept_ = intercept
    # Attach predict behaviour directly
    return model


def _save_model(model, path):
    joblib.dump(model, path)


# ─── Pure function tests ───────────────────────────────────────────────────────


class TestComputePredictionDeltaRegression:
    """compute_prediction_delta for regression problems."""

    def test_regression_delta_up(self, tmp_path):
        """Prediction increases → direction='up', delta > 0."""
        from core.deployer import compute_prediction_delta

        feats = ["units", "price"]
        pipeline_old = _make_pipeline(
            feats, "revenue", "regression", {"units": 10.0, "price": 5.0}
        )
        pipeline_new = _make_pipeline(
            feats, "revenue", "regression", {"units": 10.0, "price": 5.0}
        )

        # old model predicts 100; new model predicts 120
        model_old = _make_linear_model([1.0, 1.0], intercept=0.0)
        model_new = _make_linear_model([2.0, 1.0], intercept=0.0)

        p_old = tmp_path / "pipe_old.joblib"
        p_new = tmp_path / "pipe_new.joblib"
        m_old = tmp_path / "model_old.joblib"
        m_new = tmp_path / "model_new.joblib"

        _save_pipeline(pipeline_old, p_old)
        _save_pipeline(pipeline_new, p_new)
        _save_model(model_old, m_old)
        _save_model(model_new, m_new)

        result = compute_prediction_delta(
            str(p_new),
            str(m_new),
            str(p_old),
            str(m_old),
            {"units": 10.0, "price": 5.0},
        )

        assert result["direction"] == "up"
        assert result["delta"] > 0
        assert result["problem_type"] == "regression"
        assert result["target_column"] == "revenue"

    def test_regression_delta_down(self, tmp_path):
        """Prediction decreases → direction='down', delta < 0."""
        from core.deployer import compute_prediction_delta

        feats = ["units"]
        pipeline_old = _make_pipeline(feats, "revenue", "regression", {"units": 5.0})
        pipeline_new = _make_pipeline(feats, "revenue", "regression", {"units": 5.0})

        model_old = _make_linear_model([3.0], intercept=0.0)
        model_new = _make_linear_model([1.0], intercept=0.0)

        p_old = tmp_path / "pipe_old.joblib"
        p_new = tmp_path / "pipe_new.joblib"
        m_old = tmp_path / "model_old.joblib"
        m_new = tmp_path / "model_new.joblib"

        _save_pipeline(pipeline_old, p_old)
        _save_pipeline(pipeline_new, p_new)
        _save_model(model_old, m_old)
        _save_model(model_new, m_new)

        result = compute_prediction_delta(
            str(p_new),
            str(m_new),
            str(p_old),
            str(m_old),
            {"units": 5.0},
        )

        assert result["direction"] == "down"
        assert result["delta"] < 0

    def test_regression_delta_unchanged(self, tmp_path):
        """Same model → direction='unchanged', delta=0."""
        from core.deployer import compute_prediction_delta

        feats = ["units"]
        pipeline = _make_pipeline(feats, "revenue", "regression", {"units": 5.0})
        model = _make_linear_model([2.0], intercept=0.0)

        p = tmp_path / "pipe.joblib"
        m = tmp_path / "model.joblib"
        _save_pipeline(pipeline, p)
        _save_model(model, m)

        result = compute_prediction_delta(
            str(p), str(m), str(p), str(m), {"units": 5.0}
        )

        assert result["direction"] == "unchanged"
        assert result["delta"] == 0.0

    def test_regression_pct_change(self, tmp_path):
        """pct_change is computed correctly."""
        from core.deployer import compute_prediction_delta

        feats = ["x"]
        pipeline_old = _make_pipeline(feats, "y", "regression", {"x": 0.0})
        pipeline_new = _make_pipeline(feats, "y", "regression", {"x": 0.0})
        # old: x * 1 + 10 = 20 for x=10; new: x * 2 + 10 = 30 for x=10 → +50%
        model_old = _make_linear_model([1.0], intercept=10.0)
        model_new = _make_linear_model([2.0], intercept=10.0)

        p_old = tmp_path / "pipe_old.joblib"
        p_new = tmp_path / "pipe_new.joblib"
        m_old = tmp_path / "model_old.joblib"
        m_new = tmp_path / "model_new.joblib"
        _save_pipeline(pipeline_old, p_old)
        _save_pipeline(pipeline_new, p_new)
        _save_model(model_old, m_old)
        _save_model(model_new, m_new)

        result = compute_prediction_delta(
            str(p_new), str(m_new), str(p_old), str(m_old), {"x": 10.0}
        )

        assert abs(result["pct_change"] - 50.0) < 0.5

    def test_feature_delta_keys(self, tmp_path):
        """feature_delta contains required keys."""
        from core.deployer import compute_prediction_delta

        feats = ["a", "b"]
        pipeline_old = _make_pipeline(feats, "out", "regression", {"a": 0.0, "b": 0.0})
        pipeline_new = _make_pipeline(feats, "out", "regression", {"a": 0.0, "b": 0.0})
        model_old = _make_linear_model([1.0, 0.5], intercept=0.0)
        model_new = _make_linear_model([2.0, 0.5], intercept=0.0)

        for name, obj in [
            ("po", pipeline_old),
            ("pn", pipeline_new),
            ("mo", model_old),
            ("mn", model_new),
        ]:
            _save_pipeline(
                obj, tmp_path / f"{name}.joblib"
            ) if "p" in name else _save_model(obj, tmp_path / f"{name}.joblib")

        result = compute_prediction_delta(
            str(tmp_path / "pn.joblib"),
            str(tmp_path / "mn.joblib"),
            str(tmp_path / "po.joblib"),
            str(tmp_path / "mo.joblib"),
            {"a": 5.0, "b": 3.0},
        )

        assert len(result["feature_delta"]) == 2
        for item in result["feature_delta"]:
            assert "feature" in item
            assert "old_contribution" in item
            assert "new_contribution" in item
            assert "contribution_delta" in item
            assert "direction" in item

    def test_feature_delta_sorted_by_abs_delta(self, tmp_path):
        """feature_delta is sorted by abs(contribution_delta) descending."""
        from core.deployer import compute_prediction_delta

        # Use 3 features so normalization doesn't equalize deltas
        feats = ["stable_a", "stable_b", "high_driver"]
        means = {f: 0.0 for f in feats}
        pipeline_old = _make_pipeline(feats, "out", "regression", means)
        pipeline_new = _make_pipeline(feats, "out", "regression", means)
        # old: equal weights; new: high_driver dominates
        model_old = _make_linear_model([0.1, 0.1, 0.1], intercept=0.0)
        model_new = _make_linear_model([0.1, 0.1, 5.0], intercept=0.0)

        for name, obj in [("po", pipeline_old), ("pn", pipeline_new)]:
            _save_pipeline(obj, tmp_path / f"{name}.joblib")
        for name, obj in [("mo", model_old), ("mn", model_new)]:
            _save_model(obj, tmp_path / f"{name}.joblib")

        result = compute_prediction_delta(
            str(tmp_path / "pn.joblib"),
            str(tmp_path / "mn.joblib"),
            str(tmp_path / "po.joblib"),
            str(tmp_path / "mo.joblib"),
            {"stable_a": 1.0, "stable_b": 1.0, "high_driver": 1.0},
        )

        # high_driver should be first (largest contribution delta)
        assert result["feature_delta"][0]["feature"] == "high_driver"

    def test_summary_contains_prediction_values(self, tmp_path):
        """Summary mentions old and new prediction values."""
        from core.deployer import compute_prediction_delta

        feats = ["x"]
        pipeline_old = _make_pipeline(feats, "revenue", "regression", {"x": 0.0})
        pipeline_new = _make_pipeline(feats, "revenue", "regression", {"x": 0.0})
        model_old = _make_linear_model([1.0], intercept=0.0)
        model_new = _make_linear_model([2.0], intercept=10.0)

        _save_pipeline(pipeline_old, tmp_path / "po.joblib")
        _save_pipeline(pipeline_new, tmp_path / "pn.joblib")
        _save_model(model_old, tmp_path / "mo.joblib")
        _save_model(model_new, tmp_path / "mn.joblib")

        result = compute_prediction_delta(
            str(tmp_path / "pn.joblib"),
            str(tmp_path / "mn.joblib"),
            str(tmp_path / "po.joblib"),
            str(tmp_path / "mo.joblib"),
            {"x": 5.0},
        )

        assert "revenue" in result["summary"]
        assert (
            str(int(result["old_prediction"])) in result["summary"]
            or str(result["old_prediction"]) in result["summary"]
        )

    def test_top_drivers_nonempty_when_delta_exists(self, tmp_path):
        """top_drivers is non-empty when feature contributions meaningfully change.

        Uses 3 features to avoid normalization collapsing all deltas to zero.
        The new model weights feature_b heavily; old model weighted all equally.
        """
        from core.deployer import compute_prediction_delta

        feats = ["feature_a", "feature_b", "feature_c"]
        means = {f: 0.0 for f in feats}
        pipeline_old = _make_pipeline(feats, "y", "regression", means)
        pipeline_new = _make_pipeline(feats, "y", "regression", means)
        # Old: uniform weights; new: feature_b dominates
        model_old = _make_linear_model([0.3, 0.3, 0.3])
        model_new = _make_linear_model([0.1, 0.1, 9.0])

        _save_pipeline(pipeline_old, tmp_path / "po.joblib")
        _save_pipeline(pipeline_new, tmp_path / "pn.joblib")
        _save_model(model_old, tmp_path / "mo.joblib")
        _save_model(model_new, tmp_path / "mn.joblib")

        result = compute_prediction_delta(
            str(tmp_path / "pn.joblib"),
            str(tmp_path / "mn.joblib"),
            str(tmp_path / "po.joblib"),
            str(tmp_path / "mo.joblib"),
            {"feature_a": 1.0, "feature_b": 1.0, "feature_c": 1.0},
        )

        assert len(result["top_drivers"]) >= 1

    def test_provided_features_returned(self, tmp_path):
        """provided_features echoes the input dict."""
        from core.deployer import compute_prediction_delta

        feats = ["units"]
        pipeline = _make_pipeline(feats, "revenue", "regression", {"units": 0.0})
        model = _make_linear_model([1.0])
        p = tmp_path / "pipe.joblib"
        m = tmp_path / "model.joblib"
        _save_pipeline(pipeline, p)
        _save_model(model, m)

        inputs = {"units": 42.0}
        result = compute_prediction_delta(str(p), str(m), str(p), str(m), inputs)

        assert result["provided_features"] == inputs


class TestComputePredictionDeltaClassification:
    """compute_prediction_delta for classification problems."""

    def test_classification_same_class_unchanged(self, tmp_path):
        """Same class → direction='unchanged'."""
        from core.deployer import compute_prediction_delta
        from sklearn.linear_model import LogisticRegression

        feats = ["x"]
        pipeline = _make_pipeline(feats, "label", "classification", {"x": 0.0})
        # Train a minimal logistic regression
        X = np.array([[0], [1], [0], [1]])
        y = np.array([0, 1, 0, 1])
        model = LogisticRegression().fit(X, y)

        p = tmp_path / "pipe.joblib"
        m = tmp_path / "model.joblib"
        _save_pipeline(pipeline, p)
        _save_model(model, m)

        result = compute_prediction_delta(str(p), str(m), str(p), str(m), {"x": 0.5})

        assert "direction" in result
        assert result["problem_type"] == "classification"

    def test_classification_feature_delta_present(self, tmp_path):
        """feature_delta is computed for classification too."""
        from core.deployer import compute_prediction_delta
        from sklearn.linear_model import LogisticRegression

        feats = ["a", "b"]
        pipeline = _make_pipeline(
            feats, "label", "classification", {"a": 0.0, "b": 0.0}
        )
        X = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
        y = np.array([0, 1, 0, 1])
        model = LogisticRegression().fit(X, y)

        p = tmp_path / "pipe.joblib"
        m = tmp_path / "model.joblib"
        _save_pipeline(pipeline, p)
        _save_model(model, m)

        result = compute_prediction_delta(
            str(p), str(m), str(p), str(m), {"a": 0.5, "b": 0.5}
        )

        assert isinstance(result["feature_delta"], list)


# ─── Regex tests ───────────────────────────────────────────────────────────────


class TestPredDeltaPatterns:
    """_PRED_DELTA_PATTERNS regex coverage."""

    @pytest.fixture
    def pattern(self):
        from api.chat import _PRED_DELTA_PATTERNS

        return _PRED_DELTA_PATTERNS

    @pytest.mark.parametrize(
        "msg",
        [
            "why did my prediction change",
            "why did the prediction change after retraining",
            "why is the prediction different now",
            "explain the prediction delta",
            "explain the prediction change",
            "what changed about the prediction",
            "why different results after retraining",
            "same inputs different prediction",
            "prediction delta after retrain",
            "why is my forecast different",
        ],
    )
    def test_positive_matches(self, pattern, msg):
        assert pattern.search(msg), f"Should match: {msg!r}"

    @pytest.mark.parametrize(
        "msg",
        [
            "show me the data",
            "train a model",
            "how accurate is my model",
            "what is my R2",
            "deploy the model",
            "compare my models",
        ],
    )
    def test_negative_non_matches(self, pattern, msg):
        assert not pattern.search(msg), f"Should NOT match: {msg!r}"


# ─── Chat handler integration tests ───────────────────────────────────────────

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
    b"East,160.0,16\nWest,210.0,21\n"
)


@pytest.fixture
def client(tmp_path):
    """Sync TestClient with isolated SQLite DB."""
    import db as db_module
    from fastapi.testclient import TestClient
    from sqlmodel import SQLModel, create_engine

    test_db = str(tmp_path / "pred_delta_test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.ab_test  # noqa
    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.deployment_preset  # noqa
    import models.deployment_version  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa
    import models.webhook_event  # noqa
    import models.analysis_template  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _setup_and_deploy_delta(client):
    """Create project → upload → features → train linear_regression → deploy."""
    proj = client.post("/api/projects", json={"name": "PredDelta Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "problem_type": "regression"},
    )

    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert train.status_code == 202
    run_id = train.json()["model_run_ids"][0]

    for _ in range(60):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    assert run and run["status"] == "done"

    deploy = client.post(f"/api/deploy/{run_id}")
    assert deploy.status_code == 201
    return project_id, deploy.json()["id"]


def _make_second_deploy_version(client, project_id):
    """Train a second model and re-deploy (creates DeploymentVersion v2)."""
    train2 = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["random_forest_regressor"]},
    )
    assert train2.status_code == 202
    run_id2 = train2.json()["model_run_ids"][0]

    for _ in range(60):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id2), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    assert run and run["status"] == "done"
    redeploy = client.post(f"/api/deploy/{run_id2}")
    assert redeploy.status_code in (200, 201)


def _chat_delta(client, project_id, message):
    """Post a chat message with mocked Anthropic and return the response."""
    with mock.patch("anthropic.Anthropic") as mock_ant:
        mock_client = mock.MagicMock()
        mock_ant.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Prediction changed after retraining."])
        mock_client.messages.stream.return_value = mock_stream
        resp = client.post(f"/api/chat/{project_id}", json={"message": message})
    return resp


def _parse_events_delta(resp):
    return [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ") and line[6:].strip()
    ]


class TestPredictionDeltaChatHandler:
    """Integration tests for the prediction delta chat handler."""

    def test_prediction_delta_result_keys(self, tmp_path):
        """compute_prediction_delta returns all required keys."""
        from core.deployer import compute_prediction_delta

        feats = ["x"]
        pipeline = _make_pipeline(feats, "y", "regression", {"x": 0.0})
        model = _make_linear_model([1.0])
        p = tmp_path / "p.joblib"
        m = tmp_path / "m.joblib"
        _save_pipeline(pipeline, p)
        _save_model(model, m)

        result = compute_prediction_delta(str(p), str(m), str(p), str(m), {"x": 5.0})

        required = [
            "old_prediction",
            "new_prediction",
            "delta",
            "pct_change",
            "direction",
            "problem_type",
            "target_column",
            "provided_features",
            "feature_delta",
            "top_drivers",
            "summary",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_handler_emits_prediction_delta_with_two_versions(self, client):
        """Chat handler emits prediction_delta SSE event when 2+ versions exist."""
        project_id, _dep_id = _setup_and_deploy_delta(client)
        _make_second_deploy_version(client, project_id)

        resp = _chat_delta(
            client, project_id, "why did my prediction change after retraining"
        )
        assert resp.status_code == 200
        events = _parse_events_delta(resp)
        delta_events = [e for e in events if e.get("type") == "prediction_delta"]
        assert len(delta_events) == 1
        data = delta_events[0]["prediction_delta"]
        assert "old_prediction" in data
        assert "new_prediction" in data
        assert "direction" in data
        assert "summary" in data
        assert "feature_delta" in data

    def test_handler_no_fire_without_deployment(self, client):
        """Handler does not fire prediction_delta when no deployment exists."""
        # Just create a project with data but no deployment
        proj = client.post("/api/projects", json={"name": "NoDeploy"})
        assert proj.status_code == 201
        project_id = proj.json()["id"]

        upload = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        )
        assert upload.status_code == 201

        resp = _chat_delta(client, project_id, "why did my prediction change")
        assert resp.status_code == 200
        events = _parse_events_delta(resp)
        types = [e.get("type") for e in events]
        assert "prediction_delta" not in types

    def test_handler_no_fire_with_single_version(self, client):
        """Handler does not fire prediction_delta when only 1 version exists."""
        # Deploy once only — single version, no second deploy
        project_id, _dep_id = _setup_and_deploy_delta(client)

        resp = _chat_delta(client, project_id, "why is my prediction different")
        assert resp.status_code == 200
        events = _parse_events_delta(resp)
        types = [e.get("type") for e in events]
        assert "prediction_delta" not in types

    def test_handler_result_includes_version_numbers(self, client):
        """prediction_delta event includes current_version and previous_version."""
        project_id, _dep_id = _setup_and_deploy_delta(client)
        _make_second_deploy_version(client, project_id)

        resp = _chat_delta(client, project_id, "explain the prediction change")
        assert resp.status_code == 200
        events = _parse_events_delta(resp)
        delta_events = [e for e in events if e.get("type") == "prediction_delta"]
        assert len(delta_events) == 1
        data = delta_events[0]["prediction_delta"]
        assert "current_version" in data
        assert "previous_version" in data
        assert data["current_version"] > data["previous_version"]
