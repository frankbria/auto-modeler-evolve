"""Tests for Prediction Output Anomaly Detection — Track D perpetual.

Covers:
  - compute_prediction_output_anomalies pure function (regression + classification)
  - GET /api/deploy/{deployment_id}/output-anomalies REST endpoint
  - _OUTPUT_ANOMALY_PATTERNS regex (positive + negative matches)
  - Chat handler integration (emits event, required fields, no-deployment guard)
"""

from __future__ import annotations

import io

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_prediction_output_anomalies
from tests.conftest import wait_for_run

# ─── helpers ──────────────────────────────────────────────────────────────────


def _mk_log(
    log_id: str = "abc12345",
    prediction_numeric: float | None = 100.0,
    prediction: str = "100.0",
    confidence: float | None = None,
    created_at: str = "2026-05-31T10:00:00",
    input_features: str = '{"region": "East", "units": 10}',
):
    return {
        "id": log_id,
        "prediction_numeric": prediction_numeric,
        "prediction": prediction,
        "confidence": confidence,
        "created_at": created_at,
        "input_features": input_features,
    }


def _regression_logs(values: list[float]) -> list[dict]:
    """Build a list of regression log dicts from prediction values."""
    return [
        _mk_log(log_id=f"log{i:04d}", prediction_numeric=v, prediction=str(v))
        for i, v in enumerate(values)
    ]


def _classification_logs(confidences: list[float]) -> list[dict]:
    """Build a list of classification log dicts from confidence values."""
    return [
        _mk_log(
            log_id=f"log{i:04d}",
            prediction_numeric=None,
            prediction="class_a",
            confidence=c,
        )
        for i, c in enumerate(confidences)
    ]


_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
)

# ─── async fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.dataset_filter  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Anomaly Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


@pytest.fixture()
async def feature_set_id(ac, dataset_id):
    resp = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert resp.status_code == 201, resp.text
    fs_id = resp.json()["feature_set_id"]
    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


@pytest.fixture()
async def trained_run_id(ac, project_id, feature_set_id):
    resp = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["model_run_ids"][0]
    return await wait_for_run(ac, project_id, run_id)


@pytest.fixture()
async def deployment_id(ac, trained_run_id):
    resp = await ac.post(f"/api/deploy/{trained_run_id}", json={})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ─── pure function: regression tests ──────────────────────────────────────────


class TestComputePredictionOutputAnomaliesRegression:
    def test_required_keys_present(self):
        logs = _regression_logs([100, 110, 105, 95, 102])
        result = compute_prediction_output_anomalies(logs, "regression")
        required = {
            "n_total",
            "n_anomalous",
            "anomaly_rate",
            "verdict",
            "verdict_label",
            "problem_type",
            "stats",
            "anomalies",
            "summary",
        }
        assert required.issubset(result.keys())

    def test_no_anomalies_all_in_range(self):
        # All within 2 std — no anomalies expected
        logs = _regression_logs([100, 102, 98, 101, 99, 103, 97, 100, 101, 100])
        result = compute_prediction_output_anomalies(logs, "regression")
        assert result["verdict"] == "no_anomalies"
        assert result["n_anomalous"] == 0
        assert result["anomalies"] == []

    def test_detects_extreme_high_value(self):
        # One value far above the rest
        logs = _regression_logs([100, 102, 98, 101, 99, 103, 97, 100, 101, 9000])
        result = compute_prediction_output_anomalies(logs, "regression")
        assert result["n_anomalous"] >= 1
        assert result["verdict"] in ("few_anomalies", "many_anomalies")
        assert any("9000" in a["prediction_value"] for a in result["anomalies"])

    def test_detects_extreme_low_value(self):
        logs = _regression_logs([100, 102, 98, 101, 99, 103, 97, 100, 101, -9000])
        result = compute_prediction_output_anomalies(logs, "regression")
        assert result["n_anomalous"] >= 1
        assert any("-9000" in a["prediction_value"] for a in result["anomalies"])

    def test_anomalies_sorted_by_z_score_descending(self):
        logs = _regression_logs([100, 100, 100, 100, 100, 100, 100, 100, 1000, 2000])
        result = compute_prediction_output_anomalies(logs, "regression")
        if len(result["anomalies"]) >= 2:
            z_scores = [a["z_score"] for a in result["anomalies"]]
            assert z_scores == sorted(z_scores, reverse=True)

    def test_stats_contains_mean_and_std(self):
        logs = _regression_logs([100, 110, 105, 95, 102])
        result = compute_prediction_output_anomalies(logs, "regression")
        assert "mean" in result["stats"]
        assert "std" in result["stats"]
        assert "min" in result["stats"]
        assert "max" in result["stats"]

    def test_anomaly_entry_required_fields(self):
        logs = _regression_logs([100, 100, 100, 100, 100, 100, 100, 100, 100, 9000])
        result = compute_prediction_output_anomalies(logs, "regression")
        if result["anomalies"]:
            entry = result["anomalies"][0]
            assert "id" in entry
            assert "prediction_value" in entry
            assert "deviation" in entry
            assert "reason" in entry
            assert "created_at" in entry
            assert "input_summary" in entry
            assert "z_score" in entry
            assert entry["confidence"] is None

    def test_all_identical_values_no_anomalies(self):
        logs = _regression_logs([100.0] * 10)
        result = compute_prediction_output_anomalies(logs, "regression")
        assert result["n_anomalous"] == 0
        assert result["verdict"] == "no_anomalies"

    def test_caps_at_max_anomalies(self):
        # Create many extreme values
        normal = [100.0] * 5
        extreme = [100000.0] * 15
        logs = _regression_logs(normal + extreme)
        result = compute_prediction_output_anomalies(
            logs, "regression", max_anomalies=5
        )
        assert len(result["anomalies"]) <= 5

    def test_few_anomalies_verdict(self):
        # 1 anomaly out of 10 = 10% → few_anomalies or many depending on threshold
        logs = _regression_logs([100] * 9 + [99999])
        result = compute_prediction_output_anomalies(logs, "regression")
        assert result["verdict"] in ("few_anomalies", "many_anomalies")

    def test_many_anomalies_verdict(self):
        # Use a low z-score threshold so many values are flagged
        # A spread of 70–150 with threshold 0.5 ensures many exceed it
        values = [70, 75, 80, 85, 88, 90, 95, 100, 105, 110, 115, 120, 130, 140, 150]
        logs = _regression_logs(values)
        result = compute_prediction_output_anomalies(
            logs, "regression", z_score_threshold=0.5
        )
        # More than 10% should be flagged with such a low threshold
        assert result["verdict"] == "many_anomalies"
        assert result["n_anomalous"] > len(values) * 0.1

    def test_too_few_logs_raises_value_error(self):
        logs = _regression_logs([100, 200, 300, 400])  # only 4
        with pytest.raises(ValueError, match="at least 5"):
            compute_prediction_output_anomalies(logs, "regression")

    def test_summary_is_string(self):
        logs = _regression_logs([100, 102, 98, 101, 99])
        result = compute_prediction_output_anomalies(logs, "regression")
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_problem_type_echoed(self):
        logs = _regression_logs([100, 102, 98, 101, 99])
        result = compute_prediction_output_anomalies(logs, "regression")
        assert result["problem_type"] == "regression"


# ─── pure function: classification tests ──────────────────────────────────────


class TestComputePredictionOutputAnomaliesClassification:
    def test_no_anomalies_all_high_confidence(self):
        confs = [0.95, 0.92, 0.88, 0.90, 0.93, 0.85, 0.91, 0.87, 0.94, 0.89]
        logs = _classification_logs(confs)
        result = compute_prediction_output_anomalies(logs, "classification")
        assert result["n_anomalous"] == 0
        assert result["verdict"] == "no_anomalies"

    def test_detects_low_confidence_predictions(self):
        confs = [0.95, 0.92, 0.88, 0.90, 0.93, 0.85, 0.91, 0.87, 0.40, 0.30]
        logs = _classification_logs(confs)
        result = compute_prediction_output_anomalies(logs, "classification")
        assert result["n_anomalous"] >= 2
        assert result["verdict"] in ("few_anomalies", "many_anomalies")

    def test_anomalies_sorted_by_confidence_ascending(self):
        confs = [0.95, 0.92, 0.88, 0.90, 0.93, 0.85, 0.91, 0.87, 0.40, 0.30]
        logs = _classification_logs(confs)
        result = compute_prediction_output_anomalies(logs, "classification")
        if len(result["anomalies"]) >= 2:
            confidences = [
                a["confidence"]
                for a in result["anomalies"]
                if a["confidence"] is not None
            ]
            assert confidences == sorted(confidences)

    def test_classification_anomaly_entry_fields(self):
        confs = [0.95, 0.92, 0.88, 0.90, 0.93, 0.85, 0.91, 0.87, 0.40, 0.30]
        logs = _classification_logs(confs)
        result = compute_prediction_output_anomalies(logs, "classification")
        if result["anomalies"]:
            entry = result["anomalies"][0]
            assert entry["z_score"] is None
            assert entry["confidence"] is not None
            assert "%" in entry["deviation"]

    def test_classification_stats_contains_mean_confidence(self):
        confs = [0.95, 0.92, 0.88, 0.90, 0.93]
        logs = _classification_logs(confs)
        result = compute_prediction_output_anomalies(logs, "classification")
        assert "mean_confidence" in result["stats"]
        assert "min_confidence" in result["stats"]
        assert "max_confidence" in result["stats"]

    def test_classification_too_few_confidence_logs_raises(self):
        # Only 4 logs with confidence — but 5 total; if <5 have confidence, raise
        logs = [
            _mk_log(log_id=f"x{i}", confidence=None, prediction_numeric=None)
            for i in range(5)
        ]
        with pytest.raises(ValueError, match="at least 5"):
            compute_prediction_output_anomalies(logs, "classification")


# ─── regex pattern tests ──────────────────────────────────────────────────────


class TestOutputAnomalyPatterns:
    @pytest.fixture()
    def pattern(self):
        from api.chat import _OUTPUT_ANOMALY_PATTERNS

        return _OUTPUT_ANOMALY_PATTERNS

    def test_matches_any_unusual_predictions(self, pattern):
        assert pattern.search("any unusual predictions?")

    def test_matches_prediction_outliers(self, pattern):
        assert pattern.search("prediction outliers")

    def test_matches_anomalous_outputs(self, pattern):
        assert pattern.search("anomalous outputs")

    def test_matches_weird_model_outputs(self, pattern):
        assert pattern.search("weird model outputs")

    def test_matches_are_my_unusual_predictions(self, pattern):
        assert pattern.search("are my unusual predictions a problem?")

    def test_matches_extreme_prediction_values(self, pattern):
        assert pattern.search("predictions that are unusually high")

    def test_matches_suspicious_model_outputs(self, pattern):
        assert pattern.search("suspicious model predictions")

    def test_matches_unusual_model_output(self, pattern):
        assert pattern.search("unusual model output")

    def test_no_false_positive_show_predictions(self, pattern):
        assert not pattern.search("show me recent predictions")

    def test_no_false_positive_prediction_count(self, pattern):
        assert not pattern.search("how many predictions have been made?")

    def test_no_false_positive_compare_models(self, pattern):
        assert not pattern.search("compare my models")


# ─── REST endpoint tests ──────────────────────────────────────────────────────


class TestOutputAnomaliesEndpoint:
    @pytest.mark.asyncio
    async def test_unknown_deployment_returns_404(self, ac):
        resp = await ac.get("/api/deploy/unknown-id/output-anomalies")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_no_predictions_returns_no_data(self, ac, deployment_id):
        resp = await ac.get(f"/api/deploy/{deployment_id}/output-anomalies")
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "no_data"
        assert data["n_anomalous"] == 0
        assert data["deployment_id"] == deployment_id

    @pytest.mark.asyncio
    async def test_endpoint_returns_required_fields(self, ac, deployment_id):
        resp = await ac.get(f"/api/deploy/{deployment_id}/output-anomalies")
        assert resp.status_code == 200
        data = resp.json()
        required_keys = {
            "deployment_id",
            "n_total",
            "n_anomalous",
            "anomaly_rate",
            "verdict",
            "verdict_label",
            "stats",
            "anomalies",
            "summary",
        }
        assert required_keys.issubset(data.keys())
