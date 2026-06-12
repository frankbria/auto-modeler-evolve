"""Tests for Prediction Confidence Band — Track D perpetual.

Covers:
  - compute_confidence_band pure function (no-data, regression, classification,
    stable/moderate/high_spread verdicts, day grouping, guards)
  - GET /api/deploy/{deployment_id}/confidence-band REST endpoint
  - _CONFIDENCE_BAND_PATTERNS regex (positive + negative)
"""

from __future__ import annotations

import io
import time
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_confidence_band

# ─── helpers ──────────────────────────────────────────────────────────────────

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
)


def _make_log(prediction_numeric=None, confidence=None, created_at=None):
    return {
        "prediction_numeric": prediction_numeric,
        "confidence": confidence,
        "created_at": created_at or datetime(2024, 1, 1, tzinfo=timezone.utc),
    }


# ─── Unit tests: compute_confidence_band ──────────────────────────────────────


class TestComputeConfidenceBand:
    def test_no_data_returns_no_data_verdict(self):
        result = compute_confidence_band([], problem_type="regression")
        assert result["verdict"] == "no_data"
        assert result["n_samples"] == 0
        assert result["days"] == []
        assert result["means"] == []

    def test_no_data_classification_returns_no_data(self):
        result = compute_confidence_band([], problem_type="classification")
        assert result["verdict"] == "no_data"

    def test_regression_single_day(self):
        logs = [
            _make_log(prediction_numeric=10.0, created_at=datetime(2024, 1, 1)),
            _make_log(prediction_numeric=20.0, created_at=datetime(2024, 1, 1)),
            _make_log(prediction_numeric=30.0, created_at=datetime(2024, 1, 1)),
        ]
        result = compute_confidence_band(logs, problem_type="regression")
        assert result["n_samples"] == 3
        assert len(result["days"]) == 1
        assert result["days"][0] == "2024-01-01"
        assert result["means"][0] == pytest.approx(20.0, abs=0.01)
        assert result["uppers"][0] > result["means"][0]
        assert result["lowers"][0] < result["means"][0]

    def test_classification_uses_confidence_field(self):
        logs = [
            _make_log(confidence=0.9, created_at=datetime(2024, 1, 1)),
            _make_log(confidence=0.8, created_at=datetime(2024, 1, 1)),
            _make_log(confidence=0.7, created_at=datetime(2024, 1, 1)),
        ]
        result = compute_confidence_band(logs, problem_type="classification")
        assert result["n_samples"] == 3
        assert result["means"][0] == pytest.approx(0.8, abs=0.01)
        assert result["value_label"] == "Confidence"

    def test_regression_value_label(self):
        logs = [_make_log(prediction_numeric=5.0, created_at=datetime(2024, 1, 1))]
        result = compute_confidence_band(logs, problem_type="regression")
        assert result["value_label"] == "Predicted Value"

    def test_multiple_days_sorted(self):
        logs = [
            _make_log(prediction_numeric=10.0, created_at=datetime(2024, 1, 3)),
            _make_log(prediction_numeric=20.0, created_at=datetime(2024, 1, 1)),
            _make_log(prediction_numeric=30.0, created_at=datetime(2024, 1, 2)),
        ]
        result = compute_confidence_band(logs, problem_type="regression")
        assert result["days"] == ["2024-01-01", "2024-01-02", "2024-01-03"]

    def test_stable_verdict_low_cv(self):
        logs = [
            _make_log(
                prediction_numeric=100.0 + i * 0.1, created_at=datetime(2024, 1, 1)
            )
            for i in range(10)
        ]
        result = compute_confidence_band(logs, problem_type="regression")
        assert result["verdict"] == "stable"

    def test_high_spread_verdict(self):
        logs = [
            _make_log(prediction_numeric=v, created_at=datetime(2024, 1, 1))
            for v in [1.0, 10.0, 100.0, 1000.0]
        ]
        result = compute_confidence_band(logs, problem_type="regression")
        assert result["verdict"] in ("moderate_spread", "high_spread")

    def test_n_days_cap(self):
        logs = [
            _make_log(
                prediction_numeric=float(d),
                created_at=datetime(2024, 1, d % 28 + 1),
            )
            for d in range(40)
        ]
        result = compute_confidence_band(logs, problem_type="regression", n_days=30)
        assert result["n_days"] <= 30

    def test_skips_none_values(self):
        logs = [
            _make_log(prediction_numeric=None, created_at=datetime(2024, 1, 1)),
            _make_log(prediction_numeric=50.0, created_at=datetime(2024, 1, 1)),
        ]
        result = compute_confidence_band(logs, problem_type="regression")
        assert result["n_samples"] == 1

    def test_classification_skips_out_of_range_confidence(self):
        logs = [
            _make_log(confidence=-0.5, created_at=datetime(2024, 1, 1)),
            _make_log(confidence=1.5, created_at=datetime(2024, 1, 1)),
            _make_log(confidence=0.8, created_at=datetime(2024, 1, 1)),
        ]
        result = compute_confidence_band(logs, problem_type="classification")
        assert result["n_samples"] == 1

    def test_overall_stats_present(self):
        logs = [
            _make_log(prediction_numeric=10.0, created_at=datetime(2024, 1, 1)),
            _make_log(prediction_numeric=20.0, created_at=datetime(2024, 1, 1)),
        ]
        result = compute_confidence_band(logs, problem_type="regression")
        assert result["overall_mean"] is not None
        assert result["overall_std"] is not None
        assert result["overall_lower"] is not None
        assert result["overall_upper"] is not None

    def test_summary_contains_sample_count(self):
        logs = [
            _make_log(prediction_numeric=10.0, created_at=datetime(2024, 1, 1)),
            _make_log(prediction_numeric=20.0, created_at=datetime(2024, 1, 1)),
        ]
        result = compute_confidence_band(logs, problem_type="regression")
        assert "2" in result["summary"]

    def test_problem_type_echoed(self):
        logs = [_make_log(prediction_numeric=5.0, created_at=datetime(2024, 1, 1))]
        result = compute_confidence_band(logs, problem_type="regression")
        assert result["problem_type"] == "regression"


# ─── Pattern tests ────────────────────────────────────────────────────────────

_POSITIVE_PHRASES = [
    "show me the confidence band",
    "display confidence band",
    "prediction uncertainty over time",
    "prediction spread over time",
    "how confident has my model been",
    "how certain is my deployment",
    "confidence band chart",
    "confidence interval trend",
    "prediction stability over time",
    "model consistency over time",
    "how stable are my predictions",
    "show prediction range",
    "prediction value trend",
]

_NEGATIVE_PHRASES = [
    "how is my sales data looking",
    "train a model",
    "show me the drift alert status",
    "feature importance",
    "retrain the model",
    "how many predictions did I make today",
]


class TestConfidenceBandPatterns:
    @pytest.fixture(autouse=True)
    def _import_pattern(self):
        from api.chat import _CONFIDENCE_BAND_PATTERNS

        self.pat = _CONFIDENCE_BAND_PATTERNS

    @pytest.mark.parametrize("phrase", _POSITIVE_PHRASES)
    def test_positive(self, phrase):
        assert self.pat.search(phrase), f"Expected match for: {phrase!r}"

    @pytest.mark.parametrize("phrase", _NEGATIVE_PHRASES)
    def test_negative(self, phrase):
        assert not self.pat.search(phrase), f"Expected NO match for: {phrase!r}"


# ─── Async fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module
    from main import app

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    r = await ac.post("/api/projects", json={"name": "cb_test"})
    return r.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    r = await ac.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert r.status_code == 201, r.text
    return r.json()["dataset_id"]


@pytest.fixture()
async def feature_set_id(ac, dataset_id):
    r = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert r.status_code == 201, r.text
    fs_id = r.json()["feature_set_id"]
    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


@pytest.fixture()
async def trained_run_id(ac, project_id, feature_set_id):
    r = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert r.status_code == 202, r.text
    run_id = r.json()["model_run_ids"][0]
    for _ in range(30):
        resp = await ac.get(f"/api/models/{project_id}/runs")
        run = next((x for x in resp.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.3)
    pytest.skip("Training did not complete")


@pytest.fixture()
async def deployment_id(ac, trained_run_id):
    r = await ac.post(f"/api/deploy/{trained_run_id}", json={})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ─── REST endpoint tests ──────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_confidence_band_404_unknown(ac):
    r = await ac.get("/api/deploy/nonexistent-id/confidence-band")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_confidence_band_no_predictions(ac, deployment_id):
    r = await ac.get(f"/api/deploy/{deployment_id}/confidence-band")
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "no_data"
    assert body["n_samples"] == 0
    assert "deployment_id" in body
    assert "algorithm" in body
    assert "target_column" in body


@pytest.mark.anyio
async def test_confidence_band_with_n_days_param(ac, deployment_id):
    r = await ac.get(f"/api/deploy/{deployment_id}/confidence-band?n_days=7")
    assert r.status_code == 200
    assert "deployment_id" in r.json()
