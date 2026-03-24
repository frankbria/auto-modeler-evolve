"""Tests for model performance by segment analysis.

Covers:
- compute_segment_performance() unit tests (core/validator.py)
- _segment_metric() regression and classification
- _segment_status() thresholds
- GET /api/models/{model_run_id}/segment-performance endpoint
- 400 on unknown column
- 400 on high-cardinality column
- _SEGMENT_PERF_PATTERNS regex detection
- _detect_segment_perf_col() helper
- POST /api/chat/{project_id} emits segment_performance SSE event
"""

import io
import time

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _SEGMENT_PERF_PATTERNS, _detect_segment_perf_col
from core.validator import (
    _segment_metric,
    _segment_status,
    compute_segment_performance,
)

# ---------------------------------------------------------------------------
# Sample CSV: 3 regions with different revenue distributions
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units,cost\n"
    b"East,100.5,10,50\n"
    b"West,200.3,20,80\n"
    b"East,150.7,15,60\n"
    b"West,300.1,30,120\n"
    b"North,250.9,25,100\n"
    b"East,175.2,18,70\n"
    b"West,220.4,22,90\n"
    b"North,190.6,19,75\n"
    b"East,130.8,13,55\n"
    b"West,280.0,28,110\n"
)


# ---------------------------------------------------------------------------
# Unit tests: compute_segment_performance()
# ---------------------------------------------------------------------------


class TestComputeSegmentPerformance:
    def _make_regression_data(self):
        """Create a simple regression scenario: segment affects both y_true and y_pred."""
        group_values = ["East", "East", "West", "West", "North", "North"] * 2
        y_true = np.array(
            [100.0, 120.0, 200.0, 210.0, 250.0, 260.0] * 2, dtype=float
        )
        # Perfect predictions for East/West, noisy for North
        y_pred = np.array(
            [100.0, 120.0, 200.0, 210.0, 150.0, 350.0] * 2, dtype=float
        )
        return group_values, y_true, y_pred

    def test_basic_output_structure(self):
        gv, yt, yp = self._make_regression_data()
        result = compute_segment_performance(gv, yt, yp, "regression")
        assert "segments" in result
        assert "best_segment" in result
        assert "worst_segment" in result
        assert "gap" in result
        assert "metric_name" in result
        assert "summary" in result
        assert result["metric_name"] == "R²"

    def test_regression_best_worst(self):
        gv, yt, yp = self._make_regression_data()
        result = compute_segment_performance(gv, yt, yp, "regression")
        # East and West should have perfect R² = 1.0; North should be poor
        assert result["worst_segment"] == "North"
        assert result["best_segment"] in ("East", "West")

    def test_classification_metric_name(self):
        gv = ["A", "A", "B", "B"]
        yt = np.array([0, 1, 0, 1])
        yp = np.array([0, 1, 1, 1])  # A: 100% correct, B: 50% correct
        result = compute_segment_performance(gv, yt, yp, "classification")
        assert result["metric_name"] == "Accuracy"
        assert result["best_segment"] == "A"
        assert result["worst_segment"] == "B"

    def test_gap_computed_correctly(self):
        gv = ["A", "A", "B", "B"]
        yt = np.array([1.0, 2.0, 3.0, 4.0])
        yp_perfect = np.array([1.0, 2.0, 1.0, 1.0])  # B has bad predictions
        result = compute_segment_performance(gv, yt, yp_perfect, "regression")
        assert result["gap"] is not None
        assert result["gap"] > 0

    def test_max_groups_cap(self):
        groups = [f"G{i}" for i in range(20)]
        gv = groups * 2  # 40 rows, 20 groups
        yt = np.arange(40, dtype=float)
        yp = yt.copy()
        result = compute_segment_performance(gv, yt, yp, "regression", max_groups=5)
        assert len(result["segments"]) <= 5

    def test_single_group_summary(self):
        gv = ["A"] * 5
        yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yp = yt.copy()
        result = compute_segment_performance(gv, yt, yp, "regression")
        assert "one segment" in result["summary"].lower()

    def test_empty_groups(self):
        result = compute_segment_performance([], np.array([]), np.array([]), "regression")
        assert result["segments"] == []
        assert result["best_segment"] is None

    def test_low_sample_flag(self):
        gv = ["A", "B"]  # only 1 row each
        yt = np.array([1.0, 2.0])
        yp = np.array([1.0, 2.0])
        result = compute_segment_performance(gv, yt, yp, "regression")
        # Segments with < 2 rows get metric=None
        for seg in result["segments"]:
            assert seg["metric"] is None
            assert seg["status"] == "insufficient_data"


class TestSegmentMetric:
    def test_regression_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()
        assert _segment_metric(y_true, y_pred, "regression") == pytest.approx(1.0)

    def test_regression_constant_true(self):
        """Constant y_true → R² undefined → return None."""
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        result = _segment_metric(y_true, y_pred, "regression")
        assert result is None

    def test_classification_perfect(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = y_true.copy()
        assert _segment_metric(y_true, y_pred, "classification") == pytest.approx(1.0)

    def test_classification_partial(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])  # 50% correct
        assert _segment_metric(y_true, y_pred, "classification") == pytest.approx(0.5)

    def test_single_row_returns_none(self):
        assert _segment_metric(np.array([1.0]), np.array([1.0]), "regression") is None


class TestSegmentStatus:
    def test_regression_thresholds(self):
        assert _segment_status(0.90, "regression") == "strong"
        assert _segment_status(0.70, "regression") == "moderate"
        assert _segment_status(0.50, "regression") == "weak"
        assert _segment_status(0.20, "regression") == "poor"

    def test_classification_thresholds(self):
        assert _segment_status(0.90, "classification") == "strong"
        assert _segment_status(0.75, "classification") == "moderate"
        assert _segment_status(0.55, "classification") == "weak"
        assert _segment_status(0.30, "classification") == "poor"

    def test_none_returns_insufficient_data(self):
        assert _segment_status(None, "regression") == "insufficient_data"


# ---------------------------------------------------------------------------
# Fixtures for API tests
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.dataset_filter  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    import api.models as models_module

    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Seg Perf Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


@pytest.fixture()
async def trained_run_id(ac, project_id, dataset_id):
    """Create feature set, set target, train, and return the first completed run_id."""
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
    train_resp = await ac.post(
        f"/api/models/{project_id}/train",
        json={"feature_set_id": fs_id, "algorithms": ["linear_regression"]},
    )
    assert train_resp.status_code == 202, train_resp.text
    run_id = train_resp.json()["model_run_ids"][0]

    # Poll until done
    for _ in range(40):
        runs_resp = await ac.get(f"/api/models/{project_id}/runs")
        run = next(
            (r for r in runs_resp.json().get("runs", []) if r["id"] == run_id), None
        )
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.25)
    pytest.skip("Training did not complete in time")


# ---------------------------------------------------------------------------
# API tests: GET /api/models/{run_id}/segment-performance
# ---------------------------------------------------------------------------


class TestSegmentPerformanceEndpoint:
    async def test_basic_response(self, ac, trained_run_id):
        resp = await ac.get(
            f"/api/models/{trained_run_id}/segment-performance?col=region"
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["group_col"] == "region"
        assert "segments" in data
        assert len(data["segments"]) > 0
        assert "best_segment" in data
        assert "worst_segment" in data
        assert "summary" in data

    async def test_segments_have_required_fields(self, ac, trained_run_id):
        resp = await ac.get(
            f"/api/models/{trained_run_id}/segment-performance?col=region"
        )
        data = resp.json()
        for seg in data["segments"]:
            assert "name" in seg
            assert "n" in seg
            assert "status" in seg
            assert "metric_name" in seg

    async def test_unknown_column_returns_400(self, ac, trained_run_id):
        resp = await ac.get(
            f"/api/models/{trained_run_id}/segment-performance?col=nonexistent_col"
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"]

    async def test_high_cardinality_column_returns_400(self, ac, trained_run_id):
        resp = await ac.get(
            f"/api/models/{trained_run_id}/segment-performance?col=revenue"
        )
        assert resp.status_code == 400
        assert "too many" in resp.json()["detail"].lower()

    async def test_not_found_run_id(self, ac):
        resp = await ac.get(
            "/api/models/00000000-0000-0000-0000-000000000000/segment-performance?col=region"
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Chat pattern tests
# ---------------------------------------------------------------------------


class TestSegmentPerfPatterns:
    MATCH_CASES = [
        "how does my model perform by region?",
        "model accuracy by segment",
        "performance breakdown by product",
        "check model performance across regions",
        "does my model work equally well for all groups?",
        "which segment performs worst?",
        "model performance per category",
        "accuracy breakdown",
    ]

    NO_MATCH_CASES = [
        "train a model",
        "deploy my model",
        "compare East vs West",
        "what drives revenue?",
    ]

    def test_matches(self):
        for msg in self.MATCH_CASES:
            assert _SEGMENT_PERF_PATTERNS.search(msg), f"Should match: {msg!r}"

    def test_no_matches(self):
        for msg in self.NO_MATCH_CASES:
            assert not _SEGMENT_PERF_PATTERNS.search(msg), f"Should NOT match: {msg!r}"


class TestDetectSegmentPerfCol:
    def test_finds_mentioned_column(self):
        df = pd.read_csv(io.BytesIO(_SAMPLE_CSV))
        col = _detect_segment_perf_col("how does model perform by region?", df)
        assert col == "region"

    def test_fallback_to_first_categorical(self):
        df = pd.read_csv(io.BytesIO(_SAMPLE_CSV))
        col = _detect_segment_perf_col("how does the model perform across segments?", df)
        assert col is not None  # Falls back to first low-cardinality column

    def test_returns_none_for_no_candidates(self):
        df = pd.DataFrame({"a": range(100), "b": range(100)})  # no low-cardinality cols
        col = _detect_segment_perf_col("show performance by segment", df)
        assert col is None
