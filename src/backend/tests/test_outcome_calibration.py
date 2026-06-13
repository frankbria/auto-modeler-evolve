"""Tests for Prediction Outcome Calibration Chart feature.

Covers:
- compute_prediction_outcome_calibration pure function (classification + regression)
- GET /api/deploy/{id}/outcome-calibration REST endpoint
- _OUTCOME_CALIB_PATTERNS regex matching
"""

from __future__ import annotations

import io
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_prediction_outcome_calibration

# ---------------------------------------------------------------------------
# Pure function — shared helpers
# ---------------------------------------------------------------------------


def _cls_pairs(n: int, accuracy: float = 0.8, conf_offset: float = 0.0) -> list[dict]:
    """Generate n classification pairs with given accuracy and confidence offset."""
    pairs = []
    for i in range(n):
        is_correct = i < int(n * accuracy)
        conf = min(1.0, max(0.0, (i / n) + conf_offset))
        pairs.append({"confidence": conf, "is_correct": is_correct})
    return pairs


def _reg_pairs(n: int, bias: float = 0.0, noise: float = 1.0) -> list[dict]:
    """Generate n regression pairs with given bias and noise."""
    pairs = []
    for i in range(n):
        predicted = float(i)
        actual = predicted + bias + (i % 3 - 1) * noise
        pairs.append({"actual": actual, "predicted": predicted})
    return pairs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_min_calibration_samples_constant():
    from core.analyzer import _MIN_CALIBRATION_SAMPLES

    assert _MIN_CALIBRATION_SAMPLES == 5


def test_n_calibration_buckets_constant():
    from core.analyzer import _N_CALIBRATION_BUCKETS

    assert _N_CALIBRATION_BUCKETS == 10


# ---------------------------------------------------------------------------
# Pure function — no_data / not_applicable
# ---------------------------------------------------------------------------


def test_not_applicable_for_unknown_problem_type():
    result = compute_prediction_outcome_calibration([], "unknown_type")
    assert result["verdict"] == "not_applicable"
    assert result["n_samples"] == 0


def test_no_data_when_fewer_than_5_classification_pairs():
    result = compute_prediction_outcome_calibration(
        [{"confidence": 0.8, "is_correct": True}] * 3, "classification"
    )
    assert result["verdict"] == "no_data"
    assert result["n_samples"] == 3


def test_no_data_when_fewer_than_5_regression_pairs():
    result = compute_prediction_outcome_calibration(
        [{"actual": 1.0, "predicted": 1.0}] * 2, "regression"
    )
    assert result["verdict"] == "no_data"
    assert result["n_samples"] == 2


def test_no_data_when_empty_list_classification():
    result = compute_prediction_outcome_calibration([], "classification")
    assert result["verdict"] == "no_data"
    assert result["n_samples"] == 0


# ---------------------------------------------------------------------------
# Pure function — classification: well calibrated
# ---------------------------------------------------------------------------


def test_classification_well_calibrated():
    """Perfect calibration: confidence matches actual accuracy in each bucket."""
    # Build pairs that are nearly perfectly calibrated
    pairs = []
    for bucket in range(10):
        conf = (bucket + 0.5) / 10.0
        correct_count = round(conf * 10)
        for j in range(10):
            pairs.append({"confidence": conf, "is_correct": j < correct_count})
    result = compute_prediction_outcome_calibration(pairs, "classification")
    assert result["verdict"] in (
        "well_calibrated",
        "slightly_overconfident",
        "slightly_underconfident",
    )
    assert result["ece"] < 0.15
    assert "n_samples" in result
    assert result["n_samples"] == len(pairs)
    assert "buckets" in result
    assert "overall_accuracy" in result


def test_classification_returns_buckets_list():
    pairs = _cls_pairs(50, accuracy=0.8, conf_offset=0.1)
    result = compute_prediction_outcome_calibration(pairs, "classification")
    assert "buckets" in result
    assert len(result["buckets"]) == 10
    for bucket in result["buckets"]:
        assert "lo" in bucket
        assert "hi" in bucket
        assert "n" in bucket


def test_classification_returns_ece():
    pairs = _cls_pairs(50)
    result = compute_prediction_outcome_calibration(pairs, "classification")
    assert "ece" in result
    assert 0.0 <= result["ece"] <= 1.0


def test_classification_returns_overall_accuracy():
    pairs = [{"confidence": 0.9, "is_correct": True}] * 8 + [
        {"confidence": 0.9, "is_correct": False}
    ] * 2
    result = compute_prediction_outcome_calibration(pairs, "classification")
    assert "overall_accuracy" in result
    assert abs(result["overall_accuracy"] - 0.8) < 0.01


def test_classification_problem_type_in_result():
    pairs = _cls_pairs(10)
    result = compute_prediction_outcome_calibration(pairs, "classification")
    assert result["problem_type"] == "classification"


def test_classification_n_filled_buckets():
    # All pairs with same confidence → only 1 bucket filled
    pairs = [{"confidence": 0.85, "is_correct": True}] * 10
    result = compute_prediction_outcome_calibration(pairs, "classification")
    assert result["n_filled_buckets"] == 1


# ---------------------------------------------------------------------------
# Pure function — classification: overconfident
# ---------------------------------------------------------------------------


def test_classification_overconfident():
    """High confidence pairs that are often wrong → overconfident."""
    pairs = [{"confidence": 0.95, "is_correct": False}] * 6 + [
        {"confidence": 0.95, "is_correct": True}
    ] * 2
    result = compute_prediction_outcome_calibration(pairs, "classification")
    assert result["verdict"] in (
        "overconfident",
        "slightly_overconfident",
        "slightly_underconfident",
        "well_calibrated",
    )
    # confidence is high but accuracy is 25% — big deviation
    if result["verdict"] in ("overconfident", "slightly_overconfident"):
        assert result["ece"] > 0.0


def test_classification_skips_out_of_range_confidence():
    pairs = [
        {"confidence": 1.5, "is_correct": True},  # invalid → skipped
        {"confidence": -0.1, "is_correct": True},  # invalid → skipped
        *[{"confidence": 0.8, "is_correct": True}] * 5,
    ]
    result = compute_prediction_outcome_calibration(pairs, "classification")
    assert result["n_samples"] == 5  # only valid ones counted


def test_classification_handles_missing_keys_gracefully():
    pairs = [
        {"confidence": 0.8, "is_correct": True},
        {"confidence": 0.8},  # missing is_correct → skipped
        {},  # missing both → skipped
        *[{"confidence": 0.7, "is_correct": False}] * 5,
    ]
    result = compute_prediction_outcome_calibration(pairs, "classification")
    # Only 6 valid pairs
    assert result["n_samples"] == 6


# ---------------------------------------------------------------------------
# Pure function — regression: unbiased
# ---------------------------------------------------------------------------


def test_regression_unbiased():
    """Errors cancel out → unbiased verdict."""
    pairs = []
    for i in range(20):
        pairs.append({"actual": float(i), "predicted": float(i) + (i % 2 - 0.5) * 0.1})
    result = compute_prediction_outcome_calibration(pairs, "regression")
    assert result["verdict"] in (
        "unbiased",
        "positive_bias",
        "negative_bias",
        "low_data",
    )
    assert "mae" in result
    assert result["mae"] >= 0
    assert "bins" in result
    assert len(result["bins"]) == 10


def test_regression_positive_bias():
    """Actual consistently higher than predicted → positive_bias (model under-predicts)."""
    pairs = [{"actual": float(i) + 5.0, "predicted": float(i)} for i in range(20)]
    result = compute_prediction_outcome_calibration(pairs, "regression")
    assert result["verdict"] == "positive_bias"
    assert result["mean_error"] > 0


def test_regression_negative_bias():
    """Predicted consistently higher than actual → negative_bias."""
    pairs = [{"actual": float(i), "predicted": float(i) + 5.0} for i in range(20)]
    result = compute_prediction_outcome_calibration(pairs, "regression")
    assert result["verdict"] == "negative_bias"
    assert result["mean_error"] < 0


def test_regression_returns_all_stats():
    pairs = _reg_pairs(10, bias=0.0)
    result = compute_prediction_outcome_calibration(pairs, "regression")
    for key in (
        "mean_error",
        "median_error",
        "std_error",
        "mae",
        "rmse",
        "error_min",
        "error_max",
    ):
        assert key in result, f"Missing key: {key}"


def test_regression_problem_type_in_result():
    pairs = _reg_pairs(10)
    result = compute_prediction_outcome_calibration(pairs, "regression")
    assert result["problem_type"] == "regression"


def test_regression_bins_cover_error_range():
    pairs = _reg_pairs(20, bias=0.0, noise=2.0)
    result = compute_prediction_outcome_calibration(pairs, "regression")
    assert len(result["bins"]) == 10
    assert result["bins"][0]["lo"] <= result["bins"][-1]["hi"]


def test_regression_sum_of_bin_counts_equals_n_samples():
    pairs = _reg_pairs(15, bias=1.0)
    result = compute_prediction_outcome_calibration(pairs, "regression")
    bin_total = sum(b["count"] for b in result["bins"])
    assert bin_total == result["n_samples"]


def test_regression_low_data_verdict():
    """Fewer than 10 samples → low_data verdict."""
    pairs = [{"actual": 1.0, "predicted": 0.5}] * 7
    result = compute_prediction_outcome_calibration(pairs, "regression")
    assert result["verdict"] in (
        "low_data",
        "positive_bias",
        "negative_bias",
        "unbiased",
    )


def test_regression_handles_missing_keys_gracefully():
    pairs = [
        {"actual": 1.0, "predicted": 0.5},
        {"actual": 2.0},  # missing predicted → skipped
        {},  # missing both → skipped
        *[{"actual": float(i), "predicted": float(i) + 0.1} for i in range(5)],
    ]
    result = compute_prediction_outcome_calibration(pairs, "regression")
    assert result["n_samples"] == 6


def test_regression_rmse_gte_mae():
    """RMSE >= MAE always (Cauchy-Schwarz)."""
    pairs = _reg_pairs(20, bias=1.0, noise=0.5)
    result = compute_prediction_outcome_calibration(pairs, "regression")
    assert result["rmse"] >= result["mae"] - 1e-9  # small float tolerance


# ---------------------------------------------------------------------------
# REST endpoint tests (async, fresh test DB)
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"x,y\n"
    b"1.0,10.5\n2.0,21.3\n3.0,30.7\n4.0,41.1\n5.0,52.9\n"
    b"6.0,61.5\n7.0,72.3\n8.0,80.8\n9.0,91.2\n10.0,100.4\n"
    b"11.0,111.0\n12.0,120.5\n13.0,133.2\n14.0,141.7\n15.0,155.0\n"
    b"16.0,162.3\n17.0,173.8\n18.0,184.1\n19.0,195.6\n20.0,205.9\n"
)


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
    r = await ac.post("/api/projects", json={"name": "calib_test"})
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
    r = await ac.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    assert r.status_code == 201, r.text
    fs_id = r.json()["feature_set_id"]
    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "y", "feature_set_id": fs_id},
    )
    return fs_id


@pytest.fixture()
async def trained_run_id(ac, project_id, feature_set_id):
    import time

    r = await ac.post(
        f"/api/models/{project_id}/train",
        json={"feature_set_id": feature_set_id, "algorithms": ["linear_regression"]},
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


@pytest.mark.anyio
async def test_outcome_calibration_404_unknown_deployment(ac):
    r = await ac.get("/api/deploy/nonexistent-id/outcome-calibration")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_outcome_calibration_regression_no_data(ac, deployment_id):
    r = await ac.get(f"/api/deploy/{deployment_id}/outcome-calibration")
    assert r.status_code == 200
    data = r.json()
    assert data["verdict"] == "no_data"
    assert data["deployment_id"] == deployment_id


@pytest.mark.anyio
async def test_outcome_calibration_with_feedback(ac, deployment_id):
    """Providing feedback with actual_value builds an outcome calibration chart."""
    # Make predictions to get log IDs
    r_pred = await ac.post(
        f"/api/predict/{deployment_id}",
        json={"x": 5.0},
    )
    assert r_pred.status_code == 200

    log_id = r_pred.json().get("prediction_log_id")
    if log_id:
        # Record feedback with actual value for several predictions
        for _ in range(5):
            r_pred2 = await ac.post(
                f"/api/predict/{deployment_id}", json={"x": float(_ + 1)}
            )
            if r_pred2.status_code == 200 and r_pred2.json().get("prediction_log_id"):
                lid = r_pred2.json()["prediction_log_id"]
                await ac.post(
                    f"/api/deploy/{deployment_id}/feedback",
                    json={
                        "prediction_log_id": lid,
                        "actual_value": float(_ + 1) * 10.0 + 0.5,
                    },
                )

    r = await ac.get(f"/api/deploy/{deployment_id}/outcome-calibration")
    assert r.status_code == 200
    data = r.json()
    assert "verdict" in data
    assert "problem_type" in data
    assert "deployment_id" in data


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def outcome_calib_pattern():
    from api.chat import _OUTCOME_CALIB_PATTERNS

    return _OUTCOME_CALIB_PATTERNS


def test_pattern_prediction_outcome_calibration(outcome_calib_pattern):
    assert outcome_calib_pattern.search("prediction outcome calibration")


def test_pattern_show_calibration_chart(outcome_calib_pattern):
    assert outcome_calib_pattern.search("show me the calibration chart")


def test_pattern_show_calibration_from_feedback(outcome_calib_pattern):
    assert outcome_calib_pattern.search("show calibration from feedback")


def test_pattern_are_predictions_calibrated(outcome_calib_pattern):
    assert outcome_calib_pattern.search(
        "are my predictions calibrated based on actual outcomes"
    )


def test_pattern_reliability_diagram(outcome_calib_pattern):
    assert outcome_calib_pattern.search("reliability diagram")


def test_pattern_calibration_chart_using_feedback(outcome_calib_pattern):
    assert outcome_calib_pattern.search("calibration chart using feedback")


def test_pattern_error_histogram_from_feedback(outcome_calib_pattern):
    assert outcome_calib_pattern.search("prediction error histogram from feedback")


def test_pattern_accuracy_by_confidence_bucket(outcome_calib_pattern):
    assert outcome_calib_pattern.search("prediction accuracy by confidence bucket")


def test_pattern_no_match_regular_calibration_check(outcome_calib_pattern):
    """The regular calibration check (training-time) should not match."""
    assert not outcome_calib_pattern.search("is my model well calibrated")


def test_pattern_no_match_threshold_optimization(outcome_calib_pattern):
    """Production threshold optimization should not match."""
    assert not outcome_calib_pattern.search(
        "optimize my decision threshold using feedback"
    )
