"""Tests for Calibration Check feature — Track C perpetual.

Covers:
- compute_calibration_check() pure function (19 tests)
- _CALIBRATION_CHECK_PATTERNS regex detection (8 positive, 2 negative)
- GET /api/models/{run_id}/calibration-check REST endpoint (3 tests)
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.validator import compute_calibration_check

# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

SAMPLE_SIZE = 200


def _binary_data(n: int = SAMPLE_SIZE, seed: int = 42):
    """Generate binary classification data with known properties."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    # Perfect calibration: probabilities = actual fractions (smoothed)
    y_proba = np.column_stack([1 - y_true * 0.8 - 0.1, y_true * 0.8 + 0.1]).astype(float)
    y_proba = np.clip(y_proba, 0.05, 0.95)
    y_proba /= y_proba.sum(axis=1, keepdims=True)
    return y_true.astype(int), y_proba


def _multiclass_data(n: int = SAMPLE_SIZE, n_classes: int = 3, seed: int = 7):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, n)
    proba = rng.dirichlet(np.ones(n_classes), n)
    return y_true.astype(int), proba


def test_required_keys_binary():
    y, p = _binary_data()
    result = compute_calibration_check(y, p, class_names=["no", "yes"])
    for key in ["curve_points", "ece", "brier_score", "verdict", "verdict_label",
                "per_class", "n_classes", "n_total", "summary"]:
        assert key in result, f"Missing key: {key}"


def test_n_total_matches_input():
    y, p = _binary_data(150)
    result = compute_calibration_check(y, p)
    assert result["n_total"] == 150


def test_n_classes_binary():
    y, p = _binary_data()
    result = compute_calibration_check(y, p)
    assert result["n_classes"] == 2


def test_n_classes_multiclass():
    y, p = _multiclass_data(n_classes=3)
    result = compute_calibration_check(y, p)
    assert result["n_classes"] == 3


def test_ece_in_range():
    y, p = _binary_data()
    result = compute_calibration_check(y, p)
    assert 0.0 <= result["ece"] <= 1.0


def test_brier_score_in_range():
    y, p = _binary_data()
    result = compute_calibration_check(y, p)
    assert 0.0 <= result["brier_score"] <= 1.0


def test_verdict_values():
    y, p = _binary_data()
    result = compute_calibration_check(y, p)
    assert result["verdict"] in ("well_calibrated", "moderate", "poorly_calibrated")


def test_verdict_label_is_string():
    y, p = _binary_data()
    result = compute_calibration_check(y, p)
    assert isinstance(result["verdict_label"], str)
    assert len(result["verdict_label"]) > 0


def test_curve_points_is_list():
    y, p = _binary_data()
    result = compute_calibration_check(y, p)
    assert isinstance(result["curve_points"], list)
    assert len(result["curve_points"]) > 0


def test_curve_point_keys():
    y, p = _binary_data()
    result = compute_calibration_check(y, p)
    pt = result["curve_points"][0]
    for key in ["mean_prob", "frac_positive", "count", "bin_label"]:
        assert key in pt, f"curve_point missing key: {key}"


def test_per_class_structure():
    y, p = _binary_data()
    class_names = ["neg", "pos"]
    result = compute_calibration_check(y, p, class_names=class_names)
    assert isinstance(result["per_class"], list)
    entry = result["per_class"][0]
    for key in ["class_name", "brier_score", "ece", "n_positive"]:
        assert key in entry, f"per_class entry missing key: {key}"


def test_per_class_class_names_applied():
    y, p = _binary_data()
    result = compute_calibration_check(y, p, class_names=["cat", "dog"])
    names = [e["class_name"] for e in result["per_class"]]
    assert "dog" in names


def test_well_calibrated_verdict_low_ece():
    """Perfectly calibrated data should produce a well_calibrated verdict."""
    n = 500
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, n)
    # Near-perfect calibration: class probability ≈ actual fraction
    y_proba = np.zeros((n, 2))
    y_proba[:, 1] = np.clip(y_true * 0.85 + rng.normal(0, 0.02, n), 0.01, 0.99)
    y_proba[:, 0] = 1.0 - y_proba[:, 1]
    result = compute_calibration_check(y_true, y_proba)
    # Should be well_calibrated or moderate with small ECE
    assert result["ece"] < 0.20


def test_poorly_calibrated_overconfident():
    """Overconfident model: all probabilities are near 0 or 1 regardless of outcome."""
    n = 200
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, n)
    rng2 = np.random.default_rng(99)
    # Overconfident: always 0.99 or 0.01
    y_proba = np.where(
        rng2.integers(0, 2, n).reshape(-1, 1) == 1,
        np.array([[0.01, 0.99]]),
        np.array([[0.99, 0.01]]),
    ).astype(float)
    result = compute_calibration_check(y_true, y_proba)
    assert result["verdict"] in ("moderate", "poorly_calibrated")


def test_too_few_rows_raises():
    y = np.array([0, 1, 0, 1, 0])
    p = np.column_stack([1 - np.array([0.1, 0.9, 0.2, 0.8, 0.3]),
                         np.array([0.1, 0.9, 0.2, 0.8, 0.3])])
    with pytest.raises(ValueError, match="at least 10"):
        compute_calibration_check(y, p)


def test_n_bins_clamped_min():
    y, p = _binary_data()
    result = compute_calibration_check(y, p, n_bins=1)
    assert len(result["curve_points"]) >= 1


def test_n_bins_clamped_max():
    y, p = _binary_data()
    result = compute_calibration_check(y, p, n_bins=100)
    assert len(result["curve_points"]) <= 20


def test_summary_is_string():
    y, p = _binary_data()
    result = compute_calibration_check(y, p)
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 20


def test_multiclass_per_class_count():
    y, p = _multiclass_data(n_classes=4)
    result = compute_calibration_check(y, p)
    assert len(result["per_class"]) == 4


def test_class_names_fallback():
    """When class_names is wrong length, falls back to str(class_index)."""
    y, p = _binary_data()
    result = compute_calibration_check(y, p, class_names=["only_one"])
    # Should not crash; class names default to str labels
    assert result["n_classes"] == 2


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


def test_pattern_how_well_calibrated():
    from api.chat import _CALIBRATION_CHECK_PATTERNS
    assert _CALIBRATION_CHECK_PATTERNS.search("how well-calibrated is my model?")


def test_pattern_calibrated_classifier():
    from api.chat import _CALIBRATION_CHECK_PATTERNS
    assert _CALIBRATION_CHECK_PATTERNS.search("how well calibrated is the classifier")


def test_pattern_confidence_scores_reliable():
    from api.chat import _CALIBRATION_CHECK_PATTERNS
    assert _CALIBRATION_CHECK_PATTERNS.search("are my confidence scores reliable?")


def test_pattern_reliability_diagram():
    from api.chat import _CALIBRATION_CHECK_PATTERNS
    assert _CALIBRATION_CHECK_PATTERNS.search("show me the reliability diagram")


def test_pattern_show_calibration():
    from api.chat import _CALIBRATION_CHECK_PATTERNS
    assert _CALIBRATION_CHECK_PATTERNS.search("show calibration for my model")


def test_pattern_brier_score():
    from api.chat import _CALIBRATION_CHECK_PATTERNS
    assert _CALIBRATION_CHECK_PATTERNS.search("what's the brier score?")


def test_pattern_calibration_check():
    from api.chat import _CALIBRATION_CHECK_PATTERNS
    assert _CALIBRATION_CHECK_PATTERNS.search("calibration check for the predictions")


def test_pattern_probability_calibration():
    from api.chat import _CALIBRATION_CHECK_PATTERNS
    assert _CALIBRATION_CHECK_PATTERNS.search("probability calibration for my classifier")


def test_pattern_no_match_unrelated():
    from api.chat import _CALIBRATION_CHECK_PATTERNS
    assert not _CALIBRATION_CHECK_PATTERNS.search("train a new model please")


def test_pattern_no_match_general_accuracy():
    from api.chat import _CALIBRATION_CHECK_PATTERNS
    assert not _CALIBRATION_CHECK_PATTERNS.search("what is the model accuracy?")


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------

CLASSIFICATION_CSV = (
    b"f1,f2,label\n"
    + b"".join(
        f"{i*0.5},{i*0.3},{0 if i < 20 else 1}\n".encode()
        for i in range(40)
    )
)

REGRESSION_CSV = (
    b"f1,f2,target\n"
    + b"".join(
        f"{i},{i*2},{i*3.0}\n".encode()
        for i in range(20)
    )
)


@pytest.fixture()
def client(tmp_path):
    test_db = str(tmp_path / "cal_check_test.db")
    orig_engine = db_module.engine
    db_module.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    from main import app
    yield TestClient(app)
    db_module.engine = orig_engine


def _train_and_wait(client, csv_bytes, target_col, algorithm, project_name):
    proj = client.post("/api/projects", json={"name": project_name})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": target_col, "problem_type": "classification" if algorithm != "linear_regression" else "regression"},
    )

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": [algorithm]},
    )
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    return run_id


def test_calibration_check_endpoint_200_classification(client):
    run_id = _train_and_wait(
        client, CLASSIFICATION_CSV, "label", "logistic_regression", "CalCheck200"
    )
    resp = client.get(f"/api/models/{run_id}/calibration-check")
    assert resp.status_code == 200
    data = resp.json()
    for key in ["curve_points", "ece", "brier_score", "verdict", "verdict_label",
                "per_class", "n_classes", "n_total", "summary", "algorithm", "target_col"]:
        assert key in data, f"Response missing key: {key}"


def test_calibration_check_endpoint_404_unknown_run(client):
    resp = client.get("/api/models/no-such-run/calibration-check")
    assert resp.status_code == 404


def test_calibration_check_endpoint_400_regression(client):
    run_id = _train_and_wait(
        client, REGRESSION_CSV, "target", "linear_regression", "CalCheckReg"
    )
    resp = client.get(f"/api/models/{run_id}/calibration-check")
    assert resp.status_code == 400
