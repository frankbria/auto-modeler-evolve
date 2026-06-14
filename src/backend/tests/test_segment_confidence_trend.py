"""Tests for compute_segment_confidence_trend pure function, regex patterns, and REST endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.analyzer import compute_segment_confidence_trend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now():
    return datetime.utcnow()


def _cls_log(seg_val: str, confidence: float, days_ago: int = 0) -> dict:
    ts = _now() - timedelta(days=days_ago)
    return {
        "prediction_numeric": None,
        "confidence": confidence,
        "created_at": ts,
        "input_features_dict": {"category": seg_val},
    }


def _reg_log(seg_val: str, pred: float, days_ago: int = 0) -> dict:
    ts = _now() - timedelta(days=days_ago)
    return {
        "prediction_numeric": pred,
        "confidence": None,
        "created_at": ts,
        "input_features_dict": {"region": seg_val},
    }


# ---------------------------------------------------------------------------
# Pure-function: basic structure
# ---------------------------------------------------------------------------


def test_empty_logs_returns_no_data():
    result = compute_segment_confidence_trend([], "category")
    assert result["verdict"] == "no_data"
    assert result["n_segments"] == 0


def test_required_keys_present():
    result = compute_segment_confidence_trend([], "category")
    for key in (
        "segment_column",
        "problem_type",
        "metric",
        "n_segments",
        "n_samples",
        "segments",
        "most_confident_segment",
        "least_confident_segment",
        "calibration_gap",
        "verdict",
        "summary",
        "n_days",
    ):
        assert key in result, f"Missing key: {key}"


def test_single_log_returns_no_data():
    logs = [_cls_log("Premium", 0.8)]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    assert result["verdict"] == "no_data"


def test_classification_no_confidence_field_returns_no_confidence_data():
    logs = [
        {
            "prediction_numeric": None,
            "confidence": None,
            "created_at": _now(),
            "input_features_dict": {"category": "A"},
        },
        {
            "prediction_numeric": None,
            "confidence": None,
            "created_at": _now(),
            "input_features_dict": {"category": "B"},
        },
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    assert result["verdict"] == "no_confidence_data"
    assert result["metric"] == "confidence"


def test_classification_uses_confidence_field():
    logs = [
        _cls_log("Premium", 0.80, days_ago=10),
        _cls_log("Premium", 0.60, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    assert result["n_segments"] == 1
    seg = result["segments"][0]
    assert seg["last_mean"] < seg["first_mean"]


def test_regression_metric_is_prediction_spread():
    result = compute_segment_confidence_trend([], "region", "regression")
    assert result["metric"] == "prediction_spread"


def test_classification_metric_is_confidence():
    result = compute_segment_confidence_trend([], "category", "classification")
    assert result["metric"] == "confidence"


def test_segment_entry_required_fields():
    logs = [
        _cls_log("Premium", 0.80, days_ago=5),
        _cls_log("Premium", 0.70, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    assert result["n_segments"] >= 1
    seg = result["segments"][0]
    for key in (
        "name",
        "n_samples",
        "n_days_with_data",
        "direction",
        "direction_label",
        "change_pct",
        "first_mean",
        "last_mean",
        "daily_stats",
    ):
        assert key in seg, f"Missing segment key: {key}"


def test_classification_deteriorating_when_confidence_declines():
    # Confidence drops significantly over multiple days → deteriorating
    logs = [
        _cls_log("Basic", 0.90, days_ago=10),
        _cls_log("Basic", 0.80, days_ago=5),
        _cls_log("Basic", 0.55, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    basic = next(s for s in result["segments"] if s["name"] == "Basic")
    assert basic["direction"] == "deteriorating"
    assert basic["change_pct"] < 0


def test_classification_improving_when_confidence_rises():
    logs = [
        _cls_log("VIP", 0.55, days_ago=10),
        _cls_log("VIP", 0.70, days_ago=5),
        _cls_log("VIP", 0.90, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    vip = next(s for s in result["segments"] if s["name"] == "VIP")
    assert vip["direction"] == "improving"
    assert vip["change_pct"] > 0


def test_classification_stable_for_small_change():
    logs = [
        _cls_log("Standard", 0.80, days_ago=3),
        _cls_log("Standard", 0.81, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    std = next(s for s in result["segments"] if s["name"] == "Standard")
    assert std["direction"] == "stable"


def test_two_classification_segments():
    logs = [
        _cls_log("Good", 0.85, days_ago=10),
        _cls_log("Good", 0.88, days_ago=0),
        _cls_log("Bad", 0.85, days_ago=10),
        _cls_log("Bad", 0.50, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    assert result["n_segments"] == 2


def test_calibration_gap_detected():
    # Two segments with very different confidence levels (gap > 0.15)
    logs = [
        _cls_log("HighConf", 0.95, days_ago=5),
        _cls_log("HighConf", 0.95, days_ago=0),
        _cls_log("LowConf", 0.70, days_ago=5),
        _cls_log("LowConf", 0.70, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    assert result["calibration_gap"] is True
    assert result["verdict"] in (
        "calibration_gap",
        "stable",
        "mixed",
        "uniform_decline",
        "uniform_improving",
    )


def test_no_calibration_gap_when_segments_close():
    # Both segments have similar confidence (gap < 0.15)
    logs = [
        _cls_log("A", 0.80, days_ago=5),
        _cls_log("A", 0.80, days_ago=0),
        _cls_log("B", 0.82, days_ago=5),
        _cls_log("B", 0.82, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    assert result["calibration_gap"] is False


def test_most_and_least_confident_segment_identified():
    logs = [
        _cls_log("HighConf", 0.95, days_ago=5),
        _cls_log("HighConf", 0.95, days_ago=0),
        _cls_log("LowConf", 0.60, days_ago=5),
        _cls_log("LowConf", 0.60, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    assert result["most_confident_segment"] == "HighConf"
    assert result["least_confident_segment"] == "LowConf"


def test_regression_computes_spread():
    # For regression, metric = prediction_spread (CV%)
    logs = [
        _reg_log("East", float(v), days_ago=d)
        for v, d in [(100, 5), (110, 5), (90, 5), (100, 0), (105, 0), (95, 0)]
    ]
    result = compute_segment_confidence_trend(logs, "region", "regression")
    # Should return segments with mean = CV%
    assert result["metric"] == "prediction_spread"
    assert result["n_segments"] >= 1


def test_max_segments_cap():
    logs = []
    for i in range(12):
        for day in (5, 0):
            logs.append(_cls_log(f"Seg{i}", 0.7 + i * 0.02, days_ago=day))
    result = compute_segment_confidence_trend(
        logs, "category", "classification", max_segments=8
    )
    assert result["n_segments"] <= 8


def test_n_days_filter_excludes_old_logs():
    old = {
        "prediction_numeric": None,
        "confidence": 0.99,
        "created_at": _now() - timedelta(days=60),
        "input_features_dict": {"category": "Old"},
    }
    recent = _cls_log("Recent", 0.75, days_ago=5)
    result = compute_segment_confidence_trend(
        [old, recent], "category", "classification", n_days=30
    )
    # Old log excluded — only 1 log total, so no_data
    assert result["n_samples"] <= 1


def test_uniform_decline_verdict():
    logs = [
        _cls_log("A", 0.90, days_ago=10),
        _cls_log("A", 0.60, days_ago=0),
        _cls_log("B", 0.88, days_ago=10),
        _cls_log("B", 0.58, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    assert result["verdict"] == "uniform_decline"


def test_uniform_improving_verdict():
    logs = [
        _cls_log("A", 0.55, days_ago=10),
        _cls_log("A", 0.90, days_ago=0),
        _cls_log("B", 0.58, days_ago=10),
        _cls_log("B", 0.92, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    assert result["verdict"] == "uniform_improving"


def test_daily_stats_structure():
    logs = [
        _cls_log("Premium", 0.80, days_ago=5),
        _cls_log("Premium", 0.78, days_ago=5),
        _cls_log("Premium", 0.60, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    prem = next(s for s in result["segments"] if s["name"] == "Premium")
    for stat in prem["daily_stats"]:
        assert "date" in stat
        assert "mean" in stat
        assert "count" in stat


def test_sorted_by_abs_change_desc():
    logs = [
        _cls_log("BigDrop", 0.90, days_ago=10),
        _cls_log("BigDrop", 0.50, days_ago=0),
        _cls_log("SmallDrop", 0.80, days_ago=10),
        _cls_log("SmallDrop", 0.78, days_ago=0),
    ]
    result = compute_segment_confidence_trend(logs, "category", "classification")
    if result["n_segments"] >= 2:
        first_change = abs(result["segments"][0]["change_pct"])
        second_change = abs(result["segments"][1]["change_pct"])
        assert first_change >= second_change


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------

from api.chat import _SEGMENT_CONF_TREND_PATTERNS as _PAT  # noqa: E402

POSITIVE_CASES = [
    "confidence by segment",
    "confidence per region",
    "model confidence per category",
    "prediction confidence by region",
    "is my model less confident for West",
    "is the model less confident about Premium customers",
    "confidence trend by segment",
    "confidence trend for each region",
    "which segment has the lowest confidence",
    "which category has the worst confidence",
    "model uncertainty by region",
    "model uncertainty across segments",
    "confidence breakdown by segment",
    "low confidence for each category",
]

NEGATIVE_CASES = [
    "train my model",
    "prediction value trend by region",
    "segment drift analysis",
    "show feature importance",
    "what is my model accuracy",
    "show recent predictions",
    "how many predictions today",
]


@pytest.mark.parametrize("msg", POSITIVE_CASES)
def test_positive_matches(msg: str):
    assert _PAT.search(msg), f"Expected match: {msg!r}"


@pytest.mark.parametrize("msg", NEGATIVE_CASES)
def test_negative_no_match(msg: str):
    assert not _PAT.search(msg), f"Expected no match: {msg!r}"


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _create_deployment(engine) -> str:
    """Create a minimal project/run/deployment for REST tests."""
    import json as _json
    from sqlmodel import Session
    from models.deployment import Deployment
    from models.model_run import ModelRun
    from models.project import Project

    with Session(engine) as s:
        proj = Project(owner_id="test-default-owner", name=f"sct-test-{id(engine)}")
        s.add(proj)
        s.commit()
        s.refresh(proj)
        run = ModelRun(
            project_id=proj.id,
            algorithm="random_forest",
            status="completed",
            metrics=_json.dumps({"accuracy": 0.88}),
        )
        s.add(run)
        s.commit()
        s.refresh(run)
        dep = Deployment(
            model_run_id=run.id,
            project_id=proj.id,
            endpoint_path=f"/api/predict/{run.id}",
            dashboard_url=f"/predict/{run.id}",
            is_active=True,
            problem_type="classification",
        )
        s.add(dep)
        s.commit()
        return dep.id


def test_endpoint_404_unknown_deployment(client):
    resp = client.get("/api/deploy/nonexistent-id/segment-confidence-trend")
    assert resp.status_code == 404


def test_endpoint_returns_no_data_without_logs(client):
    from db import engine

    dep_id = _create_deployment(engine)
    resp = client.get(
        f"/api/deploy/{dep_id}/segment-confidence-trend?segment_col=category"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] in ("no_data", "no_confidence_data")
    assert data["deployment_id"] == dep_id


def test_endpoint_no_segment_col_returns_no_data(client):
    from db import engine

    dep_id = _create_deployment(engine)
    resp = client.get(f"/api/deploy/{dep_id}/segment-confidence-trend")
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "no_data"


def test_endpoint_required_fields_in_response(client):
    from db import engine

    dep_id = _create_deployment(engine)
    resp = client.get(
        f"/api/deploy/{dep_id}/segment-confidence-trend?segment_col=category"
    )
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "deployment_id",
        "segment_column",
        "problem_type",
        "metric",
        "n_segments",
        "n_samples",
        "segments",
        "most_confident_segment",
        "least_confident_segment",
        "calibration_gap",
        "verdict",
        "summary",
        "n_days",
    ):
        assert key in data, f"Missing field: {key}"
