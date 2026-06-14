"""Tests for compute_segment_prediction_trend pure function, regex patterns, and REST endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.analyzer import compute_segment_prediction_trend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now():
    return datetime.utcnow()


def _log(
    seg_val: str, pred: float, days_ago: int = 0, confidence: float | None = None
) -> dict:
    ts = _now() - timedelta(days=days_ago)
    return {
        "prediction_numeric": pred,
        "confidence": confidence,
        "created_at": ts,
        "input_features_dict": {"region": seg_val, "units": 5.0},
    }


def _cls_log(seg_val: str, confidence: float, days_ago: int = 0) -> dict:
    ts = _now() - timedelta(days=days_ago)
    return {
        "prediction_numeric": None,
        "confidence": confidence,
        "created_at": ts,
        "input_features_dict": {"category": seg_val},
    }


# ---------------------------------------------------------------------------
# Pure-function: basic structure
# ---------------------------------------------------------------------------


def test_empty_logs_returns_no_data():
    result = compute_segment_prediction_trend([], "region")
    assert result["verdict"] == "no_data"
    assert result["n_segments"] == 0
    assert result["n_samples"] == 0


def test_required_keys_present_no_data():
    result = compute_segment_prediction_trend([], "region")
    for key in (
        "segment_column",
        "problem_type",
        "n_segments",
        "n_samples",
        "segments",
        "most_improved_segment",
        "most_declining_segment",
        "verdict",
        "summary",
        "n_days",
    ):
        assert key in result, f"Missing key: {key}"


def test_single_log_not_enough_data():
    logs = [_log("East", 100.0, days_ago=0)]
    result = compute_segment_prediction_trend(logs, "region")
    assert result["verdict"] == "no_data"
    assert result["n_samples"] == 0 or result["n_segments"] == 0


def test_missing_segment_col_excluded():
    logs = [
        {
            "prediction_numeric": 50.0,
            "confidence": None,
            "created_at": _now(),
            "input_features_dict": {"units": 5.0},
        },
        {
            "prediction_numeric": 60.0,
            "confidence": None,
            "created_at": _now(),
            "input_features_dict": {"units": 6.0},
        },
    ]
    result = compute_segment_prediction_trend(logs, "region")
    assert result["verdict"] == "no_data"


def test_two_segments_two_logs_builds_result():
    logs = [
        _log("East", 100.0, days_ago=5),
        _log("East", 110.0, days_ago=0),
        _log("West", 200.0, days_ago=5),
        _log("West", 190.0, days_ago=0),
    ]
    result = compute_segment_prediction_trend(logs, "region")
    assert result["n_segments"] == 2
    names = {s["name"] for s in result["segments"]}
    assert "East" in names
    assert "West" in names


def test_segment_entry_required_fields():
    logs = [
        _log("East", 100.0, days_ago=5),
        _log("East", 120.0, days_ago=0),
    ]
    result = compute_segment_prediction_trend(logs, "region")
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


def test_trending_up_detection():
    logs = [
        _log("East", 100.0, days_ago=10),
        _log("East", 130.0, days_ago=5),
        _log("East", 160.0, days_ago=0),
    ]
    result = compute_segment_prediction_trend(logs, "region")
    east = next(s for s in result["segments"] if s["name"] == "East")
    assert east["direction"] == "trending_up"
    assert east["change_pct"] > 0


def test_trending_down_detection():
    logs = [
        _log("West", 200.0, days_ago=10),
        _log("West", 160.0, days_ago=5),
        _log("West", 120.0, days_ago=0),
    ]
    result = compute_segment_prediction_trend(logs, "region")
    west = next(s for s in result["segments"] if s["name"] == "West")
    assert west["direction"] == "trending_down"
    assert west["change_pct"] < 0


def test_stable_direction_small_change():
    # < 2% per period change → stable
    logs = [
        _log("North", 100.0, days_ago=3),
        _log("North", 100.5, days_ago=0),
    ]
    result = compute_segment_prediction_trend(logs, "region")
    north = next(s for s in result["segments"] if s["name"] == "North")
    assert north["direction"] == "stable"


def test_diverging_verdict():
    logs = [
        _log("East", 100.0, days_ago=10),
        _log("East", 150.0, days_ago=0),
        _log("West", 200.0, days_ago=10),
        _log("West", 150.0, days_ago=0),
    ]
    result = compute_segment_prediction_trend(logs, "region")
    assert result["verdict"] == "diverging"
    assert result["most_improved_segment"] == "East"
    assert result["most_declining_segment"] == "West"


def test_all_improving_verdict():
    logs = [
        _log("East", 100.0, days_ago=10),
        _log("East", 160.0, days_ago=0),
        _log("West", 100.0, days_ago=10),
        _log("West", 155.0, days_ago=0),
    ]
    result = compute_segment_prediction_trend(logs, "region")
    assert result["verdict"] == "all_improving"
    assert result["most_improved_segment"] is not None
    assert result["most_declining_segment"] is None


def test_all_declining_verdict():
    logs = [
        _log("East", 200.0, days_ago=10),
        _log("East", 130.0, days_ago=0),
        _log("West", 200.0, days_ago=10),
        _log("West", 140.0, days_ago=0),
    ]
    result = compute_segment_prediction_trend(logs, "region")
    assert result["verdict"] == "all_declining"
    assert result["most_declining_segment"] is not None
    assert result["most_improved_segment"] is None


def test_classification_uses_confidence():
    logs = [
        _cls_log("Premium", 0.60, days_ago=10),
        _cls_log("Premium", 0.90, days_ago=0),
    ]
    result = compute_segment_prediction_trend(
        logs, "category", problem_type="classification"
    )
    assert result["n_segments"] == 1
    seg = result["segments"][0]
    assert seg["last_mean"] > seg["first_mean"]
    assert seg["direction"] == "trending_up"


def test_max_segments_cap():
    logs = []
    for i in range(12):
        for day in (5, 0):
            logs.append(_log(f"Seg{i}", float(100 + i), days_ago=day))
    result = compute_segment_prediction_trend(logs, "region", max_segments=8)
    assert result["n_segments"] <= 8


def test_daily_stats_structure():
    logs = [
        _log("East", 100.0, days_ago=5),
        _log("East", 110.0, days_ago=5),
        _log("East", 120.0, days_ago=0),
    ]
    result = compute_segment_prediction_trend(logs, "region")
    east = next(s for s in result["segments"] if s["name"] == "East")
    for stat in east["daily_stats"]:
        assert "date" in stat
        assert "mean" in stat
        assert "count" in stat
    assert east["n_days_with_data"] >= 1


def test_n_days_param_filters_old_logs():
    old_log = {
        "prediction_numeric": 999.0,
        "confidence": None,
        "created_at": _now() - timedelta(days=60),
        "input_features_dict": {"region": "East"},
    }
    recent = _log("East", 100.0, days_ago=5)
    result = compute_segment_prediction_trend([old_log, recent], "region", n_days=30)
    # Old log excluded; only 1 log total → no_data (< 2 predictions)
    assert result["n_samples"] <= 1


def test_sorted_by_abs_change():
    logs = [
        _log("BigDrop", 200.0, days_ago=10),
        _log("BigDrop", 100.0, days_ago=0),
        _log("SmallUp", 100.0, days_ago=10),
        _log("SmallUp", 105.0, days_ago=0),
    ]
    result = compute_segment_prediction_trend(logs, "region")
    if result["n_segments"] >= 2:
        first_change = abs(result["segments"][0]["change_pct"])
        second_change = abs(result["segments"][1]["change_pct"])
        assert first_change >= second_change


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------

from api.chat import _SEGMENT_PRED_TREND_PATTERNS as _PAT  # noqa: E402

POSITIVE_CASES = [
    "prediction trend by segment",
    "prediction trend per region",
    "prediction trends for category",
    "prediction trends across segments",
    "which segment is improving",
    "which category is declining",
    "what segment is trending down",
    "segment-level prediction trend",
    "how are my predictions changing for each region",
    "how are predictions changing by segment",
    "average prediction for West trending",
    "average prediction trending up",
    "which region has the highest prediction",
    "which segment has the most improved prediction",
]

NEGATIVE_CASES = [
    "train my model",
    "what is my model accuracy",
    "show feature importance",
    "which features drifted most",
    "segment drift analysis",
    "prediction analytics",
    "show recent predictions",
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
        proj = Project(owner_id="test-default-owner", name=f"spt-test-{id(engine)}")
        s.add(proj)
        s.commit()
        s.refresh(proj)
        run = ModelRun(
            project_id=proj.id,
            algorithm="linear_regression",
            status="completed",
            metrics=_json.dumps({"r2": 0.8}),
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
        )
        s.add(dep)
        s.commit()
        return dep.id


def test_endpoint_404_unknown_deployment(client):
    resp = client.get("/api/deploy/nonexistent-id/segment-prediction-trend")
    assert resp.status_code == 404


def test_endpoint_returns_no_data_without_logs(client):
    from db import engine

    dep_id = _create_deployment(engine)
    resp = client.get(
        f"/api/deploy/{dep_id}/segment-prediction-trend?segment_col=region"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "no_data"
    assert data["deployment_id"] == dep_id


def test_endpoint_no_categorical_feature_returns_no_data(client):
    from db import engine

    dep_id = _create_deployment(engine)
    # No segment_col + no pipeline → no_data
    resp = client.get(f"/api/deploy/{dep_id}/segment-prediction-trend")
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "no_data"


def test_endpoint_required_fields_in_response(client):
    from db import engine

    dep_id = _create_deployment(engine)
    resp = client.get(
        f"/api/deploy/{dep_id}/segment-prediction-trend?segment_col=region"
    )
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "deployment_id",
        "segment_column",
        "problem_type",
        "n_segments",
        "n_samples",
        "segments",
        "most_improved_segment",
        "most_declining_segment",
        "verdict",
        "summary",
        "n_days",
    ):
        assert key in data, f"Missing field: {key}"
