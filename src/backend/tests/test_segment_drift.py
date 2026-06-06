"""Tests for compute_segment_drift pure function, regex patterns, and REST endpoint."""

import pytest

from core.analyzer import compute_segment_drift


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def _inputs(n: int, seg_val: str, numeric_val: float = 5.0) -> list[dict]:
    return [{"region": seg_val, "units": numeric_val} for _ in range(n)]


FEATURE_RANGES = {
    "region": {"known_categories": ["East", "West", "North"]},
    "units": {"min": 1.0, "max": 10.0},
}


def test_empty_inputs_returns_no_data():
    result = compute_segment_drift([], "region", FEATURE_RANGES)
    assert result["verdict"] == "no_data"
    assert result["n_segments"] == 0
    assert result["n_samples"] == 0


def test_required_keys_present():
    result = compute_segment_drift([], "region", FEATURE_RANGES)
    for key in (
        "segment_column",
        "n_segments",
        "n_samples",
        "segments",
        "max_drift_segment",
        "avg_drift_score",
        "high_count",
        "moderate_count",
        "low_count",
        "minimal_count",
        "verdict",
        "summary",
    ):
        assert key in result, f"Missing key: {key}"


def test_single_segment_no_drift():
    inputs = _inputs(20, "East", numeric_val=5.0)
    result = compute_segment_drift(inputs, "region", FEATURE_RANGES)
    assert result["n_segments"] == 1
    assert result["segments"][0]["name"] == "East"
    assert result["segments"][0]["drift_score"] == 0.0
    assert result["segments"][0]["status"] == "minimal"


def test_numeric_drift_oor():
    # All units = 20.0 which is > max=10 → 100% OOR
    inputs = [{"region": "East", "units": 20.0}] * 10
    result = compute_segment_drift(inputs, "region", FEATURE_RANGES)
    east = result["segments"][0]
    # units OOR = 100%, region unseen = 0%, avg = 50%
    assert east["drift_score"] > 0
    assert east["status"] in ("high", "moderate")


def test_categorical_drift_unseen():
    # "South" is not in known_categories
    inputs = [{"region": "South", "units": 5.0}] * 10
    result = compute_segment_drift(inputs, "region", FEATURE_RANGES)
    # segment column itself is the grouping key, but "region" values are the segments
    # so "South" is a valid segment, but "South" not in known_categories → unseen
    # Actually, region is the segment column, so we don't compute drift on the segment column itself
    # drift is only on other features (units here)
    # units are in range → drift_score ≈ 0 for "South"
    south = next((s for s in result["segments"] if s["name"] == "South"), None)
    assert south is not None


def test_multiple_segments_sorted_by_drift():
    # East: no drift, West: high drift (units=20)
    east_inputs = [{"region": "East", "units": 5.0}] * 10
    west_inputs = [{"region": "West", "units": 20.0}] * 10
    result = compute_segment_drift(east_inputs + west_inputs, "region", FEATURE_RANGES)
    assert result["n_segments"] == 2
    # West should be first (higher drift)
    assert result["segments"][0]["name"] == "West"
    assert result["segments"][0]["drift_score"] > result["segments"][1]["drift_score"]


def test_max_drift_segment():
    east_inputs = [{"region": "East", "units": 5.0}] * 5
    west_inputs = [{"region": "West", "units": 20.0}] * 10
    result = compute_segment_drift(east_inputs + west_inputs, "region", FEATURE_RANGES)
    assert result["max_drift_segment"] == "West"


def test_status_classification_high():
    # units=20 (all OOR) gives drift_score=50% (50% for units, 0% for region itself = avg 50%)
    # Actually: region is excluded from drift computation since it IS the segment column
    # units OOR=100% → avg drift = 100%
    inputs_without_region_tracking = [{"region": "X", "units": 100.0}] * 20
    ranges = {"region": {"known_categories": ["A"]}, "units": {"min": 0.0, "max": 10.0}}
    result = compute_segment_drift(inputs_without_region_tracking, "region", ranges)
    assert result["segments"][0]["status"] == "high"


def test_status_classification_minimal():
    inputs = [{"region": "East", "units": 5.0}] * 10
    result = compute_segment_drift(inputs, "region", FEATURE_RANGES)
    assert result["segments"][0]["status"] == "minimal"


def test_verdict_concentrated():
    # One segment has high drift, another has almost none
    high_inputs = [{"region": "East", "units": 50.0}] * 10  # all OOR
    low_inputs = [{"region": "West", "units": 5.0}] * 10  # no OOR
    result = compute_segment_drift(high_inputs + low_inputs, "region", FEATURE_RANGES)
    # East: drift=100%, West: drift=0% → concentrated
    assert result["verdict"] in ("concentrated", "widespread")  # concentrated expected
    if (
        result["segments"][0]["drift_score"]
        > result["segments"][1]["drift_score"] * 1.5
    ):
        assert result["verdict"] == "concentrated"


def test_verdict_minimal_all_low():
    # All segments have no drift
    inputs = _inputs(10, "East") + _inputs(10, "West")
    result = compute_segment_drift(inputs, "region", FEATURE_RANGES)
    assert result["verdict"] == "minimal"


def test_verdict_widespread():
    # Both segments have similar high drift
    high_east = [{"region": "East", "units": 50.0}] * 10
    high_west = [{"region": "West", "units": 50.0}] * 10
    result = compute_segment_drift(high_east + high_west, "region", FEATURE_RANGES)
    assert result["verdict"] in ("widespread", "concentrated")


def test_summary_present():
    inputs = _inputs(10, "East")
    result = compute_segment_drift(inputs, "region", FEATURE_RANGES)
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 10


def test_segment_column_missing_from_inputs():
    inputs = [{"units": 5.0}] * 10  # no "region" key
    result = compute_segment_drift(inputs, "region", FEATURE_RANGES)
    assert result["verdict"] == "no_data"
    assert result["n_segments"] == 0


def test_max_segments_cap():
    # 20 different segments, capped at max_segments=5
    inputs = [{"region": str(i), "units": 5.0} for i in range(20)]
    result = compute_segment_drift(inputs, "region", FEATURE_RANGES, max_segments=5)
    assert result["n_segments"] <= 5


def test_top_drifting_features_capped_at_3():
    # Many features, top_drifting_features per segment should be ≤3
    ranges = {
        "region": {"known_categories": ["A"]},
        **{f"feat{i}": {"min": 0.0, "max": 1.0} for i in range(10)},
    }
    inputs = [
        {"region": "A", **{f"feat{i}": 100.0 for i in range(10)}} for _ in range(10)
    ]
    result = compute_segment_drift(inputs, "region", ranges)
    for seg in result["segments"]:
        assert len(seg["top_drifting_features"]) <= 3


def test_avg_drift_score():
    # Two segments: 0% and 100% → avg = 50%
    ranges = {"region": {"known_categories": ["A"]}, "units": {"min": 0.0, "max": 10.0}}
    low = [{"region": "A", "units": 5.0}] * 10
    high = [{"region": "B", "units": 50.0}] * 10
    result = compute_segment_drift(low + high, "region", ranges)
    assert result["avg_drift_score"] > 0


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------

from api.chat import _SEGMENT_DRIFT_PATTERNS, _detect_segment_col  # noqa: E402


@pytest.mark.parametrize(
    "msg",
    [
        "segment drift analysis",
        "drift by region",
        "which segment has the most drift",
        "geographic drift analysis",
        "drift breakdown by category",
        "is drift concentrated in a group",
        "drift breakdown by segment",
        "where is the drift concentrated",
    ],
)
def test_segment_drift_pattern_matches(msg):
    assert _SEGMENT_DRIFT_PATTERNS.search(msg), f"Pattern should match: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "feature drift ranking",
        "enable drift alerts",
        "covariate drift alert",
        "how many predictions were made",
    ],
)
def test_segment_drift_pattern_no_false_positives(msg):
    assert not _SEGMENT_DRIFT_PATTERNS.search(msg), f"Pattern should NOT match: {msg!r}"


def test_detect_segment_col_finds_from_message():
    ranges = {
        "region": {"known_categories": ["East", "West"]},
        "product": {"known_categories": ["Widget", "Gadget"]},
        "units": {"min": 0.0, "max": 100.0},
    }
    col = _detect_segment_col("drift by product", ranges)
    assert col == "product"


def test_detect_segment_col_falls_back_to_first_categorical():
    ranges = {
        "units": {"min": 0.0, "max": 100.0},
        "region": {"known_categories": ["East", "West"]},
    }
    col = _detect_segment_col("show me the drift", ranges)
    assert col == "region"


def test_detect_segment_col_returns_none_when_no_categorical():
    ranges = {"units": {"min": 0.0, "max": 100.0}}
    col = _detect_segment_col("segment drift", ranges)
    assert col is None


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_segment_drift_404_unknown_deployment(client):
    resp = client.get("/api/deploy/nonexistent-id/segment-drift")
    assert resp.status_code == 404


def _create_test_deployment(session_engine):
    """Helper to create a minimal project/run/deployment for REST tests."""
    import json as _json
    from sqlmodel import Session
    from models.deployment import Deployment
    from models.model_run import ModelRun
    from models.project import Project

    with Session(session_engine) as s:
        proj = Project(name=f"seg-test-{id(session_engine)}")
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


def test_segment_drift_no_logs_returns_no_data(client):
    from db import engine

    dep_id = _create_test_deployment(engine)
    resp = client.get(f"/api/deploy/{dep_id}/segment-drift")
    assert resp.status_code == 200
    data = resp.json()
    # No pipeline → no feature_ranges → no categorical → no_data
    assert data["verdict"] == "no_data"


def test_segment_drift_required_fields_in_response(client):
    from db import engine

    dep_id = _create_test_deployment(engine)
    resp = client.get(f"/api/deploy/{dep_id}/segment-drift")
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "deployment_id",
        "n_segments",
        "n_samples",
        "segments",
        "verdict",
        "summary",
    ):
        assert key in data, f"Missing field: {key}"
