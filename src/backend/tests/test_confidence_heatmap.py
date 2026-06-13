"""Tests for compute_confidence_heatmap pure function, regex patterns, and REST endpoint."""

from __future__ import annotations

import pytest

from core.analyzer import compute_confidence_heatmap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cls_log(
    vx, vy, confidence: float, feat_x: str = "age", feat_y: str = "region"
) -> dict:
    return {
        "prediction_numeric": None,
        "confidence": confidence,
        "input_features_dict": {feat_x: vx, feat_y: vy},
    }


def _reg_log(vx, vy, pred: float, feat_x: str = "age", feat_y: str = "region") -> dict:
    return {
        "prediction_numeric": pred,
        "confidence": None,
        "input_features_dict": {feat_x: vx, feat_y: vy},
    }


def _make_cls_logs(n: int = 30) -> list[dict]:
    logs = []
    for i in range(n):
        vx = i % 5  # 5 numeric bins
        vy = "East" if i % 2 == 0 else "West"
        conf = 0.5 + (i % 5) * 0.1
        logs.append(_cls_log(vx, vy, conf))
    return logs


# ---------------------------------------------------------------------------
# Pure-function: structure and edge cases
# ---------------------------------------------------------------------------


def test_empty_logs_returns_insufficient():
    result = compute_confidence_heatmap([], "age", "region")
    assert result["verdict"] == "insufficient_data"
    assert result["cells"] == []


def test_fewer_than_5_returns_insufficient():
    logs = [_cls_log(1, "East", 0.8)] * 4
    result = compute_confidence_heatmap(logs, "age", "region")
    assert result["verdict"] == "insufficient_data"
    assert "Not enough" in result["summary"]


def test_required_keys_present():
    logs = _make_cls_logs(30)
    result = compute_confidence_heatmap(logs, "age", "region")
    for key in (
        "feature_x",
        "feature_y",
        "x_labels",
        "y_labels",
        "cells",
        "low_confidence_zones",
        "overall_min",
        "overall_max",
        "overall_mean",
        "n_samples",
        "n_cells_total",
        "n_cells_populated",
        "verdict",
        "summary",
    ):
        assert key in result, f"Missing key: {key}"


def test_classification_verdict_gaps_found():
    logs = []
    # Low confidence for vx=0, vy=West
    for _ in range(10):
        logs.append(_cls_log(0, "West", 0.45))
    # High confidence everywhere else
    for _ in range(10):
        logs.append(_cls_log(1, "East", 0.92))
    for _ in range(10):
        logs.append(_cls_log(2, "East", 0.88))
    result = compute_confidence_heatmap(logs, "age", "region", n_bins=3)
    assert result["verdict"] == "gaps_found"
    assert len(result["low_confidence_zones"]) >= 1


def test_classification_verdict_uniform_high():
    logs = []
    for vx in range(3):
        for vy in ["East", "West"]:
            for _ in range(5):
                logs.append(_cls_log(vx, vy, 0.92))
    result = compute_confidence_heatmap(logs, "age", "region", n_bins=3)
    assert result["verdict"] == "uniform_high"
    assert result["low_confidence_zones"] == []


def test_low_confidence_threshold_is_065():
    logs = []
    for _ in range(10):
        logs.append(_cls_log("A", "X", 0.64))  # just below threshold
    for _ in range(10):
        logs.append(_cls_log("B", "Y", 0.92))
    result = compute_confidence_heatmap(logs, "age", "region")
    low = [z for z in result["low_confidence_zones"] if z["x_label"] == "A"]
    assert len(low) >= 1
    assert low[0]["avg_confidence"] < 0.65


def test_low_confidence_zones_sorted_ascending():
    logs = []
    for _ in range(10):
        logs.append(_cls_log("A", "X", 0.40))
    for _ in range(10):
        logs.append(_cls_log("B", "Y", 0.55))
    for _ in range(10):
        logs.append(_cls_log("C", "Z", 0.92))
    result = compute_confidence_heatmap(logs, "age", "region")
    zones = result["low_confidence_zones"]
    if len(zones) >= 2:
        for i in range(len(zones) - 1):
            assert zones[i]["avg_confidence"] <= zones[i + 1]["avg_confidence"]


def test_cell_is_low_confidence_flag():
    logs = []
    for _ in range(10):
        logs.append(_cls_log("A", "X", 0.45))
    for _ in range(10):
        logs.append(_cls_log("B", "Y", 0.90))
    result = compute_confidence_heatmap(logs, "age", "region")
    low_cells = [c for c in result["cells"] if c["is_low_confidence"]]
    ok_cells = [c for c in result["cells"] if not c["is_low_confidence"]]
    assert all(c["avg_confidence"] < 0.65 for c in low_cells)
    assert all(c["avg_confidence"] >= 0.65 for c in ok_cells)


def test_cell_n_samples_counts():
    logs = [_cls_log("A", "X", 0.8)] * 12
    result = compute_confidence_heatmap(logs, "age", "region")
    cells = [c for c in result["cells"] if c["x_label"] == "A" and c["y_label"] == "X"]
    assert len(cells) == 1
    assert cells[0]["n_samples"] == 12


def test_min_samples_per_cell_excludes_sparse():
    logs = []
    for _ in range(10):
        logs.append(_cls_log("A", "X", 0.9))
    # Only 2 samples for B/Y — below min_samples_per_cell=3
    for _ in range(2):
        logs.append(_cls_log("B", "Y", 0.3))
    result = compute_confidence_heatmap(logs, "age", "region", min_samples_per_cell=3)
    cell_labels = {(c["x_label"], c["y_label"]) for c in result["cells"]}
    assert ("B", "Y") not in cell_labels


def test_numeric_x_creates_bins():
    logs = [_cls_log(i, "East", 0.8) for i in range(20)]
    result = compute_confidence_heatmap(logs, "age", "region", n_bins=4)
    assert len(result["x_labels"]) == 4
    for label in result["x_labels"]:
        assert "–" in label  # numeric bin label format


def test_categorical_y_keeps_values():
    logs = []
    for vy in ["North", "South", "East"]:
        for _ in range(5):
            logs.append(_cls_log(1, vy, 0.8))
    result = compute_confidence_heatmap(logs, "age", "region")
    assert set(result["y_labels"]) == {"North", "South", "East"}


def test_regression_uses_cv_metric():
    logs = []
    # Low spread (consistent) → high certainty
    for _ in range(10):
        logs.append(_reg_log("A", "X", 100.0))
    for _ in range(10):
        logs.append(_reg_log("A", "X", 101.0))
    # High spread → lower certainty
    for _ in range(10):
        logs.append(_reg_log("B", "Y", 10.0))
    for _ in range(10):
        logs.append(_reg_log("B", "Y", 10000.0))
    result = compute_confidence_heatmap(
        logs, "age", "region", problem_type="regression"
    )
    consistent_cells = [c for c in result["cells"] if c["x_label"] == "A"]
    spread_cells = [c for c in result["cells"] if c["x_label"] == "B"]
    if consistent_cells and spread_cells:
        assert consistent_cells[0]["avg_confidence"] > spread_cells[0]["avg_confidence"]


def test_n_bins_clamped_to_range():
    logs = [_cls_log(i, "East", 0.8) for i in range(30)]
    result_min = compute_confidence_heatmap(logs, "age", "region", n_bins=1)
    assert len(result_min["x_labels"]) >= 2
    result_max = compute_confidence_heatmap(logs, "age", "region", n_bins=100)
    assert len(result_max["x_labels"]) <= 6


def test_feature_x_y_in_result():
    logs = _make_cls_logs(30)
    result = compute_confidence_heatmap(logs, "age", "region")
    assert result["feature_x"] == "age"
    assert result["feature_y"] == "region"


def test_n_samples_counts_valid_rows():
    logs = _make_cls_logs(30)
    # Add a row with missing feature_y — should be excluded
    logs.append(
        {
            "confidence": 0.8,
            "prediction_numeric": None,
            "input_features_dict": {"age": 1},
        }
    )
    result = compute_confidence_heatmap(logs, "age", "region")
    assert result["n_samples"] == 30


def test_overall_stats_present():
    logs = _make_cls_logs(30)
    result = compute_confidence_heatmap(logs, "age", "region", n_bins=3)
    assert result["overall_min"] is not None
    assert result["overall_max"] is not None
    assert result["overall_mean"] is not None
    assert result["overall_min"] <= result["overall_mean"] <= result["overall_max"]


def test_cells_contain_idx():
    logs = _make_cls_logs(30)
    result = compute_confidence_heatmap(logs, "age", "region", n_bins=3)
    for cell in result["cells"]:
        assert "x_idx" in cell
        assert "y_idx" in cell
        assert 0 <= cell["x_idx"] < len(result["x_labels"])
        assert 0 <= cell["y_idx"] < len(result["y_labels"])


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

from api.chat import _CONF_HEATMAP_PATTERNS  # noqa: E402


@pytest.mark.parametrize(
    "msg",
    [
        "show me a confidence heatmap",
        "which input combinations make my model least confident?",
        "feature confidence grid for my deployment",
        "show me low confidence zones",
        "confidence by age and income",
        "model uncertainty heatmap by two features",
        "which combinations of values confuse my model?",
        "where is my model uncertain for age and region?",
    ],
)
def test_conf_heatmap_pattern_matches(msg: str):
    assert _CONF_HEATMAP_PATTERNS.search(msg), f"Pattern did not match: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "confidence trend by segment",
        "show overall confidence distribution",
        "what's the model accuracy?",
        "segment drift analysis",
    ],
)
def test_conf_heatmap_pattern_no_false_positives(msg: str):
    assert not _CONF_HEATMAP_PATTERNS.search(msg), f"False positive: {msg!r}"


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
        proj = Project(name=f"ch-test-{id(engine)}")
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
    resp = client.get("/api/deploy/nonexistent-id/confidence-heatmap")
    assert resp.status_code == 404


def test_endpoint_returns_no_data_without_logs(client):
    from db import engine

    dep_id = _create_deployment(engine)
    resp = client.get(
        f"/api/deploy/{dep_id}/confidence-heatmap?feature_x=age&feature_y=region"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "verdict" in data
    assert data["verdict"] in (
        "insufficient_data",
        "no_data",
        "gaps_found",
        "uniform_high",
        "uniform_moderate",
    )


def test_endpoint_returns_deployment_id(client):
    from db import engine

    dep_id = _create_deployment(engine)
    resp = client.get(
        f"/api/deploy/{dep_id}/confidence-heatmap?feature_x=age&feature_y=region"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("deployment_id") == dep_id


def test_endpoint_n_bins_query(client):
    from db import engine

    dep_id = _create_deployment(engine)
    resp = client.get(
        f"/api/deploy/{dep_id}/confidence-heatmap?feature_x=age&feature_y=region&n_bins=3"
    )
    assert resp.status_code == 200


def test_endpoint_required_fields_in_response(client):
    from db import engine

    dep_id = _create_deployment(engine)
    resp = client.get(
        f"/api/deploy/{dep_id}/confidence-heatmap?feature_x=age&feature_y=region"
    )
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "deployment_id",
        "feature_x",
        "feature_y",
        "x_labels",
        "y_labels",
        "cells",
        "low_confidence_zones",
        "n_samples",
        "verdict",
        "summary",
    ):
        assert key in data, f"Missing key: {key}"
