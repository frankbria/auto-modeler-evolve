"""Tests for Canary Deployment support.

Covers:
- _CANARY_PATTERNS regex detection
- _CANARY_PROMOTE_RE / _CANARY_CANCEL_RE sub-patterns
- _CANARY_TRAFFIC_RE / _CANARY_VERSION_RE extractors
- compute_canary_comparison() pure function
- REST endpoints: start, cancel, promote, status
"""

import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_pattern_canary_deployment():
    from api.chat import _CANARY_PATTERNS

    assert _CANARY_PATTERNS.search("canary deployment")


def test_pattern_start_canary():
    from api.chat import _CANARY_PATTERNS

    assert _CANARY_PATTERNS.search("start a canary")


def test_pattern_enable_canary():
    from api.chat import _CANARY_PATTERNS

    assert _CANARY_PATTERNS.search("enable canary")


def test_pattern_route_traffic_to_new_model():
    from api.chat import _CANARY_PATTERNS

    assert _CANARY_PATTERNS.search("route 10% of traffic to new model")


def test_pattern_canary_status():
    from api.chat import _CANARY_PATTERNS

    assert _CANARY_PATTERNS.search("canary status")


def test_pattern_how_is_canary_doing():
    from api.chat import _CANARY_PATTERNS

    assert _CANARY_PATTERNS.search("how is my canary doing?")


def test_pattern_promote_canary():
    from api.chat import _CANARY_PATTERNS

    assert _CANARY_PATTERNS.search("promote the canary to production")


def test_pattern_cancel_canary():
    from api.chat import _CANARY_PATTERNS

    assert _CANARY_PATTERNS.search("cancel canary")


def test_pattern_canary_vs_production():
    from api.chat import _CANARY_PATTERNS

    assert _CANARY_PATTERNS.search("canary vs production")


def test_pattern_traffic_split():
    from api.chat import _CANARY_PATTERNS

    assert _CANARY_PATTERNS.search("traffic splitting for new model")


def test_pattern_negative_unrelated():
    from api.chat import _CANARY_PATTERNS

    assert not _CANARY_PATTERNS.search("what is the average revenue?")
    assert not _CANARY_PATTERNS.search("train a new model")


# ---------------------------------------------------------------------------
# Sub-pattern tests
# ---------------------------------------------------------------------------


def test_promote_re():
    from api.chat import _CANARY_PROMOTE_RE

    assert _CANARY_PROMOTE_RE.search("promote the canary")
    assert not _CANARY_PROMOTE_RE.search("cancel the canary")


def test_cancel_re():
    from api.chat import _CANARY_CANCEL_RE

    assert _CANARY_CANCEL_RE.search("cancel canary")
    assert _CANARY_CANCEL_RE.search("stop the canary")
    assert not _CANARY_CANCEL_RE.search("promote the canary")


def test_traffic_re():
    from api.chat import _CANARY_TRAFFIC_RE

    m = _CANARY_TRAFFIC_RE.search("route 10% of traffic to v2")
    assert m and int(m.group(1)) == 10


def test_version_re():
    from api.chat import _CANARY_VERSION_RE

    m = _CANARY_VERSION_RE.search("start canary with v1 at 20% traffic")
    assert m and int(m.group(1)) == 1


# ---------------------------------------------------------------------------
# compute_canary_comparison — pure function
# ---------------------------------------------------------------------------


def _make_logs(n_control: int, n_canary: int, control_val: float, canary_val: float):
    logs = []
    for _ in range(n_control):
        logs.append(
            {
                "served_by_canary": False,
                "prediction_numeric": control_val,
                "confidence": 0.80,
            }
        )
    for _ in range(n_canary):
        logs.append(
            {
                "served_by_canary": True,
                "prediction_numeric": canary_val,
                "confidence": 0.85,
            }
        )
    return logs


def test_canary_comparison_insufficient_data():
    from core.deployer import compute_canary_comparison

    logs = _make_logs(5, 5, 100.0, 110.0)
    result = compute_canary_comparison(logs, "run_a", "run_b", 10)
    assert result["verdict"] == "insufficient_data"
    assert not result["has_enough_data"]


def test_canary_comparison_no_significant_difference():
    from core.deployer import compute_canary_comparison

    logs = _make_logs(20, 20, 100.0, 101.0)  # 1% difference
    result = compute_canary_comparison(logs, "run_a", "run_b", 10)
    assert result["verdict"] == "no_significant_difference"
    assert result["has_enough_data"]


def test_canary_comparison_regression_higher():
    from core.deployer import compute_canary_comparison

    logs = _make_logs(20, 20, 100.0, 125.0)  # 25% higher
    result = compute_canary_comparison(logs, "run_a", "run_b", 10, "regression")
    assert result["direction"] == "canary_higher"
    assert result["delta"] > 0
    assert "canary" in result["summary"].lower()


def test_canary_comparison_regression_lower():
    from core.deployer import compute_canary_comparison

    logs = _make_logs(20, 20, 100.0, 70.0)  # 30% lower
    result = compute_canary_comparison(logs, "run_a", "run_b", 10, "regression")
    assert result["direction"] == "canary_lower"
    assert result["delta"] < 0


def test_canary_comparison_classification_better():
    from core.deployer import compute_canary_comparison

    # For classification use confidence column
    logs = []
    for _ in range(20):
        logs.append(
            {"served_by_canary": False, "prediction_numeric": None, "confidence": 0.70}
        )
    for _ in range(20):
        logs.append(
            {"served_by_canary": True, "prediction_numeric": None, "confidence": 0.90}
        )
    result = compute_canary_comparison(logs, "run_a", "run_b", 10, "classification")
    assert result["verdict"] == "canary_better"
    assert result["metric_label"] == "Avg confidence"


def test_canary_comparison_counts():
    from core.deployer import compute_canary_comparison

    logs = _make_logs(15, 12, 100.0, 115.0)
    result = compute_canary_comparison(logs, "run_a", "run_b", 10)
    assert result["control_count"] == 15
    assert result["canary_count"] == 12


def test_canary_comparison_empty_logs():
    from core.deployer import compute_canary_comparison

    result = compute_canary_comparison([], "run_a", "run_b", 10)
    assert result["verdict"] == "insufficient_data"
    assert result["control_count"] == 0
    assert result["canary_count"] == 0


def test_canary_comparison_returns_traffic_pct():
    from core.deployer import compute_canary_comparison

    logs = _make_logs(15, 15, 100.0, 110.0)
    result = compute_canary_comparison(logs, "run_a", "run_b", 20)
    assert result["traffic_pct"] == 20


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


def _make_engine(tmp_path):
    db_path = str(tmp_path / "test.db")
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    SQLModel.metadata.create_all(engine)
    return engine


def _make_deployment_with_versions(tmp_path, session):
    """Create two DeploymentVersions and a Deployment pointing to the second."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from api.deploy import ModelRun
    from core.deployer import build_prediction_pipeline, save_pipeline
    from models.deployment import Deployment
    from models.deployment_version import DeploymentVersion

    df = pd.DataFrame(
        {
            "units": np.arange(1, 21, dtype=float),
            "revenue": np.arange(1, 21, dtype=float) * 10.0,
        }
    )
    pipeline = build_prediction_pipeline(df, ["units"], "revenue", "regression")

    # Version 1 (older model)
    p1 = str(tmp_path / "pipeline_v1.joblib")
    m1 = str(tmp_path / "model_v1.joblib")
    save_pipeline(pipeline, p1)
    model_v1 = LinearRegression().fit(df[["units"]].values, df["revenue"].values)
    joblib.dump(model_v1, m1)
    run1 = ModelRun(
        project_id="proj-1",
        algorithm="linear_regression",
        status="completed",
        model_path=m1,
    )
    session.add(run1)
    session.flush()

    # Version 2 (current model)
    p2 = str(tmp_path / "pipeline_v2.joblib")
    m2 = str(tmp_path / "model_v2.joblib")
    save_pipeline(pipeline, p2)
    model_v2 = LinearRegression().fit(df[["units"]].values, df["revenue"].values)
    joblib.dump(model_v2, m2)
    run2 = ModelRun(
        project_id="proj-1",
        algorithm="random_forest",
        status="completed",
        model_path=m2,
    )
    session.add(run2)
    session.flush()

    dep = Deployment(
        model_run_id=run2.id,
        project_id="proj-1",
        endpoint_path="/api/predict/test",
        dashboard_url="/predict/test",
        pipeline_path=p2,
        algorithm="random_forest",
        problem_type="regression",
        target_column="revenue",
        feature_names=json.dumps(["units"]),
        current_version_number=2,
    )
    session.add(dep)
    session.flush()

    dv1 = DeploymentVersion(
        deployment_id=dep.id,
        version_number=1,
        model_run_id=run1.id,
        algorithm="linear_regression",
        problem_type="regression",
        target_column="revenue",
        pipeline_path=p1,
        is_current=False,
    )
    dv2 = DeploymentVersion(
        deployment_id=dep.id,
        version_number=2,
        model_run_id=run2.id,
        algorithm="random_forest",
        problem_type="regression",
        target_column="revenue",
        pipeline_path=p2,
        is_current=True,
    )
    session.add(dv1)
    session.add(dv2)
    session.commit()

    return dep, run1, run2, dv1, dv2


def test_start_canary_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from main import app

    engine = _make_engine(tmp_path)
    with Session(engine) as session:
        dep, run1, run2, dv1, dv2 = _make_deployment_with_versions(tmp_path, session)
        dep_id = dep.id

    def override_session():
        with Session(engine) as s:
            yield s

    from db import get_session

    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    resp = client.post(
        f"/api/deploy/{dep_id}/canary/start",
        json={"version_number": 1, "traffic_pct": 10},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["canary_is_active"] is True
    assert data["canary_traffic_pct"] == 10

    app.dependency_overrides.clear()


def test_start_canary_invalid_traffic_pct(tmp_path):
    from fastapi.testclient import TestClient
    from main import app

    engine = _make_engine(tmp_path)
    with Session(engine) as session:
        dep, *_ = _make_deployment_with_versions(tmp_path, session)
        dep_id = dep.id

    def override_session():
        with Session(engine) as s:
            yield s

    from db import get_session

    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    resp = client.post(
        f"/api/deploy/{dep_id}/canary/start",
        json={"version_number": 1, "traffic_pct": 75},
    )
    assert resp.status_code == 400
    app.dependency_overrides.clear()


def test_canary_status_inactive(tmp_path):
    from fastapi.testclient import TestClient
    from main import app

    engine = _make_engine(tmp_path)
    with Session(engine) as session:
        dep, *_ = _make_deployment_with_versions(tmp_path, session)
        dep_id = dep.id

    def override_session():
        with Session(engine) as s:
            yield s

    from db import get_session

    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    resp = client.get(f"/api/deploy/{dep_id}/canary/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["canary_is_active"] is False
    assert len(data["available_versions"]) == 2

    app.dependency_overrides.clear()


def test_cancel_canary_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from main import app

    engine = _make_engine(tmp_path)
    with Session(engine) as session:
        dep, run1, run2, dv1, dv2 = _make_deployment_with_versions(tmp_path, session)
        dep_id = dep.id
        # Manually enable canary
        dep.canary_is_active = True
        dep.canary_run_id = run1.id
        dep.canary_pipeline_path = dv1.pipeline_path
        dep.canary_traffic_pct = 10
        session.add(dep)
        session.commit()

    def override_session():
        with Session(engine) as s:
            yield s

    from db import get_session

    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    resp = client.post(f"/api/deploy/{dep_id}/canary/cancel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["canary_is_active"] is False

    app.dependency_overrides.clear()


def test_promote_canary_endpoint(tmp_path):
    from fastapi.testclient import TestClient
    from main import app

    engine = _make_engine(tmp_path)
    with Session(engine) as session:
        dep, run1, run2, dv1, dv2 = _make_deployment_with_versions(tmp_path, session)
        dep_id = dep.id
        dep.canary_is_active = True
        dep.canary_run_id = run1.id
        dep.canary_pipeline_path = dv1.pipeline_path
        dep.canary_traffic_pct = 10
        session.add(dep)
        session.commit()

    def override_session():
        with Session(engine) as s:
            yield s

    from db import get_session

    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    resp = client.post(f"/api/deploy/{dep_id}/canary/promote")
    assert resp.status_code == 200
    data = resp.json()
    assert data["promoted"] is True
    assert data["new_version_number"] == 3  # was v2, now v3

    app.dependency_overrides.clear()


def test_promote_canary_no_active(tmp_path):
    from fastapi.testclient import TestClient
    from main import app

    engine = _make_engine(tmp_path)
    with Session(engine) as session:
        dep, *_ = _make_deployment_with_versions(tmp_path, session)
        dep_id = dep.id

    def override_session():
        with Session(engine) as s:
            yield s

    from db import get_session

    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    resp = client.post(f"/api/deploy/{dep_id}/canary/promote")
    assert resp.status_code == 422

    app.dependency_overrides.clear()


def test_canary_status_with_active_canary(tmp_path):
    from fastapi.testclient import TestClient
    from main import app

    engine = _make_engine(tmp_path)
    with Session(engine) as session:
        dep, run1, run2, dv1, dv2 = _make_deployment_with_versions(tmp_path, session)
        dep_id = dep.id
        dep.canary_is_active = True
        dep.canary_run_id = run1.id
        dep.canary_pipeline_path = dv1.pipeline_path
        dep.canary_traffic_pct = 15
        session.add(dep)
        session.commit()

    def override_session():
        with Session(engine) as s:
            yield s

    from db import get_session

    app.dependency_overrides[get_session] = override_session
    client = TestClient(app)

    resp = client.get(f"/api/deploy/{dep_id}/canary/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["canary_is_active"] is True
    assert data["canary_traffic_pct"] == 15
    assert data["comparison"] is not None

    app.dependency_overrides.clear()
