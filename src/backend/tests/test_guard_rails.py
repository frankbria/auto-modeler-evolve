"""Tests for prediction input guard rails: validate_prediction_inputs() + API integration."""

import io
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

SAMPLE_CSV = b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-01,Widget B,South,850.00,8
2024-01-02,Widget A,East,2100.75,18
2024-01-02,Widget C,West,450.25,4
2024-01-03,Widget B,North,1650.00,15
2024-01-04,Widget A,South,980.00,9
2024-01-04,Widget C,North,1100.25,11
2024-01-05,Widget B,East,1750.00,16
2024-01-05,Widget A,West,2300.50,20
2024-01-06,Widget C,South,620.75,6
2024-01-07,Widget A,North,1400.00,12
2024-01-08,Widget B,East,1900.00,17
2024-01-09,Widget C,West,800.00,7
2024-01-10,Widget A,South,1050.00,10
2024-01-11,Widget B,North,1600.00,14
2024-01-12,Widget C,East,950.00,8
2024-01-13,Widget A,West,2200.00,19
2024-01-14,Widget B,South,1100.00,10
2024-01-15,Widget C,North,700.00,6
2024-01-16,Widget A,East,1800.00,16
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models  # noqa — registers all tables

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api

    models_api.MODELS_DIR = tmp_path / "models"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def deployed(client):
    """Full pipeline: project → upload → features → set_target → train → deploy."""
    proj = client.post("/api/projects", json={"name": "Guard Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    apply = client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert apply.status_code in (200, 201)
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue"},
    )

    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert train.status_code == 202
    run_id = train.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    deploy = client.post(f"/api/deploy/{run_id}")
    assert deploy.status_code == 201
    dep = deploy.json()
    return {"deployment_id": dep["id"], "project_id": project_id}


# ---------------------------------------------------------------------------
# Unit tests: validate_prediction_inputs
# ---------------------------------------------------------------------------


def _make_pipeline():
    """Build a minimal pipeline for unit testing validate_prediction_inputs."""
    import pandas as pd
    from core.deployer import build_prediction_pipeline

    df = pd.DataFrame(
        {
            "units": list(range(10, 21)),  # 10..20, p5≈10, p95≈20
            "region": ["North"] * 5 + ["South"] * 3 + ["East"] * 2 + ["West"] * 1,
            "revenue": [100.0 * i for i in range(1, 12)],
        }
    )
    pipeline = build_prediction_pipeline(
        df, ["units", "region"], "revenue", "regression"
    )
    return pipeline


def test_validate_no_warnings_for_in_range_numeric():
    from core.deployer import validate_prediction_inputs

    pipeline = _make_pipeline()
    warnings = validate_prediction_inputs({"units": 15}, pipeline)
    assert warnings == []


def test_validate_out_of_range_numeric():
    from core.deployer import validate_prediction_inputs

    pipeline = _make_pipeline()
    # p5 of 10..20 is ≈10.5; p95 is ≈19.5; 500 is way out of range
    warnings = validate_prediction_inputs({"units": 500}, pipeline)
    assert len(warnings) == 1
    w = warnings[0]
    assert w["feature"] == "units"
    assert w["severity"] in ("out_of_range", "extreme_outlier")
    assert "units" in w["message"]
    assert isinstance(w["expected_min"], float)
    assert isinstance(w["expected_max"], float)


def test_validate_extreme_outlier_below_min():
    from core.deployer import validate_prediction_inputs

    pipeline = _make_pipeline()
    # min is 10; -100 is below min → extreme_outlier
    warnings = validate_prediction_inputs({"units": -100}, pipeline)
    assert len(warnings) == 1
    assert warnings[0]["severity"] == "extreme_outlier"


def test_validate_unknown_category():
    from core.deployer import validate_prediction_inputs

    pipeline = _make_pipeline()
    warnings = validate_prediction_inputs({"region": "Mars"}, pipeline)
    assert len(warnings) == 1
    w = warnings[0]
    assert w["feature"] == "region"
    assert w["severity"] == "unknown_category"
    assert "Mars" in w["message"]
    assert "known_categories" in w


def test_validate_known_category_no_warning():
    from core.deployer import validate_prediction_inputs

    pipeline = _make_pipeline()
    warnings = validate_prediction_inputs({"region": "North"}, pipeline)
    assert warnings == []


def test_validate_multiple_warnings():
    from core.deployer import validate_prediction_inputs

    pipeline = _make_pipeline()
    warnings = validate_prediction_inputs(
        {"units": 9999, "region": "Atlantis"}, pipeline
    )
    assert len(warnings) == 2


def test_validate_empty_provided_features():
    from core.deployer import validate_prediction_inputs

    pipeline = _make_pipeline()
    warnings = validate_prediction_inputs({}, pipeline)
    assert warnings == []


def test_validate_none_value_skipped():
    from core.deployer import validate_prediction_inputs

    pipeline = _make_pipeline()
    # None values should be skipped gracefully
    warnings = validate_prediction_inputs({"units": None}, pipeline)
    assert warnings == []


def test_feature_ranges_stored_in_pipeline():
    """Pipeline built from deployer has feature_ranges populated."""
    pipeline = _make_pipeline()
    assert "units" in pipeline.feature_ranges
    ranges = pipeline.feature_ranges["units"]
    assert "p5" in ranges
    assert "p95" in ranges
    assert "min" in ranges
    assert "max" in ranges
    assert ranges["min"] <= ranges["p5"] <= ranges["p95"] <= ranges["max"]


def test_categorical_known_categories_stored():
    pipeline = _make_pipeline()
    assert "region" in pipeline.feature_ranges
    known = pipeline.feature_ranges["region"]["known_categories"]
    assert "North" in known
    assert "South" in known


def test_predict_single_returns_guard_rail_warnings():
    """predict_single with provided_features returns guard_rail_warnings when out of range."""
    import pandas as pd
    from core.deployer import build_prediction_pipeline, predict_single, save_pipeline
    import joblib
    from pathlib import Path
    import tempfile
    from sklearn.linear_model import LinearRegression

    df = pd.DataFrame(
        {
            "units": list(range(10, 21)),
            "revenue": [100.0 * i for i in range(1, 12)],
        }
    )
    pipeline = build_prediction_pipeline(df, ["units"], "revenue", "regression")

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline_path = Path(tmpdir) / "pipeline.joblib"
        model_path = Path(tmpdir) / "model.joblib"

        save_pipeline(pipeline, pipeline_path)

        X = df[["units"]].values.astype(float)
        y = df["revenue"].values
        model = LinearRegression().fit(X, y)
        joblib.dump(model, model_path)

        # In-range: no warnings
        result_ok = predict_single(
            str(pipeline_path),
            str(model_path),
            {"units": 15.0},
            provided_features={"units": 15.0},
        )
        assert "guard_rail_warnings" not in result_ok

        # Out-of-range: warnings
        result_warn = predict_single(
            str(pipeline_path),
            str(model_path),
            {"units": 9999.0},
            provided_features={"units": 9999.0},
        )
        assert "guard_rail_warnings" in result_warn
        assert len(result_warn["guard_rail_warnings"]) == 1


def test_predict_single_no_warnings_when_provided_features_none():
    """predict_single with provided_features=None never returns guard_rail_warnings."""
    import pandas as pd
    from core.deployer import build_prediction_pipeline, predict_single, save_pipeline
    import joblib
    from pathlib import Path
    import tempfile
    from sklearn.linear_model import LinearRegression

    df = pd.DataFrame(
        {
            "units": list(range(10, 21)),
            "revenue": [100.0 * i for i in range(1, 12)],
        }
    )
    pipeline = build_prediction_pipeline(df, ["units"], "revenue", "regression")

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline_path = Path(tmpdir) / "pipeline.joblib"
        model_path = Path(tmpdir) / "model.joblib"
        save_pipeline(pipeline, pipeline_path)
        X = df[["units"]].values.astype(float)
        y = df["revenue"].values
        model = LinearRegression().fit(X, y)
        joblib.dump(model, model_path)

        # No provided_features (default=None) → no validation → no warnings even for extreme value
        result = predict_single(
            str(pipeline_path),
            str(model_path),
            {"units": 99999.0},
        )
        assert "guard_rail_warnings" not in result


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


def test_predict_in_range_no_warnings(client, deployed):
    dep_id = deployed["deployment_id"]
    resp = client.post(f"/api/predict/{dep_id}", json={"units": 12, "region": "North"})
    assert resp.status_code == 200
    data = resp.json()
    # In-range values — guard_rail_warnings should be absent or empty
    warnings = data.get("guard_rail_warnings", [])
    assert warnings == []


def test_predict_extreme_units_returns_warnings(client, deployed):
    dep_id = deployed["deployment_id"]
    resp = client.post(
        f"/api/predict/{dep_id}", json={"units": 99999, "region": "North"}
    )
    assert resp.status_code == 200
    data = resp.json()
    warnings = data.get("guard_rail_warnings", [])
    assert len(warnings) >= 1
    assert any(w["feature"] == "units" for w in warnings)
    assert any(w["severity"] in ("out_of_range", "extreme_outlier") for w in warnings)


def test_predict_unknown_region_returns_warning(client, deployed):
    dep_id = deployed["deployment_id"]
    resp = client.post(
        f"/api/predict/{dep_id}", json={"units": 12, "region": "Atlantis"}
    )
    assert resp.status_code == 200
    data = resp.json()
    warnings = data.get("guard_rail_warnings", [])
    assert len(warnings) >= 1
    region_warning = next((w for w in warnings if w["feature"] == "region"), None)
    assert region_warning is not None
    assert region_warning["severity"] == "unknown_category"


def test_predict_warnings_include_message(client, deployed):
    dep_id = deployed["deployment_id"]
    resp = client.post(
        f"/api/predict/{dep_id}", json={"units": 99999, "region": "North"}
    )
    assert resp.status_code == 200
    warnings = resp.json().get("guard_rail_warnings", [])
    assert len(warnings) >= 1
    assert all("message" in w for w in warnings)
    assert all(isinstance(w["message"], str) for w in warnings)


def test_guard_rail_patterns_in_api_response_structure(client, deployed):
    """Warning dict has expected keys."""
    dep_id = deployed["deployment_id"]
    resp = client.post(f"/api/predict/{dep_id}", json={"units": 99999})
    assert resp.status_code == 200
    warnings = resp.json().get("guard_rail_warnings", [])
    if warnings:
        w = warnings[0]
        assert "feature" in w
        assert "provided_value" in w
        assert "severity" in w
        assert "message" in w
