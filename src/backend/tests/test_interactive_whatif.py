"""Tests for interactive what-if feature schema extension.

Verifies that GET /api/deploy/{id} returns feature_schema entries with
min, max, p5, p95 for numeric features and options for categoricals.
"""

import io
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

SAMPLE_CSV = b"""product,region,units,revenue
Widget A,North,10,1200.5
Widget B,South,8,850.0
Widget A,East,18,2100.75
Widget C,West,4,450.25
Widget B,North,15,1650.0
Widget A,South,9,980.0
Widget C,North,11,1100.25
Widget B,East,16,1750.0
Widget A,West,20,2300.5
Widget C,South,6,620.75
Widget A,North,12,1300.0
Widget B,South,9,950.0
Widget A,East,20,2200.0
Widget C,West,5,520.0
Widget B,North,16,1700.0
Widget A,South,10,1050.0
Widget C,North,12,1150.0
Widget B,East,17,1800.0
Widget A,West,21,2350.0
Widget C,South,7,670.0
"""


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.conversation  # noqa
    import models.model_run  # noqa
    import models.deployment  # noqa
    import models.prediction_log  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    from main import app

    with TestClient(app) as c:
        yield c


def _deploy(
    client, csv_data=SAMPLE_CSV, target="revenue", algorithm="linear_regression"
):
    """Set up upload → train → deploy. Returns deployment_id."""
    proj = client.post("/api/projects", json={"name": "WhatIf Test"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(csv_data), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": target})

    train_resp = client.post(
        f"/api/models/{project_id}/train", json={"algorithms": [algorithm]}
    )
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert run["status"] == "done"

    deploy_resp = client.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code == 201
    return deploy_resp.json()["id"]


class TestFeatureSchemaRanges:
    """feature_schema now includes min/max/p5/p95 for numeric features."""

    def test_numeric_features_have_range_fields(self, client):
        dep_id = _deploy(client)
        r = client.get(f"/api/deploy/{dep_id}")
        assert r.status_code == 200
        schema = r.json().get("feature_schema", [])
        assert schema, "feature_schema should be non-empty"

        numeric_entries = [e for e in schema if e.get("type") == "numeric"]
        assert numeric_entries, "should have at least one numeric feature"

        for entry in numeric_entries:
            assert "min" in entry, f"numeric feature {entry['name']} missing 'min'"
            assert "max" in entry, f"numeric feature {entry['name']} missing 'max'"
            assert "p5" in entry, f"numeric feature {entry['name']} missing 'p5'"
            assert "p95" in entry, f"numeric feature {entry['name']} missing 'p95'"

    def test_p5_le_mean_le_p95(self, client):
        """p5 ≤ mean ≤ p95 for well-behaved numeric data."""
        dep_id = _deploy(client)
        r = client.get(f"/api/deploy/{dep_id}")
        schema = r.json().get("feature_schema", [])

        for entry in schema:
            if entry.get("type") != "numeric":
                continue
            p5 = entry.get("p5")
            p95 = entry.get("p95")
            mean = entry.get("mean")
            if p5 is not None and p95 is not None:
                assert p5 <= p95, f"{entry['name']}: p5 > p95"
            if mean is not None and p5 is not None and p95 is not None:
                assert p5 <= mean <= p95, f"{entry['name']}: mean not in [p5, p95]"

    def test_min_le_p5_and_p95_le_max(self, client):
        """min ≤ p5 and p95 ≤ max."""
        dep_id = _deploy(client)
        r = client.get(f"/api/deploy/{dep_id}")
        schema = r.json().get("feature_schema", [])

        for entry in schema:
            if entry.get("type") != "numeric":
                continue
            mn = entry.get("min")
            mx = entry.get("max")
            p5 = entry.get("p5")
            p95 = entry.get("p95")
            if mn is not None and p5 is not None:
                assert mn <= p5 + 1e-9, f"{entry['name']}: min > p5"
            if mx is not None and p95 is not None:
                assert p95 <= mx + 1e-9, f"{entry['name']}: p95 > max"

    def test_categorical_features_have_options_not_ranges(self, client):
        """Categorical features still have options, not min/max/p5/p95."""
        dep_id = _deploy(client)
        r = client.get(f"/api/deploy/{dep_id}")
        schema = r.json().get("feature_schema", [])

        cat_entries = [e for e in schema if e.get("type") == "categorical"]
        assert cat_entries, "should have at least one categorical feature"

        for entry in cat_entries:
            assert "options" in entry, f"categorical {entry['name']} missing 'options'"
            assert "min" not in entry, (
                f"categorical {entry['name']} should not have 'min'"
            )
            assert "p5" not in entry, (
                f"categorical {entry['name']} should not have 'p5'"
            )

    def test_numeric_values_are_floats(self, client):
        """min/max/p5/p95 are numeric (float/int), not None."""
        dep_id = _deploy(client)
        r = client.get(f"/api/deploy/{dep_id}")
        schema = r.json().get("feature_schema", [])

        for entry in schema:
            if entry.get("type") != "numeric":
                continue
            for field in ("min", "max", "p5", "p95"):
                val = entry.get(field)
                assert isinstance(val, (int, float)), (
                    f"{entry['name']}.{field} should be numeric, got {type(val)}"
                )

    def test_schema_preserved_alongside_new_fields(self, client):
        """mean, median, std still present after adding range fields."""
        dep_id = _deploy(client)
        r = client.get(f"/api/deploy/{dep_id}")
        schema = r.json().get("feature_schema", [])

        for entry in schema:
            if entry.get("type") != "numeric":
                continue
            assert "mean" in entry, f"{entry['name']} missing mean"
            assert "median" in entry, f"{entry['name']} missing median"

    def test_units_feature_range_includes_training_values(self, client):
        """For the 'units' feature, min should be ≤ 4 and max ≥ 21 (from sample CSV)."""
        dep_id = _deploy(client)
        r = client.get(f"/api/deploy/{dep_id}")
        schema = r.json().get("feature_schema", [])

        units_entry = next((e for e in schema if e["name"] == "units"), None)
        assert units_entry, "units feature should be in schema"
        assert units_entry["min"] <= 4 + 1e-9
        assert units_entry["max"] >= 20 - 1e-9
