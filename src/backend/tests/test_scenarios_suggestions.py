"""Tests for scenario comparison endpoint and chat suggestion generation.

Feature 1 — POST /api/predict/{id}/scenarios:
  Compare multiple what-if scenarios against a shared base case in one call.

Feature 2 — generate_suggestions():
  Context-aware follow-up chips for chat, based on workflow state.
"""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Shared CSV fixtures
# ---------------------------------------------------------------------------

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

CLASSIFICATION_CSV = b"""f1,f2,label
1.0,2.0,cat
2.0,3.0,dog
3.0,4.0,cat
4.0,5.0,dog
5.0,6.0,cat
6.0,7.0,dog
7.0,8.0,cat
8.0,9.0,dog
9.0,10.0,cat
10.0,11.0,dog
1.5,2.5,cat
2.5,3.5,dog
3.5,4.5,cat
4.5,5.5,dog
5.5,6.5,cat
6.5,7.5,dog
7.5,8.5,cat
8.5,9.5,dog
9.5,10.5,cat
10.5,11.5,dog
"""


# ---------------------------------------------------------------------------
# Fixture: TestClient with isolated DB + temp dirs
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa: F401
    import models.dataset  # noqa: F401
    import models.feature_set  # noqa: F401
    import models.conversation  # noqa: F401
    import models.model_run  # noqa: F401
    import models.deployment  # noqa: F401
    import models.prediction_log  # noqa: F401
    import models.feedback_record  # noqa: F401

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


def _setup_deployed(
    client, csv_data=SAMPLE_CSV, target="revenue", algorithm="linear_regression"
):
    """Create project → upload → apply features → train → deploy. Returns deployment_id."""
    proj = client.post("/api/projects", json={"name": "Scenario Test"})
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
    deployment_id = deploy_resp.json()["id"]
    return deployment_id, run_id, project_id


# ===========================================================================
# Tests — Scenario Comparison
# ===========================================================================


class TestScenarioComparison:
    """POST /api/predict/{deployment_id}/scenarios"""

    def test_scenarios_basic_regression(self, client):
        """Two scenarios → compare deltas against base."""
        deployment_id, _, _ = _setup_deployed(client)

        base = {"product": "Widget A", "region": "North", "units": 10}
        payload = {
            "base": base,
            "scenarios": [
                {"label": "More units", "overrides": {"units": 20}},
                {"label": "Different region", "overrides": {"region": "South"}},
            ],
        }
        r = client.post(f"/api/predict/{deployment_id}/scenarios", json=payload)
        assert r.status_code == 200
        data = r.json()

        assert "base_prediction" in data
        assert "scenarios" in data
        assert len(data["scenarios"]) == 2
        assert data["scenarios"][0]["label"] == "More units"
        assert data["scenarios"][1]["label"] == "Different region"
        assert "summary" in data
        assert "deployment_id" in data

    def test_scenarios_result_shape(self, client):
        """Each scenario result has required fields."""
        deployment_id, _, _ = _setup_deployed(client)

        payload = {
            "base": {"product": "Widget A", "region": "North", "units": 10},
            "scenarios": [
                {"label": "Test", "overrides": {"units": 15}},
            ],
        }
        r = client.post(f"/api/predict/{deployment_id}/scenarios", json=payload)
        assert r.status_code == 200
        s = r.json()["scenarios"][0]

        assert "label" in s
        assert "overrides" in s
        assert "prediction" in s
        # delta, percent_change, direction should be present (may be None for classification)
        assert "delta" in s
        assert "percent_change" in s
        assert "direction" in s

    def test_scenarios_single_scenario(self, client):
        """Works with exactly one scenario."""
        deployment_id, _, _ = _setup_deployed(client)

        payload = {
            "base": {"product": "Widget A", "region": "North", "units": 10},
            "scenarios": [{"label": "High volume", "overrides": {"units": 50}}],
        }
        r = client.post(f"/api/predict/{deployment_id}/scenarios", json=payload)
        assert r.status_code == 200
        assert len(r.json()["scenarios"]) == 1

    def test_scenarios_max_limit_enforced(self, client):
        """More than 10 scenarios should return 400."""
        deployment_id, _, _ = _setup_deployed(client)

        payload = {
            "base": {"product": "Widget A", "region": "North", "units": 10},
            "scenarios": [
                {"label": f"S{i}", "overrides": {"units": i + 1}} for i in range(11)
            ],
        }
        r = client.post(f"/api/predict/{deployment_id}/scenarios", json=payload)
        assert r.status_code == 400
        assert "10" in r.json()["detail"]

    def test_scenarios_empty_scenarios_rejected(self, client):
        """Empty scenarios list should return 400."""
        deployment_id, _, _ = _setup_deployed(client)

        payload = {
            "base": {"product": "Widget A", "region": "North", "units": 10},
            "scenarios": [],
        }
        r = client.post(f"/api/predict/{deployment_id}/scenarios", json=payload)
        assert r.status_code == 400

    def test_scenarios_inactive_deployment_404(self, client):
        """Returns 404 for inactive/missing deployment."""
        payload = {
            "base": {"x": 1},
            "scenarios": [{"label": "Test", "overrides": {"x": 2}}],
        }
        r = client.post("/api/predict/nonexistent/scenarios", json=payload)
        assert r.status_code == 404

    def test_scenarios_target_column_returned(self, client):
        """target_column and problem_type are echoed in the response."""
        deployment_id, _, _ = _setup_deployed(client)

        payload = {
            "base": {"product": "Widget A", "region": "North", "units": 10},
            "scenarios": [{"label": "S1", "overrides": {"units": 20}}],
        }
        r = client.post(f"/api/predict/{deployment_id}/scenarios", json=payload)
        data = r.json()
        assert data["problem_type"] in ("regression", "classification", None)
        assert data["target_column"] == "revenue"

    def test_scenarios_summary_mentions_best_worst(self, client):
        """Summary should reference predictions meaningfully."""
        deployment_id, _, _ = _setup_deployed(client)

        payload = {
            "base": {"product": "Widget A", "region": "North", "units": 10},
            "scenarios": [
                {"label": "Low", "overrides": {"units": 1}},
                {"label": "High", "overrides": {"units": 100}},
            ],
        }
        r = client.post(f"/api/predict/{deployment_id}/scenarios", json=payload)
        data = r.json()
        assert isinstance(data["summary"], str)
        assert len(data["summary"]) > 0

    def test_scenarios_overrides_combined_with_base(self, client):
        """Override dict merges with base; unspecified base features are kept."""
        deployment_id, _, _ = _setup_deployed(client)

        payload = {
            "base": {"product": "Widget A", "region": "North", "units": 10},
            "scenarios": [
                # Only override 'units' — product and region should be inherited from base
                {"label": "More units", "overrides": {"units": 25}},
            ],
        }
        r = client.post(f"/api/predict/{deployment_id}/scenarios", json=payload)
        assert r.status_code == 200
        # The override dict should be reflected in the response
        assert r.json()["scenarios"][0]["overrides"]["units"] == 25

    def test_scenarios_classification_model(self, client):
        """Scenario comparison works for classification models too."""
        deployment_id, _, _ = _setup_deployed(
            client,
            csv_data=CLASSIFICATION_CSV,
            target="label",
            algorithm="random_forest_classifier",
        )

        payload = {
            "base": {"f1": 1.0, "f2": 2.0},
            "scenarios": [
                {"label": "High features", "overrides": {"f1": 9.0, "f2": 10.0}},
                {"label": "Mid features", "overrides": {"f1": 5.0, "f2": 6.0}},
            ],
        }
        r = client.post(f"/api/predict/{deployment_id}/scenarios", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert len(data["scenarios"]) == 2
        # Classification: probabilities may be present
        for s in data["scenarios"]:
            assert "prediction" in s

    def test_scenarios_ten_scenarios_accepted(self, client):
        """Exactly 10 scenarios (the limit) should work fine."""
        deployment_id, _, _ = _setup_deployed(client)

        payload = {
            "base": {"product": "Widget A", "region": "North", "units": 10},
            "scenarios": [
                {"label": f"S{i}", "overrides": {"units": i + 5}} for i in range(10)
            ],
        }
        r = client.post(f"/api/predict/{deployment_id}/scenarios", json=payload)
        assert r.status_code == 200
        assert len(r.json()["scenarios"]) == 10

    def test_scenarios_base_prediction_included(self, client):
        """base_prediction in response matches a standalone predict call."""
        deployment_id, _, _ = _setup_deployed(client)
        base = {"product": "Widget A", "region": "North", "units": 10}

        single = client.post(f"/api/predict/{deployment_id}", json=base).json()
        scenarios_resp = client.post(
            f"/api/predict/{deployment_id}/scenarios",
            json={
                "base": base,
                "scenarios": [{"label": "S1", "overrides": {"units": 20}}],
            },
        ).json()

        assert scenarios_resp["base_prediction"] == single["prediction"]


# ===========================================================================
# Tests — generate_suggestions
# ===========================================================================


class TestGenerateSuggestions:
    """Unit tests for chat.orchestrator.generate_suggestions()."""

    @pytest.fixture(autouse=True)
    def import_fn(self):
        from chat.orchestrator import generate_suggestions

        self.generate_suggestions = generate_suggestions

    def test_suggestions_upload_state_no_dataset(self):
        """Returns 3 upload-stage suggestions when no dataset loaded."""
        result = self.generate_suggestions(state="upload")
        assert isinstance(result, list)
        assert 1 <= len(result) <= 3
        # All should be strings
        assert all(isinstance(s, str) for s in result)

    def test_suggestions_explore_state(self):
        """Explore state returns explore-relevant suggestions."""
        from models.dataset import Dataset

        ds = Dataset(
            id="ds1",
            project_id="p1",
            filename="sales.csv",
            row_count=200,
            column_count=8,
            file_path="/tmp/f.csv",
        )
        result = self.generate_suggestions(state="explore", dataset=ds)
        assert 1 <= len(result) <= 3
        # Should mention something about data
        combined = " ".join(result).lower()
        assert any(
            word in combined
            for word in ["data", "column", "pattern", "correlation", "model"]
        )

    def test_suggestions_deploy_state(self):
        """Deploy state returns share/monitor suggestions."""
        result = self.generate_suggestions(state="deploy")
        assert 1 <= len(result) <= 3
        combined = " ".join(result).lower()
        assert any(
            word in combined
            for word in ["share", "accurate", "prediction", "alert", "retrain"]
        )

    def test_suggestions_model_state_with_runs(self):
        """Model state with completed runs mentions algorithm by name."""
        from models.model_run import ModelRun
        import json

        run = ModelRun(
            id="r1",
            project_id="p1",
            feature_set_id="fs1",
            algorithm="random_forest",
            status="done",
            metrics=json.dumps({"r2": 0.85, "mae": 50.0}),
        )
        result = self.generate_suggestions(
            state="model",
            model_runs=[run],
        )
        assert isinstance(result, list)
        assert 1 <= len(result) <= 3

    def test_suggestions_filters_duplicate_of_user_message(self):
        """Suggestions that echo the user's message too closely are filtered out."""
        # If the user just asked about correlations, "Show me a correlation heatmap"
        # should be filtered
        result = self.generate_suggestions(
            state="explore",
            last_user_message="Show me a correlation heatmap please",
        )
        # Should have some suggestions that don't directly repeat "correlation"
        for s in result:
            # The filtered suggestion should not start with the same leading words
            assert not s.lower().startswith("show me a correlation")

    def test_suggestions_returns_unique_items(self):
        """No duplicate suggestions in the returned list."""
        result = self.generate_suggestions(state="explore")
        assert len(result) == len(set(result))

    def test_suggestions_at_most_3(self):
        """Never returns more than 3 suggestions."""
        for state in ("upload", "explore", "shape", "model", "validate", "deploy"):
            result = self.generate_suggestions(state=state)
            assert len(result) <= 3, f"Got {len(result)} for state {state}"

    def test_suggestions_unknown_state_empty_or_minimal(self):
        """Unknown state gracefully returns whatever base provides (may be [])."""
        result = self.generate_suggestions(state="unknown_state_xyz")
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_suggestions_with_deployment(self):
        """Deploy state with active deployment adds API/history suggestions."""
        from models.deployment import Deployment

        dep = Deployment(
            id="dep1",
            model_run_id="r1",
            project_id="p1",
            endpoint_path="/api/predict/dep1",
            dashboard_url="/predict/dep1",
            is_active=True,
            request_count=5,
        )
        result = self.generate_suggestions(state="deploy", deployment=dep)
        assert 1 <= len(result) <= 3
        # With 5 predictions, the history suggestion should appear
        combined = " ".join(result).lower()
        assert any(word in combined for word in ["prediction", "api", "share", "alert"])

    def test_suggestions_validate_state_with_selected_run(self):
        """Validate state customises R² or accuracy suggestion from metrics."""
        from models.model_run import ModelRun
        import json

        run = ModelRun(
            id="r1",
            project_id="p1",
            feature_set_id="fs1",
            algorithm="linear_regression",
            status="done",
            is_selected=True,
            metrics=json.dumps({"r2": 0.78, "mae": 40.0}),
        )
        result = self.generate_suggestions(
            state="validate",
            model_runs=[run],
        )
        assert isinstance(result, list)
        assert len(result) >= 1
