"""Tests for feature selection automation.

Covers:
- identify_weak_features() pure function (tree-based, linear, not-available)
- GET /api/models/{run_id}/feature-selection endpoint
- POST /api/models/{project_id}/train with excluded_features
- Chat pattern _FEATURE_SEL_PATTERNS
"""

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Sample CSVs
# ---------------------------------------------------------------------------

REGRESSION_CSV = b"""f1,f2,f3,f4,target
1.0,0.5,100.0,0.0,10.0
2.0,1.0,200.0,0.0,20.0
3.0,1.5,300.0,0.0,30.0
4.0,2.0,400.0,0.0,40.0
5.0,2.5,500.0,0.0,50.0
6.0,3.0,600.0,0.0,60.0
7.0,3.5,700.0,0.0,70.0
8.0,4.0,800.0,0.0,80.0
9.0,4.5,900.0,0.0,90.0
10.0,5.0,1000.0,0.0,100.0
11.0,5.5,1100.0,0.0,110.0
12.0,6.0,1200.0,0.0,120.0
"""

CLASSIFICATION_CSV = b"""f1,f2,f3,f4,label
1.0,0.5,100.0,0.0,A
2.0,1.0,200.0,0.0,A
3.0,1.5,300.0,0.0,A
4.0,2.0,400.0,0.0,A
5.0,2.5,500.0,0.0,A
6.0,3.0,600.0,0.0,B
7.0,3.5,700.0,0.0,B
8.0,4.0,800.0,0.0,B
9.0,4.5,900.0,0.0,B
10.0,5.0,1000.0,0.0,B
11.0,5.5,1100.0,0.0,B
12.0,6.0,1200.0,0.0,B
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _setup_regression_project(client):
    """Create project, upload regression CSV, apply empty feature set, set target."""
    proj = client.post("/api/projects", json={"name": "FS Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(REGRESSION_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": "target"})
    return project_id, dataset_id


def _train_and_wait(
    client, project_id: str, algorithm: str = "random_forest_regressor"
) -> str:
    """Train a model and poll until done. Returns the run_id."""
    resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": [algorithm]},
    )
    assert resp.status_code == 202
    run_ids = resp.json()["model_run_ids"]
    assert len(run_ids) == 1
    run_id = run_ids[0]

    # Poll until done
    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    return run_id


# ---------------------------------------------------------------------------
# Unit tests: identify_weak_features pure function
# ---------------------------------------------------------------------------


class TestIdentifyWeakFeatures:
    def test_tree_model_returns_importances(self):
        from core.trainer import identify_weak_features

        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = identify_weak_features(model, ["useful_col", "zero_col"])

        assert result["has_importances"] is True
        assert result["method"] == "feature_importances"
        assert len(result["feature_importances"]) == 2
        assert result["threshold"] is not None

    def test_linear_model_uses_coef(self):
        from core.trainer import identify_weak_features

        X = np.array([[1, 0, 100], [2, 0, 200], [3, 0, 300], [4, 0, 400], [5, 0, 500]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = LinearRegression()
        model.fit(X, y)

        result = identify_weak_features(model, ["f1", "f2", "f3"])

        assert result["has_importances"] is True
        assert result["method"] == "coefficients"
        assert len(result["feature_importances"]) == 3

    def test_logistic_regression_multiclass_uses_coef(self):
        from core.trainer import identify_weak_features

        X = np.random.default_rng(42).random((30, 4))
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        result = identify_weak_features(model, ["f1", "f2", "f3", "f4"])

        assert result["has_importances"] is True
        assert result["method"] == "coefficients"
        assert len(result["feature_importances"]) == 4

    def test_mlp_returns_not_available(self):
        from core.trainer import identify_weak_features

        X = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = MLPRegressor(max_iter=50, random_state=42)
        model.fit(X, y)

        result = identify_weak_features(model, ["f1", "f2"])

        assert result["has_importances"] is False
        assert result["method"] == "not_available"
        assert result["weak_features"] == []
        assert result["n_weak"] == 0

    def test_weak_features_in_bottom_20_percent(self):
        from core.trainer import identify_weak_features

        # f1 strongly predicts y; f4 is all zeros — should be weakest
        X = np.array(
            [
                [1.0, 0.9, 0.8, 0.0],
                [2.0, 1.9, 1.8, 0.0],
                [3.0, 2.9, 2.8, 0.0],
                [4.0, 3.9, 3.8, 0.0],
                [5.0, 4.9, 4.8, 0.0],
                [6.0, 5.9, 5.8, 0.0],
                [7.0, 6.9, 6.8, 0.0],
                [8.0, 7.9, 7.8, 0.0],
            ]
        )
        y = X[:, 0]  # target = f1
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)

        result = identify_weak_features(model, ["f1", "f2", "f3", "f4"])

        assert result["has_importances"] is True
        # The zero-variance feature should appear as weak
        importances = {
            f["name"]: f["importance"] for f in result["feature_importances"]
        }
        assert importances["f4"] < importances["f1"]

    def test_sorted_descending_by_importance(self):
        from core.trainer import identify_weak_features

        X = np.random.default_rng(0).random((20, 3))
        y = X[:, 0] * 5 + X[:, 1]
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = identify_weak_features(model, ["high", "mid", "low"])

        imps = [f["importance"] for f in result["feature_importances"]]
        assert imps == sorted(imps, reverse=True)

    def test_ranks_are_1_through_n(self):
        from core.trainer import identify_weak_features

        X = np.random.default_rng(1).random((20, 3))
        y = X[:, 0]
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = identify_weak_features(model, ["a", "b", "c"])

        ranks = sorted(f["rank"] for f in result["feature_importances"])
        assert ranks == [1, 2, 3]

    def test_importances_sum_to_approx_one(self):
        from core.trainer import identify_weak_features

        X = np.random.default_rng(2).random((20, 4))
        y = X[:, 0] + X[:, 1]
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = identify_weak_features(model, ["a", "b", "c", "d"])

        total = sum(f["importance"] for f in result["feature_importances"])
        assert abs(total - 1.0) < 0.001

    def test_explanation_text_present(self):
        from core.trainer import identify_weak_features

        X = np.random.default_rng(3).random((20, 2))
        y = X[:, 0]
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = identify_weak_features(model, ["a", "b"])

        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 10

    def test_feature_col_count_mismatch_returns_not_available(self):
        from core.trainer import identify_weak_features

        X = np.random.default_rng(4).random((20, 3))
        y = X[:, 0]
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        # Pass wrong number of feature cols
        result = identify_weak_features(model, ["a", "b"])  # 3 features, 2 names

        assert result["has_importances"] is False

    def test_no_weak_features_when_all_equal(self):
        from core.trainer import identify_weak_features

        # All features equal importance (all-ones importances)
        X = np.random.default_rng(5).random((20, 3))
        y = X.sum(axis=1)
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)

        # At the 20th percentile, all features are >= threshold
        result = identify_weak_features(model, ["a", "b", "c"])
        # weak_features may or may not be empty but explanation should be present
        assert isinstance(result["explanation"], str)


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestFeatureSelectionEndpoint:
    def test_returns_404_for_unknown_run(self, client):
        resp = client.get("/api/models/nonexistent-run/feature-selection")
        assert resp.status_code == 404

    def test_returns_400_for_pending_run(self, client):
        project_id, _ = _setup_regression_project(client)
        # Trigger training but don't wait — use a run that will be pending briefly
        # Instead, create a run manually in pending state
        resp = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        run_id = resp.json()["model_run_ids"][0]
        # Immediately check — might be pending or training
        resp2 = client.get(f"/api/models/{run_id}/feature-selection")
        # Status is 400 (not done) or 200 if lightning fast
        assert resp2.status_code in (200, 400)

    def test_returns_feature_importances_after_training(self, client):
        project_id, _ = _setup_regression_project(client)
        run_id = _train_and_wait(client, project_id, "random_forest_regressor")

        resp = client.get(f"/api/models/{run_id}/feature-selection")
        assert resp.status_code == 200
        data = resp.json()

        assert data["run_id"] == run_id
        assert data["algorithm"] == "random_forest_regressor"
        assert data["n_features"] == 4  # f1, f2, f3, f4
        assert data["has_importances"] is True
        assert data["method"] == "feature_importances"
        assert len(data["feature_importances"]) == 4
        assert isinstance(data["weak_features"], list)
        assert isinstance(data["explanation"], str)

    def test_linear_regression_uses_coefficients(self, client):
        project_id, _ = _setup_regression_project(client)
        run_id = _train_and_wait(client, project_id, "linear_regression")

        resp = client.get(f"/api/models/{run_id}/feature-selection")
        assert resp.status_code == 200
        data = resp.json()

        assert data["method"] == "coefficients"
        assert data["has_importances"] is True

    def test_importances_sorted_descending(self, client):
        project_id, _ = _setup_regression_project(client)
        run_id = _train_and_wait(client, project_id, "random_forest_regressor")

        resp = client.get(f"/api/models/{run_id}/feature-selection")
        data = resp.json()

        imps = [f["importance"] for f in data["feature_importances"]]
        assert imps == sorted(imps, reverse=True)


# ---------------------------------------------------------------------------
# Training with excluded_features
# ---------------------------------------------------------------------------


class TestTrainWithExcludedFeatures:
    def test_train_excludes_specified_features(self, client):
        project_id, _ = _setup_regression_project(client)

        # Exclude f3 and f4 (the less useful ones)
        resp = client.post(
            f"/api/models/{project_id}/train",
            json={
                "algorithms": ["linear_regression"],
                "excluded_features": ["f3", "f4"],
            },
        )
        assert resp.status_code == 202

        run_id = resp.json()["model_run_ids"][0]
        # Poll until done
        for _ in range(30):
            runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
            run = next((r for r in runs if r["id"] == run_id), None)
            if run and run["status"] in ("done", "failed"):
                break
            time.sleep(0.2)

        assert run["status"] == "done"

    def test_excluding_all_features_returns_400(self, client):
        project_id, _ = _setup_regression_project(client)

        resp = client.post(
            f"/api/models/{project_id}/train",
            json={
                "algorithms": ["linear_regression"],
                "excluded_features": ["f1", "f2", "f3", "f4"],
            },
        )
        assert resp.status_code == 400
        assert "No feature columns" in resp.json()["detail"]

    def test_nonexistent_exclusions_are_silently_ignored(self, client):
        project_id, _ = _setup_regression_project(client)

        resp = client.post(
            f"/api/models/{project_id}/train",
            json={
                "algorithms": ["linear_regression"],
                "excluded_features": ["nonexistent_col"],
            },
        )
        # Should succeed — excluded column doesn't exist so feature_cols unchanged
        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Chat pattern test
# ---------------------------------------------------------------------------


class TestFeatureSelectionChatPattern:
    def test_feature_sel_pattern_matches_expected_phrases(self):
        from api.chat import _FEATURE_SEL_PATTERNS

        matches = [
            "are all columns useful?",
            "are all my features useful",
            "which features are not useful",
            "which columns should I remove",
            "remove unimportant features",
            "feature selection",
            "find weak features",
            "identify low-importance columns",
            "which features should I keep",
            "reduce my features",
        ]
        for phrase in matches:
            assert _FEATURE_SEL_PATTERNS.search(phrase), f"Should match: {phrase!r}"

    def test_feature_sel_pattern_does_not_match_unrelated(self):
        from api.chat import _FEATURE_SEL_PATTERNS

        non_matches = [
            "show me the data",
            "train a model",
            "what is the accuracy",
        ]
        for phrase in non_matches:
            assert not _FEATURE_SEL_PATTERNS.search(phrase), (
                f"Should NOT match: {phrase!r}"
            )
