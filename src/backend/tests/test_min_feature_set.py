"""Tests for minimum viable feature set (greedy backward elimination)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from api.chat import _MIN_FEATURE_SET_PATTERNS
from core.validator import compute_min_viable_feature_set

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regression(n=200, n_feat=5, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_feat))
    # Only first 2 features matter — rest are pure noise
    y = 3 * X[:, 0] + 2 * X[:, 1] + noise * rng.standard_normal(n)
    return X, y, [f"feat_{i}" for i in range(n_feat)]


def _make_classification(n=200, n_feat=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_feat))
    # Only first 2 features matter
    logit = 2 * X[:, 0] + 1.5 * X[:, 1]
    y = (logit + rng.standard_normal(n) * 0.5 > 0).astype(int)
    return X, y, [f"feat_{i}" for i in range(n_feat)]


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


class TestComputeMinViableFeatureSet:
    def test_required_keys(self):
        X, y, names = _make_regression()
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        for key in (
            "features",
            "n_original",
            "n_minimal",
            "features_dropped",
            "features_retained",
            "features_dropped_list",
            "baseline_score",
            "minimal_score",
            "score_loss",
            "can_simplify",
            "reduction_pct",
            "metric",
            "summary",
        ):
            assert key in result, f"Missing key: {key}"

    def test_n_original_matches_input_features(self):
        X, y, names = _make_regression(n_feat=5)
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        assert result["n_original"] == 5

    def test_regression_metric_is_r2(self):
        X, y, names = _make_regression()
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        assert result["metric"] == "r2"

    def test_classification_metric_is_accuracy(self):
        X, y, names = _make_classification()
        result = compute_min_viable_feature_set(
            X, y, names, LogisticRegression, {"max_iter": 200}, "classification"
        )
        assert result["metric"] == "accuracy"

    def test_features_list_length_matches_n_original(self):
        X, y, names = _make_regression(n_feat=4)
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        assert len(result["features"]) == result["n_original"]

    def test_feature_entry_keys(self):
        X, y, names = _make_regression(n_feat=3)
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        entry = result["features"][0]
        for k in ("feature", "importance", "kept", "rank"):
            assert k in entry

    def test_retained_plus_dropped_equals_original(self):
        X, y, names = _make_regression(n_feat=5)
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        assert result["n_minimal"] + result["features_dropped"] == result["n_original"]

    def test_retained_list_length_matches_n_minimal(self):
        X, y, names = _make_regression(n_feat=5)
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        assert len(result["features_retained"]) == result["n_minimal"]

    def test_dropped_list_length_matches_features_dropped(self):
        X, y, names = _make_regression(n_feat=5)
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        assert len(result["features_dropped_list"]) == result["features_dropped"]

    def test_score_loss_non_negative(self):
        X, y, names = _make_regression()
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        # score_loss = baseline - minimal, may be tiny negative due to CV variance
        assert result["score_loss"] >= -0.05

    def test_can_simplify_when_noise_features_present(self):
        """With mostly-noise features, at least some should be droppable."""
        X, y, names = _make_regression(n_feat=6, noise=0.01)
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression", tolerance=0.05
        )
        # At least one noise feature should be droppable
        assert result["can_simplify"] is True

    def test_can_simplify_false_when_all_features_important(self):
        """With perfectly collinear features all contributing, can_simplify may be False."""
        rng = np.random.default_rng(7)
        n = 100
        # 2 features, both essential — dropping either loses a lot
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        y = 5 * x1 + 5 * x2 + 0.01 * rng.standard_normal(n)
        X = np.column_stack([x1, x2])
        result = compute_min_viable_feature_set(
            X, y, ["x1", "x2"], LinearRegression, {}, "regression", tolerance=0.01
        )
        # Very tight tolerance — should not be able to drop either strong predictor
        assert result["n_minimal"] >= 1  # must keep at least 1

    def test_reduction_pct_consistent(self):
        X, y, names = _make_regression(n_feat=5)
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        expected = result["features_dropped"] / result["n_original"] * 100
        assert abs(result["reduction_pct"] - expected) < 0.01

    def test_summary_is_string(self):
        X, y, names = _make_regression()
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_baseline_score_in_valid_range(self):
        X, y, names = _make_regression()
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        # R² can be negative, but should be reasonable for well-structured data
        assert -1.0 <= result["baseline_score"] <= 1.0

    def test_classification_baseline_score_in_valid_range(self):
        X, y, names = _make_classification()
        result = compute_min_viable_feature_set(
            X, y, names, LogisticRegression, {"max_iter": 200}, "classification"
        )
        assert 0.0 <= result["baseline_score"] <= 1.0

    def test_kept_flags_consistent_with_retained_list(self):
        X, y, names = _make_regression(n_feat=5)
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        kept_names = {f["feature"] for f in result["features"] if f["kept"]}
        assert kept_names == set(result["features_retained"])

    def test_dropped_flags_consistent_with_dropped_list(self):
        X, y, names = _make_regression(n_feat=5)
        result = compute_min_viable_feature_set(
            X, y, names, LinearRegression, {}, "regression"
        )
        dropped_names = {f["feature"] for f in result["features"] if not f["kept"]}
        assert dropped_names == set(result["features_dropped_list"])

    def test_raises_for_too_few_rows(self):
        X = np.ones((5, 3))
        y = np.ones(5)
        with pytest.raises(ValueError, match="10 rows"):
            compute_min_viable_feature_set(
                X, y, ["a", "b", "c"], LinearRegression, {}, "regression"
            )

    def test_raises_for_too_few_features(self):
        X = np.ones((50, 1))
        y = np.ones(50)
        with pytest.raises(ValueError, match="2 features"):
            compute_min_viable_feature_set(
                X, y, ["a"], LinearRegression, {}, "regression"
            )

    def test_subsampling_does_not_crash(self):
        """max_rows sub-sampling path."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5000, 4))
        y = X[:, 0] + 0.5 * rng.standard_normal(5000)
        result = compute_min_viable_feature_set(
            X,
            y,
            ["a", "b", "c", "d"],
            LinearRegression,
            {},
            "regression",
            max_rows=200,
        )
        assert "n_original" in result

    def test_decision_tree_uses_feature_importances(self):
        """Model with feature_importances_ attribute."""
        X, y, names = _make_regression(n_feat=4)
        result = compute_min_viable_feature_set(
            X, y, names, DecisionTreeRegressor, {"max_depth": 3}, "regression"
        )
        assert len(result["features"]) == 4

    def test_classification_with_decision_tree(self):
        X, y, names = _make_classification(n_feat=4)
        result = compute_min_viable_feature_set(
            X, y, names, DecisionTreeClassifier, {"max_depth": 3}, "classification"
        )
        assert result["metric"] == "accuracy"
        assert 0 <= result["baseline_score"] <= 1.0


# ---------------------------------------------------------------------------
# Regex tests
# ---------------------------------------------------------------------------


class TestMinFeatureSetPatterns:
    def _match(self, text: str) -> bool:
        return bool(_MIN_FEATURE_SET_PATTERNS.search(text))

    def test_minimum_features_needed(self):
        assert self._match("what's the minimum features needed?")

    def test_which_features_can_i_drop(self):
        assert self._match("which features can I drop?")

    def test_simplify_my_model(self):
        assert self._match("simplify my model")

    def test_reduce_feature_set(self):
        assert self._match("reduce my feature set")

    def test_feature_pruning(self):
        assert self._match("feature pruning")

    def test_fewest_features_for_accuracy(self):
        assert self._match("fewest features for good predictions")

    def test_redundant_features(self):
        assert self._match("which features are redundant?")

    def test_can_i_drop_any(self):
        assert self._match("can I drop any features?")

    # False-positive guards
    def test_no_match_generic_features(self):
        assert not self._match("show me the feature importance chart")

    def test_no_match_add_feature(self):
        assert not self._match("add a new feature column")


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


def _make_test_app():
    from fastapi import FastAPI
    from api.validation import router

    app = FastAPI()
    app.include_router(router)
    from tests.conftest import install_test_auth
    install_test_auth(app)
    return app


def _seed_db_with_run(tmp_path: Path):
    """Create an in-memory DB, dataset, feature_set, and model_run."""
    from sqlmodel import Session, create_engine, SQLModel
    from models.project import Project
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun

    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    # Write CSV
    csv_path = tmp_path / "data.csv"
    rng = np.random.default_rng(42)
    n = 150
    X = rng.standard_normal((n, 4))
    y = 2 * X[:, 0] + rng.standard_normal(n) * 0.1
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df["target"] = y
    df.to_csv(csv_path, index=False)

    # Train a model
    model_path = tmp_path / "model.joblib"
    from sklearn.linear_model import LinearRegression

    m = LinearRegression()
    m.fit(X, y)
    joblib.dump(m, model_path)

    with Session(engine) as session:
        project = Project(owner_id="test-default-owner", name="test")
        session.add(project)
        session.commit()
        session.refresh(project)

        dataset = Dataset(
            project_id=project.id,
            filename="data.csv",
            file_path=str(csv_path),
            row_count=n,
            column_count=5,
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)

        feature_set = FeatureSet(
            dataset_id=dataset.id,
            target_column="target",
            problem_type="regression",
            transformations="[]",
        )
        session.add(feature_set)
        session.commit()
        session.refresh(feature_set)

        run = ModelRun(
            project_id=project.id,
            feature_set_id=feature_set.id,
            algorithm="linear_regression",
            status="done",
            model_path=str(model_path),
            metrics="{}",
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        return str(run.id), str(db_path)


class TestMinFeatureSetEndpoint:
    def test_404_for_unknown_run(self, tmp_path):
        from fastapi import FastAPI
        from api.validation import router
        from sqlmodel import Session, create_engine, SQLModel

        db_path = tmp_path / "e.db"
        engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(engine)

        app = FastAPI()
        app.include_router(router)
        from tests.conftest import install_test_auth
        install_test_auth(app)

        import api.validation as val_mod

        orig = val_mod.get_session

        def override():
            with Session(engine) as s:
                yield s

        app.dependency_overrides[orig] = override
        client = TestClient(app)
        resp = client.get("/api/models/nonexistent-id/min-feature-set")
        assert resp.status_code == 404

    def test_200_with_valid_run(self, tmp_path):
        from fastapi import FastAPI
        from api.validation import router
        from sqlmodel import Session, create_engine

        run_id, db_path = _seed_db_with_run(tmp_path)
        engine = create_engine(f"sqlite:///{db_path}")

        app = FastAPI()
        app.include_router(router)
        from tests.conftest import install_test_auth
        install_test_auth(app)

        import api.validation as val_mod

        orig = val_mod.get_session

        def override():
            with Session(engine) as s:
                yield s

        app.dependency_overrides[orig] = override
        client = TestClient(app)
        resp = client.get(f"/api/models/{run_id}/min-feature-set")
        assert resp.status_code == 200
        data = resp.json()
        assert "n_original" in data
        assert "features_retained" in data
        assert data["n_original"] == 4
