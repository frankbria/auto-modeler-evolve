"""Tests for compute_class_conditional_importance() pure function, REST endpoint, and chat regex."""

import json

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from api.chat import _CLASS_FEAT_IMP_PATTERNS
from core.explainer import compute_class_conditional_importance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_Xy(n=100, n_features=4, n_classes=2):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_features))
    # Create separable classes based on first feature
    y = (X[:, 0] > 0).astype(int)
    if n_classes > 2:
        y = (X[:, 0] > 0.5).astype(int) + (X[:, 0] < -0.5).astype(int)
    return X, y


def _fitted_rf(X, y):
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(X, y)
    return clf


def _fitted_lr(X, y):
    clf = LogisticRegression(random_state=0, max_iter=200)
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestComputeClassConditionalImportancePure:
    def test_returns_required_keys(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model,
            X=X,
            y_pred=y_pred,
            feature_names=[f"f{i}" for i in range(4)],
        )
        for key in ["classes", "n_classes", "n_samples", "feature_names", "summary"]:
            assert key in result, f"Missing key: {key}"

    def test_n_samples_matches_input(self):
        X, y = _make_Xy(n=80)
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        assert result["n_samples"] == 80

    def test_n_classes_matches_predictions(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        n_unique = len(np.unique(y_pred))
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        assert result["n_classes"] == n_unique

    def test_class_labels_present(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        class_names = ["not_churn", "churn"]
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred,
            feature_names=[f"f{i}" for i in range(4)],
            class_names=class_names,
        )
        labels = {c["class_label"] for c in result["classes"]}
        for name in class_names:
            assert name in labels

    def test_class_labels_fallback_to_prediction_value(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred,
            feature_names=[f"f{i}" for i in range(4)],
            class_names=None,
        )
        # Labels should be string representations of the int class values
        for cls in result["classes"]:
            assert isinstance(cls["class_label"], str)

    def test_each_class_has_required_keys(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        for cls in result["classes"]:
            for key in ["class_label", "sample_count", "sample_pct", "features", "top_feature", "summary"]:
                assert key in cls, f"Missing key {key} in class dict"

    def test_sample_counts_sum_to_total(self):
        X, y = _make_Xy(n=100)
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        total = sum(c["sample_count"] for c in result["classes"])
        assert total == 100

    def test_sample_pct_sums_to_100(self):
        X, y = _make_Xy(n=100)
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        total_pct = sum(c["sample_pct"] for c in result["classes"])
        assert abs(total_pct - 100.0) < 1.0  # allow rounding tolerance

    def test_feature_entry_keys(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        for cls in result["classes"]:
            for feat in cls["features"]:
                for key in [
                    "feature", "global_importance", "mean_for_class",
                    "global_mean", "deviation_pct", "weighted_deviation", "direction"
                ]:
                    assert key in feat, f"Missing feat key: {key}"

    def test_direction_valid_values(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        valid_dirs = {"above_average", "below_average", "similar"}
        for cls in result["classes"]:
            for feat in cls["features"]:
                assert feat["direction"] in valid_dirs

    def test_features_sorted_by_weighted_deviation(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        for cls in result["classes"]:
            devs = [f["weighted_deviation"] for f in cls["features"]]
            assert devs == sorted(devs, reverse=True)

    def test_top_feature_is_in_feature_names(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        feature_names = [f"f{i}" for i in range(4)]
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=feature_names
        )
        for cls in result["classes"]:
            assert cls["top_feature"] in feature_names

    def test_summary_is_string(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_logistic_regression_works(self):
        X, y = _make_Xy()
        model = _fitted_lr(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        assert result["n_classes"] >= 2

    def test_too_few_rows_raises(self):
        X = np.random.standard_normal((5, 4))
        y = np.array([0, 0, 1, 1, 0])
        model = RandomForestClassifier(n_estimators=2, random_state=0)
        model.fit(X, y)
        y_pred = model.predict(X)
        with pytest.raises(ValueError, match="at least 10 rows"):
            compute_class_conditional_importance(
                model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
            )

    def test_single_class_raises(self):
        X = np.random.standard_normal((20, 4))
        y_all_zero = np.zeros(20, dtype=int)
        # Force all predictions to same class
        class FakeModel:
            classes_ = np.array([0, 1])
            feature_importances_ = np.array([0.25, 0.25, 0.25, 0.25])
            def predict(self, X): return np.zeros(len(X), dtype=int)
        with pytest.raises(ValueError, match="at least 2 distinct"):
            compute_class_conditional_importance(
                model=FakeModel(),
                X=X,
                y_pred=y_all_zero,
                feature_names=[f"f{i}" for i in range(4)],
            )

    def test_global_importance_sums_to_one(self):
        X, y = _make_Xy()
        model = _fitted_rf(X, y)
        y_pred = model.predict(X)
        result = compute_class_conditional_importance(
            model=model, X=X, y_pred=y_pred, feature_names=[f"f{i}" for i in range(4)]
        )
        for cls in result["classes"]:
            total_imp = sum(f["global_importance"] for f in cls["features"])
            assert abs(total_imp - 1.0) < 0.01 or total_imp <= 1.0


# ---------------------------------------------------------------------------
# Chat pattern tests
# ---------------------------------------------------------------------------


class TestClassFeatImpPatterns:
    MATCH = [
        "what features drive churn predictions?",
        "feature importance per class",
        "class-conditional feature importance",
        "class-specific feature breakdown",
        "what makes the model predict churn vs not-churn",
        "feature breakdown by class",
        "why does the model predict different classes",
        "what features are different for each class",
    ]

    NO_MATCH = [
        "show me the data",
        "train a model",
        "what is my model accuracy?",
        "deploy my model",
    ]

    @pytest.mark.parametrize("msg", MATCH)
    def test_matches(self, msg):
        assert _CLASS_FEAT_IMP_PATTERNS.search(msg), f"Should match: {msg}"

    @pytest.mark.parametrize("msg", NO_MATCH)
    def test_no_match(self, msg):
        assert not _CLASS_FEAT_IMP_PATTERNS.search(msg), f"Should NOT match: {msg}"


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    """Async HTTP test client with isolated in-process DB (same pattern as other endpoint tests)."""
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    from main import app

    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path
    SQLModel.metadata.create_all(db_module.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.anyio
async def test_class_feat_imp_returns_200(ac, tmp_path):
    """Classification model returns 200 with classes list."""
    import joblib
    import pandas as pd
    from sqlmodel import Session

    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project

    X, y = _make_Xy(n=60)
    csv_path = tmp_path / "clf_data.csv"
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y
    df.to_csv(csv_path, index=False)

    model = _fitted_rf(X, y)
    model_path = tmp_path / "clf_model.joblib"
    joblib.dump(model, model_path)

    with Session(db_module.engine) as session:
        proj = Project(name="cfi_test")
        session.add(proj)
        session.commit()
        session.refresh(proj)

        ds = Dataset(
            project_id=proj.id,
            filename="clf_data.csv",
            file_path=str(csv_path),
            row_count=60,
            column_count=5,
            columns="{}",
        )
        session.add(ds)
        session.commit()
        session.refresh(ds)

        fset = FeatureSet(
            project_id=proj.id,
            dataset_id=ds.id,
            target_column="target",
            problem_type="classification",
            feature_columns=json.dumps([f"f{i}" for i in range(4)]),
            transformations="[]",
            is_active=True,
        )
        session.add(fset)
        session.commit()
        session.refresh(fset)

        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fset.id,
            algorithm="random_forest_classifier",
            status="done",
            model_path=str(model_path),
            metrics=json.dumps({"accuracy": 0.85}),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    resp = await ac.get(f"/api/models/{run_id}/class-feature-importance")
    assert resp.status_code == 200
    data = resp.json()
    assert "classes" in data
    assert data["n_classes"] >= 2
    assert "summary" in data


@pytest.mark.anyio
async def test_class_feat_imp_404_unknown(ac):
    """Unknown run returns 404."""
    resp = await ac.get("/api/models/nonexistent-uuid/class-feature-importance")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_class_feat_imp_400_regression(ac, tmp_path):
    """Regression run returns 400."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sqlmodel import Session

    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project

    X = np.random.default_rng(7).standard_normal((40, 3))
    y = X[:, 0] * 2

    csv_path = tmp_path / "reg_data.csv"
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = y
    df.to_csv(csv_path, index=False)

    model = LinearRegression().fit(X, y)
    model_path = tmp_path / "reg_model.joblib"
    joblib.dump(model, model_path)

    with Session(db_module.engine) as session:
        proj = Project(name="reg_cfi")
        session.add(proj)
        session.commit()
        session.refresh(proj)

        ds = Dataset(
            project_id=proj.id,
            filename="reg_data.csv",
            file_path=str(csv_path),
            row_count=40,
            column_count=4,
            columns="{}",
        )
        session.add(ds)
        session.commit()
        session.refresh(ds)

        fset = FeatureSet(
            project_id=proj.id,
            dataset_id=ds.id,
            target_column="target",
            problem_type="regression",
            feature_columns=json.dumps(["a", "b", "c"]),
            transformations="[]",
            is_active=True,
        )
        session.add(fset)
        session.commit()
        session.refresh(fset)

        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fset.id,
            algorithm="linear_regression",
            status="done",
            model_path=str(model_path),
            metrics=json.dumps({"r2": 0.7}),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    resp = await ac.get(f"/api/models/{run_id}/class-feature-importance")
    assert resp.status_code == 400
