"""Tests for ensemble methods: VotingRegressor/Classifier + StackingRegressor/Classifier."""

import io
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.trainer import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    _build_ensemble_estimators,
    _ensemble_vote_explanation,
    _stacking_weight_explanation,
    prepare_features,
    train_single_model,
    recommend_models,
)


# ---------------------------------------------------------------------------
# Small datasets for fast testing
# ---------------------------------------------------------------------------

REGRESSION_CSV = b"""a,b,c,target
1.0,2.0,3.0,10.0
2.0,3.0,4.0,20.0
3.0,4.0,5.0,30.0
4.0,5.0,6.0,40.0
5.0,6.0,7.0,50.0
6.0,7.0,8.0,60.0
7.0,8.0,9.0,70.0
8.0,9.0,10.0,80.0
9.0,10.0,11.0,90.0
10.0,11.0,12.0,100.0
11.0,12.0,13.0,110.0
12.0,13.0,14.0,120.0
"""

CLASSIFICATION_CSV = b"""a,b,c,label
1.0,2.0,0.5,cat
2.0,3.0,1.0,dog
3.0,4.0,0.5,cat
4.0,5.0,1.0,dog
5.0,6.0,0.5,cat
6.0,7.0,1.0,dog
7.0,8.0,0.5,cat
8.0,9.0,1.0,dog
9.0,10.0,0.5,cat
10.0,11.0,1.0,dog
11.0,12.0,0.5,cat
12.0,13.0,1.0,dog
"""


@pytest.fixture
def reg_Xy():
    df = pd.read_csv(io.StringIO(REGRESSION_CSV.decode()))
    X, y, _ = prepare_features(df, ["a", "b", "c"], "target", "regression")
    return X, y


@pytest.fixture
def cls_Xy():
    df = pd.read_csv(io.StringIO(CLASSIFICATION_CSV.decode()))
    X, y, _ = prepare_features(df, ["a", "b", "c"], "label", "classification")
    return X, y


# ---------------------------------------------------------------------------
# Registry: ensemble algorithms present
# ---------------------------------------------------------------------------


def test_voting_regressor_in_registry():
    assert "voting_regressor" in REGRESSION_ALGORITHMS
    info = REGRESSION_ALGORITHMS["voting_regressor"]
    assert info["is_ensemble"] is True
    assert info["ensemble_type"] == "voting"
    assert len(info["base_algorithms"]) >= 2


def test_stacking_regressor_in_registry():
    assert "stacking_regressor" in REGRESSION_ALGORITHMS
    info = REGRESSION_ALGORITHMS["stacking_regressor"]
    assert info["is_ensemble"] is True
    assert info["ensemble_type"] == "stacking"


def test_voting_classifier_in_registry():
    assert "voting_classifier" in CLASSIFICATION_ALGORITHMS
    info = CLASSIFICATION_ALGORITHMS["voting_classifier"]
    assert info["is_ensemble"] is True
    assert info["ensemble_type"] == "voting"


def test_stacking_classifier_in_registry():
    assert "stacking_classifier" in CLASSIFICATION_ALGORITHMS
    info = CLASSIFICATION_ALGORITHMS["stacking_classifier"]
    assert info["is_ensemble"] is True
    assert info["ensemble_type"] == "stacking"


def test_ensemble_algos_have_name_and_plain_english():
    for key in ["voting_regressor", "stacking_regressor"]:
        info = REGRESSION_ALGORITHMS[key]
        assert info["name"]
        assert info["plain_english"]
        assert info["best_for"]
    for key in ["voting_classifier", "stacking_classifier"]:
        info = CLASSIFICATION_ALGORITHMS[key]
        assert info["name"]
        assert info["plain_english"]
        assert info["best_for"]


# ---------------------------------------------------------------------------
# _build_ensemble_estimators
# ---------------------------------------------------------------------------


def test_build_ensemble_estimators_regression():
    base_keys = ["linear_regression", "random_forest_regressor"]
    estimators = _build_ensemble_estimators(base_keys, REGRESSION_ALGORITHMS)
    assert len(estimators) == 2
    names = [e[0] for e in estimators]
    assert "linear_regression" in names
    assert "random_forest_regressor" in names
    # Each estimator is a sklearn object
    for name, est in estimators:
        assert hasattr(est, "fit")


def test_build_ensemble_estimators_skips_missing():
    # Non-existent key should be silently skipped
    base_keys = ["linear_regression", "nonexistent_algo"]
    estimators = _build_ensemble_estimators(base_keys, REGRESSION_ALGORITHMS)
    assert len(estimators) == 1
    assert estimators[0][0] == "linear_regression"


def test_build_ensemble_estimators_classification():
    base_keys = ["logistic_regression", "random_forest_classifier"]
    estimators = _build_ensemble_estimators(base_keys, CLASSIFICATION_ALGORITHMS)
    assert len(estimators) == 2


# ---------------------------------------------------------------------------
# _ensemble_vote_explanation
# ---------------------------------------------------------------------------


def test_ensemble_vote_explanation_regression(reg_Xy):
    X, y = reg_Xy
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    estimators = [
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor(n_estimators=5, random_state=42)),
    ]
    for name, est in estimators:
        est.fit(X, y)
    votes = _ensemble_vote_explanation(
        ["lr", "rf"], [e for _, e in estimators], X, y, "regression"
    )
    assert "lr" in votes
    assert "rf" in votes
    assert isinstance(votes["lr"], float)
    assert isinstance(votes["rf"], float)


def test_ensemble_vote_explanation_classification(cls_Xy):
    X, y = cls_Xy
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    estimators = [
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(n_estimators=5, random_state=42)),
    ]
    for name, est in estimators:
        est.fit(X, y)
    classes = [0, 1]
    votes = _ensemble_vote_explanation(
        ["lr", "rf"], [e for _, e in estimators], X, y, "classification", classes=classes
    )
    assert "lr" in votes
    assert "rf" in votes
    # Each should be a dict of class->count
    for v in votes.values():
        assert isinstance(v, dict)


# ---------------------------------------------------------------------------
# _stacking_weight_explanation
# ---------------------------------------------------------------------------


def test_stacking_weight_explanation_sums_to_one():
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    meta = LogisticRegression(max_iter=1000)
    # Manually set coef_ to simulate a fitted meta-learner
    meta.coef_ = np.array([[0.6, 0.3, 0.1]])
    weights = _stacking_weight_explanation(["lr", "rf", "gb"], meta)
    assert set(weights.keys()) == {"lr", "rf", "gb"}
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.01


def test_stacking_weight_explanation_no_coef():
    from sklearn.linear_model import LinearRegression
    meta = LinearRegression()  # not fitted — no coef_
    weights = _stacking_weight_explanation(["lr", "rf"], meta)
    assert weights == {}


# ---------------------------------------------------------------------------
# train_single_model with ensemble algorithms
# ---------------------------------------------------------------------------


def test_train_voting_regressor(tmp_path, reg_Xy):
    X, y = reg_Xy
    result = train_single_model(X, y, "voting_regressor", "regression", tmp_path, "run1")
    assert "metrics" in result
    assert "r2" in result["metrics"]
    assert "model_path" in result
    assert "ensemble_type" in result["metrics"]
    assert result["metrics"]["ensemble_type"] == "voting"
    assert "ensemble_votes" in result["metrics"]
    assert Path(result["model_path"]).exists()


def test_train_stacking_regressor(tmp_path, reg_Xy):
    X, y = reg_Xy
    result = train_single_model(X, y, "stacking_regressor", "regression", tmp_path, "run2")
    assert "metrics" in result
    assert "r2" in result["metrics"]
    assert result["metrics"]["ensemble_type"] == "stacking"
    assert "stacking_weights" in result["metrics"]
    assert Path(result["model_path"]).exists()


def test_train_voting_classifier(tmp_path, cls_Xy):
    X, y = cls_Xy
    result = train_single_model(X, y, "voting_classifier", "classification", tmp_path, "run3")
    assert "metrics" in result
    assert "accuracy" in result["metrics"]
    assert result["metrics"]["ensemble_type"] == "voting"
    assert "ensemble_votes" in result["metrics"]
    assert Path(result["model_path"]).exists()


def test_train_stacking_classifier(tmp_path, cls_Xy):
    X, y = cls_Xy
    result = train_single_model(X, y, "stacking_classifier", "classification", tmp_path, "run4")
    assert "metrics" in result
    assert "accuracy" in result["metrics"]
    assert result["metrics"]["ensemble_type"] == "stacking"
    assert Path(result["model_path"]).exists()


def test_ensemble_summary_in_summary_string(tmp_path, reg_Xy):
    X, y = reg_Xy
    result = train_single_model(X, y, "voting_regressor", "regression", tmp_path, "run5")
    # summary should mention the base models
    assert "Combines" in result["summary"] or "voting" in result["summary"].lower()


def test_ensemble_classifier_has_ensemble_summary(tmp_path, cls_Xy):
    X, y = cls_Xy
    result = train_single_model(X, y, "voting_classifier", "classification", tmp_path, "run6")
    # ensemble_summary field should be set for classification voting
    if "ensemble_summary" in result["metrics"]:
        assert "voted" in result["metrics"]["ensemble_summary"].lower()


def test_voting_regressor_metrics_include_train_test_size(tmp_path, reg_Xy):
    X, y = reg_Xy
    result = train_single_model(X, y, "voting_regressor", "regression", tmp_path, "run7")
    assert "train_size" in result["metrics"]
    assert "test_size" in result["metrics"]
    assert result["metrics"]["train_size"] > 0
    assert result["metrics"]["test_size"] > 0


# ---------------------------------------------------------------------------
# recommend_models includes ensemble algorithms
# ---------------------------------------------------------------------------


def test_recommendations_include_ensemble_regression():
    recs = recommend_models("regression", n_rows=500, n_features=5)
    keys = [r["algorithm"] for r in recs]
    assert "voting_regressor" in keys
    assert "stacking_regressor" in keys


def test_recommendations_include_ensemble_classification():
    recs = recommend_models("classification", n_rows=500, n_features=5)
    keys = [r["algorithm"] for r in recs]
    assert "voting_classifier" in keys
    assert "stacking_classifier" in keys


def test_ensemble_recommendations_have_recommended_because():
    recs = recommend_models("regression", n_rows=500, n_features=5)
    for r in recs:
        if "voting" in r["algorithm"] or "stacking" in r["algorithm"]:
            assert r["recommended_because"]
            assert len(r["recommended_because"]) > 0


def test_ensemble_recommendations_small_dataset():
    recs = recommend_models("regression", n_rows=100, n_features=3)
    for r in recs:
        if "voting" in r["algorithm"] or "stacking" in r["algorithm"]:
            # Small dataset warning should mention rows or individual models
            assert r["recommended_because"]


# ---------------------------------------------------------------------------
# API: training with ensemble algorithms
# ---------------------------------------------------------------------------


SAMPLE_CSV = b"""a,b,c,target
1.0,2.0,3.0,10.0
2.0,3.0,4.0,20.0
3.0,4.0,5.0,30.0
4.0,5.0,6.0,40.0
5.0,6.0,7.0,50.0
6.0,7.0,8.0,60.0
7.0,8.0,9.0,70.0
8.0,9.0,10.0,80.0
9.0,10.0,11.0,90.0
10.0,11.0,12.0,100.0
11.0,12.0,13.0,110.0
12.0,13.0,14.0,120.0
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

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module
    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def project_with_feature_set(client):
    proj = client.post("/api/projects", json={"name": "Ensemble Test"})
    assert proj.status_code == 201, proj.text
    pid = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": pid},
        files={"file": ("test.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    dataset_id = upload.json()["dataset_id"]

    apply = client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert apply.status_code == 201, apply.text

    target = client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "target"},
    )
    assert target.status_code == 200, target.text
    return pid


def test_api_recommendations_include_ensemble(client, project_with_feature_set):
    pid = project_with_feature_set
    resp = client.get(f"/api/models/{pid}/recommendations")
    assert resp.status_code == 200
    keys = [r["algorithm"] for r in resp.json()["recommendations"]]
    assert "voting_regressor" in keys
    assert "stacking_regressor" in keys


def test_api_train_voting_regressor(client, project_with_feature_set):
    pid = project_with_feature_set
    resp = client.post(
        f"/api/models/{pid}/train",
        json={"algorithms": ["voting_regressor"]},
    )
    assert resp.status_code == 202
    run_ids = resp.json()["model_run_ids"]
    assert len(run_ids) == 1

    # Poll until done (with timeout)
    deadline = time.time() + 30
    final_run = None
    while time.time() < deadline:
        runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_ids[0]), None)
        if run and run["status"] in ("done", "failed"):
            final_run = run
            break
        time.sleep(0.2)

    assert final_run is not None
    assert final_run["status"] == "done", f"Training failed: {final_run.get('error_message')}"
    assert final_run["metrics"] is not None
    assert "r2" in final_run["metrics"]
    assert final_run["metrics"].get("ensemble_type") == "voting"


def test_api_train_stacking_regressor(client, project_with_feature_set):
    pid = project_with_feature_set
    resp = client.post(
        f"/api/models/{pid}/train",
        json={"algorithms": ["stacking_regressor"]},
    )
    assert resp.status_code == 202
    run_ids = resp.json()["model_run_ids"]

    deadline = time.time() + 30
    final_run = None
    while time.time() < deadline:
        runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_ids[0]), None)
        if run and run["status"] in ("done", "failed"):
            final_run = run
            break
        time.sleep(0.2)

    assert final_run is not None
    assert final_run["status"] == "done", f"Training failed: {final_run.get('error_message')}"
    assert final_run["metrics"].get("ensemble_type") == "stacking"
