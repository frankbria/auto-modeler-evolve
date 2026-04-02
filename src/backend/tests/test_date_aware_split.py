"""Tests for date-aware (chronological) train/test split.

Covers:
- chronological_split() pure function
- train_single_model() with split_strategy="chronological"
- GET /api/models/{project_id}/split-strategy endpoint
- POST /api/models/{project_id}/train with split_strategy="chronological"
- split_strategy recorded in model run metrics
- chat intent: _TIME_SPLIT_PATTERNS
"""

import io

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Sample CSV with a date column for chronological split
# ---------------------------------------------------------------------------

DATE_CSV = b"""date,revenue,region
2023-01-01,100,East
2023-02-01,120,East
2023-03-01,110,West
2023-04-01,130,East
2023-05-01,150,West
2023-06-01,140,East
2023-07-01,160,West
2023-08-01,170,East
2023-09-01,180,West
2023-10-01,190,East
2023-11-01,200,West
2023-12-01,210,East
2024-01-01,220,West
2024-02-01,230,East
2024-03-01,240,West
"""

# CSV without any date column
NO_DATE_CSV = b"""feature1,feature2,target
1.0,2.0,10.0
2.0,3.0,20.0
3.0,4.0,30.0
4.0,5.0,40.0
5.0,6.0,50.0
6.0,7.0,60.0
7.0,8.0,70.0
8.0,9.0,80.0
9.0,10.0,90.0
10.0,11.0,100.0
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


def _setup_project(client, csv_bytes: bytes, target: str = "revenue"):
    """Create project, upload CSV, apply empty feature set, set regression target."""
    proj = client.post("/api/projects", json={"name": "Split Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    apply = client.post(
        f"/api/features/{dataset_id}/apply", json={"transformations": []}
    )
    assert apply.status_code == 201

    target_resp = client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": target},
    )
    assert target_resp.status_code == 200
    return project_id, dataset_id


# ---------------------------------------------------------------------------
# Unit tests: chronological_split()
# ---------------------------------------------------------------------------


def test_chronological_split_basic():
    from core.trainer import chronological_split

    train_idx, test_idx = chronological_split(10)
    assert len(train_idx) == 8
    assert len(test_idx) == 2
    assert list(train_idx) == list(range(8))
    assert list(test_idx) == [8, 9]


def test_chronological_split_no_overlap():
    from core.trainer import chronological_split

    train_idx, test_idx = chronological_split(20)
    combined = set(train_idx) | set(test_idx)
    assert len(combined) == 20
    assert set(train_idx) & set(test_idx) == set()


def test_chronological_split_small():
    """With <10 rows the caller falls back to train=test, but chronological_split itself still works."""
    from core.trainer import chronological_split

    train_idx, test_idx = chronological_split(3)
    assert len(train_idx) == 2
    assert len(test_idx) == 1


def test_chronological_split_custom_test_size():
    from core.trainer import chronological_split

    train_idx, test_idx = chronological_split(100, test_size=0.3)
    assert len(test_idx) == 30
    assert len(train_idx) == 70


def test_chronological_split_minimum_test():
    """test_size fraction rounds to at least 1 test row."""
    from core.trainer import chronological_split

    train_idx, test_idx = chronological_split(5, test_size=0.1)
    assert len(test_idx) >= 1


# ---------------------------------------------------------------------------
# Unit tests: train_single_model with split_strategy
# ---------------------------------------------------------------------------


def test_train_single_model_records_split_strategy_random(tmp_path):
    import numpy as np

    from core.trainer import train_single_model

    X = np.random.default_rng(42).random((20, 3))
    y = np.random.default_rng(42).random(20)

    result = train_single_model(
        X, y, "linear_regression", "regression",
        tmp_path, "run-random", split_strategy="random"
    )
    assert result["metrics"]["split_strategy"] == "random"
    assert "date_col_used" not in result["metrics"]
    assert "split_explanation" not in result["metrics"]


def test_train_single_model_records_split_strategy_chronological(tmp_path):
    import numpy as np

    from core.trainer import train_single_model

    # Sorted data: ascending revenue signal
    X = np.arange(20).reshape(-1, 1).astype(float)
    y = np.arange(20).astype(float)

    result = train_single_model(
        X, y, "linear_regression", "regression",
        tmp_path, "run-chrono",
        split_strategy="chronological",
        date_col_used="sale_date",
    )
    assert result["metrics"]["split_strategy"] == "chronological"
    assert result["metrics"]["date_col_used"] == "sale_date"
    assert "split_explanation" in result["metrics"]
    assert "sale_date" in result["metrics"]["split_explanation"]
    # Chronological split: train = first 80%, test = last 20%
    assert result["metrics"]["train_size"] == 16
    assert result["metrics"]["test_size"] == 4


def test_train_single_model_chronological_vs_random_sizes(tmp_path):
    """Both strategies produce same train/test sizes for ≥10 rows."""
    import numpy as np
    from core.trainer import train_single_model

    X = np.random.default_rng(0).random((20, 2))
    y = np.random.default_rng(0).random(20)

    r_random = train_single_model(X, y, "linear_regression", "regression", tmp_path, "run-a")
    r_chrono = train_single_model(
        X, y, "linear_regression", "regression", tmp_path, "run-b",
        split_strategy="chronological"
    )
    assert r_random["metrics"]["train_size"] == r_chrono["metrics"]["train_size"]
    assert r_random["metrics"]["test_size"] == r_chrono["metrics"]["test_size"]


# ---------------------------------------------------------------------------
# API: GET /api/models/{project_id}/split-strategy
# ---------------------------------------------------------------------------


def test_split_strategy_detects_date_column(client):
    project_id, _ = _setup_project(client, DATE_CSV, target="revenue")
    resp = client.get(f"/api/models/{project_id}/split-strategy")
    assert resp.status_code == 200
    data = resp.json()
    assert data["recommended"] == "chronological"
    assert data["date_col"] == "date"
    assert "explanation" in data
    assert "date" in data["explanation"]


def test_split_strategy_no_date_column(client):
    project_id, _ = _setup_project(client, NO_DATE_CSV, target="target")
    resp = client.get(f"/api/models/{project_id}/split-strategy")
    assert resp.status_code == 200
    data = resp.json()
    assert data["recommended"] == "random"
    assert data["date_col"] is None


def test_split_strategy_404_missing_project(client):
    resp = client.get("/api/models/nonexistent-id/split-strategy")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API: POST /api/models/{project_id}/train with split_strategy
# ---------------------------------------------------------------------------


def test_train_with_chronological_split(client):
    project_id, _ = _setup_project(client, DATE_CSV, target="revenue")
    resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "split_strategy": "chronological"},
    )
    assert resp.status_code == 202
    run_ids = resp.json()["model_run_ids"]
    assert len(run_ids) == 1


def test_train_with_random_split_explicit(client):
    project_id, _ = _setup_project(client, DATE_CSV, target="revenue")
    resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "split_strategy": "random"},
    )
    assert resp.status_code == 202


def test_train_with_null_split_strategy(client):
    """null split_strategy falls back to random — backwards compatible."""
    project_id, _ = _setup_project(client, DATE_CSV, target="revenue")
    resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "split_strategy": None},
    )
    assert resp.status_code == 202


def test_train_invalid_split_strategy(client):
    project_id, _ = _setup_project(client, DATE_CSV, target="revenue")
    resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "split_strategy": "weekly"},
    )
    assert resp.status_code == 400
    assert "split_strategy" in resp.json()["detail"]


def test_train_without_split_strategy_field(client):
    """Omitting split_strategy entirely is backwards compatible."""
    project_id, _ = _setup_project(client, DATE_CSV, target="revenue")
    resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert resp.status_code == 202


# ---------------------------------------------------------------------------
# chat: _TIME_SPLIT_PATTERNS
# ---------------------------------------------------------------------------


def test_time_split_patterns_match():
    from api.chat import _TIME_SPLIT_PATTERNS

    should_match = [
        "use time-based split",
        "use a chronological split",
        "enable temporal split",
        "switch to time-based splitting",
        "train on older data",
        "test on newer data",
        "use a time series split",
        "please use chronological split",
        "respect the date order",
        "split by date",
        "random split",
    ]
    for phrase in should_match:
        assert _TIME_SPLIT_PATTERNS.search(phrase), f"Pattern did not match: {phrase!r}"


def test_time_split_patterns_no_false_positives():
    from api.chat import _TIME_SPLIT_PATTERNS

    should_not_match = [
        "train a model to predict revenue",
        "what is the average sales",
        "show me the correlation heatmap",
        "deploy my model",
        "how does my model perform",
    ]
    for phrase in should_not_match:
        assert not _TIME_SPLIT_PATTERNS.search(phrase), f"Pattern falsely matched: {phrase!r}"
