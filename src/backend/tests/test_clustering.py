"""Tests for K-means clustering: compute_clusters() + /api/data/{id}/clusters endpoint."""

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import (
    _build_cluster_description,
    _build_cluster_summary,
    compute_clusters,
)

# ---------------------------------------------------------------------------
# Unit tests — compute_clusters
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    """Simple 2-cluster dataset: two clearly separated groups."""
    return pd.DataFrame(
        {
            "x": [1.0, 1.1, 1.2, 10.0, 10.1, 10.2, 1.0, 10.0, 1.1, 10.1],
            "y": [1.0, 1.1, 0.9, 10.0, 9.9, 10.1, 1.2, 10.2, 0.8, 9.8],
        }
    )


@pytest.fixture
def multifeature_df():
    """Dataset with numeric + categorical columns."""
    return pd.DataFrame(
        {
            "revenue": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
                        700.0, 800.0, 900.0, 1000.0, 150.0, 250.0],
            "units": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 15, 25],
            "cost": [50.0, 80.0, 110.0, 140.0, 180.0, 210.0,
                     250.0, 280.0, 310.0, 340.0, 60.0, 90.0],
            "region": ["East"] * 6 + ["West"] * 6,
        }
    )


def test_compute_clusters_basic(simple_df):
    result = compute_clusters(simple_df)
    assert "error" not in result
    assert result["n_clusters"] >= 2
    assert len(result["clusters"]) == result["n_clusters"]
    assert result["features_used"] == ["x", "y"]
    assert result["rows_clustered"] == 10
    assert result["auto_k"] is True
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 10


def test_compute_clusters_finds_two_groups(simple_df):
    """The clearly separated dataset should yield exactly 2 clusters."""
    result = compute_clusters(simple_df, n_clusters=2)
    assert result["n_clusters"] == 2
    assert result["auto_k"] is False
    # Each cluster should have at least 1 row
    for c in result["clusters"]:
        assert c["size"] >= 1
    # Sizes should sum to total rows
    total_size = sum(c["size"] for c in result["clusters"])
    assert total_size == 10


def test_compute_clusters_explicit_k(multifeature_df):
    result = compute_clusters(multifeature_df, n_clusters=3)
    assert result["n_clusters"] == 3
    assert result["auto_k"] is False
    assert len(result["clusters"]) == 3


def test_compute_clusters_feature_selection(multifeature_df):
    """Clustering on a subset of features."""
    result = compute_clusters(multifeature_df, feature_cols=["revenue", "units"])
    assert "error" not in result
    assert result["features_used"] == ["revenue", "units"]
    # "cost" and "region" should not be in features_used
    assert "cost" not in result["features_used"]
    assert "region" not in result["features_used"]


def test_compute_clusters_ignores_categorical(multifeature_df):
    """Non-numeric columns should not appear in features_used."""
    result = compute_clusters(multifeature_df)
    assert "region" not in result["features_used"]


def test_compute_clusters_cluster_structure(simple_df):
    """Each cluster should have the required keys."""
    result = compute_clusters(simple_df, n_clusters=2)
    for c in result["clusters"]:
        assert "cluster_id" in c
        assert "size" in c
        assert "size_pct" in c
        assert "centroid" in c
        assert "distinguishing" in c
        assert "description" in c
        assert isinstance(c["description"], str)
        assert isinstance(c["centroid"], dict)


def test_compute_clusters_size_pct_sums_to_100(simple_df):
    result = compute_clusters(simple_df, n_clusters=2)
    total_pct = sum(c["size_pct"] for c in result["clusters"])
    assert abs(total_pct - 100.0) < 1.0  # rounding tolerance


def test_compute_clusters_distinguishing_features(multifeature_df):
    """Distinguishing features should have the expected keys."""
    result = compute_clusters(multifeature_df, n_clusters=2)
    for c in result["clusters"]:
        for d in c["distinguishing"]:
            assert "feature" in d
            assert "cluster_mean" in d
            assert "global_mean" in d
            assert d["direction"] in ("above", "below")
            assert d["magnitude"] >= 0


def test_compute_clusters_no_numeric_cols():
    """DataFrame with no numeric columns should return an error."""
    df = pd.DataFrame({"cat": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]})
    result = compute_clusters(df)
    assert "error" in result


def test_compute_clusters_too_few_rows():
    """DataFrame with fewer than MIN_ROWS_FOR_CLUSTERING should return error."""
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    result = compute_clusters(df)
    assert "error" in result


def test_compute_clusters_with_missing_values():
    """Missing values should be handled by dropping rows."""
    # 14 total rows, 3 with NaN → 11 valid (above MIN_ROWS_FOR_CLUSTERING=10)
    df = pd.DataFrame(
        {
            "x": [1.0, None, 1.1, 10.0, 10.1, 10.2, 1.0, 10.0, 1.1, 10.1,
                  1.2, 10.3, None, None],
            "y": [1.0, 1.1, 0.9, 10.0, 9.9, 10.1, 1.2, 10.2, 0.8, 9.8,
                  1.3, 9.7, None, 5.0],
        }
    )
    result = compute_clusters(df)
    assert "error" not in result
    # Rows with NaN should be dropped
    assert result["rows_clustered"] < len(df)
    assert result["rows_clustered"] == 11  # 14 - 3 NaN rows


def test_compute_clusters_invalid_feature_col(simple_df):
    """Requesting a non-existent column should fall back to valid numeric columns."""
    result = compute_clusters(simple_df, feature_cols=["x", "nonexistent"])
    assert "error" not in result
    assert "nonexistent" not in result["features_used"]


def test_compute_clusters_k_clamped_to_max():
    """n_clusters > 8 should be clamped to 8."""
    df = pd.DataFrame(
        {
            "x": list(range(50)),
            "y": list(range(50)),
        }
    )
    result = compute_clusters(df, n_clusters=20)
    assert result["n_clusters"] <= 8


def test_compute_clusters_sorted_by_size(multifeature_df):
    """Clusters should be sorted by size descending."""
    result = compute_clusters(multifeature_df, n_clusters=2)
    sizes = [c["size"] for c in result["clusters"]]
    assert sizes == sorted(sizes, reverse=True)


def test_build_cluster_description_with_features():
    distinguishing = [
        {"feature": "revenue", "cluster_mean": 800.0, "global_mean": 400.0,
         "direction": "above", "magnitude": 1.2},
        {"feature": "cost", "cluster_mean": 50.0, "global_mean": 200.0,
         "direction": "below", "magnitude": 0.8},
    ]
    desc = _build_cluster_description(0, 5, 10, distinguishing)
    assert "Group 1" in desc
    assert "50%" in desc
    assert "revenue" in desc


def test_build_cluster_description_no_features():
    desc = _build_cluster_description(1, 3, 10, [])
    assert "Group 2" in desc
    assert "3" in desc


def test_build_cluster_summary():
    clusters = [
        {"size": 7, "size_pct": 70.0},
        {"size": 3, "size_pct": 30.0},
    ]
    summary = _build_cluster_summary(clusters, ["revenue", "cost"], 10, 10)
    assert "2" in summary  # 2 groups
    assert "10" in summary  # 10 rows


# ---------------------------------------------------------------------------
# API endpoint tests — GET /api/data/{id}/clusters
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"x,y,category\n"
    b"1.0,1.0,A\n"
    b"1.1,1.1,A\n"
    b"1.2,0.9,A\n"
    b"10.0,10.0,B\n"
    b"10.1,9.9,B\n"
    b"10.2,10.1,B\n"
    b"1.0,1.2,A\n"
    b"10.0,10.2,B\n"
    b"1.1,0.8,A\n"
    b"10.1,9.8,B\n"
)


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def dataset_id(ac):
    proj_resp = await ac.post("/api/projects", json={"name": "Cluster Test"})
    project_id = proj_resp.json()["id"]
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("cluster.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


async def test_clusters_endpoint_auto_k(ac, dataset_id):
    resp = await ac.get(f"/api/data/{dataset_id}/clusters")
    assert resp.status_code == 200
    data = resp.json()
    assert "n_clusters" in data
    assert "clusters" in data
    assert len(data["clusters"]) == data["n_clusters"]
    assert data["auto_k"] is True


async def test_clusters_endpoint_explicit_k(ac, dataset_id):
    resp = await ac.get(f"/api/data/{dataset_id}/clusters", params={"n_clusters": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_clusters"] == 2
    assert data["auto_k"] is False
    assert len(data["clusters"]) == 2


async def test_clusters_endpoint_feature_selection(ac, dataset_id):
    resp = await ac.get(f"/api/data/{dataset_id}/clusters", params={"features": "x"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["features_used"] == ["x"]


async def test_clusters_endpoint_invalid_feature(ac, dataset_id):
    resp = await ac.get(
        f"/api/data/{dataset_id}/clusters", params={"features": "nonexistent"}
    )
    assert resp.status_code == 400


async def test_clusters_endpoint_invalid_k_too_low(ac, dataset_id):
    resp = await ac.get(f"/api/data/{dataset_id}/clusters", params={"n_clusters": 1})
    assert resp.status_code == 400


async def test_clusters_endpoint_invalid_k_too_high(ac, dataset_id):
    resp = await ac.get(f"/api/data/{dataset_id}/clusters", params={"n_clusters": 9})
    assert resp.status_code == 400


async def test_clusters_endpoint_dataset_not_found(ac):
    resp = await ac.get("/api/data/00000000-0000-0000-0000-000000000000/clusters")
    assert resp.status_code == 404


async def test_clusters_endpoint_response_shape(ac, dataset_id):
    resp = await ac.get(f"/api/data/{dataset_id}/clusters", params={"n_clusters": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "features_used" in data
    assert "rows_clustered" in data
    assert "summary" in data
    for c in data["clusters"]:
        assert "cluster_id" in c
        assert "size" in c
        assert "size_pct" in c
        assert "centroid" in c
        assert "distinguishing" in c
        assert "description" in c


# ---------------------------------------------------------------------------
# Chat pattern tests — _CLUSTER_PATTERNS
# ---------------------------------------------------------------------------

from api.chat import _CLUSTER_PATTERNS  # noqa: E402


@pytest.mark.parametrize(
    "message",
    [
        "cluster my data",
        "cluster my customers",
        "find natural groups",
        "segment my customers",
        "segment my data",
        "what groups exist in my data",
        "customer segmentation",
        "k-means analysis",
        "identify clusters in the dataset",
        "discover natural segments",
    ],
)
def test_cluster_patterns_match(message):
    assert _CLUSTER_PATTERNS.search(message), f"Pattern should match: {message!r}"


@pytest.mark.parametrize(
    "message",
    [
        "what is the revenue total",
        "show me the correlation matrix",
        "train a model",
        "filter to East region",
    ],
)
def test_cluster_patterns_no_match(message):
    assert not _CLUSTER_PATTERNS.search(message), f"Pattern should NOT match: {message!r}"
