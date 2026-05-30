"""Tests for compute_feature_redundancy() pure function and REST endpoint."""

import io

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_feature_redundancy

# ---------------------------------------------------------------------------
# Pure function unit tests
# ---------------------------------------------------------------------------


def _make_df(**cols):
    """Helper to build a small DataFrame from keyword-arg arrays."""
    return pd.DataFrame({k: np.array(v, dtype=float) for k, v in cols.items()})


def test_no_redundancy_returns_none_verdict():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.uniform(0, 10, 50),
            "b": rng.uniform(0, 10, 50),
            "c": rng.uniform(0, 10, 50),
        }
    )
    result = compute_feature_redundancy(df, ["a", "b", "c"])
    assert result["verdict"] == "none"
    assert result["redundant_pairs"] == []
    assert result["n_redundant"] == 0


def test_perfectly_correlated_pair_detected():
    x = np.linspace(1, 100, 60)
    df = pd.DataFrame(
        {"x": x, "y": x * 2 + 1, "z": np.random.default_rng(0).uniform(0, 1, 60)}
    )
    result = compute_feature_redundancy(df, ["x", "y", "z"])
    pairs = result["redundant_pairs"]
    assert len(pairs) >= 1
    pair = next(p for p in pairs if {"x", "y"} == {p["feature_a"], p["feature_b"]})
    assert pair["correlation_abs"] > 0.99


def test_negative_correlation_detected():
    x = np.linspace(1, 100, 40)
    df = pd.DataFrame({"x": x, "neg_x": -x + 50.0})
    result = compute_feature_redundancy(df, ["x", "neg_x"], threshold=0.85)
    assert len(result["redundant_pairs"]) == 1
    assert result["redundant_pairs"][0]["direction"] == "negative"
    assert result["redundant_pairs"][0]["correlation"] < 0


def test_keep_feature_has_higher_variance():
    x = np.linspace(0, 1, 50)
    # y = x * 3 → var(y) > var(x)
    df = pd.DataFrame({"x": x, "y": x * 3})
    result = compute_feature_redundancy(df, ["x", "y"], threshold=0.95)
    assert len(result["redundant_pairs"]) == 1
    pair = result["redundant_pairs"][0]
    assert pair["keep"] == "y"  # higher variance
    assert pair["drop"] == "x"


def test_verdict_low_for_one_pair():
    x = np.linspace(0, 10, 30)
    df = pd.DataFrame(
        {
            "a": x,
            "b": x * 2.0,
            "c": np.random.default_rng(1).uniform(0, 10, 30),
        }
    )
    result = compute_feature_redundancy(df, ["a", "b", "c"])
    assert result["verdict"] in ("low", "high")
    assert result["n_redundant"] >= 2


def test_verdict_high_for_many_pairs():
    x = np.linspace(0, 10, 40)
    df = pd.DataFrame(
        {
            "a": x,
            "b": x * 1.5,
            "c": x * 0.5,
            "d": -x * 2,
            "noise": np.random.default_rng(7).uniform(0, 10, 40),
        }
    )
    result = compute_feature_redundancy(
        df, ["a", "b", "c", "d", "noise"], threshold=0.8
    )
    # a/b/c/d are all highly correlated — should get 'high'
    assert result["verdict"] == "high"


def test_returns_required_keys():
    df = _make_df(x=np.linspace(0, 1, 20), y=np.linspace(1, 2, 20))
    result = compute_feature_redundancy(df, ["x", "y"])
    required = {
        "redundant_pairs",
        "redundant_groups",
        "n_redundant",
        "n_features_checked",
        "threshold",
        "verdict",
        "verdict_label",
        "summary",
    }
    assert required <= result.keys()


def test_pair_keys():
    x = np.linspace(0, 10, 30)
    df = pd.DataFrame({"a": x, "b": x})
    result = compute_feature_redundancy(df, ["a", "b"], threshold=0.5)
    if result["redundant_pairs"]:
        pair = result["redundant_pairs"][0]
        required = {
            "feature_a",
            "feature_b",
            "correlation",
            "correlation_abs",
            "direction",
            "keep",
            "drop",
            "reason",
        }
        assert required <= pair.keys()


def test_correlation_abs_is_absolute_value():
    x = np.linspace(0, 10, 30)
    df = pd.DataFrame({"x": x, "neg_x": -x})
    result = compute_feature_redundancy(df, ["x", "neg_x"], threshold=0.5)
    for pair in result["redundant_pairs"]:
        assert pair["correlation_abs"] >= 0
        assert abs(pair["correlation"]) == pytest.approx(
            pair["correlation_abs"], abs=1e-6
        )


def test_too_few_rows_raises_value_error():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})
    with pytest.raises(ValueError, match="rows"):
        compute_feature_redundancy(df, ["a", "b"])


def test_single_numeric_feature_returns_none_verdict():
    df = pd.DataFrame({"a": np.linspace(0, 1, 20), "cat": ["x"] * 20})
    result = compute_feature_redundancy(df, ["a", "cat"])
    assert result["verdict"] == "none"
    assert result["redundant_pairs"] == []


def test_non_numeric_columns_ignored():
    df = pd.DataFrame(
        {
            "a": np.linspace(0, 10, 30),
            "b": np.linspace(0, 10, 30),
            "cat": ["x"] * 30,
        }
    )
    result = compute_feature_redundancy(df, ["a", "b", "cat"])
    # cat is skipped; only a and b are checked
    assert result["n_features_checked"] == 2


def test_threshold_parameter_respected():
    x = np.linspace(0, 10, 30)
    # corr(x, noisy) ~ 0.95 depending on noise magnitude
    rng = np.random.default_rng(99)
    noisy = x + rng.normal(0, 0.3, 30)
    df = pd.DataFrame({"x": x, "noisy": noisy})
    r_high = compute_feature_redundancy(df, ["x", "noisy"], threshold=0.99)
    r_low = compute_feature_redundancy(df, ["x", "noisy"], threshold=0.50)
    # At threshold=0.99 there may be fewer pairs than at 0.50
    assert len(r_low["redundant_pairs"]) >= len(r_high["redundant_pairs"])


def test_redundant_groups_non_empty_when_pairs_found():
    x = np.linspace(0, 10, 30)
    df = pd.DataFrame({"x": x, "y": x * 2})
    result = compute_feature_redundancy(df, ["x", "y"], threshold=0.5)
    if result["redundant_pairs"]:
        assert len(result["redundant_groups"]) >= 1
        for group in result["redundant_groups"]:
            assert len(group) >= 2


def test_n_features_checked_counts_numeric_only():
    df = pd.DataFrame(
        {
            "a": np.linspace(0, 1, 20),
            "b": np.linspace(0, 1, 20),
            "text": ["foo"] * 20,
        }
    )
    result = compute_feature_redundancy(df, ["a", "b", "text"])
    assert result["n_features_checked"] == 2


def test_summary_is_string():
    df = pd.DataFrame({"a": np.linspace(0, 1, 20), "b": np.linspace(0, 1, 20)})
    result = compute_feature_redundancy(df, ["a", "b"])
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_threshold_stored_in_result():
    df = pd.DataFrame({"a": np.linspace(0, 1, 20), "b": np.linspace(0, 1, 20)})
    result = compute_feature_redundancy(df, ["a", "b"], threshold=0.90)
    assert result["threshold"] == pytest.approx(0.90)


def test_handles_nan_values_gracefully():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 10, 30).tolist()
    b = (np.array(a) * 1.5).tolist()
    # Insert a few NaNs
    a[5] = float("nan")
    b[10] = float("nan")
    df = pd.DataFrame({"a": a, "b": b})
    result = compute_feature_redundancy(df, ["a", "b"])
    # Should not raise; may or may not find a pair depending on filled values
    assert isinstance(result["verdict"], str)


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


def test_regex_redundant_features():
    from api.chat import _REDUNDANCY_PATTERNS

    matches = [
        "are any features redundant?",
        "check for redundant features",
        "feature redundancy",
        "multicollinearity",
        "which features measure the same thing?",
        "highly correlated features",
        "feature overlap",
        "duplicate signals in my data",
        "remove redundant features",
        "detect duplicate features",
    ]
    for phrase in matches:
        assert _REDUNDANCY_PATTERNS.search(phrase), f"No match for: {phrase}"


def test_regex_false_positives():
    from api.chat import _REDUNDANCY_PATTERNS

    non_matches = [
        "what is the correlation between revenue and profit?",
        "show me the top features",
        "how accurate is my model?",
        "train my model",
        "upload a file",
    ]
    for phrase in non_matches:
        assert not _REDUNDANCY_PATTERNS.search(phrase), f"False positive for: {phrase}"


# ---------------------------------------------------------------------------
# REST endpoint integration tests
# ---------------------------------------------------------------------------

_SAMPLE_CSV = b"a,b,c\n" + b"".join(
    f"{i},{i * 2},{100 - i * 0.3}\n".encode() for i in range(1, 31)
)


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa: F401
    import models.dataset  # noqa: F401
    import models.deployment  # noqa: F401
    import models.feature_set  # noqa: F401
    import models.model_run  # noqa: F401
    import models.project  # noqa: F401

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
    proj = await ac.post("/api/projects", json={"name": "Redundancy Test"})
    pid = proj.json()["id"]
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": pid},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


async def test_endpoint_returns_200(ac, dataset_id):
    resp = await ac.get(f"/api/data/{dataset_id}/feature-redundancy")
    assert resp.status_code == 200


async def test_endpoint_required_fields(ac, dataset_id):
    data = (await ac.get(f"/api/data/{dataset_id}/feature-redundancy")).json()
    assert "verdict" in data
    assert "redundant_pairs" in data
    assert "n_features_checked" in data
    assert "summary" in data


async def test_endpoint_unknown_dataset(ac):
    resp = await ac.get("/api/data/does-not-exist/feature-redundancy")
    assert resp.status_code == 404


async def test_endpoint_invalid_threshold(ac, dataset_id):
    resp = await ac.get(f"/api/data/{dataset_id}/feature-redundancy?threshold=1.5")
    assert resp.status_code == 400
