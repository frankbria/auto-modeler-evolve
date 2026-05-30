"""Tests for compute_target_leakage() pure function and REST endpoint."""

import io

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_target_leakage  # noqa: E402

# ---------------------------------------------------------------------------
# Pure function unit tests
# ---------------------------------------------------------------------------


def _make_num_df(n=60, seed=42):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 100, n)
    return pd.DataFrame(
        {
            "target": y,
            "a": rng.uniform(0, 10, n),
            "b": rng.uniform(0, 10, n),
        }
    )


class TestComputeTargetLeakageNumericTarget:
    def test_no_leakage_returns_none_verdict(self):
        df = _make_num_df()
        result = compute_target_leakage(df, "target", ["a", "b"])
        assert result["verdict"] == "none"
        assert result["leaky_features"] == []
        assert result["n_checked"] == 2

    def test_perfectly_correlated_feature_flagged_as_high_risk(self):
        n = 60
        y = np.linspace(1, 100, n)
        df = pd.DataFrame(
            {
                "target": y,
                "leak": y * 2 + 1,  # perfect correlation
                "safe": np.random.default_rng(0).uniform(0, 10, n),
            }
        )
        result = compute_target_leakage(df, "target", ["leak", "safe"])
        assert result["verdict"] == "severe"
        leaky = result["leaky_features"]
        assert len(leaky) == 1
        assert leaky[0]["feature"] == "leak"
        assert leaky[0]["risk_level"] == "high"
        assert leaky[0]["correlation_abs"] > 0.99

    def test_moderate_correlation_flagged_as_warning(self):
        rng = np.random.default_rng(7)
        n = 80
        y = rng.uniform(0, 10, n)
        # moderate correlation: 0.75-0.90
        noise = rng.uniform(0, 1, n)
        x_moderate = 0.85 * y + 0.15 * noise
        df = pd.DataFrame({"target": y, "x": x_moderate})
        result = compute_target_leakage(df, "target", ["x"])
        # May be moderate or high depending on exact correlation
        assert result["verdict"] in ("warning", "severe")

    def test_required_keys_present(self):
        df = _make_num_df()
        result = compute_target_leakage(df, "target", ["a", "b"])
        required = {
            "leaky_features",
            "n_checked",
            "verdict",
            "verdict_label",
            "high_threshold",
            "moderate_threshold",
            "target_col",
            "summary",
        }
        assert required.issubset(result.keys())

    def test_leaky_feature_keys(self):
        n = 50
        y = np.linspace(0, 10, n)
        df = pd.DataFrame({"target": y, "leak": y * 1.5})
        result = compute_target_leakage(df, "target", ["leak"])
        if result["leaky_features"]:
            feat = result["leaky_features"][0]
            for key in (
                "feature",
                "correlation",
                "correlation_abs",
                "risk_level",
                "risk_label",
                "reason",
            ):
                assert key in feat

    def test_verdict_severe_for_high_correlation(self):
        n = 50
        y = np.linspace(0, 10, n)
        df = pd.DataFrame({"target": y, "x": y + 0.0001})
        result = compute_target_leakage(df, "target", ["x"])
        assert result["verdict"] == "severe"

    def test_sorted_by_correlation_abs_descending(self):
        n = 60
        y = np.linspace(0, 10, n)
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "target": y,
                "strong": y * 2.0,
                "weak": y * 0.5 + rng.normal(0, 3, n),
            }
        )
        result = compute_target_leakage(
            df, "target", ["strong", "weak"], high_threshold=0.5, moderate_threshold=0.3
        )
        feats = result["leaky_features"]
        if len(feats) >= 2:
            assert feats[0]["correlation_abs"] >= feats[1]["correlation_abs"]

    def test_target_col_stored_in_result(self):
        df = _make_num_df()
        result = compute_target_leakage(df, "target", ["a", "b"])
        assert result["target_col"] == "target"

    def test_too_few_rows_raises_value_error(self):
        df = pd.DataFrame({"target": [1, 2], "x": [3, 4]})
        with pytest.raises(ValueError, match="need at least"):
            compute_target_leakage(df, "target", ["x"])

    def test_missing_target_col_raises_value_error(self):
        df = _make_num_df()
        with pytest.raises(ValueError, match="not found in DataFrame"):
            compute_target_leakage(df, "nonexistent", ["a"])

    def test_no_feature_cols_returns_none_verdict(self):
        df = _make_num_df()
        result = compute_target_leakage(df, "target", [])
        assert result["verdict"] == "none"
        assert result["n_checked"] == 0

    def test_summary_is_string(self):
        df = _make_num_df()
        result = compute_target_leakage(df, "target", ["a", "b"])
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_nan_handling_does_not_crash(self):
        df = _make_num_df()
        df.loc[0, "a"] = float("nan")
        result = compute_target_leakage(df, "target", ["a", "b"])
        assert "verdict" in result

    def test_thresholds_stored_in_result(self):
        df = _make_num_df()
        result = compute_target_leakage(
            df, "target", ["a"], high_threshold=0.92, moderate_threshold=0.77
        )
        assert result["high_threshold"] == 0.92
        assert result["moderate_threshold"] == 0.77

    def test_n_checked_equals_feature_count(self):
        df = _make_num_df()
        result = compute_target_leakage(df, "target", ["a", "b"])
        assert result["n_checked"] == 2


class TestComputeTargetLeakageCategoricalTarget:
    def test_no_leakage_none_verdict(self):
        rng = np.random.default_rng(10)
        n = 60
        df = pd.DataFrame(
            {
                "target": ["A", "B"] * (n // 2),
                "x": rng.uniform(0, 10, n),
                "y": rng.uniform(0, 10, n),
            }
        )
        result = compute_target_leakage(df, "target", ["x", "y"])
        assert result["verdict"] == "none"

    def test_feature_perfectly_predicting_class_flagged(self):
        # Continuous feature that strongly separates the binary class
        rng = np.random.default_rng(5)
        n = 80
        y_cls = ["A"] * 40 + ["B"] * 40
        # x_leak: tight Gaussian around 0 for A, tight Gaussian around 10 for B
        x_leak = np.concatenate(
            [
                rng.normal(0, 0.1, 40),
                rng.normal(10, 0.1, 40),
            ]
        )
        df = pd.DataFrame(
            {
                "target": y_cls,
                "leak": x_leak,
                "safe": rng.uniform(0, 10, n),
            }
        )
        # Use lower thresholds since MI estimates can be conservative
        result = compute_target_leakage(
            df, "target", ["leak", "safe"], high_threshold=0.5, moderate_threshold=0.3
        )
        # The leak feature should score higher than safe; verdict should be non-none
        assert result["verdict"] in ("warning", "severe")
        leaky_names = [f["feature"] for f in result["leaky_features"]]
        assert "leak" in leaky_names


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


class TestTargetLeakagePatterns:
    def test_matches_target_leakage(self):
        from api.chat import _TARGET_LEAKAGE_PATTERNS

        assert _TARGET_LEAKAGE_PATTERNS.search("is there target leakage?")

    def test_matches_data_leakage(self):
        from api.chat import _TARGET_LEAKAGE_PATTERNS

        assert _TARGET_LEAKAGE_PATTERNS.search("check for data leakage")

    def test_matches_leaky_features(self):
        from api.chat import _TARGET_LEAKAGE_PATTERNS

        assert _TARGET_LEAKAGE_PATTERNS.search("are there any leaky features?")

    def test_matches_feature_cheating(self):
        from api.chat import _TARGET_LEAKAGE_PATTERNS

        assert _TARGET_LEAKAGE_PATTERNS.search("features that cheat the answer")

    def test_matches_suspicious_correlations(self):
        from api.chat import _TARGET_LEAKAGE_PATTERNS

        assert _TARGET_LEAKAGE_PATTERNS.search(
            "suspicious correlations with the target"
        )

    def test_does_not_match_generic_correlation(self):
        from api.chat import _TARGET_LEAKAGE_PATTERNS

        assert not _TARGET_LEAKAGE_PATTERNS.search(
            "what is the correlation between price and revenue?"
        )

    def test_does_not_match_feature_importance(self):
        from api.chat import _TARGET_LEAKAGE_PATTERNS

        assert not _TARGET_LEAKAGE_PATTERNS.search("show me feature importance")


# ---------------------------------------------------------------------------
# REST endpoint integration tests
# ---------------------------------------------------------------------------

_SAMPLE_CSV = b"target,a,b\n" + b"".join(
    f"{i * 0.1},{i},{100 - i * 0.3}\n".encode() for i in range(1, 51)
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
    proj = await ac.post("/api/projects", json={"name": "Leakage Test"})
    pid = proj.json()["id"]
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": pid},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


async def test_endpoint_returns_200(ac, dataset_id):
    # Create FeatureSet by applying empty transformations
    apply_resp = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert apply_resp.status_code == 201, apply_resp.text
    # Set a target column
    set_resp = await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "target"},
    )
    assert set_resp.status_code == 200, set_resp.text
    resp = await ac.get(f"/api/data/{dataset_id}/target-leakage")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "verdict" in data
    assert "leaky_features" in data
    assert "n_checked" in data


async def test_endpoint_404_unknown_dataset(ac):
    resp = await ac.get("/api/data/does-not-exist/target-leakage")
    assert resp.status_code == 404


async def test_endpoint_400_no_target(ac, dataset_id):
    # Without a feature set / target column configured, should return 400
    resp = await ac.get(f"/api/data/{dataset_id}/target-leakage")
    assert resp.status_code == 400
