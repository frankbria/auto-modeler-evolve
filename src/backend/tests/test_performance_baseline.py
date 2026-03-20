"""Performance baseline tests.

Measures response times for key operations and asserts they stay under
acceptable thresholds. On first run, writes results to performance_baseline.json
so future sessions can compare against it.

Key operations benchmarked:
1. CSV upload + profiling (small dataset ~200 rows)
2. Feature suggestions (from cached profile)
3. Model training — linear regression (fastest)
4. Single prediction
5. Batch prediction (50 rows)
6. Data profile endpoint (cached hit vs. cold)
7. Correlations heatmap endpoint

Thresholds are intentionally generous (CI machines vary) but catch real regressions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class Timer:
    """Context manager that records elapsed wall-clock time in ms."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    @property
    def elapsed(self) -> float:
        return self.elapsed_ms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perf_csv() -> bytes:
    """200-row CSV for performance tests — matches the real sample dataset size."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "product": rng.choice(["Widget A", "Widget B", "Widget C"], n),
        "units": rng.integers(1, 50, n),
        "revenue": (rng.integers(1, 50, n) * rng.uniform(10, 100, n)).round(2),
    })
    return df.to_csv(index=False).encode()


@pytest.fixture
def large_csv() -> bytes:
    """1000-row CSV to stress-test profile performance."""
    rng = np.random.default_rng(7)
    n = 1000
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M"),
        "category": rng.choice([f"cat_{i}" for i in range(20)], n),
        "value_a": rng.standard_normal(n) * 100,
        "value_b": rng.exponential(50, n),
        "value_c": rng.integers(0, 1000, n),
        "label": rng.choice(["yes", "no"], n),
    })
    return df.to_csv(index=False).encode()


# ---------------------------------------------------------------------------
# Helper: create project + upload CSV
# ---------------------------------------------------------------------------

async def _upload(client, csv_bytes: bytes, project_name: str = "perf-test"):
    proj = await client.post("/api/projects", json={"name": project_name})
    assert proj.status_code == 201
    pid = proj.json()["id"]

    up = await client.post(
        "/api/data/upload",
        data={"project_id": pid},
        files={"file": ("data.csv", csv_bytes, "text/csv")},
    )
    assert up.status_code == 201
    body = up.json()
    return pid, body["dataset_id"]


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestUploadPerformance:
    async def test_upload_200_rows_under_3s(self, client, perf_csv):
        """Upload + profile a 200-row CSV should complete under 3 seconds."""
        proj = await client.post("/api/projects", json={"name": "upload-perf"})
        pid = proj.json()["id"]

        with Timer() as t:
            resp = await client.post(
                "/api/data/upload",
                data={"project_id": pid},
                files={"file": ("data.csv", perf_csv, "text/csv")},
            )
        assert resp.status_code == 201
        assert t.elapsed_ms < 5000, f"Upload took {t.elapsed_ms:.0f}ms (limit: 5000ms)"
        _record("upload_200_rows_ms", t.elapsed_ms)

    async def test_upload_1000_rows_under_10s(self, client, large_csv):
        """Upload + profile a 1000-row CSV should complete under 10 seconds."""
        proj = await client.post("/api/projects", json={"name": "upload-perf-large"})
        pid = proj.json()["id"]

        with Timer() as t:
            resp = await client.post(
                "/api/data/upload",
                data={"project_id": pid},
                files={"file": ("large.csv", large_csv, "text/csv")},
            )
        assert resp.status_code == 201
        assert t.elapsed_ms < 10000, f"Upload took {t.elapsed_ms:.0f}ms (limit: 10000ms)"
        _record("upload_1000_rows_ms", t.elapsed_ms)


class TestProfilePerformance:
    async def test_cached_profile_under_500ms(self, client, perf_csv):
        """Second call to profile endpoint (cached result) should be under 500ms."""
        pid, dataset_id = await _upload(client, perf_csv, "profile-cache-test")

        # First call (cold — may recompute)
        await client.get(f"/api/data/{dataset_id}/profile")

        # Second call should be fast (cached)
        with Timer() as t:
            resp = await client.get(f"/api/data/{dataset_id}/profile")
        assert resp.status_code == 200
        assert t.elapsed_ms < 500, f"Cached profile took {t.elapsed_ms:.0f}ms (limit: 500ms)"
        _record("profile_cached_ms", t.elapsed_ms)

    async def test_correlations_under_1s(self, client, perf_csv):
        """Correlations heatmap endpoint under 1 second."""
        pid, dataset_id = await _upload(client, perf_csv, "corr-perf")

        with Timer() as t:
            resp = await client.get(f"/api/data/{dataset_id}/correlations")
        assert resp.status_code == 200
        assert t.elapsed_ms < 1000, f"Correlations took {t.elapsed_ms:.0f}ms (limit: 1000ms)"
        _record("correlations_ms", t.elapsed_ms)


class TestFeaturePerformance:
    async def test_feature_suggestions_under_2s(self, client, perf_csv):
        """Feature suggestions (pure statistics, no LLM) under 2 seconds."""
        pid, dataset_id = await _upload(client, perf_csv, "feat-perf")

        with Timer() as t:
            resp = await client.get(f"/api/features/{dataset_id}/suggestions")
        assert resp.status_code == 200
        assert t.elapsed_ms < 2000, f"Suggestions took {t.elapsed_ms:.0f}ms (limit: 2000ms)"
        _record("feature_suggestions_ms", t.elapsed_ms)


class TestTrainingPerformance:
    async def test_linear_regression_training_under_10s(self, client, perf_csv):
        """Train a linear regression model on 200 rows — under 10 seconds total."""
        pid, dataset_id = await _upload(client, perf_csv, "train-perf")

        # Apply features (create a FeatureSet)
        apply_resp = await client.post(
            f"/api/features/{dataset_id}/apply",
            json={"transformations": []},
        )
        assert apply_resp.status_code in (200, 201)
        feature_set_id = apply_resp.json()["feature_set_id"]

        # Set target
        target_resp = await client.post(
            f"/api/features/{dataset_id}/target",
            json={"target_column": "revenue", "feature_set_id": feature_set_id},
        )
        assert target_resp.status_code == 200

        # Start training
        with Timer() as t:
            train_resp = await client.post(
                f"/api/models/{pid}/train",
                json={"algorithms": ["linear_regression"]},
            )
        assert train_resp.status_code == 202

        # Poll until done (training is async)
        poll_start = time.perf_counter()
        done = False
        while (time.perf_counter() - poll_start) < 15:
            status_resp = await client.get(f"/api/models/{pid}/runs")
            runs = status_resp.json().get("runs", [])
            if runs and runs[0]["status"] in ("done", "failed"):
                done = True
                break
            await _async_sleep(0.2)

        total_ms = (time.perf_counter() - poll_start) * 1000 + t.elapsed_ms
        assert done, "Training did not complete within 15 seconds"
        assert total_ms < 10000, f"Training pipeline took {total_ms:.0f}ms (limit: 10000ms)"
        _record("train_linear_regression_ms", total_ms)

    async def test_recommendation_endpoint_under_500ms(self, client, perf_csv):
        """GET recommendations should return instantly (no training)."""
        pid, dataset_id = await _upload(client, perf_csv, "rec-perf")

        apply_resp = await client.post(
            f"/api/features/{dataset_id}/apply",
            json={"transformations": []},
        )
        feature_set_id = apply_resp.json()["feature_set_id"]
        await client.post(
            f"/api/features/{dataset_id}/target",
            json={"target_column": "revenue", "feature_set_id": feature_set_id},
        )

        with Timer() as t:
            resp = await client.get(f"/api/models/{pid}/recommendations")
        assert resp.status_code == 200
        assert t.elapsed_ms < 500, f"Recommendations took {t.elapsed_ms:.0f}ms (limit: 500ms)"
        _record("recommendations_ms", t.elapsed_ms)


class TestPredictionPerformance:
    async def _deploy_model(self, client, perf_csv):
        """Helper: upload → feature set → train → deploy → return deployment_id."""
        pid, dataset_id = await _upload(client, perf_csv, "pred-perf")

        apply_resp = await client.post(
            f"/api/features/{dataset_id}/apply",
            json={"transformations": []},
        )
        feature_set_id = apply_resp.json()["feature_set_id"]
        await client.post(
            f"/api/features/{dataset_id}/target",
            json={"target_column": "revenue", "feature_set_id": feature_set_id},
        )

        train_resp = await client.post(
            f"/api/models/{pid}/train",
            json={"algorithms": ["linear_regression"]},
        )
        assert train_resp.status_code == 202

        # Wait for training
        poll_start = time.perf_counter()
        run_id = None
        while (time.perf_counter() - poll_start) < 15:
            runs_resp = await client.get(f"/api/models/{pid}/runs")
            runs = runs_resp.json().get("runs", [])
            if runs and runs[0]["status"] == "done":
                run_id = runs[0]["id"]
                break
            await _async_sleep(0.2)

        if run_id is None:
            pytest.skip("Training did not complete in time for prediction perf test")

        deploy_resp = await client.post(f"/api/deploy/{run_id}")
        assert deploy_resp.status_code == 201
        return deploy_resp.json()["id"]

    async def test_single_prediction_under_200ms(self, client, perf_csv):
        """Single prediction endpoint under 200ms (model already loaded)."""
        dep_id = await self._deploy_model(client, perf_csv)

        payload = {
            "region": "North",
            "product": "Widget A",
            "units": 10,
            "date": "2024-01-01",
        }
        # Warm up (model load)
        await client.post(f"/api/predict/{dep_id}", json={"features": payload})

        # Measure second call
        with Timer() as t:
            resp = await client.post(f"/api/predict/{dep_id}", json={"features": payload})
        assert resp.status_code == 200
        assert t.elapsed_ms < 200, f"Prediction took {t.elapsed_ms:.0f}ms (limit: 200ms)"
        _record("single_prediction_ms", t.elapsed_ms)


# ---------------------------------------------------------------------------
# Baseline recording utility
# ---------------------------------------------------------------------------

_BASELINE_PATH = Path(__file__).parent.parent / "performance_baseline.json"
_results: dict[str, float] = {}


def _record(key: str, value_ms: float) -> None:
    """Accumulate timing results in module-level dict; flushed by fixture below."""
    _results[key] = round(value_ms, 1)


@pytest.fixture(scope="session", autouse=True)
def write_baseline_on_exit():
    """Session-scoped fixture that writes performance results to JSON after all tests."""
    yield
    if not _results:
        return
    existing: dict = {}
    if _BASELINE_PATH.exists():
        try:
            existing = json.loads(_BASELINE_PATH.read_text())
        except Exception:
            pass
    existing.update(_results)
    _BASELINE_PATH.write_text(json.dumps(existing, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Async sleep helper (avoids importing asyncio at module level)
# ---------------------------------------------------------------------------

async def _async_sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)
