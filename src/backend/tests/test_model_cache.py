"""Cached model/pipeline loaders for the prediction hot path (issue #19).

predict_single used to ``joblib.load`` the model + pipeline on every request.
The cached loaders key on (path, mtime): repeated loads of the same artifact
hit the cache; a redeploy (new path or rewritten file → new mtime) misses it.
"""

import os
import time

from core import deployer


def _count_joblib_loads(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        deployer.joblib, "load", lambda path: calls.append(str(path)) or "OBJ"
    )
    return calls


def test_repeated_load_hits_cache(tmp_path, monkeypatch):
    deployer._load_joblib_cached.cache_clear()
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"x")
    calls = _count_joblib_loads(monkeypatch)

    first = deployer.load_model_cached(artifact)
    second = deployer.load_model_cached(artifact)

    assert first == second == "OBJ"
    assert len(calls) == 1, "second load should hit the cache, not re-deserialize"


def test_mtime_change_invalidates_cache(tmp_path, monkeypatch):
    """A redeploy that rewrites the artifact (new mtime) forces a reload."""
    deployer._load_joblib_cached.cache_clear()
    artifact = tmp_path / "pipeline.joblib"
    artifact.write_bytes(b"x")
    calls = _count_joblib_loads(monkeypatch)

    deployer.load_pipeline_cached(artifact)
    # Simulate redeploy overwriting the file at a later mtime.
    future = time.time() + 10
    os.utime(artifact, (future, future))
    deployer.load_pipeline_cached(artifact)

    assert len(calls) == 2, "stale (path, mtime) key must miss after a rewrite"


def test_distinct_paths_load_separately(tmp_path, monkeypatch):
    """A redeploy to a new artifact path loads the new model (different key)."""
    deployer._load_joblib_cached.cache_clear()
    a = tmp_path / "run_a.joblib"
    b = tmp_path / "run_b.joblib"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    calls = _count_joblib_loads(monkeypatch)

    deployer.load_model_cached(a)
    deployer.load_model_cached(b)

    assert len(calls) == 2
