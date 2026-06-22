"""Issue #14 — every artifact dir constant must resolve inside the per-test
tmp_path, never the real ``<backend>/data`` tree.

Before the autouse isolation fixture, the api modules captured these dirs at
import time (before ``DATA_DIR`` was set), so a fixture that forgot to patch them
let training/upload/deploy endpoints write into the real source tree (reproduced:
14k+ leaked upload dirs). This test fails loudly if any constant points outside
tmp_path again.
"""

from pathlib import Path

import pytest


def _real_data_root() -> Path:
    from core import storage

    return storage._DEFAULT_ROOT


@pytest.mark.parametrize(
    "module_path, attr",
    [
        ("api.models", "MODELS_DIR"),
        ("api.data", "UPLOAD_DIR"),
        ("api.data", "_DB_UPLOADS_DIR"),
        ("api.deploy", "DEPLOY_DIR"),
        ("api.templates", "UPLOAD_DIR"),
        ("core.scheduler", "BATCH_OUTPUT_DIR"),
    ],
)
def test_artifact_dir_constants_isolated(tmp_path, set_test_env, module_path, attr):
    import importlib

    module = importlib.import_module(module_path)
    value = Path(getattr(module, attr)).resolve()
    real = _real_data_root().resolve()

    assert real not in value.parents and value != real, (
        f"{module_path}.{attr} = {value} points inside the real data tree {real}; "
        "tests would pollute the source tree"
    )
    assert (
        tmp_path.resolve() in value.parents or value == tmp_path.resolve()
    ), f"{module_path}.{attr} = {value} is not under the per-test tmp_path {tmp_path}"


async def test_upload_lands_in_tmp(client, tmp_path, sample_csv_content):
    """An actual upload writes under tmp_path, leaving the real tree untouched."""
    real_uploads = _real_data_root() / "uploads"
    before = (
        set(p.name for p in real_uploads.iterdir()) if real_uploads.exists() else set()
    )

    proj = await client.post("/api/projects", json={"name": "Isolation Test"})
    project_id = proj.json()["id"]
    resp = await client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", sample_csv_content, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text

    after = (
        set(p.name for p in real_uploads.iterdir()) if real_uploads.exists() else set()
    )
    assert after == before, f"upload leaked into real tree: new dirs {after - before}"
