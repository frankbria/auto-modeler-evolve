"""core.storage must stay in lockstep with the api-module write paths.

The cascade/janitor remove artifacts via ``core.storage``; the api modules write
them via their own ``*_DIR`` constants. If the two layouts ever drift, deletes
would silently miss files. These tests pin the layout so a divergence fails loudly.
"""

import importlib

from core import storage


def test_storage_layout_matches_api_constants(monkeypatch):
    """With DATA_DIR unset, storage resolves to the same dirs the writers use.

    The api modules expose their write dirs as module-level ``*_DIR`` constants.
    Many other tests reassign those globals to a ``tmp_path`` in-process (without
    restoring them), so the *live* attribute can be polluted by test-execution
    order. Reload each module first to read its **declared** constant value — the
    real invariant — instead of whatever a previous test left behind.
    """
    monkeypatch.delenv("DATA_DIR", raising=False)

    api_data = importlib.reload(importlib.import_module("api.data"))
    api_models = importlib.reload(importlib.import_module("api.models"))
    api_deploy = importlib.reload(importlib.import_module("api.deploy"))
    scheduler = importlib.reload(importlib.import_module("core.scheduler"))

    assert storage.uploads_dir().resolve() == api_data.UPLOAD_DIR.resolve()
    assert storage.db_uploads_dir().resolve() == api_data._DB_UPLOADS_DIR.resolve()
    assert storage.models_dir().resolve() == api_models.MODELS_DIR.resolve()
    assert storage.deployments_dir().resolve() == api_deploy.DEPLOY_DIR.resolve()
    assert storage.batch_outputs_dir().resolve() == scheduler.BATCH_OUTPUT_DIR.resolve()
    assert storage.data_root().name == "data"


def test_pipeline_path_matches_deploy_naming():
    """Pipeline filename convention matches api/deploy.py's ``{run.id}_pipeline.joblib``."""
    run_id = "abc123"
    assert storage.deployment_pipeline_path(run_id).name == f"{run_id}_pipeline.joblib"
