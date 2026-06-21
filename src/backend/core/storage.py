"""Single source of truth for on-disk artifact paths.

Every artifact AutoModeler writes lives under one data root. Historically each
api module hardcoded ``Path(__file__).parent.parent / "data" / ...`` at import
time. That layout is duplicated here in one place so the cascade-delete and the
artifact janitor remove files from *exactly* the directories the writers use.

The root resolves at **call time** from the ``DATA_DIR`` environment variable,
falling back to ``<backend>/data`` — identical to the historical constants when
``DATA_DIR`` is unset (i.e. in production). Tests set ``DATA_DIR`` to a temp dir
so artifact I/O stays isolated from the real source tree.

Layout (relative to the data root):
    uploads/{project_id}/...                 — dataset CSVs
    db_uploads/{project_id}/...              — extract-db source databases
    models/{project_id}/{model_run_id}.joblib — trained models
    deployments/{model_run_id}_pipeline.joblib — serialized prediction pipelines
    batch_outputs/{schedule_id}_{ts}.csv     — scheduled batch prediction results
"""

from __future__ import annotations

import os
from pathlib import Path

# ``<backend>/data`` — matches the legacy api-module constants exactly when
# ``DATA_DIR`` is unset. ``core/storage.py`` lives in ``<backend>/core``, so two
# parents up is ``<backend>``.
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data"


def data_root() -> Path:
    """Return the base data directory (``DATA_DIR`` env override, else default)."""
    env = os.environ.get("DATA_DIR")
    return Path(env) if env else _DEFAULT_ROOT


def uploads_dir() -> Path:
    return data_root() / "uploads"


def db_uploads_dir() -> Path:
    return data_root() / "db_uploads"


def models_dir() -> Path:
    return data_root() / "models"


def deployments_dir() -> Path:
    return data_root() / "deployments"


def batch_outputs_dir() -> Path:
    return data_root() / "batch_outputs"


def logs_dir() -> Path:
    return data_root() / "logs"


# --- Per-entity helpers -----------------------------------------------------


def project_upload_dir(project_id: str) -> Path:
    return uploads_dir() / project_id


def project_db_uploads_dir(project_id: str) -> Path:
    return db_uploads_dir() / project_id


def project_models_dir(project_id: str) -> Path:
    return models_dir() / project_id


def deployment_pipeline_path(model_run_id: str) -> Path:
    return deployments_dir() / f"{model_run_id}_pipeline.joblib"


def project_artifact_dirs(project_id: str) -> list[Path]:
    """Project-scoped artifact directories (named by ``project_id``)."""
    return [
        project_upload_dir(project_id),
        project_db_uploads_dir(project_id),
        project_models_dir(project_id),
    ]
