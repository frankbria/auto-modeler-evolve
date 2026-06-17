"""Artifact janitor — reclaim disk from artifacts with no referencing DB row.

The cascade delete (``core.cascade``) keeps the happy path clean, but artifacts
can still leak: a crash between writing a file and committing its row, a failed
delete, or rows removed out-of-band. Left unchecked, ``data/uploads`` &c. grow
until the shared VPS disk fills and the SQLite writer goes offline.

This module sweeps every artifact directory and removes anything whose owning
row no longer exists:
  - ``uploads/{project_id}``, ``db_uploads/{project_id}``, ``models/{project_id}``
    — directory name is a ``Project.id`` with no surviving Project row.
  - ``deployments/{model_run_id}_pipeline.joblib`` — no surviving ModelRun.
  - ``batch_outputs/{schedule_id}_*.csv`` — no surviving BatchSchedule.

``enforce_upload_retention`` additionally caps how long *orphaned* uploads (the
heaviest artifacts) may linger. Directory/file names are server-generated UUIDs
(no underscores), so the id is the filename prefix before the first ``_``.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from sqlmodel import Session, select

from core import storage
from models.batch_schedule import BatchSchedule
from models.model_run import ModelRun
from models.project import Project

logger = logging.getLogger(__name__)

_PIPELINE_SUFFIX = "_pipeline.joblib"


def _age_seconds(path: Path) -> float:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:  # pragma: no cover - filesystem race
        return 0.0


def _orphan_project_dirs(base: Path, live_project_ids: set[str]) -> list[Path]:
    if not base.exists():
        return []
    return [
        child
        for child in base.iterdir()
        if child.is_dir() and child.name not in live_project_ids
    ]


def enforce_upload_retention(session: Session, max_age_days: float) -> int:
    """Remove orphaned upload directories older than ``max_age_days``.

    Live projects' uploads are never touched. Returns the number removed.
    """
    live_project_ids = set(session.exec(select(Project.id)).all())
    cutoff = max(0.0, max_age_days) * 86400
    removed = 0
    for d in _orphan_project_dirs(storage.uploads_dir(), live_project_ids):
        if _age_seconds(d) >= cutoff:
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                removed += 1
    return removed


def collect_orphans(session: Session, *, upload_grace_days: float = 0) -> dict:
    """Remove every artifact whose owning DB row is gone.

    ``upload_grace_days`` spares orphaned *upload* dirs younger than the grace
    window — a brand-new upload whose Project row hasn't committed yet should
    survive a concurrent sweep. Other artifact types are removed immediately.
    Returns a summary of what was removed.
    """
    live_project_ids = set(session.exec(select(Project.id)).all())
    live_run_ids = set(session.exec(select(ModelRun.id)).all())
    live_schedule_ids = set(session.exec(select(BatchSchedule.id)).all())

    removed_dirs = 0

    # db_uploads + models: orphan dirs removed immediately.
    for base in (storage.db_uploads_dir(), storage.models_dir()):
        for d in _orphan_project_dirs(base, live_project_ids):
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                removed_dirs += 1

    # uploads: orphan dirs removed, respecting the grace window.
    grace = max(0.0, upload_grace_days) * 86400
    for d in _orphan_project_dirs(storage.uploads_dir(), live_project_ids):
        if _age_seconds(d) >= grace:
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                removed_dirs += 1

    removed_files = 0

    # Deployment pipelines: {model_run_id}_pipeline.joblib with no live ModelRun.
    dep_base = storage.deployments_dir()
    if dep_base.exists():
        for f in dep_base.iterdir():
            if f.is_file() and f.name.endswith(_PIPELINE_SUFFIX):
                run_id = f.name[: -len(_PIPELINE_SUFFIX)]
                if run_id not in live_run_ids:
                    f.unlink(missing_ok=True)
                    removed_files += 1

    # Batch outputs: {schedule_id}_{ts}.csv with no live BatchSchedule.
    batch_base = storage.batch_outputs_dir()
    if batch_base.exists():
        for f in batch_base.iterdir():
            if f.is_file() and "_" in f.name:
                schedule_id = f.name.split("_", 1)[0]
                if schedule_id not in live_schedule_ids:
                    f.unlink(missing_ok=True)
                    removed_files += 1

    return {"removed_dirs": removed_dirs, "removed_files": removed_files}


def run_janitor(session: Session, *, upload_grace_days: float = 1) -> dict:
    """Convenience entry point: GC all orphan artifacts. Best-effort, never raises.

    Called on app startup with a 1-day grace on uploads so concurrent in-flight
    uploads are not reaped. Returns the removal summary (empty on error).
    """
    try:
        summary = collect_orphans(session, upload_grace_days=upload_grace_days)
        if summary["removed_dirs"] or summary["removed_files"]:
            logger.info(
                "Janitor reclaimed %d dir(s) and %d file(s)",
                summary["removed_dirs"],
                summary["removed_files"],
            )
        return summary
    except Exception:  # noqa: BLE001 - janitor must never break startup
        logger.exception("Janitor sweep failed")
        return {"removed_dirs": 0, "removed_files": 0}
