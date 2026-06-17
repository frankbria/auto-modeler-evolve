"""Artifact janitor — orphan GC + upload retention (issue #4 / P2.1)."""

import os
import time
from uuid import uuid4

from sqlmodel import Session

import db
from core import storage
from core.janitor import collect_orphans, enforce_upload_retention, run_janitor
from models.batch_schedule import BatchSchedule
from models.deployment import Deployment
from models.model_run import ModelRun


def _mkdir_with_file(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "f").write_text("x")
    return d


def _seed_live_project(project_id="live-project"):
    """Create a live project + model_run + deployment + schedule with artifacts."""
    with Session(db.engine) as s:
        from models.project import Project

        if s.get(Project, project_id) is None:
            s.add(Project(id=project_id, owner_id="test-default-owner", name="live"))
        run = ModelRun(project_id=project_id, algorithm="rf")
        s.add(run)
        s.commit()
        s.refresh(run)
        dep = Deployment(
            model_run_id=run.id,
            project_id=project_id,
            endpoint_path="/api/predict/x",
            dashboard_url="/predict/x",
        )
        s.add(dep)
        s.commit()
        s.refresh(dep)
        sched = BatchSchedule(deployment_id=dep.id)
        s.add(sched)
        s.commit()
        s.refresh(sched)
        return {"project_id": project_id, "run_id": run.id, "schedule_id": sched.id}


async def test_collect_orphans_removes_orphan_dirs_keeps_live(client):
    live = _seed_live_project()

    # Live artifacts.
    live_upload = _mkdir_with_file(storage.project_upload_dir(live["project_id"]))
    live_models = _mkdir_with_file(storage.project_models_dir(live["project_id"]))

    # Orphan artifacts (project id not in DB).
    orphan_id = f"ghost-{uuid4()}"
    orphan_upload = _mkdir_with_file(storage.project_upload_dir(orphan_id))
    orphan_db = _mkdir_with_file(storage.project_db_uploads_dir(orphan_id))
    orphan_models = _mkdir_with_file(storage.project_models_dir(orphan_id))

    with Session(db.engine) as s:
        summary = collect_orphans(s)

    assert live_upload.exists()
    assert live_models.exists()
    assert not orphan_upload.exists()
    assert not orphan_db.exists()
    assert not orphan_models.exists()
    assert summary["removed_dirs"] == 3


async def test_collect_orphans_removes_orphan_pipelines_and_batch_outputs(client):
    live = _seed_live_project()

    live_pipeline = storage.deployment_pipeline_path(live["run_id"])
    _mkdir_with_file(live_pipeline.parent)
    live_pipeline.write_text("x")
    live_batch = storage.batch_outputs_dir() / f"{live['schedule_id']}_20240101.csv"
    _mkdir_with_file(live_batch.parent)
    live_batch.write_text("x")

    orphan_pipeline = storage.deployment_pipeline_path(str(uuid4()))
    orphan_pipeline.write_text("x")
    orphan_batch = storage.batch_outputs_dir() / f"{uuid4()}_20240101.csv"
    orphan_batch.write_text("x")

    with Session(db.engine) as s:
        summary = collect_orphans(s)

    assert live_pipeline.exists()
    assert live_batch.exists()
    assert not orphan_pipeline.exists()
    assert not orphan_batch.exists()
    assert summary["removed_files"] == 2


async def test_collect_orphans_respects_upload_grace(client):
    """A freshly created orphan upload survives when within the grace window."""
    orphan_id = f"ghost-{uuid4()}"
    fresh = _mkdir_with_file(storage.project_upload_dir(orphan_id))

    with Session(db.engine) as s:
        collect_orphans(s, upload_grace_days=1)

    assert fresh.exists()  # younger than the 1-day grace → kept


async def test_enforce_upload_retention_age_gated(client):
    orphan_id = f"ghost-{uuid4()}"
    old = _mkdir_with_file(storage.project_upload_dir(orphan_id))
    # Backdate to 10 days ago.
    ten_days = time.time() - 10 * 86400
    os.utime(old, (ten_days, ten_days))

    with Session(db.engine) as s:
        # Retention of 30 days → too young to reap.
        assert enforce_upload_retention(s, max_age_days=30) == 0
        assert old.exists()
        # Retention of 7 days → reaped.
        assert enforce_upload_retention(s, max_age_days=7) == 1
    assert not old.exists()


async def test_run_janitor_never_raises_and_reports(client):
    orphan_id = f"ghost-{uuid4()}"
    old = _mkdir_with_file(storage.project_upload_dir(orphan_id))
    os.utime(old, (time.time() - 5 * 86400, time.time() - 5 * 86400))

    with Session(db.engine) as s:
        summary = run_janitor(s, upload_grace_days=1)

    assert not old.exists()
    assert summary["removed_dirs"] >= 1
