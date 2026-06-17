"""Transactional cascade delete for a Project and all its descendants.

``delete_project`` historically removed only the ``Project`` row, orphaning ~23
child tables (4 levels deep) and leaking every on-disk artifact. This module
deletes the whole ownership subtree rooted at a project in **one transaction /
single commit** (zero orphan rows), then removes the project's artifact
directories and files.

Ownership graph (rooted at Project):
  - direct (project_id):  Dataset, ModelRun, Deployment, Conversation, AnalysisTemplate
  - via dataset_id:       FeatureSet, DatasetFilter
  - via deployment_id:    PredictionLog, FeedbackRecord, DeploymentChangelog,
                          DeploymentPreset, DashboardFieldConfig, InputValidationRule,
                          PredictionAlertRule, SavedScenario, WebhookConfig, WebhookEvent,
                          WeeklyDigestConfig, GoalSeekRecord, BatchSchedule, BatchJobRun,
                          DeploymentVersion
  - via deployment.id:    ABTest (champion_id / challenger_id)

Rows are deleted leaf-first so the result is consistent regardless of whether
SQLite foreign-key enforcement is on. Artifact removal happens **after** a
successful commit (the filesystem is not part of the DB transaction); any file
that cannot be removed is reaped later by ``core.janitor``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from sqlalchemy import delete, or_
from sqlmodel import Session, select

from core import storage
from core.path_safety import UnsafePathError, assert_within
from models.ab_test import ABTest
from models.analysis_template import AnalysisTemplate
from models.batch_schedule import BatchJobRun, BatchSchedule
from models.conversation import Conversation
from models.dashboard_field_config import DashboardFieldConfig
from models.dataset import Dataset
from models.dataset_filter import DatasetFilter
from models.deployment import Deployment
from models.deployment_changelog import DeploymentChangelog
from models.deployment_preset import DeploymentPreset
from models.deployment_version import DeploymentVersion
from models.feature_set import FeatureSet
from models.feedback_record import FeedbackRecord
from models.goal_seek_record import GoalSeekRecord
from models.input_validation_rule import InputValidationRule
from models.model_run import ModelRun
from models.prediction_alert_rule import PredictionAlertRule
from models.prediction_log import PredictionLog
from models.project import Project
from models.saved_scenario import SavedScenario
from models.webhook_config import WebhookConfig
from models.webhook_event import WebhookEvent
from models.weekly_digest_config import WeeklyDigestConfig

logger = logging.getLogger(__name__)

# Child tables keyed by a ``deployment_id`` column, deleted before their Deployment.
_DEPLOYMENT_CHILDREN = (
    PredictionLog,
    FeedbackRecord,
    DeploymentChangelog,
    DeploymentPreset,
    DashboardFieldConfig,
    InputValidationRule,
    PredictionAlertRule,
    SavedScenario,
    WebhookConfig,
    WebhookEvent,
    WeeklyDigestConfig,
    GoalSeekRecord,
    BatchSchedule,
    BatchJobRun,
    DeploymentVersion,
)


def _safe_unlink(path_str: str | None, base_dir: Path) -> bool:
    """Remove a stored file path if it resolves inside ``base_dir``. Best-effort.

    Returns True if a file was removed. A path that escapes ``base_dir`` (a
    poisoned/legacy absolute path) is skipped rather than followed.
    """
    if not path_str:
        return False
    try:
        confined = assert_within(base_dir, Path(path_str))
    except UnsafePathError:
        logger.warning("Skipping artifact outside data dir: %s", path_str)
        return False
    try:
        if confined.is_file():
            confined.unlink()
            return True
    except OSError as exc:  # pragma: no cover - filesystem race
        logger.warning("Could not remove artifact %s: %s", confined, exc)
    return False


def _collect_artifact_targets(
    project_id: str,
    model_run_ids: list[str],
    deployments: list[Deployment],
    batch_schedules: list[BatchSchedule],
    batch_job_runs: list[BatchJobRun],
) -> tuple[list[Path], list[tuple[str | None, Path]]]:
    """Build the (directories, (file, base_dir)) removal lists for a project."""
    dirs = storage.project_artifact_dirs(project_id)

    deployments_base = storage.deployments_dir()
    batch_base = storage.batch_outputs_dir()
    files: list[tuple[str | None, Path]] = []

    # Deployment pipelines (shared dir, named by model_run_id).
    for run_id in model_run_ids:
        files.append((str(storage.deployment_pipeline_path(run_id)), deployments_base))
    for dep in deployments:
        files.append((dep.pipeline_path, deployments_base))
        files.append((dep.canary_pipeline_path, deployments_base))

    # Scheduled batch outputs (shared dir, named by schedule_id).
    for sched in batch_schedules:
        files.append((sched.last_output_path, batch_base))
    for job in batch_job_runs:
        files.append((job.output_path, batch_base))

    return dirs, files


def delete_project_cascade(session: Session, project_id: str) -> dict:
    """Delete a project, every descendant row, and its artifacts.

    The caller is responsible for ownership authorization (this function trusts
    ``project_id``). All row deletions happen in a single transaction; the
    artifact filesystem cleanup runs after the commit succeeds.

    Returns a summary dict: removed directory and file counts.
    """
    # --- Gather child ids / rows (before deleting) ---------------------------
    datasets = session.exec(
        select(Dataset).where(Dataset.project_id == project_id)
    ).all()
    dataset_ids = [d.id for d in datasets]

    model_runs = session.exec(
        select(ModelRun).where(ModelRun.project_id == project_id)
    ).all()
    model_run_ids = [m.id for m in model_runs]

    deployments = session.exec(
        select(Deployment).where(Deployment.project_id == project_id)
    ).all()
    deployment_ids = [d.id for d in deployments]

    batch_schedules: list[BatchSchedule] = []
    batch_job_runs: list[BatchJobRun] = []
    if deployment_ids:
        batch_schedules = session.exec(
            select(BatchSchedule).where(BatchSchedule.deployment_id.in_(deployment_ids))
        ).all()
        batch_job_runs = session.exec(
            select(BatchJobRun).where(BatchJobRun.deployment_id.in_(deployment_ids))
        ).all()

    dirs, files = _collect_artifact_targets(
        project_id, model_run_ids, deployments, batch_schedules, batch_job_runs
    )

    # --- Delete rows leaf-first in one transaction ---------------------------
    try:
        if deployment_ids:
            for model in _DEPLOYMENT_CHILDREN:
                session.execute(
                    delete(model).where(model.deployment_id.in_(deployment_ids))
                )
            session.execute(
                delete(ABTest).where(
                    or_(
                        ABTest.champion_id.in_(deployment_ids),
                        ABTest.challenger_id.in_(deployment_ids),
                    )
                )
            )
        if dataset_ids:
            session.execute(
                delete(FeatureSet).where(FeatureSet.dataset_id.in_(dataset_ids))
            )
            session.execute(
                delete(DatasetFilter).where(DatasetFilter.dataset_id.in_(dataset_ids))
            )

        session.execute(delete(Deployment).where(Deployment.project_id == project_id))
        session.execute(delete(ModelRun).where(ModelRun.project_id == project_id))
        session.execute(delete(Dataset).where(Dataset.project_id == project_id))
        session.execute(
            delete(Conversation).where(Conversation.project_id == project_id)
        )
        session.execute(
            delete(AnalysisTemplate).where(AnalysisTemplate.project_id == project_id)
        )
        session.execute(delete(Project).where(Project.id == project_id))
        session.commit()
    except Exception:
        session.rollback()
        raise

    # --- Remove artifacts (best-effort, post-commit) -------------------------
    removed_dirs = 0
    for d in dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            if not d.exists():
                removed_dirs += 1

    removed_files = 0
    for path_str, base in files:
        if _safe_unlink(path_str, base):
            removed_files += 1

    return {"removed_dirs": removed_dirs, "removed_files": removed_files}
