"""Cascade project-delete + artifact removal (issue #4 / P2.1).

Deleting a project must remove every descendant row (no orphans) and every
on-disk artifact, in one transaction. These tests seed a project with a row in
*every* child table plus all four artifact directory types, delete via the API,
and assert zero surviving rows + removed files.
"""

from uuid import uuid4

from sqlmodel import Session, select

import db
from core import storage
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


def _write(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def _seed_full_subtree(project_id: str) -> dict:
    """Insert one row in every child table + create every artifact for a project."""
    with Session(db.engine) as s:
        dataset = Dataset(project_id=project_id, filename="d.csv", file_path="d.csv")
        model_run = ModelRun(project_id=project_id, algorithm="random_forest")
        s.add(dataset)
        s.add(model_run)
        s.commit()
        s.refresh(dataset)
        s.refresh(model_run)

        deployment = Deployment(
            model_run_id=model_run.id,
            project_id=project_id,
            endpoint_path="/api/predict/x",
            dashboard_url="/predict/x",
        )
        s.add(deployment)
        s.commit()
        s.refresh(deployment)

        schedule = BatchSchedule(deployment_id=deployment.id)
        s.add(schedule)
        s.commit()
        s.refresh(schedule)

        webhook = WebhookConfig(
            deployment_id=deployment.id, url="https://example.com/h"
        )
        s.add(webhook)
        s.commit()
        s.refresh(webhook)

        rows = [
            FeatureSet(dataset_id=dataset.id),
            DatasetFilter(
                dataset_id=dataset.id,
                conditions="[]",
                filter_summary="all",
                original_rows=10,
                filtered_rows=10,
            ),
            Conversation(project_id=project_id),
            AnalysisTemplate(project_id=project_id, name="tpl"),
            PredictionLog(
                deployment_id=deployment.id, input_features="{}", prediction='"y"'
            ),
            FeedbackRecord(deployment_id=deployment.id),
            DeploymentChangelog(
                id=str(uuid4()),
                deployment_id=deployment.id,
                change_type="deployed",
                description="x",
            ),
            DeploymentPreset(deployment_id=deployment.id, name="preset"),
            DashboardFieldConfig(deployment_id=deployment.id, feature_name="f"),
            InputValidationRule(
                deployment_id=deployment.id, feature_name="f", rule_type="not_null"
            ),
            PredictionAlertRule(
                deployment_id=deployment.id, name="alert", condition_type="confidence"
            ),
            SavedScenario(deployment_id=deployment.id, name="scenario"),
            WebhookEvent(
                webhook_id=webhook.id,
                deployment_id=deployment.id,
                event_type="drift_detected",
            ),
            WeeklyDigestConfig(deployment_id=deployment.id),
            GoalSeekRecord(
                deployment_id=deployment.id,
                target_column="t",
                problem_type="regression",
                algorithm_plain="rf",
                target_value_str="5",
                achieved_value_str="4",
                achieved=False,
                summary="x",
            ),
            BatchJobRun(schedule_id=schedule.id, deployment_id=deployment.id),
            DeploymentVersion(
                deployment_id=deployment.id,
                version_number=1,
                model_run_id=model_run.id,
            ),
            ABTest(champion_id=deployment.id, challenger_id=deployment.id),
        ]
        s.add_all(rows)
        s.commit()

        ids = {
            "dataset_id": dataset.id,
            "model_run_id": model_run.id,
            "deployment_id": deployment.id,
            "schedule_id": schedule.id,
            "webhook_id": webhook.id,
        }

    # --- artifacts on disk (via the same storage helpers the cascade uses) ---
    pipeline = _write(storage.deployment_pipeline_path(ids["model_run_id"]))
    batch_out = _write(
        storage.batch_outputs_dir() / f"{ids['schedule_id']}_20240101.csv"
    )
    with Session(db.engine) as s:
        dep = s.get(Deployment, ids["deployment_id"])
        dep.pipeline_path = str(pipeline)
        sched = s.get(BatchSchedule, ids["schedule_id"])
        sched.last_output_path = str(batch_out)
        s.add(dep)
        s.add(sched)
        s.commit()

    upload_dir = storage.project_upload_dir(project_id)
    db_upload_dir = storage.project_db_uploads_dir(project_id)
    models_dir = storage.project_models_dir(project_id)
    _write(upload_dir / "d.csv")
    _write(db_upload_dir / "src.sqlite")
    _write(models_dir / f"{ids['model_run_id']}.joblib")

    ids["artifact_dirs"] = [upload_dir, db_upload_dir, models_dir]
    ids["artifact_files"] = [pipeline, batch_out]
    return ids


# Every (model, id-attribute) pair that must be empty after the delete.
_CHILD_CHECKS = [
    (Dataset, "project_id"),
    (ModelRun, "project_id"),
    (Deployment, "project_id"),
    (Conversation, "project_id"),
    (AnalysisTemplate, "project_id"),
    (FeatureSet, "dataset_id"),
    (DatasetFilter, "dataset_id"),
    (PredictionLog, "deployment_id"),
    (FeedbackRecord, "deployment_id"),
    (DeploymentChangelog, "deployment_id"),
    (DeploymentPreset, "deployment_id"),
    (DashboardFieldConfig, "deployment_id"),
    (InputValidationRule, "deployment_id"),
    (PredictionAlertRule, "deployment_id"),
    (SavedScenario, "deployment_id"),
    (WebhookConfig, "deployment_id"),
    (WebhookEvent, "deployment_id"),
    (WeeklyDigestConfig, "deployment_id"),
    (GoalSeekRecord, "deployment_id"),
    (BatchSchedule, "deployment_id"),
    (BatchJobRun, "deployment_id"),
    (DeploymentVersion, "deployment_id"),
]


async def test_delete_removes_all_child_rows_and_artifacts(client):
    pid = (await client.post("/api/projects", json={"name": "Doomed"})).json()["id"]
    ids = _seed_full_subtree(pid)

    # Pre-condition: artifacts exist.
    for d in ids["artifact_dirs"]:
        assert d.exists()
    for f in ids["artifact_files"]:
        assert f.exists()

    resp = await client.delete(f"/api/projects/{pid}")
    assert resp.status_code == 204

    # The project itself is gone.
    assert (await client.get(f"/api/projects/{pid}")).status_code == 404

    with Session(db.engine) as s:
        assert s.get(Project, pid) is None
        for model, attr in _CHILD_CHECKS:
            link = {
                "project_id": pid,
                "dataset_id": ids["dataset_id"],
                "deployment_id": ids["deployment_id"],
            }[attr]
            remaining = s.exec(select(model).where(getattr(model, attr) == link)).all()
            assert remaining == [], f"orphan rows left in {model.__name__}"
        # ABTest references the deployment by champion/challenger id.
        assert (
            s.exec(
                select(ABTest).where(ABTest.champion_id == ids["deployment_id"])
            ).all()
            == []
        )

    # All artifacts removed.
    for d in ids["artifact_dirs"]:
        assert not d.exists(), f"artifact dir survived: {d}"
    for f in ids["artifact_files"]:
        assert not f.exists(), f"artifact file survived: {f}"


async def test_delete_is_scoped_to_one_project(client):
    """A sibling project's rows and artifacts are untouched."""
    keep = (await client.post("/api/projects", json={"name": "Keep"})).json()["id"]
    doomed = (await client.post("/api/projects", json={"name": "Doomed"})).json()["id"]
    keep_ids = _seed_full_subtree(keep)
    _seed_full_subtree(doomed)

    assert (await client.delete(f"/api/projects/{doomed}")).status_code == 204

    with Session(db.engine) as s:
        assert s.get(Project, keep) is not None
        assert s.exec(select(Dataset).where(Dataset.project_id == keep)).all() != []
        assert (
            s.exec(
                select(PredictionLog).where(
                    PredictionLog.deployment_id == keep_ids["deployment_id"]
                )
            ).all()
            != []
        )
    for d in keep_ids["artifact_dirs"]:
        assert d.exists()


async def test_cross_tenant_delete_forbidden_and_nondestructive(client, second_client):
    pid = (await client.post("/api/projects", json={"name": "Mine"})).json()["id"]
    ids = _seed_full_subtree(pid)

    # Another tenant cannot delete it (404, not 403, to avoid existence leaks).
    assert (await second_client.delete(f"/api/projects/{pid}")).status_code == 404

    # Nothing was removed.
    with Session(db.engine) as s:
        assert s.get(Project, pid) is not None
        assert s.exec(select(Dataset).where(Dataset.project_id == pid)).all() != []
    for d in ids["artifact_dirs"]:
        assert d.exists()


async def test_sqlite_foreign_keys_pragma_enabled(client):
    """PRAGMA foreign_keys is ON for connections from the engine."""
    with db.engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
