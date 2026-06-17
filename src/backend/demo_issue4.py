"""Outcome-evidence demo for issue #4 (cascade project delete + artifact GC).

Runs fully isolated: artifacts go under a temp DATA_DIR and rows into a temp
SQLite DB, so nothing touches the real data tree. Prints before/after evidence
for each acceptance criterion.
"""

import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

# Isolate artifact root BEFORE importing app modules that read it.
_TMP = Path(tempfile.mkdtemp(prefix="demo_issue4_"))
os.environ["DATA_DIR"] = str(_TMP)

from sqlalchemy import text  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402

from core import storage  # noqa: E402
from core.cascade import delete_project_cascade  # noqa: E402
from core.janitor import collect_orphans, enforce_upload_retention  # noqa: E402

# Importing models registers them on SQLModel.metadata for create_all.
from models.ab_test import ABTest  # noqa: E402,F401
from models.analysis_template import AnalysisTemplate  # noqa: E402
from models.batch_schedule import BatchJobRun, BatchSchedule  # noqa: E402
from models.conversation import Conversation  # noqa: E402
from models.dashboard_field_config import DashboardFieldConfig  # noqa: E402
from models.dataset import Dataset  # noqa: E402
from models.dataset_filter import DatasetFilter  # noqa: E402
from models.deployment import Deployment  # noqa: E402
from models.deployment_changelog import DeploymentChangelog  # noqa: E402
from models.deployment_preset import DeploymentPreset  # noqa: E402
from models.deployment_version import DeploymentVersion  # noqa: E402
from models.feature_set import FeatureSet  # noqa: E402
from models.feedback_record import FeedbackRecord  # noqa: E402
from models.goal_seek_record import GoalSeekRecord  # noqa: E402
from models.input_validation_rule import InputValidationRule  # noqa: E402
from models.model_run import ModelRun  # noqa: E402
from models.prediction_alert_rule import PredictionAlertRule  # noqa: E402
from models.prediction_log import PredictionLog  # noqa: E402
from models.project import Project  # noqa: E402
from models.saved_scenario import SavedScenario  # noqa: E402
from models.user import User  # noqa: E402
from models.webhook_config import WebhookConfig  # noqa: E402
from models.webhook_event import WebhookEvent  # noqa: E402
from models.weekly_digest_config import WeeklyDigestConfig  # noqa: E402

# Child models keyed by project_id / dataset_id / deployment_id we count.
CHILD_TABLES = [
    ("Dataset", Dataset),
    ("ModelRun", ModelRun),
    ("Deployment", Deployment),
    ("Conversation", Conversation),
    ("AnalysisTemplate", AnalysisTemplate),
    ("FeatureSet", FeatureSet),
    ("DatasetFilter", DatasetFilter),
    ("PredictionLog", PredictionLog),
    ("FeedbackRecord", FeedbackRecord),
    ("DeploymentChangelog", DeploymentChangelog),
    ("DeploymentPreset", DeploymentPreset),
    ("DashboardFieldConfig", DashboardFieldConfig),
    ("InputValidationRule", InputValidationRule),
    ("PredictionAlertRule", PredictionAlertRule),
    ("SavedScenario", SavedScenario),
    ("WebhookConfig", WebhookConfig),
    ("WebhookEvent", WebhookEvent),
    ("WeeklyDigestConfig", WeeklyDigestConfig),
    ("GoalSeekRecord", GoalSeekRecord),
    ("DeploymentVersion", DeploymentVersion),
    ("BatchSchedule", BatchSchedule),
    ("BatchJobRun", BatchJobRun),
    ("ABTest", ABTest),
]

# Importing db registers the global ``@event.listens_for(Engine, "connect")``
# PRAGMA foreign_keys=ON listener — in the real app db is always imported, so the
# pragma applies to every engine (including this isolated demo one).
import db  # noqa: E402,F401

engine = create_engine(f"sqlite:///{_TMP/'demo.db'}", echo=False)
SQLModel.metadata.create_all(engine)


def _write(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    return p


def seed(pid: str) -> None:
    with Session(engine) as s:
        # Commit the owner first: PRAGMA foreign_keys=ON enforces owner_id→user.id,
        # and the bare column FK gives no ORM save-order dependency.
        s.add(User(id="owner-1", email="o@e.test", hashed_password="x"))
        s.commit()
        s.add(Project(id=pid, owner_id="owner-1", name="demo"))
        s.commit()
        ds = Dataset(project_id=pid, filename="d.csv", file_path="d.csv")
        run = ModelRun(project_id=pid, algorithm="random_forest")
        s.add(ds)
        s.add(run)
        s.commit()
        s.refresh(ds)
        s.refresh(run)
        dep = Deployment(
            model_run_id=run.id,
            project_id=pid,
            endpoint_path="/api/predict/x",
            dashboard_url="/predict/x",
        )
        s.add(dep)
        s.commit()
        s.refresh(dep)
        sched = BatchSchedule(deployment_id=dep.id)
        webhook = WebhookConfig(deployment_id=dep.id, url="https://e.test/h")
        s.add(sched)
        s.add(webhook)
        s.commit()
        s.refresh(sched)
        s.refresh(webhook)
        s.add_all(
            [
                Conversation(project_id=pid),
                AnalysisTemplate(project_id=pid, name="tpl"),
                FeatureSet(dataset_id=ds.id),
                DatasetFilter(
                    dataset_id=ds.id,
                    conditions="[]",
                    filter_summary="all",
                    original_rows=1,
                    filtered_rows=1,
                ),
                PredictionLog(
                    deployment_id=dep.id, input_features="{}", prediction='"y"'
                ),
                FeedbackRecord(deployment_id=dep.id),
                DeploymentChangelog(
                    id=str(uuid4()),
                    deployment_id=dep.id,
                    change_type="deployed",
                    description="x",
                ),
                DeploymentPreset(deployment_id=dep.id, name="p"),
                DashboardFieldConfig(deployment_id=dep.id, feature_name="f"),
                InputValidationRule(
                    deployment_id=dep.id, feature_name="f", rule_type="not_null"
                ),
                PredictionAlertRule(
                    deployment_id=dep.id, name="alert", condition_type="confidence"
                ),
                SavedScenario(deployment_id=dep.id, name="scenario"),
                WebhookEvent(
                    webhook_id=webhook.id,
                    deployment_id=dep.id,
                    event_type="drift_detected",
                ),
                WeeklyDigestConfig(deployment_id=dep.id),
                GoalSeekRecord(
                    deployment_id=dep.id,
                    target_column="t",
                    problem_type="regression",
                    algorithm_plain="rf",
                    target_value_str="5",
                    achieved_value_str="4",
                    achieved=False,
                    summary="x",
                ),
                DeploymentVersion(
                    deployment_id=dep.id, version_number=1, model_run_id=run.id
                ),
                BatchJobRun(schedule_id=sched.id, deployment_id=dep.id),
                ABTest(champion_id=dep.id, challenger_id=dep.id),
            ]
        )
        s.commit()
        # Artifacts on disk (project-scoped dirs + a deployment pipeline file).
        _write(storage.project_upload_dir(pid) / "data.csv")
        _write(storage.project_db_uploads_dir(pid) / "src.db")
        _write(storage.project_models_dir(pid) / f"{run.id}.joblib")
        _write(storage.deployment_pipeline_path(run.id))


def counts() -> list[tuple[str, int]]:
    out = []
    with Session(engine) as s:
        out.append(("Project", len(s.exec(select(Project)).all())))
        for name, model in CHILD_TABLES:
            out.append((name, len(s.exec(select(model)).all())))
    return out


def artifact_dirs(pid: str) -> list[tuple[str, bool]]:
    items = [(str(d), d.exists()) for d in storage.project_artifact_dirs(pid)]
    pj = list(storage.deployments_dir().glob("*_pipeline.joblib"))
    items.append((f"deployments/*_pipeline.joblib ({len(pj)} file)", bool(pj)))
    return items


PID = "demo-project-1"
seed(PID)

print("=" * 72)
print("CRITERION 1 — cascade delete: all child rows + artifacts in one commit")
print("=" * 72)
print("\nRow counts BEFORE delete:")
total_before = 0
for name, n in counts():
    total_before += n
    print(f"  {name:24} {n}")
print(f"  {'TOTAL ROWS':24} {total_before}")

print("\nArtifact dirs/files BEFORE delete:")
for label, exists in artifact_dirs(PID):
    print(f"  [{'EXISTS' if exists else 'gone  '}] {label}")

with Session(engine) as s:
    summary = delete_project_cascade(s, PID)

print(f"\n>>> delete_project_cascade() summary: {summary}")

print("\nRow counts AFTER delete:")
total_after = 0
for name, n in counts():
    total_after += n
    print(f"  {name:24} {n}")
print(f"  {'TOTAL ROWS':24} {total_after}")

print("\nArtifact dirs/files AFTER delete:")
for label, exists in artifact_dirs(PID):
    print(f"  [{'EXISTS' if exists else 'gone  '}] {label}")

assert total_after == 0, "FAIL: orphan rows remain"
print(f"\nRESULT: {total_before} rows -> 0, all artifacts removed. PASS")

print("\n" + "=" * 72)
print("CRITERION 2 — PRAGMA foreign_keys=ON on a fresh connection")
print("=" * 72)
with engine.connect() as conn:
    fk = conn.execute(text("PRAGMA foreign_keys")).scalar()
print(f"\n  PRAGMA foreign_keys -> {fk}   ({'ON' if fk == 1 else 'OFF'})")
assert fk == 1
print("RESULT: foreign-key enforcement is ON. PASS")

print("\n" + "=" * 72)
print("CRITERION 3 — janitor GCs orphan artifacts, keeps live, enforces retention")
print("=" * 72)
# Live project (still in DB) + an orphan project (no DB row).
with Session(engine) as s:
    s.add(User(id="owner-2", email="o2@e.test", hashed_password="x"))
    s.commit()
    s.add(Project(id="live-project", owner_id="owner-2", name="live"))
    s.commit()
live_dir = _write(storage.project_upload_dir("live-project") / "keep.csv").parent
orphan_dir = _write(storage.project_upload_dir("ghost-project") / "junk.csv").parent
orphan_models = _write(storage.project_models_dir("ghost-project") / "m.joblib").parent
print(f"\n  live   upload dir exists: {live_dir.exists()}  ({live_dir})")
print(f"  orphan upload dir exists: {orphan_dir.exists()}  ({orphan_dir})")
print(f"  orphan models dir exists: {orphan_models.exists()}  ({orphan_models})")
with Session(engine) as s:
    jsummary = collect_orphans(s, upload_grace_days=0)
print(f"\n>>> collect_orphans() summary: {jsummary}")
print(f"\n  live   upload dir exists: {live_dir.exists()}  (kept)")
print(f"  orphan upload dir exists: {orphan_dir.exists()}  (reaped)")
print(f"  orphan models dir exists: {orphan_models.exists()}  (reaped)")
assert live_dir.exists() and not orphan_dir.exists() and not orphan_models.exists()

# Retention: backdate an orphan upload 10 days; 30-day retention keeps it, 7-day reaps it.
old = _write(storage.project_upload_dir("old-ghost") / "old.csv").parent
ten_days_ago = __import__("time").time() - 10 * 86400
os.utime(old, (ten_days_ago, ten_days_ago))
with Session(engine) as s:
    kept = enforce_upload_retention(s, max_age_days=30)
    print(
        f"\n  enforce_upload_retention(30d): removed {kept}, dir exists={old.exists()}"
    )
    reaped = enforce_upload_retention(s, max_age_days=7)
    print(
        f"  enforce_upload_retention(7d):  removed {reaped}, dir exists={old.exists()}"
    )
assert kept == 0 and reaped == 1 and not old.exists()
print("RESULT: orphans reaped, live kept, retention age-gated. PASS")

# Cleanup temp tree.
shutil.rmtree(_TMP, ignore_errors=True)
print("\nALL CRITERIA DEMONSTRATED WITH OUTCOME EVIDENCE.")
