"""Background batch prediction scheduler.

A daemon thread wakes every 60 seconds, finds schedules whose next_run is
in the past, and executes the batch prediction job against the deployment's
training dataset.  Results are saved to data/batch_outputs/<schedule_id>_<ts>.csv.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

BATCH_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "batch_outputs"


# ---------------------------------------------------------------------------
# Next-run computation
# ---------------------------------------------------------------------------


def compute_next_run(
    frequency: str,
    run_hour: int,
    run_minute: int,
    day_of_week: int | None,
    day_of_month: int | None,
    after: datetime | None = None,
) -> datetime:
    """Return the next UTC datetime this schedule should fire."""
    now = after or datetime.now(UTC).replace(tzinfo=None)

    if frequency == "daily":
        candidate = now.replace(hour=run_hour, minute=run_minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    if frequency == "weekly":
        dow = day_of_week if day_of_week is not None else 0  # default Monday
        days_ahead = (dow - now.weekday()) % 7
        candidate = now.replace(
            hour=run_hour, minute=run_minute, second=0, microsecond=0
        ) + timedelta(days=days_ahead)
        if candidate <= now:
            candidate += timedelta(weeks=1)
        return candidate

    # monthly
    dom = day_of_month if day_of_month is not None else 1
    # clamp to valid range
    dom = max(1, min(dom, 28))
    candidate = now.replace(
        day=dom, hour=run_hour, minute=run_minute, second=0, microsecond=0
    )
    if candidate <= now:
        # advance one month
        if now.month == 12:
            candidate = candidate.replace(year=now.year + 1, month=1)
        else:
            candidate = candidate.replace(month=now.month + 1)
    return candidate


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def _run_job(schedule_id: str) -> None:
    """Execute one batch prediction job for the given schedule."""
    # Import here to avoid circular imports at module load
    from core.deployer import predict_batch
    from db import engine
    from models.batch_schedule import BatchJobRun, BatchSchedule
    from models.deployment import Deployment
    from models.model_run import ModelRun
    from sqlmodel import Session, select

    with Session(engine) as session:
        schedule = session.get(BatchSchedule, schedule_id)
        if not schedule or not schedule.is_active:
            return

        deployment = session.get(Deployment, schedule.deployment_id)
        if not deployment or not deployment.is_active:
            schedule.last_error = "Deployment not found or inactive"
            session.add(schedule)
            session.commit()
            return

        run = session.exec(
            select(ModelRun).where(ModelRun.id == deployment.model_run_id)
        ).first()
        if not run or not run.model_path or not Path(run.model_path).exists():
            schedule.last_error = "Model file not found"
            session.add(schedule)
            session.commit()
            return

        if not deployment.pipeline_path or not Path(deployment.pipeline_path).exists():
            schedule.last_error = "Pipeline file not found"
            session.add(schedule)
            session.commit()
            return

        # Find the dataset CSV used for this deployment
        from models.dataset import Dataset
        from models.feature_set import FeatureSet

        feature_set = session.exec(
            select(FeatureSet).where(FeatureSet.id == run.feature_set_id)
        ).first()
        if not feature_set:
            schedule.last_error = "Feature set not found"
            session.add(schedule)
            session.commit()
            return

        dataset = session.exec(
            select(Dataset).where(Dataset.id == feature_set.dataset_id)
        ).first()
        if not dataset or not dataset.file_path or not Path(dataset.file_path).exists():
            schedule.last_error = "Dataset file not found"
            session.add(schedule)
            session.commit()
            return

        # Create job run record
        job_run = BatchJobRun(
            schedule_id=schedule_id,
            deployment_id=schedule.deployment_id,
        )
        session.add(job_run)
        session.commit()
        session.refresh(job_run)
        job_run_id = job_run.id

    # Execute outside the original session so we can update atomically
    with Session(engine) as session:
        job_run = session.get(BatchJobRun, job_run_id)
        schedule = session.get(BatchSchedule, schedule_id)
        if not job_run or not schedule:
            return

        try:
            csv_bytes = Path(dataset.file_path).read_bytes()
            result_bytes = predict_batch(
                deployment.pipeline_path,
                run.model_path,
                csv_bytes,
            )

            BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            out_file = BATCH_OUTPUT_DIR / f"{schedule_id}_{ts}.csv"
            out_file.write_bytes(result_bytes)

            row_count = len(result_bytes.decode().splitlines()) - 1  # exclude header

            now = datetime.now(UTC).replace(tzinfo=None)
            job_run.status = "success"
            job_run.completed_at = now
            job_run.output_path = str(out_file)
            job_run.row_count = row_count

            schedule.last_run = now
            schedule.last_output_path = str(out_file)
            schedule.last_row_count = row_count
            schedule.last_error = None
            schedule.next_run = compute_next_run(
                schedule.frequency,
                schedule.run_hour,
                schedule.run_minute,
                schedule.day_of_week,
                schedule.day_of_month,
                after=now,
            )

        except Exception as exc:
            now = datetime.now(UTC).replace(tzinfo=None)
            err = str(exc)[:500]
            job_run.status = "failed"
            job_run.completed_at = now
            job_run.error = err

            schedule.last_error = err
            schedule.last_run = now
            schedule.next_run = compute_next_run(
                schedule.frequency,
                schedule.run_hour,
                schedule.run_minute,
                schedule.day_of_week,
                schedule.day_of_month,
                after=now,
            )
            logger.error("Batch job %s failed: %s", schedule_id, err)

        session.add(job_run)
        session.add(schedule)
        session.commit()


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------


def _scheduler_loop() -> None:
    """Check every 60 seconds for due batch jobs."""
    from db import engine
    from models.batch_schedule import BatchSchedule
    from sqlmodel import Session, select

    while True:
        try:
            now = datetime.now(UTC).replace(tzinfo=None)
            with Session(engine) as session:
                due = session.exec(
                    select(BatchSchedule).where(
                        BatchSchedule.is_active == True,  # noqa: E712
                        BatchSchedule.next_run <= now,
                    )
                ).all()
                due_ids = [s.id for s in due]

            for sid in due_ids:
                try:
                    _run_job(sid)
                except Exception as exc:
                    logger.error("Scheduler: job %s raised: %s", sid, exc)

        except Exception as exc:
            logger.error("Scheduler loop error: %s", exc)

        time.sleep(60)


_scheduler_thread: threading.Thread | None = None


def start_scheduler() -> None:
    """Start the background scheduler daemon thread (idempotent)."""
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        return
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        name="BatchScheduler",
        daemon=True,
    )
    _scheduler_thread.start()
    logger.info("Batch scheduler started")
