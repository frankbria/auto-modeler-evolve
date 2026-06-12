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
        candidate = now.replace(
            hour=run_hour, minute=run_minute, second=0, microsecond=0
        )
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

    # Fire batch_complete webhook (outside DB session — best-effort)
    try:
        from core.webhook import EVENT_BATCH_COMPLETE, dispatch_webhooks

        dispatch_webhooks(
            schedule.deployment_id,
            EVENT_BATCH_COMPLETE,
            {
                "schedule_id": schedule_id,
                "status": job_run.status,
                "row_count": job_run.row_count,
                "error": job_run.error,
                "completed_at": (
                    job_run.completed_at.isoformat() if job_run.completed_at else None
                ),
            },
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Weekly monitoring digest
# ---------------------------------------------------------------------------

_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def should_send_weekly_digest(
    day_of_week: int,
    send_hour: int,
    last_sent_at: "datetime | None",
    now: "datetime",
) -> bool:
    """Return True when a weekly digest is due.

    Fires when all of the following are true:
    - today's weekday matches day_of_week
    - current hour >= send_hour
    - not already sent today (last_sent_at is None or was before today)
    """
    if now.weekday() != day_of_week:
        return False
    if now.hour < send_hour:
        return False
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if last_sent_at is not None and last_sent_at >= today_start:
        return False
    return True


def _run_weekly_digest(config_id: str) -> None:
    """Compute monitoring digest for the deployment and dispatch via webhook."""
    from db import engine
    from models.deployment import Deployment
    from models.prediction_log import PredictionLog
    from models.weekly_digest_config import WeeklyDigestConfig
    from sqlmodel import Session, select

    from core.analyzer import (
        compute_deployment_monitoring_digest,
        compute_prediction_output_anomalies,
        compute_prediction_value_trend,
    )
    from core.webhook import EVENT_WEEKLY_DIGEST, dispatch_webhooks

    with Session(engine) as session:
        cfg = session.get(WeeklyDigestConfig, config_id)
        if not cfg or not cfg.enabled:
            return
        deployment = session.get(Deployment, cfg.deployment_id)
        if not deployment or not deployment.is_active:
            return

        # Load recent prediction logs for signal computation
        logs = session.exec(
            select(PredictionLog)
            .where(PredictionLog.deployment_id == deployment.id)
            .order_by(PredictionLog.created_at.desc())  # type: ignore[arg-type]
            .limit(200)
        ).all()

        logs_data = [
            {
                "prediction_numeric": r.prediction_numeric,
                "confidence": r.confidence,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in logs
        ]

        problem_type = deployment.problem_type or "regression"

        # Compute individual signals (best-effort — never crash)
        anomaly_result = None
        value_trend_result = None
        dist_shift_result = None
        readiness_result = None

        try:
            if len(logs_data) >= 5:
                anomaly_result = compute_prediction_output_anomalies(
                    logs_data, problem_type
                )
        except Exception:
            pass

        try:
            if problem_type == "regression" and len(logs_data) >= 2:
                value_trend_result = compute_prediction_value_trend(
                    logs_data, "day", 30
                )
        except Exception:
            pass

        # Usage 7-day comparison
        from datetime import timedelta

        now = datetime.now(UTC).replace(tzinfo=None)
        cutoff_7d = now - timedelta(days=7)
        cutoff_14d = now - timedelta(days=14)
        usage_7d = sum(1 for r in logs if r.created_at and r.created_at >= cutoff_7d)
        usage_prev_7d = sum(
            1 for r in logs if r.created_at and cutoff_14d <= r.created_at < cutoff_7d
        )

        digest = compute_deployment_monitoring_digest(
            anomaly_result,
            value_trend_result,
            dist_shift_result,
            readiness_result,
            usage_7d,
            usage_prev_7d,
            problem_type,
        )

        # Stamp sent time
        cfg.last_sent_at = now
        session.add(cfg)
        session.commit()

    # Dispatch webhook (outside session)
    dispatch_webhooks(
        deployment.id,
        EVENT_WEEKLY_DIGEST,
        {
            "overall_health": digest["overall_health"],
            "health_score": digest["health_score"],
            "signals_total": len(digest.get("signals", [])),
            "signals_firing": sum(
                1
                for s in digest.get("signals", [])
                if s.get("severity") in ("amber", "red")
            ),
            "priority_actions": digest.get("priority_actions", []),
            "summary": digest.get("summary", ""),
            "algorithm": deployment.algorithm or "",
            "target_column": deployment.target_column or "",
        },
    )


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------


def _scheduler_loop() -> None:
    """Check every 60 seconds for due batch jobs, weekly digests, and low-activity alerts."""
    from db import engine
    from models.batch_schedule import BatchSchedule
    from models.deployment import Deployment
    from models.weekly_digest_config import WeeklyDigestConfig
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

                # Find weekly digests due to send
                digest_cfgs = session.exec(
                    select(WeeklyDigestConfig).where(
                        WeeklyDigestConfig.enabled == True,  # noqa: E712
                    )
                ).all()
                digest_ids = [
                    c.id
                    for c in digest_cfgs
                    if should_send_weekly_digest(
                        c.day_of_week, c.send_hour, c.last_sent_at, now
                    )
                ]

                # Collect deployments with low-activity alerts configured
                active_deps = session.exec(
                    select(Deployment).where(
                        Deployment.is_active == True,  # noqa: E712
                        Deployment.low_activity_threshold_per_day != None,  # noqa: E711
                    )
                ).all()
                low_activity_dep_ids = [d.id for d in active_deps]

                # Collect deployments with high-activity burst alerts configured
                burst_deps = session.exec(
                    select(Deployment).where(
                        Deployment.is_active == True,  # noqa: E712
                        Deployment.high_activity_threshold_per_hour != None,  # noqa: E711
                    )
                ).all()
                high_activity_dep_ids = [d.id for d in burst_deps]

                # Collect deployments with latency alerts configured
                latency_deps = session.exec(
                    select(Deployment).where(
                        Deployment.is_active == True,  # noqa: E712
                        Deployment.latency_alert_threshold_ms != None,  # noqa: E711
                    )
                ).all()
                latency_dep_ids = [d.id for d in latency_deps]

                # Collect deployments with accuracy-triggered auto-rollback enabled
                rollback_deps = session.exec(
                    select(Deployment).where(
                        Deployment.is_active == True,  # noqa: E712
                        Deployment.auto_rollback_enabled == True,  # noqa: E712
                        Deployment.auto_rollback_accuracy_threshold != None,  # noqa: E711
                    )
                ).all()
                rollback_dep_ids = [d.id for d in rollback_deps]

                # Collect deployments with prediction value trend alert enabled
                pred_value_deps = session.exec(
                    select(Deployment).where(
                        Deployment.is_active == True,  # noqa: E712
                        Deployment.pred_value_alert_enabled == True,  # noqa: E712
                        Deployment.pred_value_alert_pct != None,  # noqa: E711
                    )
                ).all()
                pred_value_dep_ids = [d.id for d in pred_value_deps]

                # Collect deployments with input distribution drift alert enabled
                input_dist_deps = session.exec(
                    select(Deployment).where(
                        Deployment.is_active == True,  # noqa: E712
                        Deployment.input_dist_drift_alert_enabled == True,  # noqa: E712
                    )
                ).all()
                input_dist_dep_ids = [d.id for d in input_dist_deps]

                # Collect deployments with degradation-triggered auto-retrain enabled
                degradation_retrain_deps = session.exec(
                    select(Deployment).where(
                        Deployment.is_active == True,  # noqa: E712
                        Deployment.degradation_retrain_enabled == True,  # noqa: E712
                        Deployment.degradation_retrain_accuracy_threshold != None,  # noqa: E711
                    )
                ).all()
                degradation_retrain_dep_ids = [d.id for d in degradation_retrain_deps]

            for sid in due_ids:
                try:
                    _run_job(sid)
                except Exception as exc:
                    logger.error("Scheduler: job %s raised: %s", sid, exc)

            for did in digest_ids:
                try:
                    t = threading.Thread(
                        target=_run_weekly_digest,
                        args=(did,),
                        daemon=True,
                    )
                    t.start()
                except Exception as exc:
                    logger.error("Scheduler: weekly digest %s raised: %s", did, exc)

            for dep_id in low_activity_dep_ids:
                try:
                    from api.deploy import _check_and_fire_low_activity_alert

                    _check_and_fire_low_activity_alert(dep_id)
                except Exception as exc:
                    logger.error(
                        "Scheduler: low-activity check %s raised: %s", dep_id, exc
                    )

            for dep_id in high_activity_dep_ids:
                try:
                    from api.deploy import _check_and_fire_high_activity_burst

                    _check_and_fire_high_activity_burst(dep_id)
                except Exception as exc:
                    logger.error(
                        "Scheduler: high-activity burst check %s raised: %s",
                        dep_id,
                        exc,
                    )

            for dep_id in latency_dep_ids:
                try:
                    from api.deploy import _check_and_fire_latency_alert

                    _check_and_fire_latency_alert(dep_id)
                except Exception as exc:
                    logger.error(
                        "Scheduler: latency alert check %s raised: %s",
                        dep_id,
                        exc,
                    )

            for dep_id in rollback_dep_ids:
                try:
                    from api.deploy import _check_and_fire_accuracy_rollback

                    _check_and_fire_accuracy_rollback(dep_id)
                except Exception as exc:
                    logger.error(
                        "Scheduler: accuracy rollback check %s raised: %s",
                        dep_id,
                        exc,
                    )

            for dep_id in pred_value_dep_ids:
                try:
                    from api.deploy import _check_and_fire_pred_value_trend_alert

                    _check_and_fire_pred_value_trend_alert(dep_id)
                except Exception as exc:
                    logger.error(
                        "Scheduler: pred value trend alert check %s raised: %s",
                        dep_id,
                        exc,
                    )

            for dep_id in input_dist_dep_ids:
                try:
                    from api.deploy import _check_and_fire_input_dist_drift_alert

                    _check_and_fire_input_dist_drift_alert(dep_id)
                except Exception as exc:
                    logger.error(
                        "Scheduler: input dist drift alert check %s raised: %s",
                        dep_id,
                        exc,
                    )

            for dep_id in degradation_retrain_dep_ids:
                try:
                    from api.deploy import _check_and_trigger_degradation_retrain

                    _check_and_trigger_degradation_retrain(dep_id)
                except Exception as exc:
                    logger.error(
                        "Scheduler: degradation retrain check %s raised: %s",
                        dep_id,
                        exc,
                    )

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
