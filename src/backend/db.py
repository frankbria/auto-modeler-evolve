import sqlite3
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, SQLModel, create_engine

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_DIR}/automodeler.db"
engine = create_engine(DATABASE_URL, echo=False)


def db_path() -> Path:
    """On-disk path of the SQLite database file (matches ``DATABASE_URL``)."""
    return DATA_DIR / "automodeler.db"


@event.listens_for(Engine, "connect")
def _configure_sqlite_connection(dbapi_connection, connection_record):
    """Set durability/integrity PRAGMAs on every SQLite connection.

    - ``foreign_keys=ON``: SQLite ships FK enforcement OFF per-connection;
      turning it on makes declared FKs (``Project.owner_id → user.id``) protect
      against dangling references. App-level cascade still lives in
      ``core.cascade`` (the on-disk DB has no FK constraints to enforce).
    - ``journal_mode=WAL``: lets the background daemon threads read while one
      writes, fixing the documented "database is locked"/"readonly database"
      failures. WAL is a persistent database property; re-issuing it per connect
      is idempotent.
    - ``busy_timeout=30000``: a writer waits up to 30s for a lock instead of
      failing immediately — this is per-connection, so it must be set on connect.

    Registered on the ``Engine`` class so it covers the production engine and the
    per-test engines created in the test fixtures alike.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


def check_integrity(path: Path | None = None) -> None:
    """Run ``PRAGMA quick_check`` and raise on corruption (fail-fast).

    A missing file is fine — a fresh DB is created on first boot. A present but
    corrupt file (torn pages, truncation, "not a database") raises ``RuntimeError``
    so the app refuses to boot on bad data instead of serving silently-wrong
    results. Restore from the latest backup (see ``docs/BACKUP.md``).
    """
    path = path or db_path()
    if not path.exists():
        return
    try:
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute("PRAGMA quick_check").fetchall()
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(
            f"Database integrity check failed for {path}: {exc}. "
            "Restore from the latest backup (docs/BACKUP.md)."
        ) from exc
    if rows != [("ok",)]:
        raise RuntimeError(
            f"Database integrity check failed for {path}: {rows}. "
            "Restore from the latest backup (docs/BACKUP.md)."
        )


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    # Inline migrations: add columns that may be missing in existing DBs
    _apply_migrations()


def _apply_migrations(target_engine=None, migrations=None):
    """Add any columns missing from pre-existing tables.

    ``target_engine``/``migrations`` are injectable for tests; production calls
    pass nothing and use the module engine and the canonical list below.
    """
    target_engine = target_engine or engine
    if migrations is None:
        migrations = [
            ("project", "auto_retrain", "INTEGER NOT NULL DEFAULT 0"),
            ("deployment", "api_key_enabled", "INTEGER NOT NULL DEFAULT 0"),
            ("deployment", "api_key_hash", "TEXT"),
            ("deployment", "api_key_salt", "TEXT"),
            ("deployment", "current_version_number", "INTEGER NOT NULL DEFAULT 1"),
            ("deployment", "environment", "TEXT NOT NULL DEFAULT 'staging'"),
            ("predictionlog", "response_ms", "REAL"),
            ("predictionlog", "ab_variant", "TEXT"),
            ("deployment", "rate_limit_rpm", "INTEGER"),
            ("deployment", "monthly_quota", "INTEGER"),
            ("deployment", "quota_alert_threshold_pct", "INTEGER"),
            ("deployment", "accuracy_alert_threshold", "REAL"),
            ("deployment", "accuracy_alert_fired", "INTEGER NOT NULL DEFAULT 0"),
            ("deployment", "confidence_threshold", "REAL"),
            ("deployment", "dashboard_title", "TEXT"),
            ("deployment", "dashboard_description", "TEXT"),
            ("project", "last_milestone_state", "TEXT"),
            ("project", "last_insight_dataset_id", "TEXT"),
            ("project", "last_type_check_dataset_id", "TEXT"),
            ("deployment", "sla_alert_last_fired_at", "TEXT"),
            ("project", "last_ensemble_suggest_run_count", "INTEGER"),
            ("project", "last_low_accuracy_guidance_run_count", "INTEGER"),
            ("deployment", "feature_drift_alert_enabled", "INTEGER NOT NULL DEFAULT 0"),
            ("deployment", "feature_drift_alert_last_fired_at", "TEXT"),
            ("deployment", "low_activity_threshold_per_day", "INTEGER"),
            ("deployment", "low_activity_alert_last_fired_at", "TEXT"),
            ("deployment", "high_activity_threshold_per_hour", "INTEGER"),
            ("deployment", "high_activity_burst_last_fired_at", "TEXT"),
            ("deployment", "latency_alert_threshold_ms", "INTEGER"),
            ("deployment", "latency_alert_last_fired_at", "TEXT"),
            ("deployment", "auto_rollback_enabled", "INTEGER NOT NULL DEFAULT 0"),
            ("deployment", "auto_rollback_accuracy_threshold", "REAL"),
            ("deployment", "auto_rollback_triggered_at", "TEXT"),
            ("deployment", "pred_value_alert_enabled", "INTEGER NOT NULL DEFAULT 0"),
            ("deployment", "pred_value_alert_pct", "REAL"),
            ("deployment", "pred_value_alert_last_fired_at", "TEXT"),
            (
                "deployment",
                "input_dist_drift_alert_enabled",
                "INTEGER NOT NULL DEFAULT 0",
            ),
            (
                "deployment",
                "input_dist_drift_severity_threshold",
                "TEXT NOT NULL DEFAULT 'medium'",
            ),
            ("deployment", "input_dist_drift_alert_last_fired_at", "TEXT"),
            ("deployment", "canary_run_id", "TEXT"),
            ("deployment", "canary_pipeline_path", "TEXT"),
            ("deployment", "canary_traffic_pct", "INTEGER"),
            ("deployment", "canary_is_active", "INTEGER NOT NULL DEFAULT 0"),
            ("deployment", "canary_started_at", "TEXT"),
            ("predictionlog", "served_by_canary", "INTEGER NOT NULL DEFAULT 0"),
            ("deployment", "degradation_retrain_enabled", "INTEGER NOT NULL DEFAULT 0"),
            ("deployment", "degradation_retrain_accuracy_threshold", "REAL"),
            ("deployment", "degradation_retrain_last_triggered_at", "TEXT"),
            ("modelrun", "test_indices", "TEXT"),
            ("modelrun", "evaluation", "TEXT"),
        ]
    with target_engine.connect() as conn:
        for table, col, definition in migrations:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {definition}"))
                conn.commit()
            except OperationalError as exc:
                conn.rollback()
                # Idempotent re-run: the column already exists. Anything else
                # (locked DB, missing table, bad SQL) is real schema drift —
                # fail loud so startup aborts instead of booting "healthy" over
                # an incompletely-migrated DB that 500s at request time.
                if "duplicate column name" in str(exc).lower():
                    continue
                raise RuntimeError(
                    f"Migration failed adding column {table}.{col}: {exc}"
                ) from exc
        # Index migrations: create_all() makes these on fresh DBs, but existing
        # DBs need an explicit CREATE INDEX. IF NOT EXISTS keeps it idempotent
        # (issue #19 — composite index for prediction-log time-window queries).
        index_migrations = [
            (
                "ix_predictionlog_dep_created",
                "predictionlog",
                "(deployment_id, created_at)",
            ),
        ]
        tables = {
            r[0]
            for r in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
        }
        for name, table, cols in index_migrations:
            # create_all() always makes the table before this runs in production;
            # skip when absent so partial/minimal engines (some tests) don't trip.
            if table not in tables:
                continue
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON {table} {cols}"))
            conn.commit()


def get_session():
    with Session(engine) as session:
        yield session
