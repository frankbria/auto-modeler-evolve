"""The (deployment_id, created_at) composite index exists (issue #19).

Time-window/sort analytics queries on PredictionLog are near-universal; without
the composite index they full-scan. These tests cover both creation paths:
fresh DB via ``create_all`` and an existing DB via ``_apply_migrations``.
"""

from sqlalchemy import text
from sqlmodel import SQLModel, create_engine

import models.prediction_log  # noqa: F401 — registers the table on metadata
from db import _apply_migrations

INDEX_NAME = "ix_predictionlog_dep_created"


def _index_names(engine) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("PRAGMA index_list(predictionlog)")).fetchall()
    # PRAGMA index_list columns: seq, name, unique, origin, partial
    return {row[1] for row in rows}


def test_composite_index_created_on_fresh_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    SQLModel.metadata.create_all(engine)
    assert INDEX_NAME in _index_names(engine)


def test_index_migration_creates_index_on_existing_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    # Simulate a pre-#19 DB: predictionlog table with no composite index.
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE predictionlog "
                "(id TEXT PRIMARY KEY, deployment_id TEXT, created_at TEXT)"
            )
        )
        conn.commit()
    assert INDEX_NAME not in _index_names(engine)

    # migrations=[] skips column adds; the index migration still runs.
    _apply_migrations(target_engine=engine, migrations=[])
    assert INDEX_NAME in _index_names(engine)

    # Idempotent: a second run does not raise.
    _apply_migrations(target_engine=engine, migrations=[])
    assert INDEX_NAME in _index_names(engine)
