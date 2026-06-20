"""Migration framework (#11): fail-loud ALTER loop.

The old loop wrapped every ALTER in ``except Exception: pass``, so a partial
migration (locked DB, missing table) silently left columns missing and the app
booted "healthy", then 500'd at request time with ``no such column``. The fix
ignores ONLY ``duplicate column name`` (idempotent re-run) and aborts on
everything else.
"""

import pytest
from sqlmodel import create_engine, text

import db as db_module


def _engine_with_table(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'm.db'}")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE project (id INTEGER PRIMARY KEY)"))
    return eng


def test_duplicate_column_is_idempotent(tmp_path):
    """Re-running an already-applied migration must not raise."""
    eng = _engine_with_table(tmp_path)
    mig = [("project", "flag", "INTEGER NOT NULL DEFAULT 0")]
    db_module._apply_migrations(eng, mig)  # adds it
    db_module._apply_migrations(eng, mig)  # column exists -> ignored
    with eng.connect() as conn:
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(project)"))]
    assert "flag" in cols


def test_partial_migration_aborts_at_boot(tmp_path):
    """A failed ALTER (here: missing table) aborts instead of silently passing."""
    eng = _engine_with_table(tmp_path)
    mig = [
        ("project", "flag", "INTEGER NOT NULL DEFAULT 0"),
        ("nope", "x", "TEXT"),  # table doesn't exist -> real failure
    ]
    with pytest.raises(RuntimeError, match="Migration failed adding column nope.x"):
        db_module._apply_migrations(eng, mig)
