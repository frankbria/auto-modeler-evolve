"""DB durability (#10): WAL/busy_timeout/FK PRAGMAs, quick_check fail-fast,
and a coordinated backup → wipe → restore round trip."""

import sqlite3

import pytest
from sqlmodel import create_engine, text

import db as db_module
from core import backup


def _make_db(path, value="hello"):
    """Create a tiny SQLite DB with one row so we can prove data survives."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (v TEXT)")
    conn.execute("INSERT INTO t VALUES (?)", (value,))
    conn.commit()
    conn.close()


# --- PRAGMAs (criterion 2) --------------------------------------------------


def test_connection_pragmas_are_set(tmp_path):
    """Every connection gets WAL + 30s busy_timeout + FK enforcement."""
    engine = create_engine(f"sqlite:///{tmp_path / 'p.db'}")
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 30000
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


# --- quick_check fail-fast (criterion 3) ------------------------------------


def test_check_integrity_passes_on_good_db(tmp_path):
    good = tmp_path / "good.db"
    _make_db(good)
    db_module.check_integrity(good)  # must not raise


def test_check_integrity_missing_file_is_ok(tmp_path):
    db_module.check_integrity(tmp_path / "absent.db")  # fresh DB → no raise


def test_check_integrity_raises_on_corrupt_db(tmp_path):
    corrupt = tmp_path / "corrupt.db"
    corrupt.write_bytes(b"this is not a sqlite database at all")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        db_module.check_integrity(corrupt)


# --- backup / restore round trip (criterion 1) ------------------------------


def test_backup_then_restore_recovers_db_and_artifacts(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"
    _make_db(db_file, value="original")

    # An artifact that must travel with the DB.
    artifact = data_root / "models" / "proj1" / "run.joblib"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"\x00MODEL\x00")

    dest = tmp_path / "backups"
    archive = backup.create_backup(dest, db_file=db_file, data_root=data_root)
    assert archive.exists()

    # Simulate disaster: wipe the DB and the artifact.
    db_file.unlink()
    artifact.unlink()

    backup.restore_backup(archive, db_file=db_file, data_root=data_root)

    conn = sqlite3.connect(str(db_file))
    assert conn.execute("SELECT v FROM t").fetchone() == ("original",)
    conn.close()
    assert artifact.read_bytes() == b"\x00MODEL\x00"


def test_backup_excludes_live_wal_shm(tmp_path):
    """The live DB and its WAL/SHM sidecars are not double-captured in data/."""
    import tarfile

    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"
    _make_db(db_file)
    (data_root / "automodeler.db-wal").write_bytes(b"wal")
    (data_root / "automodeler.db-shm").write_bytes(b"shm")

    archive = backup.create_backup(tmp_path / "b", db_file=db_file, data_root=data_root)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "automodeler.db" in names
    assert not any(n.startswith("data/automodeler.db") for n in names)


def test_restore_rejects_path_traversal(tmp_path):
    """A malicious archive member can't escape the data root."""
    import io
    import tarfile

    from core.path_safety import UnsafePathError

    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"

    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        payload = b"pwned"
        info = tarfile.TarInfo(name="data/../../escape.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(UnsafePathError):
        backup.restore_backup(evil, db_file=db_file, data_root=data_root)
    assert not (tmp_path / "escape.txt").exists()
