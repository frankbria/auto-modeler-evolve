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
    assert "data/automodeler.db" not in names
    assert "data/automodeler.db-wal" not in names
    assert "data/automodeler.db-shm" not in names


def test_backup_includes_artifacts_named_like_the_db(tmp_path):
    """Exclusion is by exact path, not basename: a real artifact whose name
    starts with 'automodeler.db' must still be backed up (codex P1b)."""
    import tarfile

    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"
    _make_db(db_file)
    decoy = data_root / "uploads" / "p1" / "automodeler.db.csv"
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"col\n1\n")

    archive = backup.create_backup(tmp_path / "b", db_file=db_file, data_root=data_root)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "data/uploads/p1/automodeler.db.csv" in names


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


def test_backup_dest_inside_data_root_skips_prior_archives(tmp_path):
    """dest_dir under data_root must not let backups swallow each other
    (would grow exponentially) — codex P1."""
    import tarfile

    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"
    _make_db(db_file)
    dest = data_root / "backups"

    first = backup.create_backup(dest, db_file=db_file, data_root=data_root)
    second = backup.create_backup(dest, db_file=db_file, data_root=data_root)

    with tarfile.open(second) as tar:
        names = tar.getnames()
    assert not any("backups" in n for n in names)
    assert first.exists()  # the earlier archive was not consumed


def test_backup_rejects_data_root_as_dest(tmp_path):
    """Backing up into the data root itself would recurse — reject it (codex P1)."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"
    _make_db(db_file)
    with pytest.raises(ValueError, match="data root"):
        backup.create_backup(data_root, db_file=db_file, data_root=data_root)


def test_backup_dest_above_data_root_still_captures_artifacts(tmp_path):
    """A destination above data_root must not silently drop every artifact
    (the over-broad ancestor exclusion bug — codex P1)."""
    import tarfile

    data_root = tmp_path / "backend" / "data"
    data_root.mkdir(parents=True)
    db_file = data_root / "automodeler.db"
    _make_db(db_file)
    art = data_root / "models" / "m.joblib"
    art.parent.mkdir()
    art.write_bytes(b"M")

    dest = tmp_path / "backend"  # ancestor of data_root
    archive = backup.create_backup(dest, db_file=db_file, data_root=data_root)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "data/models/m.joblib" in names


def test_restore_rejects_db_less_archive(tmp_path):
    """An archive with no DB snapshot must not wipe the live WAL/SHM — codex P2."""
    import io
    import tarfile

    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"
    _make_db(db_file)
    wal = db_file.with_name("automodeler.db-wal")
    wal.write_bytes(b"precious-wal-pages")

    dbless = tmp_path / "dbless.tar.gz"
    with tarfile.open(dbless, "w:gz") as tar:
        payload = b"x\n"
        info = tarfile.TarInfo(name="data/uploads/p/x.csv")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="no database"):
        backup.restore_backup(dbless, db_file=db_file, data_root=data_root)
    assert wal.read_bytes() == b"precious-wal-pages"  # live WAL untouched


def test_restore_replaces_managed_tree_not_overlay(tmp_path):
    """Restoring an older backup must drop files created after the snapshot
    (true restore, not overlay) — codex P2."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"
    _make_db(db_file)
    kept = data_root / "models" / "p" / "a.joblib"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"A")

    archive = backup.create_backup(tmp_path / "b", db_file=db_file, data_root=data_root)

    # A file created *after* the snapshot, inside a managed root.
    stale = data_root / "models" / "p" / "post_snapshot.joblib"
    stale.write_bytes(b"STALE")

    backup.restore_backup(archive, db_file=db_file, data_root=data_root)

    assert kept.read_bytes() == b"A"
    assert not stale.exists()  # post-snapshot file gone after restore


def test_restore_rejects_duplicate_artifact_members(tmp_path):
    """A crafted archive with the same artifact path twice must be rejected
    before any live mutation — codex P2."""
    import io
    import tarfile

    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"
    _make_db(db_file, value="live")

    evil = tmp_path / "dup.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        good_db = tmp_path / "snap.db"
        _make_db(good_db)
        tar.add(good_db, arcname="automodeler.db")
        for payload in (b"one", b"two"):
            info = tarfile.TarInfo(name="data/models/p/x.joblib")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="duplicate"):
        backup.restore_backup(evil, db_file=db_file, data_root=data_root)

    conn = sqlite3.connect(str(db_file))
    assert conn.execute("SELECT v FROM t").fetchone() == ("live",)  # untouched
    conn.close()


def test_restore_rejects_artifact_aliasing_the_db(tmp_path):
    """A crafted artifact entry that resolves to the DB path must be rejected
    before any live mutation (codex P1)."""
    import io
    import tarfile

    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"
    _make_db(db_file, value="live")

    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        good_db = tmp_path / "snap.db"
        _make_db(good_db, value="snapshot")
        tar.add(good_db, arcname="automodeler.db")
        junk = b"corrupt-the-db"
        info = tarfile.TarInfo(name="data/automodeler.db")  # aliases db_file
        info.size = len(junk)
        tar.addfile(info, io.BytesIO(junk))

    with pytest.raises(ValueError, match="aliases the database"):
        backup.restore_backup(evil, db_file=db_file, data_root=data_root)

    conn = sqlite3.connect(str(db_file))
    assert conn.execute("SELECT v FROM t").fetchone() == ("live",)  # untouched
    conn.close()


def test_restore_leaves_live_state_intact_on_corrupt_archive(tmp_path):
    """A corrupt DB snapshot is rejected before any live file is replaced."""
    import tarfile

    data_root = tmp_path / "data"
    data_root.mkdir()
    db_file = data_root / "automodeler.db"
    _make_db(db_file, value="live")
    artifact = data_root / "models" / "m.joblib"
    artifact.parent.mkdir()
    artifact.write_bytes(b"LIVE-MODEL")

    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        import io

        junk = b"not a database"
        info = tarfile.TarInfo(name="automodeler.db")
        info.size = len(junk)
        tar.addfile(info, io.BytesIO(junk))
        good = b"NEW-MODEL"
        info2 = tarfile.TarInfo(name="data/models/m.joblib")
        info2.size = len(good)
        tar.addfile(info2, io.BytesIO(good))

    with pytest.raises(RuntimeError, match="integrity check failed"):
        backup.restore_backup(bad, db_file=db_file, data_root=data_root)

    # Live DB and artifact must be untouched.
    conn = sqlite3.connect(str(db_file))
    assert conn.execute("SELECT v FROM t").fetchone() == ("live",)
    conn.close()
    assert artifact.read_bytes() == b"LIVE-MODEL"
