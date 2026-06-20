"""Coordinated backup/restore of the SQLite DB + on-disk artifacts.

All durable state lives in two places that must be snapshotted *together* or a
restore yields deployments that 404 on every prediction: the SQLite database and
the artifact tree (uploads / models / deployments / batch_outputs). In
production ``DATA_DIR`` is unset, so both live under ``<backend>/data`` and a
single archive captures them in one window.

The DB is copied via SQLite's **online backup API** (``sqlite3.Connection.backup``)
so a live, concurrently-written DB is captured without torn pages — a plain
``cp`` of a WAL-mode DB can grab a half-written page. The artifact files are
write-mostly-once, so a same-window ``tar`` of them is consistent enough.

Archive layout (``tar.gz``)::

    automodeler.db        # consistent online-backup snapshot of the DB
    data/<relpath>...     # artifact tree (everything under data_root except the
                          # live automodeler.db / -wal / -shm)

Usage::

    python -m core.backup backup  /var/backups/automodeler
    python -m core.backup restore /var/backups/automodeler/automodeler-backup-....tar.gz

See ``docs/BACKUP.md`` for the cron/timer schedule and the restore drill.
"""

from __future__ import annotations

import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

from core import storage
from core.path_safety import assert_within
from db import db_path

_DB_ARCNAME = "automodeler.db"
_DATA_PREFIX = "data"


def _snapshot_db(src: Path, dst: Path) -> None:
    """Copy ``src`` DB to ``dst`` using SQLite's online backup API."""
    source = sqlite3.connect(str(src))
    try:
        target = sqlite3.connect(str(dst))
        try:
            with target:
                source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def create_backup(
    dest_dir: Path | str,
    *,
    db_file: Path | None = None,
    data_root: Path | None = None,
) -> Path:
    """Write a timestamped ``tar.gz`` of the DB + artifacts to ``dest_dir``.

    Returns the path to the created archive.

    Consistency model: the DB is captured atomically (online backup API) but the
    artifact tar is a best-effort same-window copy, not a global snapshot. A
    concurrent project delete/upload can leave the archive with orphaned files
    (DB has no row → reaped by ``core.janitor`` on restore) or, more rarely, a
    DB row whose file was created/removed mid-walk (serving fails closed on a
    missing artifact; the next retrain/redeploy regenerates). Run backups during
    low write activity; coordinating a true global snapshot is out of scope.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    db_file = db_file or db_path()
    data_root = data_root or storage.data_root()

    # Exclude exactly the live DB file and its WAL/SHM sidecars (captured via the
    # online backup API) — match by resolved path, never by basename, so a real
    # artifact like ``uploads/<id>/automodeler.db.csv`` is still backed up.
    excluded = {
        db_file.resolve(),
        db_file.with_name(db_file.name + "-wal").resolve(),
        db_file.with_name(db_file.name + "-shm").resolve(),
    }

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    archive = dest_dir / f"automodeler-backup-{stamp}.tar.gz"

    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / _DB_ARCNAME
        if db_file.exists():
            _snapshot_db(db_file, snapshot)

        with tarfile.open(archive, "w:gz") as tar:
            if snapshot.exists():
                tar.add(snapshot, arcname=_DB_ARCNAME)
            if data_root.exists():
                for path in sorted(data_root.rglob("*")):
                    if not path.is_file() or path.resolve() in excluded:
                        continue
                    rel = path.relative_to(data_root)
                    tar.add(path, arcname=f"{_DATA_PREFIX}/{rel.as_posix()}")

    return archive


def restore_backup(
    archive: Path | str,
    *,
    db_file: Path | None = None,
    data_root: Path | None = None,
) -> None:
    """Restore the DB and artifact tree from an archive made by ``create_backup``.

    The live DB file is replaced and artifact files are extracted into
    ``data_root``. Every archive member is confined with ``assert_within`` before
    extraction to defend against path-traversal in an untrusted archive.
    """
    archive = Path(archive)
    db_file = db_file or db_path()
    data_root = data_root or storage.data_root()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if name == _DB_ARCNAME:
                target = db_file
            elif name.startswith(f"{_DATA_PREFIX}/"):
                rel = name[len(_DATA_PREFIX) + 1 :]
                target = assert_within(data_root, data_root / rel)
            else:
                continue  # unknown member — ignore
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Stream — a model/dataset member can be larger than free RAM.
            with extracted, open(target, "wb") as out:
                shutil.copyfileobj(extracted, out)

    # Drop any stale WAL/SHM left from before the restore so the restored DB
    # file is the single source of truth on next open.
    for suffix in ("-wal", "-shm"):
        sidecar = db_file.with_name(db_file.name + suffix)
        if sidecar.exists():
            sidecar.unlink()


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="AutoModeler backup/restore")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup", help="Create a backup archive")
    b.add_argument("dest_dir", help="Directory to write the archive into")
    r = sub.add_parser("restore", help="Restore from a backup archive")
    r.add_argument("archive", help="Path to a .tar.gz created by 'backup'")
    args = parser.parse_args(argv)

    if args.cmd == "backup":
        path = create_backup(args.dest_dir)
        print(f"Backup written: {path}")
    elif args.cmd == "restore":
        restore_backup(args.archive)
        print(f"Restored from: {args.archive}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(_main())
