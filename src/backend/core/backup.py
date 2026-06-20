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

import os
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

from core import storage
from core.path_safety import assert_within
from db import check_integrity, db_path

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
    # If the destination lives under data_root, never tar the archives themselves
    # (each new backup would otherwise swallow every prior one → exponential).
    dest_resolved = dest_dir.resolve()

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
                    if not path.is_file():
                        continue
                    resolved = path.resolve()
                    if resolved in excluded:
                        continue
                    # Skip anything under the backup destination directory.
                    if resolved == dest_resolved or dest_resolved in resolved.parents:
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

    Extracts the whole archive into a staging directory and validates the staged
    DB *before* touching live state, so a truncated/corrupt/path-traversing
    archive can't leave the live DB or artifacts half-overwritten. A DB-less
    archive is rejected — restoring one would wipe the live WAL/SHM (losing any
    committed-but-not-checkpointed pages) while leaving a stale DB behind.

    Every artifact member is confined with ``assert_within`` to defend against
    path-traversal in an untrusted archive.
    """
    archive = Path(archive)
    db_file = db_file or db_path()
    data_root = data_root or storage.data_root()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    # Stage on the same filesystem as the live DB so the final swap is an atomic
    # os.replace (a default /tmp staging dir is often a separate fs → EXDEV).
    with tempfile.TemporaryDirectory(dir=db_file.parent) as staging_str:
        staging = Path(staging_str)
        staged_db: Path | None = None
        staged_artifacts: list[tuple[Path, Path]] = []  # (staged, live target)

        # --- Extract everything into staging (no live mutation yet) ----------
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                if name == _DB_ARCNAME:
                    dest = staged_db = staging / _DB_ARCNAME
                    target = db_file
                elif name.startswith(f"{_DATA_PREFIX}/"):
                    rel = name[len(_DATA_PREFIX) + 1 :]
                    target = assert_within(data_root, data_root / rel)  # validate
                    dest = assert_within(staging, staging / _DATA_PREFIX / rel)
                else:
                    continue  # unknown member — ignore
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Stream — a model/dataset member can be larger than free RAM.
                with extracted, open(dest, "wb") as out:
                    shutil.copyfileobj(extracted, out)
                if name != _DB_ARCNAME:
                    staged_artifacts.append((dest, target))

        if staged_db is None:
            raise ValueError(
                f"Archive {archive} contains no database snapshot — refusing to "
                "restore (would wipe live WAL/SHM and leave a stale DB)."
            )
        # Fail before touching live state if the snapshot itself is corrupt.
        check_integrity(staged_db)

        # --- Swap staged files into place ------------------------------------
        # ponytail: the DB swap is atomic (os.replace); the multi-file artifact
        # move is best-effort — a disk-full mid-move could partially restore, but
        # the archive is already fully extracted+validated so that window is tiny.
        os.replace(staged_db, db_file)
        for staged, target in staged_artifacts:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged), str(target))

    # Drop any stale WAL/SHM from before the restore so the restored DB file is
    # the single source of truth on next open (safe now: the DB was replaced).
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
