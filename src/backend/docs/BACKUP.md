# Backup & Restore Runbook (#10)

AutoModeler's durable state is the SQLite database **plus** the on-disk artifact
tree (uploads, models, deployment pipelines, batch outputs). They must be
backed up and restored **together** — a restore that recovers one but not the
other yields deployments that 404 on every prediction.

## What protects the data

| Concern | Mechanism |
|---|---|
| Concurrent writers locking the DB | `journal_mode=WAL` + `busy_timeout=30000` (set per connection in `db.py`) |
| Dangling references | `foreign_keys=ON` |
| Booting on a corrupt DB | `PRAGMA quick_check` at startup → **fail-fast** (app refuses to boot; `db.check_integrity`) |
| Disaster recovery | Coordinated backup archive via `core.backup` |

## Backup

Backups use SQLite's **online backup API**, so they are safe to take against a
live, running server (no torn pages — unlike a plain `cp` of a WAL-mode DB). The
same archive captures the artifact tree in the same window.

```bash
# From src/backend (the directory containing db.py / core/):
python -m core.backup backup /var/backups/automodeler
# → writes /var/backups/automodeler/automodeler-backup-YYYYMMDDTHHMMSS.tar.gz
```

The archive contains:

```
automodeler.db        # consistent DB snapshot (online backup API)
data/<relpath>...     # artifact tree, minus the live .db / .db-wal / .db-shm
```

### Schedule it (cron)

Daily at 02:00, keeping 14 days of archives:

```cron
0 2 * * *  cd /srv/automodeler/src/backend && /usr/bin/python -m core.backup backup /var/backups/automodeler && find /var/backups/automodeler -name 'automodeler-backup-*.tar.gz' -mtime +14 -delete
```

### Or systemd timer

`/etc/systemd/system/automodeler-backup.service`:

```ini
[Service]
Type=oneshot
WorkingDirectory=/srv/automodeler/src/backend
ExecStart=/usr/bin/python -m core.backup backup /var/backups/automodeler
```

`/etc/systemd/system/automodeler-backup.timer`:

```ini
[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl enable --now automodeler-backup.timer
```

## Restore (the drill)

> Stop the app first so nothing writes mid-restore.

```bash
python -m core.backup restore /var/backups/automodeler/automodeler-backup-YYYYMMDDTHHMMSS.tar.gz
```

This replaces the live DB file (clearing any stale `-wal`/`-shm`) and extracts
the artifact tree back under the data root. Then start the app — the startup
`quick_check` confirms the restored DB is sound.

**Restore is tested in CI**, not just documented: `tests/test_backup_restore.py`
backs up a DB + artifact, wipes both, restores, and asserts the row and the
artifact bytes come back (and that a path-traversal archive is rejected).

## Notes

- In production `DATA_DIR` is unset, so the DB and all artifacts live under one
  `<backend>/data` root and a single archive captures them coherently. If you
  relocate data with `DATA_DIR`, the DB still lives next to `db.py`'s data dir;
  pass explicit `db_file=` / `data_root=` to `core.backup` if you split them.
- Keep at least one off-box copy of the archives (rsync/object storage) — a
  local backup does not survive disk-full or hardware loss.
