# Lessons

## SQLite FK enforcement + SQLAlchemy save-order (issue #4, 2026-06-16)

**Pattern:** Enabling `PRAGMA foreign_keys=ON` makes the existing
`Project.owner_id → user.id` FK actually enforce. Many tests insert `Project`
rows (directly or via the `backfill` fixture) without a matching `User`, so the
suite broke with `FOREIGN KEY constraint failed`.

**Wrong fix attempted:** Adding the missing `User` to the session inside a
`before_flush` listener (`session.add(user)`). This does NOT work — without an
ORM `relationship()` between Project and User, a bare column-level FK gives the
unit-of-work no save-order dependency, so the pending `Project` insert races
ahead of the just-added `User`.

**Correct fix:** Emit the `User` INSERT **immediately** on the flush connection
inside `before_flush` (`session.connection().execute(insert(User.__table__)...)`)
so the row exists before the ORM emits the `Project` insert. Core inserts bypass
Python-side defaults, so non-null columns without a SQL default (e.g.
`created_at`) must be supplied explicitly.

## Brownfield SQLite can't be retrofitted with FK constraints
SQLite cannot `ALTER TABLE ADD CONSTRAINT` a foreign key; the existing on-disk
DB has none. So `ondelete=CASCADE` on child tables would never protect prod data
and would break the test insert patterns. App-level transactional cascade
(`core/cascade.py`) is the reliable mechanism; PRAGMA only enforces the one
declared `owner_id` FK going forward.

## Import-time path constants ignore env → test pollution
`UPLOAD_DIR`/`MODELS_DIR`/`DEPLOY_DIR` are import-time `Path(__file__)...`
constants that ignore `DATA_DIR`, so test artifact writes leaked into the real
`src/backend/data` tree. `core/storage.py` resolves the root at call time from
`DATA_DIR`; the cascade/janitor use it so tests isolate to `tmp_path`.

## Full-suite-only failures = global-state pollution / clock flakiness (issue #4)
Three tests passed in isolation but failed in the 53-min full run. None were
production bugs — all were test fragility surfaced only by the long, ordered run:
- **`test_storage_layout_matches_api_constants`**: ~150 test files do
  `data_module.UPLOAD_DIR = tmp_path / "uploads"` (direct assignment, **no**
  `monkeypatch`), so the api-module `*_DIR` globals stay polluted for every later
  test. My anti-drift test was the only one asserting the *declared* value. Fix:
  `importlib.reload(...)` each module inside the test to read its declared
  constant (immune to prior-test reassignment) — don't read a global other tests
  freely mutate.
- **`test_uptime_degraded_day_status` / `test_uptime_active_day_has_predictions`**
  (pre-existing, unchanged since #24): pure-function tests that placed logs
  "1 hour"/"~15-40 min" ago and asserted they bucket to *today*. `utcnow()` day
  bucketing slips to the previous calendar day when the suite runs in the first
  UTC hour (local MST = UTC-7, so a ~17:00-18:00 MST run is 00:00-01:00 UTC).
  Fix: anchor test logs to **noon UTC today**, collapsing the flake window to the
  midnight microsecond. Lesson: never assert "today" on a relative-to-now offset
  small enough to cross midnight.
