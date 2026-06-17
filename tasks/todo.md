# Issue #4 — [P2.1] Cascade project delete + artifact GC (orphans, unbounded disk)

**Branch:** `fix/4-cascade-delete-artifact-gc`
**Severity:** HIGH (blocker) · data-integrity

## Problem
`delete_project` does `session.delete(project); session.commit()` with no cascade and
no artifact removal. ~23 child tables (4 levels deep) become orphans; trained models,
uploads, db_uploads and deployment pipelines are never removed. Orphaned Deployment rows
can still serve `/api/predict/{id}`. Disk grows unbounded → fills shared VPS → SQLite
writer goes offline. (Audit also cites test pollution: thousands of leaked dirs in the
real data tree, caused by import-time artifact-dir constants that ignore `DATA_DIR`.)

## Ownership graph (rooted at Project)
- **Direct (project_id):** Dataset, ModelRun, Deployment, Conversation, AnalysisTemplate
- **via dataset_id:** FeatureSet, DatasetFilter
- **via deployment_id:** PredictionLog, FeedbackRecord, DeploymentChangelog, DeploymentPreset,
  DashboardFieldConfig, InputValidationRule, PredictionAlertRule, SavedScenario,
  WebhookConfig, WebhookEvent, WeeklyDigestConfig, GoalSeekRecord, BatchSchedule,
  BatchJobRun, DeploymentVersion
- **via champion_id/challenger_id (= deployment.id):** ABTest

## Artifacts on disk
- `data/uploads/{project_id}/` (datasets)
- `data/db_uploads/{project_id}/` (extract-db)
- `data/models/{project_id}/{model_run_id}.joblib` (trained models)
- `data/deployments/{model_run_id}_pipeline.joblib` (deployment pipelines)

## Plan

1. **`core/storage.py`** — single source of truth for artifact paths. Base dir resolves at
   call time from `DATA_DIR` env, falling back to `<backend>/data` (current behaviour).
   Helpers: `uploads_dir()`, `db_uploads_dir()`, `models_dir()`, `deployments_dir()`,
   `project_upload_dir(pid)`, `project_db_uploads_dir(pid)`, `project_models_dir(pid)`,
   `deployment_pipeline_path(run_id)`, `project_artifact_paths(pid, run_ids)`.
   - Point the existing `UPLOAD_DIR / _DB_UPLOADS_DIR / MODELS_DIR / DEPLOY_DIR` constants at
     these helpers so writes and removals share one resolution (single source of truth).
   - Add a test asserting storage layout matches the api-module constants (anti-drift).

2. **`core/cascade.py`** — `delete_project_cascade(session, project_id)`:
   - Collect deployment_ids, dataset_ids, model_run_ids for the project.
   - Bulk-`delete()` all descendant rows in dependency order (grandchildren by
     deployment_id → children by dataset_id → direct children by project_id → project),
     all in **one transaction / single commit** → zero orphan rows.
   - After commit, remove artifact dirs/files via `core/storage` (guarded by
     `path_safety.assert_within`). Best-effort on files (FS isn't transactional); the
     janitor reaps any leftover.

3. **Wire `delete_project`** (api/projects.py) to call `delete_project_cascade`.

4. **`core/janitor.py`** — GC artifacts with no referencing DB row:
   - `collect_orphans(session)`: remove `uploads/`, `db_uploads/`, `models/` subdirs whose
     `{project_id}` has no Project row; remove `deployments/*_pipeline.joblib` whose
     `model_run_id` has no ModelRun/Deployment row.
   - `enforce_upload_retention(session, max_age_days=None)`: age sweep for orphaned uploads.
   - `run_janitor(session)`: convenience wrapper, returns a summary (dirs/files removed).
   - Wire best-effort invocation on app startup (lifespan), guarded so it never blocks boot.

5. **PRAGMA `foreign_keys=ON`** via SQLAlchemy `Engine "connect"` event in `db.py`
   (applies to all engines incl. test). Low risk: only the existing `owner_id` FK is
   declared, so enforcement adds "no project without a real user" without breaking inserts.
   **Deviation (criterion 2):** we do NOT add hard child-FK constraints with `ondelete=CASCADE`.
   Rationale: (a) the existing SQLite DB cannot gain FK constraints without a full table
   rebuild, so they'd never protect production data anyway; (b) enforcing child FKs would
   break existing insert/test patterns (the `backfill` fixture seeds only parent Projects).
   The transactional app-level cascade delivers the identical guarantee (zero orphans),
   verified by tests. Will confirm the full suite stays green with PRAGMA on; if it breaks
   unrelated tests, scope PRAGMA to the production engine and document.

6. **Tests** (`tests/test_project_cascade_delete.py`, `tests/test_janitor.py`):
   - Seed a project with rows in every child table + artifact dirs/files (via storage).
   - `DELETE /api/projects/{id}` → assert zero rows across ALL child tables + all artifact
     dirs/files removed (the criterion-4 assertion).
   - Cross-tenant: `second_client` DELETE → 404, nothing removed.
   - Janitor: orphan dir removed; live-project dir kept; retention sweep.
   - PRAGMA: assert `PRAGMA foreign_keys` returns 1 on a fresh connection.

## Acceptance criteria (from issue #4)
- [x] `delete_project` transactional: deletes all child rows + removes artifact dirs in one commit
      → `core/cascade.py::delete_project_cascade`, wired into `api/projects.py`.
- [x] `PRAGMA foreign_keys=ON` enabled (`db.py` Engine connect listener). Hard child-FK
      `ondelete=CASCADE` intentionally NOT added (documented deviation — brownfield SQLite
      can't be retrofitted + would break insert/test patterns; app-level cascade is the
      mechanism).
- [x] Janitor GCs artifact files with no referencing row + enforces upload retention
      → `core/janitor.py` (`collect_orphans`, `enforce_upload_retention`, `run_janitor`),
      best-effort on startup in `main.py` lifespan.
- [x] Test asserts zero orphan rows + removed dirs after delete
      → `tests/test_project_cascade_delete.py`, `tests/test_janitor.py`,
      `tests/test_storage_layout.py`.

## Implementation notes / extra changes
- **`core/storage.py`** (new): single source of truth for artifact paths; call-time
  `DATA_DIR` resolution (default = `<backend>/data`, identical to the legacy constants).
- **`tests/conftest.py`**: extended the autouse `backfill` fixture to also seed the owner
  `User` for any flushed `Project` (PRAGMA now enforces `owner_id → user.id`). The user
  INSERT is emitted immediately on the flush connection — a deferred `session.add` would
  race the Project insert because the bare column FK gives no ORM save-order dependency.
  See tasks/lessons.md.

## Status: implementation complete; full backend suite verified green.

Full suite (6663 tests) initially reported `3 failed, 6660 passed` — all three were
**test fragility surfaced only by the full ordered 53-min run**, not production bugs
(no production code changed to fix them):
- `test_storage_layout_matches_api_constants` — read api-module `*_DIR` globals that
  ~150 other tests reassign in-place without `monkeypatch`. Fixed: reload the modules
  to read declared constants.
- `test_uptime_degraded_day_status`, `test_uptime_active_day_has_predictions` —
  pre-existing flaky tests (unchanged since #24) that assert relative-to-now logs land
  on "today"; slips a day when run in the first UTC hour. Fixed: anchor logs to noon UTC.

See tasks/lessons.md for the patterns.
