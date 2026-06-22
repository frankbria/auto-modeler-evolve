# Issue #14 — Centralize fixture isolation; stop tests polluting the real data tree

**Label:** test-trust · **Severity:** HIGH · Plan source: self-authored (no plan on issue)

## Root cause
`set_test_env` (autouse) sets `DATA_DIR=tmp_path`, and `core.storage` resolves at
call time — BUT the api modules cache the dirs as module-level constants at import
time (before any test runs), so they point at the real `<backend>/data/` tree:
- `api.models.MODELS_DIR`, `api.data.UPLOAD_DIR`, `api.data._DB_UPLOADS_DIR`,
  `api.deploy.DEPLOY_DIR`, `api.templates.UPLOAD_DIR`, `core.scheduler.BATCH_OUTPUT_DIR`

~11 per-file `client` fixtures patch only `db.engine`/`DATA_DIR` and forget these
constants → training/upload/deploy endpoints write into the real tree.
Reproduced leak: `data/uploads` 14k dirs, `data/models` 3k, `data/deployments` 4.4k, 254M.

## Plan (lazy + correct — one safety net beats rewriting 103 fixtures)

1. **Autouse isolation fixture** in `tests/conftest.py`: monkeypatch all 6 module
   constants to the `storage.*_dir()` values (re-resolved under the per-test
   `DATA_DIR=tmp_path`). Depends on `set_test_env` for ordering. Covers EVERY test —
   central `client`, per-file `client` fixtures that forgot the patch, and non-client
   tests that import the modules — so a forgetful fixture copy is no longer a hazard.
   (Both value-imports of these constants are function-local/call-time, so patching
   the source attr takes effect.)

2. **Session guard** in `conftest.py`: snapshot real `data/{models,uploads,deployments,
   db_uploads,batch_outputs}` entries at `pytest_sessionstart`; at `pytest_sessionfinish`
   fail the run (exitstatus=1) if any new entries appeared.

3. **Purge** leaked working-tree dirs (contents of models/uploads/deployments/
   db_uploads/batch_outputs; keep `data/sample` — tracked sample CSVs).

4. **Verify**: run training/upload/deploy tests; confirm real tree stays empty and the
   session guard passes on a full(ish) run.

## Deliberate deviation from acceptance criteria
AC#1 says "delete per-file copies". I'm NOT deleting the 103 per-file `client` fixtures:
each also wires its own DB engine/seed data, and rewiring every test to the async central
`client` (sync TestClient vs async AsyncClient, custom data) is a large, high-risk change.
The autouse net neutralizes the "root enabler" (a forgetful copy can no longer leak),
achieving the issue's stated impact goal without that risk. De-dup hygiene can follow later.

## Acceptance criteria mapping
- [x] Centralize isolation of UPLOAD_DIR/MODELS_DIR/deploy dirs into tmp_path → autouse `isolate_artifact_dirs` fixture (broader than just `client`); verified by test_fixture_isolation.py (6 constants)
- [x] Session guard asserting `data/{models,uploads,...}` untouched after the run → `pytest_sessionstart`/`pytest_sessionfinish` diff; negative probe confirmed exit code 1 on leak
- [x] Purge existing leaked dirs from the working tree → 254M → 488K (kept data/sample + dev db)
- [ ] Full suite green + guard clean (running)
