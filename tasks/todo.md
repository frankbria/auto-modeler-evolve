# Issue #6 — [P2.3] Stop auto-retrain rebinding live feature_set before training succeeds

**Source plan**: issue body (`PLAN_SOURCE = comment` — plan lives in the issue body itself).
**Severity**: HIGH (blocker), data-integrity.

## Root cause
`core/retrain.py::_do_trigger`:
1. Mutates `feature_set.dataset_id = new_dataset_id` (line 69) and **commits** (line 81) on the
   feature_set that the **currently-deployed** ModelRun still references — *before* training runs.
2. Training can `return None` (lines 89–110: missing dataset file, empty `feature_cols`) or raise in
   the background thread. Either way the live model's feature_set is permanently rebound to a new
   dataset with no rollback → provenance + retrain reproducibility corrupted.
3. Lines 118–119 mutate `_training_queues[project_id]` / `_training_counters[project_id]` without
   `_lock`, racing `start_training`/chat-driven training (which do hold `_lock`) → clobbered
   queue/counter and SSE sentinel races.

## Fix design — Option A: create a new run-scoped FeatureSet (NOT mutate the existing one)
- The existing feature_set (referenced by the deployed run) is **never touched**.
- A new `FeatureSet` row (copy of `transformations` / `column_mapping` / `target_column` /
  `problem_type`, `is_active=True`) points at `new_dataset_id`; the new `ModelRun` references it.
- All early-return ("skipped retrain") checks move **before any DB write**, so a skipped retrain
  leaves the DB completely untouched.
- Queue/counter setup is wrapped in `with _lock:` (matching `api/models.py:528` `start_training`).
- `column_mapping` is now copied for provenance completeness (previously dropped).

## Acceptance criteria (from issue)
- [x] Do not mutate the existing feature_set in place — create a new FeatureSet pointing at
      `new_dataset_id`.
- [x] A skipped/failed retrain must leave prior state untouched.
- [x] Acquire `_lock` around queue/counter setup.
- [x] Add a test forcing a failed retrain and asserting `feature_set.dataset_id` is unchanged.

## Adapted plan steps

### Step 1 — RED: tests first (TDD)
**File**: `src/backend/tests/test_auto_retrain.py` (append)
- `test_retrain_skipped_leaves_feature_set_unchanged`: project with a selected/done ModelRun whose
  FeatureSet points at `OLD`; call `trigger_auto_retrain(project, NEW)` where the NEW dataset's file
  does not exist on disk. Assert: returns `None`; original `FeatureSet.dataset_id == OLD`; no new
  FeatureSet or ModelRun rows were created. (This test FAILS on current code — it reproduces the bug.)
- `test_retrain_empty_feature_cols_leaves_state_untouched`: NEW dataset file exists but yields zero
  feature columns. Assert returns `None` and original feature_set unchanged, no new rows.
- `test_retrain_creates_new_feature_set_not_in_place`: success path — verify a SECOND FeatureSet row
  is created with `dataset_id == NEW`, the original FeatureSet keeps `dataset_id == OLD`, and the new
  ModelRun references the NEW feature_set's id.
- `test_retrain_queue_counter_set_under_lock`: smoke test that after a triggered retrain the project's
  queue/counter entries exist (the `_lock` acquisition is a code-level guarantee; this asserts the
  observable side-effect).

### Step 2 — GREEN: fix `core/retrain.py::_do_trigger`
- Capture `transformations`, `column_mapping`, `target_column`, `problem_type`, `algorithm` from the
  selected run's feature_set (read-only).
- Read new `Dataset`; if missing/no `file_path`/file-missing → `return None` (**before any write**).
- Exit session; load df, apply transforms; if `not feature_cols` → `return None` (**before any write**).
- Open a session: create NEW `FeatureSet(dataset_id=new_dataset_id, ...copy)`, flush for id, create
  `ModelRun(feature_set_id=new_fs.id, status="pending")`, commit, refresh.
- Wrap `_training_queues`/`_training_counters` setup in `with _lock:`.
- Launch background thread unchanged (now pointing at `new_run_id`).

### Step 3 — REFACTOR: lint/typecheck, run the new + existing retrain tests, then the broader suite
- `uv run ruff check` + `uv run black --check` on touched files.
- `uv run pytest tests/test_auto_retrain.py -q` (new tests pass, existing still pass).
- `uv run pytest -q` (no regressions; retrain-adjacent tests in particular).

## Test strategy → acceptance mapping
| Criterion | Test |
|---|---|
| Don't mutate existing feature_set | `test_retrain_creates_new_feature_set_not_in_place` |
| Skipped/failed leaves state untouched | `test_retrain_skipped_leaves_feature_set_unchanged` + `test_retrain_empty_feature_cols_leaves_state_untouched` |
| Acquire `_lock` | code review + `test_retrain_queue_counter_set_under_lock` |
| Test forcing failed retrain asserting dataset_id unchanged | `test_retrain_skipped_leaves_feature_set_unchanged` |

## Deviations / notes
- Chose Option A (new FeatureSet) over Option B (defer `dataset_id` to `status='done'`) because B
  would still mutate the *shared* feature_set that the old deployed run references — it just delays
  the corruption, it doesn't prevent it. A creates clean per-run provenance.
- Scope is strictly `core/retrain.py` + its tests. `MODELS_DIR` hardcoding in retrain.py:16 is a
  known `core.storage` concern but is **out of scope** for this issue (YAGNI) and not required by any
  acceptance criterion; left untouched.
