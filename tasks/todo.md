# Issue #19 — Fix performance hazards in prediction analytics & chat hot paths

**Source:** issue body (Phase 5.2, HIGH). Plan adapted to current codebase (line numbers verified live; issue's audit line numbers were stale).

## Adapted Plan

### Step 1 — Composite index `(deployment_id, created_at)` on PredictionLog
- `models/prediction_log.py`: add `__table_args__ = (Index("ix_predictionlog_dep_created", "deployment_id", "created_at"),)`.
  Drop the standalone `deployment_id` index (composite covers it as a left-prefix).
- `db.py`: add a `CREATE INDEX IF NOT EXISTS ...` step for existing SQLite DBs (current `_apply_migrations` only does ADD COLUMN). Idempotent.
- Test: assert the index exists via `PRAGMA index_list`.

### Step 2 — LRU model/pipeline cache keyed by `(path, mtime)` in the predict hot path
- `core/deployer.py`: add `load_pipeline_cached(path)` / `load_model_cached(path)` using
  `functools.lru_cache` keyed by `(path, os.path.getmtime(path))`. mtime keying = automatic
  invalidation: redeploy points the deployment at a new artifact path (or rewrites the file →
  new mtime), so the next load misses. No explicit invalidation hook needed.
- Use them in `predict_single` (deployer.py:316-317). Leave batch/explain/sweep loaders alone (not per-request).
- Test: patch `joblib.load`, assert 2 predictions on same artifact load once; a new path reloads.

### Step 3 — SQL pagination/sort: `get_prediction_logs` (deploy.py ~1746)
- Replace full `.all()` + Python `sorted()` + slice with `func.count` total +
  `order_by(created_at.desc()).offset(offset).limit(limit)`. Response shape unchanged.

### Step 4 — SQL aggregation: `get_deployment_analytics` (deploy.py ~1658)
- day counts → `GROUP BY func.date(created_at)`; total → `func.count`; recent_avg → `func.avg`;
  class_counts → `GROUP BY prediction`; histogram min/max → `func.min/max` + ONE bounded fetch for the
  10-bucket pass. Apply a bounded time window. Response shape unchanged.

### Step 5 — Kill N+1 in `deployments_overview` (deploy.py ~481)
- One grouped query for 7d/today counts over all owned deployment ids (replaces N count queries).
- One `select(Project).where(id.in_(...))` (replaces N `session.get`).

### Step 6 — Chat: window history + Anthropic prompt caching (chat.py)
- Window `api_messages` to the last ~30 turns (tail slice keeps last=user → contract preserved).
- `system` becomes a 2-block list: block 0 = stable base (`build_system_prompt(..., recent_messages=None)`,
  marked `cache_control={"type":"ephemeral"}`); block 1 = per-turn additions (cards + recent-conversation context).
  Factor recent-context formatting out of `build_system_prompt` so it lives in block 1.
  # ponytail: caches the largest stable prefix; hits only above Anthropic's min-cacheable floor.
- Update `test_anthropic_contract.py`: `system` is now `list[block]`, block 0 has `cache_control`, last msg still user.

## Acceptance Criteria
- [ ] Sort/pagination/aggregation pushed into SQL (LIMIT/OFFSET, COUNT/GROUP BY/AVG, bounded windows) — Steps 3,4
- [ ] N+1 replaced with a single grouped query — Step 5
- [ ] LRU model/pipeline cache keyed by `(path, mtime)`, invalidated on redeploy — Step 2
- [ ] Chat history windowed + prompt caching on the static system prompt — Step 6
- [ ] Composite index on `(deployment_id, created_at)` — Step 1

## Known limitations (deferred — not in acceptance criteria)
- Other analytics endpoints (drift, audit, confidence-trend, value-trend, covariate, usage) share the
  "materialize + Python-aggregate" pattern; several already cap their fetch. Full SQL rewrite of their
  per-feature/JSON statistics is out of scope. Noted as PR follow-up.
