# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## What This Is

**AutoModeler** — an AI-powered, conversational data modeling platform. Users upload
data, explore it through chat, build ML models with AI guidance, and deploy predictions
as APIs and dashboards. No code required.

Built on the **code-evolve** framework: a self-evolving project builder that reads
`vision.md` and `spec.md`, then autonomously implements features session after session.

## How It Works

1. `scripts/evolve.sh` orchestrates each evolution session
2. It reads `vision.md`, `spec.md`, current project state, and GitHub issues
3. Claude Code implements the next highest-priority work from the spec
4. Build verification runs (auto-detected from project stack)
5. Journal entry is written, issue responses posted
6. Changes are committed, tagged, and pushed

## Project Structure

```
auto-modeler-evolve/
├── vision.md              # Project vision — the "why" (DO NOT lose sight of this)
├── spec.md                # Technical spec with feature checklist
├── IDENTITY.md            # Agent constitution (DO NOT MODIFY)
├── JOURNAL.md             # Evolution log (append at top, never delete)
├── LEARNINGS.md           # Cached knowledge from research
├── DAY_COUNT              # Current evolution day
├── scripts/
│   ├── evolve.sh          # Master orchestrator (DO NOT MODIFY)
│   ├── format_issues.py   # Issue sanitization (DO NOT MODIFY)
│   └── detect_stack.sh    # Stack detection for build verification
├── skills/                # Agent behavior definitions
│   ├── evolve/            # Build features from spec
│   ├── self-assess/       # Gap analysis: spec vs implementation
│   ├── communicate/       # Journal and issue responses
│   ├── research/          # Web research
│   └── plan/              # Planning from vision/spec
├── .github/workflows/
│   ├── evolve.yml         # Cron trigger (every 4h)
│   └── ci.yml             # Build verification on push/PR
└── src/                   # THE APPLICATION
    ├── backend/           # FastAPI (Python, uv)
    └── frontend/          # Next.js (TypeScript, npm)
```

## Tech Stack

### Backend
- **Language:** Python 3.12+
- **Framework:** FastAPI
- **Package Manager:** uv (`uv venv`, `uv sync`, `uv run pytest`)
- **Database:** SQLite via SQLModel
- **ML:** scikit-learn, pandas, numpy, joblib
- **LLM:** Anthropic SDK (Claude) for chat orchestration
- **Testing:** pytest + pytest-bdd (no mocking — real services)
- **Linting:** ruff + black

### Frontend
- **Framework:** Next.js 15 (App Router)
- **UI Kit:** Shadcn/UI (Nova template, gray color scheme)
- **Icons:** Hugeicons (@hugeicons/react) — NOT lucide-react
- **Font:** Nunito Sans
- **Charts:** Recharts
- **State:** Zustand
- **Testing:** Jest (unit) + Playwright (E2E)

## Established Patterns (follow these exactly)

### Chat Intent → SSE Card Pattern
Every conversational feature follows this exact pipeline. Do not deviate:

1. **Regex constant** at module level in `src/backend/api/chat.py`:
   ```python
   _FOO_PATTERNS = re.compile(r"(?i)\b(phrase one|phrase two|...)\b", re.IGNORECASE)
   ```
   - Do NOT add a trailing `\b` after alternation groups ending in `\w` — it causes false negatives
   - Include 6–9 natural-language variants covering how business analysts actually speak

2. **Handler block** inside `send_message()`, guarded by `_FOO_PATTERNS.search(body.message)`:
   ```python
   foo_event: dict | None = None
   if _FOO_PATTERNS.search(body.message) and ctx["dataset"]:
       try:
           # ... compute result ...
           foo_event = {...}
           system_prompt += "\n\n## <Context for LLM narration>"
       except Exception:  # noqa: BLE001
           pass  # Nice-to-have; never crash chat
   ```
   The `except Exception: pass` is intentional — rich cards are enhancement, not core.

3. **SSE emit** in the generator's yield section, after LLM streaming:
   ```python
   if foo_event:
       yield f"data: {json.dumps({'type': 'foo_result', 'foo_result': foo_event})}\n\n"
   ```

4. **Frontend card component** at `src/frontend/components/<domain>/<feature>-card.tsx`
   - TypeScript types in `src/frontend/lib/types.ts`
   - API method in `src/frontend/lib/api.ts`
   - Zustand action in `src/frontend/lib/store.ts` (`attachFooToLastMessage`)
   - Unit tests in `src/frontend/__tests__/<feature>-card.test.tsx`

### Working DataFrame Convention
Always call `_load_working_df` with positional args in this exact order:
```python
_df = _load_working_df(file_path, _active_filter_conditions)
```
NOT `_load_working_df(dataset, session)` — that is a recurring bug pattern.

### Pure Functions in `core/`
Analytics functions (e.g., `compute_segment_performance()`) must:
- Accept plain Python arrays/lists, not ORM objects
- Have no database dependencies
- Be fully testable in isolation without a running server

### Artifact Storage & Cascade Delete (#4)
On-disk artifacts (uploads, db_uploads, models, deployment pipelines, batch
outputs) have ONE source of truth: `core/storage.py`, which resolves the root at
call time from `DATA_DIR` (default `<backend>/data`). Follow this exactly:
- **Never hardcode** `Path(__file__).parent.parent / "data" / ...`. New artifact
  paths go through a `core.storage` helper; writer modules point their `*_DIR`
  constants at those helpers so writes and the cascade/janitor cleanup share one
  root (a writer that ignores `DATA_DIR` while cleanup honors it silently leaks
  every file once `DATA_DIR` is set — that was the #4 footgun).
- **Deleting a parent must cascade.** `delete_project` calls
  `core.cascade.delete_project_cascade` — a leaf-first bulk delete of the whole
  ownership subtree in one transaction (zero orphans), then artifact removal
  (path_safety-guarded) post-commit. Adding a new child table? Add it to the
  cascade and to `tests/test_project_cascade_delete.py`'s coverage.
- **Orphans are reaped** by `core/janitor.py` (startup sweep). Any new artifact
  type needs a janitor rule + upload-style retention if it can leak.
- SQLite FK enforcement is ON (`db.py` `PRAGMA foreign_keys`); brownfield child
  FKs are NOT retrofitted — the app-level cascade is the mechanism.

### Column Cardinality Guards
When accepting a column as a grouping/segmenting parameter:
- Reject if `n_unique > 50` (absolute high-cardinality)
- Reject if `n_unique >= n_rows * 0.8` (relative near-unique — catches continuous numeric columns)
- Return HTTP 400 with a plain-English explanation

### Authentication & Owner/Tenant Authorization
Auth is **backend-native** (FastAPI owns it — BetterAuth is frontend-only and cannot
protect the Python API). Follow this exactly when adding endpoints:
- **Auth**: `User` model + `/api/auth` (register/login/me); bcrypt passwords, JWT
  (HS256) signed with `AUTH_SECRET` (see `src/backend/.env.example`). `get_current_user`
  → 401 on missing/invalid token.
- **Ownership** lives on `Project.owner_id` (non-nullable). Child entities cascade-scope
  through their owning project — do NOT add `owner_id` to child tables.
- **Every new management router** gets `dependencies=[Depends(require_owner)]` (auth +
  path-resource ownership in one line). `require_owner` is **soft on missing** resources
  (only blocks cross-tenant *existing* ones) so handlers keep their own 404/empty
  semantics; cross-tenant → 404 (never 403, to avoid existence leaks).
- **Resource ids in the body/query** (not the path) are NOT covered by `require_owner` —
  authorize them explicitly with `get_owned_*` / `assert_owns_project` (see
  `api/data.py` upload/merge/join-keys).
- **Public surface**: `/api/predict/*` serving stays unauthenticated (per-deployment API
  keys); the deploy router uses `require_owner_allow_public`. The anonymous
  `/predict/[id]` page loads a single shared deployment via the public mirrors
  `GET /api/predict/{id}/{info,presets,dashboard-config,dashboard-metadata}` — thin
  handlers that reject inactive deployments and call `_verify_api_key`. NEVER load the
  public page from an owner-scoped `/api/deploy/*` endpoint (that was the #25 IDOR).
- **Frontend is auth-aware** (#25): `lib/auth-token.ts` stores the JWT in localStorage;
  the single `apiFetch` chokepoint in `lib/api.ts` injects `Authorization: Bearer` on
  every management call and, on 401, clears the token + redirects to `/login` (public
  `/api/predict/*` URLs are exempt — they use API keys, not the user JWT). A
  `<RequireAuth>` client guard wraps owner-scoped pages; `/login` + `/predict/[id]` stay
  public. Header-less callers (SSE/EventSource, file downloads) can't add the Bearer
  header — streams use the header-authenticated `streamSSE`, and owner-scoped downloads
  use `downloadFile(url, fallbackFilename?)` / the `useDownload` hook (#28): fetch the
  blob through `apiFetch`, read the filename from `Content-Disposition`, trigger a
  client-side object-URL download. NEVER render an owner-scoped `<a href download>` /
  `window.open` (it 401s) and do NOT put the JWT in a URL query.
- **Tests**: the shared `client` fixture is transparently authenticated; use `anon_client`
  for 401 assertions and `second_client` for cross-tenant (IDOR) tests
  (`tests/test_tenant_isolation.py`).

### Ingestion Security (SSRF / path-traversal / file-read — #3)
Any handler that makes an outbound request to a user-influenced URL, writes a file under
a user-influenced name, or opens a user-supplied path MUST reuse these guards — do NOT
hand-roll checks (that is how P1.3 happened):
- **Outbound URLs** (URL import, webhooks): `core.ssrf_guard.assert_safe_url(url,
  allow_unresolved=...)` before connecting (`allow_unresolved=True` at *registration*,
  `False` at *fetch/dispatch*), and fetch via `safe_urlopen` (re-validates every redirect
  hop). The guard allowlists by routability (`not ip.is_global`), so loopback/link-local/
  RFC1918/ULA/CGNAT are all blocked.
- **File writes**: `core.path_safety.sanitize_filename(name)` then `resolve_within(base_dir,
  name)` — never join a raw filename onto an upload dir.
- **Opening an uploaded path** (e.g. `extract-db` `db_path`): `assert_within(base_dir, path)`
  to confine it (out-of-dir → 404, no existence leak), and open SQLite via
  `_open_readonly_sqlite` (`mode=ro` + extension loading off + ATTACH/DETACH denied).
- `UnsafeURLError`/`UnsafePathError` are mapped to HTTP 400 by app-level handlers in
  `main.py`; for confinement-as-not-found cases catch `UnsafePathError` and raise 404.
- Negative tests live in `tests/test_ingestion_security.py`.

## UX North Star

This is built for **business analysts**, not data scientists. Every interaction should
feel like chatting with a smart colleague. Key principles:

1. **Chat is the primary interface** — everything can be done through conversation
2. **No jargon without explanation** — always include plain-English equivalents
3. **Show, don't tell** — every insight gets a visualization
4. **Progressive disclosure** — start simple, reveal complexity on demand
5. **Fail gracefully** — no stack traces, always suggest next steps

## Running Locally

```bash
# Full evolution cycle (uses OAuth from `claude auth login` or env var)
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... ./scripts/evolve.sh

# With specific model
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... MODEL=claude-opus-4-6 ./scripts/evolve.sh

# Force run (bypass schedule gate)
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... FORCE_RUN=true ./scripts/evolve.sh
```

## Safety Rules

These files are protected and must never be modified by the agent:
- `IDENTITY.md` — agent constitution
- `scripts/evolve.sh` — orchestrator
- `scripts/format_issues.py` — input sanitization
- `.github/workflows/` — CI/CD safety net

## State Files

| File | Purpose | Mutability |
|------|---------|-----------|
| vision.md | North star | Human-edited |
| spec.md | Blueprint with checklist (Phase 8+ items are perpetual) | Human + agent (checkboxes) |
| JOURNAL.md | Session log | Append-only (top) |
| LEARNINGS.md | Cached knowledge | Append (new sections) |
| BACKLOG.md | Evolution coordination + ideation board | Agent-managed (read before, update after) |
| DAY_COUNT | Evolution day | Written each run |
