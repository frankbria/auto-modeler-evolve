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

### Column Cardinality Guards
When accepting a column as a grouping/segmenting parameter:
- Reject if `n_unique > 50` (absolute high-cardinality)
- Reject if `n_unique >= n_rows * 0.8` (relative near-unique — catches continuous numeric columns)
- Return HTTP 400 with a plain-English explanation

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
