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
- **ML:** scikit-learn, pandas, numpy
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
# Full evolution cycle
ANTHROPIC_API_KEY=sk-... ./scripts/evolve.sh

# With specific model
ANTHROPIC_API_KEY=sk-... MODEL=claude-opus-4-6 ./scripts/evolve.sh

# Force run (bypass schedule gate)
ANTHROPIC_API_KEY=sk-... FORCE_RUN=true ./scripts/evolve.sh
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
