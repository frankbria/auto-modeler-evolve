# AutoModeler

AI-powered, conversational data modeling platform. Upload a spreadsheet, chat with an
AI assistant, and walk away with a validated ML model running behind a live API and
interactive dashboard. No code required.

Built for **business analysts** who know their data but don't write code.

## What It Does

1. **Upload** a CSV -- AutoModeler instantly profiles it (row counts, types, patterns, anomalies)
2. **Explore** via natural language -- "Which products are trending up?" returns charts and stats
3. **Shape** features -- say "suggest features" or "apply transformations" to engineer features through conversation
4. **Model** -- recommends and trains appropriate algorithms, shows plain-English comparisons
5. **Validate** -- cross-validation, confusion matrices, feature importance, per-row explanations, and segment-level performance breakdowns
6. **Deploy** -- one click to a live prediction API endpoint + shareable dashboard

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12+, FastAPI, SQLModel (SQLite) |
| ML | scikit-learn, pandas, numpy |
| LLM | Anthropic SDK (Claude) for chat orchestration |
| Frontend | Next.js 15, React 19, TypeScript |
| UI | Shadcn/UI (Nova), Tailwind CSS, Hugeicons |
| Charts | Recharts |
| State | Zustand |
| Package Mgmt | uv (backend), npm (frontend) |

## Project Structure

```
src/
├── backend/
│   ├── api/           # FastAPI route handlers (chat, data, features, models, validation, deploy)
│   ├── core/          # Business logic (analyzer, feature_engine, trainer, validator, explainer, deployer)
│   ├── chat/          # LLM orchestration (prompts, narration, state machine)
│   ├── models/        # SQLModel database models
│   ├── db.py          # Database setup
│   └── main.py        # App entry point
└── frontend/
    ├── app/           # Next.js App Router pages
    │   ├── project/[id]/  # Workspace (chat + data + features + models + validation + deploy)
    │   └── predict/[id]/  # Public prediction dashboard
    ├── components/    # React components (chat, data, features, models, deploy, ui)
    └── lib/           # API client, Zustand stores, types
```

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic OAuth token (via `claude auth login` or `ANTHROPIC_AUTH_TOKEN` env var)

### Backend

```bash
cd src/backend
uv venv
uv sync
ANTHROPIC_AUTH_TOKEN=sk-ant-oat01-... uv run uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd src/frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:3000` and expects the backend at `http://localhost:8000`.

### Running Tests

```bash
# Backend (1557 tests)
cd src/backend
uv run pytest

# Frontend (680 tests)
cd src/frontend
npm test
```

## API Overview

| Endpoint | Purpose |
|----------|---------|
| `POST /api/chat/{project_id}` | Streamed chat (SSE) |
| `POST /api/data/upload` | CSV upload + auto-profiling |
| `GET /api/data/{id}/profile` | Data quality report |
| `GET /api/data/{id}/query` | Natural language data queries |
| `GET /api/features/{id}/suggestions` | AI feature suggestions |
| `POST /api/features/{id}/apply` | Apply transformations |
| `POST /api/models/{id}/train` | Train models |
| `GET /api/models/{id}/compare` | Compare trained models |
| `GET /api/models/{id}/segment-performance` | Per-segment R² / accuracy breakdown |
| `GET /api/validate/{id}/explain` | Feature importance (SHAP-lite) |
| `GET /api/models/{id}/report` | Download PDF model report |
| `POST /api/deploy/{id}` | Deploy model as API |
| `POST /api/predict/{id}` | Make predictions |
| `POST /api/predict/{id}/batch` | Batch predictions (CSV) |

## Conversational Capabilities

The chat interface understands natural language for every step of the workflow. Key chat-triggered features:

| Say something like... | What happens |
|-----------------------|-------------|
| "suggest features" / "recommend transformations" | Inline `FeatureSuggestCard` with transform badges + one-click Apply All |
| "apply the feature suggestions" | Applies all transformations, emits `FeatureSuggestCard` confirmation |
| "how does my model perform by region?" | Inline `SegmentPerformanceCard` showing per-group R² or accuracy with best/worst labels |
| "model accuracy by product" | Same segment breakdown, auto-detected grouping column |
| "generate a report" / "pdf report" | Inline `ReportReadyCard` with direct download link |

## How It's Built

AutoModeler is built on the [code-evolve](https://github.com/frankbria/code-evolve)
framework -- a self-evolving project builder. An AI agent reads `vision.md` and
`spec.md`, then autonomously implements features session after session.

```bash
# Run an evolution session (uses OAuth from `claude auth login` or env var)
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... ./scripts/evolve.sh
```

See `spec.md` for the full feature checklist and `JOURNAL.md` for the build log.

## Current Status

**Phases 1-8 in progress** (foundation through advanced conversational features).
2,237 tests passing (1,557 backend + 680 frontend).

Recently shipped:
- **Model performance by segment** — ask "how does my model perform by region?" for a per-group breakdown with gap analysis and retraining recommendations
- **Chat-driven feature engineering** — the full Upload → Explore → Shape → Train → Deploy workflow is now 100% conversational
- **Chat-triggered PDF reports** — "generate a report" delivers a shareable PDF inline in chat

Phase 9 (onboarding, project management, export, responsive design) is next.

## License

MIT
