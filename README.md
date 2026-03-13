# AutoModeler

AI-powered, conversational data modeling platform. Upload a spreadsheet, chat with an
AI assistant, and walk away with a validated ML model running behind a live API and
interactive dashboard. No code required.

Built for **business analysts** who know their data but don't write code.

## What It Does

1. **Upload** a CSV -- AutoModeler instantly profiles it (row counts, types, patterns, anomalies)
2. **Explore** via natural language -- "Which products are trending up?" returns charts and stats
3. **Shape** features -- AI suggests transformations, you approve/reject through conversation
4. **Model** -- recommends and trains appropriate algorithms, shows plain-English comparisons
5. **Validate** -- cross-validation, confusion matrices, feature importance, per-row explanations
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
- Anthropic API key

### Backend

```bash
cd src/backend
uv venv
uv sync
ANTHROPIC_API_KEY=sk-... uv run uvicorn main:app --reload --port 8000
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
# Backend (155 tests)
cd src/backend
uv run pytest

# Frontend
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
| `GET /api/validate/{id}/explain` | Feature importance (SHAP-lite) |
| `POST /api/deploy/{id}` | Deploy model as API |
| `POST /api/predict/{id}` | Make predictions |
| `POST /api/predict/{id}/batch` | Batch predictions (CSV) |

## How It's Built

AutoModeler is built on the [code-evolve](https://github.com/frankbria/code-evolve)
framework -- a self-evolving project builder. An AI agent reads `vision.md` and
`spec.md`, then autonomously implements features session after session.

```bash
# Run an evolution session
ANTHROPIC_API_KEY=sk-... ./scripts/evolve.sh
```

See `spec.md` for the full feature checklist and `JOURNAL.md` for the build log.

## Current Status

**Phases 1-6 complete** (foundation through deployment). Phase 7 (polish: onboarding,
project management, chat memory, export, responsive design) is next.

## License

MIT
