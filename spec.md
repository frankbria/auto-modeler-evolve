# Specification

## Tech Stack

### Backend (Python)
- **Framework:** FastAPI
- **Package Manager:** uv
- **ML Libraries:** scikit-learn (core), pandas, numpy
- **Feature Engineering:** feature-engine, category-encoders
- **Explainability:** shap, lime
- **Data Profiling:** ydata-profiling (formerly pandas-profiling)
- **Model Serialization:** joblib
- **LLM Integration:** Anthropic SDK (Claude) for chat + analysis narration
- **Testing:** pytest + pytest-bdd
- **Linting:** ruff + black

### Frontend (TypeScript)
- **Framework:** Next.js 15 (App Router)
- **UI:** Shadcn/UI + Tailwind CSS (Nova template, gray scheme)
- **Icons:** Hugeicons (@hugeicons/react)
- **Font:** Nunito Sans
- **Charts:** Recharts (or Nivo for complex visualizations)
- **Chat UI:** Custom conversational interface with streaming responses
- **File Upload:** react-dropzone
- **State Management:** Zustand (lightweight, minimal boilerplate)
- **Testing:** Jest (unit) + Playwright (E2E)

### Database
- **Primary:** SQLite (lightweight, zero-config — perfect for single-user/small-team)
- **Purpose:** Store projects, datasets metadata, model runs, chat history
- **File Storage:** Local filesystem for uploaded CSVs and serialized models

### Deployment
- **Model Serving:** FastAPI endpoint (same server, dedicated route)
- **Dashboard:** Auto-generated Next.js page per deployed model
- **Target:** Single VPS deployment (monorepo, shared process)

---

## Architecture

### Monorepo Structure
```
auto-modeler-evolve/
├── src/
│   ├── backend/                  # FastAPI application
│   │   ├── api/                  # Route handlers
│   │   │   ├── chat.py           # Chat/conversation endpoints
│   │   │   ├── data.py           # Upload, preview, profiling
│   │   │   ├── features.py       # Feature engineering endpoints
│   │   │   ├── models.py         # Training, comparison, selection
│   │   │   ├── validation.py     # Model validation & explainability
│   │   │   └── deploy.py         # Model deployment & prediction
│   │   ├── core/                 # Business logic
│   │   │   ├── analyzer.py       # Data analysis & pattern detection
│   │   │   ├── feature_engine.py # Feature suggestion & transformation
│   │   │   ├── trainer.py        # Model training & comparison
│   │   │   ├── validator.py      # Cross-validation & metrics
│   │   │   ├── explainer.py      # SHAP/LIME explanations
│   │   │   └── deployer.py       # Model packaging & serving
│   │   ├── chat/                 # Chat orchestration
│   │   │   ├── orchestrator.py   # Conversation state machine
│   │   │   ├── prompts.py        # LLM prompt templates
│   │   │   └── narration.py      # Plain-English explanations
│   │   ├── models/               # Database models (SQLModel)
│   │   │   ├── project.py        # Project metadata
│   │   │   ├── dataset.py        # Dataset records
│   │   │   ├── feature_set.py    # Feature engineering history
│   │   │   ├── model_run.py      # Training runs & results
│   │   │   └── conversation.py   # Chat history
│   │   ├── db.py                 # Database connection & migrations
│   │   └── main.py               # FastAPI app entry point
│   │
│   └── frontend/                 # Next.js application
│       ├── app/
│       │   ├── page.tsx           # Landing / project list
│       │   ├── project/[id]/
│       │   │   ├── page.tsx       # Project workspace (chat + panels)
│       │   │   ├── data/page.tsx  # Data explorer
│       │   │   └── deploy/page.tsx# Deployment dashboard
│       │   └── predict/[id]/
│       │       └── page.tsx       # Public prediction dashboard
│       ├── components/
│       │   ├── chat/              # Chat interface components
│       │   ├── data/              # Data preview, stats, charts
│       │   ├── features/          # Feature cards, approval UI
│       │   ├── models/            # Model comparison, selection
│       │   ├── deploy/            # Deployment status, dashboard
│       │   └── ui/               # Shadcn components
│       └── lib/
│           ├── api.ts             # Backend API client
│           ├── store.ts           # Zustand stores
│           └── types.ts           # Shared TypeScript types
```

### Communication Pattern
- **Chat-first:** All user interactions flow through the chat interface
- **Side panels:** Data previews, charts, and model comparisons render in a
  split-panel layout alongside the chat
- **Streaming:** LLM responses stream to the frontend via SSE (Server-Sent Events)
- **Async jobs:** Long-running tasks (training, profiling) use background workers
  with progress updates pushed to the chat

### Conversation State Machine
The chat orchestrator tracks where the user is in the workflow:

```
START → UPLOAD → EXPLORE → SHAPE → MODEL → VALIDATE → DEPLOY
  ↑                                                       |
  └───────── (new project) ────────────────────────────────┘
```

Users can jump between states freely ("go back to exploring") but the AI gently
guides them forward through the natural flow.

---

## Features (Priority Order)

### Phase 1: Foundation (Days 0-3)
> Goal: A working chat interface that accepts data and shows basic analysis.

- [x] **Project scaffolding** — FastAPI backend + Next.js frontend in monorepo, with
      shared dev server configuration, CORS, and health check endpoint
- [x] **Database setup** — SQLite via SQLModel, with Project and Dataset tables,
      migrations via Alembic
- [x] **File upload** — Drag-and-drop CSV upload with progress indicator, file
      validation (size limits, CSV parsing), storage to local filesystem
- [x] **Data preview** — After upload, show first 10 rows in a clean table, column
      types, row count, and basic stats (min, max, mean, nulls) in a summary card
- [x] **Chat interface shell** — Split-panel layout: chat on the left, data/viz on
      the right. Text input with send button, message history, typing indicator.
      Streaming responses from backend via SSE
- [x] **Basic chat orchestration** — Connect chat to Claude API. System prompt
      includes dataset context (columns, types, sample rows). User can ask questions
      about their data and get natural-language answers

### Phase 2: Analysis & Exploration (Days 4-7)
> Goal: Users can ask questions and get visual, insightful answers about their data.

- [x] **Auto-profiling** — On upload, generate comprehensive data profile: distributions,
      correlations, missing value patterns, outlier detection. Cache results in DB
- [x] **Natural language data queries** — User asks "which region has highest sales?"
      → backend generates pandas query → returns result as text + chart
- [x] **Chart generation** — Bar, line, scatter, histogram, heatmap. Backend generates
      chart configs (Recharts-compatible JSON), frontend renders them inline in chat
- [x] **Pattern detection** — Automated insights: trends, seasonality, correlations,
      anomalies. Surfaced proactively in chat ("I noticed something interesting...")
- [x] **Data quality report** — Missing values, duplicates, type mismatches, outliers.
      Presented as actionable suggestions ("Column X has 12% missing — want to fill
      them with the median?")

### Phase 3: Feature Engineering (Days 8-11)
> Goal: AI suggests and applies feature transformations with user approval.

- [x] **Feature suggestions** — Based on column types and patterns, suggest
      transformations: date decomposition, categorical encoding, binning, log
      transforms, interaction features. Each with plain-English explanation
- [x] **Approval workflow** — Each suggestion shown as a card: what it does, why it
      might help, preview of the result. User approves/rejects/modifies via chat
      or button click
- [x] **Feature application** — Apply approved transformations, update dataset view,
      show before/after comparison
- [x] **Target variable selection** — Guide user to pick what they want to predict.
      Suggest classification vs regression based on target column type. Explain the
      difference in plain language
- [x] **Feature importance preview** — Quick correlation/mutual-information analysis
      to show which features are likely most predictive, before training

### Phase 4: Model Training (Days 12-16)
> Goal: Train, compare, and select models through conversation.

- [x] **Problem type detection** — Auto-detect classification vs regression from
      target variable. Confirm with user in plain language
- [x] **Model recommendations** — Suggest 2-4 appropriate algorithms based on dataset
      size, feature count, and problem type. Explain each in non-technical terms
      ("Random Forest: like asking 100 experts and taking a vote")
- [x] **Training execution** — Train recommended models with sensible defaults.
      Background thread training with real-time SSE push via EventSource subscription.
- [x] **Model comparison dashboard** — Side-by-side metrics (accuracy, precision,
      recall, R², MAE). Plain-English summary + auto-recommendation of best model
- [x] **Model selection** — User picks preferred model via button in Models tab;
      is_selected stored, chat acknowledgement sent

### Phase 5: Validation & Explainability (Days 17-20)
> Goal: Build trust through transparency — show what the model gets right and wrong.

- [x] **Cross-validation results** — K-fold validation with confidence intervals.
      Presented as "This model is consistently accurate, not just lucky on one split"
- [x] **Confusion matrix / error analysis** — For classification: visual confusion
      matrix with plain-English annotations. For regression: residual plots with
      explanations. Highlight where the model struggles
- [x] **Feature importance (SHAP)** — Global feature importance chart using sklearn
      native importances (tree: feature_importances_, linear: coef_). "The top 3
      factors driving predictions are: region, season, and product category"
- [x] **Individual prediction explanations** — Feature contribution waterfall for
      single predictions using linear attribution approximation.
      "For this specific case, the model predicted high revenue because..."
- [x] **Confidence & limitations** — Honest assessment of model limitations. "This
      model hasn't seen data from Q4 — predictions for holiday season may be less
      reliable"

### Phase 6: Deployment (Days 21-25)
> Goal: One-click deployment of model as API + interactive dashboard.

- [x] **Model packaging** — Serialize trained model + feature pipeline as a single
      deployable artifact. Include metadata: training date, features used, metrics
- [x] **Prediction API** — Auto-generated FastAPI endpoint: POST /api/predict/{model_id}
      with JSON input → JSON prediction output. Auto-generated OpenAPI docs
- [x] **Prediction dashboard** — Auto-generated Next.js page for each deployed model.
      Form with input fields matching feature columns. Submit → see prediction +
      explanation. Shareable URL
- [x] **Batch prediction** — Upload a CSV of new data → get predictions for all rows.
      Download results as CSV with prediction + confidence columns
- [x] **Deployment management** — List deployed models, view usage stats, undeploy.
      Simple status dashboard

### Phase 7: Polish & Delight (Days 26-30)
> Goal: Make it feel like working with a brilliant, patient colleague.

- [x] **Onboarding flow** — Empty-state panel with contextual description on first visit;
      guided tooltips in upload area. Sample dataset (200-row sales CSV) loads with one click.
- [x] **Project management** — Create, rename, delete, duplicate projects. Project
      list with last-modified, model status, quick stats (dataset name, row count, model count)
- [x] **Chat memory across sessions** — Resume conversations with "Welcome back" context
      message summarising last active time and conversation snippet
- [x] **Export & sharing** — Download model as .joblib pickle; PDF model report
      (reportlab, includes metrics, feature importance, confidence/limitations);
      public sharing link with one-click copy-to-clipboard in deployment panel.
- [x] **Responsive design** — Collapsible side panel toggle; topbar breadcrumb navigation;
      horizontal tab scroll; mobile Chat/Data toggle in topbar switches panels full-screen
      on small viewports; side-by-side layout preserved on md+ breakpoint.

### Phase 8: Continuous Evolution (Perpetual)
> Goal: Move beyond the initial spec. Research, ideate, and implement — guided by the
> vision, not a fixed checklist. Balance quality hardening with scope expansion.
>
> These items are **never checked off**. Each session, pick work from one or more of
> these tracks based on what will have the most impact right now.

#### Track A — Quality Hardening

- [x] **Gap analysis** — Compare what spec.md claims is done against the actual code.
      Does every [x] item truly work end-to-end? Are there shallow implementations that
      pass tests but don't deliver the full user experience described in the spec? Fix
      discrepancies and journal what you find.
      *Day 3 (18:00): Full analysis pass. All [x] items verified present. Two real gaps found and fixed: (1) NL query returns 500 instead of graceful fallback when API key missing; (2) self-demo revealed training requires `apply` before `set_target` — workflow is correct but demo script needed updating. No missing Phase 1-7 features detected.*
- [x] **E2E test build-out** — Expand Playwright coverage to the full user journey:
      upload CSV → explore data → ask questions → get charts → approve features → train
      models → validate → deploy → predict. Each critical path should have its own test
      file. Target: every spec phase has at least one E2E scenario.
      *Day 2 (10:00): 33 E2E tests — upload.spec.ts (10), training.spec.ts (8), deploy.spec.ts (9), home.spec.ts (6). Also fixed 2 real UX bugs: dataset state not restored on navigation, ModelTrainingPanel not loading existing runs on mount.*
- [~] **Unit test coverage to 100%** — Identify uncovered backend modules and frontend
      components. Write targeted tests for edge cases, error paths, and boundary
      conditions. Use `pytest --cov` and Jest coverage reports to find gaps.
      *Day 2 (20:05): query_engine.py 14%→92%, total backend 92%→95%. Remaining: frontend Jest coverage; explainer.py and validator.py edge paths.*
      *Day 3 (00:09): chart_builder 73%→100%, orchestrator 78%→100%, api/chat 37%→98%, total backend 94%→97%. 400 tests pass. Remaining: frontend Jest coverage.*
      *Day 3 (18:00): frontend Jest set up (next/jest + @testing-library/react + jest-fetch-mock); 69 unit tests covering store mutations, API client HTTP shapes, ChartMessage rendering (all 6 types), cn() utility. Frontend + backend = 469 total tests.*
      *Day 3 (06:00): 4 new frontend test suites (deployment-panel 17, model-training-panel 15, validation-panel 25, feature-suggestions-panel 25); api.ts 100% coverage. Total: 150 frontend + 530 backend = 680 tests. api.ts 100%, deployment-panel 99%, validation-panel 89%, feature-suggestions 38% (sub-components not yet tested).*
      *Day 3 (16:03): PipelinePanel (10 tests), DatasetListPanel (20 tests), FeatureImportancePanel (8 tests) — 38 new frontend tests for all feature-suggestions.tsx sub-components. Plus 2 new api.test.ts tests for uploadFromUrl. Total: 190 frontend + 545 backend = 735 tests, all passing.*
      *Day 3 (10:00): 74 new backend tests targeting api/features, api/validation, api/chat, api/deploy error paths; 15 frontend tests for app/page.tsx + predict/[id]/page.tsx (first app/ page coverage). SQLite connector adds 14 more backend tests. Total: 205 frontend + 630 backend = 835 tests. Backend coverage: 98%.*
- [x] **Integration tests** — Build tests that exercise real cross-boundary flows:
      upload → profile → chat about data (hits Claude API mock or stub) → train → deploy
      → predict. These complement E2E by testing backend flows without browser overhead.
      *Day 2 (14:00): 11 integration tests in test_integration_flow.py — cover upload, profile, feature suggestions, training, compare, deploy, single predict, batch predict, undeploy, multi-model comparison, narration, validation, and explainability. All 11 pass.*
- [x] **Self-demo capability** — Build a scripted demo that can run autonomously to
      prove the platform works. Upload sample data, run through the full workflow, capture
      screenshots or output at each stage. This becomes the smoke test and the showcase.
      *Day 3 (18:00): scripts/demo.py — 15-step autonomous smoke test (upload→NL query→feature suggestions→apply→target→train→compare→validate→importance→deploy→predict→batch→undeploy). 15/15 PASS in ~3s. Also fixed a real backend bug: NL query threw unhandled TypeError (not caught by anthropic.APIError) when ANTHROPIC_API_KEY missing.*
- [x] **Error resilience audit** — Systematically test failure modes: corrupt CSV,
      empty dataset, single-row data, all-null columns, model training failure, deployment
      of a terrible model. Verify every failure produces a helpful user-facing message.
      *Day 2 (20:05): 22 new tests; fixed 2 real bugs (NaN in preview rows, inf in histogram); training/deploy edge cases covered. Remaining: model training failure + terribly-performing model path.*
      *Day 3 (00:09): model training failure (run→failed, error_message populated), partial failure (1 algo fails, others continue), terrible model (low R², still deployable), constant target, all-failed narration. All paths covered.*
- [x] **Performance baseline** — Measure and record response times for key operations
      (upload profiling, model training, prediction). Establish baselines so future
      changes can be compared. Identify and fix any obvious bottlenecks.
      *Day 3 (04:31): 8 performance tests with real timings — upload 200 rows: 28ms, cached profile: 2ms, correlations: 2ms, feature suggestions: 6ms, linear regression train: 218ms, single prediction: 4ms. Results persisted to performance_baseline.json for future comparison.*

#### Track B — Vision-Driven Innovation

- [x] **Research external models and data sources** — Investigate integrating external
      ML models (XGBoost, LightGBM, neural networks via scikit-learn MLPClassifier),
      additional data connectors (Excel, Google Sheets, database connections), or
      pre-trained models for common use cases (sales forecasting, churn prediction).
      Document findings in LEARNINGS.md before implementing.
      *Day 3 (04:31): XGBoost 3.2.0 and LightGBM 4.6.0 integrated into trainer.py algorithm registries (both regression + classification). Optional imports with graceful fallback if not installed. feature_importances_ accessible — compatible with existing explainer.py. 16 tests; all pass. xgboost/lightgbm added to pyproject.toml.*
      *Day 3 (12:03): Excel/XLSX upload (.xlsx/.xls) via openpyxl — converts to CSV on ingest; all downstream endpoints work unchanged. MLPRegressor/MLPClassifier added to algorithm registry with size-aware recommendation messages. 21 new tests; 530 total. Frontend dropzone updated to accept xlsx/xls.*
      *Day 3 (16:03): Google Sheets + CSV URL import — POST /api/data/upload-url converts Sheets share links to CSV export URL (with gid/tab preservation), downloads, profiles, and creates Dataset. urllib.request, no external deps. Frontend "Import from Google Sheets or CSV URL" toggle in UploadPanel. 15 backend tests; api.ts uploadFromUrl() covered in frontend tests.*
      *Day 3 (10:00): SQLite database connector — POST /api/data/upload-db (upload .db/.sqlite/.sqlite3, list tables) + POST /api/data/extract-db (SELECT query or full table → Dataset CSV). stdlib sqlite3 + pandas.read_sql_query, zero new deps. 14 integration tests; api.ts uploadDb()/extractDb() client methods.*
- [x] **Smarter chat orchestration** — Evolve the conversation AI: richer prompt
      templates (prompts.py), narrative explanations (narration.py), proactive insights
      ("I noticed your R² dropped when I removed feature X — want to add it back?"),
      and multi-turn reasoning about model selection trade-offs.
      *Day 2 (22:00): _call_claude() helper with API-key guard + fallback; narrate_data_insights_ai() calls Claude after upload with build_proactive_insight_prompt; narrate_training_with_ai() calls Claude with build_model_comparison_narrative_prompt for 2+ models; _detect_model_regression() compares latest vs best previous run (>2% threshold) and injects "I noticed your R² dropped..." into system prompt; build_system_prompt gains recent_messages param for multi-turn continuity (last 4 messages, 300-char cap); api/data.py and api/models.py wired. 20 new tests; 464 total, all pass.*
- [x] **Advanced visualizations** — Heat maps for correlation matrices, interactive
      scatter plots with brushing/linking, time-series decomposition charts, model
      comparison radar charts. Each viz should be triggered naturally through chat.
      *Day 2 (20:05): correlation heatmap added — build_correlation_heatmap(), /api/data/{id}/correlations endpoint, frontend HeatmapChart CSS-grid renderer with color scale.*
      *Day 2 (14:00): radar chart for model comparison — build_model_comparison_radar() normalizes all metrics to 0-1, /api/models/{id}/comparison-radar endpoint returns 204 when <2 models.*
      *Day 3 (00:09): time-series decomposition — detect_time_columns(), build_timeseries_chart() (original + rolling avg + OLS trend), GET /api/data/{id}/timeseries endpoint. 21 tests.*
      *Day 3 (08:04): scatter brushing — click-to-highlight in InteractiveScatterChart; selected point shown with reference lines + coordinates label + Clear button; normal points dim to 35% opacity when one is selected.*
- [x] **Data transformation pipeline** — Support for multi-step, reorderable
      transformation pipelines with undo. Let users build complex feature engineering
      flows through conversation.
      *Day 3 (08:04): 3 new endpoints on FeatureSet — GET /steps (list), POST /steps (append one step), DELETE /steps/{index} (undo). Each mutation re-applies the full pipeline and returns updated preview + new_columns. Frontend PipelinePanel component shows ordered steps with per-step Undo buttons; loaded on mount from GET /steps. api.ts getSteps/addStep/removeStep client methods. 14 tests, all pass.*
- [x] **Multi-dataset support** — Allow joining/merging multiple CSVs within a project.
      The chat guides the user through selecting join keys and resolving conflicts.
      *Day 3 (02:00): suggest_join_keys() ranks common columns by uniqueness ratio; merge_datasets() handles inner/left/right/outer joins with automatic suffix for conflicting column names. 3 new endpoints (list, join-keys, merge); DatasetListPanel in Data tab with guided merge UI (select 2 datasets → auto-suggest join key → pick join type → preview result). 31 tests, all pass.*
- [x] **Template projects** — Pre-built project templates for common use cases (sales
      forecasting, customer churn, demand prediction) with sample data, pre-configured
      features, and guided conversation flows.
      *Day 3 (04:31): 3 templates (sales_forecast/customer_churn/demand_forecast) with GET /api/templates, GET /api/templates/{id}, POST /api/templates/{id}/apply. Each template ships with sample CSV (200/300/250 rows), pre-configured target column + problem type, suggested algorithms, and a conversation starter message. 20 tests, all pass.*

#### Track C — Coordination

- [x] **Update BACKLOG.md** — Before starting work, check BACKLOG.md for what the other
      bot instance is working on or has recently explored. Write your chosen focus at the
      top before implementing. After the session, move completed items to the "Done" section
      and add any new ideas you discovered.
      *Day 3 (08:04): BACKLOG updated at session start and end each session from Day 2 onward.*

---

## Data Model

### Core Entities

```
Project
├── id: UUID
├── name: str
├── description: str (optional)
├── created_at: datetime
├── updated_at: datetime
├── status: enum (exploring, modeling, deployed)
└── settings: JSON (preferences, defaults)

Dataset
├── id: UUID
├── project_id: FK → Project
├── filename: str
├── file_path: str (local filesystem)
├── row_count: int
├── column_count: int
├── columns: JSON (name, dtype, stats)
├── profile: JSON (cached profiling results)
├── uploaded_at: datetime
└── size_bytes: int

FeatureSet
├── id: UUID
├── dataset_id: FK → Dataset
├── transformations: JSON (ordered list of applied transforms)
├── column_mapping: JSON (original → engineered features)
├── target_column: str
├── created_at: datetime
└── is_active: bool

ModelRun
├── id: UUID
├── project_id: FK → Project
├── feature_set_id: FK → FeatureSet
├── algorithm: str
├── hyperparameters: JSON
├── metrics: JSON (accuracy, precision, recall, R², etc.)
├── training_duration_ms: int
├── model_path: str (serialized model file)
├── is_selected: bool
├── is_deployed: bool
├── created_at: datetime
└── shap_values_path: str (optional, cached SHAP)

Conversation
├── id: UUID
├── project_id: FK → Project
├── messages: JSON (list of {role, content, timestamp, metadata})
├── state: enum (upload, explore, shape, model, validate, deploy)
└── updated_at: datetime

Deployment
├── id: UUID
├── model_run_id: FK → ModelRun
├── endpoint_path: str (/api/predict/{model_id})
├── dashboard_url: str
├── is_active: bool
├── request_count: int
├── created_at: datetime
└── last_predicted_at: datetime (optional)
```

---

## API Design

### Chat & Conversation
- `POST /api/chat/{project_id}` — Send message, get streamed response (SSE)
- `GET /api/chat/{project_id}/history` — Get conversation history

### Data Management
- `POST /api/data/upload` — Upload CSV, create dataset, return preview
- `GET /api/data/{dataset_id}/preview` — First N rows + column stats
- `GET /api/data/{dataset_id}/profile` — Full data profile (cached)
- `GET /api/data/{dataset_id}/query` — Natural language → query → result

### Feature Engineering
- `GET /api/features/{dataset_id}/suggestions` — AI-generated feature suggestions
- `POST /api/features/{dataset_id}/apply` — Apply selected transformations
- `GET /api/features/{feature_set_id}/preview` — Preview transformed data

### Model Training
- `POST /api/models/{project_id}/train` — Start training run (background job)
- `GET /api/models/{project_id}/status` — Training progress
- `GET /api/models/{project_id}/compare` — Compare trained models
- `POST /api/models/{model_run_id}/select` — Select model for deployment

### Validation
- `GET /api/validate/{model_run_id}/metrics` — Detailed validation metrics
- `GET /api/validate/{model_run_id}/explain` — SHAP/feature importance
- `GET /api/validate/{model_run_id}/explain/{row_index}` — Single prediction explanation

### Deployment
- `POST /api/deploy/{model_run_id}` — Deploy model (create endpoint + dashboard)
- `POST /api/predict/{deployment_id}` — Make prediction (public endpoint)
- `POST /api/predict/{deployment_id}/batch` — Batch prediction (CSV in, CSV out)
- `GET /api/deployments` — List active deployments
- `DELETE /api/deploy/{deployment_id}` — Undeploy model

### Project Management
- `POST /api/projects` — Create project
- `GET /api/projects` — List projects
- `GET /api/projects/{id}` — Get project details
- `DELETE /api/projects/{id}` — Delete project

---

## Testing Strategy

### Backend (pytest + pytest-bdd)
- **Unit tests:** Core logic (analyzer, feature_engine, trainer, validator)
- **Integration tests:** API endpoints with real SQLite database, real file uploads
- **BDD scenarios:** End-to-end user stories ("Given I upload a sales CSV, When I
  ask for a revenue prediction model, Then I get a trained model with metrics")
- **No mocking:** Real services, real files, real ML training (use small datasets)

### Frontend (Jest + Playwright)
- **Unit tests:** Component rendering, store logic, API client
- **E2E tests:** Full user flows — upload → explore → train → deploy
- **Visual regression:** Chart rendering, responsive layout

### Quality Gates
- Coverage: >85%
- Pass rate: 100%
- All E2E scenarios pass before merge

---

## UX Principles (for the AI agent building this)

1. **Chat is king.** The chat panel should feel like the natural way to do everything.
   Side panels are for displaying results, not for input forms.

2. **No jargon without explanation.** If you use a technical term, immediately follow
   it with a plain-English equivalent. "R² (how well the model fits your data, from
   0 to 1 — higher is better)"

3. **Show, don't tell.** Every insight should come with a visualization. Don't just
   say "there's a correlation" — show the scatter plot.

4. **Celebrate progress.** When a model trains successfully, when accuracy is high,
   when deployment completes — acknowledge it. Not with confetti, but with warm,
   confident language.

5. **Fail gracefully.** Bad data? Say what's wrong and suggest how to fix it. Model
   performs poorly? Explain why and suggest next steps. Never show a stack trace.

6. **Speed matters.** Show loading states, stream responses, cache aggressively.
   The user should never wonder "is it doing something?"
