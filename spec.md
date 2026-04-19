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

### Phase 8: UI/UX, Accessibility & CX Hardening

> Audit-driven improvements from a full UI/UX, a11y, and CX review. Target audience is
> business analysts (non-technical users). Items are ordered high → low impact within
> each track and should be picked up by evolve sessions before moving to Phase 9.

#### Track A — Accessibility (WCAG 2.1 AA)

- [x] **Skip navigation link** — Add a visually-hidden skip link as the first focusable element in `app/layout.tsx` (`<a href="#main-content" className="sr-only focus:not-sr-only">Skip to main content</a>`) and `id="main-content"` on the `<main>` element so keyboard users can bypass the nav bar on every page.

- [x] **Tab panel ARIA pattern** — The right-panel tab bar (`app/project/[id]/page.tsx`) and validation sub-tabs (`components/validation/validation-panel.tsx`) use bare `<button>` elements with no `role="tab"`, `aria-selected`, or enclosing `role="tablist"`. Tab panels have no `role="tabpanel"` or `aria-labelledby`. Apply the full ARIA tab widget pattern to both navigation levels.

- [x] **Feature suggestion rows keyboard accessible** — In `components/features/feature-suggestions.tsx`, suggestion rows use `<div onClick>` with a decorative checkbox `<div>` inside, giving keyboard users no access and no `aria-checked`. Replace the outer div with a `<button>` or `<input type="checkbox">` with a `<label>`, and expose selection state via `aria-checked` or `checked`.

- [x] **Emoji and Unicode status icons annotated** — Components including `training-started-card.tsx`, `deployed-card.tsx`, `feature-suggestions-chat-card.tsx`, `data-story-card.tsx`, and `readiness-check-card.tsx` use raw emoji (✅, ⚙️, 📄, ✓, ✗, ⚠) in `<span>` elements with no `aria-label`. Mark decorative emoji `aria-hidden="true"` when adjacent text conveys the meaning; add `role="img" aria-label="..."` where the emoji carries unique meaning.

- [x] **Expand/collapse buttons expose state** — "Show more / Show less" toggles in `anomaly-card.tsx` and `dictionary-card.tsx` have no `aria-expanded`. Add `aria-expanded={showAll}` and `aria-controls` pointing to the list's `id` on every progressive-disclosure toggle.

- [x] **Algorithm card selection state** — `AlgorithmCard` in `components/models/model-training-panel.tsx` renders as `<button>` but has no `aria-pressed`. Add `aria-pressed={selected}` so screen readers announce whether the algorithm is currently selected.

- [x] **Heatmap cell keyboard and focus** — In `components/chat/chart-message.tsx`, heatmap cells (`role="button"`, `tabIndex={0}`) handle only `Enter`, not `Space`. They also apply `outline: "none"` removing the focus ring entirely. Add a Space key handler and replace the inline outline removal with `focus-visible:ring-2` class.

- [x] **Chart SVG accessibility** — Recharts charts in `model-training-panel.tsx`, `validation-panel.tsx`, and `chart-message.tsx` produce unlabeled SVGs. Wrap each chart in a `<figure>` with `<figcaption>` describing the data, or pass `title`/`desc` via Recharts props, so screen readers announce meaningful context instead of raw SVG path data.

- [x] **Deployment analytics sparkbar accessible** — In `deployment-panel.tsx`, the `AnalyticsMiniChart` sparkbar is purely visual. Add `aria-label="Predictions over last 7 days: [values]"` to the container element.

#### Track B — Ease of Use / CX for Business Analysts

- [x] **Undeploy confirmation dialog** — In `deployment-panel.tsx`, the "Undeploy" button fires `handleUndeploy` immediately with no confirmation. Undeploying breaks all live users of the prediction API. Add an inline confirmation pattern (replace the button temporarily with "Are you sure? This will break the live prediction link." + Confirm/Cancel) before executing the action.

- [x] **Plain-English metric explanations in training panel** — `MetricsRow` in `model-training-panel.tsx` shows R², MAE, RMSE, F1, Precision as bare numbers. Add a tooltip or inline explanation for each metric (e.g., "R² 0.84 — your model explains 84% of variation in the data. Higher is better.") to match the plain-English style used in `ModelCardView`.

- [x] **"Train more" confirmation before clearing results** — Clicking "Train more" in `model-training-panel.tsx` silently clears `runs` and `comparison` from the UI with no warning. Add a confirmation before clearing, or redesign to keep existing results visible alongside the new training configuration form.

- [x] **Chat input multi-line support** — The chat input in `app/project/[id]/page.tsx` is a single-line `<Input>`. Shift+Enter is caught but does nothing. Replace with a `<Textarea>` that auto-grows (`field-sizing-content` or resize observer) using Shift+Enter for newlines and Enter to send.

- [x] **Copy chat message to clipboard** — Assistant message bubbles have no copy action. Add a copy-to-clipboard button (visible on hover or via a `...` menu) so business analysts can copy model summaries, chart insights, and data stories to share with colleagues.

- [x] **"Defaults" defined in What-If analysis** — The What-If card in `deployment-panel.tsx` says "+N more features use defaults" without defining defaults. Add a footnote: "Remaining features use the median value from the training dataset."

- [x] **Project loading skeleton** — The project workspace shows only the text "Loading project..." during the initial fetch. Replace with a skeleton layout or spinner so users know data is loading, not broken.

- [x] **Validation empty state navigates to Models tab** — The empty state in `validation-panel.tsx` says "Select a model in the Models tab first" but provides no navigation action. Add a button that calls the parent's tab-switch callback to take users directly to the Models tab.

- [x] **`handleExplain` silent failure feedback** — In `app/predict/[id]/page.tsx`, errors in `handleExplain` are swallowed silently — the loading spinner stops with no feedback. Show an inline message: "Explanation unavailable for this prediction."

- [x] **Suggestion chips labeled and visually distinct** — Suggestion chips in `app/project/[id]/page.tsx` have no header label and are styled similarly to message bubbles. Add a "Try asking:" label above the chip row and use a small caret icon on each chip to signal they are clickable prompts, not system messages.

- [x] **No-dataset right panel call-to-action** — When `!currentDataset`, the right panel is completely blank. Render a prominent upload card with step-by-step instructions and a visible drag-and-drop zone so first-time users are not confused by an empty panel.

#### Track C — Consistent UX Patterns

- [x] **Standardize data card colors to design tokens** — `forecast-chart.tsx`, `readiness-check-card.tsx`, `group-stats-card.tsx`, and `correlation-bar-card.tsx` use hardcoded Tailwind gray/blue colors (`text-gray-800`, `bg-gray-100`, `stroke="#2563eb"`) instead of semantic tokens (`text-foreground`, `bg-muted`, `text-primary`). These break in dark mode. Audit and replace all hardcoded colors in these files with CSS variable-based tokens.

- [x] **Unify expand/collapse toggle pattern** — "Show more / Show less" toggles across the codebase use three different implementations (colors, underline behavior, sizing differ). Standardize on `<Button variant="ghost" size="sm">` everywhere.

- [x] **Standardize Badge usage** — Inline `<span>` elements styled to look like badges exist alongside the design-system `<Badge>` component throughout the codebase. They differ in border-radius, padding, and font-weight. Replace all ad-hoc badge spans with the `<Badge>` component using `className` for color-only overrides.

- [x] **Unify feature importance bar scaling** — `ImportanceBar` in `model-card-view.tsx` uses a `× 5` magic-number scale; `FeatureImportancePanel` in `feature-suggestions.tsx` uses percentage-of-max. The same feature can appear at different widths in different views. Extract a single `<ImportanceBar importance={0..1} />` component used everywhere.

- [x] **Page heading hierarchy** — The project workspace has no `<h1>` (the project name is in a `<span>` in the breadcrumb). The prediction page has `<h1>`. `CardTitle` renders as a `<div>` in many components. Establish a hierarchy: `<h1>` per page, `<h2>` for major sections, `<h3>` for card titles via `asChild` or `as` props.

#### Track D — Data Visualization Polish

- [x] **Radar chart metric labels** — `ModelRadarChart` in `model-training-panel.tsx` shows raw identifiers (`r2`, `mae`) as axis labels. Replace with plain-English labels ("Accuracy", "Avg Error") and correct the note to say "All metrics scaled so a larger area = better performance" (the current "higher is better on every axis" is inaccurate for un-inverted MAE).

- [x] **Y-axis label fallback** — In `chart-message.tsx`, bar and line charts only show the Y-axis label when `y_label` is truthy. Default to `y_keys[0]` when `y_label` is absent so axes are never unlabeled.

- [x] **Residuals chart labels and guidance order** — In `validation-panel.tsx`, the residuals scatter has no Y-axis label and the interpretive text appears below the chart. Move guidance above the chart and add `"Residual (actual − predicted)"` as the Y-axis label.

- [x] **Color-only encoding in correlation and group-stats** — `correlation-bar-card.tsx` uses blue/red to encode positive/negative (fails for colorblind users). `group-stats-card.tsx` uses 4 blue shades to encode rank (also color-only). Add directional arrows (`↑`/`↓`) to correlation bars and numeric rank labels (1, 2, 3…) to group-stats rows.

- [x] **Forecast chart tick formatter** — In `forecast-chart.tsx`, `tickFormatter` slices date strings to 8 characters, producing "2024-01-" for full datetimes. Replace with a period-aware formatter producing human-friendly labels ("Jan 2024", "Q1 2024") based on the `period_label` value.

- [x] **Version history chart domain** — In `model-training-panel.tsx`, `VersionHistoryCard` uses `domain={[-0.1, 1]}` for regression, clipping models with R² < -0.1. Use `domain={["auto", "auto"]}` or compute domain from data to accommodate all values.

#### Track E — Workflow Guidance

- [x] **Workflow stepper includes Feature Engineering** — `WorkflowProgress` covers Upload → Train → Validate → Deploy but skips Feature Engineering. Users following the stepper will be directed to Train before setting a target column or applying transformations. Add a "Features" step between Upload and Train, or update the Upload step description: "Explore your data and configure features before training."

- [x] **Workflow stepper visible on mobile** — The `WorkflowProgress` is inside the right panel, which is hidden when `mobileView === "chat"`. Mobile users have no workflow indicator while chatting. Move the stepper to the top bar (below the breadcrumb) so it remains visible regardless of which panel is active.

- [x] **Validate step tied to actual validation** — In `workflow-progress.tsx`, the Validate step is marked `done` as soon as a deployment exists (`hasDeployment`), not when the user actually runs validation. Track validation completion separately (e.g., cross-validation results present) and use that as the Validate step signal to prevent the false "all done" state for users who skipped validation.

- [x] **Training-started card navigates to Models tab** — In `training-started-card.tsx`, the prompt "Check the Models tab for real-time progress →" uses a Unicode arrow with no navigation action. Make "Models tab" a `<button>` that fires the tab-switch callback, so a single click takes the user to their running training job.

### Phase 9: Continuous Evolution (Perpetual)
> Goal: Move beyond the initial spec. Research, ideate, and implement — guided by the
> vision, not a fixed checklist. Balance quality hardening with scope expansion.
>
> These items are **never checked off**. Each session, pick work from one or more of
> these tracks based on what will have the most impact right now.
>
> **PRIORITY ORDER (as of Day 19):**
> 1. **Track D — Deployment Depth** ← highest priority; this is AutoModeler's biggest
>    competitive differentiator and the most underbuilt area relative to the vision.
> 2. **Track C — Model Building Depth** ← second priority; richer training = better models.
> 3. **Track E — End-to-End Polish** ← third; close friction in the "lunch break" flow.
> 4. **Track B — Vision-Driven Innovation** ← only if D/C/E have nothing obvious to do.
> 5. **Track A — Quality Hardening** ← test coverage has reached its target (see below).
>    **DO NOT add new tests purely for coverage.** Backend is at 99%, frontend at 91%.
>    Both exceed the 85% target. Write tests only for new features. Stop chasing 100%.
>    The Explore phase (analytics cards) is **saturated** — do not add more chat-triggered
>    analysis card types. Focus on what happens *after* a model is built.

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
- [x] **Unit test coverage to 100%** — Identify uncovered backend modules and frontend
      components. Write targeted tests for edge cases, error paths, and boundary
      conditions. Use `pytest --cov` and Jest coverage reports to find gaps.
      *Day 2 (20:05): query_engine.py 14%→92%, total backend 92%→95%. Remaining: frontend Jest coverage; explainer.py and validator.py edge paths.*
      *Day 3 (00:09): chart_builder 73%→100%, orchestrator 78%→100%, api/chat 37%→98%, total backend 94%→97%. 400 tests pass. Remaining: frontend Jest coverage.*
      *Day 3 (18:00): frontend Jest set up (next/jest + @testing-library/react + jest-fetch-mock); 69 unit tests covering store mutations, API client HTTP shapes, ChartMessage rendering (all 6 types), cn() utility. Frontend + backend = 469 total tests.*
      *Day 3 (06:00): 4 new frontend test suites (deployment-panel 17, model-training-panel 15, validation-panel 25, feature-suggestions-panel 25); api.ts 100% coverage. Total: 150 frontend + 530 backend = 680 tests. api.ts 100%, deployment-panel 99%, validation-panel 89%, feature-suggestions 38% (sub-components not yet tested).*
      *Day 3 (16:03): PipelinePanel (10 tests), DatasetListPanel (20 tests), FeatureImportancePanel (8 tests) — 38 new frontend tests for all feature-suggestions.tsx sub-components. Plus 2 new api.test.ts tests for uploadFromUrl. Total: 190 frontend + 545 backend = 735 tests, all passing.*
      *Day 3 (10:00): 74 new backend tests targeting api/features, api/validation, api/chat, api/deploy error paths; 15 frontend tests for app/page.tsx + predict/[id]/page.tsx (first app/ page coverage). SQLite connector adds 14 more backend tests. Total: 205 frontend + 630 backend = 835 tests. Backend coverage: 98%.*
      *Day 3 (20:02): 53 new targeted backend tests in test_final_coverage.py covering 20+ modules — multiclass explainer, validator bias/confidence, deployer classification, feature_engine edge cases, query_engine null returns, narration error paths, analyzer inf/NaN, report_generator, API 404/silencing paths. 686 backend tests, 99% coverage (9196 stmts, 73 missing). Remaining uncovered: ImportError branches (xgboost/lgbm when installed) + SSE streaming paths — both architecturally impossible without library removal or live connections.*
      *Day 3 (14:00): 49 new workspace page tests (app/project/[id]/page.tsx 0%→91%); also fixed scrollIntoView jsdom stub; excluded types.ts + layout.tsx from coverage (pure declarations, no runtime code). Frontend coverage 63%→91% statements. 254 frontend + 686 backend = 940 total tests. Both stacks exceed 85% spec target.*
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

- [x] **Prediction logging & monitoring** — Track every prediction made via PredictionLog
      SQLModel table (inputs, output, timestamp, confidence). Analytics endpoint returns
      per-day counts, histogram distribution, recent average. Logs endpoint returns paginated
      prediction history. DeploymentPanel upgraded with AnalyticsCard (mini bar chart + totals)
      and ReadinessCard (score + checklist). Chat detects readiness intent and emits structured
      readiness events alongside Claude response.
      *Day 4 (00:08): PredictionLog model; GET /api/deploy/{id}/analytics + /logs; GET
      /api/models/{id}/readiness (6-check scorecard); chat intent detection; frontend panel
      upgrades. 34 backend + 12 frontend = 46 new tests. Total: 720 backend + 266 frontend = 986.*
- [x] **Hyperparameter auto-tuning** — POST /api/models/{run_id}/tune uses RandomizedSearchCV to find
      better settings for the selected model. Returns before/after metrics, improvement %, and best params.
      TuningCard in ModelTrainingPanel shows the result inline. Non-tunable algorithms (linear regression)
      return a graceful explanation. get_tuning_grid() exposes per-algorithm param grids.
      *Day 4 (04:44): 25 backend + 13 frontend = 38 new tests. Total: ~770 backend + 282 frontend = ~1052.*
- [x] **AI project narrative generator** — POST /api/projects/{id}/narrative synthesises all project
      artifacts (dataset, features, model metrics, deployment status) into a plain-English executive
      summary. Uses Claude when ANTHROPIC_API_KEY is present; falls back to structured static narrative.
      Perfect for "share with VP" use case. api.projects.narrative() in frontend API client.
      *Day 4 (04:44): 21 backend tests. Context dict includes dataset, features, model, deployment.*
- [x] **Prediction feedback loop** — Users record actual outcomes after predictions:
      POST /api/predict/{id}/feedback stores FeedbackRecord (actual_value/actual_label,
      is_correct auto-computed by comparing stored prediction to provided label).
      GET /api/deploy/{id}/feedback-accuracy aggregates real-world MAE (regression) or
      accuracy (classification) across all recorded feedback. FeedbackCard in
      DeploymentPanel shows stats and feedback form. api.ts submitFeedback() + feedbackAccuracy().
      *Day 4 (08:06): 21 backend tests, all passing. Closes the loop between predictions and reality.*
- [x] **Smart model health dashboard + guided retraining** — GET /api/deploy/{id}/health computes a
      unified 0-100 health score combining model age (freshness), feedback accuracy (real-world
      performance), and drift (distribution stability). POST /api/models/{project_id}/retrain
      one-click retrains using existing feature set and selected algorithm. Chat detects "model health",
      "should I retrain", "update model" etc. → injects health context into system prompt + emits
      {type: health} SSE event. ModelHealthCard in DeploymentPanel shows score, status badge,
      per-component breakdown, recommendations, and Retrain button.
      *Day 4 (02:00): 27 backend + 12 frontend = 39 new tests. Total: 854 backend + 294 frontend = 1148.*
- [x] **Live prediction explanation on public dashboard** — The predict/[id] shareable dashboard
      now explains WHY the model made each prediction. POST /api/predict/{deployment_id}/explain
      returns feature contributions (importance × normalised deviation from training mean) sorted
      by absolute impact, plus a plain-English summary and top_drivers list. PredictionPipeline
      stores feature_means + feature_stds at build time for accurate attribution.
      Frontend: "Why this prediction?" button appears after each result → ContributionBar waterfall
      chart with red (pushed down) / blue (pushed up) bars showing each feature's impact vs
      the training average. FeatureContribution + PredictionExplanation types added.
      Closes the vision's "Not a black box" promise for the shareable analyst dashboard.
      *Day 4 (12:04): 11 backend + 6 frontend = 17 new tests. Total: ~870 backend + 306 frontend.*
- [x] **Prediction session history on public dashboard** — The predict/[id] page tracks all
      predictions made in the current browser session. After the first prediction, a "Session
      History" section appears showing a table of all past predictions (sequence #, time, result)
      with a "Download CSV" button that exports the full session including all feature inputs.
      PredictionHistoryRecord type added. History capped at 20 entries per session.
      *Day 4 (06:00): pure frontend implementation. 4 new page tests. Closes the analyst use
      case: "take predictions back to my VP" without needing the full deployment panel.*
- [x] **Model version history timeline** — GET /api/models/{project_id}/history returns all
      training runs sorted oldest-first with primary metric (r2/accuracy), trend direction
      (improving/declining/stable/insufficient_data, computed via linear regression slope with
      2%-of-mean stability floor), best/latest metric. _compute_trend helper is independently
      testable. Frontend VersionHistoryCard in ModelTrainingPanel: mini Recharts LineChart of
      primary metric over time, summary stats row (Best/Latest/Runs), per-run table with
      Current/Live badges. Card only appears when 2+ completed runs exist. History loaded on
      mount and refreshed after SSE all_done event. Closes the retrain-feedback loop: analysts
      can see "is my model actually improving across sessions?"
      *Day 4 (16:04): 19 backend + 18 frontend = 37 new tests. Total: 911 backend + 343 frontend = 1254.*

- [x] **Bulk scenario comparison** — POST /api/predict/{id}/scenarios accepts a base feature dict and up to 10 labelled
      override sets. Returns the base prediction plus per-scenario result (delta, percent_change, direction, probabilities),
      best/worst outcome identification, and a plain-English summary. Perfect for "what if revenue if region = X vs Y vs Z?"
      VP meeting prep. api.deploy.scenarios() in frontend API client; ScenarioComparison + ScenarioResult types added.
      *Day 4 (20:03): 12 backend + 10 frontend = 22 new tests. Total: 951 backend + 348 frontend = 1299.*
- [x] **Anomaly detection** — POST /api/data/{dataset_id}/anomalies runs IsolationForest across selected numeric features to find
      multi-dimensional outliers (e.g., row where revenue, quantity, AND category together look suspicious — not just one column at a time).
      Returns per-row anomaly scores 0-100, top-N anomalous records with feature values, plain-English summary. Chat detects
      "find anomalies", "unusual records", "outliers", "suspicious" → injects summary into system prompt + emits {type: "anomalies"}
      SSE event. AnomalyCard in Data tab shows summary, features analysed, top rows table with score badges, and manual re-scan button.
      *Day 4 (14:00): 22 backend + 11 frontend = 33 new tests. Total: 978 backend + 359 frontend = 1337.*
- [x] **Chat follow-up suggestion chips** — After each AI response, the backend emits a {type: "suggestions"} SSE event
      with 2-3 context-aware follow-up questions chosen from a per-state pool (6 states × 4-6 suggestions each) plus dynamic
      additions based on available project artefacts (best algorithm name, accuracy metric, deployment request count).
      Frontend shows clickable pill-shaped chips below the chat input; clicking prefills the input without sending.
      generate_suggestions() in orchestrator.py independently testable. Directly implements the "smart colleague"
      vision principle for non-technical users who don't know what to ask.
      *Day 4 (20:03): included in 22 tests above.*
- [x] **Conversational data cleaning** — Users can say "fill missing revenue with median", "remove duplicate rows",
      "drop rows where quantity < 0", "cap outliers in sales at 99%", or "drop column X" and the dataset is
      updated in-place (CSV rewritten, profile recomputed, Dataset record updated). core/cleaner.py provides five
      pure-function operations: remove_duplicates, fill_missing (mean/median/mode/zero/value), filter_rows
      (gt/lt/eq/ne/gte/lte/contains/notcontains), cap_outliers (percentile clip), drop_column. Chat detects
      cleaning intent via _CLEAN_PATTERNS regex + _detect_clean_op() param extractor → emits {type: cleaning_suggestion}
      SSE event (suggested_operation + quality_summary). Upholds "explain before executing" vision principle:
      operation is SUGGESTED not auto-applied; CleaningCard in Data tab shows quality summary, suggested op
      description, and a one-click Apply button. api.ts clean() + CleaningSuggestion/CleanResult types.
      *Day 4 (20:00): 39 backend + 12 frontend = 51 new tests. Total: 1017 backend + 371 frontend = 1388.*
- [x] **Model monitoring alerts + chat-triggered visualizations** — Proactive system-wide health alerts:
      GET /api/projects/{id}/alerts scans all active deployments for four alert types: stale_model
      (>60 days=warning, >90=critical), no_predictions (deployed >1 day with 0 requests), drift_detected
      (from PredictionLog when ≥40 logs), poor_feedback (classification accuracy <70% or regression
      pct_error >30% from FeedbackRecord). Returns sorted list (critical-first). AlertsCard in
      DeploymentPanel: "Check for Alerts" button, severity badges, recommendation text, "Show N more"
      collapse, externalAlerts prop for chat SSE push. Chat detects "any alerts?", "monitor", "check my
      models" → injects summary + emits {type: alerts} SSE event. Chat also detects "show model history"
      → {type: history} and "how many predictions" → {type: analytics} SSE events for triggering
      existing panels from conversational queries. Three compiled regex pattern groups added to chat.py.
      *Day 4 (10:00): 23 backend + 13 frontend = 36 new tests. Total: 934 backend + 338 frontend = 1272.*

- [x] **Prediction drift detection + what-if analysis** — Two new Phase 8 capabilities:
      (1) GET /api/deploy/{id}/drift compares early vs recent prediction distributions using
      only PredictionLog — no schema migration needed. Regression: z-score of mean shift;
      classification: total variation distance. Returns drift_score 0–100, status, explanation.
      Chat detects drift keywords and emits {type: drift} SSE events. DriftCard in DeploymentPanel
      shows baseline/recent stats. (2) POST /api/predict/{id}/whatif accepts base features +
      overrides, runs predict_single twice, returns delta/percent_change/direction + plain-English
      summary. WhatIfCard in DeploymentPanel with feature input form. Fixed 4 pre-existing
      test_prediction_monitoring failures (Anthropic mock missing).
      *Day 3 (18:00): 18 backend + 3 frontend = 21 new tests. Total: 738 backend + 269 frontend = 1007 tests, all passing.*

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
      *Day 4 (06:00): box plot chart type — build_boxplot() with Tukey-fence whiskers, grouped or single-column; GET /api/data/{id}/boxplot endpoint; BoxPlotChart SVG renderer in chat-message.tsx. 19 backend + 8 frontend = 27 new tests.*
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

- [x] **Workflow progress stepper** — Visual 4-step indicator (Upload → Train → Validate → Deploy)
      showing completion status derived from existing React state. Rendered above the right-panel
      tab bar once data is uploaded; each step is clickable to jump to the relevant tab; active
      step highlighted in primary color, completed steps show checkmark. `hasDeployment` state
      tracks deployment dynamically (seeded from project.has_deployment, updated via onDeployed).
      Tab buttons given `data-testid="tab-{name}"` for reliable test targeting.
      *Day 5 (04:00): 10 new frontend tests. Total: 381 frontend + 1017 backend = 1398.*
      Also: auto-fixed 149 ruff lint errors (F401/F841/E401/F541) across backend test files and
      API modules; fixed jest.config.js ESLint error.

- [x] **Prediction confidence intervals** — Regression predictions now return a 95% prediction
      interval (lower, upper) computed from residual std on training data. Classification
      predictions return a `confidence` field = max(predict_proba). `PredictionPipeline` stores
      `residual_std`; at deploy time `api/deploy.py` loads the model and computes std(y_true - y_pred)
      on training data, storing it in the pipeline file. `predict_single()` returns `confidence_interval`
      dict when residual_std > 0. Frontend: `ConfidenceIntervalBadge` on predict/[id] page shows
      "95% prediction interval: X – Y" below the main prediction value; classification shows a green
      "Model confidence: N%" badge. `ConfidenceInterval` type added to types.ts. Directly implements
      the "not a black box" vision promise for the shareable analyst dashboard.
      *Day 9 (00:05): 14 backend + 6 frontend = 20 new tests. Total: 1053 backend + 401 frontend = 1454.*

- [x] **Model Improvement Advisor** — When an analyst asks "how do I improve my model?" or "make my
      predictions better", AutoModeler analyses the current trained model's metrics, feature set, and
      training choices, then returns ranked, plain-English improvement suggestions ordered by expected
      impact. `core/advisor.py` provides `compute_improvement_suggestions()` — a pure function that runs
      9 independent checks: weak features (call feature selection), ensemble potential (R² < 0.80),
      date features unused, small training dataset, class imbalance unhandled, calibration missing,
      hyperparameter tuning available, too few features, linear model on nonlinear data. Each suggestion
      carries `difficulty` (easy/medium/hard) and `expected_impact` (low/moderate/high), sorted
      high-impact-first then by ease of action. `GET /api/models/{project_id}/improvement-suggestions`
      endpoint loads the selected (or best) run, derives context (dataset size, date col, weak features,
      calibration flags), and returns the ranked list. `_IMPROVEMENT_PATTERNS` (14 NL variants: "how do
      I improve my model", "make my predictions better", "increase accuracy", "give me suggestions",
      "what's wrong with my model") in `chat.py` — distinct from `_TUNE_PATTERNS` (hyperparameter-only).
      Handler injects top suggestions into the system prompt with metric context so Claude can present
      them naturally. `{type:"model_improvement"}` SSE event. `ModelImprovementCard` (violet border, 💡
      icon) in chat: suggestion count + metric badges in header, per-suggestion rows with category icon
      (🎯/🤖/📊/⚖️), title + impact badge + difficulty badge, explanation text, legend row.
      `ModelImprovementResult`/`ImprovementSuggestion`/`ImprovementDifficulty`/`ImprovementImpact`/
      `ImprovementAction` TypeScript types; `api.models.improvementSuggestions()` client method;
      `attachModelImprovementToLastMessage` Zustand action. Directly implements the vision's "smart
      colleague" promise — a colleague who proactively tells you how to get better results.
      *Day 24 (04:41): 41 backend + 13 frontend = 54 new tests. Total: 2419 backend + 1147 frontend = 3566, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Smart Model Selection Advisor** — When an analyst asks "which model should I use?", "pick the
      most explainable model", "I need the most accurate model", or "fastest model for real-time API",
      AutoModeler scores all completed runs against an analyst-chosen criteria and returns a ranked
      recommendation in plain English. `compute_model_selection(runs, criteria)` pure function in
      `core/advisor.py` supports five criteria: **accuracy** (primary metric wins), **explainability**
      (linear > logistic > decision_tree > RF > XGB > MLP > ensemble rank-inverted to 0-1),
      **stability** (cross-validation coefficient of variation — lower CoV = more stable), **speed**
      (algorithm complexity rank), and **balanced** (0.40×accuracy + 0.30×explainability + 0.30×stability).
      Each run gets per-component scores (0-1 each) plus a composite `score` for sorting. Returns winner
      (with `algorithm_plain`, `why` plain-English justification, component score bars) + all runs ranked
      1-N + one-sentence `summary`. `GET /api/models/{project_id}/model-selection?criteria=` endpoint
      (validates criteria; 400 on unknown). `_MODEL_SELECT_PATTERNS` (15 NL variants: "which model should
      I use", "pick the best model", "most explainable model", "which model is most accurate", "fastest
      model", "low latency model") + `_detect_selection_criteria()` helper in `chat.py` (extracts criteria
      from message: explainability/accuracy/speed/stability keywords, defaults to balanced). System prompt
      injection names winner + ranked list; `{type:"model_selection"}` SSE event.
      `ModelSelectionCard` (indigo border, 🏆 icon) in chat: criteria badge + N-models badge, winner
      highlight box (algorithm name + `why` justification), four component score mini-bars (accuracy/
      explainability/stability/speed), ranked run list (trophy for rank 1, selected/deployed badges),
      summary sentence. `ModelSelectionResult`/`ModelSelectionRun`/`SelectionCriteria` TypeScript types;
      `api.models.modelSelection()` client method; `attachModelSelectionToLastMessage()` Zustand action.
      Closes the "I don't know which model to choose" analyst gap — a smart colleague would say "for
      explainability, use the linear model; for accuracy, use XGBoost."
      *Day 24 (04:00): 42 backend + 18 frontend = 60 new tests. Total: 2461 backend + 1165 frontend = 3626, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Auto-Retrain on Upload** — When enabled, uploading new data automatically triggers a background
      retrain using the project's currently selected model and algorithm — so the model stays current
      without the analyst having to remember to do it. `Project.auto_retrain` bool field (default `False`).
      `GET/PUT /api/projects/{project_id}/auto-retrain` endpoints. `core/retrain.py` provides
      `trigger_auto_retrain(project_id, dataset_id, session)` — finds the selected model run, updates the
      active feature set to point at the new dataset, and fires `_train_in_background()` as a daemon thread.
      `data.py` upload handler calls `trigger_auto_retrain()` after creating the dataset when the flag is on.
      `_AUTO_RETRAIN_PATTERNS` (14 NL variants: "enable auto-retrain", "turn on auto retrain", "keep model
      fresh", "retrain when I upload new data") + chat handler in `chat.py` detects enable/disable intent,
      toggles the DB flag, and emits `{type:"auto_retrain"}` SSE event. `AutoRetrainCard` (teal border, 🔄
      icon) shows Enabled/Disabled badge, selected algorithm label, toggle button, and explains that the
      model will retrain on each upload when enabled.
      *Day 24 (05:30): 14 backend + 10 frontend = 24 new tests.*

- [x] **Conversation Export as HTML Report** — Analysts can say "export this conversation", "download the
      analysis report", or "share this report" and receive a `ConversationExportCard` in the chat with a
      direct download link. `_CONV_EXPORT_PATTERNS` (13 NL variants) in `chat.py` detects the intent and
      emits `{type:"conversation_export"}` SSE event carrying the download URL, message count, and dataset
      name. `GET /api/chat/{project_id}/export` endpoint calls `_build_export_html()` — a pure function that
      assembles a fully self-contained HTML document (no external CSS/JS dependencies) from project metadata,
      dataset info (filename, row/column counts), best model results (algorithm + primary metric + summary),
      and the full conversation transcript (user/assistant messages, HTML-escaped, chronological). The HTML
      is returned as `Content-Disposition: attachment` with a project-name-based filename. `ConversationExportCard`
      (emerald border, 📄 icon) renders in chat: message count badge, dataset name badge, download description,
      and a "Download HTML Report" `<a>` link that triggers the browser download. `ConversationExportInfo`
      TypeScript type; `attachConversationExportToLastMessage` Zustand action. Directly closes the vision's
      "share the analysis journey with your VP" use case — a permanent, offline artifact of the analyst's
      full data exploration, model building, and validation work.
      *Day 24 (12:00): 14 backend + 10 frontend = 24 new tests. Total: 2489 backend + 1195 frontend = 3684, all passing. Backend lint: clean. Frontend build: clean.*

- [x] **Proactive Model Health Alerts** — When an analyst returns to a project that has deployed models
      showing signs of degradation (stale age, low usage), AutoModeler proactively surfaces health alerts
      directly in the chat — no dashboard-hunting required. `compute_deployment_health_item()` pure function
      in `core/analyzer.py` scores each deployment across two signals: **age** (0–100 based on days since
      deploy: 100 if <30 days, 20 if >180 days) and **usage** (0–100 based on request count and idle time).
      Combined score: `age × 0.55 + usage × 0.45`. Status: **healthy** (≥75), **warning** (50–74),
      **critical** (<50). Per-item `top_issue` and `recommendation` are in plain English. `compute_project_health_summary()`
      aggregates all active deployment items for a project: total/healthy/warning/critical counts, alerts list
      (warning + critical items only), overall status (worst-case escalation), and a one-sentence summary.
      `GET /api/projects/{project_id}/health-summary` endpoint returns this for all active deployments in a
      project. `_HEALTH_SUMMARY_PATTERNS` (9 NL variants: "how are my models doing?", "any issues with my
      deployments?", "model health check", "do I need to retrain?", "model drift") triggers a `{type:"health_summary"}`
      SSE event from chat. **Proactive injection**: on project load, if the project has deployments and the
      analyst is on a returning visit, `page.tsx` calls `api.projects.healthSummary()` and injects a
      `health_summary` field into the welcome-back message — so the `ProjectHealthCard` appears automatically
      without the analyst asking. `ProjectHealthCard` (status-adaptive border: emerald/amber/red): overall
      status heading, plain-English summary, count badges (total/healthy/warning/critical), per-alert rows
      with model name, health score progress bar, top issue, recommendation, "View Deployment" and "Retrain
      Model" CTA buttons that switch the right panel tab. Direct implementation of the "smart colleague"
      vision promise: a colleague who taps you on the shoulder and says "Hey, your model is aging — want
      me to retrain it?"
      *Day 24 (20:00): 16 backend + 14 frontend = 30 new tests. Total: 2505 backend + 1209 frontend = 3714, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Prediction Opportunity Discovery** — When an analyst doesn't know what to predict, they can ask
      "what can I predict?", "what should I model?", "suggest a target", or "what prediction opportunities
      are there?" and AutoModeler analyzes the dataset columns to suggest the best prediction targets.
      `compute_prediction_opportunities(col_stats, row_count)` pure function in `core/analyzer.py` scores
      every eligible column: numeric columns become regression candidates; categoricals with 2–20 unique
      values become classification candidates; ID-like columns (name pattern `_id`, `_key`, `pk`, etc.)
      and near-unique categoricals are excluded; columns with >30% missing data are excluded; constant
      columns (zero std/variance) are excluded. Feasibility score (0-100) rewards: low missing data (+20),
      enough predictors (+15), high business value (+10). Business value is classified by column name:
      `revenue`/`sales`/`profit`/`churn` → "high"; `price`/`cost`/`quantity`/`rate` → "medium"; otherwise
      "low". Results ranked by feasibility then business value, capped at 5. `_HIGH_VALUE_NAMES` and
      `_MEDIUM_VALUE_NAMES` regex constants drive the classification. `_example_question()` generates a
      domain-appropriate natural-language question for each opportunity. `GET /api/data/{id}/prediction-
      opportunities` endpoint returns opportunities with `total` count. `_PREDICT_OPP_PATTERNS` (9 NL
      variants: "what can I predict", "what should I model", "suggest a target", "what columns can I
      predict", "prediction opportunities") + system prompt injection (top suggestion + full list) +
      `{type:"prediction_opportunities"}` SSE event. `PredictionOpportunitiesCard` (purple border, 🎯 icon)
      in chat: count + high-value badges, ranked opportunity rows with problem-type badge (violet=regression,
      amber=classification), business-value badge, reason text, example question in quotes, feasibility
      score bar, optional "Set target" button per row via `onSelectTarget` callback. Closes the "business
      analyst who knows their data but doesn't know what to model" cold-start gap from the vision.
      *Day 25 (04:00): 24 backend + 19 frontend = 43 new tests. Total: 2529 backend + 1228 frontend = 3757, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Dataset Distribution Comparison** — When an analyst asks "what changed in my data?", "how does my
      new dataset compare?", "are there distribution shifts?", or "has my data changed?", AutoModeler
      compares the two most recent datasets for the project and surfaces a structured drift report.
      `compute_dataset_comparison(old_df, new_df)` is a pure function in `core/analyzer.py` that
      computes: row count changes (with ± % change), schema changes (new/dropped columns), per-column
      numeric distribution shifts (mean before/after, % change, severity low/medium/high — high if >30%
      mean shift), categorical changes (new categories, dropped categories, frequency shifts — reported
      when top frequency shift ≥10%), an overall drift score (0–100) weighting row changes, schema
      changes, and per-column severity, and a plain-English summary. `GET /api/data/compare?baseline_id=&new_id=`
      endpoint compares two datasets by ID and returns the full report. `_DATASET_COMPARE_PATTERNS` (9
      NL variants: "what changed in my data", "how does my new data compare", "distribution shift",
      "has my data changed", "new vs old data", "differences between datasets", "is my new data different")
      detects intent in chat, queries the two most-recent datasets for the project, injects drift score +
      column highlights into the system prompt, and emits `{type:"dataset_comparison"}` SSE event.
      `DatasetComparisonCard` (orange border, 📊 icon): drift score badge (green/yellow/red), change count
      badge, baseline vs new filenames with row/column counts, row-change %, schema change section
      (new/dropped column lists), numeric drift table (old avg → new avg, % change direction arrow, severity
      badge), categorical drift rows (new categories in green, dropped in red, frequency shift %).
      Closes the "I uploaded new data — what changed?" analyst gap: the smart colleague who says "I noticed
      your revenue average jumped 150% and two new product categories appeared — you should probably retrain."
      *Day 25 (12:00): 23 backend + 18 frontend = 41 new tests. Total: 2552 backend + 1246 frontend = 3798, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Inline Multi-Feature Prediction via Chat** — Analysts can say "run a prediction for Region=East, Units=150",
      "make a prediction with product=Widget, quantity=100", or "calculate what the revenue would be given units=300"
      and receive an `InlinePredictionCard` directly in the conversation — no navigation to the deployment dashboard
      required. `_INLINE_PRED_PATTERNS` (8 NL variants: "run/make/give me a prediction for/with", "what would X be
      if", "score/classify this record", "run the model on", "model output for") detects the intent. The handler is
      guarded to fire only when a deployment exists and `whatif_chat_event` did not already fire (avoiding double-prediction).
      `_extract_multi_feature_prediction(message, feature_names)` parses `key=value`, `key: value`, and `key is value`
      patterns from the message using `_KV_PAIR_RE`, normalising keys case-insensitively and via underscore→space
      mapping against known feature names; numeric strings are cast to float. Features not extracted from the message
      are filled with training-data means (`pipeline.feature_means`). `predict_single()` runs the full prediction.
      The `{type:"inline_prediction"}` SSE event carries: `prediction`, `probabilities` (classification), `confidence_interval`
      (regression), `confidence` (classification), `provided_features` (the values parsed from the message),
      `defaults_used_count`, `total_features`, `summary` (plain English), `target_column`, `problem_type`. Claude's system
      prompt is injected with the result so it can narrate naturally. `InlinePredictionCard` (blue border, 🔮 icon): for
      regression — large prediction value + 95% CI or confidence %; for classification — probability bars per class
      (sorted descending); in both cases: feature badges showing `key=value` for provided inputs, italic "N features
      used training-data averages" note when defaults were applied. `InlinePredictionResult` TypeScript type;
      `attachInlinePredictionToLastMessage` Zustand action; SSE wired in page.tsx. Directly implements the vision's
      "Conversation over configuration" principle — analysts never need to leave the chat to run a quick prediction.
      *Day 25 (20:00): 17 backend + 15 frontend = 32 new tests. Total: 2569 backend + 1252 frontend = 3821, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Goal-Driven Training** — Analysts can say "I need 85% accuracy", "reach 0.85 R² for revenue prediction",
      "hit 80% F1 score", or "train a model until it reaches 90% accuracy" and AutoModeler autonomously tries
      algorithms in order (fast → accurate: linear/logistic, Random Forest, Gradient Boosting) until the target
      is achieved — then stops early. If no algorithm hits the goal, hyperparameter tuning is attempted on the best
      candidate. `_GOAL_TRAIN_PATTERNS` (8 NL variants) in `chat.py` detects intent. `_extract_goal_target()` helper
      extracts (metric, threshold) from the message: `85%` → `("accuracy", 0.85)`, `0.85 R²` → `("r2", 0.85)`.
      `run_goal_driven_training(X, y, problem_type, goal_metric, goal_target, model_dir, base_id)` pure function
      in `core/trainer.py` sub-samples datasets >5,000 rows for speed (trial mode), trains each algorithm via
      `train_single_model()`, stops on first success, falls back to `tune_model()` on best if goal not met.
      Returns `{goal_metric, goal_target, achieved, winner_algorithm, winner_algorithm_name, winner_score, trials,
      tried_tuning, summary}` — each trial records `{algorithm_name, score, achieved_goal}`. `{type:"goal_training"}`
      SSE event. `GoalTrainingCard` (emerald border if achieved, amber if not, 🎯 icon): "Goal Achieved ✓" or "Best
      Effort" badge, goal-target badge ("R² ≥ 0.75"), winner highlight box with score, trials table (algorithm, score,
      ✓/✗), tuning note, plain-English summary. `GoalTrainingResult`/`GoalTrainingTrial` TypeScript types;
      `attachGoalTrainingToLastMessage` Zustand action; SSE wired in page.tsx. Directly closes the vision gap: analysts
      who say "I need at least 80% accuracy" get an autonomous answer instead of manually comparing runs.
      *Day 26 (04:00): 26 backend + 16 frontend = 42 new tests. Total: 2595 backend + 1268 frontend = 3863, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Prediction Sensitivity Analysis** — Analysts can ask "how sensitive is revenue to units?",
      "sweep price from 10 to 100", "run a sensitivity analysis on quantity", or "show me how the
      prediction changes as units varies" and receive a `SensitivityCard` with a line chart showing
      predicted output as a single feature sweeps through a range of values. `_SENSITIVITY_PATTERNS`
      (8 NL variants: "sensitivity analysis on/for", "how sensitive is X to Y", "sweep X from A to B",
      "vary X from A to B", "run a sensitivity", "how does prediction change as X varies", "effect of X
      on prediction", "show how X affects prediction") in `chat.py`. `_detect_sensitivity_request()`
      extracts feature name (longest-match scan against feature list), range (explicit "from X to Y"
      or ±50% around training mean), and step count (default 10, explicit "N steps"). Handler is
      guarded to fire only when a deployment exists and neither what-if nor inline prediction already
      fired (avoiding duplicate prediction cards). `run_sensitivity_analysis(pipeline_path, model_path,
      feature_name, sweep_values, base_features)` pure function in `core/deployer.py` sweeps one
      feature across supplied values, holds all others at training means, collects regression predictions
      (or top-class confidence for classification), computes min/max/change_pct, and builds a
      plain-English summary ("As units varies from 5 to 20, revenue increases by 300% (500 → 2000)").
      `{type:"sensitivity"}` SSE event. `SensitivityCard` (teal border, 🎚️ icon): feature → target
      heading, Regression/Classification badge, change % badge (↑ emerald / ↓ rose), min/max prediction
      boxes, Recharts line chart (feature value X axis, prediction Y axis) for regression or confidence
      curve for classification, fallback table of (feature value, predicted class) when no numeric curve.
      `SensitivityResult` TypeScript type; `attachSensitivityToLastMessage` Zustand action; SSE handler
      wired in workspace page. Directly closes the "how much does my prediction move if this input
      changes?" analyst question — complementary to what-if (single value) and inline prediction (explicit
      multi-feature inputs). A business analyst presenting to a VP can now say "here's the curve showing
      exactly how revenue responds to changes in units sold."
      *Day 26 (12:00): 24 backend + 17 frontend = 41 new tests. Total: 2619 backend + 1285 frontend = 3904, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Guided Onboarding Wizard** — When a first-time analyst says "guide me", "help me get started",
      "walk me through the steps", "what should I do first?", or "onboarding", AutoModeler responds with
      an `OnboardingGuideCard` showing their exact progress through the 6-step workflow. `_ONBOARDING_PATTERNS`
      (8 NL variants) in `chat.py` detects the intent. `compute_onboarding_state(has_dataset, message_count,
      has_target, has_model_run, has_cross_val, has_deployment)` pure function in `core/onboarding.py` maps
      progress flags to a structured state: `step_index`, `total_steps`, `completion_pct`, `steps` (each with
      `title`, `description`, `hint`, `suggested_action`, `suggested_tab`, `icon`, `is_done`, `is_current`),
      `current_step`, `is_complete`, `summary`. Six steps: Upload → Explore → Set target → Train → Validate
      → Deploy. `GET /api/projects/{project_id}/onboarding` endpoint derives state from project records
      (dataset, conversation count, feature set, model runs with cross-val, deployment). Chat handler emits
      `{type:"onboarding_guide"}` SSE event with full state; Claude's system prompt is injected with step
      context for natural narration. `OnboardingGuideCard` (blue border, 🧭 icon): "Getting Started Guide"
      heading, completion % badge, progress bar (aria-valuenow), step list (✓ for done steps, current-step
      icon for active, ○ for upcoming), current step description + italic hint tip, CTA button that fires
      `onSwitchTab` callback to navigate to the relevant panel tab. Complete state shows celebration message.
      `OnboardingGuideResult`/`OnboardingStep` TypeScript types; `attachOnboardingGuideToLastMessage` Zustand
      action; SSE wired in page.tsx. Directly closes the "I just uploaded data but don't know what to do
      next" cold-start gap — the single biggest barrier for business analysts adopting new tools.
      *Day 26 (20:00): 26 backend + 16 frontend = 42 new tests. Total: 2645 backend + 1301 frontend = 3946, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Data Version History Timeline** — Analysts can ask "show my upload history", "data version timeline",
      "how has my data changed over time", or "what datasets do I have?" and receive a `DataVersionHistoryCard`
      in chat showing a timeline of every dataset upload for the project. `_VERSION_HISTORY_PATTERNS` (8 NL
      variants) in `chat.py` detects intent. `compute_version_history(datasets, dataframes)` pure function in
      `core/analyzer.py` builds the timeline: for each consecutive upload pair it calls the existing
      `compute_dataset_comparison()` and extracts a `drift_from_previous` dict with `drift_score`, `summary`,
      `changed_columns`, `new_columns`, `dropped_columns`, and `row_count_change_pct`. Returns `version_count`,
      `versions` list (each with `version`, `dataset_id`, `filename`, `row_count`, `column_count`, `uploaded_at`,
      `size_bytes`, `drift_from_previous`), `overall_stability` (stable/moderate/high, from max drift across
      transitions), and plain-English `summary`. `GET /api/data/{project_id}/version-history` REST endpoint
      orders datasets by `uploaded_at` ascending, builds dicts and DataFrames, returns the history. Chat
      handler injects stability + summary into the system prompt so Claude narrates the timeline naturally.
      `{type:"version_history"}` SSE event. `DataVersionHistoryCard` (adaptive border, 📂 icon): stability
      badge (Stable/Moderate Drift/High Drift) + version count header; timeline rendered latest-first with
      version number badge (blue for latest), filename, upload date, row/column/size info, and "Latest" label
      on newest; between each pair a drift connector shows drift score, changed column count, and row % change
      with color coding (green <20, amber 20–49, red ≥50). `DataVersionDrift`, `DataVersionEntry`,
      `DataVersionHistoryResult` TypeScript types; `attachVersionHistoryToLastMessage` Zustand action; SSE
      handler and render wired in page.tsx. Closes the "how has my data evolved across uploads?" gap — a
      business analyst who uploads Q2 data needs to immediately know how different it is from Q1 before
      deciding whether to retrain their model.
      *Day 27 (04:00): 22 backend + 18 frontend = 40 new tests. Total: 2667 backend + 1319 frontend = 3986, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Feature Interaction Analysis** — Analysts can ask "how do units and price interact?", "show me the
      interaction between region and category", "joint effect of units and price", "feature interaction heatmap",
      or "2D sensitivity" and receive an `InteractionCard` showing a 2-D prediction grid — how two features
      jointly affect the model's output. `_INTERACTION_PATTERNS` (8 NL variants) + `_detect_interaction_request()`
      in `chat.py` — scans the message for the two longest column names present. Handler is guarded to fire only
      when a deployment exists and neither sensitivity, what-if, nor inline prediction already fired (preventing
      double-prediction cards). `run_feature_interaction(pipeline_path, model_path, feature1, feature2,
      base_features, n_steps=7)` pure function in `core/deployer.py`: for numeric features, sweeps
      `[mean ± 2×std]` in n_steps; for categorical features, uses all known classes from the label encoder.
      Builds an n×m grid of predictions holding all other features at training means. Returns `{feature1,
      feature2, target_column, problem_type, row_labels, col_labels, values, min_val, max_val, summary}`.
      Plain-English summary names the prediction range and suggests looking for the highest-value combination.
      `{type:"interaction"}` SSE event. `InteractionCard` (violet border, 🔬 icon): Regression/Classification
      badge, min/max prediction boxes (regression), 2-D heatmap table with color-coded cells (rose=low through
      emerald=high for regression, violet=class for classification), Low/High legend, corner cell labels both axes,
      truncated long labels with title tooltip, summary footer. `InteractionResult` TypeScript type;
      `attachInteractionToLastMessage` Zustand action; SSE handler and render wired in page.tsx. Closes the
      "which combination of two variables produces the best outcome?" analyst question — a VP-facing insight
      that goes beyond single-feature sensitivity to show true interaction effects.
      *Day 28 (04:00): 25 backend + 19 frontend = 44 new tests. Total: 2734 backend + 1371 frontend = 4105, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Prediction Cohort Analysis + CSV Export** — After ranking predictions ("top 20 at-risk customers"), analysts can ask "who are they?", "what do the top predictions have in common?", "profile the ranked records?", or "characterize the at-risk group" and receive a `PredictionCohortCard` profiling the top-N as a cohort vs the full dataset. `_COHORT_PATTERNS` (9 NL variants) + handler (fires only when deployment + dataset exist and ranked_pred hasn't already fired) in `chat.py`. `compute_prediction_cohort(pipeline_path, model_path, df, n, direction)` pure function in `core/deployer.py`: re-runs `run_dataset_ranking()` to get top-N indices, then computes **categorical breakdown** (per-category top-N% vs overall% vs ratio, capped at 5 categories per column, skipping >20 unique) and **numeric comparison** (top-N mean vs overall mean, ratio, direction label "higher"/"lower"/"similar"; target column excluded). Generates plain-English `characterization`: "The 20 highest-scoring revenue predictions: 70% have region = 'East'; units is 80% higher on average." Returns `{target_column, problem_type, n, direction, total_scored, categorical_profile, numeric_profile, characterization}`. `{type:"prediction_cohort"}` SSE event. `PredictionCohortCard` (indigo border, 🔍 icon): Highest/Lowest badge, n-of-total badge, characterization paragraph, "Categorical Breakdown" with dual horizontal bars (indigo=top-N, slate=all rows; legend), "Numeric Averages" with per-column rows (column name, ratio badge colored rose≥1.5×/amber≥1.2×/sky≤0.7×/indigo≤0.85×/slate otherwise, top avg vs overall avg); graceful "No additional features to profile" empty state. **CSV download for ranked predictions**: "⬇ Download CSV" button added to `RankedPredictionsCard` header — client-side `buildCsv()` generates CSV from SSE data including rank, row_index, all prediction columns, and ALL feature_values columns (not just the 4 shown in the table); triggers `<a download>` with filename `{target_column}_ranked_predictions.csv`. `PredictionCohortResult`/`CohortCategoricalProfile`/`CohortNumericProfile`/`CohortCategoryEntry` TypeScript types; `attachPredictionCohortToLastMessage` Zustand action; SSE handler and render wired in page.tsx. Closes the "who ARE the top-N predictions?" analyst gap — a smart colleague who built a churn model would immediately say "these 20 are predominantly West-region customers with high monthly bills and short tenure — here's why they stand out."
      *Day 28 (20:00): 24 backend + 18 frontend = 42 new tests. Total: 2782 backend + 1406 frontend = 4188, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Dataset Ranking via Model** — Analysts can ask "which customers are most likely to churn?", "show me
      the top 20 predictions", "rank by predicted revenue", "which records are most at risk?", or "apply the
      model to all my data" and receive a `RankedPredictionsCard` showing the top N rows from the current
      dataset ranked by model prediction. Closes the gap between "having a model" and "knowing which specific
      records to act on". `_RANKED_PRED_PATTERNS` (8 NL variants: "which X are most likely to...", "rank by
      predicted...", "top/bottom N predictions", "most at risk", "best/worst opportunities", "apply the model
      to all data") + `_detect_ranked_pred_request()` in `chat.py` — extracts n (default 20, capped at 100)
      and direction (highest/lowest) from the message. Handler fires only when a deployment and dataset both
      exist and no other prediction card (interaction/sensitivity/whatif/inline-pred) has already fired.
      `run_dataset_ranking(pipeline_path, model_path, df, n=20, direction="highest")` pure function in
      `core/deployer.py`: applies `pipeline.transform_df(df)` to all rows, calls `model.predict()` for
      regression (ranks by value) or `model.predict_proba()` for classification (ranks by max class
      confidence); returns `{problem_type, target_column, direction, n, total_scored, rows, summary,
      class_names}` with each row containing `{rank, row_index, score, feature_values, prediction or
      predicted_class+confidence+probabilities}`. Empty dataset raises `ValueError`. Plain-English summary
      names total rows scored and the top predicted value/class. `{type:"ranked_predictions"}` SSE event.
      `RankedPredictionsCard` (amber border, 🏆 icon): gold/silver/bronze rank badges for top 3 rows, n-of-
      total count badge, Highest/Lowest direction badge, Regression/Classification problem-type badge, sortable
      table with prediction value (regression: sky formatted number; classification: class + confidence % in
      color-coded badge), top 4 feature columns with values, summary footer. `RankedPredictionRow` +
      `RankedPredictionsResult` TypeScript types; `attachRankedPredictionsToLastMessage` Zustand action; SSE
      handler and render wired in page.tsx. Directly implements the vision's "smart colleague" promise: a
      colleague who, after building a churn model, immediately says "these are the 20 customers most likely
      to leave — here's who to call first."
      *Day 28 (12:00): 24 backend + 17 frontend = 41 new tests. Total: 2758 backend + 1388 frontend = 4146, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Prediction Presets on the VP Dashboard** — Analysts can define named "quick-fill" scenarios (e.g., "Best Case",
      "Average Quarter", "Worst Case") via chat and have them appear as one-click buttons on the shared prediction
      dashboard, so VPs and colleagues can instantly load realistic scenarios without knowing the feature names.
      `_PRESET_SAVE_PATTERNS` (8 NL variants: "save this as a preset called X", "add a preset called X with Y=Z",
      "create a prediction preset named X", "save as a named scenario called X", "bookmark this as preset",
      "quick scenario called X") + `_PRESET_LIST_PATTERNS` (4 NL variants: "show my presets", "list saved scenarios",
      "what presets do I have", "show existing presets") in `chat.py`. `_extract_preset_definition()` parses the
      preset name (via `called|named` lookahead regex) and feature `key=value` pairs (using `=` only, excluding
      `:` to avoid collisions with the name separator). `DeploymentPreset` SQLModel table (`id`, `deployment_id`,
      `name`, `feature_values` JSON, `created_at`). `GET/POST /api/deploy/{id}/presets` + `DELETE
      /api/deploy/{id}/presets/{preset_id}` CRUD endpoints (validates non-empty name, non-empty feature_values;
      returns 422 on either). Chat handlers emit `{type:"preset_saved"}` and `{type:"preset_list"}` SSE events.
      `PresetSavedCard` (emerald border, 🎯 icon): preset name, feature-count badge, per-feature `key=value` badges.
      `PresetListCard` (indigo border, 📋 icon): count badge, per-preset rows with feature badges and Load button.
      `predict/[id]/page.tsx` augmented with `useEffect` loading presets from `GET /api/deploy/{id}/presets`; when
      any exist, a "Quick Scenarios" row of rounded pill buttons appears above the input form — clicking fills all
      fields with the preset values and clears the previous result. `DeploymentPreset`/`PresetSavedInfo`/`PresetListInfo`
      TypeScript types; `api.deploy.getPresets/createPreset/deletePreset()` client methods; `attachPresetSavedToLastMessage`
      + `attachPresetListToLastMessage` Zustand actions; SSE handlers and renders wired in `page.tsx`. Directly closes
      the "VP doesn't know what to type" cold-start gap on the shared prediction dashboard — the analyst does the
      thinking once, the VP clicks "Best Case" or "Conservative" and instantly sees the model's answer.
      *Day 29 (04:00): 25 backend + 20 frontend = 45 new tests. Total: 2807 backend + 1426 frontend = 4233, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Multi-Row Batch Prediction via Chat** — Analysts can say "predict for: Region=East, Units=100; Region=West, Units=150" and receive a `MultiPredictionCard` showing a comparison table of all scenarios in one shot — no need to run predictions one at a time. `_MULTI_ROW_PRED_PATTERNS` (6 NL variants: "batch/bulk predict for these scenarios", "run predictions for multiple records", "score these inputs", "compare these scenarios") + key detection: any message containing `;` AND matching `_INLINE_PRED_PATTERNS` triggers the multi-row path. `_extract_multi_row_predictions(message, feature_names)` splits the message on `;`, trims leading preamble from each segment (using `_trim_preamble()` which finds the first occurrence of a known feature key to avoid parsing "for:" as a k-v pair), then calls `_extract_multi_feature_prediction()` per segment; returns a list only when ≥2 rows are found. The handler is guarded to fire before `inline_pred_event` (mutual exclusion: multi-row takes priority, inline fires only when multi-row didn't). For each parsed row: merges provided features with training means, calls `predict_single()`, records `{row_index, provided_features, defaults_used_count, prediction, probabilities, confidence, confidence_interval}`. Regression summary: "N predictions for target: range min–max". Classification summary: "N predictions for target: most common = class". `{type:"multi_prediction"}` SSE event. `MultiPredictionCard` (violet border, 📊 icon): "Scenario Comparison" heading, N-scenarios + Regression/Classification badges; compact HTML table with row-index column, prediction column (regression: compact-formatted number; classification: top-class + confidence % in color-coded badge), up to 4 provided-feature columns with values (dash for missing), defaults count column when any row used defaults; summary footer; "Features not specified... used training-data averages" note. `MultiPredictionRow` + `MultiPredictionResult` TypeScript types; `attachMultiPredictionToLastMessage` Zustand action; SSE handler and render wired in `page.tsx`. Directly closes the "compare multiple what-if scenarios at once" analyst gap — distinct from Bulk Scenario Comparison (override-based) and Inline Prediction (single row).
      *Day 29 (20:00): 17 backend + 15 frontend = 32 new tests. Total: 2824 backend + 1441 frontend = 4265, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **AI-powered data dictionary** — When a dataset is uploaded (or on demand via POST), auto-generate
      plain-English descriptions for every column. `core/dictionary.py` classifies each column as
      id/metric/dimension/date/flag/text via heuristics (name hints + dtype + cardinality), then uses
      Claude (with static fallback) to generate descriptions in one batch call. Stored back into
      `Dataset.columns` JSON. GET /api/data/{id}/dictionary (fast, on-demand) + POST /api/data/{id}/dictionary
      (generates/regenerates). `DictionaryCard` in the Data tab: "Quick summary" uses static descriptions,
      "AI descriptions" calls Claude; type badges colour-coded (Metric=blue, Dimension=purple, Date=green,
      ID=gray, Flag=yellow, Text=orange); show/hide for datasets with >8 columns; Regenerate button.
      `DataDictionary` + `ColumnDescription` + `ColumnSemanticType` types; `api.data.getDictionary()` +
      `api.data.generateDictionary()` client methods. Closes the "smart colleague" promise — a colleague
      would explain what each column means to an analyst inheriting unfamiliar data.
      *Day 9 (08:07): 32 backend + 15 frontend = 47 new tests. Total: 1096 backend + 426 frontend = 1522.*

- [x] **Pivot table / cross-tabulation analysis** — Business analysts can ask "break down revenue by region
      and product" in chat and receive an interactive pivot table inline in the conversation (not in a side
      panel). `build_crosstab()` in `core/chart_builder.py` uses `pd.pivot_table` (sum/mean/count/min/max),
      caps rows/cols at configurable limits, and returns a JSON structure with col_headers, rows (with cells +
      row_total), col_totals, grand_total, and a plain-English summary. `GET /api/data/{id}/crosstab?rows=&cols=&values=&agg=`
      endpoint for direct API access. Chat intent detection via `_CROSSTAB_PATTERNS` + `_detect_crosstab_request()`
      helper that extracts row_col/col_col/value_col from natural language; the inferred crosstab is computed
      inline and injected into the system prompt so Claude can narrate the findings. `{type: "crosstab"}` SSE
      event attaches the table to the last chat message via `attachCrosstabToLastMessage()` in the Zustand store.
      `CrosstabTable` component renders a zebra-striped HTML table with truncated labels, a "Total" column
      (row sums) and "Total" row (column sums + grand total). `CrosstabResult` type; `api.data.getCrosstab()`.
      *Day 9 (04:00): 19 backend + 12 frontend = 31 new tests. Total: 1115 backend + 438 frontend = 1553.*

- [x] **Computed columns through conversation** — Business analysts can say "add a column called margin = revenue / cost"
      and receive an interactive `ComputeCard` in the Data tab (not just a text response) that shows the formula, sample
      preview values, and dtype before they confirm. `core/computed.py` uses `pd.DataFrame.eval()` for safe expression
      evaluation — only arithmetic/comparison on column names, no arbitrary Python execution. `POST /api/data/{id}/compute`
      writes the new column back to the CSV in-place and refreshes the Dataset profile. Chat intent detected via
      `_COMPUTE_PATTERNS` + `_detect_compute_request()` (extracts column name and expression, verifies ≥1 existing column
      appears in the expression). `{type: "compute_suggestion"}` SSE event pushes the suggestion to the frontend;
      `attachComputeToLastMessage()` in the Zustand store links it to the chat turn. `ComputedColumnSuggestion` +
      `ComputeResult` types; `api.data.computeColumn()` client method. Follows the "explain before executing" pattern —
      the column is never auto-added; user must click Apply.
      *Day 9 (12:00): 26 backend + 11 frontend = 37 new tests. Total: 1141 backend + 449 frontend = 1590.*

- [x] **Cross-deployment model comparison** — POST /api/predict/compare accepts 2-4 deployment IDs and
      a feature dict, returns predictions from each model version so analysts can verify whether a retrained
      model improved on their specific inputs. GET /api/deployments now accepts optional `?project_id=` filter
      for project-scoped listing. `CompareModelsCard` on the public predict/[id] page auto-detects other
      deployed versions of the same project and shows a side-by-side comparison table (algorithm, trained date,
      prediction, uncertainty) when expanded. The `/api/predict/compare` route is registered BEFORE
      `/api/predict/{deployment_id}` so FastAPI doesn't try to match "compare" as a UUID.
      `ModelComparisonResult` + `ComparisonResponse` types; `api.deploy.compareModels()` + `api.deploy.listByProject()`.
      *Day 9 (20:00): 11 backend + 10 frontend = 21 new tests. Total: 1064 backend + 411 frontend = 1475.*

- [x] **Developer API integration snippets** — When a model is deployed, analysts can share it with their developer via auto-generated copy-pasteable code examples.
      `GET /api/deploy/{id}/integration` returns curl, Python (requests), and JavaScript (fetch) snippets built from the deployment's feature schema. Custom `base_url` query param allows overriding `localhost` for production. `IntegrationCard` in DeploymentPanel: shows endpoint URL + algorithm/target info, tabbed code blocks (curl/Python/JavaScript) with copy-to-clipboard, batch prediction curl note, OpenAPI docs link. Implements the vision's "An API their developer can plug into the company's reporting tool."
      *Day 9 (16:10): 18 backend + 16 frontend = 34 new tests. Total: 1159 backend + 465 frontend = 1624.*

- [x] **Segment comparison analysis** — Business analysts can say "compare East vs West" or "difference between enterprise and SMB" and receive an inline `SegmentComparisonCard` in the chat with side-by-side mean/std/count per numeric column for each group. `compare_segments()` in `core/analyzer.py` computes Cohen's d effect sizes, identifies notable differences (abs effect > 0.5) sorted by magnitude, and returns a plain-English summary. Chat intent detected via `_COMPARE_PATTERNS` + `_detect_compare_request()` — the helper scans all DataFrame columns for one that contains both extracted terms as actual values, resolving the group column automatically. `{type: segment_comparison}` SSE event attaches via `attachSegmentToLastMessage()` in the Zustand store. `GET /api/data/{id}/compare-segments?col=&val1=&val2=` provides direct API access with 400 validation for missing values. `SegmentComparisonCard` renders a zebra-striped table with val1 in blue / val2 in purple, effect badges (similar/moderate/large/very large) colour-coded from blue to orange, direction arrows (↑ East / ↑ West), and notable-column rows highlighted in amber. `SegmentComparisonResult` + `SegmentColumnStats` + `SegmentNotableDiff` types; `api.data.compareSegments()` client method. Directly implements the analyst use case: "my VP wants to know why East region outperforms West — what's actually different?"
      *Day 9 (12:00 session 2): 22 backend + 12 frontend = 34 new tests. Total: 1181 backend + 477 frontend = 1658.*

- [x] **Time-series forecasting** — Business analysts can ask "predict next 6 months of revenue" and receive a `ForecastChart` inline in the chat. `forecast_next_periods()` in `core/forecaster.py` detects data frequency (daily/weekly/monthly/quarterly), engineers time-based features (trend index, cyclic sin/cos for month and day-of-week), trains `LinearRegression`, and returns 95% prediction intervals (residual-std CI). `detect_time_series()` auto-detects date + numeric columns; `GET /api/data/{id}/forecast?target=&periods=6` REST endpoint. Chat intent via `_FORECAST_PATTERNS + _detect_forecast_request()` → `{type:"forecast"}` SSE event. `ForecastChart` renders solid historical line + dashed forecast line + shaded CI band; `TrendBadge` shows ▲/▼/→ with % change. Closes the vision scenario: "predict next quarter's revenue by region."
      *Day 10 (00:04): 41 backend + 12 frontend = 53 new tests. Total: 1222 backend + 489 frontend = 1711.*

- [x] **Data readiness assessment** — Before an analyst clicks Train, they can ask "is my data ready?" and receive an inline `ReadinessCheckCard` with a 0-100 score, letter grade (A-F), status badge (Ready/Needs Attention/Not Ready), per-component breakdown, and actionable recommendations. `compute_data_readiness()` in `core/readiness.py` scores 5 components: row count (25pts), missing values (25pts), duplicate rows (20pts), feature diversity (15pts), data type quality (15pts). Optional `target_col` adds a class-balance advisory check that doesn't affect the weighted total. `GET /api/data/{id}/readiness-check?target=` REST endpoint. Chat intent via `_DATA_READINESS_PATTERNS` → `{type:"data_readiness"}` SSE event; `attachDataReadinessToLastMessage()` Zustand store action. `ReadinessCheckCard` shows score gauge + progress bars + status icons + recommendations; also appears in the Data tab with a lazy "Check Readiness" button. Directly implements the "explain before executing" vision principle — analysts know their data is ready before training.
      *Day 10 (08:02): 39 backend + 14 frontend = 53 new tests. Total: 1261 backend + 503 frontend = 1764.*

- [x] **Target correlation analysis** — Business analysts can ask "what drives revenue?" or "what's correlated with profit?" and receive an inline `CorrelationBarCard` in the chat showing which numeric columns are most correlated with the named target, ranked by absolute Pearson r. `analyze_target_correlations()` in `core/analyzer.py` returns a ranked list with strength labels (very strong/strong/moderate/weak/negligible), direction (positive/negative), and a plain-English summary. `GET /api/data/{id}/target-correlations?target=&top_n=10` REST endpoint (400 on non-numeric or missing column). Chat intent via `_CORRELATION_TARGET_PATTERNS` + `_detect_correlation_target_request()` (scans actual DataFrame column names against the user's message; falls back to feature-set target if no column named explicitly). `{type:"target_correlation"}` SSE event; `attachCorrelationToLastMessage()` Zustand store action. `CorrelationBarCard` renders horizontal ranked bars: blue=positive, red=negative, width proportional to strength relative to the strongest correlation; strength badges (very strong, strong, moderate, weak) color-coded. Directly implements "which factors affect my outcome?" — the #1 analyst question before modeling. `TargetCorrelationResult` + `CorrelationEntry` types; `api.data.getTargetCorrelations()` client method.
      *Day 10 (04:00): 34 backend + 11 frontend = 45 new tests. Total: 1295 backend + 515 frontend = 1810.*

- [x] **Chat-triggered correlation heatmap + click interactivity** — Business analysts can ask "show me the correlation matrix", "heatmap", or "how are my columns related" and receive the full pairwise Pearson correlation heatmap inline in the chat. `_HEATMAP_PATTERNS` in `chat.py` detects these phrases → computes from cached profile → emits `{type:"chart"}` which reuses the existing chart SSE path and `HeatmapChart` renderer. The `HeatmapChart` now supports click-to-highlight: clicking a cell shows a focused tooltip below the grid with the exact r value (blue=positive, red=negative), highlights the row and column labels, and provides a ✕ dismiss button; clicking the same cell deselects. Directly addresses the "show me how my columns relate" exploration request that previously required knowing to check the Data tab.
      *Day 10 (12:00): 9 backend + 7 frontend = 16 new tests. Total: 1350 backend + 545 frontend = 1895.*

- [x] **Column rename through conversation** — Business analysts can say "rename revenue_usd to Revenue" or "change the name of rev_q1_adj to Q1_Revenue" and have the column renamed in-place across the CSV, profile, and Dataset record. `_RENAME_PATTERNS` + `_detect_rename_request()` in `chat.py` extract old/new names with case-insensitive column matching; the rename executes synchronously (no "suggest before execute" step needed — a rename is unambiguous) and emits `{type:"rename_result"}` SSE. `POST /api/data/{id}/rename-column` provides direct REST access with full validation: column must exist, new name must be word-chars-only (no spaces/specials), no naming conflicts. `RenameResultCard` renders `old_name` ~~strikethrough~~ → `new_name` highlighted in the chat turn. `api.data.renameColumn()` client method added. Closes the "inherited cryptic column names" analyst pain point.
      *Day 10 (12:00): 18 backend + 10 frontend = 28 new tests.*

- [x] **Group-by analysis** — Business analysts can ask "show me revenue by region" or "breakdown by product" and receive an inline `GroupStatsCard` in the chat with ranked horizontal bars. `compute_group_stats()` in `core/analyzer.py` supports sum/mean/count/min/max/median aggregations, caps output at 30 groups, and returns a plain-English summary ("Highest: West (500.00). Top group is 46.3% of the total."). `GET /api/data/{id}/group-stats?group_by=&metrics=&agg=` REST endpoint (400 on invalid column). Chat intent via `_GROUP_PATTERNS` + `_detect_group_request()` — scans DataFrame columns in the user's message to auto-identify the group-by column (categorical) vs value columns (numeric); detects aggregation keyword (average/count/min/max/median). `{type:"group_stats"}` SSE event; `attachGroupStatsToLastMessage()` Zustand store action. `GroupStatsCard` renders ranked horizontal bars (blue intensity by rank), group count + total in header, summary footer. `GroupStatsResult` + `GroupStatsRow` types; `group_stats` field on `ChatMessage`. Closes the most common analyst analysis pattern — "how does X break down by Y?" — that pivot tables can't answer in a single natural question.
      *Day 10 (16:02): 28 backend + 13 frontend = 41 new tests. Total: 1323 backend + 528 frontend = 1851.*

- [x] **Chat-initiated model training** — Business analysts can say "train a model to predict revenue" in the chat and have training start immediately — no need to leave the conversation and navigate to the Models tab. `_TRAIN_PATTERNS` regex detects "train a model", "build a predictor", "start training" etc. `_detect_train_target()` scans the message for a column name ("predict X", "target is X", or any known column). Three cases handled: (A) feature set + target already set → start training directly; (B) feature set exists, no target → extract target from message, set it on the feature set, start training; (C) no feature set → create a minimal one with all columns + detected target, start training. Uses the same `_train_in_background` daemon threads and `_training_queues`/`_training_counters` shared state as the Models tab, so the `/api/models/{id}/runs/stream` SSE endpoint works seamlessly. Emits `{type:"training_started"}` SSE event after launching threads. `TrainingStartedCard` shows target column, problem type badge, algorithm chips, run count, and "Check the Models tab" CTA. `TrainingStartedResult` type; `attachTrainingStartedToLastMessage()` Zustand store action. Closes the biggest conversational workflow gap — analysts can stay in chat through the full upload → explore → train flow without context-switching to UI panels.
      *Day 10 (20:00): 18 backend + 12 frontend = 30 new tests. Total: 1368 backend + 557 frontend = 1925.*

- [x] **Automated data story** — Business analysts can ask "analyze my data", "walk me through this dataset", or "what's interesting here?" and receive a single comprehensive `DataStoryCard` that orchestrates all available analysis modules into one narrative. `generate_data_story()` in `core/storyteller.py` runs up to 4 sections: (1) Data readiness — always shown; (2) Group-by breakdown — on the best categorical column (moderate unique count); (3) Target correlations — only if `target_col` is provided; (4) Anomaly count — if numeric columns and ≥10 rows. Returns `{dataset_id, filename, row_count, col_count, readiness_score, readiness_grade, sections, summary, recommended_next_step}`. `GET /api/data/{id}/story?target=` REST endpoint. Chat intent via `_STORY_PATTERNS` → `{type:"data_story"}` SSE event; `attachDataStoryToLastMessage()` Zustand store action. `DataStoryCard` renders: header (filename, row/col count, grade badge), readiness score bar, per-section icons + titles + insights (readiness=📊, group_by=📈, correlations=🔗, anomalies=⚠️), recommended next step footer. Grade badge colors: A=green, B=blue, C=yellow, D=orange, F=red. Key implementation detail: uses `pd.api.types.is_string_dtype()` alongside `dtype == object` to correctly detect string columns under pandas 4.x which uses `StringDtype` instead of `object`. `_build_summary()` and `_recommend_next_step()` are independently exported for unit testing. `DataStory` + `DataStorySection` types; `api.data.getDataStory()` client method. Closes the "smart colleague walks you through your data in one ask" vision promise — the single most natural first question for any analyst inheriting an unfamiliar dataset.
      *Day 11 (04:00): 45 backend + 13 frontend = 58 new tests. Total: 1413 backend + 570 frontend = 1983.*

- [x] **Non-destructive data filter via chat** — Business analysts can say "focus on Q4 data", "filter to North region", or "show only revenue > 500" and all subsequent analyses run on the filtered subset without modifying the underlying CSV. Separate `DatasetFilter` SQLModel table stores one active filter per dataset (avoids ALTER TABLE migration issues). `core/filter_view.py` provides `parse_filter_request()` (NL → `list[FilterCondition]`), `apply_active_filter()` (pandas boolean indexing, AND logic), `build_filter_summary()` (plain-English description), and `validate_filter_conditions()`. `_load_working_df()` helper in `api/chat.py` centralises the "load CSV + apply active filter" pattern — replaces all 13 `pd.read_csv()` calls so every existing analysis automatically respects the active filter. `_FILTER_PATTERNS` and `_CLEAR_FILTER_PATTERNS` regex groups detect filter intents; `filter_set` and `filter_cleared` SSE events. REST: `POST /{id}/set-filter`, `DELETE /{id}/clear-filter`, `GET /{id}/active-filter`. Frontend: `FilterSetCard` renders conditions with operator symbols (`eq`→`=`, `gt`→`>`, etc.) and row-reduction stats inline in chat; `FilterBadge` in the Data tab header shows the active filter with a ✕ clear button. Zustand: `activeFilter` state + `attachFilterToLastMessage()` + `setActiveFilter()`. Operator normalization handles NL variants ("is", "equals", "greater than") → internal ops (`eq`, `gt`). `FilterCondition`, `ActiveFilter`, `FilterSetResult` types; `api.data.setFilter()`, `clearFilter()`, `getActiveFilter()` client methods. Closes the "narrow the analysis context" workflow — analysts no longer need to physically slice their CSV to focus on a segment.
      *Day 11 (12:00): 34 backend + 24 frontend = 58 new tests. Total: 1447 backend + 594 frontend = 2041.*

- [x] **Chat-driven model deployment** — Business analysts can say "deploy my model", "go live", "make it live", "publish my model", or "ship my model" in chat and have their best or selected model deployed immediately — no UI navigation required. `_DEPLOY_CHAT_PATTERNS` (9 variants) in `chat.py` detects deployment intent. The handler finds the selected model run (or falls back to the best completed run by R²/accuracy), then calls `execute_deployment()` (a helper extracted from the existing deploy route in `api/deploy.py`) — idempotent, returns the existing deployment if already active. Emits `{type:"deployed"}` SSE event. `DeployedCard` renders inline in the chat: green live indicator, algorithm label + problem type badge, primary metric (R²/accuracy), target column, dashboard link (Open →), and API endpoint URL with copy-to-clipboard. `DeployedResult` type; `attachDeployedToLastMessage()` Zustand store action. If no completed models exist, the system prompt guides the user to train first — no crash. Closes the final gap in the "full workflow through chat" vision promise: upload → explore → train → **deploy**, all without leaving the chat window.
      *Day 11 (20:00): 17 backend + 18 frontend = 35 new tests. Total: 1464 backend + 612 frontend = 2076.*

- [x] **"Explain my model" conversational model card** — Business analysts can ask "explain my model", "what does my model do", "how does my model work", "model summary", or "what drives my predictions" and receive an inline `ModelCardView` in the chat that synthesises all model evidence into plain English. `GET /api/models/{project_id}/model-card` finds the selected model run (or best completed by primary metric), loads the joblib pipeline for feature importances, and returns: `algorithm_name`, `problem_type`, `target_col`, `metric` (with `plain_english` interpretation — "explains most patterns in your data", "predicts correctly 9/10 times"), `top_features` (ranked importance bars from `compute_feature_importance()`), `limitations` (honest assessment: small dataset, low accuracy, few features), and a one-sentence `summary`. Helper functions `_algorithm_plain_name()`, `_metric_plain_english()`, `_build_limitations()` are independently exported for unit testing. `_MODEL_CARD_PATTERNS` (9 variants) in `chat.py` + system prompt injection guides Claude to narrate the card conversationally ("imagine explaining to a VP who doesn't know ML"). `{type:"model_card"}` SSE event; `attachModelCardToLastMessage()` Zustand store action; `ModelCard` + `ModelCardMetric` + `ModelCardFeature` TypeScript types. `ModelCardView` renders: algorithm chip, problem type badge, Live indicator if deployed, metric value + plain-English context, horizontal importance bars with %-labeled widths, amber limitation callout, footer stats (rows/features/target). Closes the "Not a black box" vision promise for the chat interface — analysts can understand and trust their model before sharing with their VP.
      *Day 12 (04:00): 22 backend + 16 frontend = 38 new tests. Total: 1486 backend + 628 frontend = 2114.*

- [x] **Chat-triggered PDF report generation** — Business analysts can say "generate a report", "download the model report", "pdf report", or "give me a summary I can share" and receive an inline `ReportReadyCard` in the chat with a direct download link to the PDF model report. `_REPORT_PATTERNS` (9 variants) in `chat.py` detects report intent. The handler finds the selected or best completed model run, infers `problem_type` from metrics (r2→regression, accuracy→classification), and emits `{type:"report_ready"}` SSE event with `model_run_id`, `algorithm`, `metric_name`, `metric_value`, and `download_url`. `ReportReadyCard` renders: teal border, 📄 icon, "PDF Report Ready" header, problem type badge, human-readable algorithm name, metric value, description line, and a prominent "Download PDF Report" anchor button pointing to `GET /api/models/{run_id}/report`. `ReportReady` TypeScript type; `attachReportToLastMessage()` Zustand store action. No new backend endpoint needed — reuses the existing `report_generator.py` + `/api/models/{run_id}/report` from Phase 7 export. Key bugs fixed: f-string format spec `:.4f if condition` (invalid syntax, silently swallowed by `except Exception`) + `ModelRun.problem_type` attribute access (field doesn't exist; inferred from metrics instead). Closes the "share with VP" conversational use case — analysts can trigger the shareable PDF report without navigating to the Models tab.
      *Day 12 (12:00): 16 backend + 17 frontend = 33 new tests. Total: 1502 backend + 645 frontend = 2147.*

- [x] **Model performance by segment** — Business analysts can ask "how does my model perform by region?" or "model accuracy by product" and receive an inline `SegmentPerformanceCard` in chat showing per-group R² (regression) or Accuracy (classification). `compute_segment_performance()` in `core/validator.py` aligns group labels from the original CSV with model predictions, computes per-segment metrics (skipping groups with < 2 rows), assigns `strong/moderate/weak/poor/insufficient_data` status badges, and identifies best/worst segments with a plain-English gap summary. `GET /api/models/{run_id}/segment-performance?col=` endpoint (400 on unknown column, 400 on high-cardinality or near-unique columns). `_SEGMENT_PERF_PATTERNS` (7 variants) in `chat.py` detects intent; `_detect_segment_perf_col()` extracts the grouping column from the message (scans actual DataFrame columns, falls back to first low-cardinality column). `{type:"segment_performance"}` SSE event; `SegmentPerformanceCard` renders a table with ▲ best / ▼ lowest labels, per-row performance bars, and a `!` low-sample warning. Directly implements the vision's "This model is 92% accurate overall, but struggles with new product categories" transparency promise — extending segment comparison from data exploration into model validation.
      *Day 13 (04:00): 26 backend + 12 frontend = 38 new tests. Total: 1557 backend + 680 frontend = 2237.*

- [x] **Chat-driven feature engineering** — Business analysts can say "suggest features", "recommend transformations", or "feature engineering" and receive an inline `FeatureSuggestCard` in chat showing all detected transformations (date decomposition, one-hot encoding, log transforms, interaction features) with transform type badges, preview column names, descriptions, and a prominent "Apply All" button. `_FEATURE_SUGGEST_PATTERNS` (8 variants) in `chat.py` calls `suggest_features()` with the working DataFrame (respects active filter) and column stats, then emits `{type:"feature_suggestions"}` SSE event. Analysts can also say "apply features", "apply all suggestions", or "accept the feature suggestions" — `_FEATURE_APPLY_PATTERNS` (7 variants) calls `suggest_features()` then `apply_transformations()`, creates or replaces the active `FeatureSet` in the DB, and emits `{type:"features_applied"}` SSE event. `FeatureSuggestCard` has an "Apply All" button that calls `api.features.apply()` directly (no second chat message needed) and transitions to an inline success state showing new column count. `FeaturesAppliedCard` confirms the applied transformation count, new column names, and total columns. Both handlers follow the correct `_load_working_df(file_path, _active_filter_conditions)` calling convention. `FeatureSuggestionItem`, `FeatureSuggestionsChatResult`, `FeaturesAppliedResult` TypeScript types; `attachFeatureSuggestionsToLastMessage()` + `attachFeaturesAppliedToLastMessage()` Zustand store actions. Closes the last gap in the fully conversational Upload → Explore → **Shape** → Train → Deploy workflow — analysts can now complete the entire feature engineering phase without leaving the chat window.
      *Day 12 (20:00): 29 backend + 23 frontend = 52 new tests. Total: 1531 backend + 668 frontend = 2199.*

- [x] **Column profile deep-dive** — Business analysts can say "tell me about the revenue column", "profile region", "distribution of sales", or "what values are in category" and receive an inline `ColumnProfileCard` in the chat showing a complete statistical portrait of any single column. `compute_column_profile()` in `core/analyzer.py` returns: `col_type` (numeric/categorical/date), `stats` (per-type: mean/median/std/p25/p75 for numeric; most_common/top_categories for categorical; min_date/max_date/frequency for date), `distribution` (histogram bins+counts for numeric, bar labels+counts for categorical), `issues` (auto-detected: `high_null_rate` >20%, `skewed` |skewness|>2, `constant_value`, `potential_id` ≥95% unique, `high_cardinality` >50 unique, `near_unique`, `dominant_value` >90%), and a plain-English `summary`. `GET /api/data/{dataset_id}/column-profile?col=` REST endpoint (400 on unknown column, 404 on unknown dataset). `_COLUMN_PROFILE_PATTERNS` (9 variants) + `_detect_profile_col()` in `chat.py` detects intent and extracts the column name. `{type:"column_profile"}` SSE event; `attachColumnProfileToLastMessage()` Zustand store action. `ColumnProfileCard` renders: cyan border, column name + type badge, plain-English summary, stat chip grid (Rows/Unique/Missing + type-specific chips), mini distribution chart (histogram bars for numeric, horizontal category bars for categorical), issue rows with severity icons (✗ critical, ⚠ warning, ℹ info). `ColumnProfile`, `ColumnProfileIssue`, `ColumnProfileStats`, `ColumnProfileDistribution` TypeScript types; `api.data.getColumnProfile()` client method. Closes the "what's in this column?" analyst question — the first question before any modeling or cleaning decision.
      *Day 14 (20:00): 39 backend + 16 frontend = 55 new tests. Total: 1596 backend + 700 frontend = 2296.*

- [x] **K-means customer segmentation via chat** — Business analysts can say "cluster my customers", "segment my data", "find natural groups", or "customer segmentation" and receive an inline `ClusteringCard` showing natural data groupings without needing a target column. `compute_clusters()` in `core/analyzer.py` uses sklearn KMeans with auto-k selection via silhouette score (tests k=2..8 and picks the best), StandardScaler normalization, and computes per-cluster profiles: `centroid` (mean feature values), `distinguishing` features (cluster mean deviates ≥0.5σ from global mean, sorted by magnitude, with `above`/`below` direction), plain-English `description`, `size`, and `size_pct`. `GET /api/data/{id}/clusters?features=&n_clusters=` REST endpoint (400 on unknown columns or k outside 2-8). `_CLUSTER_PATTERNS` (9 NL variants) + `_detect_cluster_features()` in `chat.py` detects clustering intent; handler computes clusters inline, injects cluster summary + descriptions into the system prompt so Claude narrates the findings, and emits `{type:"clusters"}` SSE event. `ClusteringCard` renders: violet border, cluster count badge (with `auto`/`manual` indicator), summary text, feature chips (columns used), per-cluster rows with color-coded size bars, distinguishing feature badges with ↑/↓ direction arrows, plain-English description, footer with row count and k. `ClusteringResult`, `ClusterProfile`, `ClusterDistinguishingFeature` TypeScript types; `api.data.getClusters()` client method; `attachClustersToLastMessage()` Zustand store action. Fills the unsupervised ML gap — analysts can find natural customer/product/record groupings without needing a target column, answering "are there distinct types in my data?" before training.
      *Day 15 (04:00): 39 backend + 18 frontend = 57 new tests. Total: 1635 backend + 718 frontend = 2353.*

- [x] **Time-period comparison via chat** — Business analysts can say "compare 2023 vs 2024", "Q1 vs Q2 performance", "year over year", "month over month", "H1 vs H2", or "this year vs last year" and receive an inline `TimeWindowCard` showing side-by-side numeric metric means for two date ranges. `compare_time_windows()` in `core/analyzer.py` accepts two named date windows, filters rows to each period, computes per-column means with `pct_change`, `direction` (up/down/flat), and `notable` flag (≥20% change). `_build_timewindow_summary()` generates a plain-English overview naming the biggest mover. `GET /api/data/{id}/compare-time-windows?date_col=&p1_name=&p1_start=&p1_end=&p2_name=&p2_start=&p2_end=` REST endpoint (400 on unknown column, empty period, parse errors; 404 on missing dataset). `_TIMEWINDOW_PATTERNS` (8 NL trigger variants) + `_detect_timewindow_request()` in `chat.py` — handles explicit year patterns (`\b(20\d\d)\b.*\b(20\d\d)\b`), quarter patterns (`Q[1-4](?:\s+20\d\d)?`), YoY/MoM/H1-vs-H2 keywords, and falls back to bisecting the data date range. Injects period names + summary + notable changes into the system prompt for Claude narration. `{type:"time_window_comparison"}` SSE event; `attachTimeWindowToLastMessage()` Zustand store action. `TimeWindowCard` renders: orange border, period comparison header, up/down count badges, period name chips with row counts, side-by-side table (metric | P1 mean | P2 mean | Change %), amber callout listing columns with >20% change, plain-English summary. `TimeWindowPeriod` + `TimeWindowColumn` + `TimeWindowComparison` TypeScript types; `api.data.compareTimeWindows()` client method. Closes the "how did this period compare to last?" analyst question — the first question in any performance review or VP presentation.
      *Day 15 (12:00): 27 backend + 17 frontend = 44 new tests. Total: 1662 backend + 735 frontend = 2397.*

- [x] **Top-N record ranking via chat** — Business analysts can say "show me top 10 customers by revenue", "bottom 5 products", "worst-performing orders", or "rank by margin" and receive an inline `TopNCard` in the chat showing a ranked table of individual records. `compute_top_n()` in `core/analyzer.py` uses `nlargest()`/`nsmallest()`, caps at 50 rows, drops NaN in the sort column, assigns sequential `_rank` numbers (1-based), and returns a plain-English `summary`. `GET /api/data/{id}/top-n?col=&n=10&order=desc` REST endpoint (400 on unknown or non-numeric column, 404 on missing dataset; n clamped to 1–50). `_TOPN_PATTERNS` (8 NL trigger variants) + `_detect_topn_request()` in `chat.py` — extracts `n` (digit or word: five/ten/twenty), detects `ascending` from bottom/lowest/worst keywords, matches column names against actual DataFrame columns with fallback to first numeric column. `{type:"top_n"}` SSE event; `attachTopNToLastMessage()` Zustand store action. `TopNCard` renders: emerald border (top) or rose border (bottom), medal emojis (🥇🥈🥉) for rank 1-3, numeric rank for ranks 4+, zebra-striped rows with amber highlights for top 3, sort column bolded, large values formatted with k/M suffixes, summary footer. `TopNRow` + `TopNResult` TypeScript types; `api.data.getTopN()` client method. Closes the "#1 analyst reflex" — "who are my top customers?" — that previously had no dedicated chat handler.
      *Day 15 (20:00): 44 backend + 16 frontend = 60 new tests. Total: 1706 backend + 751 frontend = 2457.*

- [x] **Chat-triggered what-if prediction analysis** — Business analysts with a deployed model can ask "what if units was 20?", "what would happen if I doubled revenue?", "change region to West", or "predict with discount = 0.15" and receive an inline `WhatIfChatCard` comparing the original vs. modified prediction. `_WHATIF_CHAT_PATTERNS` (8 NL variants) + `_detect_whatif_request()` in `chat.py` — feature-name-first parsing (checks each known feature name against three pattern types: A/was-is/equals-to, B/change-to, C/equals-sign) plus a multiplier fallback for "double/triple/halve" that returns a `__multiply__N` sentinel resolved at runtime. Handler loads `PredictionPipeline.feature_means` from the joblib file as the base feature dict, calls `predict_single()` twice (base + modified), computes delta/pct_change/direction, builds a plain-English summary, and emits `{type:"whatif_result"}` SSE event. System prompt injection guides Claude to explain the change in business terms. `WhatIfChatCard` renders: amber border + 🔀 icon, problem type badge, Hypothetical Change row showing feature old→new, side-by-side Original vs Modified prediction boxes, `DeltaBadge` with ↑/↓/→ arrow + ±% change, classification probability rows when applicable, plain-English summary footer. `WhatIfChatResult` TypeScript type; `attachWhatIfChatToLastMessage()` Zustand store action. Fills the last major gap in the "chat-first deployment workflow" — analysts can run what-if scenarios without navigating to the DeploymentPanel.
      *Day 16 (04:00): 15 backend + 17 frontend = 32 new tests. Total: 1721 backend + 768 frontend = 2489.*

- [x] **Prediction error analysis via chat** — Business analysts can ask "where was my model wrong?", "show me the prediction errors", "biggest prediction errors", or "which rows did my model get wrong?" and receive an inline `PredictionErrorCard` in the chat showing the top-N worst training predictions. `compute_prediction_errors()` in `core/validator.py` is a pure function (no DB dependencies): for regression, finds rows with largest absolute residuals sorted descending, returns signed error + abs_error + rank + optional feature values; for classification, lists incorrectly predicted rows with actual/predicted class labels (decoded from `target_classes` when available). Both paths return `errors`, `total_errors`, `error_rate`, `problem_type`, and a plain-English `summary`. `GET /api/models/{run_id}/prediction-errors?n=10` REST endpoint in `api/validation.py` (uses shared `_load_run_context()` + `_build_Xy()` helpers, resolves `target_classes` from pipeline joblib). `_PRED_ERROR_PATTERNS` (14 NL variants including pluralization — no trailing `\b` per established pattern) in `chat.py` detects intent; handler loads best/selected model run, predicts on full training set, injects top errors into system prompt, emits `{type:"prediction_errors"}` SSE event. `PredictionErrorCard` renders: rose border, algorithm + problem type badges, target column, per-row table (rank, actual→predicted, signed error badge, FeatureChips for up to 4 feature values), empty state for perfect fits, summary footer. `PredictionErrorRow` + `PredictionErrorResult` TypeScript types; `pred_errors` field on `ChatMessage`; `api.models.getPredictionErrors()` client method; `attachPredictionErrorsToLastMessage()` Zustand store action. Closes the "why did my model fail?" analyst question — the first instinct after seeing a model's accuracy number.
      *Day 16 (12:00): 24 backend + 17 frontend = 41 new tests. Total: 1745 backend + 785 frontend = 2530.*

- [x] **Chat-driven record table viewer** — Business analysts can say "show me the data", "show me my data", "preview the data", "peek at the data", "show first 20 rows", "show rows where region = East" and receive an inline `RecordTableCard` in the chat with a scrollable table of actual records. `sample_records()` in `core/analyzer.py` handles optional `FilterCondition` list (reusing `apply_active_filter()` from `filter_view.py`), caps at 50 rows, paginates via `offset`, caps display columns at 8, serialises NaN→None, and returns `columns`, `rows`, `total_rows`, `filtered_rows`, `shown_rows`, `filtered`, `condition_summary`, `summary`. `GET /api/data/{id}/records?n=20&where=&offset=` REST endpoint. `_RECORDS_PATTERNS` (13 NL variants: "show me the/my data", "display/preview/peek at the records", "let me see the data", "what does the data look like", "show first N rows", "sample the data", "show/find rows/records where") in `chat.py` — no overlap with TOPN ("show me top/bottom N") or PRED_ERROR ("show errors/mistakes"). `_detect_records_request()` extracts `n` from "first 15 rows" patterns and optional `where` clause via `parse_filter_request()`. `{type:"records"}` SSE event; `attachRecordsToLastMessage()` Zustand store action. `RecordTableCard` renders: sky-blue border, "Data Preview" header with columns count badge and amber "filtered" badge, condition summary row when filtered, scrollable table with underscore-replaced column headers, em-dash for null values, string truncation at 30 chars, footer showing shown/total row counts. Fills the most fundamental analyst gap — "let me just see the data" — that was previously missing despite a full suite of analytical cards.
      *Day 16 (20:00): 22 backend + 16 frontend = 38 new tests. Total: 1767 backend + 801 frontend = 2568.*

- [x] **Scatter plot via chat** — Business analysts can say "plot revenue vs units", "scatter revenue against cost", "show me the relationship between X and Y", "how does X relate to Y", or "scatter plot" and receive an inline scatter chart in the conversation. `_SCATTER_PATTERNS` (8 NL variants) + `_detect_scatter_request()` in `chat.py` — separator-first extraction (vs/versus/against patterns up to 30-char fragments, "between X and Y", fallback to first two numeric columns mentioned); samples 500 points when df is larger; computes Pearson r for system prompt narration ("r = 0.95, positive correlation, strong"); emits `{type:"chart", chart:{chart_type:"scatter",...}}` SSE reusing existing `InteractiveScatterChart` renderer — no new frontend component. Active data filters respected via `_load_working_df`. `except Exception: pass` guard prevents scatter failures from crashing the stream. Closes the most natural exploratory visualization request — analysts who've seen group stats or correlation numbers instinctively want to see the data plotted.
      *Day 17 (04:00): 24 backend + 9 frontend = 33 new tests. Total: 1791 backend + 810 frontend = 2601.*

- [x] **Line/trend chart via chat** — Business analysts can say "plot revenue over time", "trend of sales", "line chart of units", "chart revenue by month", "how has revenue changed over time", "show me the revenue trend", or "time series of X" and receive an inline multi-series line chart (raw values + rolling average + trend line) in the conversation. `_LINE_CHART_PATTERNS` (8 NL variants) + `_detect_line_chart_request()` in `chat.py` — uses `detect_time_columns()` to auto-detect the date column; scans message for a numeric column name (longest match first), falls back to first numeric column; builds chart via `build_timeseries_chart()`; injects trend direction + % change into system prompt; emits `{type:"chart", chart:{chart_type:"line",...}}` SSE reusing existing multi-series line chart renderer — zero new frontend component. Distinct from forecasting (which predicts future) and time-window comparison (which compares two periods): this is the "just show me the trend" visualization that every analyst reaches for first.
      *Day 17 (12:00): 25 backend + 8 frontend = 33 new tests.*

- [x] **Box plot via chat** — Business analysts can say "distribution of revenue by region", "box plot of sales", "spread of units by product", "compare distribution of revenue across segments", "whisker plot", or "show outliers in revenue by region" and receive an inline box-and-whisker chart grouped by any categorical column. `_BOXPLOT_PATTERNS` (8 NL variants) + `_detect_boxplot_request()` in `chat.py` — detects value_col (numeric) and optional group_col (categorical with ≤30 unique values) using "by/across/per/for each" clause parsing; calls `build_boxplot(df, value_col, group_col)`; injects median + group info into system prompt; emits `{type:"chart", chart:{chart_type:"boxplot",...}}` SSE reusing existing `BoxPlotChart` SVG renderer — zero new frontend component. Complements group stats (which shows means) with distributional shape, IQR, and outlier visibility per group.
      *Day 17 (12:00): 14 backend + 6 frontend = 20 new tests. Total: 1830 backend + 824 frontend = 2654.*

- [x] **Pie / donut chart via chat** — Business analysts can say "pie chart of revenue by region", "donut chart of sales by product", "show me the composition of cost by segment", "share of units by category", or "proportion chart of revenue" and receive an inline pie/donut chart in the chat. `_PIE_CHART_PATTERNS` (9 NL variants: "pie chart", "donut chart", "doughnut chart", "show me a pie/donut/doughnut", "composition/proportion/share/makeup of…by", "breakdown chart") + `_detect_pie_chart_request()` in `chat.py` — finds the numeric value column (mentioned in message or first numeric) and the categorical slice column (mentioned after "by/of/for/per/across" or first categorical with 2–30 unique values); groups the DataFrame by slice column, sums the value column, passes to `build_pie_chart()`; emits `{type:"chart", chart:{chart_type:"pie",...}}` SSE reusing the existing `PieChart` Recharts renderer in `chart-message.tsx` — zero new frontend components. Active data filters respected via `_load_working_df`. System prompt injection describes the largest slice, total, and category count. `except Exception: pass` guard prevents pie chart failures from crashing the SSE stream. Closes the "what's the composition of X by Y?" analyst question — the first thing a business analyst reaches for when presenting share and breakdown data to stakeholders.
      *Day 18 (04:00): 23 backend + 8 frontend = 31 new tests. Total: 1867 backend + 832 frontend = 2699.*

- [x] **Bar chart via chat** — Business analysts can say "bar chart of revenue by region", "column chart of sales by product", "show me a bar chart", or "vertical bar chart" and receive an inline vertical bar chart in the conversation. `_BAR_CHART_PATTERNS` (8 NL variants) + `_detect_bar_chart_request()` in `chat.py` — finds value_col (numeric, longest-match first), group_col (categorical via "by/per/for each" clause parser or first mentioned categorical), and agg keyword (sum/mean/count/max/min, default sum); groups df by group_col, aggregates value_col, calls `build_bar_chart()`; emits `{type:"chart", chart:{chart_type:"bar",...}}` SSE reusing the existing BarChart renderer — zero new frontend components. Active filters respected. Distinct from `GroupStatsCard` which shows horizontal ranked bars — this produces a proper Recharts vertical bar chart.
      *Day 18 (12:00): 20 backend + 6 frontend = 26 new tests.*

- [x] **Dataset download via chat** — Business analysts can say "download my data", "export my data", "save the data as CSV", or "export the results" and receive an inline `DataExportCard` in the chat with a direct download link. `_DOWNLOAD_PATTERNS` (8 NL variants) + handler in `chat.py` emits `{type:"data_export"}` SSE event with `{dataset_id, filename, row_count, filtered, download_url}`. `GET /api/data/{id}/download` REST endpoint reads the CSV, applies the active filter if present (returning only filtered rows with a `_filtered` filename suffix), and returns a CSV `FileResponse` with `Content-Disposition: attachment`. `DataExportCard` (indigo border, ⬇ icon, "Dataset Export Ready" header, amber Filtered badge when active, filename + row count, Download CSV anchor). `DataExportResult` TypeScript type; `api.data.downloadDatasetUrl()` client helper; `attachDataExportToLastMessage()` Zustand store action; SSE handler wired in workspace page. Closes the "take my filtered analysis back to Excel/Tableau" analyst workflow entirely from chat.
      *Day 18 (12:00): 19 backend + 13 frontend = 32 new tests. Total: 1906 backend + 851 frontend = 2757.*

- [x] **Histogram via chat** — Business analysts can say "histogram of revenue", "show me a histogram", "frequency histogram of units", "binned distribution of cost", "frequency chart of revenue", or "distribution chart of units" and receive an inline histogram chart in the conversation. `_HISTOGRAM_PATTERNS` (8 NL variants) + `_detect_histogram_col()` in `chat.py` — finds the numeric column by longest-match first (underscore/space variant), falls back to first numeric column; uses `numpy.histogram()` with adaptive bin count (min 5, max 30, `len(values) // 10`); calls existing `build_histogram()` in `core/chart_builder.py`; emits `{type:"chart", chart:{chart_type:"histogram",...}}` SSE reusing the existing `"histogram"` case in `chart-message.tsx` — zero new frontend components. Active filters respected via `_load_working_df`. Distinct from `_COLUMN_PROFILE_PATTERNS` ("distribution of X" → ColumnProfileCard) and `_BOXPLOT_PATTERNS` (grouped box plots) — requires explicit "histogram" or "frequency histogram/chart" vocabulary.
      *Day 18 (20:00): 22 backend + 0 frontend = 22 new tests.*

- [x] **Missing values overview via chat** — Business analysts can say "show me the missing values", "which columns have missing data?", "null values overview", "missing data summary", "data completeness overview", "how many missing values do I have?", or "where is my missing data?" and receive an inline `NullMapCard` in the chat. `_NULL_MAP_PATTERNS` (7 NL variants) + inline handler in `chat.py` loads the working DataFrame, computes per-column null_count/null_pct/complete_pct, sorts columns most-missing first, builds `NullMapResult` dict with dataset_id, total_rows, total_columns, columns_with_nulls, fully_complete_columns, overall_completeness, columns list, and summary sentence; emits `{type:"null_map", null_map:{...}}` SSE event. `NullMapCard` (teal border, "Data Completeness" header, overall-completeness badge, per-column table with color-coded completion bars: emerald=100%, amber≥90%, rose<90%, "N missing" badges, summary footer). `NullMapResult` / `NullMapColumn` TypeScript types; `null_map?` field on `ChatMessage`; `attachNullMapToLastMessage()` Zustand store action; SSE handler + render wired in workspace page. Closes the "where are my gaps?" analyst question distinctly from data readiness (overall score) and column profile (single-column deep dive).
      *Day 18 (20:00): 24 backend + 16 frontend = 40 new tests. Total: 1952 backend + 867 frontend = 2819.*

- [x] **Summary statistics table via chat** — Business analysts can say "summarize my data", "descriptive statistics", "summary statistics", "describe all columns", "stats for all my data", or "give me the statistics for my columns" and receive an inline `SummaryStatsCard` in the chat with a pandas-style describe() table covering all numeric and categorical columns. `compute_summary_stats(df)` in `core/analyzer.py` returns: `total_rows`, `total_cols`, `numeric_stats` (count/mean/std/min/q25/median/q75/max/null_count per column), `categorical_stats` (count/unique/top/freq/null_count per column), and a plain-English `summary`. `GET /api/data/{id}/summary-stats` REST endpoint. `_SUMMARY_STATS_PATTERNS` (7 NL variants) in `chat.py` detects intent; handler calls `compute_summary_stats()`, injects summary into system prompt, emits `{type:"summary_stats"}` SSE event. `SummaryStatsCard` renders: slate border, "≡ Summary Statistics" header with rows/columns badges, Numeric Columns table (Column/Count/Mean/Std/Min/Median/Max/Nulls with k/M suffix formatting), Categorical Columns table (Column/Count/Unique/Most Common/Freq/Nulls), empty state when no stats. `SummaryStatsResult`, `NumericColumnStats`, `CategoricalColumnStats` TypeScript types; `summary_stats?` field on `ChatMessage`; `api.data.getSummaryStats()` client method; `attachSummaryStatsToLastMessage()` Zustand store action. Closes the "what does my entire dataset look like at a glance?" analyst question — the first call after loading a new dataset.

- [x] **Category value counts via chat** — Business analysts can say "most common values in region", "frequency table for product_category", "value counts for status", "how often does each region appear", "how is my data split by region", or "top occurrences in product" and receive an inline `ValueCountCard` in the chat with a ranked frequency table for any categorical column. `compute_value_counts(df, col, n=20)` in `core/analyzer.py` returns: `column`, `total_rows`, `non_null`, `null_count`, `unique_count`, `rows` (ranked list of {value, count, pct}), `has_more` (when unique_count > n), and a plain-English `summary`. `GET /api/data/{id}/value-counts?col=&n=` REST endpoint (400 on unknown column). `_VALUE_COUNT_PATTERNS` (8 NL variants) + `_detect_value_counts_col()` in `chat.py` detect intent and extract the target column (longest-match scan across column names, fallback to first categorical column); handler emits `{type:"value_counts"}` SSE event. `ValueCountCard` renders: lime border, "# Value Counts: column_name" header with unique-count badge and optional null badge, frequency table with mini progress bar (lime-500) + count + percentage per value, "Showing top N of M" truncation notice when `has_more`, summary footer. `ValueCountResult`, `ValueCountRow` TypeScript types; `value_counts?` field on `ChatMessage`; `api.data.getValueCounts()` client method; `attachValueCountsToLastMessage()` Zustand store action. Closes the "what are the actual categories in this column?" analyst question — essential before grouping, filtering, or building any categorical model.
      *Day 19 (04:00): 78 backend + 36 frontend = 114 new tests. Total: 2030 backend + 903 frontend = 2933.*

- [x] **Pair correlation analysis via chat** — Business analysts can ask "how correlated are revenue and cost?", "correlation between X and Y?", "does price correlate with demand?", or "Pearson r for units and sales?" and receive an inline `PairCorrelationCard` with Pearson r, p-value, strength/direction labels, and plain-English interpretation. `compute_pair_correlation(df, col1, col2)` in `core/analyzer.py` uses `scipy.stats.pearsonr`, classifies strength (very strong/strong/moderate/weak/negligible by |r| thresholds), assigns direction (positive/negative/no), grades significance (highly significant p<0.001, significant p<0.01, marginally significant p<0.05, not significant), and returns a one-sentence interpretation + summary. `GET /api/data/{id}/pair-correlation?col1=&col2=` REST endpoint (400 on non-numeric or missing columns). `_PAIR_CORR_PATTERNS` (7 NL variants) + `_detect_pair_corr_cols()` in `chat.py` detect intent; `{type:"pair_correlation"}` SSE event; `attachPairCorrelationToLastMessage()` Zustand store action. `PairCorrelationCard` renders: violet border, ∼ icon, col1 vs col2 header, strength/direction badges, large r value with color-coded directional bar, p-value + significance badge, interpretation paragraph, summary footer. `PairCorrelationResult` TypeScript type; `api.data.getPairCorrelation()` client method. Distinct from `_CORRELATION_TARGET_PATTERNS` (single target → ranked all-column bars) and `_HEATMAP_PATTERNS` (all-columns matrix) — this answers "exactly how correlated are these two columns?" without switching tabs.
      *Day 19 (12:00): 61 backend + 13 frontend = 74 new tests.*

- [x] **Quick stat query via chat** — Business analysts can ask "what's the average revenue?", "total sales?", "maximum cost?", "median units?", "standard deviation of price?", or "count the rows?" and receive an inline `StatQueryCard` showing a single large aggregate value. `compute_stat_query(df, agg, col)` in `core/analyzer.py` supports count/sum/mean/median/max/min/std, formats values with k/M suffixes, infers a plain-English label (average/total/median/maximum/minimum/std dev/count), and returns `n_rows`, `n_valid`, `formatted_value`, `label`, and `summary`. `GET /api/data/{id}/stat-query?agg=&col=` REST endpoint (400 on unknown agg or column). `_STAT_QUERY_PATTERNS` (7 NL variants) + `_detect_stat_query()` in `chat.py` detect intent; `_AGG_WORD_MAP` maps "average/mean/total/sum/max/min/median/std" to internal agg keys; count intent checked before the `_AGG_WORD_MAP` loop to prevent "how many rows?" matching "total" → "sum". `{type:"stat_query"}` SSE event; `attachStatQueryToLastMessage()` Zustand store action. `StatQueryCard` renders: color-coded border by agg type (cyan=mean, blue=sum, teal=median, emerald=max, orange=min, purple=std, amber=count), agg icon (x̄/Σ/m/↑/↓/σ/#), label + column header, agg badge, large formatted value, optional row-info paragraph when n_valid < n_rows, summary footer. Distinct from `_GROUP_PATTERNS` (requires "by" grouping clause). `StatQueryResult` TypeScript type; `api.data.getStatQuery()` client method.
      *Day 19 (12:00): 928 frontend tests passing, build clean. Total: 2091 backend + 928 frontend = 3019.*

- [x] **Group trend analysis via chat** — Business analysts can ask "which regions are growing?", "fastest growing products?", "which segments are trending up?", "compare growth by product category", or "how are my regions trending over time?" and receive an inline `GroupTrendCard` ranking all groups by growth rate. `compute_group_trends(df, date_col, group_col, value_col)` in `core/analyzer.py` converts dates to a numeric day-index, fits OLS slope per group (b = cov(x,y)/var(x)), computes % change (first→last), classifies direction (up/down/flat), ranks by slope descending, and builds a plain-English summary naming the fastest grower and worst decliner. Guards: rejects if group_col has >50 unique values (high cardinality). `GET /api/data/{id}/group-trends?date_col=&group_col=&value_col=` REST endpoint (400 on missing columns or high-cardinality group_col). `_GROUP_TREND_PATTERNS` (7 NL variants: "which X are growing/trending/increasing/declining/rising", "fastest growing/declining X", "growth/trend rate by/per X", "which X have the most growth/decline", "how are X trending over time", "compare growth by X") + `_detect_group_trend_request()` in `chat.py` — auto-detects date column via `detect_time_columns()`, scans message for mentioned categorical col (longest-match, fallback first cat col) and numeric col (longest-match, fallback first numeric). Handler computes `_compute_gt()`, injects rising/falling counts + summary into system prompt. `{type:"group_trends"}` SSE event; `attachGroupTrendsToLastMessage()` Zustand store action. `GroupTrendCard` renders: orange border, 📈 icon, rising/falling/flat count badges, table with rank/#/group/first/last/change badge (color-coded +green/-rose/→muted) / direction arrow (▲▼→), summary footer. `GroupTrendRow` + `GroupTrendResult` TypeScript types; `api.data.getGroupTrends()` client method. Directly implements the vision's "Which products are trending up?" question — distinct from scatter (static relationship), correlation (strength/direction), time-window comparison (two specific periods), and line chart (single series raw trend line).
      *Day 19 (20:00): 17 backend + 13 frontend = 30 new tests. Total: 2108 backend + 941 frontend = 3049.*

#### Track C — Model Building Depth

> AutoModeler's competitive differentiation is making ML accessible to business analysts.
> Right now, model training works but lacks sophistication. These improvements make models
> meaningfully better — and help analysts understand *why* one approach beats another.

- [x] **Class imbalance handling** — When target class distribution is skewed (e.g., 95% no-churn / 5% churn),
      auto-detect imbalance (minority class < 20% of total) and offer three strategies: class weighting
      (`class_weight="balanced"`), SMOTE oversampling (via `imbalanced-learn`), and threshold tuning
      (optimize decision threshold by F1 score). Show the analyst the class distribution before training,
      explain what imbalance means in plain English, and recommend a strategy. Include before/after
      comparison in the model metrics.
      *Day 22 (04:00): `detect_class_imbalance(y)` pure function in `trainer.py` (minority < 20% threshold; returns class_distribution, minority_class, minority_ratio, recommended_strategy, plain-English explanation; recommends "smote" for severe imbalance ≥100 rows with <5% minority, "class_weight" otherwise). `train_single_model()` gains optional `imbalance_strategy` param: "class_weight" injects `class_weight="balanced"` param for LogReg/RF/LGBM and uses `compute_sample_weight` in `fit()` for GBC/XGB; "smote" applies SMOTE to training split only (falls back gracefully if imblearn unavailable); "threshold" sweeps 0.05–0.95 to maximise binary F1 (`_tune_threshold()` helper, records `optimal_threshold` in metrics). `imbalance_strategy` echoed in metrics for UI display. `imbalanced-learn 0.14.1` added to `pyproject.toml`. `GET /api/models/{project_id}/imbalance` endpoint returns detection result + project_id + problem_type (returns is_imbalanced=False with explanation for regression). `TrainRequest.imbalance_strategy` optional field; `POST .../train` validates against {"class_weight","smote","threshold",null}. `ImbalanceCard` (rose border on imbalance, emerald on balanced) in ModelTrainingPanel: distribution bar (minority bars rose-colored), plain-English explanation, three clickable strategy buttons (recommended badge, selected badge, aria-pressed, toggle-on/off). `ClassImbalanceResult`/`ClassDistributionEntry` TypeScript types; `api.models.classImbalance()` client method; `imbalance_strategy` state threaded through `handleTrain()`. `model-training-panel` updated to fetch imbalance on mount for classification problems and pass strategy to train call. 28 backend + 15 frontend = 43 new tests. Total: 2264 backend + 1060 frontend = 3324, all passing. Backend lint: clean. Frontend build + lint: clean.*
      *Day 34 (04:00): **Chat integration** — `_CLASS_IMBALANCE_PATTERNS` regex (10 natural-language variants covering analyst phrasing). Handler in `send_message()`: reads `feature_set.problem_type`; for classification, calls `detect_class_imbalance(target_col_values)`, enriches result with `project_id`/`target_column`/`problem_type`, emits `{type:"class_imbalance_check"}` SSE event and injects plain-English context into system_prompt; for regression, emits N/A event. Root-cause fix: `body.project_id` → `project_id` (path parameter — `ChatMessage` Pydantic model does not expose project_id). `ClassImbalanceChatCard` component (rose/emerald/gray states, `DistributionBar` sub-component, strategy info panel, "Go to Models tab" CTA). `attachClassImbalanceCheckToLastMessage` Zustand action. `class_imbalance_check?` field on `ChatMessage` type. SSE handler + card render in `project/[id]/page.tsx`. 22 backend + 14 frontend = 36 new tests. All passing.*

- [x] **Ensemble methods** — Add `VotingClassifier` / `VotingRegressor` (soft voting across the
      best 2-3 base models from the comparison run) and `StackingRegressor` / `StackingClassifier`
      (linear meta-learner). Surface these in the algorithm selection UI as "Ensemble (combines multiple
      models — often the most accurate choice)". Include plain-English explainability for ensemble
      decisions ("3 out of 4 models voted for 'high revenue'").
      *Day 22 (20:00): `voting_regressor`, `voting_classifier`, `stacking_regressor`, `stacking_classifier` added to REGRESSION_ALGORITHMS / CLASSIFICATION_ALGORITHMS with `is_ensemble: True` flag and `base_algorithms` list (always sklearn, no optional deps). `_build_ensemble_estimators()` constructs `(name, estimator)` tuples from the registry. `train_single_model()` detects `is_ensemble` and dispatches to `_train_ensemble_model()`. Voting builds `VotingRegressor` / `VotingClassifier(voting="soft")`; stacking builds `StackingRegressor(final_estimator=Ridge)` / `StackingClassifier(final_estimator=LogisticRegression)` with `cv=min(5, n//4)`. Explainability: `_ensemble_vote_explanation()` records per-base-model mean predictions (regression) or class vote counts (classification) in `metrics.ensemble_votes`. `_stacking_weight_explanation()` reads `final_estimator_.coef_` magnitudes and normalises them to `metrics.stacking_weights`. `ensemble_summary` plain-English field: "3 out of 3 models voted for 'cat'" / "Meta-learner trusted 'random_forest_regressor' most (55% of weight)". `EnsembleVoteRow` (violet border, 🧩 icon) in `ModelTrainingPanel`: renders inline below `MetricsRow` when `metrics.ensemble_type` is set; voting shows per-model name + prediction/vote-counts; stacking shows a horizontal bar chart of weight percentages sorted descending; both show `ensemble_summary`. `EnsembleMetricsExtra` TypeScript interface added to `types.ts`. `recommend_models()` includes ensemble algos; `_why_recommended()` has dedicated voting/stacking cases. 26 backend + 19 frontend = 45 new tests. Total: 2308 backend + 1090 frontend = 3398, all passing. Backend lint: clean. Frontend build + lint: clean.*
      *Day 36 (04:00): **Chat integration** — `_ENSEMBLE_PATTERNS` regex (8 NL variants: "should I use an ensemble", "voting classifier", "stacking regressor", "combine my models", "can an ensemble improve my accuracy", etc.) in `chat.py`. Handler block in `send_message()`: reads `feature_set.problem_type` to pick regression/classification algorithms, finds best non-ensemble completed run (by R²/accuracy), decides stacking vs voting (stacking when ≥200 rows AND ≥2 completed runs), builds two-option list, emits `{type:"ensemble_recommendation"}` SSE event. "Explain before executing" principle: card shows options with plain-English descriptions and "say 'train a voting ensemble' to start" prompts — no training triggered. `EnsembleRecommendationCard` (violet border, 🧩 icon): problem-type badge, current best score + algorithm badges, "What is an ensemble model?" callout, recommendation summary (`data-testid="ensemble-summary"`), two option rows (`data-testid="ensemble-option-{voting|stacking}"`) each with Recommended/Easy/Medium badges and plain-English description. `EnsembleOption` + `EnsembleRecommendationResult` TypeScript types; `ensemble_recommendation?` on `ChatMessage`; `attachEnsembleRecommendationToLastMessage` Zustand action; SSE handler + render wired in workspace page. Integration tests bypass background-thread DB conflict by injecting `ModelRun(status="done")` directly into test session. 16 backend + 18 frontend = 34 new tests. Total: 3370 backend + 1749 frontend = 5119, all passing. Backend lint: clean. Frontend build: clean.*
      *Day 37 (12:00): **Ensemble Training via Chat** — `_ENSEMBLE_TRAIN_PATTERNS` regex (8 NL variants: "train a voting ensemble", "train a stacking ensemble", "build a voting model", "build stacking classifier", "run a voting ensemble", "run a stacking ensemble", "create a voting/stacking ensemble", "start/try voting/stacking ensemble") and `_STACKING_RE` sub-detector in `chat.py`. Handler block fires BEFORE `_TRAIN_PATTERNS` guard to prevent double-firing; mutual exclusion via `training_started_event is not None` check. Selects `voting_regressor`/`stacking_regressor` (regression) or `voting_classifier`/`stacking_classifier` (classification) based on `feature_set.problem_type` and `_STACKING_RE` match. Creates `ModelRun(status="pending")` in its own session, starts `_train_in_background` thread, sets `training_started_event` (same schema as regular training). Injects plain-English LLM context explaining that ensembles combine multiple models for higher accuracy. Old `test_monitoring_alerts.py::TestChatAnalyticsIntent` updated to use `prediction_analytics_chat` event type name (was stale `analytics`). 22 backend tests in `test_ensemble_train_chat.py`, 5 fixes in `test_monitoring_alerts.py`. Total: 3247 backend, all passing. Backend lint: clean. Frontend build: clean.*
      *Day 38 (04:00): **Local Explanation Chat Card (Feature Contribution Waterfall)** — `_EXPLAIN_ROW_PATTERNS` regex (9 NL variants: "explain prediction for row N", "explain record/index N", "explain specific prediction", "show SHAP values", "show feature contributions", "give me local explanation", "what drove/caused/influenced this prediction", "why did the model predict", "individual/local explanation", "waterfall chart") and `_extract_row_index()` helper (parses "row N", "record N", "index N", "#N" — defaults to 0) in `chat.py`. Guard: `ctx["model_runs"]` AND `ctx["dataset"]` AND `ctx["feature_set"]` AND `not pdp_event`. Handler: finds selected/best completed run; loads CSV via `pd.read_csv`; applies transformations via `apply_transformations`; builds X/y via `prepare_features` (uses `_le_feat_cols` as feature names — not the label encoder return value); clamps row index to valid range; loads joblib model; optionally resolves class labels from `_pipeline.joblib` target_classes; calls `explain_single_prediction()` from `core/explainer.py`; caps contributions at 12 for SSE payload; injects top-3 drivers + narration instruction into system prompt. Emits `{type:"local_explanation"}` SSE event. `LocalExplanationCard` (violet border `border-violet-300 bg-violet-50`, 🔍 icon): Row/Algorithm/Target/Correct-Wrong badges; side-by-side Actual vs Predicted boxes; blue (`bg-sky-400`) bars for positive contributions, rose (`bg-rose-400`) for negative; bar width proportional to abs(contribution)/maxAbs; feature/value/impact column headers; per-bar aria-labels; figcaption summary. `LocalExplanationContribution` + `LocalExplanationResult` TypeScript interfaces; `local_explanation?` on `ChatMessage`; `attachLocalExplanationToLastMessage` Zustand action; SSE handler + render wired in workspace page. Bugfix: `prepare_features` returns `(X, y, LabelEncoder|None)` — handler was incorrectly using the 3rd return value as feature names; fixed to use `_le_feat_cols` directly. 41 backend tests (8 `TestExtractRowIndex` + 7 `TestExplainSinglePrediction` + 20 `TestExplainRowPatterns` + 6 `TestLocalExplanationChatIntegration`). Backend lint: clean. Frontend build: clean.*
      *Day 37 (20:00): **Confusion Matrix Chat Card** — `_CONFUSION_MATRIX_PATTERNS` regex (8 NL variants: "show me the confusion matrix", "confusion matrix", "where does my model make mistakes", "true/false positives/negatives", "classification accuracy by class", "precision/recall/f1 per class", "model classification breakdown") in `chat.py`. Guard: classification models only (checks `algorithm in _CM_CLS_ALGOS`). Handler: loads best/selected classification run, applies feature transformations via `_load_working_df` + `apply_transformations`, builds X/y, calls `model.predict()` to get actual predictions from the trained joblib model, resolves class names from `pipeline.target_classes` if available. Enhanced `compute_confusion_matrix()` in `core/validator.py` now returns `per_class_metrics` (per-class precision/recall/f1/support computed directly from confusion matrix rows/columns — no sklearn import needed) and `most_confused_pair` (highest off-diagonal cell, the most common misclassification). Emits `{type:"confusion_matrix_chat"}` SSE event with algorithm, algorithm_plain, target_col fields. Injects overall accuracy, most_confused_pair, and plain-English summary into system prompt. `ConfusionMatrixChatCard` (border adapts: emerald ≥85%, amber ≥70%, rose <70%, 🎯 icon): algorithm + target + accuracy badges; row-count display; 2D matrix grid with "Actual" vertical label + "Predicted" horizontal header, color-coded cells (emerald diagonal = correct, darker for higher recall; rose off-diagonal = errors; transparent for zero); per-class metrics table (Class/Precision/Recall/F1/Support); rose callout for most_confused_pair ("Most common mistake: 'X' predicted as 'Y' (N times)"); figcaption with summary. `PerClassMetric` + `ConfusionMatrixChatResult` TypeScript interfaces; `confusion_matrix_chat?` on `ChatMessage`; `attachConfusionMatrixChatToLastMessage` Zustand action; SSE handler + render wired in workspace page. Closes the "where does my model make mistakes?" conversational gap — distinct from `PredictionErrorCard` (individual wrong rows) and the validation panel confusion matrix (not chat-triggered). 28 backend + 18 frontend = 46 new tests. Backend lint: clean. Frontend build: clean.*
      *Day 36 (12:00): **Hyperparameter tuning chat card** — `_EXPLICIT_TUNE_RE` module-level constant (unambiguous tuning vocabulary: tune/tuning/optimize/hyperparameter/grid-search/random-search/go ahead and tune/run the tuning/start tuning/best params) guards inline tuning from generic "improve my model" phrases (those still route to `_IMPROVEMENT_PATTERNS`). `tune_data` block replaced by `tune_chat_event` block in `send_message()`: when `_EXPLICIT_TUNE_RE` matches AND `ctx["feature_set"]` AND `ctx["dataset"]` exist, loads CSV via `_load_working_df`, prepares X/y, creates `ModelRun(status="training")`, calls `tune_model()` (10-iter RandomizedSearchCV, 3-fold CV), updates run to `done`, emits `{type:"tune_chat"}` SSE event with before/after metrics, best_params, improved flag, improvement_pct. Non-tunable algorithms: emits `{tunable:False}` with tree-based suggestion. Tunable but missing feature_set/dataset: guidance only (no card). `TuningChatCard` (green/emerald border when improved, amber when unchanged, slate for not-tunable, 🔧 icon): before/after metrics table with colored delta column, best params display (`key = value` monospace), Improved/Unchanged badge, improvement_pct badge (± %). `TuningChatResult` TypeScript interface; `tune_chat?` on `ChatMessage`; `attachTuneChatToLastMessage` Zustand action; SSE handler + render wired in workspace page. 20 backend + 21 frontend = 41 new tests. Backend lint: clean. Frontend build: clean.*
      *Day 36 (20:00): **Cross-Validation Score Distribution chat card** — `_CV_SCORE_DIST_PATTERNS` regex (8 NL variants: "how consistent is my model", "cross-validation scores", "show me fold scores", "cv variance", "model stability check", "is my model stable", "high variance in my cv", "fold-by-fold performance") in `chat.py`. Handler block in `send_message()`: selects the selected/best completed run; loads CSV via `_load_working_df`; applies feature transformations; builds X/y via `prepare_features`; gets unfitted model from the algorithm registry; calls `run_cross_validation()` (5-fold, existing validator function); computes coefficient of variation (std/mean); classifies consistency as `stable` (CoV<5%), `moderate` (5-15%), or `variable` (>15%); emits `{type:"cv_score_distribution", cv_score_distribution:{...}}` SSE event with algorithm, metric_plain (R²/Weighted F1), scores list, mean, std, CI, n_splits, consistency, consistency_pct, summary. Injects plain-English stability context into system prompt. `CvScoreDistributionCard` (border adapts: emerald for stable, amber for moderate, rose for variable, 📊 icon): header with algorithm/problem_type/consistency badges; 3-column stats grid (mean/std/CoV); per-fold bars (colored emerald≥0.8/sky≥0.6/amber≥0.4/rose<0.4); 95% CI line; summary paragraph; figcaption explaining what high/low variance means for the analyst. `CvScoreDistributionResult` TypeScript interface; `cv_score_distribution?` on `ChatMessage`; `attachCvScoreDistributionToLastMessage` Zustand action; SSE handler + render wired in workspace page. 13 backend + 14 frontend = 27 new tests. Backend lint: clean. Frontend build: clean.*

- [x] **Date-aware train/test split** — When a date column is present, default to a chronological
      split (oldest 80% = train, newest 20% = test) rather than random shuffle. Explain why: "We used
      time-based splitting — training on past data and testing on more recent data gives a more honest
      picture of how your model will perform in the real world." Chat should detect "use time-based
      split" intent and toggle the strategy.
      *Day 22 (12:00): `chronological_split(n_rows, test_size=0.2) -> (train_idx, test_idx)` pure function in `trainer.py` — returns sequential index arrays (oldest 80% = train, newest 20% = test). `train_single_model()` gains `split_strategy: str = "random"` + `date_col_used: str | None = None` params; when `split_strategy == "chronological"` uses `chronological_split()` index arrays instead of `train_test_split(shuffle=True)`; records `split_strategy`, `date_col_used`, and `split_explanation` in metrics. `_train_in_background()` in `api/models.py` gains `split_strategy` param; if `"chronological"` calls `detect_time_columns(df)` to find the first date column, sorts df ascending by that column before `prepare_features()`, then passes `split_strategy="chronological"` and `date_col_used` to `train_single_model()` (falls back to random if no date col found). `TrainRequest` gains `split_strategy: str | None = None` (validated: random/chronological/null). `GET /api/models/{project_id}/split-strategy` endpoint returns `{recommended, date_col, explanation}` — detects time columns on first 50 rows. `_TIME_SPLIT_PATTERNS` (11 NL variants) in `chat.py`: detects "use time-based split", "chronological split", "train on older data", "time series split", "random split" etc.; emits `{type:"split_strategy"}` SSE event with `{split_strategy, date_col, explanation}` and injects guidance into system prompt. `SplitStrategyCard` in chat (sky-blue border for chronological, slate for random): header + badge (Time-based/Random), date column code label, explanation text, 80%/20% train/test legend bars for chronological. `ModelTrainingPanel`: auto-calls `splitStrategy()` on mount, auto-selects "chronological" if recommended; renders split strategy toggle (Random/Time-based buttons with aria-pressed) above algorithm selection; passes `splitStrategy` to `api.models.train()`. Split strategy shown inline in run metrics row (🗓️ Time-based split badge with date_col). 18 backend + 11 frontend = 29 new tests. Total: 2282 backend + 1071 frontend = 3353, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Feature selection automation** — After training, identify features with near-zero importance
      (bottom 20% by SHAP or coefficient magnitude) and offer to retrain without them. "I found 3
      features that aren't helping the model — removing them may reduce noise and improve predictions
      on new data." Show the user a before/after metric comparison. This closes a real analyst question:
      "Are all my columns actually useful?"
      *Day 23 (04:52): `identify_weak_features(model, feature_cols, threshold_percentile=20.0)` in `core/trainer.py` — extracts importances via `feature_importances_` (tree-based) or `|coef_|` (linear), normalises to sum=1, finds bottom-20th-percentile threshold, flags weak features. Returns `{feature_importances: [...], weak_features: [...], threshold, method, has_importances, n_weak, explanation}`. Returns `has_importances=False` for MLP/ensemble. `GET /api/models/{run_id}/feature-selection` loads the trained model from disk, reads feature columns from the active feature set, and returns the ranked list. `TrainRequest` gains `excluded_features: list[str] | None` — filtered from feature_cols before passing to `_train_in_background()`; raises HTTP 400 if all features excluded. `_FEATURE_SEL_PATTERNS` (8 NL variants) in `chat.py`: detects "are all columns useful?", "feature selection", "remove weak features", "which features should I remove" etc.; finds the most recently completed run, loads the model, calls `identify_weak_features`, emits `{type:"feature_selection"}` SSE event. `FeatureSelectionCard` (amber border, 🎯 icon) serves both as a chat card (read-only) and panel card (interactive): per-feature importance bars (amber for normal, rose for weak, ↓ weak label), rank numbers, % values, method note, n_weak badge; in panel mode adds checkboxes per feature, "Exclude N weak features on retrain" button (populates `excluded_features` for next train call), Clear button. `ModelTrainingPanel` auto-loads feature selection after training completes (best done run) and shows panel below version history. `FeatureSelectionResult` + `FeatureImportanceRow` TypeScript types; `api.models.featureSelection(runId)` client method; `attachFeatureSelectionToLastMessage` Zustand action; `feature_selection` wired in page.tsx SSE handler. 21 backend + 21 frontend = 42 new tests. Total: 2329 backend + 1111 frontend = 3440, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Training on large datasets** — Current training loads the full CSV into memory. Add a
      chunked-training path for datasets >50k rows: random sample 20k rows for training, train on
      that, report "trained on 20,000 rows (random sample — full dataset is too large for in-memory
      training)". This prevents OOM errors and keeps the experience fast.
      *Day 23 (04:00): `sample_large_dataset(df, max_rows=20_000, threshold=50_000, random_state=42)` pure function in `trainer.py` — returns (sampled_df, sample_info) with `was_sampled`, `original_rows`, `sample_rows`, and analyst-friendly `note`. Called in `_train_in_background()` before `prepare_features()`; when sampling occurs, `sample_size`, `original_dataset_size`, and `sample_note` are added to run metrics and emitted in the SSE `done` event. Reproducible (fixed seed). 8 pure-function tests cover: not-sampled at threshold, sampled above threshold, reproducibility, custom threshold, empty note. Total: 2357 backend + 1122 frontend = 3479 tests, all passing. Lint clean.*

- [x] **Calibration for classifiers** — Apply `CalibratedClassifierCV` to classifiers so that
      `predict_proba` outputs are well-calibrated (a predicted 80% confidence should be right ~80%
      of the time). Show a reliability diagram in the validation panel. Plain-English: "This chart
      shows how trustworthy the model's confidence scores are — a well-calibrated model's bars
      follow the diagonal line."
      *Day 23 (04:00): `CalibratedClassifierCV(model_class(**params), cv=3, method="sigmoid")` wraps all classifiers in `train_single_model()` except when threshold tuning, SMOTE, `_SAMPLE_WEIGHT_FIT_ALGOS` with class_weight, or <30 training rows (all cases where calibration is inappropriate or technically infeasible). The calibrated model is what gets saved to disk, so all deployment predictions benefit. `_add_calibration_metrics()` helper computes binary calibration curve (`sklearn.calibration.calibration_curve`, 10 bins, uniform strategy), Brier score, and a plain-English `calibration_note` (well-calibrated < 0.05 max deviation, reasonably calibrated < 0.15). These go into metrics as `calibration_curve`, `brier_score`, `calibration_note`, `is_calibrated`. `identify_weak_features()` updated to unwrap `CalibratedClassifierCV` via `.calibrated_classifiers_[0].estimator` before extracting feature importances. `GET /api/models/{run_id}/calibration` endpoint reads pre-computed data from DB (returns 400 when not available). `CalibrationData`/`CalibrationPoint` TypeScript types; `api.models.calibration()` client method. `ReliabilityDiagramView` component in ValidationPanel: BarChart of predicted vs actual frequency, red dashed diagonal reference line (perfect calibration), Brier score badge (color-coded green/amber/red), plain-English note, SR-accessible figure/figcaption. New "Calibration" sub-tab in ValidationPanel; shows "not available" callout for regression/threshold/SMOTE models. 20 backend + 11 frontend = 31 new tests.*

- [x] **Partial Dependence Plots (PDP) via chat** — Business analysts can ask "partial dependence for price", "marginal effect of units on revenue", "PDP for region", "average effect of units on the prediction", "population-level effect of discount", or "partial dependence plot for quantity" and receive a `PartialDependenceCard` with a line chart showing how the model output changes as one feature varies -- **averaged over the actual training distribution** (not just at fixed means, like sensitivity analysis). `compute_partial_dependence(model, X_train, feature_idx, grid_values, problem_type, class_names)` pure function in `core/explainer.py`: sweeps the feature through a grid of values (p5-p95 of training data in 20 steps), replaces the feature column in a copy of the full training set, averages model predictions across all training rows. Returns `{grid_values, mean_predictions, std_predictions, class_curves, n_training_rows, summary}`. `GET /api/models/{run_id}/partial-dependence?feature=&steps=20` endpoint in `api/validation.py` (400 on unknown feature, 404 on missing run; steps clamped 5-50). `_PDP_PATTERNS` (8 NL variants) + `_detect_pdp_feature()` in `chat.py` detect intent; handler picks the best completed run, computes PDP, injects summary into system prompt, emits `{type:"partial_dependence"}` SSE event. `PartialDependenceCard` (purple border, 📉 icon): Regression/Classification badge, algorithm badge, trend-direction badge (up/down/flat), italic "averaged over N training records" explainer, Recharts LineChart with purple mean line + violet dashed +/-1 std band + axis labels + tooltip; per-class colour legend for multiclass; constant-feature fallback message. `PartialDependenceResult` TypeScript type; `partial_dependence?` field on `ChatMessage`; `attachPartialDependenceToLastMessage` Zustand action; SSE handler and render wired in workspace page. Closes the "how does feature X affect predictions on AVERAGE across all customers?" analyst question -- more statistically rigorous than sensitivity analysis (which fixes other features at training means), directly implementing the vision's "Not a black box" promise.
      *Day 31 (20:00): 29 backend + 15 frontend = 44 new tests. Total: 2961 backend + 1527 frontend = 4488, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Calibration Check via chat** — Business analysts can ask "how well-calibrated is my model?", "are my confidence scores reliable?", "show the reliability diagram", "brier score", or "calibration check for the predictions" and receive a `CalibrationCheckCard` inline in conversation. The calibration data (Brier score, reliability curve) was already computed at training time and stored in model metrics — this feature wires it into the chat pipeline. `_CALIBRATION_CHECK_PATTERNS` regex (8 NL variants) in `chat.py`; handler loads the selected/best completed model run's metrics, checks for `is_calibrated` flag, extracts `brier_score` / `calibration_curve` / `calibration_note`, applies a quality bucket (excellent < 0.1, good < 0.2, poor ≥ 0.2), builds a plain-English `summary`, and emits `{type:"calibration_check"}` SSE event. System prompt injected with summary + narration instruction to explain what calibration means to the analyst and whether confidence scores shown on the VP prediction dashboard can be trusted. `CalibrationCheckCard` (violet border, 🎯 icon): quality badge (emerald=Excellent, blue=Good, red=Needs attention), algorithm badge, Brier score row with "(lower is better; 0 = perfect, 0.25 = random)" hint, summary paragraph, Recharts BarChart reliability diagram (blue bars = actual frequency per confidence bucket, dashed reference diagonal = perfect calibration), calibration note footnote; empty-curve fallback. `CalibrationCheckResult` TypeScript interface; `calibration_check?` on `ChatMessage`; `attachCalibrationCheckToLastMessage` Zustand action; SSE handler + render wired in workspace `page.tsx`.
      *Day 32 (04:00): 13 backend + 15 frontend = 28 new tests. Total: 2974 backend + 1542 frontend = 4516, all passing. Backend lint: clean. Frontend build + lint: clean.*

#### Track D — Deployment Depth

> **This is the highest-priority track.** The vision's "one click and you have a live API"
> promise is implemented but thin. Real deployment for business analysts requires the features below.
> These are what no-code AutoML tools consistently get wrong — and where AutoModeler can win.

- [x] **API key authentication for prediction endpoints** — Right now, any person with the
      prediction URL can call the model. Add optional API key protection: generate a key at deploy
      time, store a salted hash, require `Authorization: Bearer <key>` header. The deployment panel
      shows the key once (copy-to-clipboard), with a Regenerate button. Plain-English: "Your model
      is now protected — only people with this key can use it." This is table-stakes for any analyst
      who wants to share a model with their dev team.
      *Day 20 (04:00): `api_key_hash` + `api_key_salt` + `api_key_enabled` added to Deployment model with inline SQLite migration. `POST /api/deploy/{id}/api-key` generates `secrets.token_urlsafe(32)`, stores `sha256(salt:key)`, returns key once. `DELETE /api/deploy/{id}/api-key` removes protection. `_verify_api_key()` helper enforces `Authorization: Bearer` on predict/batch/explain endpoints using `secrets.compare_digest`. `ApiKeyCard` in DeploymentPanel: amber border, Protected/Open-access badge, Generate/Regenerate/Remove protection buttons, copy-to-clipboard for the generated key. 14 backend + 8 frontend = 22 new tests.*

- [x] **Scheduled batch prediction jobs** — Let analysts set up a recurring prediction run:
      "Run batch predictions on my sales_forecast.csv every Monday at 9am." Store schedules in a
      `BatchSchedule` SQLModel table (cron expression, dataset_id, deployment_id, last_run,
      next_run, output_path). Use APScheduler (already available via FastAPI's background tasks).
      Email/webhook notification on completion (configurable). Frontend: "Schedule" tab in the
      deployment panel with a simple form (frequency: daily/weekly/monthly + time picker).
      *Day 20 (12:00): `BatchSchedule` + `BatchJobRun` SQLModel tables (auto-created by `create_all`). Background daemon thread wakes every 60s, finds due schedules, runs batch predictions against deployment's training dataset, saves results to `data/batch_outputs/<sid>_<ts>.csv`. `compute_next_run()` computes UTC next-fire for daily/weekly/monthly frequencies. 5 endpoints: POST/GET/DELETE schedules + POST run (immediate trigger) + GET run history + GET batch-outputs download (path-traversal guarded). `ScheduleCard` in DeploymentPanel: frequency/time/day form, schedule list with next_run/last_run/last_row_count, Run Now / History / Remove per-schedule, paginated run history with download links. 19 backend + 13 frontend = 32 new tests.*

- [x] **Deployment versioning and rollback** — When a model is retrained and redeployed, the
      old version should be preserved and accessible. Maintain a `DeploymentVersion` table tracking
      each version (model_run_id, version_number, deployed_at, metrics snapshot). The deployment
      panel shows a "Version history" timeline with one-click rollback: "Restore v2 (R² 0.84)".
      Old prediction logs remain associated with the version that made them.
      *Day 20 (20:00): `DeploymentVersion` SQLModel table (auto-created). `execute_deployment()` archives current version on re-deploy, keeping endpoint URL stable. `GET /api/deploy/{id}/versions` + `POST /api/deploy/{id}/rollback/{version}` endpoints. `DeploymentVersionCard` shows indigo-bordered timeline (newest-first); Restore button with two-click confirmation (click-to-arm, click-to-confirm prevents accidental rollback); current version has "Current" badge; card only shows when 2+ versions exist. 11 backend + 13 frontend = 24 new tests.*

- [x] **Champion-challenger A/B testing** — Allow splitting live prediction traffic between two
      deployed model versions (e.g., 80% to champion, 20% to challenger). Record which version
      served each prediction in PredictionLog. The deployment panel shows a side-by-side comparison
      of champion vs challenger accuracy, confidence, and request counts in real time. Auto-promote
      challenger to champion when it achieves statistical significance (Mann-Whitney U or bootstrap).
      *Day 21 (20:00): `ABTest` SQLModel table (auto-created by `create_all`). `ab_variant` field added to `PredictionLog` (inline SQLite migration). `make_prediction()` checks for active `ABTest` where `champion_id == deployment_id`; uses `random.random()` vs `champion_split_pct/100` to route each request; logs `ab_variant="champion"` or `"challenger"` (deployment_id always set to champion so endpoint analytics are stable). Four REST endpoints: `POST /api/deploy/{id}/ab-test` (create, deactivates existing), `GET /api/deploy/{id}/ab-test` (status + per-variant metrics), `DELETE /api/deploy/{id}/ab-test` (end, no promotion), `POST /api/deploy/{id}/ab-test/promote` (copies challenger's model into champion deployment keeping endpoint URL stable, archives current model as new DeploymentVersion, records `winner="challenger"`). `_ab_variant_metrics()` computes request_count/avg_confidence/p95_ms/avg_prediction per variant. `_ab_significance()` runs Mann-Whitney U via `scipy.stats.mannwhitneyu` (α=0.05); returns "Need N more samples" note when < 5 samples per variant. `ABTestCard` (purple border, ⚗️ icon, "Live" badge) in DeploymentPanel: idle state with description + Start A/B Test button; create form with challenger ID input + split % slider (50–99, default 80) + Cancel; active test view with champion/challenger split bar (purple/amber), per-variant metrics boxes (requests/avg confidence/p95 latency/avg prediction), significance badge (green=significant, gray=not yet), Promote Challenger (two-click arm-then-confirm) + End Test + Refresh buttons. API methods: `api.deploy.getAbTest()`, `createAbTest()`, `endAbTest()`, `promoteChallenger()`. `ABTest`, `ABVariantMetrics`, `ABSignificance` TypeScript types. 27 backend + 19 frontend = 46 new tests. Total: 2227 backend + 1036 frontend = 3263.*
      *Day 33 (12:00): Chat integration wired. `_AB_TEST_PATTERNS` (8 NL variants: "how is my A/B test going?", "check A/B test", "show A/B test status", "is the challenger doing better?", "A/B test results", "promote the challenger", "end the A/B test", "stop the split test") + `_AB_PROMOTE_RE` + `_AB_END_RE` in `chat.py`. Handler detects status/promote/end action; status path calls `_ab_test_response()` from `api.deploy` and returns full metrics; promote path replicates `promote_challenger()` logic inline (archives champion, copies challenger attrs, records DeploymentVersion, ends test with winner="challenger"); end path sets `is_active=False`; no-test path returns action="none" with onboarding guidance. SSE emits `{type:"ab_test_result", ab_test_result:{action, summary, ...ABTest fields when status}}`. `ABTestChatCard` (purple border, ⚗️ icon) in chat: status view — "Live" badge + champion/challenger split bar (purple/amber) + 2-column MetricsColumn (requests/avg confidence/p95/avg prediction) + SignificanceRow + guidance footer; promoted view — "Promoted ✓" badge + URL-unchanged note; ended view — "Ended" badge + champion-remains note; none view — guidance to train second model. `ABTestChatResult` TypeScript type (action + summary + optional ABTest fields); `ab_test_result?` on `ChatMessage`; `attachABTestResultToLastMessage` Zustand action; SSE handler and render wired in `page.tsx`. Note: deployment system uses one-deployment-per-project for URL stability — A/B tests require deployments from two separate projects as champion and challenger. 16 backend + 19 frontend = 35 new tests. Total: 3051 backend + 1611 frontend = 4662, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Webhook notifications** — Let users register a webhook URL to be called when model health
      degrades, drift is detected, or a scheduled batch job completes. `POST /api/deploy/{id}/webhook`
      stores the URL + event types. Dispatch via `httpx.post()` with a signed payload (HMAC-SHA256).
      This connects AutoModeler into existing analyst workflows (Slack, Teams, Zapier integrations).
      *Day 21 (04:00): `WebhookConfig` SQLModel table (auto-created by `create_all`). `core/webhook.py` provides `dispatch_webhooks(deployment_id, event_type, payload)` — queries active webhooks, fires matching ones in daemon threads, signs each payload with HMAC-SHA256 (`X-AutoModeler-Signature` header). Three event types: `batch_complete` (fired in scheduler._run_job), `drift_detected` (fired when drift_score >= 50), `health_degraded` (fired when health_score < 60). Four REST endpoints: `POST /api/deploy/{id}/webhooks` (register, returns secret once), `GET /api/deploy/{id}/webhooks` (list, no secret), `DELETE /api/deploy/{id}/webhooks/{wid}` (soft-delete), `POST .../test` (synchronous test dispatch, returns HTTP status). `WebhookCard` (sky-blue border, "🔔 Webhook Notifications") in DeploymentPanel: signed-header explanation, webhook list with event-type badges / Test / Remove per entry, test result inline (OK/Failed), last-fired timestamp, add-webhook form with URL input + event-type checkboxes + Save/Cancel. Secret shown once after creation in amber callout with Copy button. 18 backend + 13 frontend = 31 new tests.*

- [x] **Webhook event history via chat** — Analysts can ask "what webhooks fired recently?",
      "show webhook history", or "did any webhooks fire?" and receive a `WebhookHistoryCard` inline
      in chat showing a per-event timeline: event type badge (batch_complete/drift_detected/etc.),
      webhook URL, timestamp, and HTTP status badge (200 OK / Error). Closes the gap between
      webhooks firing silently and analysts having visibility into their integration health.
      *Day 33 (20:00): `WebhookEvent` SQLModel table (`id`, `webhook_id`, `deployment_id`, `event_type`, `fired_at`, `status_code`) persists each dispatch attempt; `_dispatch_in_thread()` in `core/webhook.py` writes a row after each HTTP call. `GET /api/deploy/{id}/webhook-history` returns `{total, events, summary}` — events joined with `WebhookConfig` for URL lookup; summary is plain-English. `_WEBHOOK_HISTORY_PATTERNS` (8 NL variants) + handler block in `chat.py` (guarded by `ctx["deployment"]`); SSE `{type:"webhook_history"}`. `WebhookHistoryCard` (slate border, 🔔 icon): event count badge, summary, column header, per-event rows with color-coded event type badges, URL, timestamp, status badge. Zustand `attachWebhookHistoryToLastMessage`; SSE wired in `page.tsx`. 18 backend + 15 frontend = 33 new tests. Total: 3069 backend + 1626 frontend = 4695, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Deployment environment promotion (staging → production)** — Add an environment concept:
      each deployment is tagged `staging` or `production`. A "Promote to production" button in the
      deployment panel swaps the active endpoint, preserving the staging URL for testing. The predict
      page shows an environment badge. Plain-English: "You're looking at the staging version — your
      team is using the production version at a different URL."
      *Day 22 (04:50): `environment` field added to `Deployment` model (default `"staging"`) with inline SQLite migration. `POST /api/deploy/{id}/promote-to-production` marks this deployment as production and demotes any existing production deployment for the same project back to staging (preserving its URL for testing). `POST /api/deploy/{id}/demote-to-staging` reverses the promotion. Both endpoints are idempotent. `_deployment_response` includes `environment` in every response. `EnvironmentCard` in DeploymentPanel: amber border + "Staging" badge with "Promote to Production" button (two-click confirmation — first click shows amber confirmation dialog, second click executes); green border + "Production" badge with "Demote to staging" ghost button. Environment badge also shown inline in the "Model deployed" status row (amber=Staging, green=Production). `predict/[id]` page shows Staging/Production badge in the dashboard header so the analyst knows which version their VP is seeing. `api.deploy.promoteToProduction()` + `api.deploy.demoteToStaging()` client methods; `EnvironmentPromotionResult` TypeScript type; `environment?` optional field on `Deployment` interface (optional for backwards compat with existing tests). 9 backend + 9 frontend = 18 new tests. Total: 2236 backend + 1045 frontend = 3281, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Export as self-contained prediction service** — "Download as ZIP" exports the model
      pipeline, a minimal FastAPI server (`server.py`), `requirements.txt`, and a `README.md`
      with one-command deployment instructions (`uvicorn server:app`). This lets developers take the
      model and run it on their own infrastructure with no dependency on AutoModeler. Closes the
      vision's "An API their developer can plug into the company's reporting tool" promise.
      *Day 21 (05:04): `GET /api/deploy/{id}/export` returns a ZIP with server.py (FastAPI predict + health + root endpoints, CORS middleware, joblib model/pipeline loading), model_pipeline.joblib, model.joblib, requirements.txt, README.md. server.py is syntactically valid Python, embeds target_column, algorithm, and example payload from training data. `ExportServiceCard` (emerald border, "📦 Export as Service") in DeploymentPanel: lists 5 included files, shows uvicorn quick-start snippet, Download as ZIP button triggers blob download with correct filename. `api.deploy.exportServiceUrl()` client helper. 18 backend + 18 frontend = 36 new tests.*

- [x] **Prediction SLA monitoring** — Track p50/p95/p99 prediction latency per deployment.
      Alert (inline in deployment panel) when p95 > 500ms. Show a latency sparkline in the
      analytics card next to the prediction count sparkline. This gives developers confidence
      before wiring the API into production systems.
      *Day 21 (12:00): `response_ms` added to `PredictionLog` (Optional[float], inline SQLite migration) and populated via `time.monotonic()` around `predict_single()` in `make_prediction()`. `GET /api/deploy/{id}/sla` endpoint returns `p50_ms`/`p95_ms`/`p99_ms`/`avg_ms`/`sample_count`/`alert`/`alert_message`/`latency_by_day` — `_percentile()` helper uses linear interpolation on sorted list. `alert=True` when `p95 > 500ms`; `alert_message` names the threshold and suggests remediation. `latency_by_day` groups by day and averages ms for the sparkbar. `SlaData` TypeScript type; `api.deploy.sla()` client method. `SlaMonitorCard` (sky-blue border when healthy, red border on alert) in `DeploymentPanel`: p50/p95/p99 grid, Healthy/`p95 > 500ms` badge, `LatencySparkbar` (bars colored red when > 500ms), avg ms + sample count, red alert message callout. Logs with NULL `response_ms` (legacy rows) excluded from sample_count. 12 backend + 11 frontend = 23 new tests.*

- [x] **Per-deployment rate limiting and monthly quotas** — Allow operators to cap how
      many requests a deployed prediction endpoint can serve: per-minute RPM via a sliding
      window (in-memory deque per deployment), and a rolling 30-day prediction count from
      `PredictionLog`. `PUT /api/deploy/{id}/rate-limit` sets/removes both limits (0 = remove,
      null = remove). `GET /api/deploy/{id}/quota-status` returns used/remaining/pct_used.
      `POST /api/predict/{id}` raises HTTP 429 when either limit is exceeded. Chat understands
      "set rate limit to 100 requests per minute", "add a monthly quota of 500 predictions",
      "check my quota", "disable rate limit" — emits a `rate_limit` SSE event rendered as a
      `RateLimitCard` with an amber-bordered card, per-minute limit, quota usage fraction,
      color-coded progress bar (green/amber/red at 70%/90%), and percentage used.
      *Day 31 (04:00): `rate_limit_rpm` + `monthly_quota` fields on `Deployment` (inline SQLite migration). `_check_rate_limit()` sliding-window helper + `_check_monthly_quota()` rolling-count helper in `api/deploy.py`. `_RATE_LIMIT_PATTERNS` + 4 extraction regexes in `api/chat.py`; handler applies set/disable/status intent without crashing chat. `RateLimitCard` frontend component with `UsageBar` sub-component. Zustand `attachRateLimitToLastMessage` action. 26 backend + 17 frontend = 43 new tests; total 2915 backend + 1495 frontend = 4410.*

- [x] **Prediction input guard rails** — When a user supplies a feature value that is outside
      the model's training-data range (numeric) or is an unseen category, the prediction response
      now includes a `guard_rail_warnings` list describing exactly what was out of bounds and why
      confidence may be lower. `feature_ranges` field added to `PredictionPipeline` (backward-compatible
      default; stored at build time): numeric features store `{p5, p95, min, max}`; categorical
      features store `{known_categories: [...]}`. `validate_prediction_inputs(provided_features,
      pipeline)` pure function in `core/deployer.py` checks only user-supplied values (not defaults).
      `predict_single()` accepts optional `provided_features` kwarg and calls the validator when
      supplied; `guard_rail_warnings` omitted from the result when empty. `make_prediction()` in
      `api/deploy.py` passes `provided_features=input_data` (the user's request body). Chat inline
      prediction handler passes `_ip_extracted` (pre-default features) as `provided_features` and
      injects a warning summary into the Claude system prompt addendum. `GuardRailWarning` TypeScript
      interface added to `types.ts`; `InlinePredictionResult.guard_rail_warnings?` + `PredictionResult.
      guard_rail_warnings?` fields added. `InlinePredictionCard` shows amber-bordered warning rows when
      warnings are present: severity label (Out of range / Extreme outlier / Unknown category), message,
      typical range for numeric or known values for categorical, `role="alert"` per row, aria-label on
      the warnings section. Card border shifts from blue to amber when warnings are present. Public
      `predict/[id]/page.tsx` result section renders a warning callout block. Directly implements the
      vision's "Not a black box" and "Fail gracefully — always suggest next steps" principles: analysts
      sharing a VP dashboard now get an honest "heads up, that revenue figure is 50× the training max —
      this prediction is extrapolating, not interpolating."
      *Day 31 (12:00): 17 backend + 17 frontend = 34 new tests; total 2932 backend + 1512 frontend = 4444. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Quota Alert Notifications** — When monthly quota usage crosses a configured
      percentage threshold, registered webhooks with the `quota_alert` event type are dispatched
      exactly once — the moment the threshold is first crossed. Analysts configure via chat:
      "alert me when I hit 80% of my quota", "set quota alert at 90%", "disable quota alert".
      `quota_alert_threshold_pct` field added to `Deployment` model (inline SQLite migration).
      `EVENT_QUOTA_ALERT` added to `core/webhook.py` `ALL_EVENTS`. `_check_and_fire_quota_alert()`
      pure helper in `api/deploy.py` — fires only when `used == ceil(quota * threshold / 100)`,
      preventing repeated alerts on every subsequent prediction. Called in a daemon thread after
      each successful prediction commit. `PUT /api/deploy/{id}/quota-alert` endpoint (validates
      1-99 range; 0 or null removes; 422 for negative or >99). `GET /api/deploy/{id}/quota-status`
      now includes `quota_alert_threshold_pct` and `quota_alert_enabled`. `_QUOTA_ALERT_PATTERNS`
      (8 NL variants) in `chat.py`; handler sets/reads threshold, emits `{type:"quota_alert_config"}`
      SSE event. `QuotaAlertCard` (orange border, 🔔 icon): enabled/disabled badge, threshold
      explanation, current usage fraction + color-coded progress bar (green/amber/red), help text.
      `QuotaAlertConfig` TypeScript interface; `attachQuotaAlertConfigToLastMessage` Zustand action;
      SSE handler + render wired in workspace page. Closes the gap where analysts setting a monthly
      quota had no early warning before their VP's dashboard started returning 429 errors.
      *Day 32 (20:00): 21 backend + 16 frontend = 37 new tests. Total: 3010 backend + 1577 frontend = 4587. Backend lint: clean. Frontend build + lint: clean.*

- [x] **SLA Latency Monitoring via chat** — Analysts can ask "how fast is my model?",
      "show me the prediction latency", "p95 latency", "response time stats", or "is my API
      within SLA?" and receive an `SlaCard` showing p50/p95/p99 percentiles, average latency,
      sample count, a trend sparkline (Recharts LineChart over `latency_by_day`), and an alert
      badge + `role="alert"` message when p95 exceeds the 500ms target. `_SLA_PATTERNS` (10 NL
      variants) in `chat.py` detects intent. Handler queries `PredictionLog.response_ms` values
      for the active deployment, computes percentiles via the existing `_percentile()` helper,
      groups by day for the sparkline, and emits `{type:"sla_metrics"}` SSE event with the full
      `SlaData` payload. Claude's system prompt is injected with the latency summary and SLA
      status so it narrates in plain English. `SlaCard` (sky border, ⚡ icon) is a new chat-
      context component (distinct from the deployment-panel `SlaMonitorCard`): no-data empty
      state, percentile grid (p50/p95/p99), avg/sample-count row, sparkline when ≥2 days of
      data, red alert with `role="alert"` paragraph when p95 > 500ms, SLA target footnote.
      `attachSlaMetricsToLastMessage` Zustand action; SSE handler and render wired in `page.tsx`.
      *Day 32 (12:00): 15 backend + 19 frontend = 34 new tests; total 2989 backend + 1561 frontend = 4550. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Batch Prediction Scheduling via chat** — Analysts can say "schedule daily batch
      predictions at 9am", "run my model every Monday at 8am", "batch predictions every
      month", or "show my batch schedules" to create and list `BatchSchedule` records
      directly from the chat interface. `_SCHEDULE_PATTERNS` (7 NL arms) detects create/list
      intents. `_extract_schedule_params()` parses frequency (daily/weekly/monthly), run time
      (12h + 24h), and day-of-week names. `_build_schedule_description()` returns plain-English
      confirmations ("Every Monday at 08:00 UTC"). Handler guards on active deployment; list
      path queries all schedules; create path calls `compute_next_run()` and commits a new
      `BatchSchedule`. SSE emits `{type:"schedule_set"}` event. `ScheduleSetChatCard` (teal
      border, 🗓️): "created" view shows FrequencyBadge + description + next-run + help footer;
      "list" view shows count badge + per-schedule rows with frequency/description/next-run/
      last-row-count. `attachScheduleSetToLastMessage` Zustand action; SSE handler + render
      wired in workspace page.
      *Day 33 (04:00): 25 backend + 15 frontend = 40 new tests. Total: 3035 backend + 1592 frontend = 4627. Backend lint: clean. Frontend build + lint: clean.*

#### Track E — End-to-End Polish

> The "lunch break" success criterion: a business analyst uploads quarterly sales data and
> in 30 minutes has a shareable dashboard for their VP. Run this flow end-to-end and fix
> every friction point you find. Tests don't catch UX debt — only running it does.

- [x] **"Lunch break" flow audit** — Using `scripts/demo.py` as a starting point, run the
      complete analyst journey manually: upload → ask 2-3 questions → approve features → train
      → validate → deploy → share the prediction link. Document every moment of confusion,
      missing affordance, or required domain knowledge. Fix the top 3 friction points found.
      Journal findings honestly.
      *Day 23 (20:00): Code audit across the full flow. Top friction: (1) generic "Prediction Dashboard" title on VP-shared page; (2) no model trust context (algorithm, accuracy, date) on predict page; (3) cryptic column names and missing range hints in the form; (4) raw algorithm IDs in compare-model table; (5) session history showed prediction only, not inputs. All 5 fixed in predict/[id]/page.tsx. Feature schema extended with mean+std. See "shareable prediction page UX" item.*

- [x] **Proactive insights after upload** — The system calls `narrate_data_insights_ai()` after
      upload but the analyst still has to know to ask questions. Expand proactive suggestions:
      after upload, the AI should offer 3-5 specific, data-aware questions in suggestion chips
      (not generic "ask me anything" prompts). E.g., "I see a `date` column and a `revenue` column
      — want me to show you the revenue trend over time?" These should be generated from the
      actual profile, not hardcoded templates.
      *Day 23 (12:00): `generate_upload_suggestions(profile, col_names)` in `orchestrator.py` — generates 3-5 data-aware chips from actual column types, correlations, and missing-value profile. Upload and sample-load endpoints return `suggestions` list in response body. Frontend sets `chatSuggestions` from response, rendering chip buttons with "Try asking:" label. 9 backend unit tests + 3 API integration tests + 3 frontend tests.*

- [x] **"What can I do next?" guidance at every step** — At the end of each major action
      (upload complete, features applied, model trained, model deployed), the AI should proactively
      say what the logical next step is and offer it as a clickable suggestion. This replaces the
      current state where the analyst has to remember the workflow. Tie this to the conversation
      state machine stages.
      *Day 23 (12:00): `get_next_step_chips(state)` in `orchestrator.py` — returns 3 action-focused chips for each workflow stage (explore/shape/validate/deploy). Training stream `all_done` event includes `next_step_chips`. Chat SSE emits `{type:"next_step"}` after `deployed` and `features_applied` events. `ModelTrainingPanel.onTrainingComplete(chips)` callback bubbles chips to page. 6 backend unit tests + 1 backend training-stream test + 3 frontend tests.*

- [x] **The shareable prediction page UX** — The `predict/[id]` page is what the analyst
      shares with their VP. Run it with a fresh eye: Is it immediately obvious what to do?
      Does it explain what the model is predicting? Does the form field order match the way
      analysts think about their data? Does it look polished enough to show a VP? Fix whatever
      needs fixing.
      *Day 23 (20:00): 5 targeted UX fixes: (1) Page title now "{Target Column} Predictor" (e.g., "Revenue Predictor") — tells VP immediately what the model does. (2) `ModelContextCard` shows algorithm in plain English (algoName() maps raw IDs), accuracy in plain language ("Explains 84% of variation"), and deployment date. (3) Form heading "Your Scenario" with "pre-filled with training averages" sub-label; numeric labels show "(avg: X)" hint using new mean field from feature schema. (4) Algorithm names in compare-model table use algoName() mapping. (5) Session history shows "Key Inputs" column (first 3 feature values) so VP can see what scenario produced each prediction. Backend: get_feature_schema() extended with mean+std fields. 2 backend + 6 frontend = 8 new tests. Total: 2378 backend + 1134 frontend = 3512, all passing. Backend lint: clean. Frontend build + lint: clean.*

#### Track F — Coordination

- [x] **Update BACKLOG.md** — Before starting work, check BACKLOG.md for what the other
      bot instance is working on or has recently explored. Write your chosen focus at the
      top before implementing. After the session, move completed items to the "Done" section
      and add any new ideas you discovered.
      *Day 3 (08:04): BACKLOG updated at session start and end each session from Day 2 onward.*

- [x] **Executive Briefing Generator** — Analysts can say "write a briefing for my VP" or
      "create an executive summary" and receive a polished VP-ready `ExecutiveBriefingCard`
      directly in chat. Closes the "share results with leadership" gap: the analyst has a
      prediction dashboard URL, but previously had no structured way to communicate model value
      to a non-technical audience.
      *Day 35 (04:00): `generate_executive_briefing()` pure function in `core/storyteller.py` —
      assembles plain-English metric explanations (`_metric_explanation()` formats R², accuracy,
      RMSE with quality tiers and plain-English meaning), algorithm descriptions
      (`_algo_description()` one-sentence business explanations), 4-section briefing structure
      (What We Analyzed, How Accurate Is It?, What This Means, Deployment Status), one-sentence
      headline `summary`, and `action_items` list. `GET /api/projects/{id}/executive-briefing`
      REST endpoint gathers all project context (dataset, features, model runs, deployment,
      prediction count) and returns the full structured result. `_BRIEFING_PATTERNS` regex
      (8 natural-language variants: "write a briefing for my VP", "create an executive summary",
      "explain this to my executive team", "talking points for my VP meeting", etc.) + handler
      block in `chat.py` that emits `{type:"executive_briefing"}` SSE event and injects
      plain-English context into system_prompt. `ExecutiveBriefingCard` (emerald border, 📋 icon):
      algorithm badge, metric label badge (color-coded by quality tier), italic one-line summary,
      4 sections with uppercase headings, Recommended Actions list with → bullets, footer with
      prediction dashboard link OR deploy-prompt, copy-to-clipboard button
      (`aria-label="Copy briefing to clipboard"`). `ExecutiveBriefingResult` / `BriefingSection`
      TypeScript types; `api.projects.executiveBriefing()` client method;
      `attachExecutiveBriefingToLastMessage` Zustand action; SSE handler + card render in
      `project/[id]/page.tsx`. 22 backend + 16 frontend = 38 new tests. All passing. Backend lint:
      clean. Frontend build + lint: clean.*

- [x] **Service export via chat** — Analysts can say "package my model", "export my model as a
      service", "download the prediction service", "deploy this elsewhere", or "give my model to a
      developer" and receive a `ServiceExportChatCard` inline in chat with a direct ZIP download
      link — no navigation to the deployment panel required. Closes the last gap in the "chat-first
      deployment workflow": the developer hand-off story is now completable entirely through
      conversation.
      *Day 35 (12:00): `_SERVICE_EXPORT_PATTERNS` regex (8 NL variants: package/bundle/export/zip,
      standalone/self-contained service, deploy elsewhere, give model to developer) added to
      `chat.py`. Handler guards on `ctx["deployment"]`; extracts `algorithm`, `target_column`,
      `problem_type`, `feature_count` from the active deployment record (falls back to model run),
      and emits `{type:"service_export", service_export:{deployment_id, algorithm, target_column,
      problem_type, feature_count, download_url, included_files}}` SSE event. `ServiceExportChatCard`
      (indigo border, 📦 icon): "Model Package Ready" heading, ZIP-download badge, problem-type
      badge, formatted algorithm description, included-files list (`data-testid="included-files"`;
      per-file plain-English annotations), quickstart code block (`pip install -r requirements.txt`
      + `uvicorn server:app --host 0.0.0.0 --port 8000`), feature count, `<a download>` link with
      `aria-label`. `ServiceExportChatResult` TypeScript type; `service_export?` on `ChatMessage`;
      `attachServiceExportToLastMessage` Zustand action; SSE handler + card render wired in
      `project/[id]/page.tsx`. 13 backend + 18 frontend = 31 new tests. Total: ~3142 backend +
      1693 frontend. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Deployment Version Comparison via Chat** — Analysts can ask "compare my deployment
      versions", "did my retrain improve?", "current version vs previous", "how did my retrain
      improve", or "is the new version better?" and receive a `DeploymentVersionComparisonCard`
      inline in chat showing per-metric deltas between the current and previous deployment version.
      Handler guards on `ctx["deployment"]` and 2+ `DeploymentVersion` records. Computes
      delta/pct_change/direction/improved for r2, accuracy, mae, rmse, f1, precision, recall.
      MAE/RMSE treated as error metrics (lower=better). Algorithm-change detection with amber note.
      Plain-English summary. When <2 versions: `has_comparison=False` with onboarding guidance.
      No-comparison state shows summary only (no table). `_VERSION_COMPARE_PATTERNS` regex (8 NL
      variants). `DeploymentVersionComparisonResult`/`VersionMetricDiff` TypeScript types;
      `attachVersionComparisonToLastMessage` Zustand action; SSE handler + card render wired in
      `project/[id]/page.tsx`. 13 backend + 19 frontend = 32 new tests. Total: 3155 backend +
      1712 frontend = 4867, all passing. Backend lint: clean. Frontend build + lint: clean.*

- [x] **Learning Curve Analysis via Chat** — Analysts can ask "would more data help?", "would
      adding data improve my model?", "show me the learning curve", "do I need more data?",
      "is my training set big enough?", "did my model converge?", or "training size analysis" and
      receive a `LearningCurveCard` with a Recharts line chart showing training score vs validation
      score at increasing dataset sizes, with a plain-English convergence verdict. `compute_learning_curve(X,
      y, algorithm, problem_type, n_sizes=5, cv_folds=3)` pure function in `core/trainer.py` uses
      `sklearn.model_selection.learning_curve` — sweeps min_fraction…1.0 in n_sizes steps, computes
      mean train/val scores per fold, detects convergence when the val score improvement over the last
      two steps is < 1% of the full-data score. Returns `sizes_pct`, `train_scores`, `val_scores`,
      `converged`, `plateau_pct`, `best_val_score`, `metric_label`, `metric_key`, `n_total`,
      `algorithm_name`, `recommendation`, `summary`. `GET /api/models/{project_id}/learning-curve`
      REST endpoint. `_LEARNING_CURVE_PATTERNS` (8 NL variants) + handler in `chat.py` that selects
      the best/selected run, loads the working DataFrame via `_load_working_df` (active filters
      respected), prepares X/y, calls `compute_learning_curve()`, and emits
      `{type:"learning_curve"}` SSE event. `LearningCurveCard` (indigo border, 📈 icon): Converged
      /Still Learning badge, row count + algorithm name, summary text, Recharts dual-series LineChart
      (solid training line, solid validation line, X-axis = % of training data, Y-axis = metric),
      best val score box, plateau convergence box (when converged), recommendation callout.
      `LearningCurveResult` TypeScript type; `learning_curve?` on `ChatMessage`;
      `attachLearningCurveToLastMessage` Zustand action; SSE handler + render wired in
      `project/[id]/page.tsx`. Bug fixed Day 37: handler was using `pd.read_csv` directly instead
      of `_load_working_df` — active filters were not respected. Closes the "do I need to collect
      more data before retraining?" analyst question — distinct from CV score distribution (which
      shows consistency across folds at full size) and from the training progress view.
      *Day 37 (04:00): 25 backend + 17 frontend = 42 tests. Backend lint: clean. Frontend build: clean.*

- [x] **Developer SDK Generation via Chat** — When a model is deployed, analysts can say "generate
      a Python SDK", "create a JavaScript SDK for my model", "developer SDK", "how can my developers
      use my API", or "make it easy for developers" and receive a `SdkDownloadCard` inline in chat
      with download links for a typed Python class and JavaScript module wrapping the prediction
      endpoint. `GET /api/deploy/{id}/sdk?language=python|javascript` endpoint generates a self-
      contained SDK file: Python produces a class with `__init__(base_url)` + `predict(**features)`
      that POSTs to the prediction endpoint and returns typed results; JavaScript produces an ES
      module `class` with the same interface using `fetch`. Class name derived from target column
      (e.g. `revenue_predictor` → `RevenuePredictor`). `_SDK_PATTERNS` (8 NL variants) in `chat.py`
      guards on `ctx["deployment"]`; computes class name, builds Python/JS URL params, emits
      `{type:"sdk_download"}` SSE event. `SdkDownloadCard` (indigo border, 📦 icon): algorithm +
      problem-type info, Python and JavaScript download buttons, usage code snippet showing
      `from revenue_predictor_sdk import RevenuePredictor` + `predictor.predict(...)`.
      `SdkDownloadInfo` TypeScript type; `sdk_download?` on `ChatMessage`;
      `attachSdkDownloadToLastMessage` Zustand action; SSE handler + render wired in
      `project/[id]/page.tsx`. Closes the "how does my developer consume this model?" gap —
      the analyst can hand a developer a pre-built client library instead of raw API docs.
      *Day 37 (04:00): 27 backend + 16 frontend = 43 tests. Backend lint: clean. Frontend build: clean.*

- [x] **Cross-Project Portfolio Overview via Chat** — Analysts managing multiple projects can ask
      "show all my models", "portfolio overview", "compare all my projects", "which project is doing
      best?", "cross-project view", or "all my work" and receive a `PortfolioCard` summarising every
      project in one place. `compute_portfolio_summary(project_summaries)` pure function in
      `core/analyzer.py` aggregates: `total_projects`, `active_deployments`, `total_predictions`,
      `best_performer` (project with highest primary metric), `avg_metric`, `projects` (per-project
      rows with name, dataset, model count, best algorithm, best metric value, deployment status,
      prediction count), `summary` (plain-English overview sentence). `GET /api/projects/portfolio`
      REST endpoint queries all projects and their associated datasets, model runs, and deployments.
      `_PORTFOLIO_PATTERNS` (8 NL variants) + handler in `chat.py` that iterates all projects for
      the database session, assembles per-project summary dicts, and emits `{type:"portfolio"}` SSE
      event. `PortfolioCard` (purple border): total-projects/active-deployments/total-predictions
      stat chips, best-performer highlight row (algorithm + metric), per-project table (name,
      dataset, models trained, best score, deployed status, predictions served), plain-English
      summary footer. `PortfolioResult` TypeScript type; `portfolio?` on `ChatMessage`;
      `attachPortfolioToLastMessage` Zustand action; SSE handler + render wired in
      `project/[id]/page.tsx`. Closes the "how are all my models doing?" question for analysts
      running multiple prediction projects simultaneously.
      *Day 37 (04:00): 21 backend + 16 frontend = 37 tests. Backend lint: clean. Frontend build: clean.*

- [x] **Prediction Log Analytics via Chat** — Track D perpetual. Analysts can ask "how many
      predictions have been made?", "show prediction analytics", "prediction volume this week",
      "usage stats for my model", or "how often is my model being called?" and receive a
      `PredictionAnalyticsChatCard` inline in chat. Upgrades the existing thin stub (only
      `total_predictions`) into a full analytics card. Handler queries `PredictionLog` for the last
      30 days, computes: `total_predictions` (all-time from `Deployment.request_count`),
      `predictions_last_7_days`, `predictions_last_30_days`, `predictions_today`, `predictions_by_day`
      (14-day daily bar counts), `peak_day` (date + count of highest-volume day), `class_counts`
      (classification only — dict of predicted_class→count for last 30 days), `avg_prediction`
      (regression only — mean of `prediction_numeric`). Bug fixed: was using `model_runs.problem_type`
      which doesn't exist — fixed to `deployment.problem_type`. Emits `{type:"prediction_analytics_chat"}`
      SSE event. `PredictionAnalyticsChatCard` (sky-blue border, 📊 icon): total badge, problem-type
      badge, summary paragraph, 3-stat grid (7-day / 30-day / today), 14-day `BarChart` sparkline
      with peak day highlighted in darker blue (no chart shown when all zero), peak-day info row,
      class distribution bars with % widths (classification), avg prediction box (regression).
      `PredictionAnalyticsChatResult` TypeScript type; `prediction_analytics_chat?` on `ChatMessage`;
      `attachPredictionAnalyticsChatToLastMessage` Zustand action; SSE handler + render wired in
      `project/[id]/page.tsx`.
      *Day 37 (12:00): 16 backend + 17 frontend = 33 new tests. Backend lint: clean. Frontend build: clean.*

- [x] **Production Input Feature Distribution via Chat** — Track D perpetual. Analysts can ask
      "what values are users sending to my model?", "show production input distribution", "are my
      production inputs in range?", or "how different are production inputs from training?" and receive
      a `ProductionInputDistributionCard` inline in chat. Handler queries the last 500 `PredictionLog`
      records for the active deployment, parses `input_features` JSON, aggregates per-feature stats
      (capped at 10 features): numeric features show production mean/min/max vs training range
      (from `PredictionPipeline.feature_ranges`), with out-of-range count and percentage; categorical
      features show top-5 value distribution bars with percentage widths, plus unseen-category detection
      for values not in training `known_categories`. Guards on `ctx["deployment"]`. Emits
      `{type:"prod_input_dist"}` SSE event with `{deployment_id, sample_count, features[], summary}`.
      `ProductionInputDistributionCard` (sky-blue border, 📊 icon): sample-count/feature-count badges,
      "All inputs in range" (emerald) or "N out-of-range values" (amber) badge; per-feature rows
      colored amber (numeric OOR) or rose (unseen categorical); numeric rows show min/avg/max grid
      and training range footnote; categorical rows show horizontal percentage bars + unseen warning;
      empty state when no predictions yet; figcaption legend. `ProductionInputDistributionResult`/
      `ProdInputFeature`/`ProdInputNumericFeature`/`ProdInputCategoricalFeature` TypeScript types;
      `prod_input_dist?` on `ChatMessage`; `attachProdInputDistToLastMessage` Zustand action; SSE
      handler + render wired in `project/[id]/page.tsx`. Closes the "are users sending weird values
      to my model?" analyst question — surfaces production covariate shift before it causes silent
      accuracy degradation.
      *Day 38 (12:00): 21 backend + 15 frontend = 36 new tests. Backend lint: clean. Frontend build: clean.*

- [x] **Proactive Covariate Drift Alert** — Complements `ProductionInputDistributionCard` with a
      proactive, severity-driven alert when production inputs diverge from training distributions.
      `compute_covariate_drift_alert(all_inputs, feature_ranges)` pure function in `core/analyzer.py`:
      iterates up to 10 features, classifies each as numeric (checks out-of-range %) or categorical
      (checks unseen category %), applies medium threshold (≥15%) and high threshold (≥30%), returns
      `{has_alerts, severity, severity_label, sample_count, feature_count, alert_count, alerts, summary}`.
      `GET /api/deploy/{id}/covariate-drift` endpoint loads PredictionLogs + pipeline feature_ranges,
      calls pure function, returns result with `deployment_id`. `_COVARIATE_DRIFT_PATTERNS` regex in
      `chat.py` detects 17+ natural-language variants ("covariate drift", "input drift", "drift alert",
      etc.); handler guarded by `ctx["deployment"]` emits `{type:"covariate_drift_alert"}` SSE event.
      `CovariateDriftAlertCard` (🌊 icon, severity-colored border): severity badge, sample/feature count
      badges, per-feature `AlertRow` with HIGH/MED badge, OOR% or unseen% readout, description, guidance
      footer. `CovariateDriftAlertResult`/`CovariateDriftFeatureAlert` TypeScript types; Zustand
      `attachCovariateDriftAlertToLastMessage`; SSE handler + render + proactive welcome-back injection
      in `project/[id]/page.tsx` (auto-checks active deployment on returning visit, injects medium/high
      alerts into welcome-back message). Closes the "are my production inputs quietly drifting?" question.
      *Day 38 (20:00): 41 backend + 24 frontend = 65 new tests. Backend lint: clean. Frontend build: clean.*

- [x] **Quota Runway Analysis via Chat** — Track D perpetual. Analysts can ask "will my quota last
      the month?", "quota runway", "quota forecast", "quota projection", "quota exhaustion", "at this
      rate when will I run out?", "prediction budget analysis", or "when will my monthly quota run out?"
      and receive a `QuotaRunwayCard` inline in chat. `_QUOTA_RUNWAY_PATTERNS` regex (8 NL variants)
      guards the handler; handler queries `PredictionLog` for (1) current calendar-month count (2) last
      7-day average daily rate, then computes: remaining quota, days until exhaustion at current rate,
      projected month total, and `will_exhaust` flag (projected > quota). `QuotaRunwayCard` (📊 icon,
      rose/amber/emerald border coding by risk): progress bar (aria-valuenow, role="progressbar") with
      percentage label; used/total, remaining, days-left-in-month grid; at-risk alert (role="alert")
      showing rate, days_left_at_rate, projected total vs. limit; safe-state message with green ✓;
      unlimited state shows usage stats reassuringly; rate limit RPM footnote with hourly capacity;
      figcaption sr-only caption. `QuotaRunwayResult` TypeScript type; `quota_runway?` on `ChatMessage`;
      `attachQuotaRunwayToLastMessage` Zustand action; SSE handler + render wired in `project/[id]/page.tsx`.
      CI fix in same session: `_apply_migrations()` in `db.py` backfills missing `auto_retrain` column
      (added to Project model previously but not migrated), resolving CI failures on real-DB tests.
      *Day 39: 21 backend + 20 frontend = 41 new tests. Backend lint: clean. Frontend build: clean.*

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
- Coverage: >85% (both stacks already exceed this — do not chase 100%)
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
