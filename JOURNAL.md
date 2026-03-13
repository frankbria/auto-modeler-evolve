# Journal

## Day 1 — 20:00 — Phase 6 Complete: Deployment

Implemented all five Phase 6 features in one session. The model packaging was essentially already done — `trainer.py` has always serialized a full sklearn Pipeline + metadata into a single `.joblib` artifact; Phase 6 just needed to make it addressable. New `models/deployment.py` adds a Deployment table (model_run_id, is_active, request_count, feature_schema JSON, last_predicted_at). New `core/deployer.py` handles three concerns: `load_artifact` (loads the joblib), `get_feature_schema` (reconstructs column dtype info by introspecting the ColumnTransformer's fitted transformers — numeric vs categorical — to drive the prediction form), `predict_single` and `predict_batch`. New `api/deploy.py` splits across three FastAPI routers: `/api/deploy` for lifecycle (POST to deploy, DELETE to undeploy), `/api/deployments` for listing and details, `/api/predict` for inference (single JSON input and batch CSV upload → CSV download via StreamingResponse). Frontend adds a sixth "Deploy" tab to the project workspace with a `DeployPanel` component showing deployment status, a shareable dashboard link, request count, and batch prediction upload. New `/predict/[id]` Next.js page is the public-facing prediction dashboard: it fetches the deployment's feature schema, renders a dynamic input form (numeric → number input, categorical → text input), and shows the prediction + confidence on submit. 144 backend tests pass; Next.js build clean. All six phases from spec.md are now complete — next session begins Phase 7 (Polish & Delight: onboarding, project management, chat memory, export).

## Day 1 — 16:00 — Phase 5 Complete: Validation & Explainability

Implemented all five Phase 5 features in one session. New `core/validator.py` provides honest cross-validation (via `sklearn.base.clone()` for unfitted CV, preventing data leakage) with 95% confidence intervals computed from fold score standard deviations; classification models get a confusion matrix from an 80/20 split with per-class accuracy, regression models get actual-vs-predicted residual data for scatter visualization. New `core/explainer.py` uses sklearn's `permutation_importance` for global feature importance (model-agnostic, no SHAP dependency needed — the Pipeline wrapper made TreeExplainer fragile) and a perturbation-based individual explanation: each feature is replaced with its column median/mode and the prediction change is measured as the "impact." New `api/validation.py` exposes three endpoints: `/metrics`, `/explain`, and `/explain/{row_index}`. One TypeScript bug fixed: Recharts Tooltip `formatter` prop expects `ValueType | undefined` not `number` — replaced typed parameters with runtime `Number(v)` coercions. Frontend adds a fifth "Validate" tab with three sub-sections (CV Results, Feature Impact, Explain Row): CV metrics shown as cards with fold-score sparkbars and CI range, confusion matrix rendered as a color-coded grid (green diagonal = correct, red off-diagonal = mistakes), residuals as a scatter chart with a perfect-prediction reference line, feature importance as a horizontal bar chart, and per-row explanations as an impact bar chart with signed values. 130 backend tests pass; Next.js build clean. Next session: Phase 6 — deployment (model packaging, prediction API, dashboard).

## Day 1 — 12:00 — Phase 4 Complete: Model Training

Implemented all five Phase 4 features in one session. New `core/trainer.py` wraps sklearn estimators in Pipelines with ColumnTransformer preprocessing (median impute + optional StandardScaler for numeric; constant impute + OrdinalEncoder for categorical) — this design prevents train/test leakage and produces a single serializable joblib artifact for Phase 6 deployment. `recommend_models` selects 2-4 algorithms based on dataset size and problem type (small datasets skip GradientBoosting to avoid overfitting risk). `train_model` runs cross-validation to produce reliable metrics (accuracy/F1/precision/recall for classification; R²/MAE/RMSE for regression) then fits a final pipeline on all data for storage. One bug caught and fixed: the root `.gitignore` had a bare `models/` rule that blocked both `src/backend/models/` (Python DB models) and `src/frontend/components/models/`; replaced with targeted `data/models/` ignores and explicit negations for the source directories. `compare_models` produces ranked results with a plain-English summary. New `api/models.py` exposes recommend, train, runs, compare, and select endpoints. Frontend adds a fourth "Models" tab: enter a target column → see algorithm recommendations → checkbox-select algorithms → train → compare table with per-row Select buttons. 97 backend tests pass; Next.js build clean. Next session: Phase 5 — validation & explainability (SHAP, confusion matrix, confidence).

## Day 1 — 08:00 — Phase 3 Complete: Feature Engineering

Implemented all five Phase 3 features in one session. New `core/feature_engine.py` generates feature transformation suggestions purely from statistical analysis (no LLM needed): date-like string columns → date_decompose; right-skewed numerics (skewness > 1.5) → log_transform; low-cardinality categoricals (≤15) → one_hot; medium-cardinality (≤50) → label_encode; continuous floats with many values → bin_quartile; correlated numeric pairs (r ≥ 0.5) → interaction terms. `apply_transformations` returns a new DataFrame without mutating the input, plus a column mapping. `detect_problem_type` correctly handles float→regression, int with low cardinality→classification. `compute_feature_importance` uses sklearn mutual information, which handles mixed types. One bug fixed: the initial implementation classified float columns with few rows as classification (unique ≤ 10 threshold); fixed by separating float (always regression) from integer (cardinality check). Frontend extended with a 3-tab right panel (Data / Features / Importance), `FeatureSuggestionsPanel` with checkbox-select-and-apply UI, and `FeatureImportancePanel` with bar chart visualization. 71 backend tests pass; Next.js build clean. Next session: Phase 4 — model training.

## Day 1 — 12:04 — (auto-generated)

Session commits: no commits made.


## Day 1 — 08:09 — (auto-generated)

Session commits: no commits made.


## Day 1 — 04:00 — Phase 2 Complete: Analysis & Exploration

Implemented all five Phase 2 features: enhanced `core/analyzer.py` with full profiling (IQR-based outlier detection, histogram bins, categorical value distributions, correlation matrix, and plain-English pattern insights); new `core/chart_builder.py` generating Recharts-compatible JSON configs for bar, line, histogram, scatter, and pie charts; new `core/query_engine.py` using Claude to parse natural-language questions into structured QuerySpec dicts (safe, no code eval) and execute them against pandas DataFrames. Added `/api/data/{id}/profile` and `/api/data/{id}/query` endpoints; updated the chat SSE stream to emit optional `chart` events after the text stream. Frontend updated with a `ChartMessage` component and chart events handled inline in the message bubble, plus an Insights panel in the data view that surfaces warnings on upload. One snag: newer pandas returns dtype "str" not "object" for string columns — fixed the date-column heuristic to check both. All 40 backend tests pass; Next.js TypeScript build clean. Next session: Phase 3 — feature suggestions and approval workflow.

## Day 1 — 00:00 — Phase 1 Complete: Full Stack Bootstrap

Implemented the entire Phase 1 foundation in one session: FastAPI backend (Python/uv/SQLModel/SQLite) with project CRUD, CSV upload with pandas profiling, data preview, and Claude-powered streaming chat via SSE. Frontend bootstrapped with Next.js 15, shadcn/ui, Zustand, react-dropzone — split-panel workspace (chat left, data right) with drag-and-drop CSV upload, column stats grid, and real-time streamed responses. One snag: pytest-bdd doesn't natively await async step functions, solved by switching BDD steps to FastAPI's synchronous TestClient. All 13 backend tests pass; Next.js build compiles cleanly with no TypeScript errors. Next session: Phase 2 — auto-profiling, natural language data queries, and chart generation.

## Day 0 — 21:51 — (auto-generated)

Session commits: no commits made.


<!-- New entries go at the top, below this heading -->
