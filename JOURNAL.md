# Journal

## Day 1 — 16:20 — Phase 4 Complete: Model Training

Discovered and fixed a critical gap: the `models/` Python package (Project, Dataset, FeatureSet, Conversation) was never created — all 71 previously-reported tests were actually failing with `ModuleNotFoundError`. Fixed by creating `src/backend/models/` with five SQLModel tables and a `.gitignore` bug where `models/` was inadvertently ignoring both backend and frontend component directories (changed to `src/backend/data/models/`). Built Phase 4 on top: `core/trainer.py` with `recommend_models` (heuristic algorithm suggestions by dataset size), `prepare_features` (label-encodes categoricals, fills NAs), and `train_single_model` (Linear/RandomForest/GradientBoosting for both regression and classification, with train/test split, full metrics, joblib persistence, and plain-English summaries); `api/models.py` with five endpoints (recommendations, train, runs/status poll, compare, select), using daemon background threads — one bug found: the background thread imported `engine` at load time via `from db import engine`, so tests' monkeypatched engine wasn't picked up; fixed by using `import db as _db` and referencing `_db.engine` dynamically. Frontend adds a "Models" tab: `ModelTrainingPanel` with algorithm cards (select, show plain-English description), training progress polling (1.5s interval), per-run metrics cards (R²/MAE/RMSE or Accuracy/F1/Precision), and model selection. 97 backend tests pass; Next.js build clean. Next session: Phase 5 — validation & explainability (cross-validation, confusion matrix, SHAP).

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
