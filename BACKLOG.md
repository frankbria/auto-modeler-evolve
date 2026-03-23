# Evolution Backlog

Living document for coordinating between bot instances and tracking ideation.
**Read this before starting work. Write your focus before implementing.**

## Currently Working On

<!-- Each bot writes here BEFORE starting implementation. Format: -->
<!-- ## [Bot ID / Timestamp] — [Focus Area] -->
<!-- Brief description of what you're doing this session. -->
<!-- Remove your entry when you commit your session wrap-up. -->

## Day 12 (04:00) — "Explain My Model" conversational model card
Natural language model explanation — `GET /api/models/{project_id}/model-card` synthesizes algorithm + metrics + feature importance + limitations into a structured plain-English model card. `_MODEL_CARD_PATTERNS` chat intent detects "explain my model", "what does my model do", "how does my model work", "model summary", etc. → emits `{type:"model_card"}` SSE event. `ModelCardView` component renders inline in chat: algorithm chip, accuracy badge with plain-English context ("predicts correctly 9/10 times"), top feature drivers as horizontal bars, key limitation, action buttons (dashboard link + copy API URL). Closes the "not a black box" vision promise for analysts who want to *understand* their model before sharing it with their VP.

## Day 11 (20:00) — Done
Chat-driven deployment — execute_deployment() helper extracted from deploy_model route; _DEPLOY_CHAT_PATTERNS (9 variants) in chat.py; handler selects is_selected run or falls back to best-by-metric; emits {type:"deployed"} SSE event; DeployedCard (green live dot, algorithm/target/metric, dashboard link, copy-endpoint button); DeployedResult type; attachDeployedToLastMessage store action; no-model case gracefully guides user to train first. 17 backend + 18 frontend = 35 new tests. Total: 1464 backend + 612 frontend = 2076.

## Day 11 (12:00) — Done
Non-destructive data filter — DatasetFilter SQLModel table (one-per-dataset); core/filter_view.py (parse_filter_request, apply_active_filter, build_filter_summary, validate_filter_conditions); _load_working_df() helper replaces all 13 pd.read_csv() calls in chat.py so every analysis respects active filter; POST/DELETE/GET /api/data/{id}/set-filter|clear-filter|active-filter; _FILTER_PATTERNS + _CLEAR_FILTER_PATTERNS chat intents → {type:"filter_set"} + {type:"filter_cleared"} SSE events; FilterSetCard (conditions with operator symbols, row-reduction stats in chat); FilterBadge (Data tab header, ✕ clear button); FilterCondition + ActiveFilter + FilterSetResult types; api.data.setFilter/clearFilter/getActiveFilter; activeFilter + attachFilterToLastMessage + setActiveFilter Zustand. 34 backend + 24 frontend = 58 new tests. Total: 1447 backend + 594 frontend = 2041.

## Day 11 (04:00) — Done
Automated data story — generate_data_story() in core/storyteller.py orchestrates readiness + group-by + target correlations + anomaly count into one narrative; GET /api/data/{id}/story?target=; _STORY_PATTERNS (12 variants) + chat handler → {type:"data_story"} SSE event; DataStoryCard (grade badge, score bar, per-section icons 📊📈🔗⚠️, recommended next step footer); _build_summary() + _recommend_next_step() exported for unit testing; attachDataStoryToLastMessage Zustand action; DataStory + DataStorySection types; api.data.getDataStory(); pandas 4.x StringDtype fix. 45 backend + 13 frontend = 58 new tests. Total: 1413 backend + 570 frontend = 1983.

## Day 10 (20:00) — Done
Chat-initiated model training — _TRAIN_PATTERNS + _detect_train_target(); three cases: (A) existing feature set+target → start directly, (B) feature set+no target → set target+train, (C) no feature set → create minimal FS+train; reuses _train_in_background daemon threads + _training_queues from models.py; {type:"training_started"} SSE event; TrainingStartedCard (target, problem type badge, algorithm chips, Models tab CTA); TrainingStartedResult type; attachTrainingStartedToLastMessage store action. 18 backend + 12 frontend = 30 new tests. Total: 1368 backend + 557 frontend = 1925.

## Day 10 (12:00) — Done
Interactive heatmap chat trigger + column rename — _HEATMAP_PATTERNS emits {type:"chart"} heatmap via existing SSE path; HeatmapChart upgraded with click-to-highlight cells (tooltip shows exact r value, highlights row/col labels); _RENAME_PATTERNS + _detect_rename_request() execute rename synchronously in chat handler + {type:"rename_result"} SSE; POST /api/data/{id}/rename-column with full validation; RenameResultCard; api.data.renameColumn(). 27 backend + 17 frontend = 44 new tests. Total: 1350 backend + 545 frontend = 1895.

## Day 10 (16:02) — Done
Group-by analysis — compute_group_stats() (sum/mean/count/min/max/median, 30-group cap, sorted desc, share-of-total for sum); GET /api/data/{id}/group-stats; _GROUP_PATTERNS + _detect_group_request() (auto-detects categorical group col + numeric value cols + agg keyword); {type:"group_stats"} SSE event; GroupStatsCard (ranked horizontal bars, blue intensity by rank, header count + total, summary footer); attachGroupStatsToLastMessage Zustand action; GroupStatsResult + GroupStatsRow types. 28 backend + 13 frontend = 41 new tests. Total: 1323 backend + 528 frontend = 1851.

## Day 10 (04:00) — Done
Target correlation analysis — analyze_target_correlations() (Pearson ranked, strength labels, plain-English summary); GET /api/data/{id}/target-correlations; _CORRELATION_TARGET_PATTERNS + _detect_correlation_target_request() chat intent; {type:"target_correlation"} SSE event; CorrelationBarCard (horizontal ranked bars, blue=positive/red=negative, strength badges); TargetCorrelationResult + CorrelationEntry types; api.data.getTargetCorrelations(); attachCorrelationToLastMessage store action. 34 backend + 11 frontend = 45 new tests. Total: 1295 backend + 515 frontend = 1810.

## Day 10 (08:02) — Done
Data readiness assessment — compute_data_readiness() (5 components: row count/missing/duplicates/diversity/type quality + optional class balance advisory); GET /api/data/{id}/readiness-check; _DATA_READINESS_PATTERNS + chat intent → {type:"data_readiness"} SSE event; ReadinessCheckCard (score gauge + progress bars + status icons + recommendations; lazy button in Data tab + inline in chat); DataReadinessResult type; api.data.getReadinessCheck(); attachDataReadinessToLastMessage store action. 39 backend + 14 frontend = 53 new tests. Total: 1261 backend + 503 frontend = 1764.

## Day 10 (00:04) — Done
Time-series forecasting — forecast_next_periods() in core/forecaster.py (trend index + cyclic sin/cos features + LinearRegression + 95% CI from residual std); GET /api/data/{id}/forecast?target=&periods=6; _FORECAST_PATTERNS + _detect_forecast_request() chat intent → {type:"forecast"} SSE event; ForecastChart (solid historical line + dashed forecast line + shaded CI band, trend badge, summary). 41 backend + 12 frontend = 53 new tests. Total: 1222 backend + 489 frontend = 1711.

## Day 9 (12:00 session 2) — Done
Segment comparison analysis — compare_segments() (Cohen's d effect size, notable_diffs sorted by magnitude); GET /api/data/{id}/compare-segments (400 on missing values); _COMPARE_PATTERNS + _detect_compare_request() (scans DataFrame for column containing both terms); {type:segment_comparison} SSE event; SegmentComparisonCard (val1 blue/val2 purple, amber notable rows, effect badges, direction arrows); attachSegmentToLastMessage store action; SegmentComparisonResult types; api.data.compareSegments(). 22 backend + 12 frontend = 34 new tests. Total: 1181 backend + 477 frontend = 1658.

## Day 9 (16:10) — Done
API integration code snippets — GET /api/deploy/{id}/integration (curl/Python/JS code from pipeline feature schema; base_url param for production); IntegrationCard (tabbed code blocks, copy-to-clipboard, batch note, OpenAPI link); IntegrationSnippets type; api.deploy.getIntegration(); 18 backend + 16 frontend = 34 new tests. Total: 1159 backend + 465 frontend = 1624.

## Day 9 (12:00) — Done
Computed columns through conversation — add_computed_column() using pd.eval() (safe, no arbitrary Python); POST /api/data/{id}/compute (writes CSV in-place, recomputes profile); _COMPUTE_PATTERNS + _detect_compute_request() (extracts name/expression, validates ≥1 existing column in expression); {type:"compute_suggestion"} SSE event; ComputeCard component (formula display, sample values, Apply button); attachComputeToLastMessage Zustand store action; ComputedColumnSuggestion + ComputeResult types; api.data.computeColumn(). 26 backend + 11 frontend = 37 new tests. Total: 1141 backend + 449 frontend = 1590.

## Day 9 (04:00) — Done
Pivot table / cross-tabulation analysis — build_crosstab() (pd.pivot_table + crosstab, sum/mean/count/min/max, max_rows=15/max_cols=10 cap); GET /api/data/{id}/crosstab; _CROSSTAB_PATTERNS + _detect_crosstab_request() (3-token: value/row/col, 2-token: count mode); {type:"crosstab"} SSE event; CrosstabTable component (zebra-striped, row/col totals, truncated labels); attachCrosstabToLastMessage Zustand store action; CrosstabResult type; api.data.getCrosstab(). 19 backend + 12 frontend = 31 new tests. Total: 1115 backend + 438 frontend = 1553.

## Day 9 (08:07) — Done
AI-powered data dictionary — core/dictionary.py (classify_column_type: id/metric/dimension/date/flag/text heuristics; generate_dictionary: Claude batch + static fallback); GET/POST /api/data/{id}/dictionary; DictionaryCard in Data tab (type badges, Quick summary/AI descriptions buttons, show-more collapse, Regenerate); DataDictionary + ColumnDescription + ColumnSemanticType types; api.data.getDictionary/generateDictionary; patched Claude in tests for deterministic assertions. 32 backend + 15 frontend = 47 new tests. Total: 1096 backend + 426 frontend = 1522.

## Day 9 (20:00) — Done
Cross-deployment model comparison — POST /api/predict/compare (2-4 deployment IDs + features → per-model predictions); GET /api/deployments?project_id= filter; CompareModelsCard on predict/[id] (auto-detects siblings, dropdown + table); api.ts compareModels() + listByProject(); ModelComparisonResult + ComparisonResponse types; fixed routing order (compare before {deployment_id}); fixed 6 pre-existing tests that asserted on exact fetch call count. 11 backend + 10 frontend = 21 new tests. Total: 1064 backend + 411 frontend = 1475.

## Day 9 (00:05) — Done
Prediction confidence intervals — PredictionPipeline.residual_std stored at deploy time (std of training residuals); predict_single returns confidence_interval {lower, upper, level:0.95} for regression; classification gets confidence=max(predict_proba); ConfidenceIntervalBadge + classification confidence badge on predict/[id]; ConfidenceInterval type in types.ts; jest.config.js ESLint disable re-applied. 14 backend + 6 frontend = 20 new tests. Total: 1053 backend + 401 frontend = 1454.

## Day 8 (14:56) — Done
Dataset refresh / guided "new data" workflow — POST /api/data/{id}/refresh (replaces CSV in-place, recomputes profile, validates column compatibility against FeatureSet); _REFRESH_PATTERNS chat intent → {type:refresh_prompt} SSE event with current dataset info; RefreshCard in Data tab (compatible badge, new/removed/missing-feature columns, "Choose New File" button); api.data.refresh() + DatasetRefreshResult + RefreshPrompt types; 22 backend + 14 frontend = 36 new tests. Total: 1039 backend + 395 frontend = 1434.

## Day 5 (04:00) — Done
Workflow progress stepper — WorkflowProgress component (4-step: Upload/Train/Validate/Deploy); status derived from existing React state; clickable steps jump to tab; hasDeployment state tracks deployment dynamically; data-testid on tab buttons; 10 new tests; 381 frontend total.
Also: auto-fixed 149 ruff lint errors (F401/F841/E401/F541/E701) in backend test files and API modules; fixed jest.config.js ESLint error.


## Day 4 (20:00) — Done
Conversational data cleaning — POST /api/data/{id}/clean (remove_duplicates/fill_missing/filter_rows/cap_outliers/drop_column); core/cleaner.py pure functions; _CLEAN_PATTERNS + _detect_clean_op() chat intent; {type:cleaning_suggestion} SSE event (suggest not auto-apply); CleaningCard in Data tab (quality summary + Apply button); api.ts clean() + types; 51 new tests; 1017 backend + 371 frontend = 1388 total.

## Day 4 (10:00) — Done
Model monitoring alerts + chat-triggered visualizations — GET /api/projects/{id}/alerts (stale_model/no_predictions/drift_detected/poor_feedback alerts, critical-first sort); AlertsCard in DeploymentPanel (button + externalAlerts prop); _ALERTS_PATTERNS / _HISTORY_PATTERNS / _ANALYTICS_PATTERNS chat intent detection → {type: alerts/history/analytics} SSE events; 23 backend + 13 frontend = 36 new tests. Total: 1272 tests (934 backend + 338 frontend).


## Day 4 (06:00) — Done
Box plot chart type + prediction session history — build_boxplot() with Tukey fences; GET /api/data/{id}/boxplot; BoxPlotChart SVG; predict/[id] session history + CSV download; 38 new tests; 1203 total (892 backend + 311 frontend).

## Day 4 (02:00) — Done
Smart model health dashboard + guided retraining — GET /api/deploy/{id}/health (unified score: model age + feedback accuracy + drift → health_score 0-100, status, recommendations); POST /api/models/{project_id}/retrain (one-click retrain from existing feature set + selected algorithm); chat _HEALTH_PATTERNS intent → {type: health} SSE event; ModelHealthCard in DeploymentPanel; api.ts health/retrain methods; fixed deployment-panel.test.tsx mock. 27 backend + 12 frontend = 39 new tests. Total: 1148 tests.

## Day 4 (08:06) — Done
Prediction feedback loop — FeedbackRecord model, POST /api/predict/{id}/feedback, GET /api/deploy/{id}/feedback-accuracy, FeedbackCard in DeploymentPanel. Also fixed 2 tuner test failures. 21 new tests. Total: ~827 backend tests.



## Day 3 (20:02) — Done
99% backend coverage (686 backend + 205 frontend = 891 total tests). 53 new targeted tests across 20+ modules. Remaining 1% = ImportError branches + SSE streaming (architecturally uncoverable without uninstalling libraries). See JOURNAL Day 3 (20:02).











## Ideas to Explore

Ideas discovered during sessions. Pick from here or add new ones.

- Full E2E test suite covering upload → explore → train → deploy → predict flow
- Gap analysis: verify every [x] spec item actually works end-to-end
- Integration with XGBoost / LightGBM for better model recommendations
- prompts.py and narration.py modules for richer chat experience
- Self-demo script that exercises the full platform and captures output
- Excel / Google Sheets upload support
- Template projects for common use cases (sales forecast, churn prediction)
- Interactive correlation heatmap visualization
- Multi-dataset join/merge through conversation

## Recently Completed

- Segment comparison analysis — Day 9 (12:00 session 2) — compare_segments() Cohen's d; GET /compare-segments; _COMPARE_PATTERNS auto-column-detection; SegmentComparisonCard (blue/purple, amber notable, effect badges); 34 new tests; 1658 total (1181 backend + 477 frontend)
- Computed columns through conversation — Day 9 (12:00) — add_computed_column() pd.eval(); POST /compute endpoint; _COMPUTE_PATTERNS chat intent; ComputeCard component; 37 new tests; 1590 total (1141 backend + 449 frontend)
- Pivot table / cross-tabulation — Day 9 (04:00) — build_crosstab(); GET /crosstab endpoint; _CROSSTAB_PATTERNS chat intent; CrosstabTable component; 31 new tests; 1553 total (1115 backend + 438 frontend)
- Cross-deployment model comparison — Day 9 (20:00) — POST /api/predict/compare; GET /api/deployments?project_id=; CompareModelsCard on predict page; 21 new tests; 1475 total (1064 backend + 411 frontend)
- Anomaly detection — Day 4 (14:00) — core/anomaly.py (IsolationForest, NaN-tolerant, score 0-100); POST /api/data/{id}/anomalies; chat _ANOMALY_PATTERNS → {type:anomalies} SSE + system prompt injection; AnomalyCard (summary, features used, scored table, scan button); explore suggestion chip "Are there any unusual records?"; 33 new tests; 978 backend + 359 frontend = 1337 total
- Scenario comparison + chat suggestion chips — Day 4 (20:03) — POST /api/predict/{id}/scenarios (N labelled what-ifs → N predictions + best/worst summary); generate_suggestions() (6-state pool, dynamic artefact-aware additions); {type:suggestions} SSE event; clickable pill chips in frontend; 22 backend + 10 frontend = 32 new tests; 1299 total (951 backend + 348 frontend)

- Model version history timeline — Day 4 (16:04) — GET /api/models/{project_id}/history; _compute_trend (linear regression slope, 2%-of-mean stability floor); VersionHistoryCard (LineChart + stats + run table + Current/Live badges); history loaded on mount + SSE refresh; fixed tuning-narrative mock; 37 new tests; 1254 total (911 backend + 343 frontend)

- Live prediction explanation on public dashboard — Day 4 (12:04) — POST /api/predict/{id}/explain (feature contributions, summary, top_drivers); PredictionPipeline stores means/stds; predict/[id] page "Why this prediction?" waterfall; FeatureContribution + PredictionExplanation types; 11 backend + 6 frontend = 17 new tests; ~1182 total

- Smart model health dashboard + guided retraining — Day 4 (02:00) — GET /api/deploy/{id}/health (unified 0-100 score: age + feedback + drift); POST /api/models/{project_id}/retrain (one-click retrain); chat health intent + {type:health} SSE event; ModelHealthCard; 39 new tests; 1148 total (854 backend + 294 frontend)
- Prediction feedback loop — Day 4 (08:06) — FeedbackRecord model; POST /api/predict/{id}/feedback (actual_value/actual_label/is_correct auto-compute); GET /api/deploy/{id}/feedback-accuracy (MAE/pct_error for regression, accuracy for classification, verdict + retrain suggestion); FeedbackCard in DeploymentPanel; 21 backend tests; ~827 total
- 2 tuner test fixes — Day 4 (08:06) — test_tune_untuneable_algorithm and test_tune_full_workflow updated to match synchronous endpoint behavior
- Hyperparameter auto-tuning + AI project narrative — Day 4 (04:44) — POST /api/models/{run_id}/tune (RandomizedSearchCV, 9 algorithm grids, before/after comparison); POST /api/projects/{id}/narrative (Claude + static fallback executive summary); TuningCard in ModelTrainingPanel; 25+21 backend + 13 frontend = 59 new tests; ~1052 total
- Hyperparameter auto-tuning — Day 3 (22:00) — core/tuner.py (RandomizedSearchCV per-algo grids); POST /tune endpoint (bg thread, SSE); chat _TUNE_PATTERNS intent + {type:tune} event; api.ts.models.tune(); 22 new tests; 760 backend total
- Prediction drift detection + what-if analysis — Day 3 (18:00) — GET /drift (z-score/TVD from PredictionLog, no schema change); POST /whatif (two predictions + delta); chat drift intent + SSE event; DriftCard + WhatIfCard in DeploymentPanel; fixed 4 pre-existing test failures; 21 new tests; 1007 total (738 backend + 269 frontend)
- Prediction logging + analytics + model readiness — Day 4 (00:08) — PredictionLog model; /analytics + /logs endpoints; /readiness checklist; chat intent detection; DeploymentPanel ReadinessCard + AnalyticsCard; 46 new tests; 986 total (720 backend + 266 frontend)
- Frontend coverage 63%→91% — Day 3 (14:00) — 49 workspace page tests; scrollIntoView jsdom stub; types.ts+layout.tsx excluded from coverage; 254 frontend + 686 backend = 940 total tests; both stacks exceed 85% target

<!-- Move items here after implementation. Format: -->
<!-- - [Description] — Day N (HH:MM) — [1-line outcome] -->

- Coverage 98%→99% — Day 3 (20:02) — 53 targeted tests in test_final_coverage.py; 20+ modules covered; 686 backend tests; 9196 stmts 73 missing 99%; remaining 1% = ImportError + SSE (impossible)
- Google Sheets URL import + sub-component test coverage — Day 3 (16:03) — POST /api/data/upload-url (Sheets + CSV URL); urllib.request download; UploadPanel URL toggle in frontend; PipelinePanel/DatasetListPanel/FeatureImportancePanel 38 new tests; 735 total
- Excel/XLSX upload + Neural Network MLP — Day 3 (12:03) — openpyxl Excel ingest (convert to CSV), frontend dropzone update; MLPRegressor/MLPClassifier in algorithm registry; 21 new tests; 530 total
- Multi-dataset support — Day 3 (02:00) — suggest_join_keys + merge_datasets in core/merger.py; 3 endpoints (list/join-keys/merge); DatasetListPanel in Data tab; 31 tests; 509 total
- Data transformation pipeline with undo + scatter brushing — Day 3 (08:04) — GET/POST/DELETE /steps endpoints; PipelinePanel UI; InteractiveScatterChart with click-to-highlight; 14 new tests; 478 total; fixed pytest-asyncio missing dep
- Smarter chat orchestration — Day 2 (22:00) — _call_claude() + narrate_data_insights_ai() + narrate_training_with_ai() + _detect_model_regression() + recent_messages multi-turn context; 20 tests; 464 total
- XGBoost/LightGBM integration + performance baseline + template projects — Day 3 (04:31) — xgb/lgbm in algorithm registry (16 tests); perf_baseline.json seeded (upload 28ms, predict 4ms); 3 templates with sample datasets (20 tests); 444 total tests
- Gap analysis + frontend Jest + self-demo — Day 3 (18:00) — 69 frontend tests (store/api/components/utils); scripts/demo.py 15/15 PASS in 2.8s; fixed NL query TypeError 500; 469 total tests
- Coverage hardening + training resilience + time-series decomp — Day 3 (00:09) — 62 new tests; backend 94%→97%; model training failure path; time-series 3-series line chart; 400 total tests pass
- E2E test suite build-out (upload/training/deploy) — Day 2 (10:00) — 33 Playwright tests; fixed 2 UX bugs (dataset restore + ModelTrainingPanel runs restore); 33/33 pass
- Smarter chat orchestration (prompts.py + narration.py) — Day 2 (16:08) — auto-inject upload/training messages into chat; 44 tests; 255 total pass
- Error resilience audit + query engine tests + correlation heatmap — Day 2 (20:05) — 72 new tests; 2 real bugs fixed (NaN/inf in preview); query_engine 14%→92%; total coverage 95%; heatmap chart type + endpoint
- Integration tests + radar chart — Day 2 (14:00) — 11 integration tests (upload→deploy→predict); radar chart for model comparison with normalized metrics; 338 total backend tests pass
