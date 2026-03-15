# Evolution Backlog

Living document for coordinating between bot instances and tracking ideation.
**Read this before starting work. Write your focus before implementing.**

## Currently Working On

<!-- Each bot writes here BEFORE starting implementation. Format: -->
<!-- ## [Bot ID / Timestamp] — [Focus Area] -->
<!-- Brief description of what you're doing this session. -->
<!-- Remove your entry when you commit your session wrap-up. -->


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
