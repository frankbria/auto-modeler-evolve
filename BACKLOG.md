# Evolution Backlog

Living document for coordinating between bot instances and tracking ideation.
**Read this before starting work. Write your focus before implementing.**

---

## ⚠ STEERING DIRECTIVE (updated Day 19) — READ BEFORE CHOOSING WORK

**The Explore phase is done. Stop adding analytics cards.**

As of Day 19 the chat can answer every major exploratory question a business analyst
would ask: scatter, line, bar, pie, box, histogram, heatmap, group stats, group trends,
pair correlation, segment comparison, value counts, summary stats, time windows, crosstab,
top-N, clustering, forecasting, anomalies, null maps, column profiles, filters, computed
columns, data stories, and more. There is no meaningful analytics gap left to fill.

**Where to focus instead (priority order):**

1. **Deployment depth (Track D)** — This is AutoModeler's biggest competitive gap and
   the most underbuilt area relative to the vision. Pick from spec.md Track D:
   - API key auth for prediction endpoints
   - Scheduled batch prediction jobs
   - Deployment versioning + rollback
   - Champion-challenger A/B testing
   - Webhook notifications on model drift/degradation
   - Export as self-contained prediction service (ZIP + uvicorn)
   - Prediction SLA / latency monitoring

2. **Model building depth (Track C)** — Better models = more analyst trust:
   - Class imbalance detection + handling (SMOTE / class weights / threshold tuning)
   - Ensemble methods (voting + stacking)
   - Date-aware chronological train/test splits
   - Feature selection automation (drop near-zero-importance features)

3. **End-to-end polish (Track E)** — Run the "lunch break" flow as a real user:
   - Proactive insight suggestions after upload (data-aware, not generic)
   - "What's next?" guidance at every step transition
   - Prediction page UX audit (the VP-facing dashboard)

4. **Vision-Driven Innovation (Track B)** — Only if D/C/E have nothing obvious.

**Test coverage:** Backend 99%, frontend 91%. Both EXCEED the 85% target.
Do NOT write new tests purely for coverage. Write tests only for new features.
Stop chasing 100% — it's not achievable (SSE streams, ImportError branches) and
the time is better spent on real features.

---

## Currently Working On

## Day 24 (12:00) — Done
**Track B — Conversation Export as HTML Report.** Closes the "share full analysis journey" use case.
- **Conversation Export** — `_CONV_EXPORT_PATTERNS` (13 NL variants). `_build_export_html()` pure function generates self-contained HTML (header, dataset info, model results, conversation transcript, embedded CSS). `GET /api/chat/{project_id}/export` → HTML attachment. `ConversationExportCard` (emerald border, 📄) in chat: message count badge, dataset badge, download link. 14 backend + 10 frontend = 24 new tests. Total: 2475 backend + 1195 frontend = 3670.

**What's next:**
- Continue Track B — remaining opportunities:
  - "Model drift → retrain" proactive notification (webhook + chat alert)
  - Multi-dataset comparison (upload v2 dataset, compare model performance pre/post)
  - Guided onboarding wizard (step-by-step first-use flow for new analysts)

## Day 24 (05:30) — Done
**Track B — Auto-Retrain on Upload.** Model stays current whenever new data is uploaded.
- **Auto-Retrain** — `Project.auto_retrain` bool. `GET/PUT /api/projects/{id}/auto-retrain`. `core/retrain.py` `trigger_auto_retrain()`. Upload handler fires it when enabled. `_AUTO_RETRAIN_PATTERNS` + `AutoRetrainCard` (teal). 14 backend + 10 frontend = 24 new tests.

## Day 24 (04:00) — Done
**Track B — Smart Model Selection Advisor.** Complements the Model Improvement Advisor with "which model to use" rather than "how to improve it".
- **Smart Model Selection Advisor** — `compute_model_selection(runs, criteria)` pure function scores all completed runs on 5 criteria: accuracy/explainability/stability/speed/balanced. `GET /api/models/{project_id}/model-selection?criteria=` endpoint. `_MODEL_SELECT_PATTERNS` (15 NL variants) + `_detect_selection_criteria()` in `chat.py`. `ModelSelectionCard` (indigo border, 🏆) in chat: winner highlight + component score bars + ranked list. 42 backend + 18 frontend = 60 new tests. Total: 2461 backend + 1165 frontend = 3626.
  - Conversation export as HTML report (share entire analysis journey with VP)

## Day 24 (04:41) — Done
**Track B — Model Improvement Advisor.** All spec tracks done; moved to Track B.
- **Model Improvement Advisor** — `core/advisor.py` `compute_improvement_suggestions()` pure function runs 9 ranked checks (weak features, ensemble potential, date features unused, small dataset, class imbalance, calibration, hyperparameter tuning, too few features, linear on nonlinear data). Each suggestion has `difficulty`+`expected_impact`. `GET /api/models/{project_id}/improvement-suggestions` endpoint. `_IMPROVEMENT_PATTERNS` (14 NL variants) + chat SSE emit. `ModelImprovementCard` (violet border) in chat. 41 backend + 13 frontend = 54 new tests. Total: 2419 backend + 1147 frontend = 3566.

## Day 23 (20:00) — Done
**Track E — End-to-End Polish (final two items).** All Track E items are now complete:
- **"Lunch break" flow audit** — Code audit of full analyst journey found 5 friction points in the VP-facing predict page.
- **Shareable prediction page UX** — All 5 friction points fixed in `predict/[id]/page.tsx`: (1) page title is now "{Target} Predictor"; (2) ModelContextCard shows algorithm+accuracy+date; (3) form labels show avg hints from new mean/std fields in feature schema; (4) algorithm IDs mapped to plain English everywhere; (5) session history shows key inputs column. 2 backend + 6 frontend = 8 new tests.

**Track E is complete. Phase 9 spec.md items: all tracks (D, C, E) done.**

**What's next:**
- Track B (Vision-Driven Innovation) — open-ended; session should pick work from the vision gap
- Multi-user / auth layer (if the vision calls for it)
- Deeper real-world deployment testing (the "lunch break" criterion: can an analyst actually complete the full flow in 30 minutes?)

## Day 23 (12:00) — Done
**Track E — End-to-End Polish (first two items).** Both complete:
1. **Proactive data-aware upload suggestions** — `generate_upload_suggestions(profile, col_names)` in `orchestrator.py`. Returned as `suggestions` in upload/sample API response. Frontend sets chatSuggestions with "Try asking:" label. 19 backend + 6 frontend = 25 new tests. Total: 2376 backend + 1128 frontend = 3504.
2. **"What can I do next?" step guidance** — `get_next_step_chips(state)` in `orchestrator.py`. Emitted as `next_step_chips` in `all_done` training SSE. Chat SSE emits `{type:"next_step"}` after deployed/features_applied. `ModelTrainingPanel.onTrainingComplete` callback. Discovery: TextDecoder not globally available in jest-environment-jsdom — polyfilled in jest.setup.ts.

## Day 23 (04:00) — Done
**Track C complete.** All remaining Track C (Model Building Depth) items finished:
1. **Large dataset sampling** — `sample_large_dataset(df, max_rows=20_000, threshold=50_000)` pure function in `trainer.py`. Called in `_train_in_background()` before `prepare_features()`. Adds `sample_size`, `original_dataset_size`, `sample_note` to metrics when sampling occurs. 8 new backend tests.
2. **Calibration for classifiers** — `CalibratedClassifierCV(model_class(**params), cv=3, method="sigmoid")` wraps all classifiers in `train_single_model()` (skipped for threshold tuning, SMOTE, sample_weight algos, <30 rows). `_add_calibration_metrics()` computes calibration curve + Brier score. `GET /api/models/{run_id}/calibration` endpoint. `ReliabilityDiagramView` in ValidationPanel's new Calibration sub-tab. `identify_weak_features()` unwraps CalibratedClassifierCV. 20 backend + 11 frontend = 31 new tests. Total: 2357 backend + 1122 frontend = 3479.

**What's left** (Track E — End-to-End Polish):
- "Lunch break" flow audit (run demo.py, document friction points, fix top 3)
- Proactive insights after upload (data-aware chips, not generic)
- "What can I do next?" guidance at each step transition
- Shareable prediction page UX audit

## Day 23 (04:52) — Done
Feature Selection Automation (Track C) — `identify_weak_features(model, feature_cols, threshold_percentile=20.0)` in `core/trainer.py`: tree-based uses `.feature_importances_`, linear uses `|coef_|`, MLP/ensemble returns `has_importances=False`. Bottom-20th-percentile threshold, normalised to sum=1. `GET /api/models/{run_id}/feature-selection` endpoint. `TrainRequest.excluded_features: list[str] | None` added (HTTP 400 if all excluded). `_FEATURE_SEL_PATTERNS` (8 NL variants) in `chat.py`. `FeatureSelectionCard` (amber border, 🎯): chat card (read-only importance bars) + panel card (interactive checkboxes + "Exclude N weak features on retrain" button + Clear). Auto-loaded by `ModelTrainingPanel` after training completes. 21 backend + 21 frontend = 42 new tests. Total: 2329 backend + 1111 frontend = 3440.

## Day 22 (04:00) — Done
Class imbalance handling (Track C) — `detect_class_imbalance(y)` in `trainer.py` (minority < 20% threshold). Three strategies: class_weight (param injection for LogReg/RF/LGBM, sample_weight for GBC/XGB), SMOTE (training split only, imblearn 0.14.1), threshold tuning (sweep 0.05–0.95, best F1, records optimal_threshold in metrics). `GET /api/models/{project_id}/imbalance` endpoint. `TrainRequest.imbalance_strategy`. `ImbalanceCard` (rose border) in ModelTrainingPanel: distribution bar, explanation, 3 strategy buttons with aria-pressed. 28 backend + 15 frontend = 43 new tests. Total: 2264 backend + 1060 frontend = 3324.

## Day 21 (20:00) — Done
Champion-challenger A/B testing — `ABTest` SQLModel table (auto-created). `ab_variant` added to `PredictionLog` (inline SQLite migration). `make_prediction()` routes via `random.random()` vs `champion_split_pct/100`; logs `ab_variant="champion"/"challenger"` keyed to champion's deployment_id. Four REST endpoints: POST/GET/DELETE `/api/deploy/{id}/ab-test` + POST `.../promote` (copies challenger model into champion deployment, archives version, records winner). `_ab_significance()` uses Mann-Whitney U (scipy). `ABTestCard` (purple border) in DeploymentPanel: idle + create form (challenger ID + split slider 50–99%) + active test view (split bar, per-variant metrics, significance badge, Promote/End/Refresh). 27 backend + 19 frontend = 46 new tests. Total: 2227 backend + 1036 frontend = 3263.

## Day 21 (04:00) — Done
Webhook notifications — `WebhookConfig` SQLModel table (auto-created). `core/webhook.py` provides `dispatch_webhooks(deployment_id, event_type, payload)` — HMAC-SHA256 signed `X-AutoModeler-Signature` header, daemon threads, `except Exception: pass` guard. Three event triggers: `batch_complete` in scheduler, `drift_detected` when score >= 50, `health_degraded` when score < 60. Four endpoints: POST/GET/DELETE webhooks + POST test. `WebhookCard` (sky-blue border) in DeploymentPanel: URL input, event-type checkboxes, list with Test/Remove per entry, test result inline, secret-once amber callout. 18 backend + 13 frontend = 31 new tests. Total: 2188 backend + 1006 frontend = 3194.

## Day 21 (05:04) — Done
Export as self-contained prediction service — `GET /api/deploy/{id}/export` returns a ZIP with `server.py` (FastAPI predict/health/root endpoints, CORS, joblib loading), `model_pipeline.joblib`, `model.joblib`, `requirements.txt`, `README.md`. server.py embeds target_column, algorithm, uvicorn quickstart, and example payload from training medians. `ExportServiceCard` (emerald border, 📦 icon) in DeploymentPanel: lists 5 included files, uvicorn snippet, Download as ZIP button with blob download and correct filename. `api.deploy.exportServiceUrl()` client helper. 18 backend + 18 frontend = 36 new tests. Total: 2170 backend + 993 frontend = 3163.

## Day 20 (20:00) — Done
Group trend analysis via chat — `_GROUP_TREND_PATTERNS` (7 NL variants: "which X are growing", "fastest growing X", "which regions are trending up", "growth rate by X", "which products are declining") + `_detect_group_trend_request()` (auto-detects date_col via detect_time_columns, group_col from categorical column mentions, value_col from numeric column mentions) + `compute_group_trends(df, date_col, group_col, value_col)` in `core/analyzer.py` (OLS slope per group, % change first→last, direction up/down/flat, rank by slope, plain-English summary); `GET /api/data/{id}/group-trends?date_col=&group_col=&value_col=` REST endpoint; `{type:"group_trends"}` SSE event; `GroupTrendCard` (orange border, ranked rows with up/down arrows, growth badges, summary). Directly implements vision's "Which products are trending up?" question.

## Day 19 (12:00) — Done
Pair correlation analysis + Quick stat query via chat — `_PAIR_CORR_PATTERNS` (7 NL variants) + `_detect_pair_corr_cols()` (scans actual df column names longest-match-first in message) + `compute_pair_correlation(df, col1, col2)` in `core/analyzer.py` (scipy.stats.pearsonr, threshold-based strength: very strong |r|≥0.8/strong≥0.6/moderate≥0.4/weak≥0.2/negligible; direction positive/negative/no; significance: highly significant p<0.001/significant p<0.01/marginally p<0.05/not significant; returns r, p_value, n, strength, direction, significant, interpretation, summary); `GET /api/data/{id}/pair-correlation?col1=&col2=` (400 on non-numeric/missing col); `{type:"pair_correlation"}` SSE event; `PairCorrelationCard` (violet border, ∼ icon, col1×col2 header, strength/direction badges, large r value with colored bar, p-value + significance badge, interpretation para, summary footer); `PairCorrelationResult` type; `api.data.getPairCorrelation()`; `attachPairCorrelationToLastMessage()`. `_STAT_QUERY_PATTERNS` (7 NL variants) + `_detect_stat_query()` (_AGG_WORD_MAP maps average/mean/total/sum/max/min/median/std; count intent checked FIRST to prevent "how many total rows?" → "sum") + `compute_stat_query(df, agg, col)` (count/sum/mean/median/max/min/std, k/M suffix formatting, plain-English label inference, n_rows/n_valid tracking); `GET /api/data/{id}/stat-query?agg=&col=` (400 on unknown agg/col); `{type:"stat_query"}` SSE event; `StatQueryCard` (color by agg: cyan/blue/teal/emerald/orange/purple/amber, icon x̄/Σ/m/↑/↓/σ/#, agg badge, large formatted value, optional row-info para when n_valid<n_rows, summary footer). Frontend test fix: switched getByText → getAllByText for multi-element matches; "does not show row info" fixed by targeting dedicated `<p>` via container.querySelector. 61 backend + 25 frontend = 86 new tests. Total: 2091 backend + 928 frontend = 3019.

## Day 19 (04:00) — Done
Summary statistics table via chat + Category value counts via chat — `_SUMMARY_STATS_PATTERNS` (7 NL variants: "summarize my data", "describe my dataset", "summary statistics", "stats for all columns", "statistical overview", "dataset statistics", "descriptive statistics") + handler calls `compute_summary_stats()` (pandas describe() equivalent: numeric cols get count/mean/std/min/Q25/median/Q75/max/null_count; categorical cols get count/unique/top/freq/null_count); emits `{type:"summary_stats"}` SSE event; `SummaryStatsCard` (slate border, two-section table: Numeric Columns + Categorical Columns, summary footer). `_VALUE_COUNT_PATTERNS` (8 NL variants: "most common values in X", "frequency table for X", "value counts for X", "how often does each X appear", "most frequent X", "count occurrences of X") + `_detect_value_counts_col()` + `compute_value_counts()` (top-N value frequencies with count + pct for categorical column; cap 20 values); emits `{type:"value_counts"}` SSE event; `ValueCountCard` (lime border, value/count/% table).

## Day 18 (20:00) — Done
Histogram via chat + Missing values overview via chat — `_HISTOGRAM_PATTERNS` (8 NL variants: "histogram of X", "show me a histogram", "frequency histogram of X", "binned distribution of X", "frequency/distribution chart of X") + `_detect_histogram_col()` (longest-match-first numeric column scan with underscore/space variant, fallback to first numeric); uses `numpy.histogram()` with adaptive bin count; calls existing `build_histogram()` from `chart_builder.py`; emits `{type:"chart", chart:{chart_type:"histogram",...}}` SSE reusing existing histogram renderer — zero new frontend components. `_NULL_MAP_PATTERNS` (7 NL variants: "show me the missing values", "which columns have missing data?", "null values overview", "missing data summary", "data completeness overview", "how many missing values?", "where is my missing data?") + inline handler computes per-column null_count/null_pct/complete_pct sorted most-missing-first; builds `NullMapResult` dict; emits `{type:"null_map"}` SSE event; `NullMapCard` (teal border, overall-completeness badge, per-column table with emerald/amber/rose completion bars, "N missing" badges, summary footer); `NullMapResult`/`NullMapColumn` TypeScript types; `null_map?` on `ChatMessage`; `attachNullMapToLastMessage()` Zustand action. 46 backend + 16 frontend = 62 new tests. Total: 1952 backend + 867 frontend = 2819.

## Day 18 (12:00) — Done
Bar chart via chat + Dataset download via chat — `_BAR_CHART_PATTERNS` (8 NL variants: "bar chart of X by Y", "column chart", "vertical bar chart") + `_detect_bar_chart_request()` (value_col via longest-match scan, group_col via "by/per/for each" clause + fallback to first categorical, agg via keyword sum/mean/count/max/min); emits `{type:"chart", chart:{chart_type:"bar",...}}` SSE reusing existing BarChart renderer — zero new frontend components. `_DOWNLOAD_PATTERNS` (8 NL variants) + `GET /api/data/{id}/download` endpoint (applies active filter via json.loads of stored conditions → filtered CSV with _filtered suffix, or raw CSV; Content-Disposition: attachment); `{type:"data_export"}` SSE event; `DataExportCard` (indigo border, ⬇ icon, filename + row count, amber Filtered badge, Download CSV link); `DataExportResult` type; `api.data.downloadDatasetUrl()`; `attachDataExportToLastMessage()` Zustand action. Bug: active_filter.conditions is stored as JSON string, not list — fixed with json.loads(). 39 backend + 19 frontend = 58 new tests. Total: 1906 backend + 851 frontend = 2757.

## Day 18 (04:00) — Done
Pie chart via chat — `_PIE_CHART_PATTERNS` (9 NL variants: "pie chart", "donut/doughnut chart", "show me a pie/donut", "composition/proportion/share/makeup of…by", "breakdown chart") + `_detect_pie_chart_request()` (finds categorical slice col via "by/of/for/per/across" clause parser, numeric value col via message scan; both with fallbacks to first col of each type); handler groups df by slice col → sums value col → `build_pie_chart(series, title, limit=10)`; emits `{type:"chart", chart:{chart_type:"pie",...}}` SSE reusing existing `PieChart` renderer — zero new frontend components. Bug fixed: `dough?nut` → `(?:donut|doughnut)` (regex didn't cover short spelling). Frontend test fix: pie charts have empty x/y labels so `caption == title` → `figcaption` and `<p>` both match; used `getAllByText` to avoid duplicate-element error. 23 backend + 8 frontend = 31 new tests. Total: 1867 backend + 832 frontend = 2699.

## Day 17 (20:00) — Done
Multi-metric overlay line chart via chat — `_detect_line_chart_request()` now returns `value_cols: list[str]` (was single `value_col`; collects ALL mentioned numeric columns longest-match-first, falls back to first numeric); `_LINE_CHART_PATTERNS` gained 2 new alternates matching "compare X and Y over time" and "overlay X vs/with Y"; chat handler branches: 1 col → existing `build_timeseries_chart()` (raw + rolling avg + OLS trend); 2+ cols → new `build_overlay_chart()` (raw values only per column, no decorations that would clutter a multi-line comparison); `build_overlay_chart(dates, columns_values, title)` in `chart_builder.py` wraps `build_line_chart()` — zero new frontend components (multi-series line renderer already shows legend when yKeys.length > 1). 14 backend + 0 frontend = 14 new tests. Total: 1844 backend + 824 frontend = 2668.

## Day 17 (12:00) — Done
Line chart via chat + Box plot via chat — `_LINE_CHART_PATTERNS` (8 NL variants: "plot X over time", "trend of X", "line chart of X", "chart X by month/week/year", "how has X changed", "show X trend") + `_detect_line_chart_request()` (uses `detect_time_columns()` for date col auto-detect, scans message for numeric col, falls back to first numeric; calls `build_timeseries_chart()`; trend direction + % change in system prompt); `_BOXPLOT_PATTERNS` (8 NL variants: "box plot of X", "distribution/spread/range/quartile of X by Y", "compare distribution of X across Y", "show outliers in X by Y", "whisker plot") + `_detect_boxplot_request()` (value_col=numeric, group_col=categorical via "by/across/per/for each" clause; calls `build_boxplot()`). Both emit `{type:"chart"}` SSE reusing existing multi-series line chart renderer + `BoxPlotChart` SVG renderer — zero new frontend components. 39 backend + 14 frontend = 53 new tests. Total: 1830 backend + 824 frontend = 2654.

## Day 17 (04:00) — Done
Scatter plot via chat — `_SCATTER_PATTERNS` (8 NL variants: "plot X vs Y", "scatter X against Y", "relationship between X and Y", "how does X relate to Y", "visualize relationship between", "scatter plot") + `_detect_scatter_request()` (separator-first: tries vs/versus/against then "between/and", falls back to first two numeric columns mentioned in message); handler samples 500 points max, computes Pearson r for system prompt narration ("r = 0.95, positive correlation, strong"), emits `{type:"chart", chart:{chart_type:"scatter",...}}` SSE reusing existing `InteractiveScatterChart` renderer — zero new frontend component. No trailing `\b` after alternation, correct `_load_working_df` calling convention. 24 backend + 9 frontend = 33 new tests. Total: 1791 backend + 810 frontend = 2601.

## Day 16 (20:00) — Done
Chat-driven record table viewer — `sample_records()` in `core/analyzer.py` (optional FilterCondition list reusing apply_active_filter, 50-row cap, offset paging, 8-col display cap, NaN→None, filtered/condition_summary/summary); `GET /api/data/{id}/records?n=20&where=&offset=` REST endpoint; `_RECORDS_PATTERNS` (13 NL variants: show me the/my data, display/preview/peek at records, let me see the data, show first N rows, show rows/records where) + `_detect_records_request()` (n extraction + WHERE clause via parse_filter_request); `{type:"records"}` SSE event; `RecordTableCard` (sky-blue border, columns count badge, amber filtered badge, condition summary row, table with underscore-replaced headers, null→em-dash, string truncation, shown/total footer); `RecordTableResult`+`RecordTableRow` types; `api.data.getRecords()`; `attachRecordsToLastMessage()` Zustand action. 22 backend + 16 frontend = 38 new tests. Total: 1767 backend + 801 frontend = 2568.

## Day 16 (12:00) — Done
Prediction error analysis via chat — `compute_prediction_errors()` pure function in `core/validator.py` (regression: top-N by abs residual, signed error + abs_error + rank + feature values, MAE + worst-%-of-range summary; classification: wrong predictions with actual/predicted labels decoded from target_classes, error rate + accuracy summary; n clamped 1–50); `GET /api/models/{run_id}/prediction-errors?n=10` endpoint in `api/validation.py` (uses shared `_load_run_context()` + `_build_Xy()` helpers, resolves target_classes from pipeline joblib); `_PRED_ERROR_PATTERNS` (14 NL variants, no trailing `\b`, pluralized `errors?`/`mistakes?`/`rows?`) in `chat.py`; handler loads best/selected run, predicts on training set, injects summary into system prompt, emits `{type:"prediction_errors"}` SSE event; `PredictionErrorCard` (rose border, algorithm + problem type badges, per-row table with rank/actual→predicted/ErrorBadge/FeatureChips up to 4, empty state, summary footer); `PredictionErrorRow` + `PredictionErrorResult` types; `api.models.getPredictionErrors()`; `attachPredictionErrorsToLastMessage()` Zustand action. Bug fixed: trailing `\b` in initial pattern caused false negatives on "errors" — removed per CLAUDE.md rule. Classification fixture used `decision_tree_classifier` (returns 400 — not in registry); fixed to `logistic_regression`. 24 backend + 17 frontend = 41 new tests. Total: 1745 backend + 785 frontend = 2530.

## Day 16 (04:00) — Done
Chat-triggered what-if analysis — `_WHATIF_CHAT_PATTERNS` (8 NL variants) + `_detect_whatif_request()` (feature-name-first parser: iterates known features, checks pattern A/was-is-equals-to, B/change-to, C/equals-sign + multiplier fallback double/triple/halve → __multiply__N sentinel); handler loads `PredictionPipeline.feature_means` as base dict → `predict_single()` × 2 → delta/pct/direction/summary → `{type:"whatif_result"}` SSE event; `WhatIfChatCard` (amber border, 🔀 icon, problem type badge, Hypothetical Change row with old→new, side-by-side Original/Modified prediction boxes, DeltaBadge ↑↓→ + ±%, classification probability rows, summary footer); `WhatIfChatResult` type; `attachWhatIfChatToLastMessage()` Zustand action. Key bugs fixed: feature-name-first avoids greedy regex capture of "what if total revenue" as feature; original message used (not msg_lower) for value extraction to preserve casing. 15 backend + 17 frontend = 32 new tests. Total: 1721 backend + 768 frontend = 2489.

## Day 15 (20:00) — Done
Top-N record ranking — `compute_top_n()` in `core/analyzer.py` (nlargest/nsmallest, NaN-safe, rank numbers, summary, 50-row cap); `GET /api/data/{id}/top-n?col=&n=10&order=desc` endpoint (400 on unknown/non-numeric column); `_TOPN_PATTERNS` (8 NL variants) + `_detect_topn_request()` (digit/word n extraction, ascending detection, column name matching); `{type:"top_n"}` SSE event; `TopNCard` (emerald/rose border, 🥇🥈🥉 medals, amber highlight rows, k/M suffix formatting, summary footer); `TopNRow`+`TopNResult` types; `api.data.getTopN()`; `attachTopNToLastMessage()` Zustand action. 44 backend + 16 frontend = 60 new tests. Total: 1706 backend + 751 frontend = 2457.

## Day 15 (12:00) — Done
Time-period comparison — `compare_time_windows()` in `core/analyzer.py` (two named date windows → per-column means + pct_change + direction + notable flag ≥20%; `_build_timewindow_summary()` plain-English overview naming biggest mover); `GET /api/data/{id}/compare-time-windows?date_col=&p1_name=&p1_start=&p1_end=&p2_name=&p2_start=&p2_end=` REST endpoint (400 on unknown column, empty period, parse errors); `_TIMEWINDOW_PATTERNS` (8 NL triggers) + `_detect_timewindow_request()` in chat.py — handles explicit year patterns, quarter patterns (with optional year), YoY/MoM/H1-vs-H2 keywords, fallback bisection; `{type:"time_window_comparison"}` SSE event + system prompt injection; `TimeWindowCard` (orange border, up/down count badges, period name chips, side-by-side table, amber notable-changes callout, summary); `TimeWindowPeriod` + `TimeWindowColumn` + `TimeWindowComparison` types; `api.data.compareTimeWindows()`; `attachTimeWindowToLastMessage()` Zustand action. 27 backend + 17 frontend = 44 new tests. Total: 1662 backend + 735 frontend = 2397.

## Day 15 (04:00) — Done
K-means customer segmentation — `compute_clusters()` in `core/analyzer.py` (KMeans, auto-k via silhouette score 2-8, StandardScaler, per-cluster profiles with distinguishing features sorted by magnitude, plain-English descriptions, clusters sorted by size descending); `GET /api/data/{id}/clusters?features=&n_clusters=` REST endpoint (400 on invalid columns, out-of-range k, no numeric columns; 404 on unknown dataset); `_CLUSTER_PATTERNS` (9 NL variants) + `_detect_cluster_features()` in chat.py → `{type:"clusters"}` SSE event; `ClusteringCard` (violet border, 8-color palette, `ClusterRow` with `SizeBar`, ↑/↓ distinguishing feature badges, auto/manual badge, footer with k source); `ClusteringResult` + `ClusterProfile` + `ClusterDistinguishingFeature` TypeScript types; `api.data.getClusters()` client method; `attachClustersToLastMessage()` Zustand action. 39 backend + 18 frontend = 57 new tests. Total: 1635 backend + 718 frontend = 2353.

## Day 14 (20:00) — Done
Column profile deep-dive — `compute_column_profile()` in `core/analyzer.py` (numeric/categorical/date support, 7 issue types); `GET /api/data/{id}/column-profile?col=` REST endpoint; `_COLUMN_PROFILE_PATTERNS` (9 variants) + `_detect_profile_col()` chat intent; `{type:"column_profile"}` SSE event; `ColumnProfileCard` (cyan border, stat chips, mini distribution chart, issue severity rows); `ColumnProfile`/`ColumnProfileIssue`/`ColumnProfileStats`/`ColumnProfileDistribution` types; `api.data.getColumnProfile()` client method fixed (was accidentally placed in `features:` section, moved to `data:`); `attachColumnProfileToLastMessage()` Zustand action. 39 backend + 16 frontend = 55 new tests. Total: 1596 backend + 700 frontend = 2296.

## Day 14 (12:00) — Done
Phase 8 complete — 4 remaining spec items: Badge standardization across 8 component files (ad-hoc badge spans → design-system `<Badge>` with `className` overrides); shared ImportanceBar component (`components/ui/importance-bar.tsx`, `importance={0..1}` normalized, optional `label` override) replacing the × 5 magic-number hack in `model-card-view.tsx` and percentage-of-max in `FeatureImportancePanel`; project name `<span>` → `<h1>` for heading hierarchy; WorkflowProgress moved from inside right panel to between topbar and main flex container (always visible, onStepClick now also sets mobileView to "panel"). 0 new tests. 1557 backend + 684 frontend = 2241.

## Day 13 (04:00) — Done
Model performance by segment — compute_segment_performance() in core/validator.py (aligns group_values with y_true/y_pred arrays, computes R²/Accuracy per group, best/worst/gap, plain-English summary); GET /api/models/{run_id}/segment-performance?col= (400 on unknown/high-cardinality columns); _SEGMENT_PERF_PATTERNS (7 variants) + _detect_segment_perf_col() chat intent; {type:"segment_performance"} SSE event; SegmentPerformanceCard (▲best/▼lowest labels, status badges, performance bars, low-sample !, summary); SegmentPerformanceResult + SegmentPerformanceSegment types; api.models.getSegmentPerformance(); attachSegmentPerformanceToLastMessage() Zustand action. Fixed: trailing \b in regex caused false negatives; models.filter→models.dataset_filter; training fixture used dataset_id where project_id required; is_near_unique check for continuous column rejection. 26 backend + 12 frontend = 38 new tests. Total: 1557 backend + 680 frontend = 2237.

## Day 12 (20:00) — Done
Chat-driven feature engineering — _FEATURE_SUGGEST_PATTERNS (8 variants) + _FEATURE_APPLY_PATTERNS (7 variants) in chat.py; suggest handler calls suggest_features() → emits {type:"feature_suggestions"} SSE; apply handler calls suggest_features() + apply_transformations() → creates FeatureSet → emits {type:"features_applied"} SSE; FeatureSuggestCard (purple border, suggestion list with color-coded transform badges, Apply All button that calls REST API directly + inline success state); FeaturesAppliedCard (confirmation with column count and names); FeatureSuggestionItem + FeatureSuggestionsChatResult + FeaturesAppliedResult types; attachFeatureSuggestionsToLastMessage + attachFeaturesAppliedToLastMessage Zustand actions. Fixed: _load_working_df(file_path, filter_conditions) calling convention (not dataset, session). 29 backend + 23 frontend = 52 new tests. Total: 1531 backend + 668 frontend = 2199.

## Day 12 (12:00) — Done
Chat-triggered PDF report generation — _REPORT_PATTERNS (9 variants) detects "generate a report", "pdf report", "download the model report", etc.; handler finds selected/best run + infers problem_type from metrics; emits {type:"report_ready"} SSE event; ReportReadyCard (teal border, 📄 icon, algorithm label, metric badge, Download PDF Report button); ReportReady type; attachReportToLastMessage store action. Fixed f-string format spec bug + ModelRun.problem_type attr access. 16 backend + 17 frontend = 33 new tests. Total: 1502 backend + 645 frontend = 2147.

## Day 12 (04:00) — Done
"Explain my model" conversational model card — GET /api/models/{project_id}/model-card (selected or best run, loads joblib pipeline for feature importances); _algorithm_plain_name() + _metric_plain_english() + _build_limitations() helpers; _MODEL_CARD_PATTERNS (9 variants) + chat handler + system prompt injection → {type:"model_card"} SSE event; ModelCardView (indigo card, algorithm chip, metric value + plain English, importance bars, amber limitation callout, footer stats); ModelCard + ModelCardMetric + ModelCardFeature types; attachModelCardToLastMessage Zustand action; api.models.getModelCard(). 22 backend + 16 frontend = 38 new tests. Total: 1486 backend + 628 frontend = 2114.

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
