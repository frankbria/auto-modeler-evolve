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

## Day 61 (20:00) — Done
**Track D — Custom Prediction Alert Rules via Chat.** Analysts define business-rule-based alerts on live prediction values through conversation — "alert me when predicted revenue is below $100,000", "notify me when confidence drops below 70%", "alert me when predicted class is churn". Distinct from system-level webhook events: these fire when prediction *content* meets a condition.
- `PredictionAlertRule` SQLModel table: `condition_type` (prediction_value|confidence|predicted_class), `condition_op`, `condition_value`, `condition_class`, `trigger_count`, `last_triggered_at`.
- `EVENT_PREDICTION_ALERT` added to `core/webhook.py` ALL_EVENTS — dispatches signed HMAC webhooks on trigger.
- Three regex patterns (7+4+3 NL variants). `_extract_alert_rule_condition()` pure function (class → confidence → numeric, with operator detection + fraction normalization). `_evaluate_alert_rule()` pure function (all 5 ops + all 3 condition types, case-insensitive class match). `_fire_alert_rules()` daemon thread post-prediction.
- REST: `POST/GET/DELETE /api/deploy/{id}/alert-rules`. Chat handler: LIST / DELETE / CREATE branches. SSE: `{type:"alert_rule", action:"created|list|deleted"}`.
- `AlertRuleCard` (three states — violet/slate/rose). `AlertRuleEntry` + `AlertRuleEventResult` TypeScript types; `attachAlertRuleToLastMessage` Zustand action; `getAlertRules`/`createAlertRule`/`deleteAlertRule` API methods. Wired in page.tsx.
- 25 backend + 16 frontend tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track C: Date-aware chronological split via chat — "train with chronological split"
- Track D: API key auth for prediction endpoints
- Track E: End-to-end "lunch break" analyst flow

## Day 61 (12:00) — Done
**Track D — Webhook Management via Chat.** Analysts can register, list, remove, and test webhooks entirely through conversation — no DeploymentPanel navigation required. Four new chat patterns (`_WEBHOOK_CREATE_PATTERNS`, `_WEBHOOK_LIST_CHAT_PATTERNS`, `_WEBHOOK_REMOVE_CHAT_PATTERNS`, `_WEBHOOK_TEST_CHAT_PATTERNS`) with 6–7 NL variants each. Elif chain ensures mutual exclusion with `webhook_history`. Four SSE events + four React cards: `WebhookRegisteredCard` (emerald, 🔔 icon, secret callout with copy button), `WebhookListChatCard` (slate, 🔗, per-hook rows with event badges + relative last-fired), `WebhookRemovedChatCard` (rose, 🗑️, removed URLs), `WebhookTestChatCard` (adaptive border, ⚡, HTTP status + failure guidance). Reuses existing `WebhookConfig` model + `_do_dispatch()` — no new DB tables. 38 backend + 32 frontend = 70 new tests. Backend lint: clean (3 auto-fixed). Frontend build + lint: clean.

**What's next:**
- Track C: Date-aware chronological split via chat — "train with chronological split"
- Track D: API key auth for prediction endpoints
- Track E: End-to-end "lunch break" analyst flow

## Day 54 (12:00) — Done
**CI fix + Track D — Aggregate Production Explanation Analysis via Chat.** Restored Day 43 feature from working tree (was reverted in git). Then implemented aggregate explanation: analysts can ask "what's been driving my predictions?", "aggregate explanation", "which features are influencing my live predictions?", "patterns in my production predictions" and receive an `AggregateExplanationCard` showing feature-level statistics across the last 50 production predictions.
- `compute_aggregate_explanations(pipeline_path, model_path, input_data_list)` pure function in `core/deployer.py`. Loads model/pipeline once. Single-pass aggregation: per-feature avg_abs_contribution, positive_pct, direction_label (mostly positive/negative/mixed), top_driver_pct, sample_count.
- `GET /api/deploy/{id}/aggregate-explanations?n=50` endpoint in `api/deploy.py`. 404 on inactive deployment or no prediction logs.
- `_AGGR_EXPLAIN_PATTERNS` (8 NL variants) + handler in `chat.py`. Guard: `ctx["deployment"]`. Queries last 50 PredictionLogs, injects top features + summary into system_prompt. SSE emit `{type:"aggregate_explanation"}`.
- `AggregateExplanationCard` (violet border, 📊 icon). `DirectionBadge` (sky/rose/gray). `FeatureRow` with progress bar + top-driver badge (amber, shown when ≥30%). Full ARIA. `AggregateExplanationFeature` + `AggregateExplanationResult` TypeScript types; `attachAggregateExplanationToLastMessage` Zustand action; SSE handler + render in `page.tsx`.
- Fixed cross-file test isolation bug: `client` fixtures now patch `db.engine` at module level (vs sys.modules deletion) so `get_session()` always resolves to the test engine via Python's dynamic global lookup.
- 39 backend + 17 frontend = 56 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Webhook notifications on model drift/degradation
- Track C: Date-aware chronological split via chat — "train with chronological split"
- Track E: End-to-end "lunch break" analyst flow

## Day 43 (04:00) — Done
**Track D — Production Prediction Explanation via Chat.** Analysts can ask "explain the last prediction", "why did the model give that result?", "what drove that production prediction", "feature contributions for the most recent API call" and receive a `ProductionExplanationCard` in chat showing per-feature contributions for the most recent live `PredictionLog` record.
- `GET /api/deploy/{deployment_id}/explain-prediction?prediction_id=` in `api/deploy.py`: loads most recent `PredictionLog`, calls existing `explain_prediction()` from `core/deployer.py`, returns `contributions`, `top_drivers`, `summary` + metadata. 404 on missing/inactive deployment or no PredictionLog records.
- `_PROD_EXPLAIN_PATTERNS` (8 NL variant groups) + handler in `chat.py`. Distinct from `_EXPLAIN_ROW_PATTERNS` (training rows by index). Guard: `ctx["deployment"]`. Queries most recent PredictionLog, injects top-3 drivers into system_prompt. SSE emit `{type:"prod_prediction_explanation"}`.
- `ProductionExplanationCard` (violet border, 🔍 icon). Algorithm + problem-type badges + timestamp header. Prediction box with confidence badge. Feature contributions list with sky/rose bars + "val: X" annotations + full aria accessibility. Italic summary. `ProdPredictionContribution` + `ProdPredictionExplanationResult` TypeScript types; `attachProdPredictionExplanationToLastMessage` Zustand action; SSE handler + render in `page.tsx`.
- 35 backend + 22 frontend = 57 new tests. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track C: Date-aware chronological split via chat — "train with chronological split", "use time-based train/test split"
- Track E: End-to-end "lunch break" analyst flow — run the full upload → explore → train → validate → deploy → predict flow as a real user and fix friction points
- Track D: Webhook notifications on model drift/degradation

## Day 42 (20:00) — Done
**Track D — Batch Job Results Analytics via Chat.** Analysts can ask "show me batch results", "latest batch results", "batch prediction summary", "how did the last batch job go" and receive a `BatchJobResultCard` in chat — closing the gap between scheduled batch runs and conversational insight delivery.
- `compute_batch_job_results(output_csv_bytes, problem_type, target_column)` pure function in `core/analyzer.py`. Regression: avg/median/min/max/std + histogram (3–10 bins). Classification: class distribution + pct + avg_confidence (auto-detected, 0–1 proportions converted to %). Falls back to `has_data: False` on empty/malformed CSV.
- `GET /api/deploy/{id}/batch-results` endpoint in `api/deploy.py`: queries most recent successful `BatchJobRun`, returns distribution stats with `has_results`, `job_run_id`, `completed_at`, `row_count`.
- `_BATCH_RESULTS_PATTERNS` (8 NL variants) + handler in `chat.py`. Guard: `ctx["deployment"]`. Reads output CSV, calls pure function, injects summary into system_prompt. SSE emit `{type:"batch_job_results"}`.
- `BatchJobResultCard` (teal border, empty slate state). Regression: 4-stat grid + histogram bars. Classification: horizontal pct bars per class + avg_confidence. `role="region"` accessibility. `BatchJobResultsResult` + `BatchHistogramBin` + `BatchClassDistributionEntry` TypeScript types; `attachBatchJobResultsToLastMessage` Zustand action; SSE handler + render in `page.tsx`.
- 45 backend + 26 frontend = 71 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track C: Date-aware chronological split via chat — "train with chronological split", "use time-based train/test split"
- Track E: End-to-end "lunch break" analyst flow — run the full upload → explore → train → validate → deploy → predict flow as a real user and fix friction points
- Track D: Webhook notifications on model drift/degradation

## Day 42 (12:00) — Done
**Track C — Fairness / Bias Analysis via Chat.** Analysts can ask "is my model biased?", "check fairness by gender", "any disparate impact?", "statistical parity difference", "is my model treating everyone fairly?" and receive a `FairnessCheckCard` inline in chat with Statistical Parity Difference (SPD), Disparate Impact Ratio (DIR), and per-group accuracy/MAE metrics.
- `compute_fairness_metrics()` pure function in `core/validator.py`: classification (SPD + DIR + per-group accuracy), regression (MAE disparity ratio). Status: fair/warning/biased/insufficient_data. Global positive-label detection prevents per-group label drift. Zero-MAE disparity treated as 1.0.
- `GET /api/models/{run_id}/fairness?col=` REST endpoint in `api/validation.py` (400 on unknown col, 400 on high cardinality >50, 404 on unknown run).
- `_FAIRNESS_PATTERNS` (10 NL variants) + `_detect_fairness_col()` longest-match helper in `chat.py`. Handler auto-detects sensitive column; falls back to first low-cardinality categorical column. Fixed `np` shadowing by using `import numpy as _np_fm` inside handler.
- `FairnessCheckCard` (emerald/amber/rose/slate borders). SPD+DIR grid (classification). MAE Disparity section (regression). Per-group table. `role="alert"` for warning/biased. Accessible figcaption.
- 44 backend + 26 frontend = 70 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track C: Date-aware chronological split via chat — "train with chronological split", "use time-based train/test split"
- Track E: End-to-end "lunch break" analyst flow — run the full upload → explore → train → validate → deploy → predict flow as a real user and fix friction points
- Track D: Webhook notifications on model drift/degradation

## Day 42 (04:00) — Done
**Track C — Chat-Triggered Retrain Excluding Weak Features.** Closed the gap between `FeatureSelectionCard` (shows weak features) and taking action. Analysts can now say "retrain without weak features", "drop weak features and retrain", "remove unimportant columns and retrain", etc. and the system identifies low-importance features from the best completed model and launches a new training run with those features excluded.
- `_WEAK_FEAT_RETRAIN_PATTERNS` (8 NL variant groups) in `chat.py`. Handler fires BEFORE `_TRAIN_PATTERNS`; finds best completed `ModelRun`, calls `identify_weak_features()`, launches training with `excluded_features` applied. Mutual exclusion via `training_started_event is not None` check.
- `TrainingStartedResult.excluded_features?: string[]` TypeScript field; `TrainingStartedCard` shows rose "N feature(s) excluded" badge, strikethrough feature list, "without weak features" in description text.
- Pre-existing `ctx["project"]` → `project`, `ctx["runs"]` → `ctx["model_runs"]`, `ctx["conversation"]` → `conversation` bug fixes.
- 20 backend + 8 frontend = 28 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track C: Date-aware chronological split via chat — "train with chronological split", "use time-based train/test split" triggers `split_strategy="chronological"`
- Track E: End-to-end "lunch break" analyst flow — run the full upload → explore → train → validate → deploy → predict → audit → feedback loop as a real user and fix friction points
- Track D: Webhook notifications on model drift/degradation

## Day 41 (20:00) — Done
**Track C — Chat-Triggered Imbalance-Corrected Training.** Closed the user-experience gap: `ClassImbalanceChatCard` (Day 34) told analysts "train with class weighting" but had no handler for that phrase. Now analysts can say "train with class weighting", "apply SMOTE and retrain", "fix the imbalance and train", etc. and training launches with the correct correction applied.
- `_BALANCE_TRAIN_PATTERNS` (8 NL variant groups) + `_detect_balance_strategy()` helper in `chat.py`. Handler fires BEFORE `_TRAIN_PATTERNS`; passes `imbalance_strategy` to `_train_in_background()`. Classification only — regression gets a plain-English "N/A" response.
- `training_started_event` extended with `imbalance_strategy` field (echoed through existing SSE emitter unchanged).
- `TrainingStartedResult.imbalance_strategy?` TypeScript field; `TrainingStartedCard` shows strategy badge (blue=Class Weighting, violet=SMOTE, amber=Threshold) + strategy in description text.
- 26 backend + 5 frontend = 31 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track C: Feature selection automation via chat — "drop weak features", "remove unimportant columns" triggers dropping near-zero importance features and retraining
- Track C: Date-aware chronological split via chat — "train with chronological split", "use time-based train/test split" triggers `split_strategy="chronological"`
- Track E: End-to-end "lunch break" analyst flow — run the full upload → explore → train → validate → deploy → predict → audit → feedback loop as a real user and fix friction

## Day 41 (12:00) — Done
**Track D — Feedback Accuracy Report via Chat.** Analysts can ask "how accurate have my predictions been?", "show me feedback accuracy report", "how many predictions were correct", "how well did my model perform in production", etc. and receive a `FeedbackAccuracyCard` in chat — closing the loop between model predictions and real-world outcomes using recorded FeedbackRecords.
- `compute_feedback_accuracy_report(feedback_records, prediction_logs_map, problem_type)` pure function in `core/analyzer.py`: regression → MAE/pct_error/avg_actual/verdict; classification → accuracy/accuracy_pct/correct_count/incorrect_count/verdict; both → ISO-week weekly_trend, trend_direction (improving/stable/declining via first-half vs second-half comparison with 5% threshold).
- `_FEEDBACK_ACCURACY_PATTERNS` (10 NL variant groups) in `chat.py`. Guard: `ctx["deployment"]`. Queries FeedbackRecord by deployment_id, pairs with PredictionLog, calls pure function, injects summary+verdict into system_prompt.
- `FeedbackAccuracyCard`: empty/feedback-only/computed states, verdict badge (emerald/green/amber/red), regression MAE/% Error/Matched grid, classification Accuracy %/Correct/Incorrect grid, trend direction row, Recharts LineChart for weekly trend, adaptive border color.
- `FeedbackAccuracyReportResult` + `FeedbackAccuracyWeekly` TypeScript types; `attachFeedbackAccuracyReportToLastMessage` Zustand action; SSE handler + render in `page.tsx`.
- 42 backend + 21 frontend tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Webhook notifications on model drift/degradation — "alert me when predictions shift" or "set up drift alerts"
- Track D: Deployment versioning + rollback — "roll back to last model version", "compare v1 vs v2"
- Track E: End-to-end "lunch break" analyst flow — upload → chat → train → deploy → predict → audit → feedback loop

## Day 41 (04:00) — Done
**Track D — Confidence Trend Analysis via Chat.** Analysts can ask "how is my model confidence trending?", "are my predictions getting less reliable?", "confidence over time", etc. and receive a `ConfidenceTrendCard` in chat — a temporal chart showing whether the model is becoming more or less reliable day by day.
- `compute_confidence_trend(logs, window_days, now_utc)` pure function in `core/analyzer.py`: OLS slope trend detection (improving/stable/declining), daily_stats, peak/low day, summary.
- `GET /api/deploy/{id}/confidence-trend?window=<days>` REST endpoint: 404 for unknown/inactive; returns full trend dict + `deployment_id`.
- `_CONFIDENCE_TREND_PATTERNS` (8 NL variant groups) in `chat.py`. Guard: `ctx["deployment"]`.
- `ConfidenceTrendCard`: adaptive border/badge per direction, stats grid, Recharts LineChart sparkline, trend rate label, summary.
- `ConfidenceTrendResult` + `ConfidenceTrendDailyStat` TypeScript types; `attachConfidenceTrendToLastMessage` Zustand action; SSE handler + render in `page.tsx`.
- 34 backend + 15 frontend tests. Backend lint: clean. Frontend build: clean.

## Day 40 (20:00) — Done
**Track D — Prediction Audit Report via Chat.** Analysts can ask "deployment audit", "how is my deployment doing?", "model monitoring report", "show me a deployment summary", etc. and receive a `PredictionAuditCard` in chat — a holistic health digest combining volume, confidence distribution, SLA status, and quota in one card.
- `compute_prediction_audit(logs, deployment, now_utc)` pure function in `core/analyzer.py`: volume counts (today/7d/30d/total), confidence distribution (high/medium/low %), latency percentiles (p50/p95/avg), SLA alert flag (p95>500ms), quota tracking (used=count_30d, pct, enabled), overall status (critical/warning/healthy).
- `GET /api/deploy/{id}/prediction-audit` REST endpoint: 404 for unknown/inactive; returns full audit dict + `deployment_id`.
- `_PRED_AUDIT_PATTERNS` (8 NL variant groups) in `chat.py`. Guard: `ctx["deployment"]`.
- `PredictionAuditCard`: adaptive border per status, StatusBadge, volume grid, confidence bars, latency section with SLA badge, quota progress bar, empty state.
- `PredictionAuditResult` TypeScript type; `attachPredictionAuditToLastMessage` Zustand action; SSE handler + render in `page.tsx`.
- 45 backend tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Webhook notifications on model drift/degradation — "alert me when predictions shift" or "set up drift alerts"
- Track D: Deployment versioning + rollback — "roll back to last model version", "compare v1 vs v2"
- Track E: End-to-end "lunch break" analyst flow — upload → chat → train → deploy → predict → audit

## Day 40 (12:00) — Done
**Track D — Recent Predictions Table via Chat.** Analysts can ask "show me recent predictions", "what were the last 10 predictions", "list recent API calls", "browse predictions", "prediction log table", etc. and receive a `RecentPredictionsCard` inline in chat — a live, inspectable table of actual prediction log entries.
- `_RECENT_PRED_LOG_PATTERNS` (8 NL variant groups) + `_extract_recent_pred_n()` helper. Mutual exclusion with CSV export event.
- `GET /api/deploy/{id}/recent-predictions?n=N` REST endpoint: returns last N rows DESC with `input_summary` (≤3 k-v pairs from `input_features` JSON), confidence as %, and `total_all_time` count.
- `RecentPredictionsCard`: relative time, M/k number formatting, colour-coded confidence + latency badges, A/B variant badge, key-input badge chips, CSV download link, empty state, sr-only accessibility captions.
- `RecentPredictionsResult` TypeScript type; `attachRecentPredictionsToLastMessage` Zustand action; SSE handler + render in `page.tsx`.
- 46 backend + 30 frontend = 76 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Prediction SLA / latency monitoring — "is my API slow?", "show p95 latency" shows p50/p95/p99 latency chart + threshold alert
- Track D: Webhook notifications on model drift/degradation — "alert me when predictions shift"
- Track E: End-to-end "lunch break" analyst flow — upload → chat → train → deploy → predict → inspect recent predictions

## Day 40 (04:00) — Done
**Track D — Prediction Log CSV Export via Chat.** Analysts can ask "export prediction history", "download prediction logs", "save predictions as csv", "get my prediction history", etc. and receive a `PredictionLogExportCard` inline in chat with a direct download link.
- REST endpoint `GET /api/deploy/{id}/prediction-logs/export`: streams CSV with all `input_features` columns dynamically extracted from JSON blobs, plus `id, created_at, prediction, confidence, response_ms`. `Content-Disposition: attachment` header.
- `_PRED_LOG_EXPORT_PATTERNS` (8 NL variant groups) in `chat.py`. Guard: `ctx["deployment"]`.
- `PredictionLogExportCard` (emerald border, ⬇ icon): count badge, CSV badge, date range (first/last prediction), `<a download>` link, empty state when no predictions.
- `PredictionLogExportResult` TypeScript type; `attachPredictionLogExportToLastMessage` Zustand action; SSE handler + render wired in `page.tsx`.
- 35 backend + 15 frontend = 50 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Prediction SLA / latency monitoring — "is my API slow?", "show p95 latency" shows p50/p95/p99 latency chart + threshold alert
- Track D: Webhook notifications on model drift/degradation — "alert me when predictions shift"
- Track C: Class imbalance detection + handling (SMOTE / class weights / threshold tuning)
- Track E: Run the "lunch break" flow end-to-end as a real analyst; fix any new friction points

## Day 39 (20:00) — Done
**Track D — Prediction Usage Pattern Analysis via Chat.** Analysts can ask "when is my model busiest?", "peak traffic hours for my endpoint", "hourly usage pattern", "maintenance window for my api", etc. `compute_usage_pattern()` pure function + `GET /api/deploy/{id}/usage-pattern` REST endpoint. `_USAGE_PATTERN_PATTERNS` (8 NL variants) in `chat.py`. `UsagePatternCard` with 24-bar hour chart + 7-bar day chart, busiest period callout, maintenance window suggestion from quiet hours. 39 backend + 17 frontend = 56 new tests. Lint: clean. Build: clean.

**What's next:**
- Track D: Prediction SLA / latency monitoring — show p50/p95/p99 prediction latency in deployment panel or via "is my API slow?" chat query
- Track D: Webhook notifications on model drift/degradation — "alert me when predictions shift"
- Track C: Class imbalance detection + handling (SMOTE / class weights / threshold tuning)
- Track E: Run the "lunch break" flow end-to-end as a real analyst; fix any new friction points

## Day 39 (12:00) — Done
**Track D — Deployment Cost Estimate via Chat.** Forwards-looking capacity planning: analysts can ask "how much would 1000 predictions cost?", "estimate prediction cost", "how many users can my model handle?", or "prediction capacity planning" and receive an inline `CostEstimateCard` with quota impact bar, daily capacity, days-to-serve, and recommended rate limit. `_COST_ESTIMATE_PATTERNS` (8 NL variants), `_extract_cost_n()` (k/m suffixes, comma formatting). 34 backend + 22 frontend = 56 new tests. Lint: clean. Build: clean.

## Day 38 (20:00) — Done
**Track D — Proactive Covariate Drift Alert via Chat.** Complement to Day 38 12:00's reactive `ProductionInputDistributionCard`: proactively surfaces input drift alerts when an analyst asks "are my inputs drifting?" or on workspace load when a deployed model has significant OOR inputs. 41 backend + 24 frontend tests, all passing. Lint: clean. Build: clean.

## Day 38 (12:00) — Done
**Track D — Production Input Distribution Chat Card.** Analysts can now ask "what values are users sending to my model?", "show production input distribution", or "are my production inputs in range?" and receive a `ProductionInputDistributionCard` inline in chat — per-feature production stats vs training ranges, with out-of-range and unseen-category detection.
- `_PROD_INPUT_DIST_PATTERNS` regex (8 NL variants) in `chat.py`. Guard: `ctx["deployment"]`.
- Handler: queries last 500 `PredictionLog` records, parses `input_features` JSON, aggregates numeric (mean/min/max vs training range from PredictionPipeline.feature_ranges) and categorical (top-5 value counts + unseen detection) features (capped at 10).
- `ProductionInputDistributionCard` (sky-blue border, 📊 icon): amber tint for OOR numeric, rose tint for unseen categorical, empty state, legend.
- 21 backend + 15 frontend = 36 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track E: Run the "lunch break" flow end-to-end as a real analyst; audit friction in the VP-sharing flow
- Track D: Covariate drift alert — proactively notify when production input means shift significantly from training baselines
- Track C: Feature selection automation — suggest dropping near-zero-importance features to simplify models

## Day 38 (04:00) — Done
**Track C — Local Explanation Chat Card (Feature Contribution Waterfall).** Analysts can now ask "explain this prediction", "what drove this result?", "show SHAP values for row 5", or "why did the model predict that?" and receive a `LocalExplanationCard` inline in chat — a waterfall chart showing each feature's contribution to the selected row's prediction.
- `_EXPLAIN_ROW_PATTERNS` regex (9 NL variants) + `_extract_row_index()` helper in `chat.py`. Guard: `ctx["model_runs"]` AND `ctx["dataset"]` AND `ctx["feature_set"]` AND `not pdp_event`.
- Handler: finds selected/best completed run; loads CSV; applies transformations; builds X/y; calls `explain_single_prediction()` from `core/explainer.py` (existing); caps contributions at 12; injects top-3 drivers into system prompt.
- Bugfix: `prepare_features` returns `(X, y, LabelEncoder|None)` — handler was passing `None` as feature names; fixed to use `_le_feat_cols` directly.
- `LocalExplanationCard` (violet border, 🔍 icon): Row/Algorithm/Target/Correct-Wrong badges; Actual vs Predicted side-by-side; blue/rose bars proportional to contribution magnitude; figcaption summary.
- `LocalExplanationContribution` + `LocalExplanationResult` TypeScript types; `attachLocalExplanationToLastMessage` Zustand action; SSE handler + render wired in `page.tsx`.
- 41 backend tests (unit + integration). Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Input feature distribution in production — "what values are users sending to my model?" shows distribution of production inputs vs training ranges
- Track D: Prediction SLA / latency monitoring — show p50/p95/p99 prediction latency in deployment panel
- Track E: Run the "lunch break" flow end-to-end as a real analyst; audit friction

## Day 37 (20:00) — Done
**Track C — Confusion Matrix Chat Card.** "Show me the confusion matrix" / "where does my model make mistakes?" / "precision per class" now renders a `ConfusionMatrixChatCard` inline in chat. Enhanced `compute_confusion_matrix()` with `per_class_metrics` (precision/recall/f1/support per class) and `most_confused_pair` (most common misclassification). Classification-only guard; loads fitted model from joblib. 28 backend + 18 frontend = 46 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track C: SHAP waterfall via chat — "explain this specific prediction" shows individual feature contributions (SHAP values) as a waterfall chart for the selected training row
- Track D: Input feature distribution in production — "what values are users sending to my model?" shows distribution of production inputs vs training ranges
- Track E: Run the "lunch break" flow end-to-end as a real analyst; audit friction

## Day 37 (16:00) — Done
**Track C — Ensemble Training via Chat.** The ensemble recommendation card (Day 36 04:00) told analysts to say "train a voting ensemble" to proceed — but that phrase had no handler. Fixed: `_ENSEMBLE_TRAIN_PATTERNS` regex (8 NL variants) + `_STACKING_RE` sub-detector. Handler fires before `_TRAIN_PATTERNS` to prevent double-fire; selects `voting_regressor`/`stacking_regressor`/`voting_classifier`/`stacking_classifier` based on problem type and stacking keyword; creates `ModelRun(status="pending")` and starts `_train_in_background` thread.
- Bug fix: `test_monitoring_alerts.py::TestChatAnalyticsIntent` was using stale event type `"analytics"` — updated to `"prediction_analytics_chat"` (3 failing + 2 negative checks fixed).
- 22 new backend tests in `test_ensemble_train_chat.py`. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Deployment cost estimate via chat ("how much would 1000 predictions cost?", "estimate my monthly prediction cost") — surfacing the rate limit and quota configs in terms of business cost.
- Track D: Prediction SLA / latency monitoring — show p50/p95/p99 prediction latency in the deployment panel.
- Track E: Run the "lunch break" flow end-to-end as a real analyst; audit friction in the VP-sharing flow.

## Day 37 (12:00) — Done
**Track D — Prediction Log Analytics Chat Card.** Upgraded the thin `_ANALYTICS_PATTERNS` stub into a full analytics card. Analysts can ask "how many predictions have been made?", "show prediction analytics", or "prediction volume report" and receive a `PredictionAnalyticsChatCard` with 14-day daily sparkline, 7d/30d/today stats, peak day, class distribution (classification), avg prediction (regression).
- Bug fixed: handler was reading `model_run.problem_type` (field doesn't exist on `ModelRun`) — fixed to `deployment.problem_type`.
- 16 backend + 17 frontend = 33 new tests. Backend lint: clean. Frontend build: clean.

## Day 37 (04:00) — Done
**Audit + bug fix session.** Discovered three fully-implemented but undocumented features (Learning Curve Analysis, Developer SDK Generation, Cross-Project Portfolio Overview) and added them to spec.md. Fixed active-filter bug in learning curve chat handler (`pd.read_csv` → `_load_working_df`).

## Day 36 (20:00) — Done
**Track C — CV Score Distribution Chat Card.** Analysts can now ask "how consistent is my model?", "show fold scores", "cv variance", or "is my model stable?" and receive an inline `CvScoreDistributionCard` showing per-fold CV scores as labeled bars, mean ± std, CoV%, 95% CI, and a stability classification (stable/moderate/variable).
- `_CV_SCORE_DIST_PATTERNS` regex (8 NL variants covering consistency, fold scores, cv variance, stability checks).
- Handler in `send_message()`: calls `run_cross_validation()`, classifies by CoV (std/mean) — <5% stable, 5–15% moderate, >15% variable.
- `CvScoreDistributionCard` (emerald/amber/rose border by stability, 📊 icon, per-fold bars, stats grid, 95% CI, figcaption).
- `CvScoreDistributionResult` TypeScript type; `cv_score_distribution?` on `ChatMessage`; Zustand action; SSE handler + render in `page.tsx`.
- 13 backend + 14 frontend = 27 new tests. Ruff lint: clean. Frontend build: clean.

## Day 36 (12:00) — Done
**Track C — Hyperparameter Tuning Chat Card.** Analysts can now say "tune my model", "go ahead and tune it", "optimize hyperparameters", or "run the tuning" and receive an inline `TuningChatCard` showing before/after metrics, best params, and improvement percentage — all within the conversation, without navigating to the Models panel.
- `_EXPLICIT_TUNE_RE` constant (unambiguous vocabulary: tune/tuning/optimize/hyperparameter/grid-search/best params) guards inline tuning from generic "improve my model" phrases (those still route to `_IMPROVEMENT_PATTERNS`).
- `tune_chat_event` block in `send_message()`: loads CSV, prepares X/y, creates ModelRun, calls `tune_model()` (10-iter RandomizedSearchCV, 3-fold CV), updates run to done, emits `{type:"tune_chat"}` with original_metrics, tuned_metrics, best_params, improved, improvement_pct.
- `TuningChatCard` (emerald border when improved, amber when unchanged, slate when not-tunable, 🔧 icon): before/after metrics table with delta column, best params in monospace, Improved/Unchanged badge, ±% badge.
- `TuningChatResult` TypeScript type; `tune_chat?` on `ChatMessage`; `attachTuneChatToLastMessage` Zustand action; SSE handler + render wired in `page.tsx`.
- 20 backend + 21 frontend = 41 new tests. Ruff lint: clean. Frontend build: clean.

## Day 36 (04:00) — Done
**Track C — Ensemble Method Recommendation via Chat.** Analysts can now ask "should I use an ensemble?", "best ensemble for this problem?", "voting classifier", "stacking regressor", or "can an ensemble improve my accuracy?" and receive an `EnsembleRecommendationCard` inline in chat. The card explains what ensembles are, recommends stacking or voting based on dataset size and number of completed runs, and shows both options with plain-English descriptions and training prompts. No training is triggered — "explain before executing".
- `_ENSEMBLE_PATTERNS` (8 NL variants) + handler in `chat.py`. Guards on `ctx["model_runs"]`. Recommends stacking (≥200 rows AND ≥2 runs) or voting. Emits `{type:"ensemble_recommendation"}` SSE event.
- `EnsembleRecommendationCard` (violet border, 🧩 icon): problem-type/score/algorithm badges, "What is an ensemble?" callout, summary, two option rows with Recommended/Easy/Medium badges and plain-English prompts. `EnsembleOption` + `EnsembleRecommendationResult` types; `attachEnsembleRecommendationToLastMessage` Zustand action; SSE wired in `page.tsx`.
- 16 backend + 18 frontend = 34 new tests. Total: 3370 backend + 1749 frontend = 5119, all passing. Backend lint: clean. Frontend build: clean.

## Day 35 (20:00) — Done
**Track D — Deployment Version Comparison via Chat.** Analysts can now ask "did my retrain improve?", "compare my deployment versions", or "is the new version better?" and receive a `DeploymentVersionComparisonCard` inline in chat showing per-metric deltas between the current and previous deployment version. Closes the "was this retrain worth it?" conversational gap.
- `_VERSION_COMPARE_PATTERNS` (8 NL variants) + handler in `chat.py`. Guards on `ctx["deployment"]` and 2+ `DeploymentVersion` records. Computes delta/pct_change/direction/improved for r2, accuracy, mae, rmse, f1, precision, recall (respecting higher_is_better — MAE/RMSE lower is better). Algorithm-change detection. <2 versions emits has_comparison=False with onboarding guidance.
- `DeploymentVersionComparisonCard`: border by outcome (emerald/rose/amber/slate), version range badge, improved/declined badges, date info, algorithm-changed note, metric table with directional arrows, summary footer, MAE/RMSE note. `DeploymentVersionComparisonResult`/`VersionMetricDiff` types; `attachVersionComparisonToLastMessage` Zustand action; SSE wired in `page.tsx`.
- 13 backend + 19 frontend = 32 new tests. Total: 3155 backend + 1712 frontend = 4867, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track E: Run the "lunch break" flow end-to-end as a real analyst; audit friction in the VP-sharing flow.
- Track D: Webhook notifications on model drift/degradation — "alert me when predictions shift".
- Track D: Prediction SLA/latency monitoring — track p50/p95 per endpoint, surface in chat ("is my API slow?").
- Track C: Cross-validation score distribution — show CV fold variance in training results so analysts know if model is consistent.

## Day 35 (12:00) — Done
**Track D — Service Export Chat Integration.** Analysts can now say "package my model", "export my model as a service", or "deploy this elsewhere" and receive a `ServiceExportChatCard` inline in chat with a direct ZIP download link — no navigation to the deployment panel required. Closes the developer hand-off story through pure conversation.
- `_SERVICE_EXPORT_PATTERNS` (8 NL variants) + handler in `chat.py`. Guards on `ctx["deployment"]`; extracts algorithm/target/problem_type/feature_count from Deployment record; emits `{type:"service_export", service_export:{deployment_id, algorithm, target_column, problem_type, feature_count, download_url, included_files}}` SSE event.
- `ServiceExportChatCard` (indigo border, 📦 icon): ZIP-download badge, problem-type badge, formatted algorithm name, included-files list with per-file plain-English annotations, quickstart code block (pip install + uvicorn), feature count, `<a download>` link with aria-label. Zustand `attachServiceExportToLastMessage`; SSE wired in `page.tsx`.
- 13 backend + 18 frontend = 31 new tests. Total: 3142 backend + 1693 frontend = 4835, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track C: Ensemble methods via chat — "what's the best ensemble for this problem?" — VotingClassifier/Regressor, StackingClassifier/Regressor.
- Track E: Run the "lunch break" flow end-to-end as a real analyst; audit friction in the VP-sharing flow.
- Track D: Deployment version comparison — "how does the current model compare to last week?" — diff of metrics between deployment versions.

## Day 35 (04:00) — Done
**Track B/E — Executive Briefing Generator.** Analysts can now say "write a briefing for my VP" or "create an executive summary" and receive a polished `ExecutiveBriefingCard` inline in chat — closing the "share results with leadership" gap.
- `generate_executive_briefing()` pure function in `core/storyteller.py`: assembles plain-English metric explanations (quality tiers: excellent/good/moderate/developing), algorithm descriptions, 4-section briefing (What We Analyzed, How Accurate Is It?, What This Means, Deployment Status), one-sentence headline summary, and action items.
- `GET /api/projects/{id}/executive-briefing` REST endpoint; `_BRIEFING_PATTERNS` (8 NL variants) + handler + SSE `{type:"executive_briefing"}` in `chat.py`.
- `ExecutiveBriefingCard` (emerald border, 📋 icon): algorithm badge, metric badge (color-coded by quality), italic summary, 4 sections, Recommended Actions list, prediction dashboard link OR deploy-prompt, copy-to-clipboard button.
- 22 backend + 16 frontend = 38 new tests. Total: 3129 backend + 1675 frontend = 4804, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track C: Ensemble methods via chat — "what's the best ensemble for this problem?" — VotingClassifier/Regressor, StackingClassifier/Regressor, with plain-English explanation of which base models voted and how confident each was.
- Track D: Export as self-contained prediction service (ZIP + uvicorn) — "package my model for deployment anywhere".
- Track E: Run the "lunch break" flow end-to-end as a real analyst; audit friction in the VP-sharing flow.

## Day 34 (12:00) — Done
**Track D — Cross-Deployment Webhook Health Summary via Chat.** Analysts can now ask "are my webhooks working?", "any webhook failures?", "webhook health", or "check webhook status" and receive a `WebhookHealthSummaryCard` inline in conversation showing the health of every webhook across all active deployments in the project.
- `_WEBHOOK_HEALTH_PATTERNS` (8 NL variants) + mutual-exclusion guard (`not _WEBHOOK_HISTORY_PATTERNS.search(...)`) so health and history cards don't both fire on the same message.
- Handler aggregates `WebhookConfig` + `WebhookEvent` rows per deployment: per-webhook stats (total events, failed events, success rate, last event, status: healthy/warning/critical/no_events), per-deployment rollup, overall project status (healthy/warning/critical/no_events/no_webhooks).
- SSE `{type:"webhook_health_summary"}`. `WebhookHealthSummaryCard` (border color adapts: emerald=healthy, amber=warning, red=critical, slate=no_events/no_webhooks): 🔗 icon, overall status badge + webhook count badge, summary paragraph, per-deployment section with per-webhook URL + event stats + status badge, stats footer, guidance footer.
- 16 backend + 19 frontend = 35 new tests. Total: 3107 backend + 1659 frontend = 4766, all passing. Backend lint: clean. Frontend build + tests: clean.

**What's next:**
- Track C: Ensemble methods (VotingClassifier/VotingRegressor, StackingClassifier/StackingRegressor) via chat — "what's the best ensemble for this problem?".
- Track D: Export as self-contained prediction service (ZIP + uvicorn) — "package my model for deployment anywhere".
- Track E: Run the "lunch break" flow end-to-end as a real analyst; audit friction in the VP-sharing flow.

## Day 34 (04:00) — Done
**Track C — Class Imbalance Detection via Chat.** Wired the existing `detect_class_imbalance()` pure function into the chat pipeline. Analysts can now ask "is my data imbalanced?", "my minority class is rare", "should I use SMOTE?" and receive a `ClassImbalanceChatCard` inline in conversation showing the actual class distribution and a concrete strategy recommendation.
- `_CLASS_IMBALANCE_PATTERNS` (10 NL variants) + handler in `chat.py`. Root-cause bug fixed: `body.project_id` → `project_id` (path parameter); `ChatMessage` Pydantic model does not expose project_id — was silently swallowed by `except Exception: pass`, leaving `class_imbalance_event = None`.
- SSE `{type:"class_imbalance_check"}`. `ClassImbalanceChatCard` (rose/emerald/muted states): `DistributionBar` sub-component (minority bars rose-colored), strategy panel (class_weight/smote/threshold/none with hints), "Go to Models tab" CTA. Zustand `attachClassImbalanceCheckToLastMessage`; SSE wired in `page.tsx`.
- 22 backend + 14 frontend = 36 new tests. Total: 3091 backend + 1640 frontend = 4731, all passing. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track C: Ensemble methods (VotingClassifier/VotingRegressor, StackingClassifier/StackingRegressor) via chat — "what's the best ensemble for this problem?".
- Track D: Cross-deployment webhook health dashboard (all webhook failures across projects at once).
- Track E: Run the "lunch break" flow end-to-end as a real analyst; audit friction in the VP-sharing flow.

## Day 33 (20:00) — Done
**Track D — Webhook Event History via Chat.** Closed the gap between webhooks firing silently and analysts having visibility into their integration health. Analysts can now ask "what webhooks fired recently?" or "show webhook history" and receive a `WebhookHistoryCard` inline in conversation showing a per-event timeline.
- `WebhookEvent` SQLModel table persists each dispatch attempt (webhook_id, deployment_id, event_type, fired_at, status_code). `_dispatch_in_thread()` in `core/webhook.py` writes a row after each HTTP call.
- `GET /api/deploy/{id}/webhook-history` REST endpoint returns `{total, events, summary}`.
- `_WEBHOOK_HISTORY_PATTERNS` (8 NL variants) + handler + SSE `{type:"webhook_history"}` in `chat.py`. Bug fixed: missing `from models.webhook_config import WebhookConfig` local import + stale debug print.
- `WebhookHistoryCard` (slate border, 🔔 icon): event count badge, summary, per-event rows with color-coded badges, URL, timestamp, HTTP status badge (200 OK / Error). Zustand `attachWebhookHistoryToLastMessage`; SSE wired in `page.tsx`.
- 18 backend + 15 frontend = 33 new tests. Total: 3069 backend + 1626 frontend = 4695, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track C: Class imbalance detection + handling (SMOTE / class weights / threshold tuning), or ensemble methods (voting + stacking).
- Track E: Run the "lunch break" flow end-to-end as a real analyst; audit friction in the VP-sharing flow.
- Track D: Cross-deployment webhook health dashboard (view all webhook failures across projects at once).

## Day 33 (12:00) — Done
**Track D — A/B Test Chat Integration.** Wired the existing champion-challenger A/B testing infrastructure into chat. Analysts can now ask "how is my A/B test going?", "is the challenger doing better?", "promote the challenger", or "end the A/B test" and receive an `ABTestChatCard` inline in conversation — no navigation to the Deployment panel required.
- `_AB_TEST_PATTERNS` (8 NL variants) + `_AB_PROMOTE_RE` + `_AB_END_RE` in `chat.py`. Handler: status → `_ab_test_response()` with split/metrics/significance; promote → inline `promote_challenger()` replication; end → `is_active=False`; none → guidance message. SSE `{type:"ab_test_result"}`.
- `ABTestChatCard` (purple border, ⚗️ icon): status view with split bar + MetricsColumn + SignificanceRow; promoted/ended/none confirmation views. `ABTestChatResult` type; Zustand action; SSE wired in page.tsx.
- Note: one-deployment-per-project design means A/B tests require two separate projects as champion/challenger — this is expected behavior documented in the test.
- 16 backend + 19 frontend = 35 new tests. Total: 3051 backend + 1611 frontend = 4662, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: Webhook event history via chat ("what webhooks fired recently?") — the webhook system is built but has no chat-triggered history view.
- Track C: Class imbalance detection + handling (SMOTE / class weights), or ensemble methods (voting/stacking).
- Track E: Run the "lunch break" flow end-to-end as a real analyst; fix friction points.

## Day 32 (20:00) — Done
**Track D — Quota Alert Notifications.** Closes the gap between having a monthly quota configured and knowing before your VP's dashboard starts returning 429 errors. Analysts can now say "alert me when I hit 80% of my quota" or "set quota alert at 90%" and AutoModeler will fire registered webhooks exactly once the moment usage first crosses the threshold.
- `quota_alert_threshold_pct` field on `Deployment` (inline SQLite migration). `EVENT_QUOTA_ALERT` added to `webhook.py` `ALL_EVENTS`. `_check_and_fire_quota_alert()` pure helper fires only when `used == ceil(quota * threshold / 100)` — no alert spam on subsequent predictions. Runs in a background daemon thread after each prediction commit.
- `PUT /api/deploy/{id}/quota-alert` endpoint (1-99 valid; 0/null removes; 422 for invalid). `GET /api/deploy/{id}/quota-status` extended with `quota_alert_threshold_pct` + `quota_alert_enabled`. `_QUOTA_ALERT_PATTERNS` (8 NL variants) + handler in `chat.py`; emits `{type:"quota_alert_config"}` SSE event. `QuotaAlertCard` (orange border, 🔔 icon): threshold badge, explanation, usage bar. Fixed pre-existing `test_all_events_constant_has_three_entries` to `has_expected_entries` (now 4 event types).
- 21 backend + 16 frontend = 37 new tests. Total: 3010 backend + 1577 frontend = 4587, all passing. Backend lint: clean. Frontend lint: clean.

**What's next:**
- Track E: run the "lunch break" flow as a real business analyst; look for friction in the VP-sharing flow.
- Track C: feature interaction detection (interaction terms between top features), or confidence interval improvements for classification.
- Track D: cross-deployment quota dashboard (analyst view of quota usage across all their projects).

## Day 32 (12:00) — Done
**Track D — SLA Latency Monitoring via chat.** Closes the gap between the deployment panel's `SlaMonitorCard` and the conversational interface. Analysts can now ask "how fast is my model?", "show me the prediction latency", or "p95 latency?" and receive an `SlaCard` inline in chat showing p50/p95/p99 percentiles, avg latency, sample count, a daily sparkline, and an alert when p95 > 500ms — without navigating away from the conversation.
- `_SLA_PATTERNS` (10 NL variants) in `chat.py`; handler queries `PredictionLog.response_ms`, computes percentiles, groups by day for sparkline, emits `{type:"sla_metrics"}` SSE event.
- `SlaCard` (sky border, ⚡ icon): empty state, p50/p95/p99 grid, avg/count row, Recharts sparkline, `role="alert"` message when p95 > 500ms.
- 15 backend + 19 frontend = 34 new tests. Total: 2989 backend + 1561 frontend = 4550, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: advanced quota alerting, export as self-contained prediction service (ZIP).
- Track C: ensemble methods (voting + stacking), date-aware chronological train/test splits.
- Track E: run the "lunch break" flow as a real business analyst; fix remaining friction.

## Day 32 (04:00) — Done
**Track C — Calibration Check via chat.** Closes the gap between the Validation panel's Calibration sub-tab and the conversational interface. Analysts can now ask "how well-calibrated is my model?" or "brier score?" and receive a `CalibrationCheckCard` with the reliability diagram inline in chat — surfacing data that was already computed at training time but inaccessible through conversation.
- `_CALIBRATION_CHECK_PATTERNS` (8 NL variants) in `chat.py`; handler loads model run metrics, extracts is_calibrated/brier_score/calibration_curve, applies quality bucket, injects narration hint, emits `{type:"calibration_check"}` SSE event.
- `CalibrationCheckCard` (violet border, 🎯 icon): quality badge (excellent/good/needs attention), Brier score, reliability BarChart with perfect-calibration diagonal, calibration note.
- 13 backend + 15 frontend = 28 new tests. Total: 2974 backend + 1542 frontend = 4516, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: deployment SLA dashboard improvements, advanced quota alerting.
- Track C: automated feature interaction suggestions after training, class imbalance detection improvements.
- Track E: run the "lunch break" flow as a real business analyst; fix any remaining friction.

## Day 31 (20:00) — Done
**Track C — Partial Dependence Plots (PDP) via chat.** Closes the "how does feature X affect predictions on AVERAGE across all customers?" analyst question. Unlike sensitivity analysis (which fixes all other features at training means), PDP averages over the actual training distribution — statistically more accurate for datasets where features are correlated.
- `compute_partial_dependence()` pure function in `core/explainer.py` — sweeps feature across p5-p95 grid, averages predictions over all training rows; regression/binary/multiclass variants.
- `GET /api/models/{run_id}/partial-dependence?feature=&steps=20` endpoint in `api/validation.py`.
- `_PDP_PATTERNS` (8 NL variants) + `_detect_pdp_feature()` in `chat.py`; handler picks best/selected run, injects trend summary into system prompt, emits `{type:"partial_dependence"}` SSE event.
- `PartialDependenceCard` (purple border, 📉 icon): trend badge, std band chart, multiclass per-class curves.
- 29 backend + 15 frontend = 44 new tests. Total: 2961 backend + 1527 frontend = 4488, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: cross-project model comparison improvements, deployment health alerting at-scale.
- Track C: confidence calibration curves in chat ("how well-calibrated are my model's confidence scores?"), automated feature interaction suggestions after training.
- Track E: run the "lunch break" flow as a real business analyst; fix any remaining friction.



## Day 31 (12:00) — Done
**Track D — Prediction Input Guard Rails.** Closes the "Not a black box" gap for the VP-facing prediction dashboard: when a user enters a feature value outside the model's training distribution (numeric too high/low, or unseen category), the prediction response now includes `guard_rail_warnings` describing exactly what's out of bounds and why confidence may be lower.
- `feature_ranges` field on `PredictionPipeline` (backward-compatible, computed at build time): numeric → `{p5, p95, min, max}`; categorical → `{known_categories: [...]}`.
- `validate_prediction_inputs(provided_features, pipeline)` pure function in `core/deployer.py`; checks ONLY user-supplied values (not auto-filled defaults). Three severity levels: `out_of_range` (p5–p95 breach), `extreme_outlier` (min/max breach), `unknown_category`.
- `predict_single()` accepts optional `provided_features` kwarg; `make_prediction()` passes `provided_features=input_data`; chat inline-pred handler passes extracted features before defaults merge.
- `GuardRailWarning` TypeScript interface; `guard_rail_warnings?` added to `InlinePredictionResult` and `PredictionResult`. `InlinePredictionCard` shifts to amber border + warning rows (`role="alert"`) when warnings present. `predict/[id]/page.tsx` shows amber warning callout in result section.
- 17 backend + 17 frontend = 34 new tests. Total: 2932 backend + 1512 frontend = 4444, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: cross-project model comparison improvements, deployment health alerting improvements.
- Track C: automated feature selection (drop near-zero-importance features), class imbalance detection in training.
- Track E: run the "lunch break" flow as a real business analyst and fix any remaining UX friction.

## Day 31 (04:00) — Done
**Track D — Per-Deployment Rate Limiting + Monthly Quotas.** Closes the production-readiness gap: analysts sharing deployed prediction endpoints can now cap per-minute request rates and rolling 30-day prediction counts via chat ("set rate limit to 60 requests per minute", "add a monthly quota of 1000 predictions").
- `rate_limit_rpm` + `monthly_quota` fields on `Deployment` (inline SQLite migration). `_check_rate_limit()` sliding-window (in-memory deque + threading.Lock). `_check_monthly_quota()` rolling 30-day PredictionLog count. HTTP 429 on violation.
- `PUT /api/deploy/{id}/rate-limit`, `GET /api/deploy/{id}/quota-status` endpoints. `GET /api/deploy/{id}` now exposes both fields.
- `_RATE_LIMIT_PATTERNS` + 4 extraction regexes in `chat.py`; handler applies set/disable/status without crashing chat; emits `{type:"rate_limit"}` SSE event.
- `RateLimitCard` (amber border, ⚡ icon): Active/No limits badge, RPM or "Unlimited", quota fraction + color-coded `UsageBar` (green/amber/red), percentage used, remaining, help text footer.
- `RateLimitInfo` + `QuotaStatus` TypeScript types; `attachRateLimitToLastMessage` Zustand action; `setRateLimit()` + `quotaStatus()` API methods.
- 26 backend + 17 frontend = 43 new tests. Total: 2915 backend + 1495 frontend = 4410, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: prediction confidence intervals surfaced in the prediction response ("your model predicts $42k revenue ± $3.2k"), deployment health scoring (composite "deployment health" metric from SLA + drift + error rate).
- Track B: cross-project model comparison API, cross-project template sharing.
- Track C: SHAP explanation caching (avoid recomputing on every chat ask), automated feature recommendation based on correlation analysis.

## Day 30 (20:00) — Done
**Track B — Cross-Project Portfolio Overview.** Closes the "I have multiple projects — show me everything at a glance" gap. Analysts managing several prediction models across different projects can now ask "show all my models", "portfolio overview", or "which project is doing best" and receive a `PortfolioCard` SSE card in chat.
- `compute_portfolio_summary(project_summaries)` pure function in `core/analyzer.py` — aggregates total_projects, active_deployments, total_predictions, best_performer (highest metric), per-project summaries.
- `GET /api/projects/portfolio` endpoint (registered BEFORE `/{project_id}` to avoid route shadowing) — queries all projects, finds best model run + active deployment + prediction count for each.
- `_PORTFOLIO_PATTERNS` (10 NL variants: "show all my models", "portfolio overview", "compare all my projects", "which project is doing best", "cross-project view", "all my work", etc.) in `chat.py`; handler cross-queries all projects from session, emits `{type:"portfolio"}` SSE event with full summary + system prompt injection.
- `PortfolioCard` (purple border, 🗂️ icon): header badges (N projects, N deployed, N predictions total), plain-English summary, 🏆 best performer highlight box (name, algorithm, metric %), per-project rows (name, dataset, target column, metric badge, Live/Trained/No model status badge, prediction count).
- `PortfolioResult`/`PortfolioProjectSummary`/`PortfolioBestPerformer` TypeScript types; `portfolio?` field on `ChatMessage`; `attachPortfolioToLastMessage` Zustand action; SSE handler + render wired in workspace page.
- 21 backend + 16 frontend = 37 new tests. Total: 2889 backend + 1478 frontend = 4367, all passing. Backend lint: clean. Frontend build + lint: clean.

## Day 30 (12:00) — Done
**Track D — SDK Generation.** Closes the developer-handoff gap: a deployed model is a REST API, but developers still had to reverse-engineer the endpoint shape and write HTTP code from scratch. Now a single chat message ("generate a python sdk") triggers downloadable, schema-aware Python and JavaScript client libraries.
- `GET /api/deploy/{id}/sdk?language=python|javascript` — generates typed client library from deployment's feature schema; `Content-Disposition: attachment` triggers browser download.
- `_generate_python_sdk()` — full Python module: typed `predict(feature1: float, ...) → dict` and `predict_batch(rows) → list[dict]` methods with docstrings, requests dependency, error handling, regression/classification-aware return docs.
- `_generate_javascript_sdk()` — ES module class with `async predict()` / `async predictBatch()`, JSDoc, fetch-based HTTP.
- `_SDK_PATTERNS` — 8 NL variants in chat.py; SDK event → `SdkDownloadCard` via SSE + Zustand + page.tsx.
- `SdkDownloadCard` — indigo border, badge, two download links (Python .py / JavaScript .js), inline usage previews for both languages.
- 27 backend + 16 frontend = 43 new tests. Total: 2868 backend + 1462 frontend = 4330, all passing.

**What's next:**
- Track D still has gaps: API key auth for prediction endpoints (currently open to anyone who knows the URL), scheduled batch prediction jobs, deployment versioning + rollback.
- Track C: class imbalance detection (SMOTE / class weights), date-aware chronological train/test splits.

## Day 30 (04:00) — Done
**Track B — Natural Language Date Range Filtering.** Closes the "show me Q4 data" gap that the existing filter system always promised but never delivered.
- **NL Date Filter** — `parse_date_filter_request(message, df)` pure function in `core/filter_view.py`. Detects date columns by name-hint (date/time/year/month/period/quarter/week) or string-value sampling. Resolves 6 NL patterns: Q1–Q4 with optional year ("Q4 2023"), quarter word ("third quarter 2023"), year-only ("show 2024 data"), month range ("January through March 2023"), last-N ("last 6 months", "last 2 years", "last 3 weeks"), and relative ("this year", "last year", "this month", "last month"). Returns `date_range` operator with `{start, end}` ISO-date value dict. `apply_active_filter()` extended with `date_range` branch using `pd.to_datetime()` comparison. `build_filter_summary()` formats date_range as "column between START and END". `FilterCondition` TypeScript type gains `date_range` operator + `DateRangeValue` union. `FilterSetCard` renders "between START and END" for date_range conditions. `_FILTER_PATTERNS` regex in `chat.py` extended to catch date-intent phrases; chat filter handler merges date conditions alongside field conditions. 17 backend + 5 frontend = 22 new tests. Total: 2841 backend + 1446 frontend = 4287.

**What's next:**
- Track B continues deep — remaining ideas: preset delete/reorder UI on predict page, cross-project model comparison.
- Or branch into a new data input channel: CSV URL monitoring (auto-refresh on cron), direct database connector improvements.

## Day 29 (20:00) — Done
**Track B — Multi-Row Batch Prediction.** Closes the "compare multiple independent scenarios at once" gap.
- **Multi-Row Prediction** — `_MULTI_ROW_PRED_PATTERNS` (6 NL variants) + ";" trigger (any message with semicolons and inline pred pattern) in `chat.py`. `_extract_multi_row_predictions()` with `_trim_preamble()` helper strips leading preamble from each segment before k-v parsing. Handler mutually exclusive with `inline_pred_event` (multi-row takes priority). `MultiPredictionCard` (violet, 📊): scenario comparison table with row# | prediction | feature columns | defaults. 17 backend + 15 frontend = 32 new tests. Total: 2824 backend + 1441 frontend = 4265.

**What's next:**
- All spec items 100% checked. Track B is very deep.
- Consider: preset delete/reorder UI on predict page, natural language date filtering, or cross-project model comparison.
- Or deepen an existing feature: richer preset management, template sharing, or UX polish on shared predict page.

## Day 29 (04:00) — Done
**Track B — Prediction Presets on the VP Dashboard.** Closes the "VP doesn't know what to type" cold-start gap.
- **Prediction Presets** — `DeploymentPreset` SQLModel table. `GET/POST/DELETE /api/deploy/{id}/presets` CRUD endpoints. `_PRESET_SAVE_PATTERNS` (8 NL variants) + `_PRESET_LIST_PATTERNS` (4 NL variants) + `_extract_preset_definition()` helper in `chat.py`. Chat handlers persist presets to DB and emit `{type:"preset_saved"}` + `{type:"preset_list"}` SSE events. `PresetSavedCard` (emerald, 🎯) + `PresetListCard` (indigo, 📋). `predict/[id]/page.tsx`: loads presets on mount, shows "Quick Scenarios" pill buttons above the form. 25 backend + 20 frontend = 45 new tests. Total: 2807 backend + 1426 frontend = 4233.

**What's next:**
- Further VP dashboard polish: preset delete UI on prediction page, preset order/rename.
- Multi-row inline prediction table: "predict for: Region=East, Units=100; Region=West, Units=150"
- Cross-project template sharing or model comparison improvements.

## Day 28 (20:00) — Done
**Track B — Prediction Cohort Analysis + CSV Export for Ranked Predictions.** Closes the "who ARE the top predictions?" and "download this list" gaps.
- **Prediction Cohort Analysis** — `_COHORT_PATTERNS` (9 NL variants: "who are the top predictions", "what do they have in common", "profile/characterize/describe the ranked records", "cohort analysis", "tell me about the top N customers") in `chat.py`. Handler fires when deployment + dataset exist and ranked_pred_event hasn't already fired. `compute_prediction_cohort()` pure function in `core/deployer.py`: re-ranks the dataset (same as `run_dataset_ranking`), then profiles the top-N rows vs the full dataset: categorical breakdown (per-category top-N% vs overall%, ratio), numeric comparison (top-N mean vs overall mean, ratio, direction label). Generates plain-English `characterization`: "The 20 highest-scoring revenue predictions: 70% have region = 'East'; units is 80% higher on average." `PredictionCohortCard` (indigo border, 🔍): "Highest/Lowest" badge, count badge, characterization paragraph, "Categorical Breakdown" section with dual-bar chart (indigo=top-N, slate=overall), "Numeric Averages" section with per-column rows showing ratio badge (rose=much higher, amber=moderately higher, sky=lower) + top avg vs overall avg. Handles empty profiles gracefully.
- **CSV Download for Ranked Predictions** — "⬇ Download CSV" button added to `RankedPredictionsCard` header. Client-side CSV generation from SSE data: includes rank, row_index, all predicted values (class+confidence for classification, value for regression), and ALL feature columns (not just the 4 visible in the table). Filename: `{target_column}_ranked_predictions.csv`. No new backend endpoint needed.
- 24 backend + 18 frontend = 42 new tests. Total: 2782 backend + 1406 frontend = 4188.

**What's next:**
- Track B deep — consider: cross-project template sharing, multi-project model comparison, or UX polish on VP-shared predict page.
- Potential new gap: "smart prediction routing" — when analyst asks "predict for these 10 scenarios" (multi-row inline prediction), batch them as a table result.

## Day 28 (12:00) — Done
**Track B — Dataset Ranking via Model.** Closes the "which specific rows should I act on?" gap.
- **Dataset Ranking** — `_RANKED_PRED_PATTERNS` (8 NL variants: "which customers are most likely to churn", "top N predictions", "rank by predicted revenue", "most at risk", etc.) + `_detect_ranked_pred_request()` extracting n (default 20, capped 100) and direction. `run_dataset_ranking()` pure function in `core/deployer.py`: scores all rows via `pipeline.transform_df()` + model; regression uses `predict()` float values; classification ranks by max class probability. `RankedPredictionsCard` (amber border, 🏆): gold/silver/bronze rank badges; `PredictionCell` (regression: compact number; classification: "class (XX%)" with green/amber/red confidence); table with rank + prediction + up to 4 feature columns; summary footer. 24 backend + 17 frontend = 41 new tests. Total: 2758 backend + 1388 frontend = 4146.

**What's next:**
- All tracks D, C, E complete. Track B now very deep (ranking, interaction, sensitivity, what-if, forecasting, anomaly, templates, version history, onboarding).
- Consider: cross-project template sharing, multi-project model comparison, prediction export (CSV download of ranked rows), or UX polish on the VP-shared predict page.

## Day 28 (04:00) — Done
**Track B — Feature Interaction Analysis.** Closes the "which combination of two variables gives the best outcome?" gap.
- **Feature Interaction** — `_INTERACTION_PATTERNS` (8 NL variants: "interaction between X and Y", "joint effect", "2D sensitivity", "feature interaction heatmap", etc.) + `_detect_interaction_request()` longest-match extractor in `chat.py`. `run_feature_interaction()` pure function in `core/deployer.py`: sweeps numeric features over [mean ± 2×std]; categorical features use all label encoder classes. Builds n×m prediction grid. `InteractionCard` (violet border, 🔬): color-coded heatmap table (rose=low, emerald=high for regression; violet for classification), min/max boxes, Low/High legend, summary footer. 25 backend + 19 frontend = 44 new tests. Total: 2734 backend + 1371 frontend = 4105.

## Day 27 (20:00) — Done
**Track B — Saved Analysis Templates.** Closes the "replay my analysis on new data" gap.
- **Analysis Templates** — `_SAVE_TEMPLATE_PATTERNS` / `_LIST_TEMPLATES_PATTERNS` / `_REPLAY_TEMPLATE_PATTERNS` + `_extract_template_name()` in `chat.py`. `AnalysisTemplate` SQLModel table (`id`, `project_id`, `name`, `queries` JSON, `created_at`). CRUD endpoints: `GET/POST /api/projects/{id}/analysis-templates` + `DELETE /api/projects/{id}/analysis-templates/{tid}`. Chat handler for save: loads last 8 user messages from conversation history, filters out the save command itself, saves as template, emits `{type:"template_saved"}` SSE. Chat handler for list: queries all templates, emits `{type:"template_list"}`. Chat handler for replay: finds template by name match (falls back to most recent), emits `{type:"template_replay"}` with queries as clickable chips. Frontend: `TemplateSavedCard` (emerald border, 💾, shows name + queries + replay hint), `TemplateListCard` (blue, lists templates with Replay buttons), `TemplateReplayCard` (purple, queries as click-to-send buttons that fill chat input). Types, API client, Zustand actions, SSE handlers, page.tsx wiring all complete. 17 backend + 17 frontend = 34 new tests. Total: 2684 backend + 1336 frontend = 4020.

**What's next:**
- Track B is now essentially saturated — all listed backlog items are complete.
- Consider deeper collaboration features (sharing templates across projects) or
  polishing existing features based on observed usage gaps.

## Day 27 (04:00) — Done
**Track B — Data Version History Timeline.** Closes the "how has my data changed across uploads?" gap.
- **Version History** — `_VERSION_HISTORY_PATTERNS` (8 NL variants: "show my upload history", "data version timeline", "upload history", "history of my datasets", etc.) in `chat.py`. `compute_version_history()` pure function in `core/analyzer.py`: builds upload timeline from all project datasets, computes drift between consecutive pairs via `compute_dataset_comparison()`. `GET /api/data/{project_id}/version-history` endpoint. Chat handler emits `{type:"version_history"}` SSE. `DataVersionHistoryCard` (adaptive border: emerald=stable / amber=moderate / rose=high, 📂 icon): stability badge, version count, timeline rendered latest-first with version rows + drift connectors. 22 backend + 18 frontend = 40 new tests. Total: 2667 backend + 1319 frontend = 3986.

**What's next:**
- Continue Track B — remaining opportunities:
  - Saved analysis templates (replay a custom chat flow on new data)
  - Natural language column transformations ("create a column: revenue per unit = revenue / units")

## Day 26 (20:00) — Done
**Track B — Guided Onboarding Wizard.** Closes the "I don't know where to start" first-time-user gap.
- **Onboarding Wizard** — `_ONBOARDING_PATTERNS` (8 NL variants: "guide me through", "help me get started", "walk me through the steps", "show me the guide", "what should I do first", "first steps", "onboarding", "how do I use this") in `chat.py`. `compute_onboarding_state()` pure function in `core/onboarding.py`: maps 6 progress flags (has_dataset, message_count, has_target, has_model_run, has_cross_val, has_deployment) to a step-by-step state dict. `GET /api/projects/{id}/onboarding` endpoint. Chat handler emits `{type:"onboarding_guide"}` SSE event with current step, completion %, steps list, hint, and CTA action. `OnboardingGuideCard` (blue border, 🧭): progress bar, step list (checkmarks for done, icon for current, ○ for pending), current step description + hint + CTA tab-switch button. 26 backend + 16 frontend = 42 new tests. Total: 2645 backend + 1301 frontend = 3946.

**What's next:**
- Continue Track B — remaining opportunities:
  - Data version history (timeline of dataset uploads with comparison)
  - Saved analysis templates (replay a custom chat flow on new data)

## Day 26 (12:00) — Done
**Track B — Prediction Sensitivity Analysis.** Closes the "how much does my prediction change as X varies?" gap.
- **Sensitivity Analysis** — `_SENSITIVITY_PATTERNS` (8 NL variants) + `_detect_sensitivity_request()` in `chat.py`. `run_sensitivity_analysis()` pure function in `core/deployer.py`: sweeps one feature across a range, holds all others at training means, collects predictions. `SensitivityCard` (teal, 🎚️): Recharts line chart + change % badge + min/max boxes. 24 backend + 17 frontend = 41 new tests. Total: 2619 backend + 1285 frontend = 3904.

**What's next:**
- Continue Track B — remaining opportunities:
  - Guided onboarding wizard (step-by-step first-use flow for new analysts)
  - Data version history (show data changes over time as new uploads are made)
  - Saved analysis templates (re-run the same analysis flow on new data)

## Day 26 (04:00) — Done
**Track B — Goal-Driven Training.** Closes the "I need X% accuracy — just find me an algorithm that works" gap.
- **Goal-Driven Training** — `_GOAL_TRAIN_PATTERNS` (8 NL variants) + `_extract_goal_target()` in `chat.py`. `run_goal_driven_training()` pure function in `core/trainer.py`: tries linear/RF/GBoost in order, stops early on success, falls back to tuning on best. Sub-samples >5,000 rows for speed. `GoalTrainingCard` (emerald/amber, 🎯) with winner box, trials table ✓/✗, tuning note, summary. 26 backend + 16 frontend = 42 new tests. Total: 2595 backend + 1268 frontend = 3863.

**What's next:**
- Continue Track B — remaining opportunities:
  - Guided onboarding wizard (step-by-step first-use flow for new analysts)
  - Data version history (show data changes over time as new uploads are made)
  - Natural language column transformations ("create a new column: revenue per unit = revenue / units")
  - Saved analysis templates (re-run the same analysis flow on new data)

## Day 25 (20:00) — Done
**Track B — Inline Multi-Feature Prediction via Chat.** Closes the "Conversation over configuration" vision gap — analysts can now get predictions without leaving the chat.
- **Inline Prediction** — `_INLINE_PRED_PATTERNS` (8 NL variants: "run a prediction for", "make a prediction with", "give me a prediction", "what would X be if", "score this record", "run the model on", "model output for", "what does the model predict"). `_KV_PAIR_RE` + `_extract_multi_feature_prediction()` parse `key=value` pairs from natural language, normalise keys case-insensitively, cast numerics to float, fill missing from training means. `{type:"inline_prediction"}` SSE event. `InlinePredictionCard` (blue, 🔮): regression shows prediction + CI; classification shows probability bars. 17 backend + 15 frontend = 32 new tests. Total: 2569 backend + 1252 frontend = 3821.

**What's next:**
- Continue Track B — remaining opportunities:
  - Guided onboarding wizard (step-by-step first-use flow for new analysts)
  - Data version history (show data changes over time as new uploads are made)
  - "Goal-driven training" — analyst sets target accuracy, AutoModeler tries algorithms + tuning to reach it

## Day 25 (04:00) — Done
**Track B — Prediction Opportunity Discovery.** Closes the "I have data but don't know what to model" cold-start gap.
- **Prediction Opportunities** — `compute_prediction_opportunities()` pure function in `core/analyzer.py`. Exclusion filters (ID names, high-cardinality categoricals, >30% missing, constant). Regression for numeric, classification for 2-20 category cols. Feasibility score rewards completeness + predictors + business-value name patterns. `_PREDICT_OPP_PATTERNS` (9 NL variants) + system prompt injection + `{type:"prediction_opportunities"}` SSE. `PredictionOpportunitiesCard` (purple border, 🎯) with ranked rows, feasibility bars, problem-type + business-value badges. 24 backend + 19 frontend = 43 new tests. Total: 2529 backend + 1228 frontend = 3757.

**What's next:**
- Continue Track B — remaining opportunities:
  - Multi-dataset comparison (upload v2 dataset, compare model performance pre/post)
  - Guided onboarding wizard (step-by-step first-use flow for new analysts)
  - Data version history (show data changes over time as new uploads are made)

## Day 24 (20:00) — Done
**Track B — Proactive Model Health Alerts.** Smart-colleague proactive notification when deployed models are aging or idle.
- **Proactive Health Alerts** — `compute_deployment_health_item()` + `compute_project_health_summary()` pure functions in `core/analyzer.py`. Age (0–100) + usage (0–100) scores → combined health score → healthy/warning/critical status. `GET /api/projects/{id}/health-summary` endpoint. `_HEALTH_SUMMARY_PATTERNS` (9 NL variants) + `{type:"health_summary"}` SSE event. `ProjectHealthCard` (adaptive border) in chat with per-alert rows, health bars, CTA buttons. **Proactive injection**: on project load, welcome-back message automatically includes alerts if any deployment is degraded. 16 backend + 14 frontend = 30 new tests. Total: 2505 backend + 1209 frontend = 3714.

## Day 24 (12:00) — Done
**Track B — Conversation Export as HTML Report.** Closes the "share full analysis journey" use case.
- **Conversation Export** — `_CONV_EXPORT_PATTERNS` (13 NL variants). `_build_export_html()` pure function generates self-contained HTML (header, dataset info, model results, conversation transcript, embedded CSS). `GET /api/chat/{project_id}/export` → HTML attachment. `ConversationExportCard` (emerald border, 📄) in chat: message count badge, dataset badge, download link. 14 backend + 10 frontend = 24 new tests. Total: 2475 backend + 1195 frontend = 3670.

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
