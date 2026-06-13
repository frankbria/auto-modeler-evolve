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

*(none — next session should pick from the list below)*

**Shipped Day 94 (04:00):** Prediction Outcome Calibration Chart (Track D) — reliability diagram (classification) or error histogram (regression) built from production FeedbackRecord outcomes. `compute_prediction_outcome_calibration()` pure function; `GET /api/deploy/{id}/outcome-calibration` REST endpoint; `_OUTCOME_CALIB_PATTERNS` (8 NL variants); `OutcomeCalibrationCard` (Recharts LineChart reliability diagram or BarChart error histogram). 59 new tests (38 backend + 21 frontend). Baseline: 6460/3668 → **6498/3689**.

**Shipped Day 93 (20:00):** Model Retraining Completion Notification (Track D) — `retrain_complete` webhook fires when degradation-triggered auto-retrain finishes; `RetrainCompleteNotifyCard` shows last completed retrain stats + registered notification URLs. 41 new tests (21 backend + 20 frontend).

**What's next (priority order):**
- Scheduled batch prediction job history and analytics — analysts ask "show me batch job history" or "when did my last batch run?" and see a BatchJobHistoryCard with job IDs, completion times, row counts, error rates
- Model performance decay rate analysis — "how fast is my model degrading?" with a linear trend line over weekly feedback accuracy, estimated weeks until below threshold
- Prediction confidence distribution shift alert — proactive webhook when the distribution of confidence scores changes significantly from baseline (related to but distinct from covariate drift)

---

## Day 93 (04:00) — Done

**Track D: Prediction Confidence Band Chart** — complete.

`compute_confidence_band()` pure function groups PredictionLog dicts by day, computes per-day mean ± std; CoV-based stability verdict (stable/moderate_spread/high_spread/no_data). `GET /api/deploy/{id}/confidence-band` REST endpoint. `_CONFIDENCE_BAND_PATTERNS` (8 NL variants) chat handler; SSE event `confidence_band`. `ConfidenceBandCard` (sky/amber/rose border, Recharts ComposedChart shaded band + mean Line). 36 new backend tests (+36 / +0). Baseline: 6381/3630 → **6417/3630**. Two pattern bugs fixed: `show(?:\s+me)?` for "show me the confidence band"; `predictions?(?:\s+over\s+time)?` for bare "predictions" end-of-string.

---

## Day 92 (20:00) — Done

**Track D: Production Input Distribution Drift Alert** — complete.

Proactive webhook that fires when live prediction inputs diverge from the training distribution at or above a configured severity threshold. `EVENT_INPUT_DIST_DRIFT` webhook constant; 3 new `Deployment` fields (`input_dist_drift_alert_enabled`, `input_dist_drift_severity_threshold`, `input_dist_drift_alert_last_fired_at`); `_check_and_fire_input_dist_drift_alert()` checker wired into scheduler loop; `PUT /api/deploy/{id}/input-dist-drift-alert` + `GET /api/deploy/{id}/input-dist-drift-alert-status` REST endpoints; `_INPUT_DIST_DRIFT_ALERT_PATTERNS` (8 NL variants) + 3 sub-patterns in chat handler; `InputDistDriftAlertCard` frontend component (cyan border, 🌊 icon, threshold label). Reuses existing `compute_covariate_drift_alert()` pure function. 42 new tests (+24 backend / +18 frontend). Baseline: 6357/3612 → **6381/3630**.

**What's next:**
- Model retraining orchestration — detect performance degradation and suggest retraining with recent data
- Deployment comparison leaderboard improvements — rank deployments by accuracy, latency, activity
- Prediction confidence band chart — show prediction uncertainty ranges over recent predictions

---

## Day 92 (12:00) — Done

**Track D: Deployment Health Scorecard** — complete.

Consolidated A–F health grade aggregating 5 operational signals: latency p95, avg confidence, activity last-7d, feedback accuracy, and model age. Analysts ask "show deployment health scorecard" or "is my deployment healthy?" and receive a `DeploymentHealthScorecardCard` with an overall grade, score, health label, and 5 color-coded signal rows. `compute_deployment_health_scorecard()` pure function in `core/analyzer.py`; `GET /api/deploy/{id}/health-scorecard` REST endpoint; `_DEPLOY_HEALTH_SCORECARD_PATTERNS` (8 NL variants) chat handler; SSE event; full TypeScript wiring. 62 new tests (+38 backend / +24 frontend). Baseline: 6319/3603 → **6357/3627**.

**What's next:**
- Production input distribution drift alert — notify when live prediction inputs diverge from training distribution
- Model retraining orchestration — detect performance degradation and suggest retraining
- Deployment comparison leaderboard improvements

---

## Day 92 (04:00) — Done

**Track D: Canary Deployment Support** — complete.

Route a configurable % of production predictions to a new model version for live A/B testing before full rollout. Analysts ask "start a canary with 10% traffic", "compare canary vs control", or "promote the canary". `compute_canary_comparison()` pure function; 4 REST endpoints (start/cancel/promote/status); `_CANARY_PATTERNS` (10 NL variants); `CanaryCard` frontend component. 65 new tests (+30 backend / +35 frontend). Baseline: 6289/3568 → **6319/3603**.

---

## Day 91 (20:00) — Done

**Track D: Feature Interaction Heatmap REST Endpoint** — complete.

Discovered the chat feature was pre-existing (full implementation: `run_feature_interaction()`, `_INTERACTION_PATTERNS`, `InteractionCard`). Added `GET /api/deploy/{id}/feature-interaction-heatmap` REST endpoint for programmatic access. Auto-selects feature pair; `n_steps` param (2–10). Cleaned up duplicate code accidentally introduced in prior session attempts (duplicate pure function, chat patterns/handler/SSE emit, frontend types/store/card). +5 REST tests. Baseline: 6284/3568 → **6289/3568**.

**What's next:**
- Deployment health scorecard — consolidated "is my deployment healthy?" aggregating latency, drift, confidence, and quota signals
- Production input distribution drift alert — notify when live prediction inputs diverge from training distribution
- Canary deployment support — route a configurable % of predictions to new model version, compare live

---

## Day 91 (12:00) — Done

**Track D: Saved Scenario Comparison** — complete.

Analysts save named what-if configurations and compare predictions side by side. Say "save discount=0.1 quantity=100 as Q2 Optimistic", then "compare my scenarios" to see a `SavedScenariosCard` with per-scenario prediction bars, Best/Worst badges, and spread range. Scenarios persist across sessions. Features: `SavedScenario` SQLModel table; 4 REST endpoints (list/save/delete-one/clear-all); `compute_scenario_comparison()` pure function in `core/deployer.py`; three chat intents (SAVE/VIEW/DELETE) with `_parse_save_scenario_request()` helper; `SavedScenariosCard` (sky border, 📋, regression bars + classification confidence%). 49 new tests (30 backend + 19 frontend). Baseline: 6254/3549 → **6284/3568**.

**What's next:**
- Feature interaction heatmap — sweep two features jointly to reveal interaction effects (e.g., "how do units and discount interact for revenue?")
- Deployment health scorecard — consolidated "is my deployment healthy?" card with latency, drift, confidence, and quota signals in one place
- Production input distribution drift alert — notify when live prediction inputs diverge from training distribution

---

## Day 90 (20:00) — Done

**Track D: Feature Impact Sweep** — complete.

Ranked analysis of every model feature showing which value ranges produce the most extreme predictions, culminating in an optimal configuration. Analysts ask "which feature values produce the most extreme predictions?", "feature impact sweep", or "what combination gives the highest prediction?" and receive a `FeatureSweepCard` with ranked features + delta bars + optimal config. Direction auto-detected from message (maximize vs minimize). Sweeps each feature independently (O(F×N), not grid search). Features: `run_feature_sweep(pipeline_path, model_path, direction, n_steps)` in `core/deployer.py`; REST `GET /api/deploy/{id}/feature-sweep`; `_FEATURE_SWEEP_PATTERNS` (9 NL variants) + `_FEATURE_SWEEP_MINIMIZE_RE` in `chat.py`; `FeatureSweepCard` (teal border, 🔭 icon, optimal config section + ranked delta bars). 47 new tests (28 backend + 19 frontend). Baseline: 6226/3530 → **6254/3549**.

**What's next:**
- Feature interaction heatmap — sweep two features jointly to reveal interaction effects (e.g., "how do units and discount interact for revenue?")
- What-if scenario comparison card — save and compare multiple input configurations side-by-side
- Production input distribution drift alert — notify when live prediction inputs diverge from training distribution

---

## Day 90 (12:00) — Done

**Track D: Prediction Confidence Heatmap by Feature Value** — complete.

Closes the "which input combinations make my model least confident?" analyst gap — distinct from `SegmentConfidenceTrendCard` (per-segment time series) and `ConfidenceDistributionCard` (overall histogram, training-time). Analysts ask "confidence heatmap", "which input combinations make my model least confident?", "feature confidence grid", "show low confidence zones", "confidence by age and income", "model uncertainty heatmap", "which combinations confuse my model?", or "where is my model uncertain for age and region?" (8 NL variants in `_CONF_HEATMAP_PATTERNS`) and receive a `ConfidenceHeatmapCard` showing a 2D grid of average model confidence across two feature dimensions — revealing exactly which input value combinations produce unreliable predictions.

**What was built:**
- `compute_confidence_heatmap(logs_data, feature_x, feature_y, n_bins=5, *, problem_type, min_samples_per_cell=3)` pure function in `core/analyzer.py`: for numeric features creates n_bins equal-width bins (clamped 2–6); for categorical features uses sorted unique values (up to 8); computes avg confidence per (x_bin, y_bin) cell; for regression uses inverse CV% as consistency proxy; marks cells with avg < 65% as `is_low_confidence`; sorts `low_confidence_zones` ascending by confidence; verdicts: `gaps_found` / `uniform_high` / `uniform_moderate` / `insufficient_data` / `no_data`; requires ≥5 valid logs, ≥3 samples per cell; plain-English summary with worst zone details
- `GET /api/deploy/{id}/confidence-heatmap?feature_x=age&feature_y=region&n=200&n_bins=5` REST endpoint in `api/deploy.py`: auto-detects feature pair when not specified (prefers categorical + numeric for interpretability); loads up to n prediction logs; returns full heatmap result with `deployment_id`
- `_CONF_HEATMAP_PATTERNS` (8 NL variants) + `_detect_heatmap_features()` helper (extracts up to 2 feature names from message, falls back to auto-detect from feature_ranges) + handler in `chat.py`; guarded by `ctx["deployment"]`; emits `{type:"conf_heatmap"}` SSE event
- `ConfidenceHeatmapCard` at `components/deploy/confidence-heatmap-card.tsx`: rose border (gaps_found) / emerald (uniform_high) / amber (uniform_moderate); 🗺️ icon; feature pair + n-samples + cells-populated badges; stats row (min/mean/max confidence %); CSS grid heatmap with color-coded cells (rose <55%, amber 55–70%, lime 70–85%, emerald ≥85%); cell tooltip with exact confidence and sample count; color legend; rose `role="alert"` callout listing low-confidence zones with feature values, avg confidence, and sample counts
- Full TypeScript wiring: `ConfidenceHeatmapVerdict`, `ConfidenceHeatmapCell`, `ConfidenceHeatmapZone`, `ConfidenceHeatmapResult` interfaces; `conf_heatmap?` on `ChatMessage`; `attachConfHeatmapToLastMessage` Zustand action; SSE handlers (both EventSource branches) + card render in `project/[id]/page.tsx`

**Tests:** 35 backend (18 pure-function + 8 regex + 5 REST endpoint + 4 negatives) + 18 frontend (15 card component + 3 store action) = **53 new tests**. All passing. Backend lint: clean. Frontend build + TypeScript: clean.

**Baseline:** 6191 backend / 3512 frontend → **6226 backend / 3530 frontend** (+35 / +18).

**What's next:**
- Canary deployment support — route a configurable % of predictions to a new model version, compare accuracy/latency live before full promotion
- Deployment A/B test result summary via chat — compare two live deployments side by side with accuracy, latency, and prediction distribution
- Model-level feature sensitivity sweep — "which feature value range produces the most extreme predictions?"

---

## Day 90 (04:00) — Done

**Track D: Prediction Value Trend Alert** — complete.

Proactive push notification when the rolling mean prediction output shifts significantly from its recent baseline — without requiring labeled feedback. Compares early (older 50) vs. recent (newer 50) halves of the last 100 `PredictionLog` entries; fires `pred_value_trend_alert` webhook when `abs(change_pct) > pred_value_alert_pct`. Works universally for regression (`prediction_numeric`) and classification (`confidence` fallback). 24-hour cooldown, ≥20 sample minimum. Features: `EVENT_PRED_VALUE_TREND_ALERT` constant; 3 new `Deployment` fields (`pred_value_alert_enabled`, `pred_value_alert_pct`, `pred_value_alert_last_fired_at`); `_check_and_fire_pred_value_trend_alert()` in `api/deploy.py`; scheduler loop wiring; REST endpoints (PUT enable/disable, GET status); chat handler with 8 NL variants + `_PRED_VALUE_ALERT_PCT_RE`; `PredValueAlertCard` (emerald/amber/slate border, 📈 icon). 41 new tests (23 backend + 18 frontend). Baseline: 6168/3494 → **6191/3512**.

**What's next:**
- Canary deployment support — route a configurable % of predictions to a new model version, compare accuracy/latency live before full promotion
- Deployment A/B test result summary via chat — compare two live deployments side by side with accuracy, latency, and prediction distribution
- Prediction confidence heatmap by feature value — "which input combinations make my model least confident?" (Track D)

---

## Day 89 (12:00) — Done

**Track D: Prediction Latency Alert** — complete.

P95 latency webhook alert. When the rolling p95 response time over the last 100 predictions exceeds a configurable millisecond threshold, a `latency_alert` webhook fires (1-hour cooldown). Catches model slowdowns before users notice — fills the gap between SLA stats (display only) and a real push notification. Uses `PredictionLog.response_ms` (already tracked). Features: `EVENT_LATENCY_ALERT` constant, `Deployment.latency_alert_threshold_ms` + `latency_alert_last_fired_at` fields, `_check_and_fire_latency_alert()` with 100-sample p95 and 1h cooldown, scheduler loop wiring, REST endpoints (PUT enable/disable, GET status), chat handler with 8 NL variants, `LatencyAlertCard` (orange border, ⏱ icon). 37 new tests (21 backend + 16 frontend). Baseline: 6125/3460 → **6146/3476**.

**What's next:**
- Canary deployment support — route a configurable % of predictions to a new model version, compare accuracy/latency live before full promotion
- Prediction value trend alert — webhook when the rolling mean prediction shifts significantly (e.g., "alert me if average revenue prediction drops more than 15%")
- Deployment A/B test result summary via chat — compare two live deployments side by side with accuracy, latency, and prediction distribution

---

## Day 89 (20:00) — Done

**Track D: Deployment Accuracy-Triggered Auto-Rollback** — complete.

When feedback accuracy (computed from the last 50 `FeedbackRecord.is_correct` values) drops below a configurable threshold, the deployment automatically reverts to the previous `DeploymentVersion` and fires a `rollback_triggered` webhook. Closes the "my model accuracy degraded in production and I didn't catch it in time" ops gap — distinct from manual rollback (user-initiated) and accuracy alerts (notify-only). Features: `EVENT_ROLLBACK_TRIGGERED` webhook constant; 3 new `Deployment` fields (`auto_rollback_enabled`, `auto_rollback_accuracy_threshold`, `auto_rollback_triggered_at`); `_check_and_fire_accuracy_rollback()` with 24h cooldown, ≥10 feedback minimum, ≥2 version requirement; scheduler loop wiring; REST endpoints (PUT enable/disable, GET status); chat handler with 8 NL variants (enable, disable, status, threshold extraction); `AutoRollbackCard` (emerald/amber/slate border). 40 new tests (22 backend + 18 frontend). Baseline: 6146/3476 → **6168/3494**.

**What's next:**
- Canary deployment support — route a configurable % of predictions to a new model version, compare accuracy/latency live before full promotion
- Prediction value trend alert — webhook when rolling mean prediction shifts significantly
- Deployment A/B test result summary via chat — compare two live deployments side by side

---

## Day 89 (04:00) — Done

**Track D: Prediction High-Activity Burst Alert** — complete.

The symmetric counterpart to the Day 88 low-activity sentinel. Low-activity catches silent failures (nobody calling your endpoint). High-activity burst catches the opposite: runaway loops, API abuse, or unexpected demand spikes. When hourly prediction count exceeds a configurable ceiling, the registered webhook fires with a 1-hour cooldown. Uses a rolling 60-minute window (recency matters for burst detection, unlike low-activity's daily aggregation). Features: `EVENT_HIGH_ACTIVITY_BURST` webhook constant, `Deployment.high_activity_threshold_per_hour` + `high_activity_burst_last_fired_at` fields, `_check_and_fire_high_activity_burst()` with 1h rolling window and 1h cooldown, scheduler loop wiring, REST endpoints (PUT enable/disable, GET status), chat handler with 8 NL variants (4 patterns: enable, disable, status, plus threshold extraction), `HighActivityBurstCard` frontend component (amber border, 📈 icon). Also applied missing Day 88 migrations for low-activity fields. 37 new tests (21 backend + 16 frontend). Baseline: 6104/3444 → **6125/3460**.

**What's next (Track D fresh ideas):**
- Prediction latency P95/P99 monitoring — slow predictions degrade UX before counts drop; chat: "alert me if predictions take more than 500ms"; uses PredictionLog.latency_ms if tracked, or adds it
- Webhook delivery health dashboard — show last N webhook dispatches (success/failure, response codes, retry count); surface via `GET /api/deploy/{id}/webhook-delivery-history`
- Deployment rollback trigger based on accuracy drift — if rolling accuracy drops below baseline by X%, auto-disable deployment and notify; builds on existing DriftDetectionCard infrastructure
- A/B champion-challenger testing — route N% of traffic to a challenger model; compare live accuracy side-by-side before promoting

---

## Day 88 (20:00) — Done

**Track D: Prediction Low-Activity Alert via Chat** — complete.

Closes the silent integration failure gap: when an upstream CRM, pipeline, or dashboard stops calling the model endpoint, no other alert fires because nothing bad actively *happens*. The low-activity webhook fires when daily predictions drop below a configured floor, giving analysts a fast notification that their integration may be broken. Features: `EVENT_LOW_ACTIVITY` webhook constant, `Deployment.low_activity_threshold_per_day` + `low_activity_alert_last_fired_at` fields, `_check_and_fire_low_activity_alert()` with midnight-UTC cutoff and 24h cooldown, scheduler loop wiring (60s), REST endpoints (PUT enable/disable, GET status), chat handler with 8 NL variants, `LowActivityAlertCard` frontend component. 36 new tests (20 backend + 16 frontend).

---

## Day 88 (12:00) — Done

**Track C: Cost-Sensitive Threshold Analysis** — complete.

Closes the "false positives cost $10, false negatives cost $500 — what threshold should I use?" analyst gap. This is the first feature that speaks in *dollars* rather than F1/precision/recall. Distinct from `ThresholdAnalysisCard` (sweeps thresholds, no dollar costs), `PerClassThresholdCard` (optimizes per class by F1), and `custom_class_weights` (raw weight specification). The feature derives the mathematically optimal decision threshold (`threshold* = C(FP) / (C(FP) + C(FN))`) and computes expected cost at default (50%) vs optimal threshold.

**What was built:**
- `compute_cost_sensitive_threshold(y_true, y_proba_positive, fp_cost, fn_cost, positive_label)` pure function in `core/validator.py`: optimal threshold formula, binary evaluation at 0.5 and optimal threshold, cost savings %, suggested class weight (FN/FP ratio), verdict `threshold_change_recommended` / `default_near_optimal`
- `_COST_SENSITIVE_PATTERNS` (8 NL variants) + `_extract_fp_fn_costs()` helper (labelled extraction + numeric fallback) + binary-only guard + handler in `chat.py`; emits `{type:"cost_sensitive_threshold"}` SSE event
- `CostSensitiveThresholdCard`: FP/FN cost badges, optimal threshold tile with formula, side-by-side Default vs Cost-Optimal metrics rows, green savings banner, sky retrain hint with suggested class weight
- Full TypeScript wiring: `CostSensitiveThresholdResult` + `CostSensitiveMetrics` types; `attachCostSensitiveThresholdToLastMessage` Zustand action; SSE handlers + card render in `page.tsx`

**Tests:** 38 backend + 10 frontend = **48 new tests**. All passing. Baseline: 6010/3394 → **6048/3428**.

**What's next:**
- Track D: Prediction API Uptime Summary — ALREADY BUILT (check analyzer.py:7862, deploy.py:8089, chat.py:3773) — need fresh Track D idea
- Track C: Comparative Model Improvement Plan — "create a ranked improvement roadmap for all my models" (Track B priority)
- Track C: Automated Feature Engineering Suggestions via Chat — "what new features could I create from my date column?"

---

## Day 87 (20:00) — Done

**Track D: Deployment Segment Confidence Trend** — complete.

Closes the "is my model less confident about West region predictions over time?" analyst gap — distinct from `SegmentPredictionTrendCard` (tracks prediction *values*) and `SegmentDriftCard` (input distribution drift). Completes the Day 87 segment monitoring trilogy: input drift (04:00), output value trend (12:00), model certainty (20:00). Analysts ask "confidence by segment", "model confidence per region", "is my model less confident for West", "confidence trend by segment", "which segment has lowest confidence", "model uncertainty by region", "confidence breakdown by segment", or "low confidence for each category" (8 NL variants in `_SEGMENT_CONF_TREND_PATTERNS`) and receive a `SegmentConfidenceTrendCard` with per-segment confidence sparklines, calibration gap detection, and most/least confident segment callouts.

**What was built:**
- `compute_segment_confidence_trend(logs_data, segment_column, problem_type, n_days=30, max_segments=8)` pure function in `core/analyzer.py`: classification = avg confidence score; regression = CV% (std/|mean| × 100%) as consistency proxy; ±2%/period threshold; `calibration_gap` flag (max−min > 0.15); verdicts: calibration_gap/uniform_decline/uniform_improving/mixed/stable/no_confidence_data/no_data
- `GET /api/deploy/{id}/segment-confidence-trend?segment_col=category&n=200&n_days=30` REST endpoint
- `_SEGMENT_CONF_TREND_PATTERNS` (8 NL variants) + handler in `chat.py`; emits `{type:"segment_conf_trend"}` SSE event
- `SegmentConfidenceTrendCard` (amber=calibration_gap / rose=uniform_decline / emerald=uniform_improving / sky=mixed / muted=stable, 🎯 icon): calibration gap warning, most/least confident callouts, per-segment sparklines
- Full TypeScript wiring: `SegmentConfTrendResult` interfaces; `attachSegmentConfTrendToLastMessage` Zustand action; SSE handlers + card render in `page.tsx`

**Tests:** 47 backend + 20 frontend = **67 new tests**. All passing. Baseline: 5963/3373 → **6010/3394**.

**What's next:**
- Track C: Cost-Sensitive Training via Misclassification Cost Matrix — "false positives cost $1000, false negatives cost $10"
- Track D: Prediction API Uptime Summary — "has my prediction endpoint had any downtime this week?"
- Track B: Comparative Model Improvement Plan — "create a ranked improvement roadmap for all my models"

---

## Day 87 (12:00) — Done

**Track D: Deployment Prediction Value Trend by Segment** — complete.

Closes the "is my average predicted revenue trending up or down for the West region?" analyst gap — distinct from `PredictionValueTrendCard` (overall trend, no segment breakdown) and `SegmentDriftCard` (input distribution drift, not output values). Analysts say "prediction trend by segment", "prediction trends per region", "which segment is improving?", "which category is declining?", "segment-level prediction trend", "how are my predictions changing for each region?", or 2 other NL variants (8 total in `_SEGMENT_PRED_TREND_PATTERNS`) and receive a `SegmentPredictionTrendCard` with per-segment sparklines showing whether predictions are trending up, down, or stable for each group.

**What was built:**
- `compute_segment_prediction_trend(logs_data, segment_column, problem_type, n_days=30, max_segments=8)` pure function in `core/analyzer.py`: groups enriched prediction log dicts by segment column value, computes daily means, derives change_pct (first-to-last), direction (>+2%/period=trending_up, <-2%=trending_down, else stable), sorts by |change_pct| desc; verdicts: diverging/all_improving/all_declining/mixed/stable/no_data
- `GET /api/deploy/{id}/segment-prediction-trend?segment_col=region&n=200&n_days=30` REST endpoint with auto-detect first categorical feature
- `_SEGMENT_PRED_TREND_PATTERNS` (8 NL variants) + handler in `chat.py` guarded by `ctx["deployment"]`; reuses `_detect_segment_col()` for auto-detection; emits `{type:"segment_pred_trend"}` SSE event
- `SegmentPredictionTrendCard` (amber=diverging / emerald=all_improving / rose=all_declining / sky=mixed / muted=stable, 📈 icon): verdict badge, column + segments + days badges, summary, most-improved/declining callouts, per-segment rows with direction badge + change% + first/latest/samples grid + Recharts sparkline (color-coded), empty state, sr-only figcaption
- Full TypeScript wiring: `SegmentPredTrendResult` and child interfaces; `attachSegmentPredTrendToLastMessage` Zustand action; SSE handlers + card render in both EventSource branches of `page.tsx`

**Tests:** 42 backend (17 pure-function + 14 regex + 4 REST endpoint + 7 helper) + 19 frontend (16 card component + 3 store action) = **61 new tests**. All passing. Backend lint: clean. Frontend build + TypeScript + lint: clean.

**What's next:**
- Track C: Cost-Sensitive Training via Misclassification Cost Matrix — "false positives cost $1000, false negatives cost $10"
- Track B: Comparative Model Improvement Plan — "create a ranked improvement roadmap for all my models"
- Track D: Confidence Calibration Trend by Segment — "is my model less confident for West region predictions over time?"

---

## Day 87 (04:00) — Done

**Track D: Deployment Segment Drift Detection** — complete.

Closes the "is my model drifting more for one region/segment than another?" analyst gap — distinct from `DriftImportanceCard` (ranks features, not segments) and `CovariateDriftAlertCard` (binary per-feature threshold). Analysts say "segment drift analysis", "drift by region", "which segment has the most drift", "geographic drift analysis", "drift breakdown by category", "is drift concentrated in a group" (8 NL variants in `_SEGMENT_DRIFT_PATTERNS`) and receive a `SegmentDriftCard` showing each segment's drift score.

**What was built:**
- `compute_segment_drift(all_inputs, segment_column, feature_ranges, max_segments=15)` pure function in `core/analyzer.py`
- `_detect_segment_col(message, feature_ranges)` helper: finds feature name in message or falls back to first categorical
- `GET /api/deploy/{id}/segment-drift?segment_col=region` REST endpoint; auto-detects segment column
- `_SEGMENT_DRIFT_PATTERNS` (8 NL variants) + chat handler guarded by `ctx["deployment"]`; emits `{type:"segment_drift"}` SSE event
- `SegmentDriftCard` (🗺️ icon): verdict badge (concentrated/widespread/minimal/no_data), segment rows with drift bars + status badges + top drifting features, avg drift footer

**Tests:** 35 backend (15 pure-function + 12 regex + 3 REST + 5 helper) + 18 frontend (15 card + 3 store) = **53 new tests**. Backend lint: clean. Frontend build + TypeScript + lint: clean.

**What's next:**
- Track C: Cost-Sensitive Training via Misclassification Cost Matrix — "false positives cost $1000, false negatives cost $10"
- Track D: Deployment Prediction Value Trend by Segment — "is my average predicted revenue trending up or down for West region?"
- Track B: Comparative Model Improvement Plan — "create a ranked improvement roadmap for all my models"

---

## Day 86 (20:00) — Done

**Track D: Deployment Feature Drift Alert Webhook** — complete.

Closes the "I want to be automatically notified when my important features drift — without polling the dashboard" analyst gap. Extends the Day 86 (12:00) drift importance ranking feature with push notifications. Analysts say "enable feature drift alerts", "turn on drift webhook", "alert me when features drift", "notify me on critical feature drift", "feature drift alert", "webhook on feature drift", "proactive drift alerts", "automatic drift notification" (8 NL variants in `_FEATURE_DRIFT_ALERT_PATTERNS`) to receive a `FeatureDriftAlertCard` and toggle automatic webhook delivery.

**What was built:**
- `EVENT_FEATURE_DRIFT = "feature_drift"` added to `core/webhook.py` `ALL_EVENTS`
- `Deployment.feature_drift_alert_enabled` + `Deployment.feature_drift_alert_last_fired_at` fields; inline migrations in `db.py`
- `_check_and_fire_feature_drift_alert(deployment_id, critical_features, cooldown_hours=24)` helper in `api/deploy.py`: 24-hour cooldown gate, stamps `last_fired_at` before dispatching, fires `EVENT_FEATURE_DRIFT` webhooks with `{critical_feature_count, top_critical_features, message}` payload; `except Exception: pass` throughout
- `GET /api/deploy/{id}/drift-importance-ranking` wires background thread when alert enabled + action_required/attention features present
- `PUT /api/deploy/{id}/feature-drift-alert` + `GET /api/deploy/{id}/feature-drift-alert-status` REST endpoints
- `_FEATURE_DRIFT_ALERT_PATTERNS` (8 NL variants) + `_DISABLE_FEATURE_DRIFT_ALERT_RE` + `_STATUS_FEATURE_DRIFT_ALERT_RE` in `api/chat.py`; handler enables/disables/reads config; emits `{type:"feature_drift_alert_config"}` SSE event
- `FeatureDriftAlertCard` (sky border, 🔔 icon): Enabled/Disabled badge, summary text, cooldown info block, last-fired timestamp, "No alerts fired yet" message, footer help text
- Full TypeScript wiring: `FeatureDriftAlertConfig` interface; `feature_drift_alert_config?` on `ChatMessage`; `attachFeatureDriftAlertConfigToLastMessage` Zustand action; SSE handlers + card render in both EventSource branches of `page.tsx`

**Tests:** 21 backend (3 constant/model + 7 helper unit + 5 REST endpoint + 6 regex) + 16 frontend (13 card + 3 store) = **37 new tests**. All passing. Backend lint: clean. Frontend build + TypeScript + lint: clean.

**What's next:**
- Track C: Cost-Sensitive Training via Misclassification Cost Matrix — "false positives cost $1000, false negatives cost $10"
- Track B: Comparative Model Improvement Plan — "create a ranked improvement roadmap for all my models"
- Track D: Deployment Geographic/Segment Drift Detection — "is my model drifting more in one region than another?"

---

## Day 86 (12:00) — Done

**Track D: Input Feature Drift Ranking by Importance** — complete.

Closes the "which of my drifted input features actually matters for my model?" analyst gap — distinct from `CovariateDriftAlertCard` (flags ALL features above a threshold, no importance weighting). Analysts say "which features drifted the most", "drift ranking by importance", "rank my drifted features", "feature drift risk", "drift vs importance", "top risk drift features", or 2 other NL variants (8 total in `_DRIFT_IMPORTANCE_PATTERNS`) and receive a `DriftImportanceCard` showing all model features ranked by risk_score = drift% × importance%, with per-row Priority badges (Critical / High / Medium / Low / No Drift), importance + drift progress bars, and drift details.

**What was built:**
- `compute_drift_importance_ranking(all_inputs, feature_ranges, feature_importances, max_features=15)` pure function in `core/analyzer.py`: collects production values per feature, computes OOR% for numeric (min/max bounds) and unseen% for categorical (known_categories), cross-references with normalized importance_pct from `identify_weak_features()`, computes risk_score = drift_pct × importance_pct, classifies priority (critical/high/medium/low/no_drift), sorts descending by risk_score, derives verdict (action_required/attention/monitoring/clear/no_importances)
- `GET /api/deploy/{id}/drift-importance-ranking` REST endpoint in `api/deploy.py`: loads 500 PredictionLogs, parses feature_ranges from PredictionPipeline, loads model + runs `identify_weak_features()` using `Deployment.feature_names`, calls pure function
- `_DRIFT_IMPORTANCE_PATTERNS` (8 NL variants) + handler in `api/chat.py`: loads logs + pipeline + model importances inline; injects verdict + critical/high counts into system_prompt; emits `{type:"drift_importance_ranking"}` SSE event; guarded by `ctx["deployment"]`
- `DriftImportanceCard` (rose=action_required / amber=attention / sky=monitoring / emerald=clear / muted=no_importances, 🎯 icon): verdict badge, feature-count + sample-count badges, priority alert callout (role="alert") when critical/high features exist, features table with Priority badge + dual progress bars (importance=indigo, drift=color-coded) + drift details, no-importances message, summary paragraph, sr-only figcaption
- Full TypeScript wiring: `DriftImportanceFeature` + `DriftImportanceRankingResult` interfaces; `drift_importance_ranking?` on `ChatMessage`; `attachDriftImportanceRankingToLastMessage` Zustand action; SSE handlers + card render in both EventSource branches of `page.tsx`

**Tests:** 49 backend (15 pure function + 18 regex + 4 endpoint + 3 chat handler) + 20 frontend (17 card + 3 store) = **69 new tests**. All passing. Backend lint: clean. Frontend build + TypeScript + lint: clean.

**What's next:**
- Track B: Comparative Model Improvement Plan — "create a ranked improvement roadmap for all my models"
- Track C: Cost-Sensitive Training via Misclassification Cost Matrix — "false positives cost $1000, false negatives cost $10"
- Track D: Deployment Feature Drift Webhook — fire webhook when a critical-priority feature drifts above threshold

---

## Day 86 (04:00) — Done

**Track C: Per-Class Custom Weighted Training** — complete.

Closes the "I want the model to pay more attention to my rare/important class" analyst gap — distinct from balanced class weighting (auto, uses inverse frequency) and from PerClassThresholdCard (advisory, post-training). Analysts say "give 3x weight to class churn", "class fraud=10, normal=1", "per-class custom weights", "5 times more weight to positive", or 4 other NL variants (8 total in `_CUSTOM_CLASS_WEIGHT_PATTERNS`) and training launches with their exact multipliers applied per class. Smart default: when pattern matches but no specific weights are named, applies 2x to the minority class automatically.

**What was built:**
- `train_single_model()` extended with `custom_class_weights: dict | None` + `label_encoder` params; converts str class names → int indices via LabelEncoder; applies `class_weight` dict param for LR/RF/LGBM, `sample_weight` array for GBC/XGB; neural_network gracefully trains normally (no sklearn support); skips CalibratedClassifierCV when custom weights applied; imbalance_strategy takes precedence when both specified
- `_train_in_background()` extended with `custom_class_weights` param; captures LabelEncoder from `prepare_features()` and passes to trainer
- `_CUSTOM_CLASS_WEIGHT_PATTERNS` (8 NL variants) + `_CUSTOM_WEIGHT_MULTIPLIER_RE` + `_CUSTOM_WEIGHT_TIMES_RE` + `_CUSTOM_WEIGHT_KV_RE` + `_detect_custom_class_weights()` helper (multiplier/times/kv extraction, case-insensitive class name matching) in `api/chat.py`; handler fires before `_BALANCE_TRAIN_PATTERNS` and `_TRAIN_PATTERNS`; classification-only guard; emits `training_started_event` with `custom_class_weights` dict
- `TrainingStartedResult.custom_class_weights?: Record<string, number>` TypeScript field; `TrainingStartedCard` shows amber ⚖️ "Custom weights" badge + per-class `class=Nx` chips + "with custom class weights" description text

**Tests:** 25 backend (11 regex/helper unit + 8 trainer function + 6 pattern/guard) + 9 frontend (badge present/absent, chips render, content, description text) = **34 new tests**. All passing. Backend lint: clean. Frontend build + TypeScript + lint: clean.

**What's next:**
- Track D: Input Feature Drift Ranking by importance — "which of my input features drifted the most AND matters most to my model?" (cross-referencing PSI drift scores with feature importance to prioritize monitoring attention)
- Track B: Comparative Model Improvement Plan — "create a ranked improvement roadmap for all my models" (aggregates improvement suggestions across all model runs, deduplicates, ranks by cross-model impact)
- Track C: Cost-Sensitive Training via Misclassification Cost Matrix — "false positives for fraud cost $1000, false negatives cost $10" (extends per-class weights with asymmetric FP/FN costs)

---

## Day 85 (20:00) — Done

**Track C: Per-Class Threshold Tuning** — complete.

Closes the "what confidence cutoff should I use for each class?" multiclass analyst gap. Analysts say "per-class threshold tuning", "optimize threshold for each class", "class-specific thresholds", or 5 other NL variants (8 total) and receive a `PerClassThresholdCard` showing the optimal confidence threshold for each class independently via one-vs-rest F1 maximization. Violet-bordered when actionable, with direction badges (↑ Raise/↓ Lower/✓ Default), F1-gain badges, per-class recommendations, and expandable sweep charts.

**What's next:**
- Track D: Deployment-level confidence threshold configuration — "set a minimum confidence of 70% for serving predictions" (distinct from PerClassThresholdCard which is advisory)
- Track B: Comparative Model Improvement Plan — "create a ranked improvement roadmap for all my models"
- Track C: Per-class weighted training — "train with higher weight on rare/important classes"

---

## Day 85 (12:00) — Done

**Track D: Deployment Throughput Assessment** — complete.

Closes the "can my deployment handle my batch workload?" analyst gap. Uses actual measured `response_ms` from PredictionLogs to derive p50/p95/p99 latency, max RPS, and time-to-process N records. Analysts say "how long to process 1000 predictions?", "throughput assessment", "how fast can my deployment process?", or 5 other NL variants and receive a `DeploymentThroughputCard` with latency stats, serial duration estimate, and verdict (instant/fast/moderate/slow/very_slow). Distinct from `cost_estimate` (quota/rate-limit-based) and `quota_runway` (quota burn rate).

**What's next:**
- Track C: Per-class threshold tuning — "optimize the threshold for each class in my multiclass model"
- Track B: Comparative Model Improvement Plan — "create a ranked improvement roadmap for all my models"
- Track D: Input Feature Drift Ranking — "which input features changed most in distribution since I deployed?"

---

## Day 85 (04:00) — Done

**Track D: Deployment Comparison Scorecard** — complete.

Closes the "I have multiple deployments — which one is performing best in production?" analyst gap. Ranks all active project deployments by composite score (usage volume 40%, feedback accuracy 30%, freshness 20%, SLA latency 10%). Analysts say "rank my deployments by performance", "deployment scorecard", "deployment leaderboard", or 5 other NL variants and receive a `DeploymentScorecardCard` with ranked entries, rank medals for top 3, Top Performer badge, composite score bars, and per-signal pills.

**What's next:**
- Track C: Per-class threshold tuning — "optimize the threshold for each class in my multiclass model"
- Track B: Comparative Model Improvement Plan — "create a ranked improvement roadmap for all my models"
- Track D: Deployment Capacity Planning — "how long will it take to process X predictions per day?"

---

## Day 84 (20:00) — Done

**Track B: Model Promotion Readiness Check** — complete.

Closes the "is my model good enough to deploy?" analyst gap. Synthesizes all available quality signals (primary metric, CV stability, overfitting gap, Brier calibration, data volume, sample-to-feature ratio) into a single go/no-go checklist with per-gate pass/warn/fail statuses. Analysts say "promotion readiness check", "ready to promote?", "deployment gate", "go/no-go assessment", or "pre-deployment checklist" and receive a `PromotionReadinessCard` with blocking issues highlighted and a "Deploy my model" CTA button when ready.

**What was built:**
- `compute_promotion_readiness(metrics, algorithm, problem_type, n_rows, n_features)` pure function in `core/advisor.py`: 6 gates (model quality, CV stability, overfitting risk, calibration, data volume, sample-feature ratio); verdicts: ready/ready_with_warnings/not_ready; aggregates blocking_issues and warnings lists
- `GET /api/models/{run_id}/promotion-readiness` REST endpoint in `api/models.py`: loads run + feature set + CSV, computes n_rows/n_features, calls pure function; 400 for non-done runs, 404 for missing run
- `_PROMOTION_READINESS_PATTERNS` (9 NL variants: promotion readiness, ready to promote, pre-deployment checklist, deployment gate, go/no-go, production readiness check, all checks pass, run a readiness check) + handler in `api/chat.py`; loads dataset + feature set for accurate n_rows/n_features; emits `{type:"promotion_readiness"}` SSE event; injects verdict + gate summary into system_prompt
- `PromotionReadinessCard` (emerald=ready / amber=ready_with_warnings / rose=not_ready, 🚀/⚠️/🛑 icon): verdict badge, passed/warn/fail count, per-gate rows with Pass/Warning/Fail badges and recommendation text, blocking issues callout with role="alert", summary paragraph, "Deploy my model →" CTA button (ready/ready_with_warnings only)
- Full TypeScript wiring: `PromotionReadinessGate`, `PromotionReadinessResult` interfaces in `lib/types.ts`; `promotion_readiness?` on `ChatMessage`; `api.models.promotionReadiness()` client method; `attachPromotionReadinessToLastMessage` Zustand action; SSE handlers + card render in both EventSource branches of `page.tsx`

42 backend + 26 frontend = **68 new tests**. Backend lint: clean. Frontend build + TypeScript + lint: clean.

**What's next:**
- Track C: Per-class threshold tuning — "optimize the threshold independently for each class in my multiclass model"
- Track B: Comparative Model Improvement Plan — "create a ranked roadmap to improve all my models"
- Track D: Deployment Comparison Scorecard — "rank all my deployments by production performance"

---

## Day 84 (12:00) — Done

**Track D: Automated Weekly Monitoring Digest Webhook** — complete.

Closes the "passive monitoring" gap — analysts never need to remember to check their model health. AutoModeler automatically computes all monitoring signals (anomalies, drift, retraining readiness, output distribution shift) every week and dispatches a complete health report to registered `weekly_digest` webhooks (Slack, Teams, PagerDuty, Zapier).

**What was built:**
- `WeeklyDigestConfig` SQLModel table per deployment (day_of_week, send_hour, last_sent_at)
- `should_send_weekly_digest()` pure function: fires when today matches day + hour + not yet sent today
- `_run_weekly_digest()`: computes full monitoring digest inline, dispatches via `EVENT_WEEKLY_DIGEST` webhook
- Scheduler extended to check enabled digest configs in the same 60s loop as batch jobs
- `EVENT_WEEKLY_DIGEST` added to `ALL_EVENTS` in `core/webhook.py`
- REST: `GET/PUT/DELETE /api/deploy/{id}/weekly-digest-config`
- Chat: 8 NL variants, day-of-week + time parsing, enable/disable/status intent
- `WeeklyDigestConfigCard` (teal/slate border, 📅 icon)

22 backend + 18 frontend = **40 new tests**. Backend lint: clean. Frontend build + TypeScript: clean.

**What's next:**
- Track C: Per-class threshold tuning — "optimize the threshold for each class in my multiclass model"
- Track B: Model Promotion Readiness Check — "is my model ready to move to production?"
- Track D: Deployment Capacity Planning — "how long will it take to process X predictions?"

---

## Day 84 (04:00) — Done

**Track D: Deployment Prediction Distribution Comparison via Chat** — complete.

Closes the "did my retrained model actually change how it predicts in production?" analyst gap. Analysts can ask "is my new deployment predicting higher values?", "compare my deployment prediction distributions", "deployment version prediction comparison", or "old vs new deployment predictions" (8 NL variants) and receive a `DeploymentPredictionDistributionCard` comparing production prediction distributions across the two most recent active deployments.

**What was built:**
- `compute_deployment_prediction_comparison(baseline_logs, current_logs, problem_type)` pure function in `core/analyzer.py`: regression path computes mean/median/std/min/max/p25/p75 for each deployment and determines direction verdict (current_higher / current_lower / similar); classification path computes class frequency distributions and largest shift (distribution_shifted / similar).
- `GET /api/deploy/{id}/prediction-distribution-comparison?vs=<baseline_id>` REST endpoint in `api/deploy.py`.
- `_DEPLOY_PRED_DIST_COMPARE_PATTERNS` (8 NL variants) + handler in `chat.py`: auto-selects most recent OTHER active deployment, loads 200 logs per deployment, emits `{type:"deploy_pred_dist_compare"}` SSE event.
- `DeploymentPredictionDistributionCard` (emerald/rose/amber/sky/slate by verdict): stat grids for regression, class-shift table for classification.

35 backend + 19 frontend = **54 new tests**. Backend lint: clean. Frontend build + TypeScript: clean.

**What's next:**
- Track B: Auto-trigger weekly monitoring digest via webhook — scheduled digest computation + webhook dispatch
- Track C: Per-class threshold tuning — "optimize the threshold independently for each class in my multiclass model"
- Track D: Input feature importance drift — "which input features changed most in distribution since I deployed?"

---

## Day 83 (20:00) — Done

**Track C: Production Feedback Threshold Optimizer via Chat** — complete.

Closes the "was my confidence threshold right for production?" analyst gap. Analysts can ask "what confidence threshold maximizes my production F1?", "optimize my classification threshold from feedback", "best threshold based on actual outcomes", or "real-world threshold analysis" (8 NL variants) and receive a `ProductionThresholdOptimizerCard` showing the optimal confidence threshold derived from real `FeedbackRecord` outcomes. Distinct from `ThresholdAnalysisCard` (training-time data only).

**What was built:**
- `compute_production_threshold_optimizer(feedback_pairs)` pure function in `core/validator.py`: sweeps 19 thresholds (0.05–0.95), precision/recall/F1/coverage at each. Returns optimal threshold (max F1), verdict (improved/same), comparison vs 0.5 default.
- `GET /api/deploy/{id}/production-threshold-optimizer` REST endpoint in `api/deploy.py`: joins `FeedbackRecord` + `PredictionLog` for (confidence, predicted_label, actual_label) triples. `no_data` when < 5 pairs.
- `_PROD_THRESHOLD_OPT_PATTERNS` (8 NL variants) + handler in `chat.py`: classification-only guard, feedback_pairs built inline.
- `ProductionThresholdOptimizerCard` (amber=improved / emerald=same / gray=no_data): stat grid (F1/precision/recall/coverage), comparison row (overall accuracy / current F1 / F1 gain), Recharts sweep chart with dashed optimal reference line.

36 backend + 17 frontend = **53 new tests**. Backend lint: clean. Frontend build + TypeScript: clean.

**What's next:**
- Track D: Prediction value comparison across deployment versions — "is my retrained model predicting higher values on production vs my previous deployment?"
- Track B: Auto-trigger weekly monitoring digest via webhook — scheduled digest computation + webhook dispatch
- Track C: Per-class threshold tuning — "optimize the threshold independently for each class in my multiclass model"

---

## Day 83 (12:00) — Done

**Track B: Deployment Monitoring Signal Digest via Chat** — complete.

Closes the "I have 15+ monitoring cards but need to ask 6 questions to get a complete picture" analyst gap. Analysts can ask "show all my monitoring signals", "monitoring signal digest", "deployment diagnostics", or "monitoring overview" (8 NL variants) and receive a `MonitoringDigestCard` showing ALL active monitoring signal verdicts in a single compact "mission control" view.

**What was built:**
- `compute_deployment_monitoring_digest()` pure function in `core/analyzer.py`: aggregates 5 signals (output anomalies, value trend, dist shift, retraining readiness, usage activity), each mapped to green/amber/red severity. Overall health: critical/warning/watching/healthy. Score = 100 − red×25 − amber×10. Priority actions (top 3).
- `GET /api/deploy/{id}/monitoring-digest` REST endpoint
- `_MONITORING_DIGEST_PATTERNS` (8 NL variants) + handler in `chat.py`
- `MonitoringDigestCard` (blue/amber/orange/rose, 📡): signal list with severity dots, priority actions, summary

34 backend + 17 frontend = **51 new tests**. Backend lint: clean. Frontend build + TypeScript: clean.

**What's next:**
- Track C: Classification threshold optimizer using production feedback data — "what confidence threshold maximizes my real-world F1 score?"
- Track D: Prediction value comparison across deployment versions — "is my retrained model predicting higher values on production?"
- Track B: Auto-trigger weekly monitoring digest via webhook — scheduled digest computation + webhook dispatch

---

## Day 83 (04:00) — Done

**Track D: Production Prediction Value Trend via Chat** — complete.

Closes the "are my model's outputs systematically drifting upward or downward over time?" analyst gap. Analysts can ask "are my predictions trending up?", "prediction value trend", "show me how my predictions have changed over time", "regression output trend" (8 NL variants) and receive a `PredictionValueTrendCard` showing a day-by-day LineChart of mean prediction values with direction verdict (trending_up / stable / trending_down) and overall % change.

**What was built:**
- `compute_prediction_value_trend(logs_data, period, n_periods)` pure function in `core/analyzer.py`: period bucketing, per-period stats, numpy polyfit slope, direction from overall_change_pct
- `GET /api/deploy/{id}/prediction-value-trend?period=day&n=200` REST endpoint in `api/deploy.py`
- `_PRED_VALUE_TREND_PATTERNS` (8 NL variants) + handler in `chat.py`: regression-only guard, ascending-order log load
- `PredictionValueTrendCard` (emerald/rose/amber/gray, 📈): direction badge, first/last stats, net-change pct, Recharts LineChart

38 backend + 18 frontend = **56 new tests**. Backend lint: clean. Frontend build + TypeScript: clean.

**What's next:**
- Track C: Classification threshold optimizer via chat — "what confidence threshold maximizes my F1 score?" (sweeps thresholds with precision/recall tradeoff — distinct from the existing `ThresholdAnalysisCard` which is for pre-deployment advice; this would be a deployment-level card using production data)
- Track B: Multi-signal monitoring dashboard summary — auto-generate a weekly deployment health digest combining all monitoring signal verdicts
- Track D: Prediction value comparison across deployments — "is my new model predicting higher values than my old one?"

---

## Day 82 (20:00) — Done

**Track D: Retraining Readiness Assessment via Chat** — complete.

Closes the "I have all these monitoring cards but which one actually tells me when to retrain?" analyst gap. Analysts can ask "should I retrain my model?", "retrain recommendation", "is my model degrading?", "retraining readiness" (9 NL variants) and receive a `RetrainingReadinessCard` with a composite 0-100 urgency score aggregating model age, prediction anomaly rate, confidence trend, and feedback accuracy into a single verdict (stable/monitor/retrain_soon/retrain_now).

**What was built:**
- `compute_retraining_readiness(age_days, anomaly_rate, confidence_trend, feedback_verdict, psi_critical_count, output_shift_verdict)` pure function in `core/analyzer.py`
- `GET /api/deploy/{id}/retraining-readiness` REST endpoint in `api/deploy.py`
- `_RETRAIN_READINESS_PATTERNS` + handler in `chat.py`
- `RetrainingReadinessCard` with per-signal rows, score bar, recommendations, and "Retrain Now" CTA

48 backend + 18 frontend = **66 new tests**. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: Production prediction value trend chart — "are my regression predictions trending up or down over time?"
- Track C: Classification threshold optimizer — "what confidence threshold maximizes my F1 score?"
- Track B: Multi-signal monitoring dashboard summary — auto-generate a once-a-week deployment health digest

---

## Day 82 (12:00) — Done

**Track C: Minimum Viable Feature Set via Chat** — complete.

Closes the "which of my current features can I safely drop?" analyst gap. Analysts can ask "which features can I drop?", "simplify my model", "reduce my feature set", "feature pruning", "fewest features for good predictions", "which features are redundant?", "what's the minimum features needed?", or "can I drop any features?" (8 NL variants) and receive a `MinFeatureSetCard` showing the result of greedy backward elimination.

**What was built:**
- `compute_min_viable_feature_set(X, y, feature_names, model_class, model_params, problem_type, tolerance=0.02, max_rows=2000, cv=3)` pure function in `core/validator.py`: sub-samples for speed, computes baseline CV score, fits model for importances, greedily removes least-important features while score loss ≤ tolerance. Returns n_original, n_minimal, can_simplify, features_retained/dropped lists, baseline/minimal scores, reduction_pct, summary.
- `GET /api/models/{model_run_id}/min-feature-set?tolerance=0.02` REST endpoint in `api/validation.py`.
- `_MIN_FEATURE_SET_PATTERNS` (8 NL variants) + handler in `chat.py`.
- `MinFeatureSetCard` (sky border when can_simplify): score comparison grid, ranked feature list with importance bars, retained/dropped two-column summary.

35 backend + 20 frontend = **55 new tests**. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track C: Class imbalance detection + SMOTE/class-weights handling — "my model is biased toward the majority class"
- Track D: Automated retraining signal aggregation — when drift + output anomalies + PSI all point to degradation, proactively recommend retraining with a single confidence score
- Track B: Multi-model ensemble via chat — "combine my top 3 models into an ensemble"

---

## Day 81 (12:00) — Done

**Track D: Prediction Output Distribution Shift via Chat** — complete.

Closes the "is my model producing systematically different output values in production?" analyst gap. Analysts can ask "has the distribution of my predictions shifted?", "output distribution shift", "are my predictions shifting over time?", or "are my model outputs behaving differently in production?" (8 NL variants) and receive a `PredictionOutputDistributionCard` comparing the statistical distribution of production predictions vs training-time predictions using a Kolmogorov-Smirnov test.

**What was built:**
- `compute_prediction_output_distribution_shift(training_preds, production_preds, n_bins=10)` pure function in `core/analyzer.py`: runs `scipy.stats.ks_2samp()`, computes per-distribution stats (mean/std/min/p25/median/p75/p95/max), derives mean_shift and mean_shift_pct, builds aligned histograms. Verdicts: significant_shift (p < 0.01 or |shift| > 30%), moderate_shift (p < 0.05 or |shift| > 10%), stable. Raises ValueError for < 10 samples in either list.
- `GET /api/deploy/{id}/output-distribution-shift?n=100` REST endpoint in `api/deploy.py`: loads PredictionLogs for production_preds; re-runs model on training CSV to get training_preds; returns no_data when < 10 production predictions.
- `_OUTPUT_DIST_SHIFT_PATTERNS` regex (8 NL variants) + handler in `chat.py`: inline pipeline load + predict; injects KS + mean_shift_pct + verdict into system_prompt; emits `{type:"output_distribution_shift"}` SSE event.
- `PredictionOutputDistributionCard` (emerald=stable / amber=moderate_shift / rose=significant_shift / gray=no_data, 📊 icon): verdict badge, stats row (KS statistic/p-value/mean shift%), Distribution Comparison table (training vs production side-by-side), Recharts BarChart histogram overlay (gray=training, indigo=production), summary paragraph, sr-only figcaption.

**Distinct from:** `DriftCard` (compares input feature distributions with z-score/TVD), `PredictionOutputAnomalyCard` (individual outlier predictions), `compute_training_vs_production` (requires labeled feedback for accuracy comparison). Works without ground-truth labels.

37 backend + 24 frontend = **61 new tests**. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track C: Minimum viable feature set analysis — "what's the smallest set of features that preserves model accuracy?" (greedy backward elimination)
- Track B: Automated retraining recommendations — when multiple signals (drift + output anomalies + feedback accuracy) point to degradation, proactively suggest retraining
- Track D: API usage analytics dashboard enhancement — richer charts for the deployment analytics view

---

## Day 81 (04:00) — Done

**Track D: Prediction Output Anomaly Detection via Chat** — complete.

Closes the "are my production predictions making sense?" analyst gap. Analysts can ask "any unusual predictions?", "prediction outliers", "anomalous outputs", "weird model outputs", "predictions that are unusually high", or "suspicious model outputs" and receive a `PredictionOutputAnomalyCard` in chat.

**What was built:**
- `compute_prediction_output_anomalies(logs, problem_type, z_score_threshold=2.5, confidence_threshold=0.55, max_anomalies=10)` pure function in `core/analyzer.py`: regression path computes z-scores from prediction_numeric values and flags |z| > threshold; classification path flags predictions with confidence below threshold. Both paths sort anomalies appropriately (z-score desc for regression, confidence asc for classification), cap at max_anomalies, compute anomaly_rate, and return a verdict (no_anomalies/few_anomalies/many_anomalies). Returns per-entry id, prediction_value, deviation string, reason, input_summary chips (first 3 features). Handles all-identical values (std=0) gracefully with no_anomalies. Raises ValueError for < 5 logs.
- `GET /api/deploy/{id}/output-anomalies?n=50` REST endpoint in `api/deploy.py`: loads last N PredictionLogs, converts to dicts, calls pure function, returns enriched result with deployment_id. Returns no_data verdict when < 5 logs.
- `_OUTPUT_ANOMALY_PATTERNS` regex (8 NL variants) + handler in `chat.py`: guarded by `ctx["deployment"]`; loads last 50 PredictionLogs; calls `compute_prediction_output_anomalies()`; injects verdict + summary into system_prompt; emits `{type:"output_anomalies"}` SSE event. Handles the not-enough-data case gracefully.
- `PredictionOutputAnomalyCard` (rose=many_anomalies, amber=few_anomalies, emerald=no_anomalies, gray=no_data, 🔍 icon): verdict badge, problem-type badge, n-total badge, stats grid (regression: mean/std/anomaly-count; classification: mean-confidence/min-confidence/anomaly-count), per-anomaly `AnomalyRow` with prediction value, deviation badge, z-score/confidence, reason text, relative timestamp, input feature chips, summary paragraph, sr-only figcaption.

**Distinct from:** `AnomalyCard` (input data anomalies via IsolationForest), `CovariateDriftAlertCard` (input distribution shifts), `PredictionErrorCard` (training-time wrong predictions), `ConfidenceDistributionCard` (overall aggregate confidence spread). This is the first feature to look at the *outputs* of production predictions for unusual patterns.

34 backend + 22 frontend = **56 new tests**. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track C: Minimum viable feature set analysis — "what's the smallest set of features that preserves model accuracy?" (greedy backward elimination)
- Track D: Prediction output distribution shift — compare distribution of production predictions vs training predictions to detect model behavior change
- Track B: Automated retraining recommendations — when multiple signals (drift + output anomalies + feedback accuracy) point to degradation, proactively suggest retraining

---

## Day 80 (04:00) — Done

**Track C: Class-Conditional Feature Importance via Chat** — complete.

Closes the "what makes my model predict each outcome differently?" analyst gap. Analysts can ask "what features drive churn predictions?", "feature importance per class", "class-specific feature breakdown", "what makes the model predict X vs Y?", or "why does the model predict different classes?" and receive a `ClassFeatureImportanceCard` with per-class collapsible panels showing which features deviate most from the training average for each predicted class.

- `compute_class_conditional_importance(model, X, y_pred, feature_names, class_names)` pure function in `core/explainer.py`: per-class feature deviation analysis, importance-weighted, top 8 per class.
- `GET /api/models/{run_id}/class-feature-importance` endpoint: classification-only (400 regression, 404 unknown).
- `_CLASS_FEAT_IMP_PATTERNS` (8 NL variants) + handler in `chat.py`: injects per-class highlights into system_prompt.
- `ClassFeatureImportanceCard`: collapsible per-class panels, ↑/↓/≈ direction labels, proportional importance bars.

**Distinct from:** `GlobalFeatureImportance` (averages all predictions), `PDP` (marginal effect), `LocalExplanationCard` (single row). 32 backend + 20 frontend = **52 new tests**. Total: **5180 backend / 2963 frontend = 8143**.

---

## Day 79 (20:00) — Done

**Track C: Sample Size Adequacy Analysis via Chat** — complete.

Closes the "do I have enough data to trust this model?" analyst gap. Analysts can ask "do I have enough training data?", "is my dataset big enough?", "how many more rows do I need?", "is my sample size sufficient?", or "sample size check" and receive a `SampleSizeAdequacyCard` with a verdict (adequate/borderline/insufficient), coverage progress bar, feature/row ratio, optional CV stability badge, and concrete shortfall count.

- `compute_sample_size_adequacy(n_rows, n_features, problem_type, n_classes, cv_std)` pure function in `core/analyzer.py`: 10× rule of thumb, three verdicts, cv_stable check.
- `GET /api/models/{run_id}/sample-size-adequacy` endpoint: loads dataset, prepares features, reads cv_std from metrics.
- `_SAMPLE_SIZE_PATTERNS` (8 NL variants) + handler in `chat.py`: injects verdict + shortfall into system_prompt.
- `SampleSizeAdequacyCard`: emerald/amber/rose per verdict, ARIA progressbar, feature/ratio badge, CV stability badge.

**Distinct from:** `OverfittingAnalysisCard` (train/CV gap) and `DataQualityImpactCard` (outlier effect). 35 backend + 21 frontend = **56 new tests**. Total: **5148 backend / 2943 frontend = 8091**.

---

## Day 79 (12:00) — Done

**Track C: Model Confidence Distribution via Chat** — complete.

Closes the "how confident/decisive is my model overall?" analyst question. Analysts can ask "how confident is my model?", "confidence distribution", "show confidence histogram", "how certain are my predictions?", "is my model decisive or uncertain?", or "distribution of prediction probabilities" and receive a `ConfidenceDistributionCard` showing a histogram of max-class probabilities, a High/Medium/Low tier breakdown (high ≥80%, medium 50–80%, low <50%), per-class mean confidence, and a decisiveness verdict (decisive/moderate/uncertain).

- `compute_confidence_distribution(y_proba, y_pred, class_names, n_bins)` pure function in `core/validator.py`: bins max-class probabilities, computes mean/median, segments tiers, derives decisiveness verdict.
- `GET /api/models/{model_run_id}/confidence-distribution` endpoint: classification-only (400 regression/non-proba, 404 unknown).
- `_CONFIDENCE_DIST_PATTERNS` (8 NL variants) + handler in `chat.py`: finds best/selected classification run, computes distribution, injects context into system prompt.
- `ConfidenceDistributionCard`: emerald/amber/rose per decisiveness, Recharts BarChart with color-coded bins, per-class confidence progressbars, ARIA accessibility.

**Distinct from:** `ThresholdAnalysisCard` (sweeps thresholds for precision/recall) and `CalibrationCheckCard` (checks probability reliability). 31 backend + 20 frontend = **51 new tests**. Total: **5113 backend / 2922 frontend = 8035**.

---

## Day 79 (04:00) — Done

**Track C: Classification Threshold Advisor via Chat** — complete.

Closes the "what probability cutoff should I use?" analyst gap. Analysts can ask "what threshold should I use?", "optimal cutoff for my model", "precision recall tradeoff", or "help me choose a threshold" and receive a `ThresholdAnalysisCard` showing the full precision/recall/F1 sweep (0.05–0.95) with three plain-English recommendations.

- `compute_threshold_analysis(y_true, y_proba, class_names)` pure function in `core/validator.py`: sweeps 19 thresholds, computes precision/recall/F1/positive_rate at each, identifies max_f1/high_recall/high_precision options. Binary: uses positive class probability; multiclass: uses max-class confidence as proxy.
- `GET /api/models/{run_id}/threshold-analysis` endpoint in `api/validation.py`: classification-only (400 for regression), loads model, computes thresholds, returns sweep + recommendations + current_metrics + prevalence.
- `_THRESHOLD_ADVISOR_PATTERNS` (8 NL variants) + handler in `chat.py`: guards on classification model runs, computes sweep, injects plain-English guidance into system_prompt, emits `{type:"threshold_analysis"}` SSE event.
- `ThresholdAnalysisCard` (amber border, 🎯 icon): Recharts LineChart with precision/recall/F1 curves + dashed reference line at best F1, three RecommendationRow options with threshold %, scores, and business-context descriptions ("churn/fraud use cases → High Recall"), "Which threshold is right for me?" guidance table.
- Full type wiring: `ThresholdSweepPoint`, `ThresholdRecommendation`, `ThresholdAnalysisResult` TypeScript interfaces; `threshold_analysis?` on `ChatMessage`; `attachThresholdAnalysisToLastMessage` Zustand action; SSE handler + card render in `page.tsx`; `api.models.thresholdAnalysis()` client method.

**Distinct from:** `_CONFIDENCE_THRESHOLD_PATTERNS` (sets a minimum confidence gate for prediction serving) — this is a pre-deployment advisor about what cutoff produces the best classification performance trade-off.

31 backend + 21 frontend = **52 new tests**. Total: **5082 backend / 2902 frontend = 7984**, all passing. Backend lint: clean. Frontend build + lint: clean.

---

## Day 78 (20:00) — Done

**Track C: Target Leakage Detection via Chat** — complete.

Closes the "am I accidentally cheating with this feature?" analyst gap. Detects features with suspiciously high correlation with the target (Pearson for numeric targets, normalized mutual information for categorical targets). High-risk (≥ 90%) vs moderate-risk (≥ 75%) classification. `_TARGET_LEAKAGE_PATTERNS` (8 NL variants: "is there target leakage?", "check for data leakage", "any leaky features?", etc.). `TargetLeakageCard` (emerald/amber/rose by verdict, correlation bars, Risk badges, severe alert callout). Distinct from FeatureRedundancyCard (detects collinearity between features, not with target). 27 backend + 21 frontend = **48 new tests**. Total: **5051 backend / 2881 frontend = 7932**, all passing.

---

## Day 78 (12:00) — Done

**Track C: Feature Redundancy Detection via Chat** — complete.

Closes the "are my features measuring the same thing?" analyst gap. Detects all numeric feature pairs with Pearson |correlation| ≥ 0.85 (configurable), clusters them into groups via union-find, recommends which to keep (higher variance wins). `_REDUNDANCY_PATTERNS` (8 NL variants: "are any features redundant?", "multicollinearity", "which features measure the same thing?", etc.). `FeatureRedundancyCard` (emerald/amber/rose by verdict, correlation bars, Keep/Drop badges). Distinct from FeatureSelectionCard (importance-based weak feature removal). 24 backend + 21 frontend = **45 new tests**. Total: **5024 backend / 2860 frontend = 7884**, all passing.

---

## Day 78 (04:00) — Done

**Track C: Overfitting/Underfitting Detection via Chat** — complete. `OverfittingAnalysisCard`, `compute_overfitting_analysis()` pure function comparing train score vs CV score, REST endpoint, chat regex (8 NL variants) + handler + SSE emit. 46 backend + 25 frontend = 71 new tests. Verdicts: well_fit / mild_overfit / overfit / underfit.

---

## Day 77 (20:00) — Done

**Track C: Data Quality Impact on Model Performance** — complete. `DataQualityImpactCard`, `compute_data_quality_impact()` pure function using IsolationForest, REST endpoint, chat regex (9 NL variants) + handler + SSE emit. 39 backend + 22 frontend = 61 new tests.

---

## Day 77 (12:00) — Done

**Track C: Feature Engineering Impact Analysis** — complete. `FeatureEngineeringImpactCard`, `compute_feature_engineering_impact()` pure function, REST endpoint, chat regex + handler + SSE emit. 28 backend + 12 frontend = 40 new tests.

---

## Day 76 (12:00) — Done

**Track B: Counterfactual Explanation** — complete.

Closes the "what specifically needs to change for this prediction to flip?" gap. Analysts can ask "what would save this customer?", "counterfactual for row 5", or "minimum intervention to change the prediction" and receive a `CounterfactualCard` showing the minimal feature changes needed to flip a classification prediction across the decision boundary.

**What was built:**
- `compute_counterfactual()` pure function in `core/deployer.py`: greedy finite-difference gradient search, classification-only.
- `_COUNTERFACTUAL_PATTERNS` regex (9 NL variants) + handler in `chat.py`.
- `CounterfactualCard` (amber/rose): original + counterfactual prediction boxes, feature changes table with ▲/▼ arrows.
- 28 backend + 23 frontend = **51 new tests**.

---

## Day 76 (20:00) — Done

**Track B: Population-Level Counterfactual** — complete.

Closes the "what one change would save the most customers?" gap. Analysts can now ask "what change would flip the most predictions?" or "most impactful intervention for the cohort" and receive a `PopulationCounterfactualCard` showing the single feature intervention that flips the most predictions across the at-risk cohort — the operationally actionable complement to per-row counterfactual.

**What was built:**
- `compute_population_counterfactual()` pure function in `core/deployer.py`: runs greedy counterfactual for each row (max 20), aggregates primary changed feature per flip, returns dominant (feature, direction) pair + flip rate + feature_summary sorted by flip_count desc.
- `_POPULATION_CF_PATTERNS` regex (9 NL variants) + handler in `chat.py`; guard: deployment + dataset + feature_set + model_runs + classification + not per-row counterfactual.
- `PopulationCounterfactualCard` (amber/rose, 🎯): flip rate progress bar (ARIA), dominant intervention highlight, feature breakdown table, empty state, sr-only figcaption.
- `PopulationCFFeatureSummary` + `PopulationCounterfactualResult` TypeScript interfaces; `attachPopulationCounterfactualToLastMessage` Zustand action; SSE handler wired in page.tsx.
- 46 backend + 29 frontend = **75 new tests**.

---

## Day 76 (04:00) — Done

**Track B: Predictive Cohort Monitoring** — complete.

Closes the "how is my cohort trending over time?" gap. Analysts can now ask "how has my top cohort changed?" or "cohort evolution" and receive a `CohortEvolutionCard` showing composition shifts across all historical dataset uploads.

**What was built:**
- `compute_cohort_evolution()` pure function in `core/deployer.py`: reuses `compute_prediction_cohort()` per period, computes period-over-period categorical shifts (≥5pp threshold), caps at 6 most-recent periods, generates plain-English summaries.
- `_COHORT_EVOLUTION_PATTERNS` regex (9 NL variants) + handler in `chat.py`; guards: deployment exists, ≥2 scoreable DataFrames from DB, cohort_event and ranked_pred_event not already fired.
- `CohortEvolutionCard` (violet, 📈): period timeline with `PeriodNode` (mini categorical bars) and `ShiftConnector` arrows; "Notable Composition Shifts" section with `ShiftRow` (emerald/rose per direction).
- 7 new TypeScript interfaces; `attachCohortEvolutionToLastMessage` Zustand action; SSE handler wired in page.tsx.
- 47 backend + 20 frontend = 67 new tests.

---

## Day 75 (20:00) — Done

**Track B: Prediction Baseline Context on Live Dashboard** — complete.

Closes the "why is this prediction high/low?" gap on the VP-facing `predict/[id]` dashboard. Previously the ExplanationCard showed per-feature contribution bars but had no top-level context telling the analyst whether the current prediction is above or below a "typical" case.

**What was built:**

Updated `explain_prediction()` in `core/deployer.py` to:
- Build a baseline feature vector (all features at encoded training-data means)
- Compute `baseline_prediction = model.predict(baseline_x)` — what the model would output for a completely typical input
- Regression: `delta = prediction − baseline`, `pct_change`, `direction` (`above_baseline`/`below_baseline`/`at_baseline`)
- Classification (predict_proba): `current_confidence`, `baseline_confidence`, `direction` (`class_changed`/`same_class`)
- All new fields added to return dict: `baseline_prediction`, `delta`, `pct_change`, `direction`, `current_confidence`, `baseline_confidence`

`PredictionExplanation` TypeScript interface in `types.ts` extended with 6 optional fields (backward-compatible).

`BaselineComparisonBanner` component inserted inside `ExplanationCard` on `predict/[id]/page.tsx`:
- Regression: color-coded banner (emerald=above/rose=below/muted=at_baseline), ▲/▼/→ arrow + delta + pct_change, plain-English description ("Your inputs raised the prediction above what a typical case would produce")
- Classification: shows baseline class + confidence percentages, message about class change vs same-class
- `data-testid="baseline-comparison"` for test targeting; `aria-label` for a11y
- Does not render when `baseline_prediction` is absent (fully backward-compatible with old API responses)

Fixed pre-existing `test_prediction_explain.py` client fixture: missing model imports before `SQLModel.metadata.create_all()` caused sporadic `no such table: project` failures. Fixed by adding all 21 model modules before `create_all()`.

**Tests:** 7 backend (3 integration + 3 endpoint + 1 pure-function) + 15 frontend (10 page-render component tests + 5 type-level) = 22 new tests. Total: **4746 backend + 2702 frontend = 7448**, all passing. Backend lint: clean. Frontend build + lint: clean.

Key learning: `jest.resetModules()` in `beforeEach` causes `useState` hook conflicts in React Testing Library because the React module gets re-imported with a fresh instance while the old instance's hooks are still registered. Use static module-level `require()` for page components instead.

---

## Day 74 (20:00) — Done

**Track E: Analyst-Facing Model Quality Score** — complete.

`compute_model_quality_score()` pure function in `core/advisor.py`. Tier thresholds for regression (0.85/0.70/0.55) and classification (0.90/0.80/0.70). CV instability penalty: `cv_std > 0.10` downgrades label one step. Returns quality_label, quality_score (0–100), color, reasoning bullets, recommendation. `GET /api/models/{run_id}/quality-score` REST endpoint (fixed: problem_type from FeatureSet, metrics from `json.loads()`). `_QUALITY_PATTERNS` (10 NL variants) in `chat.py`; SSE → `model_quality_score` event. `ModelQualityScoreCard` in chat + `ModelQualityBadge` inline in RunCard. 35 backend + 0 frontend tests. Total: **4672 backend + 2645 frontend = 7317**, all passing. Backend lint: clean. Frontend build + lint: clean.

---

## Day 74 (12:00) — Done

**Track C: Proactive Ensemble Auto-Suggest** — complete.

`_ENSEMBLE_AUTO_THRESHOLD = 0.75` module-level constant (importable by tests). `Project.last_ensemble_suggest_run_count: Optional[int]` field + DB migration. Proactive block in `send_message()`: fires when `ensemble_event is None` (user didn't ask) AND non-ensemble done runs exist AND no ensemble trained AND best score < 0.75 AND run count differs from last stored value. Builds the same payload as the explicit `_ENSEMBLE_PATTERNS` handler (voting/stacking options, recommended algo, summary) but adds `auto_suggested: True`. Injects a "AutoModeler Proactive Suggestion" block into `system_prompt` so Claude acknowledges the low score. Persists `last_ensemble_suggest_run_count = _ae_run_count` so suggestion fires once per batch of runs and resets when new runs complete. `EnsembleRecommendationResult.auto_suggested?: boolean` type field. `EnsembleRecommendationCard`: when `auto_suggested=True`, renders amber `role="note"` banner ("💡 AutoModeler noticed your model score is below target") and changes heading to "Accuracy Below Target — Try an Ensemble". 24 backend + 11 frontend = 35 new tests. Total: **4637 backend + 2645 frontend = 7282**, all passing. Backend lint: clean. Frontend build + lint: clean.

Key learning: Module-level threshold constants should be defined at module scope (not inside the function body) so tests can import and verify them directly.

---

## Day 74 (04:00) — Done

**Track D: Cross-Deployment Prediction Comparison via Chat** — complete.

`_CROSS_DEPLOY_PRED_PATTERNS` (8 NL variants: "compare what my models would predict for X", "run my models side by side", "cross-deployment comparison", "which model gives the highest prediction?"). Handler in `send_message()`: finds all active deployments for project, extracts `key=value` features via `_extract_multi_feature_prediction()`, fills missing with `feature_means`, runs `predict_single()` on each (capped 4), computes winner (highest regression prediction / highest confidence classification), emits `{type:"cross_deploy_prediction"}` SSE event. Single-deployment guard → system_prompt hint to retrain and deploy a second version. `CrossDeployPredictionCard` (orange border, 🔀 icon): feature chips + defaults note, comparison table with 🏆 winner row, Env badge, prediction + CI/confidence, deployed date, sr-only figcaption. 25 backend + 21 frontend = 46 new tests. Total: **4613 backend + 2634 frontend = 7247**, all passing. Backend lint: clean. Frontend build + lint: clean.

Key learning: The existing `POST /api/predict/compare` REST endpoint powers the `CompareModelsCard` on the prediction page but has no chat interface — always grep for existing REST endpoints before building new ones, and ask whether a chat handler is missing.

---

## Day 73 (20:00) — Done

**Track E: "Explain this finding" Direct-Send + Track B: Goal Seek Lock Toggle** — complete.

`handleSendMessage` in `project/[id]/page.tsx` gained `directText?: string` optional param; `AutoInsightCard` + `GoalSeekCard` now call it directly (no extra Send click). Bug caught: `onClick={handleSendMessage}` on Send button passed SyntheticEvent as `directText`; fixed to `onClick={() => handleSendMessage()}`. `GoalSeekCard`: `useState<Set<string>>` lock state, 🔓/🔒 toggle per suggestion row (`aria-pressed`, `data-testid`), Re-run button builds natural-language message parsed by updated `_extract_goal_seek_target` (`_GS_KV_FIXED_RE` regex + `feature_names` whitelist). 5 backend + 15 frontend = 20 new tests. Total: **4516 backend + 2613 frontend = 7129**, all passing. Backend lint: clean. Frontend build + lint: clean.

Key learnings: Arrow wrapper `() => fn()` is mandatory when a function has optional params and is wired to `onClick` — SyntheticEvent leaks in otherwise. Lock toggle state should use `Set<string>` with functional updater for correctness.

---

## Day 73 (12:00) — Done

**Track D: Real-time SLA Latency Webhook** — complete.

`EVENT_SLA_EXCEEDED = "sla_exceeded"` added to `core/webhook.py` + `ALL_EVENTS`. `sla_alert_last_fired_at` field on `Deployment` (TEXT migration in `db.py`). `_check_and_fire_sla_alert(deployment_id, threshold_ms=500.0, cooldown_hours=1.0)` in `api/deploy.py`: queries last 50 timed PredictionLogs, skips if < 5 samples, skips if p95 ≤ threshold, checks 1-hour cooldown via `sla_alert_last_fired_at`, stamps timestamp before dispatching, fires `dispatch_webhooks()` with `{p95_ms, avg_ms, sample_count, threshold_ms, message}`. Wired into `make_prediction()` as daemon background thread. `WebhookCreateBody` default `event_types` updated; `create_webhook` docstring updated. 20 backend tests. Total: **4511 backend + 2598 frontend = 7109**, all passing. Backend lint: clean. Frontend build: unchanged.

Key learning: Cooldown gate needs to be stamped on the Deployment model BEFORE dispatching (not after) to prevent racing daemon threads from firing multiple times. Use separate Session for background functions — request Session is closed by the time the thread runs.

---

## Day 73 (04:00) — Done

**Track D: Deployment Changelog via Chat** — complete.

`DeploymentChangelog` SQLModel table (deployment_id indexed, change_type constants). `_write_changelog()` best-effort helper writes immutable entries at key lifecycle points (deploy, re-deploy, undeploy, api-key add/remove). `GET /api/deploy/{id}/changelog` REST endpoint returns last 50 entries newest-first with relative_time. `_DEPLOYMENT_CHANGELOG_PATTERNS` (9 NL variants) + handler + SSE emit in `chat.py`. `DeploymentChangelogCard` (📋 icon, change-type icons/badges, timeline layout, empty state, sr-only figcaption, aria list). 26 backend + 16 frontend = 42 new tests. Total: **4491 backend + 2598 frontend = 7089**, all passing. Backend lint: clean. Frontend build + lint: clean.

Key learning: Integration test fixtures for deploy.py endpoints must use `TestClient` (sync) not `AsyncClient` (async). The `/api/data/upload` response uses `"dataset_id"` not `"id"`. Feature sets are applied via `/api/features/{dataset_id}/apply` + `/api/features/{dataset_id}/target`.

---

## Day 72 (20:00) — Done

**Track B: Goal Seek History via Chat** — complete.

`GoalSeekRecord` SQLModel table (deployment_id indexed, MAX_HISTORY=3 per deployment). `POST /api/deploy/{id}/goal-seek` saves a record + prunes oldest beyond 3. `GET /api/deploy/{id}/goal-seek/history` REST endpoint. `_GOAL_SEEK_HISTORY_PATTERNS` (9 NL variants) + handler + SSE emit in `chat.py`. `GoalSeekHistoryCard` (violet border, EntryCards with emerald/amber per achieved, target/achieved grid, gap indicator, top-3 suggestions with direction arrows, relative timestamp, empty state). 12 backend + 14 frontend = 26 new tests. Total: **4465 backend + 2582 frontend = 7047**, all passing. Backend lint: clean. Frontend build + lint: clean.

Key learning: When checking BACKLOG items, verify they're truly not implemented before committing to them — almost all the listed "What's Next" items (SLA monitoring, API key rotation, export-as-ZIP, chronological splits, ensembles, etc.) were already done. Goal seek history was the genuine gap.

---

## Day 72 (12:00) — Done

**Track B: Goal Seek / Reverse Prediction via Chat** — complete.

`run_goal_seek()` pure function using scipy L-BFGS-B (regression: minimise |predict(x) − target|; classification: maximise predict_proba). REST endpoint `POST /api/deploy/{id}/goal-seek`. Chat handler with `_GOAL_SEEK_PATTERNS` (8 NL variants) + `_extract_goal_seek_target()` (handles K/M/B suffixes, quoted class names). `GoalSeekCard` frontend component with target/achieved grid, gap indicator, ranked suggestions, feasibility note. 30 backend + 25 frontend = 55 new tests. Total: **4453 backend + 2568 frontend = 7021**, all passing.

Key learning: Python walrus operator `:=` in statement context requires parens. Nested f-strings with `\"` escapes are invalid in Python 3.12 — extract to variable. Deployment model uses `is_active` (bool) not `status` (str); `endpoint_path` (not `endpoint_url`); no `model_path` field (lives on `ModelRun`).

---

## Day 71 (20:00) — Done

**Track E: Auto-Insight on New Dataset** — complete.

`compute_auto_insights()` pure function (6 finding types), `Project.last_insight_dataset_id` migration, handler + SSE emit in `chat.py`, `AutoInsightCard` frontend component. 18 backend + 14 frontend = 32 new tests. Total: 4407 backend + 2491 frontend = 6898, all passing.

Key learning: Python `\b` doesn't work for underscore-delimited column names — use token-split + frozenset instead.

---

## What's Next (Day 76+)

**CRITICAL NOTE:** Always grep before implementing — most candidates have already been done.
**Key learnings:**
- Day 74 04:00: Check for existing REST endpoints that lack chat handlers.
- Day 74 12:00: Module-level threshold constants must be at module scope so tests can import them.
- Day 75 20:00: `jest.resetModules()` in `beforeEach` causes React hook conflicts — use static `require()` for page components instead.

**Track D candidates:**
- Cross-deployment prediction comparison via chat ✅ DONE (Day 74 04:00)
- Real-time SLA latency webhook ✅ DONE (Day 73 12:00)
- Deployment changelog ✅ DONE (Day 73 04:00)
- NOTE: Export-as-ZIP, API key rotation, SLA monitoring all ALREADY DONE.

**Track C candidates (model building depth):**
- Ensemble auto-suggest ✅ DONE (Day 74 12:00) — proactive ensemble card when score < 0.75
- NOTE: Chronological splits, feature selection, ensembles, SMOTE, CalibratedClassifierCV all ALREADY DONE.

**Track E candidates (polish):**
- "Explain this finding" direct-send ✅ DONE (Day 73 20:00)
- Auto-suggest column types ✅ ALREADY DONE
- Training completion low-accuracy hint ✅ DONE (Day 74 12:00 + LowAccuracyGuidanceCard)
- Analyst-facing "model quality score" ✅ DONE (Day 74 20:00)

**Track B candidates (vision-driven innovation):**
- Predictive cohort monitoring ✅ DONE (Day 76 04:00)
- "Why did this prediction change?" ✅ DONE (Day 75 12:00)
- Prediction baseline context on live dashboard ✅ DONE (Day 75 20:00) — `BaselineComparisonBanner` in `ExplanationCard` showing baseline prediction vs current + delta
- Per-row counterfactual explanation ✅ DONE (Day 76 12:00) — "what would save this customer?"
- Population-level counterfactual ✅ DONE (Day 76 20:00) — "what one change helps the most customers?"

---

## Day 71 (12:00) — Done
**Track E — Proactive Milestone Messages.**

AutoModeler now celebrates workflow achievements with a "shoulder tap" — without the analyst asking. When the analyst sends their first chat message after a key state transition (upload, first model trained, first deployment), a `MilestoneCard` appears automatically in the chat.

- **Upload milestone** 🎉 "Your data is loaded!" (20% progress, emerald border): fires on first message after CSV upload; mentions row/column counts; 2 action chips (Explore my data / Check data quality)
- **Train milestone** 🎯 "First model trained!" (65% progress, amber border): fires on first message after first completed model run; names the algorithm and shows accuracy; 2 action chips (Validate / Deploy)
- **Deploy milestone** 🚀 "Your model is live!" (100% progress, violet border): fires on first message after first deployment; 2 action chips (Share dashboard / Monitor performance)

Detection: `_get_current_milestone_state()` derives upload/train/deploy from ctx. `Project.last_milestone_state` tracks what's been announced — only advances one step per message, never repeats. `_MILESTONE_ORDER` = `[None, "upload", "train", "deploy"]`. Inline DB migration adds `last_milestone_state TEXT` to project table. `{type:"milestone"}` SSE event. `MilestoneCard` component (color-coded per type). `MilestoneResult`/`MilestoneAction` TypeScript types; `attachMilestoneToLastMessage` Zustand action. SSE handler + card render wired in `project/[id]/page.tsx`.

**Tests:** 13 backend (6 pure function + 7 integration) + 12 frontend (render, icon, title, subtitle, progress bar, summary, actions, click handler, a11y figcaption, 3 milestone type colors, store action) = 25 new tests. Total: **4436 backend + 2512 frontend = 6948**, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: API key auth enhancement — let analysts rotate/revoke keys via chat ("regenerate my API key", "disable API key for this deployment")
- Track C: Ensemble methods improvement — surface ensemble automatically when individual models plateau below 0.75 R² or 75% accuracy
- Track E: "Proactive insight on upload" — when a dataset is uploaded, immediately analyze it and emit 1-2 interesting findings in the first assistant message (no asking needed)

---

## Day 71 (04:00) — Done
**Track E — "What's Next?" Workflow Guidance Card via Chat.**

Analysts can now ask "what's my next step?", "what should I do next?", "guide me", "show me my options", "where do I go from here?", "help me get started" (10 NL variants) and receive a `WhatNextCard` in chat — a rich milestone card that tells the analyst exactly where they are in the workflow and what to do next, with context-aware guidance.

The card detects the analyst's current workflow stage from context:
- **Upload stage** (no dataset) → explains how to get started, what AutoModeler can do
- **Explore stage** (has dataset, no trained model) → summarises the dataset, suggests exploring data + applying features + training a model; mentions target column if set
- **Validate stage** (has trained model, not deployed) → celebrates the model, shows algorithm + accuracy, suggests validating → deploying → comparing
- **Monitor stage** (has live deployment) → guides to sharing, monitoring, and retraining

Each card shows: stage badge, progress bar (5%/25%/65%/100% per stage), data-aware summary sentence, and 3 prioritised step rows. Each step row has an icon, title, description, and a "Try this →" button that pre-fills the chat input with the step's action string so analysts can execute it with one click.

**Backend:** `_WHAT_NEXT_PATTERNS` (10 NL variant groups) in `chat.py`. Handler block reads `ctx["dataset"]`, `ctx["model_runs"]`, `ctx["deployment"]`, `ctx["feature_set"]` to determine stage. Pure in-handler stage logic (no new DB queries). Emits `{type:"what_next"}` SSE event with `stage`, `stage_label`, `progress`, `summary`, `steps` (list of `{icon, title, description, action}`). Injects guidance summary into system_prompt so LLM acknowledges the stage in its response.

**Frontend:** `WhatNextStep` + `WhatNextResult` TypeScript interfaces in `types.ts`. `what_next?: WhatNextResult` field added to `ChatMessage`. `attachWhatNextToLastMessage` Zustand action in `store.ts`. `WhatNextCard` component (color-coded border/badge/progress-bar per stage: blue=upload, emerald=explore, amber=validate, violet=monitor). `StepRow` subcomponent with icon, title+description, "Try this →" button (`aria-label`, `data-testid`). `role="progressbar"` + `aria-valuenow/min/max`. `sr-only figcaption` for screen readers. SSE handler + card render wired in `project/[id]/page.tsx`. `onActionClick` prop calls `setChatInput(action)` so clicking "Try this →" pre-fills the chat input.

**Tests:** 34 backend (20 regex unit + 8 handler integration: upload/explore/stage-label/steps-fields/no-false-positive/summary/progress/explore-summary) + 12 frontend (heading, stage badge, progress bar, summary, 3 step rows, step content, try buttons, click handler, figcaption, upload render, monitor render, Zustand store) = 46 new tests. Total: **4423 backend + 2500 frontend = 6923**, all passing. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: Export as self-contained prediction service via chat improvement (ZIP + uvicorn, already done as REST but chat-triggered download card could be richer)
- Track E: Proactive "nice job!" milestone messages on completion (upload complete, model trained) — auto-inject a brief guide into the LLM context on key state transitions without the analyst asking
- Track C: Date-aware chronological train/test split promotion — surface it proactively when date columns are detected

---

## Day 70 (20:00) — Done
**Track B — Cross-Project Model Comparison via Chat.**

Analysts can now ask "cross-project comparison", "compare my revenue model vs my churn model", "rank all my models side by side", "which of my models performs best overall" (8 NL variants) to receive a `CrossProjectComparisonCard` with normalized 0-100 performance scores ranked head-to-head across all projects. Normalization: R²/accuracy/F1 → ×100; error metrics (MAE/RMSE) → 1/(1+x)×100. `_normalize_metric()` helper + `compute_cross_project_comparison()` pure function in `core/advisor.py`. `GET /api/projects/cross-comparison` REST endpoint (registered before `/{project_id}` to avoid path capture). `_CROSS_PROJECT_PATTERNS` regex + handler + SSE emit in `chat.py`. `CrossProjectComparisonResult` / `CrossProjectComparisonRow` TypeScript types; `api.projects.crossComparison()` method; `attachCrossProjectComparisonToLastMessage` Zustand action; `CrossProjectComparisonCard` component (indigo border, 🏆, winner highlight, score bars, rank medals 🥇🥈🥉, deployment badges, insights list, sr-only figcaption). 33 backend + 23 frontend = 56 new tests. Total: 4389 backend + 2477 frontend = 6866. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)
- Track C: Ensemble methods (voting classifier / stacking) for better accuracy
- Track D: Prediction SLA / latency monitoring — "is my model responding fast enough?"

---

## Day 70 (12:00) — Done
**Track D — Weekly Usage Report via Chat.**

Analysts can now ask "how many predictions did I get this week?", "weekly prediction summary", "weekly report", "how did I do this week" (8 NL variants) to receive a `WeeklyUsageReportCard` with this-week vs last-week count, trend (↑/↓/→), 7-day breakdown bar chart, and top input patterns table. `_WEEKLY_USAGE_PATTERNS` regex + handler in `chat.py`. `WeeklyUsageReportResult` type; `attachWeeklyUsageReportToLastMessage` Zustand action; SSE handler + card render. 21 backend + 16 frontend = 37 new tests. Total: 4356 backend + 2454 frontend = 6810. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track B: Cross-project model comparison — "how does my revenue model compare to my churn model?"
- Track E: "What's next?" guidance cards at key step transitions

---

## Day 70 (04:00) — Done
**Track D — Prediction form "Copy as link".**

VPs can now bookmark or share exact prediction scenarios as pre-filled URLs. The "Your Scenario" form card on `predict/[id]` has a new 🔗 "Copy as link" button — clicking it encodes all current input values as URL query params (`?units=100&region=North`) and copies the full URL to the clipboard (flashes "Copied!" for 2s). Loading that URL pre-fills the form so the VP arrives with their values ready. Analysts can also say "generate a pre-filled link for units=100, region=North", "create a shareable link", "copy this scenario as a link", "bookmark this scenario" (9 NL variants) to receive a `ShareLinkCard` (orange border, 🔗 icon) in chat with the full URL, feature-value chips, and a copy button. `GET /api/deploy/{id}/share-link?features={"units":"100"}` REST endpoint. `_SHARE_LINK_PATTERNS` + `_SHARE_LINK_VALUE_RE` regexes in `chat.py`. `ShareLinkResult` type; `api.deploy.getShareLink()` method; `attachShareLinkToLastMessage` Zustand action; SSE handler + card render in project page. URL param parsing reads `window.location.search` on mount (no re-render loops). 24 backend + 18 frontend = 42 new tests. Total: 4356 backend + 2454 frontend = 6810. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: Deployment usage report via chat — "how many predictions did I get this week?", trend breakdown by day, top input patterns
- Track B: Cross-project model comparison — "how does my revenue model compare to my churn model?"
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)

---

## Day 69 (20:00) — Done
**Track D — Embed Code Generator via Chat.**

Analysts can say "give me embed code for my dashboard", "iframe snippet", "SharePoint embed", or "embed into our Notion page" (9 NL variants) to receive an `EmbedCodeCard` with a ready-to-paste `<iframe>` snippet. Three size presets (Full Width / Fixed / Compact) update iframe dimensions; code block uses `window.location.origin + dashboard_url` for correct host. Copy-to-clipboard button flashes "Copied!". "Where to paste this" callout lists SharePoint / Notion / Confluence / HTML. `GET /api/deploy/{id}/embed-code` REST endpoint. 22 backend + 17 frontend = 39 new tests. Total: 4332 backend + 2436 frontend = 6768. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Prediction form "Copy as link" — generate a pre-filled URL with query params so VPs can bookmark specific scenarios (e.g., `?units=100&region=North`)
- Track D: Deployment usage report via chat — "how many predictions did I get this week?", trend breakdown by day, top input patterns
- Track B: Cross-project model comparison — "how does my revenue model compare to my churn model?"

---

## Day 69 (12:00) — Done
**Track D — Dashboard Field Ordering via Chat.**

Analysts can now control the presentation order of fields on the VP-facing prediction form through natural language: "reorder fields: units, region, product", "put units first", "move region to the top", "field order: region, units, product". `_DC_ORDER_RE` extracts ordered lists or single "put X first" targets. Handler in `send_message()` assigns `display_order = 0, 1, 2, ...` via upsert on `DashboardFieldConfig` (field was pre-existing from Day 68 04:00 but never wired). `predict/[id]/page.tsx` sorts schema by `display_order ?? Infinity`. SSE emits `action="ordered"` + `ordered_count`. `DashboardConfigCard` shows cyan `#N` position badges, cyan border + 🔢 icon + "Fields Reordered" heading + `ordered_count` badge. `_DASHBOARD_CONFIG_PATTERNS` extended with 5 ordering arms; fixed `\s+(?:as|:)` → `\s*(?:as|:)` to handle "reorder fields: X" (no space before colon). 9 backend + 6 frontend = 15 new tests. Total: 4310 backend + 2419 frontend = 6729. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)
- Track C: Ensemble methods (voting classifier / stacking) for better accuracy
- Track D: Prediction SLA / latency monitoring — "is my model responding fast enough?"

---

## Day 69 (04:00) — Done
**Track D — Prediction Dashboard Custom Title & Description via Chat.**

Analysts can now brand the VP-facing prediction URL through natural language: "set the dashboard title to 'Q2 Revenue Forecast'", "add a dashboard description: For the finance team only", "what's the dashboard title?", "clear the dashboard title". Two new columns on `Deployment` (`dashboard_title`, `dashboard_description`) with inline migrations. `GET/PUT /api/deploy/{id}/dashboard-metadata` REST endpoints. `_DASHBOARD_META_PATTERNS` (8 NL arms) + 4 extraction regexes + handler in `chat.py` (title_set/description_set/both_set/cleared/status intents). `DashboardMetadataCard` (emerald/sky/slate borders). `predict/[id]/page.tsx` loads title/description on mount, applies custom title as page h1, shows description below title. Also back-filled missing `dashboard_config` SSE handler + `DashboardConfigCard` render in `project/[id]/page.tsx` (was emitting but never rendering). 22 backend + 16 frontend = 38 new tests. Total: 4301 backend + 2413 frontend = 6714. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)
- Track C: Ensemble methods (voting classifier / stacking) for better accuracy
- Track D: Prediction SLA / latency monitoring — "is my model responding fast enough?"

---

## Day 68 (20:00) — Done
**Track D — Per-field display labels via chat.**

Analysts can now rename any VP-facing dashboard field through natural language: "label units as Monthly Units Sold", "rename region as Sales Region on the dashboard", "call channel as Distribution Channel". `_DC_LABEL_RE` regex extracts feature name + label from NL. Handler upserts `DashboardFieldConfig.display_label`, emits `action="labeled"` SSE event with `labeled_count`. `DashboardConfigCard` updated: violet `→ "label"` badge in `FieldRow`, "Field Labeled" heading/violet border/🏷️ icon for labeled action, `labeled_count` violet badge in header. `predict/[id]/page.tsx` already consumed `display_label` (Day 68 04:00). 9 backend + 6 frontend = 15 new tests. Total: 4279 backend + 2397 frontend = 6676. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)
- Track C: Ensemble methods (voting classifier / stacking) for better accuracy
- Track D: Prediction SLA / latency monitoring — "is my model responding fast enough?"

---

## Day 68 (12:00) — Done
**CI fix: Dashboard Config Integration.**

Fixed 21 backend + 11 frontend test failures from the Day 68 04:00 revert. The revert removed backend implementations (regex constants, handler, REST endpoints, model registration) while leaving test files in place. Re-implemented all missing backend pieces with corrected regex patterns. Updated `predict/[id]/page.tsx` mock sequences in `pages.test.tsx`, `confidence-interval.test.tsx`, and `compare-models.test.tsx` to account for the new `getDashboardConfig` fetch call (4→5 call counts, added dashboard config mock response). CI is green.

**What's next:**
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)
- Track C: Ensemble methods (voting classifier / stacking) for better accuracy
- Track D: Per-field display labels via chat ("rename units to 'Monthly Units Sold'")

---

## Day 68 (04:00) — Done
**Track D — Prediction Dashboard Field Configuration via Chat.**

Analysts can now curate which fields appear on the VP-facing prediction dashboard through natural language: "hide units from the dashboard", "lock region to North", "only show units and revenue", "show all fields", "what's visible on my dashboard". `DashboardFieldConfig` SQLModel table (deployment_id indexed, is_visible/is_locked/locked_value). `GET/PUT/DELETE /api/deploy/{id}/dashboard-config` REST endpoints. Six regex constants + `_extract_dashboard_feature()` helper + handler in `chat.py` (reset/status/only-show/hide/lock intents). `DashboardConfigCard` (emerald/sky/slate borders for updated/status/reset); `FieldRow` with Hidden/Locked/Visible badges. `predict/[id]/page.tsx` filters hidden fields, renders locked fields as read-only inputs, injects locked values into prediction payload, shows "Simplified view" notice. 24 backend + 16 frontend = 40 new tests. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)
- Track C: Ensemble methods (voting classifier / stacking) for better accuracy
- Track D: Per-field display labels via chat ("rename units to 'Monthly Units Sold'")

---

## Day 67 (20:00) — Done
**Track D — Prediction Input Validation Rules via Chat.**

Analysts can now define, list, and remove input validation rules on deployed prediction APIs through natural language: "validate that units is between 1 and 10000", "require region to be one of East, West, Central", "ensure customer_id is not null", "show my validation rules", "remove all validation rules". `InputValidationRule` SQLModel table (deployment_id indexed, rule_type/feature_name/min_val/max_val/allowed_values). `validate_prediction_inputs()` pure function in `core/validator.py`: range/one_of/not_null checks, returns (is_valid, violations). `make_prediction()` hook loads rules per deployment, raises HTTP 422 with plain-English violation messages. `POST/GET/DELETE /api/deploy/{id}/input-validation-rules` REST endpoints. Seven regex constants + handler in `chat.py` (LIST/DELETE/CREATE intents, range/bound/one_of/not_null extraction). `InputValidationRuleCard` (violet/slate/rose/slate borders for created/list/deleted/guidance); `RuleTypeBadge` subcomponent; `RuleRow` with `data-testid`. 41 backend + 22 frontend = 63 new tests. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: Prediction SLA / latency monitoring — "is my model responding fast enough?", `response_ms` already in `PredictionLog`
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)
- Track C: Ensemble methods (voting classifier / stacking) for better accuracy

---

## Day 67 (12:00) — Done
**Track D — Prediction Confidence Thresholding via Chat.**

Analysts can now set a minimum confidence threshold so uncertain classification predictions are flagged with `below_threshold=True` rather than silently served. "set confidence threshold to 80%", "reject low-confidence predictions", "only accept predictions above 90% confidence", 9 NL variants total. `_CONFIDENCE_THRESHOLD_PATTERNS` + `_CONFIDENCE_THRESHOLD_VALUE_RE` + `_DISABLE_CONFIDENCE_THRESHOLD_RE` in `chat.py`. Handler persists `Deployment.confidence_threshold`; `make_prediction()` compares `max(predict_proba)` against threshold — below → `below_threshold=True` + plain-English `threshold_message`; at or above → `below_threshold=False`. `PUT /api/deploy/{id}/confidence-threshold` + `GET /api/deploy/{id}/confidence-threshold-status`. `ConfidenceThresholdCard` (amber border, 🎯 icon): enabled/disabled badge, amber explanation, 30-day below-count stats. `predict/[id]/page.tsx` shows amber callout when `below_threshold=True`. Inline migration for `confidence_threshold REAL` in `db.py`. Classification-only (no-op for regression). 17 backend + 15 frontend = 32 new tests. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: Prediction SLA / latency monitoring — chat query "is my model responding fast enough?", `response_ms` already logged in `PredictionLog`
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)
- Track C: Ensemble methods (voting classifier / stacking) for better model accuracy

---

## Day 67 (04:00) — Done
**Track C — Interactive What-if Scenario Explorer with Sliders.**

Upgraded the basic text-input `WhatIfCard` in `DeploymentPanel` into a real-time interactive slider panel. Analysts can drag sliders for numeric features (bounded by p5/p95 training-data percentiles), choose from dropdowns for categorical features, and see the predicted outcome update in ~400ms. Side-by-side "Baseline (means)" vs "Your Scenario" comparison with delta badge (▲/▼/→ + % change) and confidence interval band. "Show more" toggle for >8 features. Backend: extended `get_feature_schema()` in `core/deployer.py` to include `min`, `max`, `p5`, `p95` from stored `feature_ranges`; `FeatureSchemaEntry` TypeScript type extended with optional range fields. Frontend uses debounced `api.deploy.predict()` call. 7 backend + 17 frontend = 24 new tests. Backend lint: clean. Frontend build + lint: clean.

**What's next:**
- Track D: Prediction SLA / latency monitoring — chat query "is my model responding fast enough?", dashboard metric
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)
- Track C: Ensemble methods (voting classifier / stacking) for better model accuracy

---

## Day 66 (20:00) — Done
**Track D — Deployment Rollback via Chat.**

Analysts can now list deployment version history and roll back to any previous version entirely through chat: "roll back to version 1", "revert my deployment", "show my deployment versions", "deployment version history". `_ROLLBACK_PATTERNS` (8 NL variants) + `_ROLLBACK_VERSION_RE` in `chat.py`. List mode returns all versions; rollback mode archives current version, restores target model files onto Deployment row, creates new DeploymentVersion entry keeping endpoint URL stable. `RollbackChatCard` (indigo/emerald/rose borders for list/success/error): version history table with Current/Restored badges, algorithm name, metric, date; footer guidance. 10 backend + 18 frontend = 28 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track C: What-if scenario analysis panel — simulate predictions by adjusting feature sliders interactively
- Track D: Prediction SLA / latency monitoring — chat query "is my model responding fast enough?", dashboard metric
- Track E: "What's next?" guidance cards at key step transitions (after upload, after training, after deploy)

---

## Day 66 (12:00) — Done
**Track D — Model Accuracy Degradation Alert via Chat.**

Analysts configure accuracy-based webhook alerts entirely through chat: "alert me when my model accuracy drops below 80%", "notify me when feedback accuracy falls under 75%", "check my accuracy alert", "disable accuracy alert". After each feedback submission, system checks aggregate feedback accuracy vs. threshold, fires webhook once on first crossing. `_compute_feedback_accuracy_simple()` computes live metric from FeedbackRecord table — classification: accuracy (0–1), regression: pct_error (0–100). `accuracy_alert_fired` flag prevents repeated alerts; resets when threshold changes. `PUT /api/deploy/{id}/accuracy-alert` + `GET /api/deploy/{id}/accuracy-alert-status`. `AccuracyAlertCard` (amber border, 🎯 icon): breach color coding, fired badge. 21 backend + 20 frontend = 41 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Champion-challenger A/B testing via chat — leverage existing A/B test infrastructure for conversational champion/challenger setup
- Track C: What-if scenario analysis panel — allow analysts to simulate predictions by adjusting feature values interactively
- Track D: Deployment rollback to previous version via chat — "roll back my model", "revert to version 2"

## Day 66 (04:00) — Done
**Track B — Cross-Model Feature Importance Comparison via Chat.**

Analysts can now ask "which features matter most across all my models?", "feature importance comparison", "what drives predictions?", "feature consensus", or 11 other NL variants to get a `CrossModelFeaturesCard` comparing feature importances across all completed model runs.

`compute_cross_model_feature_importance()` pure function in `core/advisor.py`: accepts runs_with_importances list, computes per-feature mean_importance/consistency (CoV-based: high/medium/variable)/agreement_count (models ranking feature ≤ 5)/n_models_with_data; caps at 15; identifies consensus_features. `_CROSS_MODEL_FEAT_PATTERNS` (13 NL variants) in `chat.py`. SSE type `cross_model_features`. `GET /api/models/{project_id}/cross-model-features` REST endpoint. `CrossModelFeaturesCard` (violet border, 🔍 icon): consensus chip callout, feature table with importance bars + consistency badges. 19 backend + 15 frontend = 34 new tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Champion-challenger A/B testing via chat — leverage existing A/B test infrastructure for conversational champion/challenger setup
- Track C: What-if scenario analysis panel — allow analysts to simulate predictions by adjusting feature values interactively
- Track B: Model drift detection alerts — "alert me when my model's accuracy drops below X%"

## Day 65 (20:00) — Done
**Track B — Automated Model Comparison Summary via Chat.**

Analysts can now ask "compare my models", "model comparison summary", "how do my trained models stack up", "model showdown", "model overview", or 11 other NL variants to get a narrative `ModelComparisonSummaryCard` for all completed runs — no criteria needed.

`compute_model_comparison_summary()` pure function in `core/advisor.py`: ranks runs by primary metric, builds display dicts (plain-English names, explainability/speed labels from new `_EXPLAINABILITY_LABEL`/`_SPEED_LABEL` dicts, CV when available, selected/deployed flags), generates up to 3 trade-off sentences (accuracy gap, explainability, stability), 2–3 sentence narrative, summary line. `_MODEL_COMPARISON_SUMMARY_PATTERNS` (15 NL variants, no trailing `\b`) in `chat.py`; fires only when `_MODEL_SELECT_PATTERNS` does NOT match (prevents collision with criteria-based selection). SSE type `model_comparison_summary`. `ModelComparisonSummaryCard` (blue border, 📊 icon): header with count/type/metric, narrative, comparison table with winner ✓ badge + status annotations, trade-offs, italic summary footer, sr-only figcaption. 23 backend + 15 frontend = 38 new tests. Lint clean. Frontend build clean.

**What's next:**
- Track D: Champion-challenger A/B testing via chat — leverage existing A/B test infrastructure conversationally
- Track C: What-if scenario analysis panel — allow analysts to simulate predictions by adjusting feature values interactively
- Track B: Feature importance comparison across models — "which features matter most across all my models?" ✓ Done Day 66

## Day 65 (12:00) — Done
**Track D — Model Card Export via Chat + Track C — Calibration Inline in Training Panel.**

Model Card Export: Analysts can ask "export model card", "download model card", "model card for compliance", "share model documentation", and 7 other NL variants to receive a `ModelCardExportCard` in chat with a "Download HTML Model Card" button. `generate_model_card_html()` pure function in `core/report_generator.py` produces a self-contained HTML model card (overview table, performance, feature importance bars, optional calibration section, intended use, limitations, deployment info). XSS-safe via `html_escape()`. `GET /api/models/{run_id}/export-model-card` endpoint returns `HTMLResponse` with `Content-Disposition: attachment`. `_MODEL_CARD_EXPORT_PATTERNS` (10 NL variants) + handler + `{type:"model_card_export"}` SSE event. `ModelCardExportCard` (indigo border, 📋 icon): algorithm/problem-type/target badges, metric + feature count + row count, training date, download anchor. `ModelCardExportInfo` TypeScript interface.

Calibration Inline: `CalibrationRow` sub-component in `RunCard` shows "🎯 Brier score: 0.XX (excellent/good/poor)" with color-coded quality label beneath `CvScoreRow`. Classification only; returns null for regression. Closes "how trustworthy are the confidence scores?" without a chat command.

25 backend (10 HTML unit + 3 endpoint integration + 12 regex) + 13 frontend = 38 new tests. Backend lint: clean. Frontend build + TypeScript + lint: clean.

**What's next:**
- Track B: Automated model comparison summary — "which model is best and why?" (comparative narrative when multiple runs exist)
- Track D: Champion-challenger A/B testing via chat — leverage existing A/B test infrastructure for conversational champion/challenger setup
- Track C: What-if scenario analysis panel — allow analysts to simulate predictions by adjusting feature values

## Day 65 (04:00) — Done
**Track D — Prediction Error Distribution Analysis.** Analysts can now ask "show me the error distribution", "residual histogram", "where does my model struggle?", "per class error rate", and 8 other NL variants to see a histogram of ALL prediction errors — distinct from `PredictionErrorCard` (top-N worst individual rows). Pure function `compute_error_distribution()` in `core/validator.py`: regression bins residuals into a 5–30 bar histogram with bias detection (unbiased/over-predicts/under-predicts via normalized mean residual); classification returns per-class error rates sorted highest-to-lowest with decoded class names. REST endpoint `GET /api/models/{run_id}/error-distribution`. `_ERROR_DIST_PATTERNS` (11 NL variants, guarded by `not pred_error_event`). SSE type `error_distribution`. `ErrorDistributionCard` with color-coded Recharts BarChart for regression + per-class table with mini bars for classification. 34 backend + 20 frontend = 54 new tests. Backend lint: clean. Frontend build + TypeScript: clean.

**What's next:**
- Track C: Calibration display in training panel — show calibration curve + Brier score alongside CV score for classification models
- Track D: Model card export — "export all my model metadata as a shareable card"
- Track B: Automated model comparison summary — when multiple runs exist, "which model is best and why?"

## Day 64 (20:00) — Done
**Track C — Cross-Validation Score in Training Panel.** After each training run, `train_single_model()` now automatically runs 5-fold cross-validation on the full dataset using an unfitted copy of the same model (reusing `run_cross_validation()` from `core/validator.py`) and stores `cv_mean`, `cv_std`, `cv_n_splits` in `ModelRun.metrics`. CV skipped for datasets < 10 rows. The training panel's `RunCard` now shows a `CvScoreRow` beneath train/test metrics: "5-fold CV R²: 0.81 ± 0.03 (stable)" with color-coded consistency label (emerald=stable std<0.05, amber=moderate <0.1, rose=variable ≥0.1). TypeScript types updated. 4 backend + 4 frontend tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track D: Prediction error distribution analysis — "show me where my model is wrong" (histogram of residuals/misclassifications by segment)
- Track C: Calibration display in training panel — show calibration curve + Brier score alongside CV score for classification models
- Track B: Model registry export — "export all my model metadata as JSON for archiving"

## Day 64 (12:00) — Done
**Track D — Training vs Production Performance Monitor.** Analysts can now ask "how is my model holding up in production?", "training vs production performance", "is my model degrading?", and 7 other NL variants to see a side-by-side comparison of training-time metrics against live production accuracy derived from submitted feedback. Pure function `compute_training_vs_production()` supports both regression (MAE comparison, lower_is_better) and classification (accuracy comparison, higher_is_better); classifies status as stable/warning/degrading/no_feedback with configurable degradation thresholds; returns weekly timeline for sparkline. REST endpoint `GET /api/deploy/{id}/training-vs-production`. SSE type `prod_performance`. `ProdPerformanceCard` component with adaptive border color, `StatusBadge`, `DegradationBadge`, side-by-side `MetricBox`, Recharts `Timeline` with training reference line, `role="alert"` callouts for warning/degrading. 38 backend unit tests, 21 frontend tests. Backend lint: clean. Frontend build + TypeScript: clean.

**What's next:**
- Track C: Cross-validation score in training panel — analysts want "how stable is this model?" (CV score ± std alongside point-estimate R²)
- Track D: Champion-challenger A/B testing via chat — leverage existing A/B test infrastructure for conversational champion/challenger setup
- Track B: Cross-project model comparison — "which of my models is performing best across all projects?"

## Day 62 (20:00) — Done
**Track E — End-to-End "Lunch Break" BDD Test Suite.** Closed the loop on the core vision promise with a machine-executable test. Six BDD scenarios in `tests/features/analyst_lunch_break.feature` + `tests/test_bdd_analyst_lunch_break.py` covering the complete analyst journey: upload CSV → data insight visible → explore via chat → train regression model → deploy endpoint → single prediction → batch prediction. Uses synchronous TestClient with polling for async training. Discovered two response-shape gaps during authoring: preview returns `column_stats` (not `columns`); prediction response uses `feature_names` list (not `input_features` dict). Both corrected in step assertions (no backend change needed — the API is correct, the docs were wrong). Updated `performance_baseline.json` to 4004 backend / 2183 frontend tests. 6 BDD scenarios, all passing. Backend lint: clean.

**What's next:**
- Track D: Production model performance monitoring — detect if live prediction accuracy is degrading vs training metrics (needs user-submitted feedback to compare)
- Track C: Cross-validation score displayed in training panel — analysts want to know "how stable is this model?" not just point-estimate R²
- Track B: Multi-project comparison insights — "which of my models is making the best predictions across all my projects?"

## Day 62 (12:00) — Done
**Track D — Multi-Deployment Status Overview via Chat.** Analysts can ask "show all my deployments", "deployment dashboard", "deployment overview", "which models are live", and 8 other NL variants to get a cross-project operational monitoring card. Pure function `compute_deployments_overview()` aggregates all active deployments: counts by environment + health status, avg health score, total predictions, sorted list (production first, then health desc). `GET /api/deploy/overview` REST endpoint (registered before `{deployment_id}` path to prevent route capture). SSE type `deployments_overview`. `DeploymentsOverviewCard` component with per-row health bars, status/environment badges, API key Protected badge, top_issue display. 23 backend + 33 frontend tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track C: Date-aware chronological split via chat — "train with chronological split"
- Track D: Deployment comparison — side-by-side metric comparison for multiple versions of same model
- Track E: End-to-end "lunch break" analyst flow — run the full upload → explore → train → validate → deploy → predict flow as a real user and fix friction points

## Day 62 (04:00) — Done
**Track D — API Key Management via Chat.** Analysts can generate, regenerate, disable, and check status of API key protection entirely through conversation — the REST endpoint and DeploymentPanel UI existed but chat was missing. Three regex patterns (8+4+4 NL variants). Elif priority chain: DISABLE > GENERATE > STATUS (resolves pattern conflict where status regex matched "api key protection" messages). GENERATE: `secrets.token_urlsafe(32)` + SHA-256 salted hash stored; raw key in SSE event once. SSE type `api_key_result` with `{action, deployment_id, is_protected, api_key?, summary}`. `ApiKeyChatCard` — four states: generated/regenerated (amber border, 🔑, copy-to-clipboard with "shown once" callout), disabled (slate border, 🔓), status (adaptive). `ApiKeyResultInfo` TypeScript interface; `attachApiKeyResultToLastMessage` Zustand action. 31 backend + 18 frontend tests. Backend lint: clean. Frontend build: clean.

**What's next:**
- Track C: Date-aware chronological split via chat — "train with chronological split"
- Track E: End-to-end "lunch break" analyst flow — run the full upload → explore → train → validate → deploy → predict flow as a real user and fix friction points
- Track D: Deployment comparison — side-by-side metric comparison when multiple model versions exist *(partially addressed by Day 62 12:00 overview; next: version-specific comparison)*

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
- Track D: API key auth for prediction endpoints ✓ (shipped Day 62)
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
