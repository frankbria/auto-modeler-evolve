export interface Project {
  id: string
  name: string
  description?: string
  created_at: string
  updated_at: string
  status: "exploring" | "modeling" | "deployed"
  // optional quick stats (returned by list and get endpoints)
  dataset_id?: string
  dataset_filename?: string
  dataset_rows?: number
  model_count?: number
  has_deployment?: boolean
}

export interface ColumnStat {
  name: string
  dtype: string
  non_null_count: number
  null_count: number
  null_pct: number
  unique_count: number
  min?: number | null
  max?: number | null
  mean?: number | null
  std?: number | null
  sample_values: (string | number | null)[]
  distribution?: NumericDistribution | CategoricalDistribution
  outliers?: OutlierInfo
}

export interface NumericDistribution {
  bins: number[]
  counts: number[]
}

export interface CategoricalDistribution {
  labels: string[]
  counts: number[]
}

export interface OutlierInfo {
  count: number
  lower_fence: number | null
  upper_fence: number | null
  pct: number
}

export interface DataInsight {
  type: string
  severity: "info" | "warning" | "critical"
  title: string
  detail: string
}

export interface Dataset {
  id: string
  project_id: string
  filename: string
  row_count: number
  column_count: number
  uploaded_at: string
}

export interface DatasetListItem {
  dataset_id: string
  filename: string
  row_count: number
  column_count: number
  uploaded_at: string
  size_bytes: number
}

export interface JoinKeySuggestion {
  name: string
  dtype_left: string
  dtype_right: string
  unique_left: number
  unique_right: number
  uniqueness_left: number
  uniqueness_right: number
  recommended: boolean
}

export interface MergeResponse {
  dataset_id: string
  filename: string
  row_count: number
  column_count: number
  join_key: string
  how: string
  conflict_columns: string[]
  preview: Record<string, unknown>[]
  column_stats: ColumnStat[]
}

export interface UploadResponse {
  dataset_id: string
  filename: string
  row_count: number
  column_count: number
  preview: Record<string, unknown>[]
  column_stats: ColumnStat[]
  insights: DataInsight[]
  suggestions?: string[]
}

export interface ChartSpec {
  chart_type: "bar" | "line" | "histogram" | "scatter" | "pie" | "heatmap" | "radar" | "boxplot"
  title: string
  data: Record<string, unknown>[]
  x_key: string
  y_keys: string[]
  x_label: string
  y_label: string
}

// ---------------------------------------------------------------------------
// Computed columns (derived metrics through conversation)
// ---------------------------------------------------------------------------

export interface ComputedColumnSuggestion {
  dataset_id: string
  name: string
  expression: string
  sample_values: (number | string | null)[]
  dtype: string
}

export interface ComputeResult {
  dataset_id: string
  compute_result: {
    column_name: string
    expression: string
    dtype: string
    sample_values: (number | string | null)[]
    row_count: number
    column_count: number
    action: "added" | "updated"
    summary: string
  }
  preview: Record<string, unknown>[]
  updated_stats: {
    row_count: number
    column_count: number
  }
}

export interface RenameResult {
  dataset_id: string
  old_name: string
  new_name: string
  column_count: number
}

export interface TrainingStartedResult {
  project_id: string
  target_column: string
  problem_type: "classification" | "regression"
  algorithms: string[]
  run_count: number
  status: "started"
}

export interface DeployedResult {
  id: string
  model_run_id: string
  project_id: string
  endpoint_path: string
  dashboard_url: string
  is_active: boolean
  algorithm: string
  problem_type: string
  target_column: string
  feature_names: string[]
  metrics: Record<string, number>
  created_at: string | null
}

// ---------------------------------------------------------------------------
// Model Card (plain-English model explanation)
// ---------------------------------------------------------------------------

export interface ModelCardMetric {
  name: string
  value: number
  display: string
  plain_english: string
}

export interface ModelCardFeature {
  feature: string
  importance: number
  rank: number
}

export interface ModelCard {
  project_id: string
  model_run_id: string
  algorithm: string
  algorithm_name: string
  problem_type: string
  target_col: string
  row_count: number
  feature_count: number
  metric: ModelCardMetric
  top_features: ModelCardFeature[]
  limitations: string[]
  summary: string
  is_selected: boolean
  is_deployed: boolean
}

// ---------------------------------------------------------------------------
// PDF Report Ready (chat-triggered report download)
// ---------------------------------------------------------------------------

export interface ReportReady {
  model_run_id: string
  algorithm: string
  problem_type: string
  metric_name: string
  metric_value: number | null
  download_url: string
}

// ---------------------------------------------------------------------------
// Dataset export (chat-triggered CSV download)
// ---------------------------------------------------------------------------

export interface DataExportResult {
  dataset_id: string
  filename: string
  row_count: number
  filtered: boolean
  download_url: string
}

// ---------------------------------------------------------------------------
// Missing values overview (chat-triggered null map)
// ---------------------------------------------------------------------------

export interface NullMapColumn {
  column: string
  null_count: number
  null_pct: number
  complete_pct: number
}

export interface NullMapResult {
  dataset_id: string
  total_rows: number
  total_columns: number
  columns_with_nulls: number
  fully_complete_columns: number
  overall_completeness: number
  columns: NullMapColumn[]
  summary: string
}

// ---------------------------------------------------------------------------
// Summary statistics table
// ---------------------------------------------------------------------------

export interface NumericColumnStats {
  column: string
  count: number
  mean: number | null
  std: number | null
  min: number | null
  q25: number | null
  median: number | null
  q75: number | null
  max: number | null
  null_count: number
}

export interface CategoricalColumnStats {
  column: string
  count: number
  unique: number
  top: string | null
  freq: number
  null_count: number
}

export interface SummaryStatsResult {
  dataset_id: string
  total_rows: number
  total_cols: number
  numeric_stats: NumericColumnStats[]
  categorical_stats: CategoricalColumnStats[]
  summary: string
}

// ---------------------------------------------------------------------------
// Category value counts
// ---------------------------------------------------------------------------

export interface ValueCountRow {
  value: string
  count: number
  pct: number
}

export interface ValueCountResult {
  dataset_id: string
  column: string
  total_rows: number
  non_null: number
  null_count: number
  unique_count: number
  rows: ValueCountRow[]
  has_more: boolean
  summary: string
}

// ---------------------------------------------------------------------------
// Pair correlation analysis
// ---------------------------------------------------------------------------

export interface PairCorrelationResult {
  dataset_id: string
  col1: string
  col2: string
  r: number | null
  p_value: number | null
  n: number
  strength: string
  direction: string
  significant: string
  interpretation?: string
  summary: string
}

// ---------------------------------------------------------------------------
// Stat query (single aggregate result)
// ---------------------------------------------------------------------------

export interface StatQueryResult {
  dataset_id: string
  agg: string
  col: string | null
  value: number
  n_rows: number
  n_valid?: number
  formatted_value: string
  label?: string
  summary: string
}

// ---------------------------------------------------------------------------
// Group trend analysis
// ---------------------------------------------------------------------------

export interface GroupTrendRow {
  group: string
  slope: number
  pct_change: number
  direction: "up" | "down" | "flat"
  first_value: number
  last_value: number
  n_periods: number
  rank: number
}

export interface GroupTrendResult {
  dataset_id: string
  date_col: string
  group_col: string
  value_col: string
  groups: GroupTrendRow[]
  rising: number
  falling: number
  flat: number
  summary: string
}

// ---------------------------------------------------------------------------
// Chat-driven feature engineering
// ---------------------------------------------------------------------------

export interface FeatureSuggestionItem {
  id: string
  column: string
  transform_type: string
  title: string
  description: string
  preview_columns: string[]
}

export interface FeatureSuggestionsChatResult {
  dataset_id: string
  suggestions: FeatureSuggestionItem[]
  count: number
}

export interface FeaturesAppliedResult {
  feature_set_id: string
  dataset_id: string
  new_columns: string[]
  total_columns: number
  applied_count: number
}

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  timestamp: string
  chart?: ChartSpec
  crosstab?: CrosstabResult
  compute?: ComputedColumnSuggestion
  segment_comparison?: SegmentComparisonResult
  forecast?: ForecastResult
  data_readiness?: DataReadinessResult
  target_correlation?: TargetCorrelationResult
  group_stats?: GroupStatsResult
  rename_result?: RenameResult
  training_started?: TrainingStartedResult
  data_story?: DataStory
  filter_set?: FilterSetResult
  deployed?: DeployedResult
  model_card?: ModelCard
  report_ready?: ReportReady
  feature_suggestions?: FeatureSuggestionsChatResult
  features_applied?: FeaturesAppliedResult
  segment_performance?: SegmentPerformanceResult
  column_profile?: ColumnProfile
  clusters?: ClusteringResult
  time_window_comparison?: TimeWindowComparison
  top_n?: TopNResult
  whatif_chat_result?: WhatIfChatResult
  pred_errors?: PredictionErrorResult
  records?: RecordTableResult
  data_export?: DataExportResult
  null_map?: NullMapResult
  summary_stats?: SummaryStatsResult
  value_counts?: ValueCountResult
  pair_correlation?: PairCorrelationResult
  stat_query?: StatQueryResult
  group_trends?: GroupTrendResult
  split_strategy?: SplitStrategyResult
  feature_selection?: FeatureSelectionResult
  model_improvement?: ModelImprovementResult
  model_selection?: ModelSelectionResult
  auto_retrain?: AutoRetrainResult
  conversation_export?: ConversationExportInfo
  health_summary?: ProjectHealthSummary
  prediction_opportunities?: PredictionOpportunitiesResult
  dataset_comparison?: DatasetComparisonResult
  inline_prediction?: InlinePredictionResult
  multi_prediction?: MultiPredictionResult
  goal_training?: GoalTrainingResult
  sensitivity?: SensitivityResult
  interaction?: InteractionResult
  ranked_predictions?: RankedPredictionsResult
  prediction_cohort?: PredictionCohortResult
  onboarding_guide?: OnboardingGuideResult
  version_history?: DataVersionHistoryResult
  learning_curve?: LearningCurveResult
  template_saved?: TemplateSavedInfo
  template_list?: TemplateListInfo
  template_replay?: TemplateReplayInfo
  preset_saved?: PresetSavedInfo
  preset_list?: PresetListInfo
  sdk_download?: SdkDownloadInfo
  portfolio?: PortfolioResult
  rate_limit?: RateLimitInfo
}

export interface SegmentPerformanceSegment {
  name: string
  n: number
  metric: number | null
  metric_name: string
  status: "strong" | "moderate" | "weak" | "poor" | "insufficient_data"
  low_sample: boolean
}

export interface SegmentPerformanceResult {
  group_col: string
  algorithm: string
  problem_type: string
  metric_name: string
  segments: SegmentPerformanceSegment[]
  best_segment: string | null
  worst_segment: string | null
  gap: number | null
  summary: string
}

export interface QueryResponse {
  question: string
  answer: string
  chart_spec: ChartSpec | null
  result_rows: Record<string, unknown>[]
}

export interface FeatureSuggestion {
  id: string
  column: string
  transform_type:
    | "date_decompose"
    | "log_transform"
    | "one_hot"
    | "label_encode"
    | "bin_quartile"
    | "interaction"
  title: string
  description: string
  preview_columns: string[]
  example_values: (string | number | null)[]
}

export interface FeatureSetResult {
  feature_set_id: string
  column_mapping: Record<string, string[]>
  new_columns: string[]
  total_columns: number
  preview: Record<string, unknown>[]
}

export interface TargetResult {
  dataset_id: string
  target_column: string
  problem_type: "classification" | "regression" | null
  reason: string
  classes: string[]
}

export interface FeatureImportanceEntry {
  column: string
  importance: number
  importance_pct: number
  rank: number
  description: string
}

export interface FeatureImportanceResult {
  dataset_id: string
  target_column: string
  problem_type: string
  features: FeatureImportanceEntry[]
}

// ---------------------------------------------------------------------------
// Model Training (Phase 4)
// ---------------------------------------------------------------------------

export interface ModelRecommendation {
  algorithm: string
  name: string
  description: string
  plain_english: string
  best_for: string
  recommended_because: string
}

export interface ClassDistributionEntry {
  class: string
  count: number
  ratio: number
}

export interface ClassImbalanceResult {
  project_id: string
  problem_type: string
  is_imbalanced: boolean
  class_distribution: ClassDistributionEntry[]
  minority_class: string | null
  minority_ratio: number | null
  recommended_strategy: "class_weight" | "smote" | "threshold" | "none"
  explanation: string
}

export interface SplitStrategyInfo {
  recommended: "chronological" | "random"
  date_col: string | null
  explanation: string
}

export interface FeatureImportanceRow {
  name: string
  importance: number | null
  rank: number
  is_weak: boolean
}

export interface FeatureSelectionResult {
  run_id: string
  algorithm: string
  target_column: string
  n_features: number
  feature_importances: FeatureImportanceRow[]
  weak_features: string[]
  threshold: number | null
  method: "feature_importances" | "coefficients" | "not_available"
  has_importances: boolean
  n_weak: number
  explanation: string
}

export interface SplitStrategyResult {
  split_strategy: "chronological" | "random"
  date_col: string | null
  explanation: string
}

export interface ModelMetricsRegression {
  r2: number
  mae: number
  rmse: number
  train_size: number
  test_size: number
}

export interface ModelMetricsClassification {
  accuracy: number
  f1: number
  precision: number
  recall: number
  train_size: number
  test_size: number
}

export interface EnsembleMetricsExtra {
  ensemble_type?: "voting" | "stacking"
  ensemble_votes?: Record<string, number | Record<string, number>>
  stacking_weights?: Record<string, number>
  ensemble_summary?: string
}

export type ModelMetrics = (ModelMetricsRegression | ModelMetricsClassification) & EnsembleMetricsExtra

export interface ModelRun {
  id: string
  algorithm: string
  status: "pending" | "training" | "done" | "failed"
  is_selected: boolean
  is_deployed: boolean
  metrics: ModelMetrics | null
  summary: string | null
  training_duration_ms: number | null
  error_message: string | null
  created_at: string
}

export interface TrainingStatus {
  project_id: string
  model_run_ids: string[]
  algorithms: string[]
  status: string
  message: string
}

export interface ModelComparison {
  project_id: string
  problem_type: string
  models: ModelRun[]
  recommendation: {
    model_run_id: string
    algorithm: string
    reason: string
  } | null
}

// ---------------------------------------------------------------------------
// Validation & Explainability (Phase 5)
// ---------------------------------------------------------------------------

export interface CrossValidationResult {
  metric: string
  scores: number[]
  mean: number | null
  std: number | null
  ci_low: number | null
  ci_high: number | null
  n_splits: number
  summary: string
}

export interface ConfusionMatrixResult {
  type: "confusion_matrix"
  matrix: number[][]
  labels: string[]
  total: number
  correct: number
  accuracy: number
  summary: string
}

export interface ResidualsResult {
  type: "residuals"
  scatter: { predicted: number; residual: number }[]
  mae: number
  bias: number
  std: number
  percentile_75: number
  percentile_90: number
  summary: string
}

export type ErrorAnalysis = ConfusionMatrixResult | ResidualsResult

export interface ConfidenceAssessment {
  overall_confidence: "high" | "medium" | "low"
  limitations: string[]
  summary: string
}

export interface ValidationMetricsResponse {
  model_run_id: string
  algorithm: string
  problem_type: string
  held_out_metrics: Record<string, number>
  cross_validation: CrossValidationResult
  error_analysis: ErrorAnalysis
  confidence: ConfidenceAssessment
}

export interface FeatureImportanceItem {
  feature: string
  importance: number
  rank: number
}

export interface GlobalExplanationResponse {
  model_run_id: string
  algorithm: string
  problem_type: string
  feature_importance: FeatureImportanceItem[]
  summary: string
}

export interface ContributionItem {
  feature: string
  value: number
  mean_value: number
  contribution: number
  direction: "positive" | "negative"
}

export interface RowExplanationResponse {
  model_run_id: string
  row_index: number
  actual_value: number | null
  prediction: number | string
  prediction_value: number
  contributions: ContributionItem[]
  summary: string
}

// ---------------------------------------------------------------------------
// Deployment (Phase 6)
// ---------------------------------------------------------------------------

export interface FeatureSchemaEntry {
  name: string
  type: "numeric" | "categorical"
  options?: string[]   // categorical choices
  median?: number      // numeric default
  mean?: number        // training average (for hints)
  std?: number         // training std dev (for hints)
}

export interface Deployment {
  id: string
  model_run_id: string
  project_id: string
  endpoint_path: string
  dashboard_url: string
  is_active: boolean
  request_count: number
  algorithm: string | null
  problem_type: string | null
  feature_names: string[]
  target_column: string | null
  metrics: Record<string, number>
  created_at: string | null
  last_predicted_at: string | null
  feature_schema?: FeatureSchemaEntry[]
  api_key_enabled: boolean
  environment?: "staging" | "production"
}

export interface ApiKeyResult {
  deployment_id: string
  api_key: string
  message: string
}

export interface ConfidenceInterval {
  lower: number
  upper: number
  level: number
  label: string
}

export interface PredictionResult {
  deployment_id: string
  prediction: number | string
  problem_type: string
  target_column: string
  feature_names: string[]
  probabilities?: Record<string, number>
  confidence?: number
  confidence_interval?: ConfidenceInterval
}

// ---------------------------------------------------------------------------
// Prediction Explanation (Phase 8)
// ---------------------------------------------------------------------------

export interface FeatureContribution {
  feature: string
  value: number
  mean_value: number
  contribution: number
  direction: "positive" | "negative"
}

export interface PredictionExplanation {
  prediction: number | string
  target_column: string
  problem_type: string
  contributions: FeatureContribution[]
  summary: string
  top_drivers: string[]
}

// ---------------------------------------------------------------------------
// Prediction Monitoring (Phase 8)
// ---------------------------------------------------------------------------

export interface DeploymentAnalytics {
  deployment_id: string
  total_predictions: number
  predictions_by_day: { date: string; count: number }[]
  prediction_distribution: { bucket: string; count: number }[]
  recent_avg: number | null
  class_counts: Record<string, number> | null
  problem_type: string | null
}

export interface PredictionLogEntry {
  id: string
  input_features: Record<string, unknown>
  prediction: unknown
  confidence: number | null
  created_at: string
}

export interface PredictionLogsResponse {
  deployment_id: string
  total: number
  offset: number
  limit: number
  logs: PredictionLogEntry[]
}

// ---------------------------------------------------------------------------
// SLA Monitoring (Phase 9 Track D)
// ---------------------------------------------------------------------------

export interface SlaData {
  deployment_id: string
  sample_count: number
  p50_ms: number | null
  p95_ms: number | null
  p99_ms: number | null
  avg_ms: number | null
  alert: boolean
  alert_message: string | null
  latency_by_day: { date: string; avg_ms: number }[]
}

// ---------------------------------------------------------------------------
// Model Readiness (Phase 8)
// ---------------------------------------------------------------------------

export interface ReadinessCheck {
  id: string
  label: string
  passed: boolean
  detail?: string
  weight: number
}

export interface ModelReadiness {
  model_run_id: string
  algorithm: string
  score: number
  verdict: "ready" | "needs_attention" | "not_ready"
  summary: string
  checks: ReadinessCheck[]
  problem_type: string
}

// ---------------------------------------------------------------------------
// Drift Detection (Phase 8)
// ---------------------------------------------------------------------------

export type DriftStatus = "insufficient_data" | "stable" | "mild_drift" | "significant_drift"

export interface DriftNumericStats {
  mean: number
  std: number
  count: number
}

export interface DriftReport {
  deployment_id: string
  status: DriftStatus
  drift_score: number | null
  explanation: string
  baseline_stats: DriftNumericStats | null
  recent_stats: DriftNumericStats | null
  baseline_dist: Record<string, number> | null
  recent_dist: Record<string, number> | null
  problem_type: string | null
}

// ---------------------------------------------------------------------------
// What-if Analysis (Phase 8)
// ---------------------------------------------------------------------------

export interface WhatIfResult {
  deployment_id: string
  original_prediction: number | string
  modified_prediction: number | string
  changed_features: string[]
  delta: number | null
  percent_change: number | null
  direction: "increase" | "decrease" | "no change" | null
  summary: string
  problem_type: string | null
  target_column: string | null
  original_probabilities?: Record<string, number>
  modified_probabilities?: Record<string, number>
}

// ---------------------------------------------------------------------------
// Hyperparameter Tuning (Phase 8)
// ---------------------------------------------------------------------------

export interface TuningResult {
  original_model_run_id: string
  tuned_model_run_id: string | null
  algorithm: string
  tunable: boolean
  original_metrics: Record<string, number> | null
  tuned_metrics: Record<string, number> | null
  best_params: Record<string, unknown> | null
  tuned_cv_score: number | null
  improved: boolean
  improvement_pct: number | null
  summary: string
  tuned_run: ModelRun | null
}

// ---------------------------------------------------------------------------
// AI Project Narrative (Phase 8)
// ---------------------------------------------------------------------------

export interface ProjectNarrative {
  project_id: string
  project_name: string
  narrative: string
  generated_at: string
  context: Record<string, unknown>
}

// ---------------------------------------------------------------------------
// Prediction Feedback (Phase 8)
// ---------------------------------------------------------------------------

export interface FeedbackRecord {
  id: string
  deployment_id: string
  prediction_log_id: string | null
  actual_value: number | null
  actual_label: string | null
  is_correct: boolean | null
  comment: string | null
  created_at: string
  message: string
}

export type FeedbackAccuracyStatus =
  | "no_feedback"
  | "feedback_only"
  | "computed"

export interface FeedbackAccuracy {
  deployment_id: string
  status: FeedbackAccuracyStatus
  total_feedback: number
  message: string
  problem_type: string | null
  // Regression fields
  paired_count?: number
  mae?: number
  pct_error?: number
  avg_actual?: number
  // Classification fields
  rated_count?: number
  correct_count?: number
  incorrect_count?: number
  unknown_count?: number
  accuracy_from_feedback?: number
  // Both
  verdict?: "excellent" | "good" | "moderate" | "poor"
}


// ---------------------------------------------------------------------------
// Model Health (Phase 8)
// ---------------------------------------------------------------------------

export type ModelHealthStatus = "healthy" | "warning" | "critical"

export interface ModelHealth {
  deployment_id: string
  health_score: number
  status: ModelHealthStatus
  model_age_days: number
  component_scores: {
    age: number
    feedback: number | null
    drift: number | null
  }
  component_notes: {
    age: string
    feedback: string
    drift: string
  }
  recommendations: string[]
  has_feedback_data: boolean
  has_drift_data: boolean
  algorithm: string | null
  problem_type: string | null
}

export interface RetrainResponse {
  project_id: string
  model_run_ids: string[]
  algorithms: string[]
  status: string
  source_run_id: string
  message: string
}

// ---------------------------------------------------------------------------
// Model Version History (Phase 8)
// ---------------------------------------------------------------------------

export type ModelTrend = "improving" | "declining" | "stable" | "insufficient_data"

export interface ModelVersionHistory {
  project_id: string
  problem_type: string
  primary_metric: string
  primary_metric_label: string
  runs: ModelRun[]
  trend: ModelTrend
  trend_summary: string
  best_metric: number | null
  latest_metric: number | null
}

// ---------------------------------------------------------------------------
// Model Monitoring Alerts (Phase 8)
// ---------------------------------------------------------------------------

export type AlertSeverity = "critical" | "warning"
export type AlertType = "stale_model" | "no_predictions" | "drift_detected" | "poor_feedback"

export interface ProjectAlert {
  deployment_id: string
  algorithm: string
  severity: AlertSeverity
  type: AlertType
  message: string
  recommendation: string
}

export interface ProjectAlerts {
  project_id: string
  alert_count: number
  critical_count: number
  warning_count: number
  alerts: ProjectAlert[]
}

// ---------------------------------------------------------------------------
// Scenario Comparison (Phase 8)
// ---------------------------------------------------------------------------

export interface ScenarioItem {
  label: string
  overrides: Record<string, string | number>
}

export interface ScenarioResult {
  label: string
  overrides: Record<string, string | number>
  prediction: string | number
  delta: number | null
  percent_change: number | null
  direction: string | null
  probabilities: Record<string, number> | null
}

export interface ScenarioComparison {
  deployment_id: string
  base_prediction: string | number
  base_probabilities: Record<string, number> | null
  problem_type: string | null
  target_column: string | null
  scenarios: ScenarioResult[]
  summary: string
}

// ---------------------------------------------------------------------------
// Anomaly Detection (Phase 8)
// ---------------------------------------------------------------------------

export interface AnomalyRecord {
  row_index: number
  anomaly_score: number
  is_anomaly: boolean
  values: Record<string, number | null>
}

export interface AnomalyResult {
  dataset_id: string
  anomaly_count: number
  total_rows: number
  contamination_used: number
  top_anomalies: AnomalyRecord[]
  summary: string
  features_used: string[]
}

// ---------------------------------------------------------------------------
// Conversational data cleaning
// ---------------------------------------------------------------------------

export interface CleanOperation {
  operation: "remove_duplicates" | "fill_missing" | "filter_rows" | "cap_outliers" | "drop_column"
  column?: string
  strategy?: "mean" | "median" | "mode" | "zero" | "value"
  fill_value?: number | string
  operator?: "gt" | "lt" | "eq" | "ne" | "gte" | "lte" | "contains" | "notcontains"
  value?: number | string
  percentile?: number
}

export interface CleanOperationResult {
  operation: string
  column?: string
  strategy?: string
  fill_value_used?: string
  operator?: string
  value?: number | string
  before_rows: number
  after_rows: number
  before_columns?: number
  after_columns?: number
  modified_count: number
  summary: string
}

export interface CleanResult {
  dataset_id: string
  operation_result: CleanOperationResult
  preview: Record<string, unknown>[]
  updated_stats: {
    row_count: number
    column_count: number
    columns: ColumnStat[]
  }
}

export interface CleaningSuggestion {
  dataset_id: string
  suggested_operation: CleanOperation | null
  quality_summary: {
    duplicate_rows: number
    missing_value_columns: Record<string, number>
    total_rows: number
  }
}

export interface DatasetRefreshResult {
  dataset_id: string
  filename: string
  row_count: number
  column_count: number
  new_columns: string[]
  removed_columns: string[]
  feature_columns_missing: string[]
  compatible: boolean
  preview: Record<string, unknown>[]
  column_stats: ColumnStat[]
}

export interface RefreshPrompt {
  dataset_id: string
  current_filename: string
  current_row_count: number
  required_columns: string[]
}

// ---------------------------------------------------------------------------
// Cross-deployment model comparison
// ---------------------------------------------------------------------------

export interface ModelComparisonResult {
  deployment_id: string
  algorithm: string | null
  trained_at: string | null
  prediction: number | string | null
  problem_type?: string
  target_column?: string
  confidence_interval?: ConfidenceInterval
  confidence?: number
  probabilities?: Record<string, number>
  error: string | null
}

export interface ComparisonResponse {
  results: ModelComparisonResult[]
}

// ---------------------------------------------------------------------------
// Data Dictionary (Phase 8)
// ---------------------------------------------------------------------------

export type ColumnSemanticType = "id" | "metric" | "dimension" | "date" | "flag" | "text" | "unknown"

export interface ColumnDescription {
  name: string
  dtype: string
  col_type: ColumnSemanticType
  description: string
  non_null_count?: number
  null_count?: number
  null_pct?: number
  unique_count?: number
  min?: number | null
  max?: number | null
  mean?: number | null
  sample_values?: (string | number | null)[]
}

export interface DataDictionary {
  dataset_id: string
  filename: string
  generated: boolean
  columns: ColumnDescription[]
}

// ---------------------------------------------------------------------------
// Cross-tabulation / pivot table
// ---------------------------------------------------------------------------

export interface CrosstabRow {
  row_label: string
  cells: (number | null)[]
  row_total: number | null
}

export interface CrosstabResult {
  row_col: string
  col_col: string
  value_col: string | null
  agg_func: string
  col_headers: string[]
  rows: CrosstabRow[]
  col_totals: (number | null)[]
  grand_total: number | null
  summary: string
}

// ---------------------------------------------------------------------------
// Developer integration snippets
// ---------------------------------------------------------------------------

export interface IntegrationSnippets {
  deployment_id: string
  endpoint_url: string
  problem_type: string | null
  target_column: string | null
  algorithm: string | null
  example_input: Record<string, unknown>
  curl: string
  python: string
  javascript: string
  openapi_url: string
  batch_url: string
  batch_note: string
}

// ---------------------------------------------------------------------------
// Segment comparison
// ---------------------------------------------------------------------------

export interface SegmentColumnStats {
  name: string
  mean1: number | null
  std1: number | null
  median1: number | null
  count1: number
  mean2: number | null
  std2: number | null
  median2: number | null
  count2: number
  effect_size: number | null
  direction: 'higher_in_val1' | 'higher_in_val2' | null
}

export interface SegmentNotableDiff {
  name: string
  effect_size: number
  direction: 'higher_in_val1' | 'higher_in_val2'
}

export interface SegmentComparisonResult {
  group_col: string
  val1: string
  val2: string
  count1: number
  count2: number
  columns: SegmentColumnStats[]
  notable_diffs: SegmentNotableDiff[]
  summary: string
}

// ---------------------------------------------------------------------------
// Time-series forecasting
// ---------------------------------------------------------------------------

export interface ForecastPoint {
  date: string
  value: number
}

export interface ForecastFuturePoint {
  date: string
  value: number
  lower: number
  upper: number
}

export interface ForecastResult {
  chart_type: 'forecast'
  date_col: string
  value_col: string
  historical: ForecastPoint[]
  forecast: ForecastFuturePoint[]
  period_label: string
  trend: 'up' | 'down' | 'stable'
  growth_pct: number
  summary: string
  ci_level: number
}

// Data Readiness Assessment
export interface ReadinessComponent {
  name: string
  score: number
  max_score: number
  status: 'good' | 'warning' | 'critical'
  detail: string
  recommendation?: string
  advisory?: boolean
}

export interface DataReadinessResult {
  dataset_id: string
  score: number
  grade: 'A' | 'B' | 'C' | 'D' | 'F'
  status: 'ready' | 'needs_attention' | 'not_ready'
  summary: string
  components: ReadinessComponent[]
  recommendations: string[]
}

// ---------------------------------------------------------------------------
// Target correlation analysis
// ---------------------------------------------------------------------------

export interface CorrelationEntry {
  column: string
  correlation: number
  strength: 'very strong' | 'strong' | 'moderate' | 'weak' | 'negligible'
  direction: 'positive' | 'negative'
}

export interface TargetCorrelationResult {
  dataset_id: string
  target_col: string
  correlations: CorrelationEntry[]
  summary: string
}


// ---------------------------------------------------------------------------
// Group-by Analysis
// ---------------------------------------------------------------------------

export interface GroupStatsRow {
  group: string
  [metric: string]: string | number | null
}

export interface GroupStatsResult {
  dataset_id: string
  group_col: string
  value_col: string
  value_cols: string[]
  agg: 'sum' | 'mean' | 'count' | 'min' | 'max' | 'median'
  rows: GroupStatsRow[]
  total: number | null
  summary: string
}

// ---------------------------------------------------------------------------
// Automated Data Story (Phase 8)
// ---------------------------------------------------------------------------

export interface DataStorySection {
  type: 'readiness' | 'group_by' | 'correlations' | 'anomalies'
  title: string
  insight: string
  data: Record<string, unknown>
}

export interface DataStory {
  dataset_id: string
  filename: string
  row_count: number
  col_count: number
  readiness_score: number
  readiness_grade: string
  sections: DataStorySection[]
  summary: string
  recommended_next_step: string
}

// ---------------------------------------------------------------------------
// Non-destructive data filter (Phase 8)
// ---------------------------------------------------------------------------

export interface DateRangeValue {
  start: string
  end: string
}

export interface FilterCondition {
  column: string
  operator: 'eq' | 'ne' | 'gt' | 'lt' | 'gte' | 'lte' | 'contains' | 'not_contains' | 'date_range'
  value: string | number | DateRangeValue
}

export interface ActiveFilter {
  dataset_id: string
  active: boolean
  filter_summary?: string
  conditions?: FilterCondition[]
  original_rows?: number
  filtered_rows?: number
  row_reduction_pct?: number
}

export interface FilterSetResult {
  dataset_id: string
  filter_summary: string
  conditions: FilterCondition[]
  original_rows: number
  filtered_rows: number
  row_reduction_pct: number
}

export interface ColumnProfileIssue {
  type: string
  severity: "critical" | "warning" | "info"
  message: string
}

export interface ColumnProfileDistribution {
  type: "histogram" | "bar" | "date" | "unknown"
  bins?: number[]
  counts?: number[]
  labels?: string[]
}

export interface ColumnProfileStats {
  total_rows: number
  null_count: number
  null_pct: number
  unique_count: number
  // numeric
  min?: number
  max?: number
  mean?: number
  median?: number
  std?: number
  p25?: number
  p75?: number
  skewness?: number
  // categorical
  most_common?: string
  most_common_pct?: number
  top_categories?: { label: string; count: number }[]
  // date
  min_date?: string
  max_date?: string
  date_range_days?: number
  estimated_frequency?: string
}

export interface ColumnProfile {
  col_name: string
  col_type: "numeric" | "categorical" | "date"
  stats: ColumnProfileStats
  distribution: ColumnProfileDistribution
  issues: ColumnProfileIssue[]
  summary: string
}

export interface ClusterDistinguishingFeature {
  feature: string
  cluster_mean: number
  global_mean: number
  direction: "above" | "below"
  magnitude: number
}

export interface ClusterProfile {
  cluster_id: number
  size: number
  size_pct: number
  centroid: Record<string, number>
  distinguishing: ClusterDistinguishingFeature[]
  description: string
}

export interface ClusteringResult {
  n_clusters: number
  features_used: string[]
  auto_k: boolean
  rows_clustered: number
  clusters: ClusterProfile[]
  summary: string
}

export interface TimeWindowPeriod {
  name: string
  start: string
  end: string
  row_count: number
}

export interface TimeWindowColumn {
  column: string
  p1_mean: number
  p2_mean: number
  pct_change: number
  direction: "up" | "down" | "flat"
  notable: boolean
}

export interface TimeWindowComparison {
  date_col: string
  period1: TimeWindowPeriod
  period2: TimeWindowPeriod
  columns: TimeWindowColumn[]
  notable_changes: string[]
  summary: string
}

// ---------------------------------------------------------------------------
// What-If Chat Analysis (Day 16)
// ---------------------------------------------------------------------------

export interface WhatIfChatResult {
  deployment_id: string
  changed_feature: string
  original_feature_value: number | string
  new_feature_value: number | string
  original_prediction: number | string
  modified_prediction: number | string
  delta: number | null
  percent_change: number | null
  direction: "increase" | "decrease" | "no change" | null
  summary: string
  problem_type: string | null
  target_column: string | null
  original_probabilities?: Record<string, number>
  modified_probabilities?: Record<string, number>
}

export interface TopNRow {
  _rank: number
  [key: string]: number | string | null
}

export interface TopNResult {
  sort_col: string
  direction: "top" | "bottom"
  ascending: boolean
  n_requested: number
  n_returned: number
  total_rows: number
  display_cols: string[]
  rows: TopNRow[]
  summary: string
}

export interface RecordTableRow {
  [key: string]: number | string | null
}

export interface RecordTableResult {
  columns: string[]
  rows: RecordTableRow[]
  total_rows: number
  filtered_rows: number
  shown_rows: number
  filtered: boolean
  condition_summary: string
  summary: string
}

export interface PredictionErrorRow {
  actual: string | number
  predicted: string | number
  error: string | number
  abs_error: number | null
  rank: number
  features?: Record<string, string | number>
}

export interface PredictionErrorResult {
  algorithm: string
  target_col: string
  problem_type: "regression" | "classification"
  errors: PredictionErrorRow[]
  total_errors: number
  error_rate: number
  summary: string
}

export interface BatchSchedule {
  id: string
  deployment_id: string
  frequency: "daily" | "weekly" | "monthly"
  run_hour: number
  run_minute: number
  day_of_week: number | null
  day_of_month: number | null
  is_active: boolean
  last_run: string | null
  next_run: string | null
  last_output_path: string | null
  last_row_count: number | null
  last_error: string | null
  created_at: string
}

export interface BatchJobRun {
  id: string
  schedule_id: string
  deployment_id: string
  started_at: string
  completed_at: string | null
  status: "running" | "success" | "failed"
  row_count: number | null
  error: string | null
  download_url: string | null
}

export interface DeploymentVersion {
  id: string
  deployment_id: string
  version_number: number
  model_run_id: string
  algorithm: string | null
  problem_type: string | null
  target_column: string | null
  metrics: Record<string, number>
  pipeline_path: string | null
  deployed_at: string | null
  is_current: boolean
}

export interface DeploymentVersionHistory {
  deployment_id: string
  current_version_number: number
  versions: DeploymentVersion[]
}

export interface RollbackResult {
  rolled_back_to_version: number
  new_version_number: number
  id: string
  model_run_id: string
  endpoint_path: string
  algorithm: string | null
  metrics: Record<string, number>
  api_key_enabled: boolean
}

export interface EnvironmentPromotionResult {
  message: string
  deployment: Deployment
}

export interface WebhookConfig {
  id: string
  deployment_id: string
  url: string
  event_types: string[]
  is_active: boolean
  created_at: string | null
  last_fired_at: string | null
  last_status_code: number | null
  /** Only present on initial creation response */
  secret?: string
}

export interface WebhookTestResult {
  webhook_id: string
  url: string
  status_code: number
  success: boolean
}

export interface ABVariantMetrics {
  request_count: number
  avg_confidence: number | null
  p95_ms: number | null
  avg_prediction: number | null
}

export interface ABSignificance {
  significant: boolean
  p_value: number | null
  note: string
}

export interface ABTest {
  id: string
  champion_id: string
  challenger_id: string
  champion_algorithm: string | null
  challenger_algorithm: string | null
  champion_split_pct: number
  challenger_split_pct: number
  is_active: boolean
  auto_promote: boolean
  created_at: string | null
  ended_at: string | null
  winner: string | null
  champion_metrics: ABVariantMetrics
  challenger_metrics: ABVariantMetrics
  significance: ABSignificance
}

export interface CalibrationPoint {
  predicted: number
  actual: number
}

export interface CalibrationData {
  run_id: string
  algorithm: string
  brier_score: number | null
  calibration_curve: CalibrationPoint[]
  calibration_note: string
  is_calibrated: boolean
}

export type ImprovementDifficulty = "easy" | "medium" | "hard"
export type ImprovementImpact = "low" | "moderate" | "high"
export type ImprovementAction =
  | "feature_selection"
  | "train_ensemble"
  | "feature_engineering"
  | "add_data"
  | "class_imbalance"
  | "calibration"
  | "hyperparameter_tuning"
  | "add_features"
  | "train_nonlinear"

export interface ImprovementSuggestion {
  rank: number
  category: string
  title: string
  explanation: string
  action: ImprovementAction
  difficulty: ImprovementDifficulty
  expected_impact: ImprovementImpact
}

export interface ModelImprovementResult {
  run_id: string
  project_id: string
  algorithm: string
  problem_type: "regression" | "classification"
  primary_metric: number
  primary_metric_name: string
  suggestions: ImprovementSuggestion[]
  summary: string
  n_suggestions: number
}

// ---------------------------------------------------------------------------
// Model Selection Advisor
// ---------------------------------------------------------------------------

export type SelectionCriteria = "accuracy" | "explainability" | "stability" | "speed" | "balanced"

export interface ModelSelectionComponentScores {
  accuracy: number
  explainability: number
  stability: number
  speed: number
}

export interface ModelSelectionRun {
  run_id: string
  algorithm: string
  algorithm_plain: string
  score: number
  primary_metric: number
  primary_metric_name: string
  component_scores: ModelSelectionComponentScores
  why: string
  is_selected: boolean
  is_deployed: boolean
  rank: number
}

export interface ModelSelectionResult {
  project_id: string
  criteria: SelectionCriteria
  criteria_description: string
  winner: ModelSelectionRun | null
  ranked_runs: ModelSelectionRun[]
  summary: string
  n_runs: number
}

export interface AutoRetrainResult {
  project_id: string
  enabled: boolean
  selected_algorithm: string | null
  has_selected_model: boolean
}

export interface ConversationExportInfo {
  project_id: string
  download_url: string
  message_count: number
  dataset_name: string | null
}

export interface DeploymentHealthItem {
  deployment_id: string
  name: string
  algorithm_plain: string
  target_column: string
  environment: string
  health_score: number
  status: "healthy" | "warning" | "critical"
  top_issue: string | null
  recommendation: string | null
  age_score: number
  usage_score: number
}

export interface ProjectHealthSummary {
  project_id: string
  total: number
  healthy: number
  warning: number
  critical: number
  alerts: DeploymentHealthItem[]
  all_items: DeploymentHealthItem[]
  overall_status: "healthy" | "warning" | "critical"
  summary: string
}

export interface PredictionOpportunity {
  target_col: string
  problem_type: "regression" | "classification"
  feasibility_score: number
  reason: string
  business_value: "high" | "medium" | "low"
  example_question: string
  predictor_count: number
}

export interface PredictionOpportunitiesResult {
  dataset_id: string
  opportunities: PredictionOpportunity[]
  total: number
}

export interface NumericDrift {
  col: string
  old_mean: number
  new_mean: number
  old_std: number
  new_std: number
  pct_change: number
  severity: "low" | "medium" | "high"
}

export interface CategoricalDrift {
  col: string
  new_categories: string[]
  dropped_categories: string[]
  top_shift_pct: number
  severity: "medium" | "high"
}

export interface DatasetComparisonResult {
  baseline_id: string
  new_id: string
  baseline_name: string
  new_name: string
  row_count_old: number
  row_count_new: number
  row_count_change_pct: number
  col_count_old: number
  col_count_new: number
  new_columns: string[]
  dropped_columns: string[]
  numeric_drifts: NumericDrift[]
  categorical_drifts: CategoricalDrift[]
  drift_score: number
  summary: string
}

// ---------------------------------------------------------------------------
// Inline Multi-Feature Prediction via Chat (Day 25)
// ---------------------------------------------------------------------------

export interface InlinePredictionResult {
  deployment_id: string
  target_column: string
  prediction: number | string
  probabilities?: Record<string, number>
  confidence_interval?: { lower: number; upper: number } | null
  confidence?: number | null
  provided_features: Record<string, number | string>
  defaults_used_count: number
  total_features: number
  summary: string
  problem_type: string | null
}

export interface MultiPredictionRow {
  row_index: number
  provided_features: Record<string, number | string>
  defaults_used_count: number
  prediction: number | string
  probabilities?: Record<string, number>
  confidence?: number | null
  confidence_interval?: { lower: number; upper: number } | null
}

export interface MultiPredictionResult {
  deployment_id: string
  target_column: string
  problem_type: string | null
  rows: MultiPredictionRow[]
  summary: string
}

export interface GoalTrainingTrial {
  algorithm: string
  algorithm_name: string
  score: number
  achieved_goal: boolean
}

export interface GoalTrainingResult {
  project_id: string
  target_col: string
  goal_metric: string
  goal_target: number
  achieved: boolean
  winner_algorithm: string
  winner_algorithm_name: string
  winner_score: number
  trials: GoalTrainingTrial[]
  tried_tuning: boolean
  summary: string
}


export interface SensitivityResult {
  feature: string
  target_column: string
  problem_type: string
  values: number[]
  predictions: (number | string)[]
  confidences: (number | null)[]
  min_pred: number | null
  max_pred: number | null
  change_pct: number | null
  summary: string
}

export interface OnboardingStep {
  name: string
  title: string
  description: string
  hint: string
  suggested_action: string
  suggested_tab: string | null
  icon: string
  is_done: boolean
  is_current: boolean
}

export interface OnboardingGuideResult {
  step_index: number
  total_steps: number
  completion_pct: number
  steps: OnboardingStep[]
  current_step: OnboardingStep | null
  is_complete: boolean
  summary: string
  project_id?: string
}

export interface DataVersionDrift {
  drift_score: number
  summary: string
  changed_columns: number
  new_columns: string[]
  dropped_columns: string[]
  row_count_change_pct: number
}

export interface DataVersionEntry {
  version: number
  dataset_id: string
  filename: string
  row_count: number
  column_count: number
  uploaded_at: string
  size_bytes: number
  drift_from_previous: DataVersionDrift | null
}

export interface DataVersionHistoryResult {
  version_count: number
  versions: DataVersionEntry[]
  overall_stability: "stable" | "moderate" | "high"
  summary: string
}

// Learning Curve Analysis
export interface LearningCurveResult {
  sizes_pct: number[]
  train_scores: number[]
  val_scores: number[]
  converged: boolean
  plateau_pct: number | null
  best_val_score: number
  metric_label: string
  metric_key: string
  n_total: number
  algorithm: string
  algorithm_name: string
  recommendation: string
  summary: string
}

// Analysis Templates
export interface AnalysisTemplate {
  id: string
  project_id: string
  name: string
  queries: string[]
  description?: string
  created_at: string
}

export interface TemplateSavedInfo {
  id: string
  name: string
  queries: string[]
  query_count: number
}

export interface TemplateListInfo {
  templates: Array<{
    id: string
    name: string
    queries: string[]
    query_count: number
    created_at: string
  }>
  count: number
}

export interface TemplateReplayInfo {
  id: string
  name: string
  queries: string[]
  query_count: number
}

export interface InteractionResult {
  feature1: string
  feature2: string
  target_column: string
  problem_type: string
  row_labels: string[]
  col_labels: string[]
  values: (number | string)[][]
  min_val: number | null
  max_val: number | null
  summary: string
}

export interface RankedPredictionRow {
  rank: number
  row_index: number
  score: number
  feature_values: Record<string, string | number | null>
  prediction?: number
  predicted_class?: string
  confidence?: number
  probabilities?: Record<string, number>
}

export interface RankedPredictionsResult {
  problem_type: string
  target_column: string
  direction: string
  n: number
  total_scored: number
  rows: RankedPredictionRow[]
  summary: string
  class_names: string[] | null
}

export interface CohortCategoryEntry {
  value: string
  top_pct: number
  overall_pct: number
  ratio: number
}

export interface CohortCategoricalProfile {
  column: string
  categories: CohortCategoryEntry[]
  dominant: string
  dominant_top_pct: number
}

export interface CohortNumericProfile {
  column: string
  top_mean: number
  overall_mean: number
  ratio: number | null
  direction: "higher" | "lower" | "similar"
}

export interface PredictionCohortResult {
  target_column: string
  problem_type: string
  n: number
  direction: string
  total_scored: number
  categorical_profile: CohortCategoricalProfile[]
  numeric_profile: CohortNumericProfile[]
  characterization: string
}

// ---------------------------------------------------------------------------
// Prediction Presets
// ---------------------------------------------------------------------------

export interface DeploymentPreset {
  id: string
  deployment_id: string
  name: string
  feature_values: Record<string, string | number>
  created_at: string | null
}

export interface PresetSavedInfo {
  id: string
  deployment_id: string
  name: string
  feature_values: Record<string, string | number>
  feature_count: number
}

export interface PresetListInfo {
  presets: Array<{
    id: string
    name: string
    feature_values: Record<string, string | number>
    feature_count: number
  }>
  count: number
  deployment_id: string
}

export interface SdkDownloadInfo {
  deployment_id: string
  target_column: string
  algorithm: string
  problem_type: string
  python_url: string
  javascript_url: string
  class_name: string
}

// ---------------------------------------------------------------------------
// Cross-Project Portfolio Overview
// ---------------------------------------------------------------------------

export interface PortfolioProjectSummary {
  project_id: string
  name: string
  dataset_filename: string | null
  row_count: number | null
  model_count: number
  best_algorithm: string | null
  best_metric_name: string | null
  best_metric_value: number | null
  best_problem_type: string | null
  best_target_column: string | null
  has_deployment: boolean
  prediction_count: number
  last_activity_at: string | null
}

export interface PortfolioBestPerformer {
  project_id: string
  name: string
  metric_name: string
  metric_value: number
  algorithm: string
  problem_type: string
  target_column: string
}

export interface PortfolioResult {
  total_projects: number
  active_deployments: number
  total_predictions: number
  best_performer: PortfolioBestPerformer | null
  projects: PortfolioProjectSummary[]
  summary: string
}

export interface RateLimitInfo {
  deployment_id: string
  rate_limit_rpm: number | null
  rate_limit_enabled: boolean
  monthly_quota: number | null
  quota_enabled: boolean
  used_this_month: number
  remaining: number | null
  pct_used: number | null
  summary: string
}

export interface QuotaStatus {
  deployment_id: string
  quota_enabled: boolean
  monthly_quota: number | null
  used_this_month: number
  remaining: number | null
  pct_used: number | null
  rate_limit_rpm: number | null
  rate_limit_enabled: boolean
  summary: string
}
