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

export type ModelMetrics = ModelMetricsRegression | ModelMetricsClassification

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

export interface FilterCondition {
  column: string
  operator: 'eq' | 'ne' | 'gt' | 'lt' | 'gte' | 'lte' | 'contains' | 'not_contains'
  value: string | number
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
