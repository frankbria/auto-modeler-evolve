export interface Project {
  id: string
  name: string
  description?: string
  created_at: string
  updated_at: string
  status: "exploring" | "modeling" | "deployed"
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
  chart_type: "bar" | "line" | "histogram" | "scatter" | "pie"
  title: string
  data: Record<string, unknown>[]
  x_key: string
  y_keys: string[]
  x_label: string
  y_label: string
}

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  timestamp: string
  chart?: ChartSpec
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
