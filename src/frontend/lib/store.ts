import { create } from "zustand"
import type {
  Project,
  Dataset,
  ChatMessage,
  ColumnStat,
  ChartSpec,
  CrosstabResult,
  ComputedColumnSuggestion,
  DataInsight,
  SegmentComparisonResult,
  ForecastResult,
  DataReadinessResult,
  TargetCorrelationResult,
  GroupStatsResult,
  RenameResult,
  TrainingStartedResult,
  DataStory,
  FilterSetResult,
  ActiveFilter,
  DeployedResult,
  ModelCard,
  ReportReady,
  FeatureSuggestionsChatResult,
  FeaturesAppliedResult,
} from "./types"

interface AppState {
  projects: Project[]
  currentProject: Project | null
  currentDataset: Dataset | null
  dataPreview: Record<string, unknown>[]
  columnStats: ColumnStat[]
  dataInsights: DataInsight[]
  messages: ChatMessage[]
  isStreaming: boolean
  activeFilter: ActiveFilter | null

  setProjects: (projects: Project[]) => void
  setCurrentProject: (project: Project | null) => void
  setDataset: (
    dataset: Dataset,
    preview: Record<string, unknown>[],
    stats: ColumnStat[],
    insights?: DataInsight[]
  ) => void
  addMessage: (message: ChatMessage) => void
  setMessages: (messages: ChatMessage[]) => void
  setStreaming: (streaming: boolean) => void
  appendToLastMessage: (content: string) => void
  attachChartToLastMessage: (chart: ChartSpec) => void
  attachCrosstabToLastMessage: (crosstab: CrosstabResult) => void
  attachComputeToLastMessage: (compute: ComputedColumnSuggestion) => void
  attachSegmentToLastMessage: (segment_comparison: SegmentComparisonResult) => void
  attachForecastToLastMessage: (forecast: ForecastResult) => void
  attachDataReadinessToLastMessage: (data_readiness: DataReadinessResult) => void
  attachCorrelationToLastMessage: (target_correlation: TargetCorrelationResult) => void
  attachGroupStatsToLastMessage: (group_stats: GroupStatsResult) => void
  attachRenameResultToLastMessage: (rename_result: RenameResult) => void
  attachTrainingStartedToLastMessage: (training_started: TrainingStartedResult) => void
  attachDataStoryToLastMessage: (data_story: DataStory) => void
  attachFilterToLastMessage: (filter_set: FilterSetResult) => void
  setActiveFilter: (filter: ActiveFilter | null) => void
  attachDeployedToLastMessage: (deployed: DeployedResult) => void
  attachModelCardToLastMessage: (model_card: ModelCard) => void
  attachReportToLastMessage: (report_ready: ReportReady) => void
  attachFeatureSuggestionsToLastMessage: (feature_suggestions: FeatureSuggestionsChatResult) => void
  attachFeaturesAppliedToLastMessage: (features_applied: FeaturesAppliedResult) => void
  attachSegmentPerformanceToLastMessage: (segment_performance: import("./types").SegmentPerformanceResult) => void
  attachColumnProfileToLastMessage: (column_profile: import("./types").ColumnProfile) => void
  attachClustersToLastMessage: (clusters: import("./types").ClusteringResult) => void
  attachTimeWindowToLastMessage: (time_window_comparison: import("./types").TimeWindowComparison) => void
  attachTopNToLastMessage: (top_n: import("./types").TopNResult) => void
  attachWhatIfChatToLastMessage: (whatif_chat_result: import("./types").WhatIfChatResult) => void
  attachPredictionErrorsToLastMessage: (pred_errors: import("./types").PredictionErrorResult) => void
  attachRecordsToLastMessage: (records: import("./types").RecordTableResult) => void
  attachDataExportToLastMessage: (data_export: import("./types").DataExportResult) => void
  attachNullMapToLastMessage: (null_map: import("./types").NullMapResult) => void
  attachSummaryStatsToLastMessage: (summary_stats: import("./types").SummaryStatsResult) => void
  attachValueCountsToLastMessage: (value_counts: import("./types").ValueCountResult) => void
  attachPairCorrelationToLastMessage: (pair_correlation: import("./types").PairCorrelationResult) => void
  attachStatQueryToLastMessage: (stat_query: import("./types").StatQueryResult) => void
  attachGroupTrendsToLastMessage: (group_trends: import("./types").GroupTrendResult) => void
  attachSplitStrategyToLastMessage: (split_strategy: import("./types").SplitStrategyResult) => void
  attachFeatureSelectionToLastMessage: (feature_selection: import("./types").FeatureSelectionResult) => void
  attachModelImprovementToLastMessage: (model_improvement: import("./types").ModelImprovementResult) => void
  attachModelSelectionToLastMessage: (model_selection: import("./types").ModelSelectionResult) => void
  attachAutoRetrainToLastMessage: (auto_retrain: import("./types").AutoRetrainResult) => void
  attachConversationExportToLastMessage: (conversation_export: import("./types").ConversationExportInfo) => void
  attachHealthSummaryToLastMessage: (health_summary: import("./types").ProjectHealthSummary) => void
  attachPredictionOpportunitiesToLastMessage: (prediction_opportunities: import("./types").PredictionOpportunitiesResult) => void
  attachDatasetComparisonToLastMessage: (dataset_comparison: import("./types").DatasetComparisonResult) => void
  attachInlinePredictionToLastMessage: (inline_prediction: import("./types").InlinePredictionResult) => void
  attachMultiPredictionToLastMessage: (multi_prediction: import("./types").MultiPredictionResult) => void
  attachGoalTrainingToLastMessage: (goal_training: import("./types").GoalTrainingResult) => void
  attachSensitivityToLastMessage: (sensitivity: import("./types").SensitivityResult) => void
  attachInteractionToLastMessage: (interaction: import("./types").InteractionResult) => void
  attachRankedPredictionsToLastMessage: (ranked_predictions: import("./types").RankedPredictionsResult) => void
  attachPredictionCohortToLastMessage: (prediction_cohort: import("./types").PredictionCohortResult) => void
  attachOnboardingGuideToLastMessage: (onboarding_guide: import("./types").OnboardingGuideResult) => void
  attachVersionHistoryToLastMessage: (version_history: import("./types").DataVersionHistoryResult) => void
  attachLearningCurveToLastMessage: (learning_curve: import("./types").LearningCurveResult) => void
  attachTemplateSavedToLastMessage: (template: import("./types").TemplateSavedInfo) => void
  attachTemplateListToLastMessage: (template_list: import("./types").TemplateListInfo) => void
  attachTemplateReplayToLastMessage: (template_replay: import("./types").TemplateReplayInfo) => void
  attachPresetSavedToLastMessage: (preset: import("./types").PresetSavedInfo) => void
  attachPresetListToLastMessage: (preset_list: import("./types").PresetListInfo) => void
  attachSdkDownloadToLastMessage: (sdk_download: import("./types").SdkDownloadInfo) => void
  attachPortfolioToLastMessage: (portfolio: import("./types").PortfolioResult) => void
  attachRateLimitToLastMessage: (rate_limit: import("./types").RateLimitInfo) => void
  attachPartialDependenceToLastMessage: (partial_dependence: import("./types").PartialDependenceResult) => void
  attachCalibrationCheckToLastMessage: (calibration_check: import("./types").CalibrationCheckResult) => void
  attachSlaMetricsToLastMessage: (sla_metrics: import("./types").SlaData) => void
  attachQuotaAlertConfigToLastMessage: (quota_alert_config: import("./types").QuotaAlertConfig) => void
  attachScheduleSetToLastMessage: (schedule_set: import("./types").ScheduleSetResult) => void
  attachABTestResultToLastMessage: (ab_test_result: import("./types").ABTestChatResult) => void
  attachWebhookHistoryToLastMessage: (webhook_history: import("./types").WebhookHistoryResult) => void
  attachClassImbalanceCheckToLastMessage: (class_imbalance_check: import("./types").ClassImbalanceResult) => void
  attachWebhookHealthSummaryToLastMessage: (webhook_health_summary: import("./types").WebhookHealthSummaryResult) => void
  attachExecutiveBriefingToLastMessage: (executive_briefing: import("./types").ExecutiveBriefingResult) => void
  attachServiceExportToLastMessage: (service_export: import("./types").ServiceExportChatResult) => void
  attachVersionComparisonToLastMessage: (version_comparison: import("./types").DeploymentVersionComparisonResult) => void
  attachEnsembleRecommendationToLastMessage: (ensemble_recommendation: import("./types").EnsembleRecommendationResult) => void
  attachTuneChatToLastMessage: (tune_chat: import("./types").TuningChatResult) => void
  attachCvScoreDistributionToLastMessage: (cv_score_distribution: import("./types").CvScoreDistributionResult) => void
  attachPredictionAnalyticsChatToLastMessage: (prediction_analytics_chat: import("./types").PredictionAnalyticsChatResult) => void
  attachConfusionMatrixChatToLastMessage: (confusion_matrix_chat: import("./types").ConfusionMatrixChatResult) => void
  attachLocalExplanationToLastMessage: (local_explanation: import("./types").LocalExplanationResult) => void
  attachProdInputDistToLastMessage: (prod_input_dist: import("./types").ProductionInputDistributionResult) => void
  attachCovariateDriftAlertToLastMessage: (covariate_drift_alert: import("./types").CovariateDriftAlertResult) => void
  attachQuotaRunwayToLastMessage: (quota_runway: import("./types").QuotaRunwayResult) => void
  attachCostEstimateToLastMessage: (cost_estimate: import("./types").CostEstimateResult) => void
  attachUsagePatternToLastMessage: (usage_pattern: import("./types").UsagePatternResult) => void
  attachPredictionLogExportToLastMessage: (prediction_log_export: import("./types").PredictionLogExportResult) => void
  attachRecentPredictionsToLastMessage: (recent_predictions: import("./types").RecentPredictionsResult) => void
  attachPredictionAuditToLastMessage: (prediction_audit: import("./types").PredictionAuditResult) => void
  attachConfidenceTrendToLastMessage: (confidence_trend: import("./types").ConfidenceTrendResult) => void
  attachFeedbackAccuracyReportToLastMessage: (feedback_accuracy_report: import("./types").FeedbackAccuracyReportResult) => void
  attachFairnessCheckToLastMessage: (fairness_check: import("./types").FairnessCheckResult) => void
  attachBatchJobResultsToLastMessage: (batch_job_results: import("./types").BatchJobResultsResult) => void
  attachProdPredictionExplanationToLastMessage: (prod_prediction_explanation: import("./types").ProdPredictionExplanationResult) => void
}

export const useAppStore = create<AppState>((set) => ({
  projects: [],
  currentProject: null,
  currentDataset: null,
  dataPreview: [],
  columnStats: [],
  dataInsights: [],
  messages: [],
  isStreaming: false,
  activeFilter: null,

  setProjects: (projects) => set({ projects }),

  setCurrentProject: (project) => set({ currentProject: project }),

  setDataset: (dataset, preview, stats, insights = []) =>
    set({
      currentDataset: dataset,
      dataPreview: preview,
      columnStats: stats,
      dataInsights: insights,
    }),

  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  setMessages: (messages) => set({ messages }),

  setStreaming: (streaming) => set({ isStreaming: streaming }),

  appendToLastMessage: (content) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, content: last.content + content }
      }
      return { messages }
    }),

  attachChartToLastMessage: (chart) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, chart }
      }
      return { messages }
    }),

  attachCrosstabToLastMessage: (crosstab) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, crosstab }
      }
      return { messages }
    }),

  attachComputeToLastMessage: (compute) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, compute }
      }
      return { messages }
    }),

  attachSegmentToLastMessage: (segment_comparison) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, segment_comparison }
      }
      return { messages }
    }),

  attachForecastToLastMessage: (forecast) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, forecast }
      }
      return { messages }
    }),

  attachDataReadinessToLastMessage: (data_readiness) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, data_readiness }
      }
      return { messages }
    }),

  attachCorrelationToLastMessage: (target_correlation) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, target_correlation }
      }
      return { messages }
    }),

  attachGroupStatsToLastMessage: (group_stats) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, group_stats }
      }
      return { messages }
    }),

  attachRenameResultToLastMessage: (rename_result) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, rename_result }
      }
      return { messages }
    }),

  attachTrainingStartedToLastMessage: (training_started) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, training_started }
      }
      return { messages }
    }),

  attachDataStoryToLastMessage: (data_story) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, data_story }
      }
      return { messages }
    }),

  attachFilterToLastMessage: (filter_set) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, filter_set }
      }
      return { messages }
    }),

  setActiveFilter: (filter) => set({ activeFilter: filter }),

  attachDeployedToLastMessage: (deployed) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, deployed }
      }
      return { messages }
    }),

  attachModelCardToLastMessage: (model_card) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, model_card }
      }
      return { messages }
    }),

  attachReportToLastMessage: (report_ready) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, report_ready }
      }
      return { messages }
    }),

  attachFeatureSuggestionsToLastMessage: (feature_suggestions) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, feature_suggestions }
      }
      return { messages }
    }),

  attachFeaturesAppliedToLastMessage: (features_applied) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, features_applied }
      }
      return { messages }
    }),

  attachSegmentPerformanceToLastMessage: (segment_performance) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, segment_performance }
      }
      return { messages }
    }),

  attachColumnProfileToLastMessage: (column_profile) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, column_profile }
      }
      return { messages }
    }),

  attachClustersToLastMessage: (clusters) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, clusters }
      }
      return { messages }
    }),

  attachTimeWindowToLastMessage: (time_window_comparison) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, time_window_comparison }
      }
      return { messages }
    }),

  attachTopNToLastMessage: (top_n) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, top_n }
      }
      return { messages }
    }),

  attachWhatIfChatToLastMessage: (whatif_chat_result) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, whatif_chat_result }
      }
      return { messages }
    }),

  attachPredictionErrorsToLastMessage: (pred_errors) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, pred_errors }
      }
      return { messages }
    }),

  attachRecordsToLastMessage: (records) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, records }
      }
      return { messages }
    }),

  attachDataExportToLastMessage: (data_export) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, data_export }
      }
      return { messages }
    }),

  attachNullMapToLastMessage: (null_map) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, null_map }
      }
      return { messages }
    }),

  attachSummaryStatsToLastMessage: (summary_stats) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, summary_stats }
      }
      return { messages }
    }),

  attachValueCountsToLastMessage: (value_counts) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, value_counts }
      }
      return { messages }
    }),

  attachPairCorrelationToLastMessage: (pair_correlation) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, pair_correlation }
      }
      return { messages }
    }),

  attachStatQueryToLastMessage: (stat_query) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, stat_query }
      }
      return { messages }
    }),

  attachGroupTrendsToLastMessage: (group_trends) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, group_trends }
      }
      return { messages }
    }),

  attachSplitStrategyToLastMessage: (split_strategy) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, split_strategy }
      }
      return { messages }
    }),

  attachFeatureSelectionToLastMessage: (feature_selection) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, feature_selection }
      }
      return { messages }
    }),

  attachModelImprovementToLastMessage: (model_improvement) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, model_improvement }
      }
      return { messages }
    }),

  attachModelSelectionToLastMessage: (model_selection) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, model_selection }
      }
      return { messages }
    }),
  attachAutoRetrainToLastMessage: (auto_retrain) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, auto_retrain }
      }
      return { messages }
    }),
  attachConversationExportToLastMessage: (conversation_export) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, conversation_export }
      }
      return { messages }
    }),
  attachHealthSummaryToLastMessage: (health_summary) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, health_summary }
      }
      return { messages }
    }),
  attachPredictionOpportunitiesToLastMessage: (prediction_opportunities) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, prediction_opportunities }
      }
      return { messages }
    }),
  attachDatasetComparisonToLastMessage: (dataset_comparison) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, dataset_comparison }
      }
      return { messages }
    }),
  attachInlinePredictionToLastMessage: (inline_prediction) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, inline_prediction }
      }
      return { messages }
    }),
  attachMultiPredictionToLastMessage: (multi_prediction) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, multi_prediction }
      }
      return { messages }
    }),
  attachGoalTrainingToLastMessage: (goal_training) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, goal_training }
      }
      return { messages }
    }),
  attachSensitivityToLastMessage: (sensitivity) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, sensitivity }
      }
      return { messages }
    }),
  attachInteractionToLastMessage: (interaction) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, interaction }
      }
      return { messages }
    }),
  attachRankedPredictionsToLastMessage: (ranked_predictions) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, ranked_predictions }
      }
      return { messages }
    }),
  attachPredictionCohortToLastMessage: (prediction_cohort) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, prediction_cohort }
      }
      return { messages }
    }),
  attachOnboardingGuideToLastMessage: (onboarding_guide) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, onboarding_guide }
      }
      return { messages }
    }),
  attachVersionHistoryToLastMessage: (version_history) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, version_history }
      }
      return { messages }
    }),
  attachLearningCurveToLastMessage: (learning_curve) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, learning_curve }
      }
      return { messages }
    }),
  attachTemplateSavedToLastMessage: (template) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, template_saved: template }
      }
      return { messages }
    }),
  attachTemplateListToLastMessage: (template_list) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, template_list }
      }
      return { messages }
    }),
  attachTemplateReplayToLastMessage: (template_replay) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, template_replay }
      }
      return { messages }
    }),
  attachPresetSavedToLastMessage: (preset) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, preset_saved: preset }
      }
      return { messages }
    }),
  attachPresetListToLastMessage: (preset_list) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, preset_list }
      }
      return { messages }
    }),
  attachSdkDownloadToLastMessage: (sdk_download) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, sdk_download }
      }
      return { messages }
    }),
  attachPortfolioToLastMessage: (portfolio) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, portfolio }
      }
      return { messages }
    }),
  attachRateLimitToLastMessage: (rate_limit) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, rate_limit }
      }
      return { messages }
    }),
  attachPartialDependenceToLastMessage: (partial_dependence) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, partial_dependence }
      }
      return { messages }
    }),
  attachCalibrationCheckToLastMessage: (calibration_check) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, calibration_check }
      }
      return { messages }
    }),
  attachSlaMetricsToLastMessage: (sla_metrics) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, sla_metrics }
      }
      return { messages }
    }),
  attachQuotaAlertConfigToLastMessage: (quota_alert_config) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, quota_alert_config }
      }
      return { messages }
    }),
  attachScheduleSetToLastMessage: (schedule_set) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, schedule_set }
      }
      return { messages }
    }),
  attachABTestResultToLastMessage: (ab_test_result) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, ab_test_result }
      }
      return { messages }
    }),
  attachWebhookHistoryToLastMessage: (webhook_history) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, webhook_history }
      }
      return { messages }
    }),
  attachClassImbalanceCheckToLastMessage: (class_imbalance_check) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, class_imbalance_check }
      }
      return { messages }
    }),
  attachWebhookHealthSummaryToLastMessage: (webhook_health_summary) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, webhook_health_summary }
      }
      return { messages }
    }),
  attachExecutiveBriefingToLastMessage: (executive_briefing) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, executive_briefing }
      }
      return { messages }
    }),
  attachServiceExportToLastMessage: (service_export) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, service_export }
      }
      return { messages }
    }),
  attachVersionComparisonToLastMessage: (version_comparison) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, version_comparison }
      }
      return { messages }
    }),
  attachEnsembleRecommendationToLastMessage: (ensemble_recommendation) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, ensemble_recommendation }
      }
      return { messages }
    }),
  attachTuneChatToLastMessage: (tune_chat) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, tune_chat }
      }
      return { messages }
    }),

  attachCvScoreDistributionToLastMessage: (cv_score_distribution) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, cv_score_distribution }
      }
      return { messages }
    }),

  attachPredictionAnalyticsChatToLastMessage: (prediction_analytics_chat) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, prediction_analytics_chat }
      }
      return { messages }
    }),

  attachConfusionMatrixChatToLastMessage: (confusion_matrix_chat) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, confusion_matrix_chat }
      }
      return { messages }
    }),

  attachLocalExplanationToLastMessage: (local_explanation) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, local_explanation }
      }
      return { messages }
    }),

  attachProdInputDistToLastMessage: (prod_input_dist) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, prod_input_dist }
      }
      return { messages }
    }),

  attachCovariateDriftAlertToLastMessage: (covariate_drift_alert) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, covariate_drift_alert }
      }
      return { messages }
    }),

  attachQuotaRunwayToLastMessage: (quota_runway) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, quota_runway }
      }
      return { messages }
    }),

  attachCostEstimateToLastMessage: (cost_estimate) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, cost_estimate }
      }
      return { messages }
    }),

  attachUsagePatternToLastMessage: (usage_pattern) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, usage_pattern }
      }
      return { messages }
    }),
  attachPredictionLogExportToLastMessage: (prediction_log_export) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, prediction_log_export }
      }
      return { messages }
    }),
  attachRecentPredictionsToLastMessage: (recent_predictions) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, recent_predictions }
      }
      return { messages }
    }),
  attachPredictionAuditToLastMessage: (prediction_audit) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, prediction_audit }
      }
      return { messages }
    }),
  attachConfidenceTrendToLastMessage: (confidence_trend) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, confidence_trend }
      }
      return { messages }
    }),
  attachFeedbackAccuracyReportToLastMessage: (feedback_accuracy_report) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, feedback_accuracy_report }
      }
      return { messages }
    }),
  attachFairnessCheckToLastMessage: (fairness_check) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, fairness_check }
      }
      return { messages }
    }),
  attachBatchJobResultsToLastMessage: (batch_job_results) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, batch_job_results }
      }
      return { messages }
    }),
  attachProdPredictionExplanationToLastMessage: (prod_prediction_explanation) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = { ...last, prod_prediction_explanation }
      }
      return { messages }
    }),
})
)
