/**
 * SSE event dispatch — extracted from app/project/[id]/page.tsx (#17).
 *
 * The chat stream emits ~185 event types. This was a 239-branch if/else chain
 * (54 of them duplicate dead code) that no test could exercise and where a
 * typo'd type silently rendered nothing. It is now a handler MAP, so the set of
 * handled types is introspectable (`SSE_EVENT_TYPES`) for a backend/frontend
 * contract test, and each handler is unit-testable in isolation.
 *
 * `createSSEHandlers(deps)` takes the component's store actions / state setters
 * and returns `{ [eventType]: (json) => void }`. Handler bodies are the
 * original branch bodies verbatim; guards became early returns.
 */
/* eslint-disable @typescript-eslint/no-explicit-any -- SSE frames are dynamic
   network JSON; payloads are narrowed at use via `as` casts. */
import type {
  ActiveFilter,
  AnomalyResult,
  CleaningSuggestion,
  ClusteringResult,
  ColumnProfile,
  ComputedColumnSuggestion,
  CrosstabResult,
  DataReadinessResult,
  DataStory,
  DeployedResult,
  FeatureSuggestionsChatResult,
  FeaturesAppliedResult,
  FilterSetResult,
  ForecastResult,
  GroupStatsResult,
  ModelCard,
  PredictionErrorResult,
  RefreshPrompt,
  RenameResult,
  ReportReady,
  SegmentComparisonResult,
  SegmentPerformanceResult,
  TargetCorrelationResult,
  TimeWindowComparison,
  TopNResult,
  TrainingStartedResult,
  WhatIfChatResult
} from "./types"

/** A dynamic SSE frame. `type` selects the handler; other keys are payloads. */
export interface SSEEvent {
  type: string
  [key: string]: any
}

export type SSEHandler = (json: SSEEvent) => void

// Store actions / setters are strongly typed at the call site; loose here so
// this module needn't import 180+ result types.
type AnyFn = (...args: any[]) => void

export interface SSEHandlerDeps {
  appendToLastMessage: AnyFn
  attachABTestResultToLastMessage: AnyFn
  attachAccuracyAlertConfigToLastMessage: AnyFn
  attachAggregateExplanationToLastMessage: AnyFn
  attachAlertRuleToLastMessage: AnyFn
  attachApiKeyResultToLastMessage: AnyFn
  attachAutoInsightToLastMessage: AnyFn
  attachAutoRetrainToLastMessage: AnyFn
  attachAutoRollbackConfigToLastMessage: AnyFn
  attachBatchJobHistoryToLastMessage: AnyFn
  attachBatchJobResultsToLastMessage: AnyFn
  attachCalibrationCheckToLastMessage: AnyFn
  attachCanaryStatusToLastMessage: AnyFn
  attachChartToLastMessage: AnyFn
  attachClassFeatureImportanceToLastMessage: AnyFn
  attachClassImbalanceCheckToLastMessage: AnyFn
  attachClustersToLastMessage: AnyFn
  attachCohortEvolutionToLastMessage: AnyFn
  attachColumnProfileToLastMessage: AnyFn
  attachColumnTypeSuggestionsToLastMessage: AnyFn
  attachComputeToLastMessage: AnyFn
  attachConfHeatmapToLastMessage: AnyFn
  attachConfidenceBandToLastMessage: AnyFn
  attachConfidenceDistributionToLastMessage: AnyFn
  attachConfidenceThresholdConfigToLastMessage: AnyFn
  attachConfidenceTrendToLastMessage: AnyFn
  attachConfusionMatrixChatToLastMessage: AnyFn
  attachConversationExportToLastMessage: AnyFn
  attachCorrelationToLastMessage: AnyFn
  attachCostEstimateToLastMessage: AnyFn
  attachCostSensitiveThresholdToLastMessage: AnyFn
  attachCounterfactualToLastMessage: AnyFn
  attachCovariateDriftAlertToLastMessage: AnyFn
  attachCrossDeployPredictionToLastMessage: AnyFn
  attachCrossModelFeaturesToLastMessage: AnyFn
  attachCrossProjectComparisonToLastMessage: AnyFn
  attachCrosstabToLastMessage: AnyFn
  attachCvScoreDistributionToLastMessage: AnyFn
  attachDashboardConfigToLastMessage: AnyFn
  attachDashboardMetadataToLastMessage: AnyFn
  attachDataExportToLastMessage: AnyFn
  attachDataQualityImpactToLastMessage: AnyFn
  attachDataReadinessToLastMessage: AnyFn
  attachDataStoryToLastMessage: AnyFn
  attachDatasetComparisonToLastMessage: AnyFn
  attachDegradationRetrainConfigToLastMessage: AnyFn
  attachDeployPredDistCompareToLastMessage: AnyFn
  attachDeployedToLastMessage: AnyFn
  attachDeploymentChangelogToLastMessage: AnyFn
  attachDeploymentHealthScorecardToLastMessage: AnyFn
  attachDeploymentScorecardToLastMessage: AnyFn
  attachDeploymentsOverviewToLastMessage: AnyFn
  attachDriftImportanceRankingToLastMessage: AnyFn
  attachEmbedCodeToLastMessage: AnyFn
  attachEnsembleRecommendationToLastMessage: AnyFn
  attachErrorCorrelationToLastMessage: AnyFn
  attachErrorDistributionToLastMessage: AnyFn
  attachExecutiveBriefingToLastMessage: AnyFn
  attachFairnessCheckToLastMessage: AnyFn
  attachFeEngineeringImpactToLastMessage: AnyFn
  attachFeatureDriftAlertConfigToLastMessage: AnyFn
  attachFeaturePsiToLastMessage: AnyFn
  attachFeatureRedundancyToLastMessage: AnyFn
  attachFeatureSelectionToLastMessage: AnyFn
  attachFeatureSuggestionsToLastMessage: AnyFn
  attachFeatureSweepToLastMessage: AnyFn
  attachFeaturesAppliedToLastMessage: AnyFn
  attachFeedbackAccuracyReportToLastMessage: AnyFn
  attachFilterToLastMessage: AnyFn
  attachForecastToLastMessage: AnyFn
  attachGoalSeekHistoryToLastMessage: AnyFn
  attachGoalSeekToLastMessage: AnyFn
  attachGoalTrainingToLastMessage: AnyFn
  attachGroupStatsToLastMessage: AnyFn
  attachGroupTrendsToLastMessage: AnyFn
  attachHealthSummaryToLastMessage: AnyFn
  attachHighActivityBurstConfigToLastMessage: AnyFn
  attachInlinePredictionToLastMessage: AnyFn
  attachInputDistDriftAlertConfigToLastMessage: AnyFn
  attachInputValidationRuleToLastMessage: AnyFn
  attachInteractionToLastMessage: AnyFn
  attachLatencyAlertConfigToLastMessage: AnyFn
  attachLearningCurveToLastMessage: AnyFn
  attachLocalExplanationToLastMessage: AnyFn
  attachLowAccuracyGuidanceToLastMessage: AnyFn
  attachLowActivityAlertConfigToLastMessage: AnyFn
  attachMilestoneToLastMessage: AnyFn
  attachMinFeatureSetToLastMessage: AnyFn
  attachModelCardExportToLastMessage: AnyFn
  attachModelCardToLastMessage: AnyFn
  attachModelComparisonSummaryToLastMessage: AnyFn
  attachModelImprovementToLastMessage: AnyFn
  attachModelQualityScoreToLastMessage: AnyFn
  attachModelSelectionToLastMessage: AnyFn
  attachModelStatusReportToLastMessage: AnyFn
  attachMonitoringDigestToLastMessage: AnyFn
  attachMonitoringNoteToLastMessage: AnyFn
  attachMultiPredictionToLastMessage: AnyFn
  attachNullMapToLastMessage: AnyFn
  attachOnboardingGuideToLastMessage: AnyFn
  attachOutcomeCalibrationToLastMessage: AnyFn
  attachOutputAnomaliesToLastMessage: AnyFn
  attachOutputDistributionShiftToLastMessage: AnyFn
  attachOverfittingAnalysisToLastMessage: AnyFn
  attachPairCorrelationToLastMessage: AnyFn
  attachPartialDependenceToLastMessage: AnyFn
  attachPerClassThresholdToLastMessage: AnyFn
  attachPerformanceDecayRateToLastMessage: AnyFn
  attachPopulationCounterfactualToLastMessage: AnyFn
  attachPortfolioToLastMessage: AnyFn
  attachPredValueAlertConfigToLastMessage: AnyFn
  attachPredictionAnalyticsChatToLastMessage: AnyFn
  attachPredictionAuditToLastMessage: AnyFn
  attachPredictionCohortToLastMessage: AnyFn
  attachPredictionDeltaToLastMessage: AnyFn
  attachPredictionErrorsToLastMessage: AnyFn
  attachPredictionLogExportToLastMessage: AnyFn
  attachPredictionOpportunitiesToLastMessage: AnyFn
  attachPredictionValueTrendToLastMessage: AnyFn
  attachPresetListToLastMessage: AnyFn
  attachPresetSavedToLastMessage: AnyFn
  attachProdInputDistToLastMessage: AnyFn
  attachProdPerformanceToLastMessage: AnyFn
  attachProdPredictionExplanationToLastMessage: AnyFn
  attachProductionThresholdOptimizerToLastMessage: AnyFn
  attachPromotionReadinessToLastMessage: AnyFn
  attachQuotaAlertConfigToLastMessage: AnyFn
  attachQuotaRunwayToLastMessage: AnyFn
  attachRankedPredictionsToLastMessage: AnyFn
  attachRateLimitToLastMessage: AnyFn
  attachRecentPredictionsToLastMessage: AnyFn
  attachRecordsToLastMessage: AnyFn
  attachRenameResultToLastMessage: AnyFn
  attachReportToLastMessage: AnyFn
  attachRetrainCompleteNotifyToLastMessage: AnyFn
  attachRetrainingReadinessToLastMessage: AnyFn
  attachRollbackChatToLastMessage: AnyFn
  attachSampleSizeAdequacyToLastMessage: AnyFn
  attachSavedScenariosToLastMessage: AnyFn
  attachScheduleSetToLastMessage: AnyFn
  attachSdkDownloadToLastMessage: AnyFn
  attachSegmentConfTrendToLastMessage: AnyFn
  attachSegmentDriftToLastMessage: AnyFn
  attachSegmentPerformanceToLastMessage: AnyFn
  attachSegmentPredTrendToLastMessage: AnyFn
  attachSegmentToLastMessage: AnyFn
  attachSensitivityToLastMessage: AnyFn
  attachServiceExportToLastMessage: AnyFn
  attachShareLinkToLastMessage: AnyFn
  attachSimilarRecordsToLastMessage: AnyFn
  attachSlaMetricsToLastMessage: AnyFn
  attachSplitStrategyToLastMessage: AnyFn
  attachStatQueryToLastMessage: AnyFn
  attachSummaryStatsToLastMessage: AnyFn
  attachTargetLeakageToLastMessage: AnyFn
  attachTemplateListToLastMessage: AnyFn
  attachTemplateReplayToLastMessage: AnyFn
  attachTemplateSavedToLastMessage: AnyFn
  attachThresholdAnalysisToLastMessage: AnyFn
  attachThroughputAssessmentToLastMessage: AnyFn
  attachTimeWindowToLastMessage: AnyFn
  attachTopNToLastMessage: AnyFn
  attachTrainingStartedToLastMessage: AnyFn
  attachTuneChatToLastMessage: AnyFn
  attachUptimeSummaryToLastMessage: AnyFn
  attachUsagePatternToLastMessage: AnyFn
  attachValueCountsToLastMessage: AnyFn
  attachVersionComparisonToLastMessage: AnyFn
  attachVersionHistoryToLastMessage: AnyFn
  attachWebhookHealthSummaryToLastMessage: AnyFn
  attachWebhookHistoryToLastMessage: AnyFn
  attachWebhookListChatToLastMessage: AnyFn
  attachWebhookRegisteredToLastMessage: AnyFn
  attachWebhookRemovedChatToLastMessage: AnyFn
  attachWebhookTestChatToLastMessage: AnyFn
  attachWeeklyDigestConfigToLastMessage: AnyFn
  attachWeeklyUsageReportToLastMessage: AnyFn
  attachWhatIfChatToLastMessage: AnyFn
  attachWhatNextToLastMessage: AnyFn
  setActiveFilter: AnyFn
  setActiveTab: AnyFn
  setAnomalyResult: AnyFn
  setChatSuggestions: AnyFn
  setCleaningSuggestion: AnyFn
  setComputeSuggestion: AnyFn
  setRefreshPrompt: AnyFn
  setStreaming: AnyFn
}

export function createSSEHandlers(deps: SSEHandlerDeps): Record<string, SSEHandler> {
  const {
    appendToLastMessage,
    attachABTestResultToLastMessage,
    attachAccuracyAlertConfigToLastMessage,
    attachAggregateExplanationToLastMessage,
    attachAlertRuleToLastMessage,
    attachApiKeyResultToLastMessage,
    attachAutoInsightToLastMessage,
    attachAutoRetrainToLastMessage,
    attachAutoRollbackConfigToLastMessage,
    attachBatchJobHistoryToLastMessage,
    attachBatchJobResultsToLastMessage,
    attachCalibrationCheckToLastMessage,
    attachCanaryStatusToLastMessage,
    attachChartToLastMessage,
    attachClassFeatureImportanceToLastMessage,
    attachClassImbalanceCheckToLastMessage,
    attachClustersToLastMessage,
    attachCohortEvolutionToLastMessage,
    attachColumnProfileToLastMessage,
    attachColumnTypeSuggestionsToLastMessage,
    attachComputeToLastMessage,
    attachConfHeatmapToLastMessage,
    attachConfidenceBandToLastMessage,
    attachConfidenceDistributionToLastMessage,
    attachConfidenceThresholdConfigToLastMessage,
    attachConfidenceTrendToLastMessage,
    attachConfusionMatrixChatToLastMessage,
    attachConversationExportToLastMessage,
    attachCorrelationToLastMessage,
    attachCostEstimateToLastMessage,
    attachCostSensitiveThresholdToLastMessage,
    attachCounterfactualToLastMessage,
    attachCovariateDriftAlertToLastMessage,
    attachCrossDeployPredictionToLastMessage,
    attachCrossModelFeaturesToLastMessage,
    attachCrossProjectComparisonToLastMessage,
    attachCrosstabToLastMessage,
    attachCvScoreDistributionToLastMessage,
    attachDashboardConfigToLastMessage,
    attachDashboardMetadataToLastMessage,
    attachDataExportToLastMessage,
    attachDataQualityImpactToLastMessage,
    attachDataReadinessToLastMessage,
    attachDataStoryToLastMessage,
    attachDatasetComparisonToLastMessage,
    attachDegradationRetrainConfigToLastMessage,
    attachDeployPredDistCompareToLastMessage,
    attachDeployedToLastMessage,
    attachDeploymentChangelogToLastMessage,
    attachDeploymentHealthScorecardToLastMessage,
    attachDeploymentScorecardToLastMessage,
    attachDeploymentsOverviewToLastMessage,
    attachDriftImportanceRankingToLastMessage,
    attachEmbedCodeToLastMessage,
    attachEnsembleRecommendationToLastMessage,
    attachErrorCorrelationToLastMessage,
    attachErrorDistributionToLastMessage,
    attachExecutiveBriefingToLastMessage,
    attachFairnessCheckToLastMessage,
    attachFeEngineeringImpactToLastMessage,
    attachFeatureDriftAlertConfigToLastMessage,
    attachFeaturePsiToLastMessage,
    attachFeatureRedundancyToLastMessage,
    attachFeatureSelectionToLastMessage,
    attachFeatureSuggestionsToLastMessage,
    attachFeatureSweepToLastMessage,
    attachFeaturesAppliedToLastMessage,
    attachFeedbackAccuracyReportToLastMessage,
    attachFilterToLastMessage,
    attachForecastToLastMessage,
    attachGoalSeekHistoryToLastMessage,
    attachGoalSeekToLastMessage,
    attachGoalTrainingToLastMessage,
    attachGroupStatsToLastMessage,
    attachGroupTrendsToLastMessage,
    attachHealthSummaryToLastMessage,
    attachHighActivityBurstConfigToLastMessage,
    attachInlinePredictionToLastMessage,
    attachInputDistDriftAlertConfigToLastMessage,
    attachInputValidationRuleToLastMessage,
    attachInteractionToLastMessage,
    attachLatencyAlertConfigToLastMessage,
    attachLearningCurveToLastMessage,
    attachLocalExplanationToLastMessage,
    attachLowAccuracyGuidanceToLastMessage,
    attachLowActivityAlertConfigToLastMessage,
    attachMilestoneToLastMessage,
    attachMinFeatureSetToLastMessage,
    attachModelCardExportToLastMessage,
    attachModelCardToLastMessage,
    attachModelComparisonSummaryToLastMessage,
    attachModelImprovementToLastMessage,
    attachModelQualityScoreToLastMessage,
    attachModelSelectionToLastMessage,
    attachModelStatusReportToLastMessage,
    attachMonitoringDigestToLastMessage,
    attachMonitoringNoteToLastMessage,
    attachMultiPredictionToLastMessage,
    attachNullMapToLastMessage,
    attachOnboardingGuideToLastMessage,
    attachOutcomeCalibrationToLastMessage,
    attachOutputAnomaliesToLastMessage,
    attachOutputDistributionShiftToLastMessage,
    attachOverfittingAnalysisToLastMessage,
    attachPairCorrelationToLastMessage,
    attachPartialDependenceToLastMessage,
    attachPerClassThresholdToLastMessage,
    attachPerformanceDecayRateToLastMessage,
    attachPopulationCounterfactualToLastMessage,
    attachPortfolioToLastMessage,
    attachPredValueAlertConfigToLastMessage,
    attachPredictionAnalyticsChatToLastMessage,
    attachPredictionAuditToLastMessage,
    attachPredictionCohortToLastMessage,
    attachPredictionDeltaToLastMessage,
    attachPredictionErrorsToLastMessage,
    attachPredictionLogExportToLastMessage,
    attachPredictionOpportunitiesToLastMessage,
    attachPredictionValueTrendToLastMessage,
    attachPresetListToLastMessage,
    attachPresetSavedToLastMessage,
    attachProdInputDistToLastMessage,
    attachProdPerformanceToLastMessage,
    attachProdPredictionExplanationToLastMessage,
    attachProductionThresholdOptimizerToLastMessage,
    attachPromotionReadinessToLastMessage,
    attachQuotaAlertConfigToLastMessage,
    attachQuotaRunwayToLastMessage,
    attachRankedPredictionsToLastMessage,
    attachRateLimitToLastMessage,
    attachRecentPredictionsToLastMessage,
    attachRecordsToLastMessage,
    attachRenameResultToLastMessage,
    attachReportToLastMessage,
    attachRetrainCompleteNotifyToLastMessage,
    attachRetrainingReadinessToLastMessage,
    attachRollbackChatToLastMessage,
    attachSampleSizeAdequacyToLastMessage,
    attachSavedScenariosToLastMessage,
    attachScheduleSetToLastMessage,
    attachSdkDownloadToLastMessage,
    attachSegmentConfTrendToLastMessage,
    attachSegmentDriftToLastMessage,
    attachSegmentPerformanceToLastMessage,
    attachSegmentPredTrendToLastMessage,
    attachSegmentToLastMessage,
    attachSensitivityToLastMessage,
    attachServiceExportToLastMessage,
    attachShareLinkToLastMessage,
    attachSimilarRecordsToLastMessage,
    attachSlaMetricsToLastMessage,
    attachSplitStrategyToLastMessage,
    attachStatQueryToLastMessage,
    attachSummaryStatsToLastMessage,
    attachTargetLeakageToLastMessage,
    attachTemplateListToLastMessage,
    attachTemplateReplayToLastMessage,
    attachTemplateSavedToLastMessage,
    attachThresholdAnalysisToLastMessage,
    attachThroughputAssessmentToLastMessage,
    attachTimeWindowToLastMessage,
    attachTopNToLastMessage,
    attachTrainingStartedToLastMessage,
    attachTuneChatToLastMessage,
    attachUptimeSummaryToLastMessage,
    attachUsagePatternToLastMessage,
    attachValueCountsToLastMessage,
    attachVersionComparisonToLastMessage,
    attachVersionHistoryToLastMessage,
    attachWebhookHealthSummaryToLastMessage,
    attachWebhookHistoryToLastMessage,
    attachWebhookListChatToLastMessage,
    attachWebhookRegisteredToLastMessage,
    attachWebhookRemovedChatToLastMessage,
    attachWebhookTestChatToLastMessage,
    attachWeeklyDigestConfigToLastMessage,
    attachWeeklyUsageReportToLastMessage,
    attachWhatIfChatToLastMessage,
    attachWhatNextToLastMessage,
    setActiveFilter,
    setActiveTab,
    setAnomalyResult,
    setChatSuggestions,
    setCleaningSuggestion,
    setComputeSuggestion,
    setRefreshPrompt,
    setStreaming
  } = deps

  return {
  token: (json) => {
      appendToLastMessage(json.content)
  },
  chart: (json) => {
      if (!(json.chart)) return
      attachChartToLastMessage(json.chart)
  },
  crosstab: (json) => {
      if (!(json.crosstab)) return
      attachCrosstabToLastMessage(json.crosstab as CrosstabResult)
  },
  suggestions: (json) => {
      if (!(Array.isArray(json.suggestions))) return
      setChatSuggestions(json.suggestions)
  },
  next_step: (json) => {
      if (!(Array.isArray(json.chips))) return
      setChatSuggestions(json.chips)
  },
  anomalies: (json) => {
      if (!(json.anomalies)) return
      setAnomalyResult(json.anomalies as AnomalyResult)
      setActiveTab("data")
  },
  cleaning_suggestion: (json) => {
      if (!(json.cleaning)) return
      setCleaningSuggestion(json.cleaning as CleaningSuggestion)
      setActiveTab("data")
  },
  refresh_prompt: (json) => {
      if (!(json.refresh)) return
      setRefreshPrompt(json.refresh as RefreshPrompt)
      setActiveTab("data")
  },
  compute_suggestion: (json) => {
      if (!(json.compute)) return
      setComputeSuggestion(json.compute as ComputedColumnSuggestion)
      attachComputeToLastMessage(json.compute as ComputedColumnSuggestion)
      setActiveTab("data")
  },
  segment_comparison: (json) => {
      if (!(json.segment_comparison)) return
      attachSegmentToLastMessage(json.segment_comparison as SegmentComparisonResult)
  },
  forecast: (json) => {
      if (!(json.forecast)) return
      attachForecastToLastMessage(json.forecast as ForecastResult)
  },
  data_readiness: (json) => {
      if (!(json.readiness)) return
      attachDataReadinessToLastMessage(json.readiness as DataReadinessResult)
  },
  target_correlation: (json) => {
      if (!(json.correlation)) return
      attachCorrelationToLastMessage(json.correlation as TargetCorrelationResult)
  },
  group_stats: (json) => {
      if (!(json.group_stats)) return
      attachGroupStatsToLastMessage(json.group_stats as GroupStatsResult)
  },
  rename_result: (json) => {
      if (!(json.rename)) return
      attachRenameResultToLastMessage(json.rename as RenameResult)
  },
  training_started: (json) => {
      if (!(json.training)) return
      attachTrainingStartedToLastMessage(json.training as TrainingStartedResult)
  },
  data_story: (json) => {
      if (!(json.story)) return
      attachDataStoryToLastMessage(json.story as DataStory)
  },
  filter_set: (json) => {
      if (!(json.filter_set)) return
      attachFilterToLastMessage(json.filter_set as FilterSetResult)
      setActiveFilter({
        dataset_id: json.filter_set.dataset_id,
        active: true,
        filter_summary: json.filter_set.filter_summary,
        conditions: json.filter_set.conditions,
        original_rows: json.filter_set.original_rows,
        filtered_rows: json.filter_set.filtered_rows,
        row_reduction_pct: json.filter_set.row_reduction_pct,
      } as ActiveFilter)
  },
  filter_cleared: () => {
      setActiveFilter(null)
  },
  deployed: (json) => {
      if (!(json.deployment)) return
      attachDeployedToLastMessage(json.deployment as DeployedResult)
  },
  model_card: (json) => {
      if (!(json.model_card)) return
      attachModelCardToLastMessage(json.model_card as ModelCard)
  },
  report_ready: (json) => {
      if (!(json.report)) return
      attachReportToLastMessage(json.report as ReportReady)
  },
  feature_suggestions: (json) => {
      if (!(json.suggestions)) return
      attachFeatureSuggestionsToLastMessage(json.suggestions as FeatureSuggestionsChatResult)
  },
  features_applied: (json) => {
      if (!(json.applied)) return
      attachFeaturesAppliedToLastMessage(json.applied as FeaturesAppliedResult)
  },
  segment_performance: (json) => {
      if (!(json.segment_performance)) return
      attachSegmentPerformanceToLastMessage(json.segment_performance as SegmentPerformanceResult)
  },
  column_profile: (json) => {
      if (!(json.column_profile)) return
      attachColumnProfileToLastMessage(json.column_profile as ColumnProfile)
  },
  clusters: (json) => {
      if (!(json.clusters)) return
      attachClustersToLastMessage(json.clusters as ClusteringResult)
  },
  time_window_comparison: (json) => {
      if (!(json.time_window)) return
      attachTimeWindowToLastMessage(json.time_window as TimeWindowComparison)
  },
  top_n: (json) => {
      if (!(json.top_n)) return
      attachTopNToLastMessage(json.top_n as TopNResult)
  },
  whatif_result: (json) => {
      if (!(json.whatif)) return
      attachWhatIfChatToLastMessage(json.whatif as WhatIfChatResult)
  },
  prediction_errors: (json) => {
      if (!(json.pred_errors)) return
      attachPredictionErrorsToLastMessage(json.pred_errors as PredictionErrorResult)
  },
  records: (json) => {
      if (!(json.records)) return
      attachRecordsToLastMessage(json.records as import("@/lib/types").RecordTableResult)
  },
  data_export: (json) => {
      if (!(json.data_export)) return
      attachDataExportToLastMessage(json.data_export as import("@/lib/types").DataExportResult)
  },
  null_map: (json) => {
      if (!(json.null_map)) return
      attachNullMapToLastMessage(json.null_map as import("@/lib/types").NullMapResult)
  },
  summary_stats: (json) => {
      if (!(json.summary_stats)) return
      attachSummaryStatsToLastMessage(json.summary_stats as import("@/lib/types").SummaryStatsResult)
  },
  value_counts: (json) => {
      if (!(json.value_counts)) return
      attachValueCountsToLastMessage(json.value_counts as import("@/lib/types").ValueCountResult)
  },
  pair_correlation: (json) => {
      if (!(json.pair_correlation)) return
      attachPairCorrelationToLastMessage(json.pair_correlation as import("@/lib/types").PairCorrelationResult)
  },
  stat_query: (json) => {
      if (!(json.stat_query)) return
      attachStatQueryToLastMessage(json.stat_query as import("@/lib/types").StatQueryResult)
  },
  group_trends: (json) => {
      if (!(json.group_trends)) return
      attachGroupTrendsToLastMessage(json.group_trends as import("@/lib/types").GroupTrendResult)
  },
  split_strategy: (json) => {
      if (!(json.split_strategy)) return
      attachSplitStrategyToLastMessage(json.split_strategy as import("@/lib/types").SplitStrategyResult)
  },
  feature_selection: (json) => {
      if (!(json.feature_selection)) return
      attachFeatureSelectionToLastMessage(json.feature_selection as import("@/lib/types").FeatureSelectionResult)
  },
  model_improvement: (json) => {
      if (!(json.model_improvement)) return
      attachModelImprovementToLastMessage(json.model_improvement as import("@/lib/types").ModelImprovementResult)
  },
  model_selection: (json) => {
      if (!(json.model_selection)) return
      attachModelSelectionToLastMessage(json.model_selection as import("@/lib/types").ModelSelectionResult)
  },
  model_quality_score: (json) => {
      if (!(json.model_quality_score)) return
      attachModelQualityScoreToLastMessage(json.model_quality_score as import("@/lib/types").ModelQualityScoreResult)
  },
  auto_retrain: (json) => {
      if (!(json.auto_retrain)) return
      attachAutoRetrainToLastMessage(json.auto_retrain as import("@/lib/types").AutoRetrainResult)
  },
  conversation_export: (json) => {
      if (!(json.conversation_export)) return
      attachConversationExportToLastMessage(json.conversation_export as import("@/lib/types").ConversationExportInfo)
  },
  health_summary: (json) => {
      if (!(json.health_summary)) return
      attachHealthSummaryToLastMessage(json.health_summary as import("@/lib/types").ProjectHealthSummary)
  },
  prediction_opportunities: (json) => {
      if (!(json.prediction_opportunities)) return
      attachPredictionOpportunitiesToLastMessage(json.prediction_opportunities as import("@/lib/types").PredictionOpportunitiesResult)
  },
  dataset_comparison: (json) => {
      if (!(json.dataset_comparison)) return
      attachDatasetComparisonToLastMessage(json.dataset_comparison as import("@/lib/types").DatasetComparisonResult)
  },
  inline_prediction: (json) => {
      if (!(json.inline_prediction)) return
      attachInlinePredictionToLastMessage(json.inline_prediction as import("@/lib/types").InlinePredictionResult)
  },
  multi_prediction: (json) => {
      if (!(json.multi_prediction)) return
      attachMultiPredictionToLastMessage(json.multi_prediction as import("@/lib/types").MultiPredictionResult)
  },
  goal_training: (json) => {
      if (!(json.goal_training)) return
      attachGoalTrainingToLastMessage(json.goal_training as import("@/lib/types").GoalTrainingResult)
  },
  sensitivity: (json) => {
      if (!(json.sensitivity)) return
      attachSensitivityToLastMessage(json.sensitivity as import("@/lib/types").SensitivityResult)
  },
  interaction: (json) => {
      if (!(json.interaction)) return
      attachInteractionToLastMessage(json.interaction as import("@/lib/types").InteractionResult)
  },
  ranked_predictions: (json) => {
      if (!(json.ranked_predictions)) return
      attachRankedPredictionsToLastMessage(json.ranked_predictions as import("@/lib/types").RankedPredictionsResult)
  },
  prediction_cohort: (json) => {
      if (!(json.prediction_cohort)) return
      attachPredictionCohortToLastMessage(json.prediction_cohort as import("@/lib/types").PredictionCohortResult)
  },
  cohort_evolution: (json) => {
      if (!(json.cohort_evolution)) return
      attachCohortEvolutionToLastMessage(json.cohort_evolution as import("@/lib/types").CohortEvolutionResult)
  },
  counterfactual: (json) => {
      if (!(json.counterfactual)) return
      attachCounterfactualToLastMessage(json.counterfactual as import("@/lib/types").CounterfactualResult)
  },
  population_counterfactual: (json) => {
      if (!(json.population_counterfactual)) return
      attachPopulationCounterfactualToLastMessage(json.population_counterfactual as import("@/lib/types").PopulationCounterfactualResult)
  },
  similar_records: (json) => {
      if (!(json.similar_records)) return
      attachSimilarRecordsToLastMessage(json.similar_records as import("@/lib/types").SimilarRecordsResult)
  },
  fe_impact: (json) => {
      if (!(json.fe_impact)) return
      attachFeEngineeringImpactToLastMessage(json.fe_impact as import("@/lib/types").FeatureEngineeringImpactResult)
  },
  data_quality_impact: (json) => {
      if (!(json.data_quality_impact)) return
      attachDataQualityImpactToLastMessage(json.data_quality_impact as import("@/lib/types").DataQualityImpactResult)
  },
  overfitting_analysis: (json) => {
      if (!(json.overfitting_analysis)) return
      attachOverfittingAnalysisToLastMessage(json.overfitting_analysis as import("@/lib/types").OverfittingAnalysisResult)
  },
  feature_redundancy: (json) => {
      if (!(json.feature_redundancy)) return
      attachFeatureRedundancyToLastMessage(json.feature_redundancy as import("@/lib/types").FeatureRedundancyResult)
  },
  target_leakage: (json) => {
      if (!(json.target_leakage)) return
      attachTargetLeakageToLastMessage(json.target_leakage as import("@/lib/types").TargetLeakageResult)
  },
  threshold_analysis: (json) => {
      if (!(json.threshold_analysis)) return

  },
  per_class_threshold: (json) => {
      if (!(json.per_class_threshold)) return
      attachPerClassThresholdToLastMessage(json.per_class_threshold as import("@/lib/types").PerClassThresholdResult)
      attachThresholdAnalysisToLastMessage(json.threshold_analysis as import("@/lib/types").ThresholdAnalysisResult)
  },
  confidence_distribution: (json) => {
      if (!(json.confidence_distribution)) return
      attachConfidenceDistributionToLastMessage(json.confidence_distribution as import("@/lib/types").ConfidenceDistributionResult)
  },
  sample_size_adequacy: (json) => {
      if (!(json.sample_size_adequacy)) return
      attachSampleSizeAdequacyToLastMessage(json.sample_size_adequacy as import("@/lib/types").SampleSizeAdequacyResult)
  },
  class_feature_importance: (json) => {
      if (!(json.class_feature_importance)) return
      attachClassFeatureImportanceToLastMessage(json.class_feature_importance as import("@/lib/types").ClassFeatureImportanceResult)
  },
  error_correlation: (json) => {
      if (!(json.error_correlation)) return
      attachErrorCorrelationToLastMessage(json.error_correlation as import("@/lib/types").ErrorCorrelationResult)
  },
  output_anomalies: (json) => {
      if (!(json.output_anomalies)) return
      attachOutputAnomaliesToLastMessage(json.output_anomalies as import("@/lib/types").PredictionOutputAnomalyResult)
  },
  output_distribution_shift: (json) => {
      if (!(json.output_distribution_shift)) return
      attachOutputDistributionShiftToLastMessage(json.output_distribution_shift as import("@/lib/types").PredictionOutputDistributionShiftResult)
  },
  feature_psi: (json) => {
      if (!(json.feature_psi)) return
      attachFeaturePsiToLastMessage(json.feature_psi as import("@/lib/types").FeaturePsiResult)
  },
  min_feature_set: (json) => {
      if (!(json.min_feature_set)) return
      attachMinFeatureSetToLastMessage(json.min_feature_set as import("@/lib/types").MinFeatureSetResult)
  },
  retraining_readiness: (json) => {
      if (!(json.retraining_readiness)) return
      attachRetrainingReadinessToLastMessage(json.retraining_readiness as import("@/lib/types").RetrainingReadinessResult)
  },
  prediction_value_trend: (json) => {
      if (!(json.prediction_value_trend)) return
      attachPredictionValueTrendToLastMessage(json.prediction_value_trend as import("@/lib/types").PredictionValueTrendResult)
  },
  monitoring_digest: (json) => {
      if (!(json.monitoring_digest)) return
      attachMonitoringDigestToLastMessage(json.monitoring_digest as import("@/lib/types").MonitoringDigestResult)
  },
  model_status_report: (json) => {
      if (!(json.model_status_report)) return
      attachModelStatusReportToLastMessage(json.model_status_report as import("@/lib/types").ModelStatusReportInfo)
  },
  production_threshold_optimizer: (json) => {
      if (!(json.production_threshold_optimizer)) return
      attachProductionThresholdOptimizerToLastMessage(json.production_threshold_optimizer as import("@/lib/types").ProductionThresholdOptimizerResult)
  },
  deploy_pred_dist_compare: (json) => {
      if (!(json.deploy_pred_dist_compare)) return
      attachDeployPredDistCompareToLastMessage(json.deploy_pred_dist_compare as import("@/lib/types").DeploymentPredictionComparisonResult)
  },
  weekly_digest_config: (json) => {
      if (!(json.weekly_digest_config)) return
      attachWeeklyDigestConfigToLastMessage(json.weekly_digest_config as import("@/lib/types").WeeklyDigestConfigResult)
  },
  promotion_readiness: (json) => {
      if (!(json.promotion_readiness)) return
      attachPromotionReadinessToLastMessage(json.promotion_readiness as import("@/lib/types").PromotionReadinessResult)
  },
  deployment_scorecard: (json) => {
      if (!(json.deployment_scorecard)) return
      attachDeploymentScorecardToLastMessage(json.deployment_scorecard as import("@/lib/types").DeploymentScorecardResult)
  },
  throughput_assessment: (json) => {
      if (!(json.throughput_assessment)) return
      attachThroughputAssessmentToLastMessage(json.throughput_assessment as import("@/lib/types").DeploymentThroughputResult)
  },
  drift_importance_ranking: (json) => {
      if (!(json.drift_importance_ranking)) return
      attachDriftImportanceRankingToLastMessage(json.drift_importance_ranking as import("@/lib/types").DriftImportanceRankingResult)
  },
  feature_drift_alert_config: (json) => {
      if (!(json.feature_drift_alert_config)) return
      attachFeatureDriftAlertConfigToLastMessage(json.feature_drift_alert_config as import("@/lib/types").FeatureDriftAlertConfig)
  },
  input_dist_drift_alert_config: (json) => {
      if (!(json.input_dist_drift_alert_config)) return
      attachInputDistDriftAlertConfigToLastMessage(json.input_dist_drift_alert_config as import("@/lib/types").InputDistDriftAlertConfig)
  },
  low_activity_alert_config: (json) => {
      if (!(json.low_activity_alert_config)) return
      attachLowActivityAlertConfigToLastMessage(json.low_activity_alert_config as import("@/lib/types").LowActivityAlertConfig)
  },
  high_activity_burst_config: (json) => {
      if (!(json.high_activity_burst_config)) return
      attachHighActivityBurstConfigToLastMessage(json.high_activity_burst_config as import("@/lib/types").HighActivityBurstConfig)
  },
  latency_alert_config: (json) => {
      if (!(json.latency_alert_config)) return
      attachLatencyAlertConfigToLastMessage(json.latency_alert_config as import("@/lib/types").LatencyAlertConfig)
  },
  auto_rollback_config: (json) => {
      if (!(json.auto_rollback_config)) return
      attachAutoRollbackConfigToLastMessage(json.auto_rollback_config as import("@/lib/types").AutoRollbackConfig)
  },
  pred_value_alert_config: (json) => {
      if (!(json.pred_value_alert_config)) return
      attachPredValueAlertConfigToLastMessage(json.pred_value_alert_config as import("@/lib/types").PredValueAlertConfig)
  },
  degradation_retrain_config: (json) => {
      if (!(json.degradation_retrain_config)) return
      attachDegradationRetrainConfigToLastMessage(json.degradation_retrain_config as import("@/lib/types").DegradationRetrainConfig)
  },
  segment_drift: (json) => {
      if (!(json.segment_drift)) return
      attachSegmentDriftToLastMessage(json.segment_drift as import("@/lib/types").SegmentDriftResult)
  },
  segment_pred_trend: (json) => {
      if (!(json.segment_pred_trend)) return
      attachSegmentPredTrendToLastMessage(json.segment_pred_trend as import("@/lib/types").SegmentPredTrendResult)
  },
  segment_conf_trend: (json) => {
      if (!(json.segment_conf_trend)) return
      attachSegmentConfTrendToLastMessage(json.segment_conf_trend as import("@/lib/types").SegmentConfTrendResult)
  },
  conf_heatmap: (json) => {
      if (!(json.conf_heatmap)) return
      attachConfHeatmapToLastMessage(json.conf_heatmap as import("@/lib/types").ConfidenceHeatmapResult)
  },
  feature_sweep: (json) => {
      if (!(json.feature_sweep)) return
      attachFeatureSweepToLastMessage(json.feature_sweep as import("@/lib/types").FeatureSweepResult)
  },
  saved_scenarios: (json) => {
      if (!(json.saved_scenarios)) return
      attachSavedScenariosToLastMessage(json.saved_scenarios as import("@/lib/types").SavedScenariosResult)
  },
  canary_status: (json) => {
      if (!(json.canary_status)) return
      attachCanaryStatusToLastMessage(json.canary_status as import("@/lib/types").CanaryStatusResult)
  },
  deployment_health_scorecard: (json) => {
      if (!(json.deployment_health_scorecard)) return
      attachDeploymentHealthScorecardToLastMessage(json.deployment_health_scorecard as import("@/lib/types").DeploymentHealthScorecardResult)
  },
  confidence_band: (json) => {
      if (!(json.confidence_band)) return
      attachConfidenceBandToLastMessage(json.confidence_band as import("@/lib/types").ConfidenceBandResult)
  },
  retrain_complete_notify: (json) => {
      if (!(json.retrain_complete_notify)) return
      attachRetrainCompleteNotifyToLastMessage(json.retrain_complete_notify as import("@/lib/types").RetrainCompleteNotifyResult)
  },
  outcome_calibration: (json) => {
      if (!(json.outcome_calibration)) return
      attachOutcomeCalibrationToLastMessage(json.outcome_calibration as import("@/lib/types").OutcomeCalibrationResult)
  },
  batch_job_history: (json) => {
      if (!(json.batch_job_history)) return
      attachBatchJobHistoryToLastMessage(json.batch_job_history as import("@/lib/types").BatchJobHistoryResult)
  },
  performance_decay_rate: (json) => {
      if (!(json.performance_decay_rate)) return
      attachPerformanceDecayRateToLastMessage(json.performance_decay_rate as import("@/lib/types").PerformanceDecayResult)
  },
  uptime_summary: (json) => {
      if (!(json.uptime_summary)) return
      attachUptimeSummaryToLastMessage(json.uptime_summary as import("@/lib/types").ApiUptimeSummaryResult)
  },
  cost_sensitive_threshold: (json) => {
      if (!(json.cost_sensitive_threshold)) return
      attachCostSensitiveThresholdToLastMessage(json.cost_sensitive_threshold as import("@/lib/types").CostSensitiveThresholdResult)
  },
  onboarding_guide: (json) => {
      if (!(json.onboarding_guide)) return
      attachOnboardingGuideToLastMessage(json.onboarding_guide as import("@/lib/types").OnboardingGuideResult)
  },
  version_history: (json) => {
      if (!(json.version_history)) return
      attachVersionHistoryToLastMessage(json.version_history as import("@/lib/types").DataVersionHistoryResult)
  },
  learning_curve: (json) => {
      if (!(json.learning_curve)) return
      attachLearningCurveToLastMessage(json.learning_curve as import("@/lib/types").LearningCurveResult)
  },
  template_saved: (json) => {
      if (!(json.template)) return
      attachTemplateSavedToLastMessage(json.template as import("@/lib/types").TemplateSavedInfo)
  },
  template_list: (json) => {
      if (!(json.template_list)) return
      attachTemplateListToLastMessage(json.template_list as import("@/lib/types").TemplateListInfo)
  },
  template_replay: (json) => {
      if (!(json.template_replay)) return
      attachTemplateReplayToLastMessage(json.template_replay as import("@/lib/types").TemplateReplayInfo)
  },
  preset_saved: (json) => {
      if (!(json.preset)) return
      attachPresetSavedToLastMessage(json.preset as import("@/lib/types").PresetSavedInfo)
  },
  preset_list: (json) => {
      if (!(json.preset_list)) return
      attachPresetListToLastMessage(json.preset_list as import("@/lib/types").PresetListInfo)
  },
  sdk_download: (json) => {
      if (!(json.sdk_download)) return
      attachSdkDownloadToLastMessage(json.sdk_download as import("@/lib/types").SdkDownloadInfo)
  },
  portfolio: (json) => {
      if (!(json.portfolio)) return
      attachPortfolioToLastMessage(json.portfolio as import("@/lib/types").PortfolioResult)
  },
  rate_limit: (json) => {
      if (!(json.rate_limit)) return
      attachRateLimitToLastMessage(json.rate_limit as import("@/lib/types").RateLimitInfo)
  },
  partial_dependence: (json) => {
      if (!(json.partial_dependence)) return
      attachPartialDependenceToLastMessage(json.partial_dependence as import("@/lib/types").PartialDependenceResult)
  },
  calibration_check: (json) => {
      if (!(json.calibration_check)) return
      attachCalibrationCheckToLastMessage(json.calibration_check as import("@/lib/types").CalibrationCheckResult)
  },
  sla_metrics: (json) => {
      if (!(json.sla_metrics)) return
      attachSlaMetricsToLastMessage(json.sla_metrics as import("@/lib/types").SlaData)
  },
  quota_alert_config: (json) => {
      if (!(json.quota_alert_config)) return
      attachQuotaAlertConfigToLastMessage(json.quota_alert_config as import("@/lib/types").QuotaAlertConfig)
  },
  schedule_set: (json) => {
      if (!(json.schedule_set)) return
      attachScheduleSetToLastMessage(json.schedule_set as import("@/lib/types").ScheduleSetResult)
  },
  ab_test_result: (json) => {
      if (!(json.ab_test_result)) return
      attachABTestResultToLastMessage(json.ab_test_result as import("@/lib/types").ABTestChatResult)
  },
  webhook_history: (json) => {
      if (!(json.webhook_history)) return
      attachWebhookHistoryToLastMessage(json.webhook_history as import("@/lib/types").WebhookHistoryResult)
  },
  class_imbalance_check: (json) => {
      if (!(json.class_imbalance_check)) return
      attachClassImbalanceCheckToLastMessage(json.class_imbalance_check as import("@/lib/types").ClassImbalanceResult)
  },
  webhook_health_summary: (json) => {
      if (!(json.webhook_health_summary)) return
      attachWebhookHealthSummaryToLastMessage(json.webhook_health_summary as import("@/lib/types").WebhookHealthSummaryResult)
  },
  executive_briefing: (json) => {
      if (!(json.executive_briefing)) return
      attachExecutiveBriefingToLastMessage(json.executive_briefing as import("@/lib/types").ExecutiveBriefingResult)
  },
  service_export: (json) => {
      if (!(json.service_export)) return
      attachServiceExportToLastMessage(json.service_export as import("@/lib/types").ServiceExportChatResult)
  },
  version_comparison: (json) => {
      if (!(json.version_comparison)) return
      attachVersionComparisonToLastMessage(json.version_comparison as import("@/lib/types").DeploymentVersionComparisonResult)
  },
  ensemble_recommendation: (json) => {
      if (!(json.ensemble_recommendation)) return
      attachEnsembleRecommendationToLastMessage(json.ensemble_recommendation as import("@/lib/types").EnsembleRecommendationResult)
  },
  tune_chat: (json) => {
      if (!(json.tune_chat)) return
      attachTuneChatToLastMessage(json.tune_chat as import("@/lib/types").TuningChatResult)
  },
  cv_score_distribution: (json) => {
      if (!(json.cv_score_distribution)) return
      attachCvScoreDistributionToLastMessage(json.cv_score_distribution as import("@/lib/types").CvScoreDistributionResult)
  },
  prediction_analytics_chat: (json) => {
      if (!(json.prediction_analytics_chat)) return
      attachPredictionAnalyticsChatToLastMessage(json.prediction_analytics_chat as import("@/lib/types").PredictionAnalyticsChatResult)
  },
  confusion_matrix_chat: (json) => {
      if (!(json.confusion_matrix_chat)) return
      attachConfusionMatrixChatToLastMessage(json.confusion_matrix_chat as import("@/lib/types").ConfusionMatrixChatResult)
  },
  local_explanation: (json) => {
      if (!(json.local_explanation)) return
      attachLocalExplanationToLastMessage(json.local_explanation as import("@/lib/types").LocalExplanationResult)
  },
  prod_input_dist: (json) => {
      if (!(json.prod_input_dist)) return
      attachProdInputDistToLastMessage(json.prod_input_dist as import("@/lib/types").ProductionInputDistributionResult)
  },
  covariate_drift_alert: (json) => {
      if (!(json.covariate_drift_alert)) return
      attachCovariateDriftAlertToLastMessage(json.covariate_drift_alert as import("@/lib/types").CovariateDriftAlertResult)
  },
  quota_runway: (json) => {
      if (!(json.quota_runway)) return
      attachQuotaRunwayToLastMessage(json.quota_runway as import("@/lib/types").QuotaRunwayResult)
  },
  cost_estimate: (json) => {
      if (!(json.cost_estimate)) return
      attachCostEstimateToLastMessage(json.cost_estimate as import("@/lib/types").CostEstimateResult)
  },
  usage_pattern: (json) => {
      if (!(json.usage_pattern)) return
      attachUsagePatternToLastMessage(json.usage_pattern as import("@/lib/types").UsagePatternResult)
  },
  prediction_log_export: (json) => {
      if (!(json.prediction_log_export)) return
      attachPredictionLogExportToLastMessage(json.prediction_log_export as import("@/lib/types").PredictionLogExportResult)
  },
  recent_predictions: (json) => {
      if (!(json.recent_predictions)) return
      attachRecentPredictionsToLastMessage(json.recent_predictions as import("@/lib/types").RecentPredictionsResult)
  },
  prediction_audit: (json) => {
      if (!(json.prediction_audit)) return
      attachPredictionAuditToLastMessage(json.prediction_audit as import("@/lib/types").PredictionAuditResult)
  },
  confidence_trend: (json) => {
      if (!(json.confidence_trend)) return
      attachConfidenceTrendToLastMessage(json.confidence_trend as import("@/lib/types").ConfidenceTrendResult)
  },
  feedback_accuracy_report: (json) => {
      if (!(json.feedback_accuracy_report)) return
      attachFeedbackAccuracyReportToLastMessage(json.feedback_accuracy_report as import("@/lib/types").FeedbackAccuracyReportResult)
  },
  fairness_check: (json) => {
      if (!(json.fairness_check)) return
      attachFairnessCheckToLastMessage(json.fairness_check as import("@/lib/types").FairnessCheckResult)
  },
  batch_job_results: (json) => {
      if (!(json.batch_job_results)) return
      attachBatchJobResultsToLastMessage(json.batch_job_results as import("@/lib/types").BatchJobResultsResult)
  },
  prod_prediction_explanation: (json) => {
      if (!(json.prod_prediction_explanation)) return
      attachProdPredictionExplanationToLastMessage(json.prod_prediction_explanation as import("@/lib/types").ProdPredictionExplanationResult)
  },
  aggregate_explanation: (json) => {
      if (!(json.aggregate_explanation)) return
      attachAggregateExplanationToLastMessage(json.aggregate_explanation as import("@/lib/types").AggregateExplanationResult)
  },
  webhook_registered: (json) => {
      if (!(json.webhook_registered)) return
      attachWebhookRegisteredToLastMessage(json.webhook_registered as import("@/lib/types").WebhookRegisteredInfo)
  },
  webhook_list_chat: (json) => {
      if (!(json.webhook_list_chat)) return
      attachWebhookListChatToLastMessage(json.webhook_list_chat as import("@/lib/types").WebhookListChatResult)
  },
  webhook_removed_chat: (json) => {
      if (!(json.webhook_removed_chat)) return
      attachWebhookRemovedChatToLastMessage(json.webhook_removed_chat as import("@/lib/types").WebhookRemovedChatInfo)
  },
  webhook_test_chat: (json) => {
      if (!(json.webhook_test_chat)) return
      attachWebhookTestChatToLastMessage(json.webhook_test_chat as import("@/lib/types").WebhookTestChatResult)
  },
  alert_rule: (json) => {
      if (!(json.alert_rule)) return
      attachAlertRuleToLastMessage(json.alert_rule as import("@/lib/types").AlertRuleEventResult)
  },
  api_key_result: (json) => {
      if (!(json.api_key_result)) return
      attachApiKeyResultToLastMessage(json.api_key_result as import("@/lib/types").ApiKeyResultInfo)
  },
  deployments_overview: (json) => {
      if (!(json.deployments_overview)) return
      attachDeploymentsOverviewToLastMessage(json.deployments_overview as import("@/lib/types").DeploymentsOverviewResult)
  },
  prod_performance: (json) => {
      if (!(json.prod_performance)) return
      attachProdPerformanceToLastMessage(json.prod_performance as import("@/lib/types").ProdPerformanceResult)
  },
  error_distribution: (json) => {
      if (!(json.error_distribution)) return
      attachErrorDistributionToLastMessage(json.error_distribution as import("@/lib/types").ErrorDistributionResult)
  },
  model_card_export: (json) => {
      if (!(json.model_card_export)) return
      attachModelCardExportToLastMessage(json.model_card_export as import("@/lib/types").ModelCardExportInfo)
  },
  model_comparison_summary: (json) => {
      if (!(json.model_comparison_summary)) return
      attachModelComparisonSummaryToLastMessage(json.model_comparison_summary as import("@/lib/types").ModelComparisonSummaryResult)
  },
  cross_model_features: (json) => {
      if (!(json.cross_model_features)) return
      attachCrossModelFeaturesToLastMessage(json.cross_model_features as import("@/lib/types").CrossModelFeatureResult)
  },
  accuracy_alert_config: (json) => {
      if (!(json.accuracy_alert_config)) return
      attachAccuracyAlertConfigToLastMessage(json.accuracy_alert_config as import("@/lib/types").AccuracyAlertConfig)
  },
  rollback_chat: (json) => {
      if (!(json.rollback_chat)) return
      attachRollbackChatToLastMessage(json.rollback_chat as import("@/lib/types").RollbackChatResult)
  },
  confidence_threshold_config: (json) => {
      if (!(json.confidence_threshold_config)) return
      attachConfidenceThresholdConfigToLastMessage(json.confidence_threshold_config as import("@/lib/types").ConfidenceThresholdConfig)
  },
  input_validation_rule: (json) => {
      if (!(json.input_validation_rule)) return
      attachInputValidationRuleToLastMessage(json.input_validation_rule as import("@/lib/types").InputValidationRuleResult)
  },
  dashboard_config: (json) => {
      if (!(json.dashboard_config)) return
      attachDashboardConfigToLastMessage(json.dashboard_config as import("@/lib/types").DashboardConfigResult)
  },
  dashboard_metadata: (json) => {
      if (!(json.dashboard_metadata)) return
      attachDashboardMetadataToLastMessage(json.dashboard_metadata as import("@/lib/types").DashboardMetadataResult)
  },
  embed_code: (json) => {
      if (!(json.embed_code)) return
      attachEmbedCodeToLastMessage(json.embed_code as import("@/lib/types").EmbedCodeResult)
  },
  share_link: (json) => {
      if (!(json.share_link)) return
      attachShareLinkToLastMessage(json.share_link as import("@/lib/types").ShareLinkResult)
  },
  weekly_usage_report: (json) => {
      if (!(json.weekly_usage_report)) return
      attachWeeklyUsageReportToLastMessage(json.weekly_usage_report as import("@/lib/types").WeeklyUsageReportResult)
  },
  cross_project_comparison: (json) => {
      if (!(json.cross_project_comparison)) return
      attachCrossProjectComparisonToLastMessage(json.cross_project_comparison as import("@/lib/types").CrossProjectComparisonResult)
  },
  what_next: (json) => {
      if (!(json.what_next)) return
      attachWhatNextToLastMessage(json.what_next as import("@/lib/types").WhatNextResult)
  },
  milestone: (json) => {
      if (!(json.milestone)) return
      attachMilestoneToLastMessage(json.milestone as import("@/lib/types").MilestoneResult)
  },
  auto_insight: (json) => {
      if (!(json.auto_insight)) return
      attachAutoInsightToLastMessage(json.auto_insight as import("@/lib/types").AutoInsightResult)
  },
  column_type_suggestions: (json) => {
      if (!(json.column_type_suggestions)) return
      attachColumnTypeSuggestionsToLastMessage(json.column_type_suggestions as import("@/lib/types").ColumnTypeSuggestionResult)
  },
  goal_seek: (json) => {
      if (!(json.goal_seek)) return
      attachGoalSeekToLastMessage(json.goal_seek as import("@/lib/types").GoalSeekResult)
  },
  goal_seek_history: (json) => {
      if (!(json.goal_seek_history)) return
      attachGoalSeekHistoryToLastMessage(json.goal_seek_history as import("@/lib/types").GoalSeekHistoryResult)
  },
  deployment_changelog: (json) => {
      if (!(json.deployment_changelog)) return
      attachDeploymentChangelogToLastMessage(json.deployment_changelog as import("@/lib/types").DeploymentChangelogResult)
  },
  cross_deploy_prediction: (json) => {
      if (!(json.cross_deploy_prediction)) return
      attachCrossDeployPredictionToLastMessage(json.cross_deploy_prediction as import("@/lib/types").CrossDeployPredictionResult)
  },
  low_accuracy_guidance: (json) => {
      if (!(json.low_accuracy_guidance)) return
      attachLowAccuracyGuidanceToLastMessage(json.low_accuracy_guidance as import("@/lib/types").LowAccuracyGuidanceResult)
  },
  prediction_delta: (json) => {
      if (!(json.prediction_delta)) return
      attachPredictionDeltaToLastMessage(json.prediction_delta as import("@/lib/types").PredictionDeltaResult)
  },
  done: () => {
      setStreaming(false)
  },

  // ---- Previously-dropped events (#17). Backend computes + LLM narrates these;
  // each had no frontend handler. They map to one generic monitoring card.
  readiness: (json) => {
      if (!json.readiness) return
      const r = json.readiness
      attachMonitoringNoteToLastMessage({
        kind: "readiness",
        title: "Model readiness",
        summary:
          (r.verdict ? `${r.verdict} — ` : "") + `Readiness score ${r.score}/100`,
        items: (r.checks ?? []).map(
          (c: { passed?: boolean; label?: string }) =>
            `${c.passed ? "✓" : "✗"} ${c.label}`
        ),
        tone: r.score >= 80 ? "good" : r.score >= 50 ? "warning" : "critical",
      })
  },
  drift: (json) => {
      if (!json.drift) return
      const d = json.drift
      attachMonitoringNoteToLastMessage({
        kind: "drift",
        title: "Prediction drift",
        summary: d.explanation ?? "Drift check complete.",
        items: [
          `Status: ${d.status}`,
          ...(d.drift_score != null ? [`Drift score: ${d.drift_score}`] : []),
        ],
        tone:
          d.status === "drift_detected"
            ? "critical"
            : d.status === "insufficient_data"
              ? "neutral"
              : "good",
      })
  },
  health: (json) => {
      if (!json.health) return
      const h = json.health
      attachMonitoringNoteToLastMessage({
        kind: "health",
        title: "Model health",
        summary:
          h.feedback_note ?? `Health score ${h.health_score} (${h.status}).`,
        items: [
          `Health score: ${h.health_score}`,
          `Status: ${h.status}`,
          ...(h.model_age_days != null ? [`Model age: ${h.model_age_days} day(s)`] : []),
          ...(h.algorithm ? [`Algorithm: ${h.algorithm}`] : []),
        ],
        tone: h.status === "healthy" ? "good" : h.status === "degraded" ? "warning" : "neutral",
      })
  },
  alerts: (json) => {
      if (!json.alerts) return
      const a = json.alerts
      attachMonitoringNoteToLastMessage({
        kind: "alerts",
        title: "Deployment alerts",
        summary:
          a.alert_count > 0
            ? `${a.alert_count} alert(s): ${a.critical_count} critical, ${a.warning_count} warning.`
            : "No active alerts — all deployments look healthy.",
        items: (a.alerts ?? []).map(
          (al: { message?: string }) => al.message ?? ""
        ),
        tone:
          a.critical_count > 0 ? "critical" : a.warning_count > 0 ? "warning" : "good",
      })
  },
  // `history` is a trigger, not data ({project_id}); the backend narration says
  // the Version History card is now visible in the Models tab, so just switch to it.
  history: (json) => {
      if (!json.history) return
      setActiveTab("models")
  },
  // Graceful chat-stream failure (#18): the backend yields this instead of
  // silently truncating the stream. Reuses the MonitoringNoteCard (critical tone).
  error: (json) => {
      if (!json.message) return
      attachMonitoringNoteToLastMessage({
        kind: "error",
        title: "Something went wrong",
        summary: json.message,
        tone: "critical",
      })
  },
  }
}

/** Every event type this frontend handles — the contract surface (#17). */
export const SSE_EVENT_TYPES: readonly string[] = Object.keys(
  createSSEHandlers(new Proxy({}, { get: () => () => {} }) as SSEHandlerDeps)
)
