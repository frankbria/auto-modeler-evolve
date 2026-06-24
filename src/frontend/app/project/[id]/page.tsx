"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { useParams, useRouter } from "next/navigation"
import { useDropzone } from "react-dropzone"
import { Button } from "@/components/ui/button"
import { RequireAuth } from "@/components/auth/require-auth"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { ChartMessage } from "@/components/chat/chart-message"
import { ModelTrainingPanel } from "@/components/models/model-training-panel"
import {
  FeatureSuggestionsPanel,
  FeatureImportancePanel,
  DatasetListPanel,
} from "@/components/features/feature-suggestions"
import { ValidationPanel } from "@/components/validation/validation-panel"
import { DeploymentPanel } from "@/components/deploy/deployment-panel"
import { AnomalyCard } from "@/components/data/anomaly-card"
import { CleaningCard } from "@/components/data/cleaning-card"
import { RefreshCard } from "@/components/data/refresh-card"
import { DictionaryCard } from "@/components/data/dictionary-card"
import { CrosstabTable } from "@/components/data/crosstab-table"
import { ComputeCard } from "@/components/data/compute-card"
import { SegmentComparisonCard } from "@/components/data/segment-comparison-card"
import { ForecastChart } from "@/components/data/forecast-chart"
import { ReadinessCheckCard } from "@/components/data/readiness-check-card"
import { CorrelationBarCard } from "@/components/data/correlation-bar-card"
import { GroupStatsCard } from "@/components/data/group-stats-card"
import { RenameResultCard } from "@/components/data/rename-result-card"
import { TrainingStartedCard } from "@/components/models/training-started-card"
import { DataStoryCard } from "@/components/data/data-story-card"
import { FilterBadge } from "@/components/data/filter-badge"
import { FilterSetCard } from "@/components/chat/filter-set-card"
import { DeployedCard } from "@/components/deploy/deployed-card"
import { ModelCardView } from "@/components/models/model-card-view"
import { ReportReadyCard } from "@/components/models/report-ready-card"
import { SegmentPerformanceCard } from "@/components/models/segment-performance-card"
import { PredictionErrorCard } from "@/components/models/prediction-error-card"
import { ColumnProfileCard } from "@/components/data/column-profile-card"
import { ClusteringCard } from "@/components/data/clustering-card"
import { TimeWindowCard } from "@/components/data/time-window-card"
import { TopNCard } from "@/components/data/top-n-card"
import { RecordTableCard } from "@/components/data/record-table-card"
import { DataExportCard } from "@/components/data/data-export-card"
import { NullMapCard } from "@/components/data/null-map-card"
import { GroupTrendCard } from "@/components/data/group-trend-card"
import { SplitStrategyCard } from "@/components/models/split-strategy-card"
import { FeatureSelectionCard } from "@/components/models/feature-selection-card"
import { ModelImprovementCard } from "@/components/models/model-improvement-card"
import { ModelSelectionCard } from "@/components/models/model-selection-card"
import { ModelQualityScoreCard } from "@/components/models/model-quality-score-card"
import { AutoRetrainCard } from "@/components/models/auto-retrain-card"
import { ConversationExportCard } from "@/components/chat/conversation-export-card"
import { ModelCardExportCard } from "@/components/chat/model-card-export-card"
import { ModelComparisonSummaryCard } from "@/components/chat/model-comparison-summary-card"
import { CrossModelFeaturesCard } from "@/components/chat/cross-model-features-card"
import { AccuracyAlertCard } from "@/components/deploy/accuracy-alert-card"
import { RollbackChatCard } from "@/components/deploy/rollback-chat-card"
import { ConfidenceThresholdCard } from "@/components/deploy/confidence-threshold-card"
import { InputValidationRuleCard } from "@/components/deploy/input-validation-rule-card"
import { ProjectHealthCard } from "@/components/chat/project-health-card"
import { PredictionOpportunitiesCard } from "@/components/models/prediction-opportunities-card"
import { DatasetComparisonCard } from "@/components/data/dataset-comparison-card"
import { InlinePredictionCard } from "@/components/models/inline-prediction-card"
import { MultiPredictionCard } from "@/components/deploy/multi-prediction-card"
import { GoalTrainingCard } from "@/components/models/goal-training-card"
import { SensitivityCard } from "@/components/deploy/sensitivity-card"
import { InteractionCard } from "@/components/deploy/interaction-card"
import { RankedPredictionsCard } from "@/components/deploy/ranked-predictions-card"
import { PredictionCohortCard } from "@/components/deploy/prediction-cohort-card"
import { CohortEvolutionCard } from "@/components/deploy/cohort-evolution-card"
import { CounterfactualCard } from "@/components/deploy/counterfactual-card"
import { PopulationCounterfactualCard } from "@/components/deploy/population-counterfactual-card"
import { SimilarRecordsCard } from "@/components/deploy/similar-records-card"
import { FeatureEngineeringImpactCard } from "@/components/models/feature-engineering-impact-card"
import { DataQualityImpactCard } from "@/components/models/data-quality-impact-card"
import { OverfittingAnalysisCard } from "@/components/models/overfitting-analysis-card"
import { FeatureRedundancyCard } from "@/components/models/feature-redundancy-card"
import { TargetLeakageCard } from "@/components/models/target-leakage-card"
import { ThresholdAnalysisCard } from "@/components/models/threshold-analysis-card"
import { PerClassThresholdCard } from "@/components/models/per-class-threshold-card"
import { ConfidenceDistributionCard } from "@/components/models/confidence-distribution-card"
import { SampleSizeAdequacyCard } from "@/components/models/sample-size-adequacy-card"
import { ClassFeatureImportanceCard } from "@/components/models/class-feature-importance-card"
import { ErrorCorrelationCard } from "@/components/chat/error-correlation-card"
import { PredictionOutputAnomalyCard } from "@/components/deploy/prediction-output-anomaly-card"
import { PredictionOutputDistributionCard } from "@/components/deploy/prediction-output-distribution-card"
import { FeaturePsiCard } from "@/components/deploy/feature-psi-card"
import { MinFeatureSetCard } from "@/components/models/min-feature-set-card"
import { RetrainingReadinessCard } from "@/components/deploy/retraining-readiness-card"
import { PredictionValueTrendCard } from "@/components/deploy/prediction-value-trend-card"
import { MonitoringDigestCard } from "@/components/deploy/monitoring-digest-card"
import { ModelStatusReportCard } from "@/components/deploy/model-status-report-card"
import { ProductionThresholdOptimizerCard } from "@/components/deploy/production-threshold-optimizer-card"
import { OnboardingGuideCard } from "@/components/chat/onboarding-guide-card"
import { DataVersionHistoryCard } from "@/components/chat/data-version-history-card"
import { LearningCurveCard } from "@/components/chat/learning-curve-card"
import {
  TemplateSavedCard,
  TemplateListCard,
  TemplateReplayCard,
} from "@/components/data/analysis-template-card"
import { PresetSavedCard } from "@/components/deploy/preset-saved-card"
import { PresetListCard } from "@/components/deploy/preset-list-card"
import { SdkDownloadCard } from "@/components/deploy/sdk-download-card"
import { PortfolioCard } from "@/components/chat/portfolio-card"
import { RateLimitCard } from "@/components/deploy/rate-limit-card"
import { PartialDependenceCard } from "@/components/validation/partial-dependence-card"
import CalibrationCheckCard from "@/components/models/calibration-check-card"
import { SlaCard } from "@/components/deploy/sla-chat-card"
import { QuotaAlertCard } from "@/components/deploy/quota-alert-card"
import { ScheduleSetChatCard } from "@/components/deploy/schedule-set-chat-card"
import { ABTestChatCard } from "@/components/deploy/ab-test-chat-card"
import { WebhookHistoryCard } from "@/components/deploy/webhook-history-card"
import { ClassImbalanceChatCard } from "@/components/models/class-imbalance-chat-card"
import { WebhookHealthSummaryCard } from "@/components/deploy/webhook-health-summary-card"
import { ExecutiveBriefingCard } from "@/components/deploy/executive-briefing-card"
import { ServiceExportChatCard } from "@/components/deploy/service-export-chat-card"
import { DeploymentVersionComparisonCard } from "@/components/deploy/deployment-version-comparison-card"
import { DeploymentPredictionDistributionCard } from "@/components/deploy/deployment-prediction-distribution-card"
import WeeklyDigestConfigCard from "@/components/deploy/weekly-digest-config-card"
import { PromotionReadinessCard } from "@/components/models/promotion-readiness-card"
import { DeploymentScorecardCard } from "@/components/deploy/deployment-scorecard-card"
import { DeploymentThroughputCard } from "@/components/deploy/deployment-throughput-card"
import { DriftImportanceCard } from "@/components/deploy/drift-importance-card"
import { FeatureDriftAlertCard } from "@/components/deploy/feature-drift-alert-card"
import { InputDistDriftAlertCard } from "@/components/deploy/input-dist-drift-alert-card"
import { LowActivityAlertCard } from "@/components/deploy/low-activity-alert-card"
import { HighActivityBurstCard } from "@/components/deploy/high-activity-burst-card"
import { LatencyAlertCard } from "@/components/deploy/latency-alert-card"
import { AutoRollbackCard } from "@/components/deploy/auto-rollback-card"
import { PredValueAlertCard } from "@/components/deploy/pred-value-alert-card"
import { DegradationRetrainCard } from "@/components/deploy/degradation-retrain-card"
import { SegmentDriftCard } from "@/components/deploy/segment-drift-card"
import { SegmentPredictionTrendCard } from "@/components/deploy/segment-prediction-trend-card"
import { SegmentConfidenceTrendCard } from "@/components/deploy/segment-confidence-trend-card"
import { ConfidenceHeatmapCard } from "@/components/deploy/confidence-heatmap-card"
import { ApiUptimeSummaryCard } from "@/components/deploy/api-uptime-summary-card"
import { CostSensitiveThresholdCard } from "@/components/models/cost-sensitive-threshold-card"
import { FeatureSweepCard } from "@/components/deploy/feature-sweep-card"
import { SavedScenariosCard } from "@/components/deploy/saved-scenarios-card"
import { CanaryCard } from "@/components/deploy/canary-card"
import { DeploymentHealthScorecardCard } from "@/components/deploy/deployment-health-scorecard-card"
import { ConfidenceBandCard } from "@/components/deploy/confidence-band-card"
import { RetrainCompleteNotifyCard } from "@/components/deploy/retrain-complete-notify-card"
import OutcomeCalibrationCard from "@/components/deploy/outcome-calibration-card"
import { BatchJobHistoryCard } from "@/components/deploy/batch-job-history-card"
import { PerformanceDecayRateCard } from "@/components/deploy/performance-decay-rate-card"
import { EnsembleRecommendationCard } from "@/components/models/ensemble-recommendation-card"
import { TuningChatCard } from "@/components/models/tuning-chat-card"
import { CvScoreDistributionCard } from "@/components/models/cv-score-distribution-card"
import { PredictionAnalyticsChatCard } from "@/components/chat/prediction-analytics-chat-card"
import { ConfusionMatrixChatCard } from "@/components/models/confusion-matrix-chat-card"
import { LocalExplanationCard } from "@/components/models/local-explanation-card"
import { ProductionInputDistributionCard } from "@/components/chat/production-input-distribution-card"
import { CovariateDriftAlertCard } from "@/components/deploy/covariate-drift-alert-card"
import { QuotaRunwayCard } from "@/components/deploy/quota-runway-card"
import { CostEstimateCard } from "@/components/deploy/cost-estimate-card"
import { UsagePatternCard } from "@/components/deploy/usage-pattern-card"
import { PredictionLogExportCard } from "@/components/deploy/prediction-log-export-card"
import { RecentPredictionsCard } from "@/components/deploy/recent-predictions-card"
import { PredictionAuditCard } from "@/components/deploy/prediction-audit-card"
import { ConfidenceTrendCard } from "@/components/deploy/confidence-trend-card"
import { FeedbackAccuracyCard } from "@/components/deploy/feedback-accuracy-card"
import { DashboardConfigCard } from "@/components/deploy/dashboard-config-card"
import { DashboardMetadataCard } from "@/components/deploy/dashboard-metadata-card"
import { EmbedCodeCard } from "@/components/deploy/embed-code-card"
import { ShareLinkCard } from "@/components/deploy/share-link-card"
import { WeeklyUsageReportCard } from "@/components/deploy/weekly-usage-report-card"
import { CrossProjectComparisonCard } from "@/components/chat/cross-project-comparison-card"
import { WhatNextCard } from "@/components/chat/what-next-card"
import { MilestoneCard } from "@/components/chat/milestone-card"
import { AutoInsightCard } from "@/components/chat/auto-insight-card"
import { MonitoringNoteCard } from "@/components/chat/monitoring-note-card"
import { ColumnTypeSuggestionCard } from "@/components/chat/column-type-suggestion-card"
import { GoalSeekCard } from "@/components/deploy/goal-seek-card"
import { GoalSeekHistoryCard } from "@/components/deploy/goal-seek-history-card"
import { DeploymentChangelogCard } from "@/components/deploy/deployment-changelog-card"
import { CrossDeployPredictionCard } from "@/components/deploy/cross-deploy-prediction-card"
import { LowAccuracyGuidanceCard } from "@/components/models/low-accuracy-guidance-card"
import { PredictionDeltaCard } from "@/components/deploy/prediction-delta-card"
import { FairnessCheckCard } from "@/components/chat/fairness-check-card"
import { BatchJobResultCard } from "@/components/chat/batch-job-result-card"
import { ProductionExplanationCard } from "@/components/chat/production-explanation-card"
import { AggregateExplanationCard } from "@/components/chat/aggregate-explanation-card"
import { WebhookRegisteredCard } from "@/components/chat/webhook-registered-card"
import { WebhookListChatCard } from "@/components/chat/webhook-list-chat-card"
import { WebhookRemovedChatCard } from "@/components/chat/webhook-removed-chat-card"
import { WebhookTestChatCard } from "@/components/chat/webhook-test-chat-card"
import { AlertRuleCard } from "@/components/chat/alert-rule-card"
import { ApiKeyChatCard } from "@/components/chat/api-key-chat-card"
import { DeploymentsOverviewCard } from "@/components/chat/deployments-overview-card"
import { ProdPerformanceCard } from "@/components/chat/prod-performance-card"
import { ErrorDistributionCard } from "@/components/chat/error-distribution-card"
import { PairCorrelationCard } from "@/components/data/pair-correlation-card"
import { StatQueryCard } from "@/components/data/stat-query-card"
import { SummaryStatsCard } from "@/components/data/summary-stats-card"
import { ValueCountCard } from "@/components/data/value-count-card"
import { WhatIfChatCard } from "@/components/deploy/whatif-chat-card"
import {
  FeatureSuggestCard,
  FeaturesAppliedCard,
} from "@/components/features/feature-suggestions-chat-card"
import { WorkflowProgress } from "@/components/ui/workflow-progress"
import { api, downloadFile, ApiError } from "@/lib/api"
import { ErrorDisplay } from "@/components/ui/error-display"
import { createSSEHandlers } from "@/lib/sse-handlers"
import { useAppStore } from "@/lib/store"
import type {
  Dataset,
  DataInsight,
  FeatureSuggestion,
  FeatureImportanceEntry,
  FeatureSetResult,
  ChatMessage as ChatMsg,
  AnomalyResult,
  CleaningSuggestion,
  CleanResult,
  RefreshPrompt,
  DatasetRefreshResult,
  ComputedColumnSuggestion,
  ComputeResult,
} from "@/lib/types"

const WELCOME_MESSAGE =
  "Hi! I'm your data modeling assistant. Upload a CSV or Excel file to get started, or ask me anything about your data."

function buildWelcomeBackMessage(projectName: string, messages: ChatMsg[]): string {
  const msgCount = messages.length
  // Find the last assistant message to summarise what was happening
  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant")
  const snippet = lastAssistant?.content?.slice(0, 120).replace(/\n/g, " ") ?? ""
  const lastActive = messages[messages.length - 1]?.timestamp
  const sinceMin = lastActive
    ? Math.round((Date.now() - new Date(lastActive).getTime()) / 60_000)
    : 0
  const sinceStr =
    sinceMin < 60
      ? `${sinceMin} minute${sinceMin !== 1 ? "s" : ""} ago`
      : `${Math.round(sinceMin / 60)} hour${Math.round(sinceMin / 60) !== 1 ? "s" : ""} ago`

  const context = snippet ? `Last we were: "${snippet}..."` : ""
  return (
    `Welcome back to **${projectName}**! You have ${msgCount} messages in this session (last active ${sinceStr}). ` +
    (context ? `${context} ` : "") +
    `What would you like to work on?`
  )
}

type RightTab = "data" | "features" | "importance" | "models" | "validate" | "deploy"

export default function ProjectWorkspace() {
  return (
    <RequireAuth>
      <ProjectWorkspaceInner />
    </RequireAuth>
  )
}

function ProjectWorkspaceInner() {
  const params = useParams<{ id: string }>()
  const router = useRouter()
  const projectId = params.id

  const {
    currentProject,
    setCurrentProject,
    currentDataset,
    dataPreview,
    columnStats,
    dataInsights,
    setDataset,
    messages,
    addMessage,
    setMessages,
    isStreaming,
    setStreaming,
    appendToLastMessage,
    attachChartToLastMessage,
    attachCrosstabToLastMessage,
    attachComputeToLastMessage,
    attachSegmentToLastMessage,
    attachForecastToLastMessage,
    attachDataReadinessToLastMessage,
    attachCorrelationToLastMessage,
    attachGroupStatsToLastMessage,
    attachRenameResultToLastMessage,
    attachTrainingStartedToLastMessage,
    attachDataStoryToLastMessage,
    attachFilterToLastMessage,
    setActiveFilter,
    activeFilter,
    attachDeployedToLastMessage,
    attachModelCardToLastMessage,
    attachReportToLastMessage,
    attachFeatureSuggestionsToLastMessage,
    attachFeaturesAppliedToLastMessage,
    attachSegmentPerformanceToLastMessage,
    attachColumnProfileToLastMessage,
    attachClustersToLastMessage,
    attachTimeWindowToLastMessage,
    attachTopNToLastMessage,
    attachWhatIfChatToLastMessage,
    attachPredictionErrorsToLastMessage,
    attachRecordsToLastMessage,
    attachDataExportToLastMessage,
    attachNullMapToLastMessage,
    attachSummaryStatsToLastMessage,
    attachValueCountsToLastMessage,
    attachPairCorrelationToLastMessage,
    attachStatQueryToLastMessage,
    attachGroupTrendsToLastMessage,
    attachSplitStrategyToLastMessage,
    attachFeatureSelectionToLastMessage,
    attachModelImprovementToLastMessage,
    attachModelSelectionToLastMessage,
    attachModelQualityScoreToLastMessage,
    attachAutoRetrainToLastMessage,
    attachConversationExportToLastMessage,
    attachHealthSummaryToLastMessage,
    attachPredictionOpportunitiesToLastMessage,
    attachDatasetComparisonToLastMessage,
    attachInlinePredictionToLastMessage,
    attachMultiPredictionToLastMessage,
    attachGoalTrainingToLastMessage,
    attachSensitivityToLastMessage,
    attachInteractionToLastMessage,
    attachOnboardingGuideToLastMessage,
    attachVersionHistoryToLastMessage,
    attachLearningCurveToLastMessage,
    attachTemplateSavedToLastMessage,
    attachTemplateListToLastMessage,
    attachTemplateReplayToLastMessage,
    attachPresetSavedToLastMessage,
    attachPresetListToLastMessage,
    attachRankedPredictionsToLastMessage,
    attachPredictionCohortToLastMessage,
    attachSdkDownloadToLastMessage,
    attachPortfolioToLastMessage,
    attachRateLimitToLastMessage,
    attachPartialDependenceToLastMessage,
    attachCalibrationCheckToLastMessage,
    attachSlaMetricsToLastMessage,
    attachQuotaAlertConfigToLastMessage,
    attachScheduleSetToLastMessage,
    attachABTestResultToLastMessage,
    attachWebhookHistoryToLastMessage,
    attachClassImbalanceCheckToLastMessage,
    attachWebhookHealthSummaryToLastMessage,
    attachExecutiveBriefingToLastMessage,
    attachServiceExportToLastMessage,
    attachVersionComparisonToLastMessage,
    attachEnsembleRecommendationToLastMessage,
    attachTuneChatToLastMessage,
    attachCvScoreDistributionToLastMessage,
    attachPredictionAnalyticsChatToLastMessage,
    attachConfusionMatrixChatToLastMessage,
    attachLocalExplanationToLastMessage,
    attachProdInputDistToLastMessage,
    attachCovariateDriftAlertToLastMessage,
    attachQuotaRunwayToLastMessage,
    attachCostEstimateToLastMessage,
    attachUsagePatternToLastMessage,
    attachPredictionLogExportToLastMessage,
    attachRecentPredictionsToLastMessage,
    attachPredictionAuditToLastMessage,
    attachConfidenceTrendToLastMessage,
    attachFeedbackAccuracyReportToLastMessage,
    attachFairnessCheckToLastMessage,
    attachBatchJobResultsToLastMessage,
    attachProdPredictionExplanationToLastMessage,
    attachAggregateExplanationToLastMessage,
    attachWebhookRegisteredToLastMessage,
    attachWebhookListChatToLastMessage,
    attachWebhookRemovedChatToLastMessage,
    attachWebhookTestChatToLastMessage,
    attachAlertRuleToLastMessage,
    attachApiKeyResultToLastMessage,
    attachDeploymentsOverviewToLastMessage,
    attachProdPerformanceToLastMessage,
    attachErrorDistributionToLastMessage,
    attachModelCardExportToLastMessage,
    attachModelComparisonSummaryToLastMessage,
    attachCrossModelFeaturesToLastMessage,
    attachAccuracyAlertConfigToLastMessage,
    attachRollbackChatToLastMessage,
    attachConfidenceThresholdConfigToLastMessage,
    attachInputValidationRuleToLastMessage,
    attachDashboardConfigToLastMessage,
    attachDashboardMetadataToLastMessage,
    attachEmbedCodeToLastMessage,
    attachShareLinkToLastMessage,
    attachWeeklyUsageReportToLastMessage,
    attachCrossProjectComparisonToLastMessage,
    attachWhatNextToLastMessage,
    attachMilestoneToLastMessage,
    attachAutoInsightToLastMessage,
    attachMonitoringNoteToLastMessage,
    attachColumnTypeSuggestionsToLastMessage,
    attachGoalSeekToLastMessage,
    attachGoalSeekHistoryToLastMessage,
    attachDeploymentChangelogToLastMessage,
    attachCrossDeployPredictionToLastMessage,
    attachLowAccuracyGuidanceToLastMessage,
    attachPredictionDeltaToLastMessage,
    attachCohortEvolutionToLastMessage,
    attachCounterfactualToLastMessage,
    attachPopulationCounterfactualToLastMessage,
    attachSimilarRecordsToLastMessage,
    attachFeEngineeringImpactToLastMessage,
    attachDataQualityImpactToLastMessage,
    attachOverfittingAnalysisToLastMessage,
    attachFeatureRedundancyToLastMessage,
    attachTargetLeakageToLastMessage,
    attachThresholdAnalysisToLastMessage,
    attachPerClassThresholdToLastMessage,
    attachConfidenceDistributionToLastMessage,
    attachSampleSizeAdequacyToLastMessage,
    attachClassFeatureImportanceToLastMessage,
    attachErrorCorrelationToLastMessage,
    attachOutputAnomaliesToLastMessage,
    attachOutputDistributionShiftToLastMessage,
    attachFeaturePsiToLastMessage,
    attachMinFeatureSetToLastMessage,
    attachRetrainingReadinessToLastMessage,
    attachPredictionValueTrendToLastMessage,
    attachMonitoringDigestToLastMessage,
    attachModelStatusReportToLastMessage,
    attachProductionThresholdOptimizerToLastMessage,
    attachDeployPredDistCompareToLastMessage,
    attachWeeklyDigestConfigToLastMessage,
    attachPromotionReadinessToLastMessage,
    attachDeploymentScorecardToLastMessage,
    attachThroughputAssessmentToLastMessage,
    attachDriftImportanceRankingToLastMessage,
    attachFeatureDriftAlertConfigToLastMessage,
    attachInputDistDriftAlertConfigToLastMessage,
    attachLowActivityAlertConfigToLastMessage,
    attachHighActivityBurstConfigToLastMessage,
    attachLatencyAlertConfigToLastMessage,
    attachAutoRollbackConfigToLastMessage,
    attachPredValueAlertConfigToLastMessage,
    attachDegradationRetrainConfigToLastMessage,
    attachSegmentDriftToLastMessage,
    attachSegmentPredTrendToLastMessage,
    attachSegmentConfTrendToLastMessage,
    attachConfHeatmapToLastMessage,
    attachUptimeSummaryToLastMessage,
    attachCostSensitiveThresholdToLastMessage,
    attachFeatureSweepToLastMessage,
    attachSavedScenariosToLastMessage,
    attachCanaryStatusToLastMessage,
    attachDeploymentHealthScorecardToLastMessage,
    attachConfidenceBandToLastMessage,
    attachRetrainCompleteNotifyToLastMessage,
    attachOutcomeCalibrationToLastMessage,
    attachBatchJobHistoryToLastMessage,
    attachPerformanceDecayRateToLastMessage,
  } = useAppStore()

  const [chatInput, setChatInput] = useState("")
  const [uploading, setUploading] = useState(false)
  const [loadingProject, setLoadingProject] = useState(true)
  // Critical project-load failure (#17): show a retry instead of a fake welcome.
  const [projectLoadError, setProjectLoadError] = useState<string | null>(null)
  const [reloadKey, setReloadKey] = useState(0)
  const [activeTab, setActiveTab] = useState<RightTab>("data")
  const [rightPanelVisible, setRightPanelVisible] = useState(true)
  // Mobile: which panel is active ("chat" | "panel")
  const [mobileView, setMobileView] = useState<"chat" | "panel">("chat")

  // Feature engineering state
  const [featureSuggestions, setFeatureSuggestions] = useState<FeatureSuggestion[]>([])
  const [loadingFeatures, setLoadingFeatures] = useState(false)
  const [targetColumn, setTargetColumn] = useState("")
  const [importanceFeatures, setImportanceFeatures] = useState<FeatureImportanceEntry[]>([])
  const [importanceProblemType, setImportanceProblemType] = useState("")
  const [loadingImportance, setLoadingImportance] = useState(false)

  // Validation state
  const [selectedModelRunId, setSelectedModelRunId] = useState<string | null>(null)
  const [selectedModelAlgorithm, setSelectedModelAlgorithm] = useState<string | null>(null)
  const [hasValidation, setHasValidation] = useState(false)
  const [hasDeployment, setHasDeployment] = useState(false)

  // Chat follow-up suggestion chips
  const [chatSuggestions, setChatSuggestions] = useState<string[]>([])

  // Anomaly detection result (populated via SSE or manual trigger)
  const [anomalyResult, setAnomalyResult] = useState<AnomalyResult | null>(null)

  // Cleaning suggestion (populated via SSE when user asks about cleaning in chat)
  const [cleaningSuggestion, setCleaningSuggestion] = useState<CleaningSuggestion | null>(null)

  // Refresh prompt (populated via SSE when user mentions having new data)
  const [refreshPrompt, setRefreshPrompt] = useState<RefreshPrompt | null>(null)

  // Computed column suggestion (populated via SSE when user asks to add a derived column)
  const [computeSuggestion, setComputeSuggestion] = useState<ComputedColumnSuggestion | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    async function load() {
      setProjectLoadError(null)
      try {
        const [project, history] = await Promise.all([
          api.projects.get(projectId),
          api.chat.history(projectId),
        ])
        setCurrentProject(project)
        if (project.has_deployment) setHasDeployment(true)

        // Restore dataset state when navigating back to an existing project
        if (project.dataset_id) {
          try {
            const data = await api.data.preview(project.dataset_id)
            const dataset: Dataset = {
              id: data.dataset_id,
              project_id: projectId,
              filename: data.filename,
              row_count: data.row_count,
              column_count: data.column_count,
              uploaded_at: project.updated_at,
            }
            setDataset(dataset, data.preview, data.column_stats, data.insights ?? [])
            setRightPanelVisible(true)
          } catch {
            // Dataset file missing — show upload panel to re-upload
          }

          // Restore selected model run ID so the Deploy tab works without re-selecting
          try {
            const runsData = await api.models.runs(projectId)
            const selected = runsData.runs.find((r) => r.is_selected)
            if (selected) {
              setSelectedModelRunId(selected.id)
              setSelectedModelAlgorithm(selected.algorithm)
            }
          } catch {
            // No runs yet or feature set missing — ignore
          }
        }

        if (history?.messages && history.messages.length > 0) {
          const msgs: ChatMsg[] = history.messages
          // Add a "welcome back" context message if this is a returning visit
          // (history has real conversation, not just the initial greeting)
          const hasConversation = msgs.some((m) => m.role === "user")
          if (hasConversation) {
            // Build the welcome-back message, then check model health proactively
            const welcomeBack: ChatMsg = {
              role: "assistant",
              content: buildWelcomeBackMessage(project.name, msgs),
              timestamp: new Date().toISOString(),
            }
            // Proactively surface model health alerts on returning visits
            let healthSummary: import("@/lib/types").ProjectHealthSummary | undefined
            let covariateDriftAlert: import("@/lib/types").CovariateDriftAlertResult | undefined
            if (project.has_deployment) {
              try {
                const hs = await api.projects.healthSummary(projectId)
                if (hs.alerts && hs.alerts.length > 0) {
                  healthSummary = hs
                }
              } catch {
                // Non-critical — never block the welcome message
              }
              // Proactively surface covariate drift if inputs are drifting significantly
              try {
                const deployments = await api.deploy.list()
                const projectDeployment = deployments.find(
                  (d: { project_id: string }) => d.project_id === projectId
                )
                if (projectDeployment) {
                  const drift = await api.deploy.covariateDrift(projectDeployment.id)
                  if (drift.has_alerts && drift.severity !== "low") {
                    covariateDriftAlert = drift
                  }
                }
              } catch {
                // Non-critical — never block the welcome message
              }
            }
            setMessages([
              ...msgs,
              {
                ...welcomeBack,
                health_summary: healthSummary,
                covariate_drift_alert: covariateDriftAlert,
              },
            ])
          } else {
            setMessages(msgs)
          }
        } else {
          setMessages([
            {
              role: "assistant",
              content: WELCOME_MESSAGE,
              timestamp: new Date().toISOString(),
            },
          ])
        }
      } catch (e) {
        // Couldn't load the project itself (404/500/network). Surface a retry
        // instead of a fake welcome message (#17). `history` is always 200, so
        // reaching here means a genuine failure to load this project.
        setProjectLoadError(
          e instanceof ApiError && e.status === 404
            ? "This project couldn't be found, or you don't have access to it."
            : "Couldn't load this project. Check your connection and try again."
        )
      } finally {
        setLoadingProject(false)
      }
    }
    load()
  }, [projectId, reloadKey, setCurrentProject, setDataset, setMessages])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Load feature suggestions when switching to features tab
  useEffect(() => {
    if (activeTab === "features" && currentDataset && featureSuggestions.length === 0) {
      setLoadingFeatures(true)
      api.features
        .suggestions(currentDataset.id)
        .then((r) => setFeatureSuggestions(r.suggestions))
        .catch(() => setFeatureSuggestions([]))
        .finally(() => setLoadingFeatures(false))
    }
  }, [activeTab, currentDataset, featureSuggestions.length])

  const handleLoadImportance = useCallback(async () => {
    if (!currentDataset || !targetColumn.trim()) return
    setLoadingImportance(true)
    try {
      const result = await api.features.importance(currentDataset.id, targetColumn.trim())
      setImportanceFeatures(result.features)
      setImportanceProblemType(result.problem_type)
    } finally {
      setLoadingImportance(false)
    }
  }, [currentDataset, targetColumn])

  // Owner-scoped model artifacts (#28): download over an authenticated fetch so
  // the bearer token rides in the header — `window.open` can't send one and 401s.
  const handleModelDownload = useCallback(
    async (runId: string) => {
      try {
        await downloadFile(api.models.downloadUrl(runId), `model_${runId}.joblib`)
      } catch {
        addMessage({
          role: "assistant",
          content:
            "I couldn't download that model file just now. Please try again in a moment.",
          timestamp: new Date().toISOString(),
        })
      }
    },
    [addMessage]
  )

  const handleModelReport = useCallback(
    async (runId: string) => {
      try {
        await downloadFile(api.models.reportUrl(runId), `report_${runId}.pdf`)
      } catch {
        addMessage({
          role: "assistant",
          content:
            "I couldn't download that report just now. Please try again in a moment.",
          timestamp: new Date().toISOString(),
        })
      }
    },
    [addMessage]
  )

  const handleFeatureApplied = useCallback(
    (result: FeatureSetResult) => {
      addMessage({
        role: "assistant",
        content: `I've applied your feature transformations. ${result.new_columns.length} new column${result.new_columns.length !== 1 ? "s" : ""} were created: ${result.new_columns.slice(0, 5).join(", ")}${result.new_columns.length > 5 ? "..." : ""}. The dataset now has ${result.total_columns} columns total.`,
        timestamp: new Date().toISOString(),
      })
    },
    [addMessage]
  )

  const handleSendMessage = useCallback(async (directText?: string) => {
    const text = (directText !== undefined ? directText : chatInput).trim()
    if (!text || isStreaming) return

    if (directText === undefined) setChatInput("")
    setChatSuggestions([])  // clear previous suggestions when a new message is sent
    addMessage({
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    })
    addMessage({
      role: "assistant",
      content: "",
      timestamp: new Date().toISOString(),
    })
    setStreaming(true)

    try {
      const response = await api.chat.send(projectId, text)
      const reader = response.body?.getReader()
      if (!reader) {
        setStreaming(false)
        return
      }

      const decoder = new TextDecoder()
      let buffer = ""

      const sseHandlers = createSSEHandlers({
        appendToLastMessage,
        attachABTestResultToLastMessage,
        attachAccuracyAlertConfigToLastMessage,
        attachAggregateExplanationToLastMessage,
        attachAlertRuleToLastMessage,
        attachApiKeyResultToLastMessage,
        attachAutoInsightToLastMessage,
        attachMonitoringNoteToLastMessage,
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
        setStreaming,
      })

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split("\n\n")
        buffer = parts.pop() ?? ""

        for (const part of parts) {
          const trimmed = part.trim()
          if (trimmed.startsWith("data: ")) {
            try {
              const json = JSON.parse(trimmed.slice(6))
              const handler = sseHandlers[json.type]
              if (handler) {
                handler(json)
              } else if (process.env.NODE_ENV !== "production") {
                // Surfaces SSE contract drift in dev: a backend event with no
                // frontend handler (the #17 silent-drop bug) instead of nothing.
                console.warn(`[sse] unhandled event type: ${json.type}`)
              }
            } catch {
              // skip malformed JSON
            }
          }
        }
      }
    } catch {
      appendToLastMessage("\n\n[Connection error. Please try again.]")
    } finally {
      setStreaming(false)
    }
  }, [
    chatInput,
    isStreaming,
    projectId,
    addMessage,
    setStreaming,
    appendToLastMessage,
    attachChartToLastMessage,
    attachCrosstabToLastMessage,
    attachComputeToLastMessage,
    attachSegmentToLastMessage,
    attachForecastToLastMessage,
    attachDataReadinessToLastMessage,
    attachCorrelationToLastMessage,
    attachGroupStatsToLastMessage,
    attachRenameResultToLastMessage,
    attachTrainingStartedToLastMessage,
    attachDataStoryToLastMessage,
    attachFilterToLastMessage,
    setActiveFilter,
    attachDeployedToLastMessage,
    attachModelCardToLastMessage,
    attachReportToLastMessage,
    attachFeatureSuggestionsToLastMessage,
    attachFeaturesAppliedToLastMessage,
    attachSegmentPerformanceToLastMessage,
    attachColumnProfileToLastMessage,
    attachClustersToLastMessage,
    attachTimeWindowToLastMessage,
    attachTopNToLastMessage,
    attachWhatIfChatToLastMessage,
    attachPredictionErrorsToLastMessage,
    attachRecordsToLastMessage,
    attachDataExportToLastMessage,
    attachNullMapToLastMessage,
    attachSummaryStatsToLastMessage,
    attachValueCountsToLastMessage,
    attachPairCorrelationToLastMessage,
    attachStatQueryToLastMessage,
    attachGroupTrendsToLastMessage,
    attachSplitStrategyToLastMessage,
    attachFeatureSelectionToLastMessage,
    attachModelImprovementToLastMessage,
    attachModelSelectionToLastMessage,
    attachModelQualityScoreToLastMessage,
    attachAutoRetrainToLastMessage,
    attachConversationExportToLastMessage,
    attachHealthSummaryToLastMessage,
    attachPredictionOpportunitiesToLastMessage,
    attachDatasetComparisonToLastMessage,
    attachInlinePredictionToLastMessage,
    attachMultiPredictionToLastMessage,
    attachGoalTrainingToLastMessage,
    attachSensitivityToLastMessage,
    attachInteractionToLastMessage,
    attachOnboardingGuideToLastMessage,
    attachVersionHistoryToLastMessage,
    attachLearningCurveToLastMessage,
    attachTemplateSavedToLastMessage,
    attachTemplateListToLastMessage,
    attachTemplateReplayToLastMessage,
    attachPresetSavedToLastMessage,
    attachPresetListToLastMessage,
    attachRankedPredictionsToLastMessage,
    attachPredictionCohortToLastMessage,
    attachSdkDownloadToLastMessage,
    attachPortfolioToLastMessage,
    attachRateLimitToLastMessage,
    attachPartialDependenceToLastMessage,
    attachCalibrationCheckToLastMessage,
    attachSlaMetricsToLastMessage,
    attachQuotaAlertConfigToLastMessage,
    attachScheduleSetToLastMessage,
    attachABTestResultToLastMessage,
    attachWebhookHistoryToLastMessage,
    attachClassImbalanceCheckToLastMessage,
    attachWebhookHealthSummaryToLastMessage,
    attachExecutiveBriefingToLastMessage,
    attachServiceExportToLastMessage,
    attachVersionComparisonToLastMessage,
    attachEnsembleRecommendationToLastMessage,
    attachTuneChatToLastMessage,
    attachCvScoreDistributionToLastMessage,
    attachPredictionAnalyticsChatToLastMessage,
    attachConfusionMatrixChatToLastMessage,
    attachLocalExplanationToLastMessage,
    attachProdInputDistToLastMessage,
    attachCovariateDriftAlertToLastMessage,
    attachQuotaRunwayToLastMessage,
    attachCostEstimateToLastMessage,
    attachUsagePatternToLastMessage,
    attachPredictionLogExportToLastMessage,
    attachRecentPredictionsToLastMessage,
    attachPredictionAuditToLastMessage,
    attachConfidenceTrendToLastMessage,
    attachFeedbackAccuracyReportToLastMessage,
    attachFairnessCheckToLastMessage,
    attachBatchJobResultsToLastMessage,
    attachProdPredictionExplanationToLastMessage,
    attachAggregateExplanationToLastMessage,
    attachWebhookRegisteredToLastMessage,
    attachWebhookListChatToLastMessage,
    attachWebhookRemovedChatToLastMessage,
    attachWebhookTestChatToLastMessage,
    attachAlertRuleToLastMessage,
    attachApiKeyResultToLastMessage,
    attachDeploymentsOverviewToLastMessage,
    attachProdPerformanceToLastMessage,
    attachErrorDistributionToLastMessage,
    attachModelCardExportToLastMessage,
    attachModelComparisonSummaryToLastMessage,
    attachCrossModelFeaturesToLastMessage,
    attachAccuracyAlertConfigToLastMessage,
    attachRollbackChatToLastMessage,
    attachConfidenceThresholdConfigToLastMessage,
    attachInputValidationRuleToLastMessage,
    attachDashboardConfigToLastMessage,
    attachDashboardMetadataToLastMessage,
    attachEmbedCodeToLastMessage,
    attachShareLinkToLastMessage,
    attachWeeklyUsageReportToLastMessage,
    attachCrossProjectComparisonToLastMessage,
    attachWhatNextToLastMessage,
    attachMilestoneToLastMessage,
    attachAutoInsightToLastMessage,
    attachMonitoringNoteToLastMessage,
    attachColumnTypeSuggestionsToLastMessage,
    attachGoalSeekToLastMessage,
    attachGoalSeekHistoryToLastMessage,
    attachDeploymentChangelogToLastMessage,
    attachCrossDeployPredictionToLastMessage,
    attachLowAccuracyGuidanceToLastMessage,
    attachPredictionDeltaToLastMessage,
    attachCohortEvolutionToLastMessage,
    attachCounterfactualToLastMessage,
    attachPopulationCounterfactualToLastMessage,
    attachSimilarRecordsToLastMessage,
    attachFeEngineeringImpactToLastMessage,
    attachDataQualityImpactToLastMessage,
    attachOverfittingAnalysisToLastMessage,
    attachFeatureRedundancyToLastMessage,
    attachTargetLeakageToLastMessage,
    attachThresholdAnalysisToLastMessage,
    attachPerClassThresholdToLastMessage,
    attachConfidenceDistributionToLastMessage,
    attachSampleSizeAdequacyToLastMessage,
    attachClassFeatureImportanceToLastMessage,
    attachErrorCorrelationToLastMessage,
    attachOutputAnomaliesToLastMessage,
    attachOutputDistributionShiftToLastMessage,
    attachFeaturePsiToLastMessage,
    attachMinFeatureSetToLastMessage,
    attachRetrainingReadinessToLastMessage,
    attachPredictionValueTrendToLastMessage,
    attachMonitoringDigestToLastMessage,
    attachModelStatusReportToLastMessage,
    attachProductionThresholdOptimizerToLastMessage,
    attachDeployPredDistCompareToLastMessage,
    attachWeeklyDigestConfigToLastMessage,
    attachPromotionReadinessToLastMessage,
    attachDeploymentScorecardToLastMessage,
    attachThroughputAssessmentToLastMessage,
    attachDriftImportanceRankingToLastMessage,
    attachFeatureDriftAlertConfigToLastMessage,
    attachInputDistDriftAlertConfigToLastMessage,
    attachLowActivityAlertConfigToLastMessage,
    attachHighActivityBurstConfigToLastMessage,
    attachLatencyAlertConfigToLastMessage,
    attachAutoRollbackConfigToLastMessage,
    attachPredValueAlertConfigToLastMessage,
    attachDegradationRetrainConfigToLastMessage,
    attachSegmentDriftToLastMessage,
    attachSegmentPredTrendToLastMessage,
    attachSegmentConfTrendToLastMessage,
    attachConfHeatmapToLastMessage,
    attachUptimeSummaryToLastMessage,
    attachCostSensitiveThresholdToLastMessage,
    attachFeatureSweepToLastMessage,
    attachSavedScenariosToLastMessage,
    attachCanaryStatusToLastMessage,
    attachDeploymentHealthScorecardToLastMessage,
    attachConfidenceBandToLastMessage,
    attachRetrainCompleteNotifyToLastMessage,
    attachOutcomeCalibrationToLastMessage,
    attachBatchJobHistoryToLastMessage,
    attachPerformanceDecayRateToLastMessage,
  ])

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (!file) return

      setUploading(true)
      try {
        const result = await api.data.upload(projectId, file)
        const dataset: Dataset = {
          id: result.dataset_id,
          project_id: projectId,
          filename: result.filename,
          row_count: result.row_count,
          column_count: result.column_count,
          uploaded_at: new Date().toISOString(),
        }
        setDataset(dataset, result.preview, result.column_stats, result.insights)
        setFeatureSuggestions([]) // reset on new upload

        if (result.insights && result.insights.length > 0) {
          const insightLines = result.insights
            .slice(0, 3)
            .map((i: DataInsight) => `- ${i.title}: ${i.detail}`)
            .join("\n")
          addMessage({
            role: "assistant",
            content: `I've analyzed **${result.filename}** (${result.row_count.toLocaleString()} rows, ${result.column_count} columns). Here's what I noticed:\n\n${insightLines}\n\nWhat would you like to explore? You can also check the **Features** tab to see transformation suggestions.`,
            timestamp: new Date().toISOString(),
          })
        }
        // Surface data-aware suggestion chips immediately after upload
        if (result.suggestions && result.suggestions.length > 0) {
          setChatSuggestions(result.suggestions)
        }
        // Show right panel on upload if hidden
        setRightPanelVisible(true)
      } catch {
        addMessage({
          role: "assistant",
          content:
            "There was a problem uploading your file. Please make sure it is a valid CSV or Excel file (.csv, .xlsx, .xls) and try again.",
          timestamp: new Date().toISOString(),
        })
      } finally {
        setUploading(false)
      }
    },
    [projectId, setDataset, addMessage]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/vnd.ms-excel": [".xls"],
    },
    maxFiles: 1,
    disabled: uploading,
  })

  const handleLoadSample = useCallback(async () => {
    setUploading(true)
    try {
      const result = await api.data.loadSample(projectId)
      const dataset: Dataset = {
        id: result.dataset_id,
        project_id: projectId,
        filename: result.filename,
        row_count: result.row_count,
        column_count: result.column_count,
        uploaded_at: new Date().toISOString(),
      }
      setDataset(dataset, result.preview, result.column_stats, result.insights ?? [])
      setFeatureSuggestions([])
      setRightPanelVisible(true)
      addMessage({
        role: "assistant",
        content: `I've loaded the sample sales dataset — **${result.row_count} rows** across 5 product lines and 4 regions. This data contains monthly sales figures with date, product, region, revenue, and units sold.\n\nYou can use this to try predicting **revenue** using the other columns. Ask me anything about the data, or jump to the **Features** tab to get started.`,
        timestamp: new Date().toISOString(),
      })
      if (result.suggestions && result.suggestions.length > 0) {
        setChatSuggestions(result.suggestions)
      }
    } catch {
      addMessage({
        role: "assistant",
        content: "There was a problem loading the sample data. Please try again.",
        timestamp: new Date().toISOString(),
      })
    } finally {
      setUploading(false)
    }
  }, [projectId, setDataset, addMessage])

  const handleImportUrl = useCallback(async (url: string) => {
    setUploading(true)
    try {
      const result = await api.data.uploadFromUrl(projectId, url)
      const dataset: Dataset = {
        id: result.dataset_id,
        project_id: projectId,
        filename: result.filename,
        row_count: result.row_count,
        column_count: result.column_count,
        uploaded_at: new Date().toISOString(),
      }
      setDataset(dataset, result.preview, result.column_stats, result.insights ?? [])
      setFeatureSuggestions([])
      setRightPanelVisible(true)
    } catch {
      addMessage({
        role: "assistant",
        content: "There was a problem importing from that URL. Make sure it is a public Google Sheets link or a direct CSV URL.",
        timestamp: new Date().toISOString(),
      })
    } finally {
      setUploading(false)
    }
  }, [projectId, setDataset, addMessage])

  if (loadingProject) {
    return (
      <div className="flex h-screen flex-col">
        <div className="flex h-10 shrink-0 items-center gap-3 border-b px-4">
          <div className="h-3 w-16 animate-pulse rounded bg-muted" />
          <div className="h-3 w-2 animate-pulse rounded bg-muted" />
          <div className="h-3 w-32 animate-pulse rounded bg-muted" />
        </div>
        <div className="flex flex-1 overflow-hidden">
          <div className="flex w-2/5 flex-col border-r gap-3 p-4">
            <div className="h-4 w-48 animate-pulse rounded bg-muted" />
            <div className="h-20 w-full animate-pulse rounded bg-muted" />
            <div className="h-4 w-32 animate-pulse rounded bg-muted" />
            <div className="h-16 w-full animate-pulse rounded bg-muted" />
          </div>
          <div className="flex flex-1 flex-col gap-3 p-4">
            <div className="flex gap-2">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="h-8 w-16 animate-pulse rounded bg-muted" />
              ))}
            </div>
            <div className="h-40 w-full animate-pulse rounded bg-muted" />
            <div className="h-24 w-full animate-pulse rounded bg-muted" />
          </div>
        </div>
      </div>
    )
  }

  if (projectLoadError) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-16">
        <ErrorDisplay
          variant="full-page"
          message={projectLoadError}
          onRetry={() => {
            setLoadingProject(true)
            setReloadKey((k) => k + 1)
          }}
        />
      </div>
    )
  }

  return (
    <div className="flex h-screen flex-col">
      {/* Top bar */}
      <div className="flex h-10 shrink-0 items-center gap-3 border-b px-4">
        <button
          onClick={() => router.push("/")}
          className="text-xs text-muted-foreground hover:text-foreground"
        >
          ← Projects
        </button>
        <span className="text-xs text-muted-foreground">/</span>
        <h1 className="text-xs font-medium truncate">
          {currentProject?.name ?? "Loading..."}
        </h1>
        <div className="ml-auto flex items-center gap-2">
          {/* Desktop: hide/show panel toggle */}
          <Button
            variant="ghost"
            size="sm"
            className="hidden md:flex h-7 px-2 text-xs"
            onClick={() => setRightPanelVisible((v) => !v)}
          >
            {rightPanelVisible ? "Hide panel" : "Show panel"}
          </Button>
          {/* Mobile: chat / panel toggle */}
          <div className="flex md:hidden rounded-md border overflow-hidden text-xs">
            <button
              onClick={() => setMobileView("chat")}
              className={`px-3 py-1 transition-colors ${mobileView === "chat" ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}
            >
              Chat
            </button>
            <button
              onClick={() => setMobileView("panel")}
              className={`px-3 py-1 transition-colors ${mobileView === "panel" ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}
            >
              Data
            </button>
          </div>
        </div>
      </div>

      {/* Workflow progress stepper — always visible regardless of active panel */}
      {currentDataset && (
        <WorkflowProgress
          hasDataset={!!currentDataset}
          hasFeatures={featureSuggestions.length > 0 || importanceFeatures.length > 0}
          hasSelectedModel={!!selectedModelRunId}
          hasValidation={hasValidation}
          hasDeployment={hasDeployment}
          onStepClick={(tab) => {
            setActiveTab(tab as RightTab)
            setMobileView("panel")
          }}
        />
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Chat Panel — full-width on mobile when active, fixed width on md+ */}
        <div
          className={`flex flex-col border-r transition-all
            ${mobileView === "chat" ? "flex" : "hidden"} md:flex
            ${rightPanelVisible ? "md:w-2/5" : "md:flex-1"}
            w-full`}
        >
          <ScrollArea className="flex-1 overflow-y-auto">
            <div className="flex flex-col gap-3 p-4">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`group relative max-w-[90%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                      msg.role === "user"
                        ? "bg-muted text-foreground"
                        : "border bg-card text-card-foreground"
                    }`}
                  >
                    {msg.role === "assistant" && msg.content && (
                      <CopyButton text={msg.content} />
                    )}
                    {msg.content}
                    {isStreaming &&
                      i === messages.length - 1 &&
                      msg.role === "assistant" &&
                      msg.content === "" && (
                        <span className="inline-flex gap-1">
                          <span className="animate-pulse">.</span>
                          <span className="animate-pulse delay-100">.</span>
                          <span className="animate-pulse delay-200">.</span>
                        </span>
                      )}
                    {msg.chart && <ChartMessage spec={msg.chart} />}
                    {msg.crosstab && <CrosstabTable result={msg.crosstab} />}
                    {msg.segment_comparison && (
                      <SegmentComparisonCard result={msg.segment_comparison} />
                    )}
                    {msg.forecast && <ForecastChart result={msg.forecast} />}
                    {msg.data_readiness && (
                      <ReadinessCheckCard result={msg.data_readiness} />
                    )}
                    {msg.target_correlation && (
                      <CorrelationBarCard result={msg.target_correlation} />
                    )}
                    {msg.group_stats && (
                      <GroupStatsCard result={msg.group_stats} />
                    )}
                    {msg.rename_result && (
                      <RenameResultCard result={msg.rename_result} />
                    )}
                    {msg.training_started && (
                      <TrainingStartedCard
                        result={msg.training_started}
                        onNavigateToModels={() => setActiveTab("models")}
                      />
                    )}
                    {msg.data_story && (
                      <DataStoryCard result={msg.data_story} />
                    )}
                    {msg.filter_set && (
                      <FilterSetCard result={msg.filter_set} />
                    )}
                    {msg.deployed && (
                      <DeployedCard result={msg.deployed} />
                    )}
                    {msg.model_card && (
                      <ModelCardView card={msg.model_card} />
                    )}
                    {msg.report_ready && (
                      <ReportReadyCard result={msg.report_ready} />
                    )}
                    {msg.feature_suggestions && (
                      <FeatureSuggestCard result={msg.feature_suggestions} />
                    )}
                    {msg.features_applied && (
                      <FeaturesAppliedCard result={msg.features_applied} />
                    )}
                    {msg.segment_performance && (
                      <SegmentPerformanceCard result={msg.segment_performance} />
                    )}
                    {msg.column_profile && (
                      <ColumnProfileCard profile={msg.column_profile} />
                    )}
                    {msg.clusters && (
                      <ClusteringCard result={msg.clusters} />
                    )}
                    {msg.time_window_comparison && (
                      <TimeWindowCard result={msg.time_window_comparison} />
                    )}
                    {msg.top_n && (
                      <TopNCard result={msg.top_n} />
                    )}
                    {msg.whatif_chat_result && (
                      <WhatIfChatCard result={msg.whatif_chat_result} />
                    )}
                    {msg.pred_errors && (
                      <PredictionErrorCard result={msg.pred_errors} />
                    )}
                    {msg.records && (
                      <RecordTableCard result={msg.records} />
                    )}
                    {msg.data_export && (
                      <DataExportCard result={msg.data_export} />
                    )}
                    {msg.null_map && (
                      <NullMapCard result={msg.null_map} />
                    )}
                    {msg.summary_stats && (
                      <SummaryStatsCard result={msg.summary_stats} />
                    )}
                    {msg.value_counts && (
                      <ValueCountCard result={msg.value_counts} />
                    )}
                    {msg.pair_correlation && (
                      <PairCorrelationCard result={msg.pair_correlation} />
                    )}
                    {msg.stat_query && (
                      <StatQueryCard result={msg.stat_query} />
                    )}
                    {msg.group_trends && (
                      <GroupTrendCard result={msg.group_trends} />
                    )}
                    {msg.split_strategy && (
                      <SplitStrategyCard result={msg.split_strategy} />
                    )}
                    {msg.feature_selection && (
                      <FeatureSelectionCard result={msg.feature_selection} projectId={projectId} />
                    )}
                    {msg.model_improvement && (
                      <ModelImprovementCard result={msg.model_improvement} />
                    )}
                    {msg.model_selection && (
                      <ModelSelectionCard result={msg.model_selection} />
                    )}
                    {msg.model_quality_score && (
                      <ModelQualityScoreCard result={msg.model_quality_score} />
                    )}
                    {msg.auto_retrain && (
                      <AutoRetrainCard result={msg.auto_retrain} />
                    )}
                    {msg.conversation_export && (
                      <ConversationExportCard info={msg.conversation_export} />
                    )}
                    {msg.health_summary && (
                      <ProjectHealthCard
                        summary={msg.health_summary}
                        onSwitchTab={setActiveTab}
                      />
                    )}
                    {msg.prediction_opportunities && (
                      <PredictionOpportunitiesCard
                        result={msg.prediction_opportunities}
                      />
                    )}
                    {msg.dataset_comparison && (
                      <DatasetComparisonCard result={msg.dataset_comparison} />
                    )}
                    {msg.inline_prediction && (
                      <InlinePredictionCard result={msg.inline_prediction} />
                    )}
                    {msg.multi_prediction && (
                      <MultiPredictionCard result={msg.multi_prediction} />
                    )}
                    {msg.goal_training && (
                      <GoalTrainingCard result={msg.goal_training} />
                    )}
                    {msg.sensitivity && (
                      <SensitivityCard result={msg.sensitivity} />
                    )}
                    {msg.interaction && (
                      <InteractionCard result={msg.interaction} />
                    )}
                    {msg.ranked_predictions && (
                      <RankedPredictionsCard result={msg.ranked_predictions} />
                    )}
                    {msg.prediction_cohort && (
                      <PredictionCohortCard result={msg.prediction_cohort} />
                    )}
                    {msg.onboarding_guide && (
                      <OnboardingGuideCard
                        guide={msg.onboarding_guide}
                        onSwitchTab={(tab) => setActiveTab(tab as RightTab)}
                      />
                    )}
                    {msg.version_history && (
                      <DataVersionHistoryCard history={msg.version_history} />
                    )}
                    {msg.learning_curve && (
                      <LearningCurveCard result={msg.learning_curve} />
                    )}
                    {msg.template_saved && (
                      <TemplateSavedCard info={msg.template_saved} />
                    )}
                    {msg.template_list && (
                      <TemplateListCard
                        info={msg.template_list}
                        onReplay={(name) => setChatInput(`replay my "${name}" template`)}
                      />
                    )}
                    {msg.template_replay && (
                      <TemplateReplayCard
                        info={msg.template_replay}
                        onQueryClick={(q) => setChatInput(q)}
                      />
                    )}
                    {msg.preset_saved && (
                      <PresetSavedCard preset={msg.preset_saved} />
                    )}
                    {msg.preset_list && (
                      <PresetListCard preset_list={msg.preset_list} />
                    )}
                    {msg.sdk_download && (
                      <SdkDownloadCard info={msg.sdk_download} />
                    )}
                    {msg.portfolio && (
                      <PortfolioCard result={msg.portfolio} />
                    )}
                    {msg.rate_limit && (
                      <RateLimitCard info={msg.rate_limit} />
                    )}
                    {msg.partial_dependence && (
                      <PartialDependenceCard result={msg.partial_dependence} />
                    )}
                    {msg.calibration_check && (
                      <CalibrationCheckCard result={msg.calibration_check} />
                    )}
                    {msg.sla_metrics && (
                      <SlaCard sla={msg.sla_metrics} />
                    )}
                    {msg.quota_alert_config && (
                      <QuotaAlertCard config={msg.quota_alert_config} />
                    )}
                    {msg.schedule_set && (
                      <ScheduleSetChatCard result={msg.schedule_set} />
                    )}
                    {msg.ab_test_result && (
                      <ABTestChatCard result={msg.ab_test_result} />
                    )}
                    {msg.webhook_history && (
                      <WebhookHistoryCard data={msg.webhook_history} />
                    )}
                    {msg.class_imbalance_check && (
                      <ClassImbalanceChatCard
                        data={msg.class_imbalance_check}
                        onSwitchTab={(tab) => setActiveTab(tab as RightTab)}
                      />
                    )}
                    {msg.webhook_health_summary && (
                      <WebhookHealthSummaryCard data={msg.webhook_health_summary} />
                    )}
                    {msg.executive_briefing && (
                      <ExecutiveBriefingCard briefing={msg.executive_briefing} />
                    )}
                    {msg.service_export && (
                      <ServiceExportChatCard result={msg.service_export} />
                    )}
                    {msg.version_comparison && (
                      <DeploymentVersionComparisonCard result={msg.version_comparison} />
                    )}
                    {msg.ensemble_recommendation && (
                      <EnsembleRecommendationCard result={msg.ensemble_recommendation} />
                    )}
                    {msg.tune_chat && (
                      <TuningChatCard result={msg.tune_chat} />
                    )}
                    {msg.cv_score_distribution && (
                      <CvScoreDistributionCard result={msg.cv_score_distribution} />
                    )}
                    {msg.prediction_analytics_chat && (
                      <PredictionAnalyticsChatCard result={msg.prediction_analytics_chat} />
                    )}
                    {msg.confusion_matrix_chat && (
                      <ConfusionMatrixChatCard result={msg.confusion_matrix_chat} />
                    )}
                    {msg.local_explanation && (
                      <LocalExplanationCard result={msg.local_explanation} />
                    )}
                    {msg.prod_input_dist && (
                      <ProductionInputDistributionCard result={msg.prod_input_dist} />
                    )}
                    {msg.covariate_drift_alert && (
                      <CovariateDriftAlertCard result={msg.covariate_drift_alert} />
                    )}
                    {msg.quota_runway && (
                      <QuotaRunwayCard result={msg.quota_runway} />
                    )}
                    {msg.cost_estimate && (
                      <CostEstimateCard result={msg.cost_estimate} />
                    )}
                    {msg.usage_pattern && (
                      <UsagePatternCard result={msg.usage_pattern} />
                    )}
                    {msg.prediction_log_export && (
                      <PredictionLogExportCard result={msg.prediction_log_export} />
                    )}
                    {msg.recent_predictions && (
                      <RecentPredictionsCard result={msg.recent_predictions} />
                    )}
                    {msg.prediction_audit && (
                      <PredictionAuditCard result={msg.prediction_audit} />
                    )}
                    {msg.confidence_trend && (
                      <ConfidenceTrendCard result={msg.confidence_trend} />
                    )}
                    {msg.feedback_accuracy_report && (
                      <FeedbackAccuracyCard result={msg.feedback_accuracy_report} />
                    )}
                    {msg.fairness_check && (
                      <FairnessCheckCard result={msg.fairness_check} />
                    )}
                    {msg.batch_job_results && (
                      <BatchJobResultCard result={msg.batch_job_results} />
                    )}
                    {msg.prod_prediction_explanation && (
                      <ProductionExplanationCard result={msg.prod_prediction_explanation} />
                    )}
                    {msg.aggregate_explanation && (
                      <AggregateExplanationCard result={msg.aggregate_explanation} />
                    )}
                    {msg.webhook_registered && (
                      <WebhookRegisteredCard info={msg.webhook_registered} />
                    )}
                    {msg.webhook_list_chat && (
                      <WebhookListChatCard result={msg.webhook_list_chat} />
                    )}
                    {msg.webhook_removed_chat && (
                      <WebhookRemovedChatCard info={msg.webhook_removed_chat} />
                    )}
                    {msg.webhook_test_chat && (
                      <WebhookTestChatCard result={msg.webhook_test_chat} />
                    )}
                    {msg.alert_rule && (
                      <AlertRuleCard result={msg.alert_rule} />
                    )}
                    {msg.api_key_result && (
                      <ApiKeyChatCard result={msg.api_key_result} />
                    )}
                    {msg.deployments_overview && (
                      <DeploymentsOverviewCard result={msg.deployments_overview} />
                    )}
                    {msg.prod_performance && (
                      <ProdPerformanceCard result={msg.prod_performance} />
                    )}
                    {msg.error_distribution && (
                      <ErrorDistributionCard result={msg.error_distribution} />
                    )}
                    {msg.model_card_export && (
                      <ModelCardExportCard info={msg.model_card_export} />
                    )}
                    {msg.model_comparison_summary && (
                      <ModelComparisonSummaryCard result={msg.model_comparison_summary} />
                    )}
                    {msg.cross_model_features && (
                      <CrossModelFeaturesCard result={msg.cross_model_features} />
                    )}
                    {msg.accuracy_alert_config && (
                      <AccuracyAlertCard config={msg.accuracy_alert_config} />
                    )}
                    {msg.rollback_chat && (
                      <RollbackChatCard result={msg.rollback_chat} />
                    )}
                    {msg.confidence_threshold_config && (
                      <ConfidenceThresholdCard config={msg.confidence_threshold_config} />
                    )}
                    {msg.input_validation_rule && (
                      <InputValidationRuleCard result={msg.input_validation_rule} />
                    )}
                    {msg.dashboard_config && (
                      <DashboardConfigCard config={msg.dashboard_config} />
                    )}
                    {msg.dashboard_metadata && (
                      <DashboardMetadataCard result={msg.dashboard_metadata} />
                    )}
                    {msg.embed_code && (
                      <EmbedCodeCard result={msg.embed_code} />
                    )}
                    {msg.share_link && (
                      <ShareLinkCard result={msg.share_link} />
                    )}
                    {msg.weekly_usage_report && (
                      <WeeklyUsageReportCard result={msg.weekly_usage_report} />
                    )}
                    {msg.cross_project_comparison && (
                      <CrossProjectComparisonCard result={msg.cross_project_comparison} />
                    )}
                    {msg.what_next && (
                      <WhatNextCard
                        result={msg.what_next}
                        onActionClick={(action) => {
                          setChatInput(action)
                        }}
                      />
                    )}
                    {msg.milestone && (
                      <MilestoneCard
                        result={msg.milestone}
                        onActionClick={(prompt) => {
                          setChatInput(prompt)
                        }}
                      />
                    )}
                    {msg.monitoring_note && (
                      <MonitoringNoteCard note={msg.monitoring_note} />
                    )}
                    {msg.auto_insight && (
                      <AutoInsightCard
                        result={msg.auto_insight}
                        onActionClick={(prompt) => {
                          handleSendMessage(prompt)
                        }}
                      />
                    )}
                    {msg.column_type_suggestions && (
                      <ColumnTypeSuggestionCard
                        result={msg.column_type_suggestions}
                        onActionClick={(prompt) => {
                          setChatInput(prompt)
                        }}
                      />
                    )}
                    {msg.goal_seek && (
                      <GoalSeekCard
                        result={msg.goal_seek}
                        onActionClick={(message) => {
                          handleSendMessage(message)
                        }}
                      />
                    )}
                    {msg.goal_seek_history && (
                      <GoalSeekHistoryCard result={msg.goal_seek_history} />
                    )}
                    {msg.deployment_changelog && (
                      <DeploymentChangelogCard result={msg.deployment_changelog} />
                    )}
                    {msg.cross_deploy_prediction && (
                      <CrossDeployPredictionCard result={msg.cross_deploy_prediction} />
                    )}
                    {msg.low_accuracy_guidance && (
                      <LowAccuracyGuidanceCard
                        result={msg.low_accuracy_guidance}
                        onActionClick={handleSendMessage}
                      />
                    )}
                    {msg.prediction_delta && (
                      <PredictionDeltaCard result={msg.prediction_delta} />
                    )}
                    {msg.cohort_evolution && (
                      <CohortEvolutionCard result={msg.cohort_evolution} />
                    )}
                    {msg.counterfactual && (
                      <CounterfactualCard data={msg.counterfactual} />
                    )}
                    {msg.population_counterfactual && (
                      <PopulationCounterfactualCard data={msg.population_counterfactual} />
                    )}
                    {msg.similar_records && (
                      <SimilarRecordsCard data={msg.similar_records} />
                    )}
                    {msg.fe_impact && (
                      <FeatureEngineeringImpactCard result={msg.fe_impact} />
                    )}
                    {msg.data_quality_impact && (
                      <DataQualityImpactCard result={msg.data_quality_impact} />
                    )}
                    {msg.overfitting_analysis && (
                      <OverfittingAnalysisCard result={msg.overfitting_analysis} />
                    )}
                    {msg.feature_redundancy && (
                      <FeatureRedundancyCard result={msg.feature_redundancy} />
                    )}
                    {msg.target_leakage && (
                      <TargetLeakageCard result={msg.target_leakage} />
                    )}
                    {msg.threshold_analysis && (
                      <ThresholdAnalysisCard result={msg.threshold_analysis} />
                    )}
                    {msg.per_class_threshold && (
                      <PerClassThresholdCard result={msg.per_class_threshold} />
                    )}
                    {msg.confidence_distribution && (
                      <ConfidenceDistributionCard result={msg.confidence_distribution} />
                    )}
                    {msg.sample_size_adequacy && (
                      <SampleSizeAdequacyCard result={msg.sample_size_adequacy} />
                    )}
                    {msg.class_feature_importance && (
                      <ClassFeatureImportanceCard result={msg.class_feature_importance} />
                    )}
                    {msg.error_correlation && (
                      <ErrorCorrelationCard result={msg.error_correlation} />
                    )}
                    {msg.output_anomalies && (
                      <PredictionOutputAnomalyCard result={msg.output_anomalies} />
                    )}
                    {msg.output_distribution_shift && (
                      <PredictionOutputDistributionCard result={msg.output_distribution_shift} />
                    )}
                    {msg.feature_psi && (
                      <FeaturePsiCard result={msg.feature_psi} />
                    )}
                    {msg.min_feature_set && (
                      <MinFeatureSetCard result={msg.min_feature_set} />
                    )}
                    {msg.retraining_readiness && (
                      <RetrainingReadinessCard
                        data={msg.retraining_readiness}
                        onActionClick={handleSendMessage}
                      />
                    )}
                    {msg.prediction_value_trend && (
                      <PredictionValueTrendCard result={msg.prediction_value_trend} />
                    )}
                    {msg.monitoring_digest && (
                      <MonitoringDigestCard result={msg.monitoring_digest} />
                    )}
                    {msg.model_status_report && (
                      <ModelStatusReportCard info={msg.model_status_report} />
                    )}
                    {msg.production_threshold_optimizer && (
                      <ProductionThresholdOptimizerCard result={msg.production_threshold_optimizer} />
                    )}
                    {msg.deploy_pred_dist_compare && (
                      <DeploymentPredictionDistributionCard result={msg.deploy_pred_dist_compare} />
                    )}
                    {msg.weekly_digest_config && (
                      <WeeklyDigestConfigCard data={msg.weekly_digest_config} />
                    )}
                    {msg.promotion_readiness && (
                      <PromotionReadinessCard
                        result={msg.promotion_readiness}
                        onActionClick={handleSendMessage}
                      />
                    )}
                    {msg.deployment_scorecard && (
                      <DeploymentScorecardCard result={msg.deployment_scorecard} />
                    )}
                    {msg.throughput_assessment && (
                      <DeploymentThroughputCard result={msg.throughput_assessment} />
                    )}
                    {msg.drift_importance_ranking && (
                      <DriftImportanceCard result={msg.drift_importance_ranking} />
                    )}
                    {msg.feature_drift_alert_config && (
                      <FeatureDriftAlertCard config={msg.feature_drift_alert_config} />
                    )}
                    {msg.input_dist_drift_alert_config && (
                      <InputDistDriftAlertCard config={msg.input_dist_drift_alert_config} />
                    )}
                    {msg.low_activity_alert_config && (
                      <LowActivityAlertCard config={msg.low_activity_alert_config} />
                    )}
                    {msg.high_activity_burst_config && (
                      <HighActivityBurstCard config={msg.high_activity_burst_config} />
                    )}
                    {msg.latency_alert_config && (
                      <LatencyAlertCard config={msg.latency_alert_config} />
                    )}
                    {msg.auto_rollback_config && (
                      <AutoRollbackCard data={msg.auto_rollback_config} />
                    )}
                    {msg.pred_value_alert_config && (
                      <PredValueAlertCard data={msg.pred_value_alert_config} />
                    )}
                    {msg.degradation_retrain_config && (
                      <DegradationRetrainCard data={msg.degradation_retrain_config} />
                    )}
                    {msg.segment_drift && (
                      <SegmentDriftCard result={msg.segment_drift} />
                    )}
                    {msg.segment_pred_trend && (
                      <SegmentPredictionTrendCard result={msg.segment_pred_trend} />
                    )}
                    {msg.segment_conf_trend && (
                      <SegmentConfidenceTrendCard result={msg.segment_conf_trend} />
                    )}
                    {msg.conf_heatmap && (
                      <ConfidenceHeatmapCard result={msg.conf_heatmap} />
                    )}
                    {msg.feature_sweep && (
                      <FeatureSweepCard result={msg.feature_sweep} />
                    )}
                    {msg.saved_scenarios && (
                      <SavedScenariosCard result={msg.saved_scenarios} />
                    )}
                    {msg.canary_status && (
                      <CanaryCard result={msg.canary_status} />
                    )}
                    {msg.deployment_health_scorecard && (
                      <DeploymentHealthScorecardCard result={msg.deployment_health_scorecard} />
                    )}
                    {msg.confidence_band && (
                      <ConfidenceBandCard data={msg.confidence_band} />
                    )}
                    {msg.retrain_complete_notify && (
                      <RetrainCompleteNotifyCard data={msg.retrain_complete_notify} />
                    )}
                    {msg.outcome_calibration && (
                      <OutcomeCalibrationCard result={msg.outcome_calibration} />
                    )}
                    {msg.batch_job_history && (
                      <BatchJobHistoryCard result={msg.batch_job_history} />
                    )}
                    {msg.performance_decay_rate && (
                      <PerformanceDecayRateCard result={msg.performance_decay_rate} />
                    )}
                    {msg.uptime_summary && (
                      <ApiUptimeSummaryCard result={msg.uptime_summary} />
                    )}
                    {msg.cost_sensitive_threshold && (
                      <CostSensitiveThresholdCard result={msg.cost_sensitive_threshold} />
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          <div className="border-t p-3">
            {/* Follow-up suggestion chips */}
            {!isStreaming && chatSuggestions.length > 0 && (
              <div className="mb-2" data-testid="suggestion-chips">
                <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Try asking:</p>
                <div className="flex flex-wrap gap-1.5">
                  {chatSuggestions.map((suggestion, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        setChatInput(suggestion)
                        setChatSuggestions([])
                      }}
                      className="flex items-center gap-1 rounded-full border border-primary/30 bg-primary/5 px-3 py-1 text-xs text-primary hover:bg-primary/10 transition-colors"
                      data-testid="suggestion-chip"
                    >
                      <span className="text-primary/60">▸</span>
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="flex gap-2 items-end">
              <Textarea
                placeholder="Ask about your data... (Shift+Enter for new line)"
                value={chatInput}
                onChange={(e) => {
                  setChatInput(e.target.value)
                  // Auto-grow: reset then allow natural height
                  e.target.style.height = "auto"
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    handleSendMessage()
                  }
                }}
                disabled={isStreaming}
                rows={1}
                className="resize-none min-h-[36px] max-h-[120px] py-2"
              />
              <Button
                onClick={() => handleSendMessage()}
                disabled={isStreaming || !chatInput.trim()}
              >
                Send
              </Button>
            </div>
          </div>
        </div>

        {/* Right Panel — full-width on mobile when active, 3/5 on md+ */}
        {(rightPanelVisible || mobileView === "panel") && (
          <div className={`flex flex-col overflow-hidden
            ${mobileView === "panel" ? "flex" : "hidden"} md:flex
            w-full md:w-3/5`}>
            {currentDataset ? (
              <>
                {/* Tab Bar */}
                <div role="tablist" aria-label="Project workspace tabs" className="flex border-b overflow-x-auto">
                  {(["data", "features", "importance", "models", "validate", "deploy"] as RightTab[]).map((tab) => {
                    const labels: Record<RightTab, string> = {
                      data: "Data",
                      features: "Features",
                      importance: "Importance",
                      models: "Models",
                      validate: "Validate",
                      deploy: "Deploy",
                    }
                    return (
                      <button
                        key={tab}
                        role="tab"
                        aria-selected={activeTab === tab}
                        aria-controls={`tabpanel-${tab}`}
                        id={`tab-${tab}`}
                        onClick={() => setActiveTab(tab)}
                        className={`shrink-0 px-4 py-2.5 text-xs font-medium capitalize transition-colors ${
                          activeTab === tab
                            ? "border-b-2 border-primary text-foreground"
                            : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {labels[tab]}
                      </button>
                    )
                  })}
                </div>

                {activeTab === "data" && (
                  <div role="tabpanel" id="tabpanel-data" aria-labelledby="tab-data" className="flex flex-1 flex-col overflow-hidden">
                    {activeFilter && (
                      <div className="px-4 pt-3">
                        <FilterBadge
                          filter={activeFilter}
                          onClear={async () => {
                            await api.data.clearFilter(currentDataset.id)
                            setActiveFilter(null)
                          }}
                        />
                      </div>
                    )}
                    <DataPreviewPanel
                      filename={currentDataset.filename}
                      rowCount={currentDataset.row_count}
                      columnCount={currentDataset.column_count}
                      preview={dataPreview}
                      stats={columnStats}
                      insights={dataInsights}
                    />
                    <div className="border-t px-4 py-3">
                      <h3 className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                        Project Datasets
                      </h3>
                      <DatasetListPanel
                        projectId={projectId}
                        onMerged={(result) => {
                          addMessage({
                            role: "assistant",
                            content: `I've merged the two datasets on **${result.join_key}** (${result.how} join). The result has ${result.row_count.toLocaleString()} rows and ${result.column_count} columns, saved as **${result.filename}**.${result.conflict_columns.length > 0 ? ` Columns that appeared in both datasets were renamed with suffixes: ${result.conflict_columns.join(", ")}.` : ""} You can now use this merged dataset for feature engineering and model training.`,
                            timestamp: new Date().toISOString(),
                          })
                        }}
                      />
                    </div>
                    {(anomalyResult || (columnStats && columnStats.some((c) => c.dtype !== "object"))) && (
                      <div className="border-t px-4 py-3">
                        <h3 className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                          Anomaly Detection
                        </h3>
                        <AnomalyCard
                          result={anomalyResult ?? undefined}
                          datasetId={currentDataset.id}
                          numericFeatures={columnStats
                            ?.filter((c) => c.dtype !== "object")
                            .map((c) => c.name)
                            .slice(0, 10)}
                        />
                      </div>
                    )}
                    {cleaningSuggestion && (
                      <div className="border-t px-4 py-3">
                        <h3 className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                          Data Cleaning
                        </h3>
                        <CleaningCard
                          suggestion={cleaningSuggestion}
                          datasetId={currentDataset.id}
                          onCleaned={(result: CleanResult) => {
                            setCleaningSuggestion(null)
                            addMessage({
                              role: "assistant",
                              content: `Done! ${result.operation_result.summary} The dataset now has ${result.updated_stats.row_count.toLocaleString()} rows.`,
                              timestamp: new Date().toISOString(),
                            })
                          }}
                        />
                      </div>
                    )}
                    {refreshPrompt && (
                      <div className="border-t px-4 py-3">
                        <h3 className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                          Update Data
                        </h3>
                        <RefreshCard
                          datasetId={currentDataset.id}
                          prompt={refreshPrompt}
                          onRefreshed={(result: DatasetRefreshResult) => {
                            setRefreshPrompt(null)
                            addMessage({
                              role: "assistant",
                              content: `Dataset updated! Your new file has ${result.row_count.toLocaleString()} rows and ${result.column_count} columns.${result.compatible ? " Your model configuration is compatible — you can retrain now." : " Warning: some feature columns are missing. You may need to re-configure features before retraining."}`,
                              timestamp: new Date().toISOString(),
                            })
                          }}
                        />
                      </div>
                    )}
                    {computeSuggestion && (
                      <div className="border-t px-4 py-3">
                        <h3 className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                          Computed Column
                        </h3>
                        <ComputeCard
                          suggestion={computeSuggestion}
                          onComputed={(result: ComputeResult) => {
                            setComputeSuggestion(null)
                            addMessage({
                              role: "assistant",
                              content: `Done! ${result.compute_result.summary}`,
                              timestamp: new Date().toISOString(),
                            })
                          }}
                        />
                      </div>
                    )}
                    <div className="border-t px-4 py-3">
                      <h3 className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                        Data Readiness
                      </h3>
                      <ReadinessCheckCard datasetId={currentDataset.id} />
                    </div>
                    <div className="border-t px-4 py-3">
                      <h3 className="mb-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                        Data Dictionary
                      </h3>
                      <DictionaryCard datasetId={currentDataset.id} />
                    </div>
                  </div>
                )}

                {activeTab === "features" && (
                  <ScrollArea role="tabpanel" id="tabpanel-features" aria-labelledby="tab-features" className="flex-1">
                    <div className="p-4">
                      <div className="mb-3">
                        <h3 className="text-sm font-semibold">Feature Suggestions</h3>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          Select transformations to apply. Approved features will be added as new columns.
                        </p>
                      </div>
                      {loadingFeatures ? (
                        <p className="text-xs text-muted-foreground">Analyzing columns...</p>
                      ) : (
                        <FeatureSuggestionsPanel
                          datasetId={currentDataset.id}
                          suggestions={featureSuggestions}
                          onApplied={handleFeatureApplied}
                        />
                      )}
                    </div>
                  </ScrollArea>
                )}

                {activeTab === "importance" && (
                  <ScrollArea role="tabpanel" id="tabpanel-importance" aria-labelledby="tab-importance" className="flex-1">
                    <div className="p-4">
                      <div className="mb-3">
                        <h3 className="text-sm font-semibold">Feature Importance</h3>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          Select a column to predict and see which features are most useful.
                        </p>
                      </div>
                      <div className="mb-4 flex gap-2">
                        <Input
                          placeholder="Target column (e.g. revenue)"
                          value={targetColumn}
                          onChange={(e) => setTargetColumn(e.target.value)}
                          className="text-xs"
                          onKeyDown={(e) => {
                            if (e.key === "Enter") handleLoadImportance()
                          }}
                        />
                        <Button
                          size="sm"
                          onClick={handleLoadImportance}
                          disabled={!targetColumn.trim() || loadingImportance}
                        >
                          {loadingImportance ? "..." : "Analyze"}
                        </Button>
                      </div>
                      {importanceFeatures.length > 0 && (
                        <FeatureImportancePanel
                          features={importanceFeatures}
                          targetColumn={targetColumn}
                          problemType={importanceProblemType}
                        />
                      )}
                    </div>
                  </ScrollArea>
                )}

                {activeTab === "validate" && currentDataset && (
                  <div role="tabpanel" id="tabpanel-validate" aria-labelledby="tab-validate" className="flex flex-1 flex-col overflow-hidden">
                    <ValidationPanel
                      projectId={projectId}
                      selectedRunId={selectedModelRunId}
                      algorithmName={selectedModelAlgorithm}
                      onNavigateToModels={() => setActiveTab("models")}
                      onValidationComplete={() => setHasValidation(true)}
                    />
                  </div>
                )}

                {activeTab === "deploy" && currentDataset && (
                  <ScrollArea role="tabpanel" id="tabpanel-deploy" aria-labelledby="tab-deploy" className="flex-1">
                    <div className="p-4">
                      <div className="mb-3">
                        <h3 className="text-sm font-semibold">Deploy Model</h3>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          One-click deployment as a live prediction API + shareable dashboard.
                        </p>
                      </div>
                      <DeploymentPanel
                        projectId={projectId}
                        selectedRunId={selectedModelRunId}
                        algorithmName={selectedModelAlgorithm}
                        onDeployed={(dep) => {
                          setHasDeployment(true)
                          addMessage({
                            role: "assistant",
                            content: `Your model is live! Share this link with anyone: ${dep.dashboard_url}\n\nThey can fill in values and get instant predictions — no code required. Developers can also use the API endpoint directly: POST ${dep.endpoint_path}`,
                            timestamp: new Date().toISOString(),
                          })
                        }}
                      />
                    </div>
                  </ScrollArea>
                )}

                {activeTab === "models" && currentDataset && (
                  <ScrollArea role="tabpanel" id="tabpanel-models" aria-labelledby="tab-models" className="flex-1">
                    <div className="p-4">
                      <div className="mb-3">
                        <h3 className="text-sm font-semibold">Model Training</h3>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          Train and compare ML models on your dataset. Make sure you have set a target column in the Features tab first.
                        </p>
                      </div>
                      <ModelTrainingPanel
                        projectId={projectId}
                        onModelSelected={(runId, algorithm) => {
                          setSelectedModelRunId(runId)
                          setSelectedModelAlgorithm(algorithm)
                          addMessage({
                            role: "assistant",
                            content: `I have selected this model for your project. You can now go to the **Validate** tab to run cross-validation, see error analysis, and understand feature importance. Or we can deploy it as a live prediction API whenever you're ready.`,
                            timestamp: new Date().toISOString(),
                          })
                        }}
                        onModelDownload={handleModelDownload}
                        onModelReport={handleModelReport}
                        onTrainingComplete={(chips) => {
                          if (chips.length > 0) setChatSuggestions(chips)
                        }}
                      />
                    </div>
                  </ScrollArea>
                )}
              </>
            ) : (
              <UploadPanel
                getRootProps={getRootProps}
                getInputProps={getInputProps}
                isDragActive={isDragActive}
                uploading={uploading}
                onLoadSample={handleLoadSample}
                onImportUrl={handleImportUrl}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  function handleCopy() {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <button
      onClick={handleCopy}
      aria-label={copied ? "Copied!" : "Copy message"}
      className="absolute right-2 top-2 hidden group-hover:flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
    >
      {copied ? "✓ Copied" : "Copy"}
    </button>
  )
}

function UploadPanel({
  getRootProps,
  getInputProps,
  isDragActive,
  uploading,
  onLoadSample,
  onImportUrl,
}: {
  getRootProps: ReturnType<typeof useDropzone>["getRootProps"]
  getInputProps: ReturnType<typeof useDropzone>["getInputProps"]
  isDragActive: boolean
  uploading: boolean
  onLoadSample: () => void
  onImportUrl: (url: string) => void
}) {
  const [urlInput, setUrlInput] = useState("")
  const [urlOpen, setUrlOpen] = useState(false)

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-6 p-8">
      {/* Step-by-step workflow */}
      <div className="w-full max-w-md">
        <p className="mb-3 text-center text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          How it works
        </p>
        <ol className="space-y-2">
          {[
            { step: 1, label: "Upload", desc: "Drop a CSV or Excel file to get started" },
            { step: 2, label: "Explore", desc: "Ask questions about your data in plain English" },
            { step: 3, label: "Shape", desc: "AI suggests features; you approve or adjust" },
            { step: 4, label: "Train", desc: "Choose a target column and train models" },
            { step: 5, label: "Validate", desc: "See what the model gets right and where it struggles" },
            { step: 6, label: "Deploy", desc: "One click — live API + shareable prediction dashboard" },
          ].map(({ step, label, desc }) => (
            <li key={step} className={`flex items-start gap-3 ${step === 1 ? "" : "opacity-50"}`}>
              <span className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold ${step === 1 ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}>
                {step}
              </span>
              <div>
                <span className="text-xs font-semibold">{label}</span>
                <span className="ml-1.5 text-xs text-muted-foreground">{desc}</span>
              </div>
            </li>
          ))}
        </ol>
      </div>

      <div
        {...getRootProps()}
        className={`flex h-48 w-full max-w-md cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-colors ${
          isDragActive
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50"
        } ${uploading ? "pointer-events-none opacity-50" : ""}`}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <p className="text-sm text-muted-foreground">Uploading...</p>
        ) : isDragActive ? (
          <p className="text-sm font-medium">Drop your file here</p>
        ) : (
          <>
            <p className="text-sm font-medium">Drop your CSV or Excel file here</p>
            <p className="mt-1 text-xs text-muted-foreground">or click to browse</p>
          </>
        )}
      </div>

      {!uploading && (
        <div className="flex flex-col items-center gap-3 w-full max-w-md">
          <div className="flex flex-col items-center gap-1">
            <p className="text-xs text-muted-foreground">Don&apos;t have a dataset handy?</p>
            <button
              onClick={onLoadSample}
              className="text-xs text-primary hover:underline underline-offset-2"
            >
              Load sample sales data (200 rows, 5 columns)
            </button>
          </div>

          <div className="flex flex-col items-center gap-1 w-full">
            <button
              onClick={() => setUrlOpen((v) => !v)}
              className="text-xs text-muted-foreground hover:text-primary hover:underline underline-offset-2"
            >
              {urlOpen ? "Cancel" : "Import from Google Sheets or CSV URL"}
            </button>
            {urlOpen && (
              <div className="flex w-full gap-2 mt-1">
                <input
                  type="url"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://docs.google.com/spreadsheets/d/..."
                  className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-xs placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && urlInput.trim()) {
                      onImportUrl(urlInput.trim())
                      setUrlInput("")
                      setUrlOpen(false)
                    }
                  }}
                />
                <button
                  onClick={() => {
                    if (urlInput.trim()) {
                      onImportUrl(urlInput.trim())
                      setUrlInput("")
                      setUrlOpen(false)
                    }
                  }}
                  disabled={!urlInput.trim()}
                  className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50 hover:bg-primary/90 transition-colors"
                >
                  Import
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function DataPreviewPanel({
  filename,
  rowCount,
  columnCount,
  preview,
  stats,
  insights,
}: {
  filename: string
  rowCount: number
  columnCount: number
  preview: Record<string, unknown>[]
  stats: import("@/lib/types").ColumnStat[]
  insights: DataInsight[]
}) {
  const columns = preview.length > 0 ? Object.keys(preview[0]) : []

  const severityClass = (s: DataInsight["severity"]) =>
    s === "critical"
      ? "bg-red-50 border-red-200 text-red-800 dark:bg-red-950 dark:border-red-900 dark:text-red-200"
      : s === "warning"
      ? "bg-amber-50 border-amber-200 text-amber-800 dark:bg-amber-950 dark:border-amber-900 dark:text-amber-200"
      : "bg-blue-50 border-blue-200 text-blue-800 dark:bg-blue-950 dark:border-blue-900 dark:text-blue-200"

  return (
    <div className="flex flex-1 min-h-0 flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 border-b px-4 py-3">
        <h2 className="text-sm font-semibold">{filename}</h2>
        <Badge variant="outline">{rowCount.toLocaleString()} rows</Badge>
        <Badge variant="outline">{columnCount} columns</Badge>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4">
          {/* Insights panel */}
          {insights.length > 0 && (
            <div className="mb-5">
              <h3 className="mb-2 text-sm font-semibold">Insights</h3>
              <div className="flex flex-col gap-2">
                {insights.map((insight, i) => (
                  <div
                    key={i}
                    className={`rounded-lg border px-3 py-2 text-xs ${severityClass(insight.severity)}`}
                  >
                    <p className="font-semibold">{insight.title}</p>
                    <p className="mt-0.5 opacity-80">{insight.detail}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Column Stats */}
          {stats.length > 0 && (
            <div className="mb-6">
              <h3 className="mb-3 text-sm font-semibold">Column Statistics</h3>
              <div className="grid grid-cols-2 gap-2 lg:grid-cols-3">
                {stats.map((col) => (
                  <Card key={col.name} size="sm">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <span className="truncate">{col.name}</span>
                        <Badge variant="secondary">{col.dtype}</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-1 text-xs text-muted-foreground">
                        <p>
                          Nulls: {col.null_count} ({col.null_pct.toFixed(1)}%)
                        </p>
                        <p>Unique: {col.unique_count}</p>
                        {col.mean != null && (
                          <p>
                            Mean: {Number(col.mean).toFixed(2)} | Std:{" "}
                            {col.std != null ? Number(col.std).toFixed(2) : "N/A"}
                          </p>
                        )}
                        {col.min != null && col.max != null && (
                          <p>
                            Range: {col.min} - {col.max}
                          </p>
                        )}
                        {col.outliers && col.outliers.count > 0 && (
                          <p className="text-amber-600 dark:text-amber-400">
                            {col.outliers.count} outlier
                            {col.outliers.count !== 1 ? "s" : ""} (
                            {col.outliers.pct.toFixed(1)}%)
                          </p>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          <Separator className="my-4" />

          {/* Data Table */}
          <h3 className="mb-3 text-sm font-semibold">
            Data Preview (first {preview.length} rows)
          </h3>
          {preview.length > 0 && (
            <div className="overflow-x-auto rounded-lg border">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b bg-muted/50">
                    {columns.map((col) => (
                      <th
                        key={col}
                        className="whitespace-nowrap px-3 py-2 font-medium"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.map((row, i) => (
                    <tr key={i} className="border-b last:border-b-0">
                      {columns.map((col) => (
                        <td
                          key={col}
                          className="max-w-[200px] truncate whitespace-nowrap px-3 py-1.5"
                        >
                          {row[col] == null ? (
                            <span className="text-muted-foreground/50">null</span>
                          ) : (
                            String(row[col])
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
