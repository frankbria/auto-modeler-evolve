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
}))
