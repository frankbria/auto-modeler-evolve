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
}))
