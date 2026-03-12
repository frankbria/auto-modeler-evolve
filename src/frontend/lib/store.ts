import { create } from "zustand"
import type {
  Project,
  Dataset,
  ChatMessage,
  ColumnStat,
} from "./types"

interface AppState {
  projects: Project[]
  currentProject: Project | null
  currentDataset: Dataset | null
  dataPreview: Record<string, unknown>[]
  columnStats: ColumnStat[]
  messages: ChatMessage[]
  isStreaming: boolean

  setProjects: (projects: Project[]) => void
  setCurrentProject: (project: Project | null) => void
  setDataset: (
    dataset: Dataset,
    preview: Record<string, unknown>[],
    stats: ColumnStat[]
  ) => void
  addMessage: (message: ChatMessage) => void
  setMessages: (messages: ChatMessage[]) => void
  setStreaming: (streaming: boolean) => void
  appendToLastMessage: (content: string) => void
}

export const useAppStore = create<AppState>((set) => ({
  projects: [],
  currentProject: null,
  currentDataset: null,
  dataPreview: [],
  columnStats: [],
  messages: [],
  isStreaming: false,

  setProjects: (projects) => set({ projects }),

  setCurrentProject: (project) => set({ currentProject: project }),

  setDataset: (dataset, preview, stats) =>
    set({ currentDataset: dataset, dataPreview: preview, columnStats: stats }),

  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  setMessages: (messages) => set({ messages }),

  setStreaming: (streaming) => set({ isStreaming: streaming }),

  appendToLastMessage: (content) =>
    set((state) => {
      const messages = [...state.messages]
      const last = messages[messages.length - 1]
      if (last && last.role === "assistant") {
        messages[messages.length - 1] = {
          ...last,
          content: last.content + content,
        }
      }
      return { messages }
    }),
}))
