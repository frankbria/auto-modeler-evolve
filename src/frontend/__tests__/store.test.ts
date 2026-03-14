/**
 * Unit tests for the Zustand app store.
 *
 * The store is the source of truth for the entire frontend — messages,
 * dataset info, streaming state. These tests cover all mutations and
 * confirm each action produces the expected state shape.
 */

import { act } from "react"
import { useAppStore } from "../lib/store"
import type { ChatMessage, Project, Dataset, ColumnStat, DataInsight } from "../lib/types"

// Reset the store between tests so each test starts from initial state.
beforeEach(() => {
  useAppStore.setState({
    projects: [],
    currentProject: null,
    currentDataset: null,
    dataPreview: [],
    columnStats: [],
    dataInsights: [],
    messages: [],
    isStreaming: false,
  })
})

const makeProject = (overrides: Partial<Project> = {}): Project => ({
  id: "proj-1",
  name: "Test Project",
  status: "exploring",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  ...overrides,
})

const makeMessage = (role: "user" | "assistant", content: string): ChatMessage => ({
  role,
  content,
  timestamp: new Date().toISOString(),
})

const makeDataset = (): Dataset => ({
  id: "ds-1",
  project_id: "proj-1",
  filename: "sales.csv",
  row_count: 100,
  column_count: 5,
  uploaded_at: "2026-01-01T00:00:00Z",
})

const makeColumnStat = (): ColumnStat => ({
  name: "revenue",
  dtype: "float64",
  non_null_count: 100,
  null_count: 0,
  null_pct: 0,
  unique_count: 80,
  sample_values: [100, 200, 300],
})

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------

describe("setProjects", () => {
  it("replaces project list", () => {
    const projects = [makeProject({ id: "a" }), makeProject({ id: "b" })]
    act(() => useAppStore.getState().setProjects(projects))
    expect(useAppStore.getState().projects).toHaveLength(2)
    expect(useAppStore.getState().projects[0].id).toBe("a")
  })

  it("can clear projects with empty array", () => {
    act(() => useAppStore.getState().setProjects([makeProject()]))
    act(() => useAppStore.getState().setProjects([]))
    expect(useAppStore.getState().projects).toHaveLength(0)
  })
})

describe("setCurrentProject", () => {
  it("sets current project", () => {
    const project = makeProject()
    act(() => useAppStore.getState().setCurrentProject(project))
    expect(useAppStore.getState().currentProject?.id).toBe("proj-1")
  })

  it("can clear current project with null", () => {
    act(() => useAppStore.getState().setCurrentProject(makeProject()))
    act(() => useAppStore.getState().setCurrentProject(null))
    expect(useAppStore.getState().currentProject).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

describe("setDataset", () => {
  it("stores dataset, preview, stats, and insights", () => {
    const dataset = makeDataset()
    const preview = [{ id: 1, revenue: 100 }]
    const stats = [makeColumnStat()]
    const insights: DataInsight[] = [
      { type: "warning", severity: "warning", title: "Missing values", detail: "2% null" },
    ]
    act(() => useAppStore.getState().setDataset(dataset, preview, stats, insights))
    const state = useAppStore.getState()
    expect(state.currentDataset?.id).toBe("ds-1")
    expect(state.dataPreview).toHaveLength(1)
    expect(state.columnStats).toHaveLength(1)
    expect(state.dataInsights).toHaveLength(1)
  })

  it("defaults insights to empty array when not provided", () => {
    act(() => useAppStore.getState().setDataset(makeDataset(), [], [makeColumnStat()]))
    expect(useAppStore.getState().dataInsights).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

describe("addMessage", () => {
  it("appends a message to empty list", () => {
    const msg = makeMessage("user", "Hello")
    act(() => useAppStore.getState().addMessage(msg))
    expect(useAppStore.getState().messages).toHaveLength(1)
    expect(useAppStore.getState().messages[0].content).toBe("Hello")
  })

  it("preserves order for multiple messages", () => {
    act(() => useAppStore.getState().addMessage(makeMessage("user", "first")))
    act(() => useAppStore.getState().addMessage(makeMessage("assistant", "second")))
    act(() => useAppStore.getState().addMessage(makeMessage("user", "third")))
    const msgs = useAppStore.getState().messages
    expect(msgs[0].content).toBe("first")
    expect(msgs[1].content).toBe("second")
    expect(msgs[2].content).toBe("third")
  })
})

describe("setMessages", () => {
  it("replaces all messages", () => {
    act(() => useAppStore.getState().addMessage(makeMessage("user", "old")))
    act(() => useAppStore.getState().setMessages([makeMessage("assistant", "new")]))
    expect(useAppStore.getState().messages).toHaveLength(1)
    expect(useAppStore.getState().messages[0].content).toBe("new")
  })
})

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

describe("setStreaming", () => {
  it("toggles streaming state", () => {
    expect(useAppStore.getState().isStreaming).toBe(false)
    act(() => useAppStore.getState().setStreaming(true))
    expect(useAppStore.getState().isStreaming).toBe(true)
    act(() => useAppStore.getState().setStreaming(false))
    expect(useAppStore.getState().isStreaming).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// appendToLastMessage — SSE streaming chunk accumulation
// ---------------------------------------------------------------------------

describe("appendToLastMessage", () => {
  it("appends text to the last assistant message", () => {
    act(() => {
      useAppStore.getState().addMessage(makeMessage("user", "What is revenue?"))
      useAppStore.getState().addMessage(makeMessage("assistant", "Revenue is "))
    })
    act(() => useAppStore.getState().appendToLastMessage("$1M"))
    expect(useAppStore.getState().messages[1].content).toBe("Revenue is $1M")
  })

  it("accumulates multiple chunks correctly", () => {
    act(() => useAppStore.getState().addMessage(makeMessage("assistant", "")))
    act(() => useAppStore.getState().appendToLastMessage("Hello"))
    act(() => useAppStore.getState().appendToLastMessage(", "))
    act(() => useAppStore.getState().appendToLastMessage("world"))
    expect(useAppStore.getState().messages[0].content).toBe("Hello, world")
  })

  it("does not append if last message is from user", () => {
    act(() => useAppStore.getState().addMessage(makeMessage("user", "user text")))
    act(() => useAppStore.getState().appendToLastMessage(" extra"))
    // User messages must not be mutated — streaming only targets assistant messages
    expect(useAppStore.getState().messages[0].content).toBe("user text")
  })

  it("does nothing when messages list is empty", () => {
    // Should not throw
    act(() => useAppStore.getState().appendToLastMessage("orphan chunk"))
    expect(useAppStore.getState().messages).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// attachChartToLastMessage — chart rendering in chat
// ---------------------------------------------------------------------------

describe("attachChartToLastMessage", () => {
  const chart = {
    chart_type: "bar" as const,
    title: "Revenue by Region",
    data: [{ region: "North", revenue: 1000 }],
    x_key: "region",
    y_keys: ["revenue"],
    x_label: "Region",
    y_label: "Revenue",
  }

  it("attaches a chart spec to the last assistant message", () => {
    act(() => useAppStore.getState().addMessage(makeMessage("assistant", "Here is the chart:")))
    act(() => useAppStore.getState().attachChartToLastMessage(chart))
    expect(useAppStore.getState().messages[0].chart).toBeDefined()
    expect(useAppStore.getState().messages[0].chart?.title).toBe("Revenue by Region")
  })

  it("does not attach chart to a user message", () => {
    act(() => useAppStore.getState().addMessage(makeMessage("user", "show me a chart")))
    act(() => useAppStore.getState().attachChartToLastMessage(chart))
    expect(useAppStore.getState().messages[0].chart).toBeUndefined()
  })

  it("attaches to correct message when multiple messages exist", () => {
    act(() => {
      useAppStore.getState().addMessage(makeMessage("user", "q1"))
      useAppStore.getState().addMessage(makeMessage("assistant", "a1"))
      useAppStore.getState().addMessage(makeMessage("user", "q2"))
      useAppStore.getState().addMessage(makeMessage("assistant", "a2"))
    })
    act(() => useAppStore.getState().attachChartToLastMessage(chart))
    const msgs = useAppStore.getState().messages
    expect(msgs[3].chart).toBeDefined()  // last assistant
    expect(msgs[1].chart).toBeUndefined() // earlier assistant unchanged
  })
})
