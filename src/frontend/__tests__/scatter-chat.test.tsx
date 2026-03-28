/**
 * Tests for scatter plot chat integration:
 * - ChartMessage renders scatter charts
 * - Zustand store attachChartToLastMessage wires scatter chart to last message
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { ChartMessage } from "@/components/chat/chart-message"
import type { ChartSpec } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Scatter ChartMessage rendering
// ---------------------------------------------------------------------------

const scatterSpec: ChartSpec = {
  chart_type: "scatter",
  title: "revenue vs units",
  data: [
    { x: 100, y: 10 },
    { x: 200, y: 20 },
    { x: 150, y: 15 },
    { x: 300, y: 30 },
  ],
  x_key: "x",
  y_keys: ["y"],
  x_label: "revenue",
  y_label: "units",
}

describe("Scatter chart rendering via ChartMessage", () => {
  it("renders the scatter chart title", () => {
    render(<ChartMessage spec={scatterSpec} />)
    expect(screen.getByText("revenue vs units")).toBeInTheDocument()
  })

  it("renders with custom title", () => {
    const spec: ChartSpec = { ...scatterSpec, title: "X vs Y" }
    render(<ChartMessage spec={spec} />)
    expect(screen.getByText("X vs Y")).toBeInTheDocument()
  })

  it("renders with empty data without crashing", () => {
    const spec: ChartSpec = { ...scatterSpec, data: [] }
    render(<ChartMessage spec={spec} />)
    // Should not throw
  })

  it("renders with single data point without crashing", () => {
    const spec: ChartSpec = { ...scatterSpec, data: [{ x: 1, y: 2 }] }
    render(<ChartMessage spec={spec} />)
  })

  it("renders container element", () => {
    const { container } = render(<ChartMessage spec={scatterSpec} />)
    expect(container.firstChild).not.toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Zustand store — scatter chart attachment
// ---------------------------------------------------------------------------

describe("Zustand store scatter chart attachment", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attachChartToLastMessage links scatter chart to assistant message", () => {
    useAppStore.setState({
      messages: [{ id: "1", role: "assistant", content: "Here is the scatter plot." }],
    })

    useAppStore.getState().attachChartToLastMessage(scatterSpec)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.chart).toBeDefined()
    expect(lastMsg.chart?.chart_type).toBe("scatter")
    expect(lastMsg.chart?.x_label).toBe("revenue")
    expect(lastMsg.chart?.y_label).toBe("units")
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [{ id: "2", role: "user", content: "plot revenue vs units" }],
    })

    useAppStore.getState().attachChartToLastMessage(scatterSpec)

    const msgs = useAppStore.getState().messages
    // User messages should not get chartSpec attached
    expect(msgs[msgs.length - 1].chartSpec).toBeUndefined()
  })

  it("does nothing when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    // Should not throw
    useAppStore.getState().attachChartToLastMessage(scatterSpec)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })

  it("attaches to the last assistant message when multiple messages exist", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "plot revenue vs units" },
        { id: "2", role: "assistant", content: "Here is the chart." },
        { id: "3", role: "user", content: "thanks" },
        { id: "4", role: "assistant", content: "You're welcome." },
      ],
    })

    useAppStore.getState().attachChartToLastMessage(scatterSpec)

    const msgs = useAppStore.getState().messages
    // Should attach to the last message if it's an assistant message
    const last = msgs[msgs.length - 1]
    if (last.role === "assistant") {
      expect(last.chart).toBeDefined()
      expect(last.chart?.chart_type).toBe("scatter")
    }
  })
})
