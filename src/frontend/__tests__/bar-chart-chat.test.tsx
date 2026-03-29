/**
 * Tests for bar chart chat integration:
 * - ChartMessage renders bar charts correctly
 * - Zustand store attachChartToLastMessage wires bar chart to last message
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { ChartMessage } from "@/components/chat/chart-message"
import type { ChartSpec } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Bar ChartMessage rendering
// ---------------------------------------------------------------------------

const barSpec: ChartSpec = {
  chart_type: "bar",
  title: "Sum of revenue by region",
  data: [
    { label: "West", value: 500 },
    { label: "North", value: 250 },
    { label: "East", value: 250 },
  ],
  x_key: "label",
  y_keys: ["value"],
  x_label: "region",
  y_label: "revenue",
}

describe("Bar chart rendering via ChartMessage", () => {
  it("renders the bar chart title", () => {
    render(<ChartMessage spec={barSpec} />)
    expect(screen.getByText("Sum of revenue by region")).toBeInTheDocument()
  })

  it("renders with custom title", () => {
    const spec: ChartSpec = { ...barSpec, title: "Revenue by Product" }
    render(<ChartMessage spec={spec} />)
    expect(screen.getByText("Revenue by Product")).toBeInTheDocument()
  })

  it("renders with empty data without crashing", () => {
    const spec: ChartSpec = { ...barSpec, data: [] }
    render(<ChartMessage spec={spec} />)
    // Should not throw
  })

  it("renders container element", () => {
    const { container } = render(<ChartMessage spec={barSpec} />)
    expect(container.firstChild).not.toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Zustand store — bar chart attachment
// ---------------------------------------------------------------------------

describe("Zustand store bar chart attachment", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attachChartToLastMessage links bar chart to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Here is the bar chart." },
      ],
    })

    useAppStore.getState().attachChartToLastMessage(barSpec)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.chart).toBeDefined()
    expect(lastMsg.chart?.chart_type).toBe("bar")
    expect(lastMsg.chart?.title).toBe("Sum of revenue by region")
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [{ id: "2", role: "user", content: "bar chart please" }],
    })

    useAppStore.getState().attachChartToLastMessage(barSpec)

    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].chart).toBeUndefined()
  })

  it("does nothing when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    useAppStore.getState().attachChartToLastMessage(barSpec)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })
})
