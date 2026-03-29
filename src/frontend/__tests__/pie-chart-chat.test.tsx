/**
 * Tests for pie chart chat integration:
 * - ChartMessage renders pie charts correctly
 * - Zustand store attachChartToLastMessage wires pie chart to last message
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { ChartMessage } from "@/components/chat/chart-message"
import type { ChartSpec } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Pie ChartMessage rendering
// ---------------------------------------------------------------------------

const pieSpec: ChartSpec = {
  chart_type: "pie",
  title: "revenue by region",
  data: [
    { name: "West", value: 500 },
    { name: "North", value: 250 },
    { name: "East", value: 250 },
  ],
  x_key: "name",
  y_keys: ["value"],
  x_label: "",
  y_label: "",
}

describe("Pie chart rendering via ChartMessage", () => {
  it("renders the pie chart title", () => {
    render(<ChartMessage spec={pieSpec} />)
    // Title appears in both <p> and <figcaption> (sr-only caption = title when no axis labels)
    expect(screen.getAllByText("revenue by region").length).toBeGreaterThan(0)
  })

  it("renders with custom title", () => {
    const spec: ChartSpec = { ...pieSpec, title: "Sales Breakdown" }
    render(<ChartMessage spec={spec} />)
    expect(screen.getAllByText("Sales Breakdown").length).toBeGreaterThan(0)
  })

  it("renders with empty data without crashing", () => {
    const spec: ChartSpec = { ...pieSpec, data: [] }
    render(<ChartMessage spec={spec} />)
    // Should not throw
  })

  it("renders with single slice without crashing", () => {
    const spec: ChartSpec = {
      ...pieSpec,
      data: [{ name: "East", value: 1000 }],
    }
    render(<ChartMessage spec={spec} />)
  })

  it("renders container element", () => {
    const { container } = render(<ChartMessage spec={pieSpec} />)
    expect(container.firstChild).not.toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Zustand store — pie chart attachment
// ---------------------------------------------------------------------------

describe("Zustand store pie chart attachment", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attachChartToLastMessage links pie chart to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Here is the pie chart." },
      ],
    })

    useAppStore.getState().attachChartToLastMessage(pieSpec)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.chart).toBeDefined()
    expect(lastMsg.chart?.chart_type).toBe("pie")
    expect(lastMsg.chart?.title).toBe("revenue by region")
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [{ id: "2", role: "user", content: "pie chart please" }],
    })

    useAppStore.getState().attachChartToLastMessage(pieSpec)

    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].chartSpec).toBeUndefined()
  })

  it("does nothing when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    useAppStore.getState().attachChartToLastMessage(pieSpec)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })
})
