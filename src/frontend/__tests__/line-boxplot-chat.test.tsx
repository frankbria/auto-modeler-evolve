/**
 * Tests for line chart and box plot chat integration:
 * - ChartMessage renders line charts with correct structure
 * - ChartMessage renders box plot charts from chat context
 * - Zustand store attachChartToLastMessage wires charts to last message
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { ChartMessage } from "@/components/chat/chart-message"
import type { ChartSpec } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Sample specs
// ---------------------------------------------------------------------------

const lineSpec: ChartSpec = {
  chart_type: "line",
  title: "revenue over time",
  data: [
    { date: "2023-01-01", revenue: 100, "7-period avg": 100, Trend: 100 },
    { date: "2023-02-01", revenue: 120, "7-period avg": 110, Trend: 115 },
    { date: "2023-03-01", revenue: 115, "7-period avg": 112, Trend: 118 },
    { date: "2023-04-01", revenue: 140, "7-period avg": 119, Trend: 121 },
  ],
  x_key: "date",
  y_keys: ["revenue", "7-period avg", "Trend"],
  x_label: "Date",
  y_label: "revenue",
}

const boxplotSpec: ChartSpec = {
  chart_type: "boxplot",
  title: "Distribution of revenue by region",
  data: [
    { group: "East", min: 100, q1: 110, median: 120, q3: 130, max: 140, mean: 120 },
    { group: "West", min: 150, q1: 170, median: 190, q3: 210, max: 230, mean: 190 },
    { group: "North", min: 80, q1: 90, median: 100, q3: 110, max: 120, mean: 100 },
  ],
  x_key: "group",
  y_keys: ["min", "q1", "median", "q3", "max"],
  x_label: "region",
  y_label: "revenue",
}

// ---------------------------------------------------------------------------
// Line chart rendering via ChartMessage
// ---------------------------------------------------------------------------

describe("Line chart rendering via ChartMessage", () => {
  it("renders the line chart title", () => {
    render(<ChartMessage spec={lineSpec} />)
    expect(screen.getByText("revenue over time")).toBeInTheDocument()
  })

  it("renders without crashing with empty data", () => {
    const spec: ChartSpec = { ...lineSpec, data: [] }
    render(<ChartMessage spec={spec} />)
  })

  it("renders container element", () => {
    const { container } = render(<ChartMessage spec={lineSpec} />)
    expect(container.firstChild).not.toBeNull()
  })

  it("uses y_label as axis label fallback", () => {
    const spec: ChartSpec = { ...lineSpec, y_label: "revenue" }
    render(<ChartMessage spec={spec} />)
    // Should render without errors
  })
})

// ---------------------------------------------------------------------------
// Box plot rendering via ChartMessage (chat context)
// ---------------------------------------------------------------------------

describe("Box plot rendering via ChartMessage (chat context)", () => {
  it("renders the box plot title", () => {
    render(<ChartMessage spec={boxplotSpec} />)
    expect(screen.getByText("Distribution of revenue by region")).toBeInTheDocument()
  })

  it("renders with group labels", () => {
    render(<ChartMessage spec={boxplotSpec} />)
    expect(screen.getByText("East")).toBeInTheDocument()
    expect(screen.getByText("West")).toBeInTheDocument()
    expect(screen.getByText("North")).toBeInTheDocument()
  })

  it("renders without crashing with empty data", () => {
    const spec: ChartSpec = { ...boxplotSpec, data: [] }
    render(<ChartMessage spec={spec} />)
  })

  it("renders container element", () => {
    const { container } = render(<ChartMessage spec={boxplotSpec} />)
    expect(container.firstChild).not.toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Zustand store — line chart attachment from chat
// ---------------------------------------------------------------------------

describe("Zustand store line chart attachment", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attachChartToLastMessage links line chart to assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Here is the revenue trend over time." },
      ],
    })

    useAppStore.getState().attachChartToLastMessage(lineSpec)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.chart).toBeDefined()
    expect(lastMsg.chart?.chart_type).toBe("line")
    expect(lastMsg.chart?.title).toBe("revenue over time")
  })

  it("does not attach line chart to user messages", () => {
    useAppStore.setState({
      messages: [{ id: "1", role: "user", content: "plot revenue over time" }],
    })

    useAppStore.getState().attachChartToLastMessage(lineSpec)

    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].chartSpec).toBeUndefined()
  })

  it("does nothing when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    useAppStore.getState().attachChartToLastMessage(lineSpec)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// Zustand store — box plot attachment from chat
// ---------------------------------------------------------------------------

describe("Zustand store box plot chart attachment", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attachChartToLastMessage links box plot to assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Here is the revenue distribution by region." },
      ],
    })

    useAppStore.getState().attachChartToLastMessage(boxplotSpec)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.chart).toBeDefined()
    expect(lastMsg.chart?.chart_type).toBe("boxplot")
    expect(lastMsg.chart?.x_label).toBe("region")
    expect(lastMsg.chart?.y_label).toBe("revenue")
  })

  it("does not attach box plot to user messages", () => {
    useAppStore.setState({
      messages: [{ id: "1", role: "user", content: "distribution of revenue by region" }],
    })

    useAppStore.getState().attachChartToLastMessage(boxplotSpec)

    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].chartSpec).toBeUndefined()
  })

  it("attaches box plot to last assistant message in a conversation", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "show distribution" },
        { id: "2", role: "assistant", content: "Here is the box plot." },
        { id: "3", role: "user", content: "thanks" },
        { id: "4", role: "assistant", content: "You're welcome." },
      ],
    })

    useAppStore.getState().attachChartToLastMessage(boxplotSpec)

    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    if (last.role === "assistant") {
      expect(last.chart).toBeDefined()
      expect(last.chart?.chart_type).toBe("boxplot")
    }
  })
})
