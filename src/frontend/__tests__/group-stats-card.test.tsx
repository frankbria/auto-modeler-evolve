/**
 * Tests for GroupStatsCard component and the attachGroupStatsToLastMessage store action.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { GroupStatsCard } from "@/components/data/group-stats-card"
import type { GroupStatsResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Fixtures -----------------------------------------------------------

const sumResult: GroupStatsResult = {
  dataset_id: "ds-1",
  group_col: "region",
  value_col: "revenue",
  value_cols: ["revenue"],
  agg: "sum",
  rows: [
    { group: "West", revenue: 500 },
    { group: "East", revenue: 330 },
    { group: "North", revenue: 250 },
  ],
  total: 1080,
  summary:
    "Grouped revenue by region (sum) — 3 groups. Highest: West (500.00). Top group is 46.3% of the total.",
}

const countResult: GroupStatsResult = {
  dataset_id: "ds-2",
  group_col: "product",
  value_col: "count",
  value_cols: ["count"],
  agg: "count",
  rows: [
    { group: "A", count: 10 },
    { group: "B", count: 5 },
  ],
  total: 15,
  summary: "Grouped count by product (count) — 2 groups. Highest: A (10).",
}

const emptyResult: GroupStatsResult = {
  dataset_id: "ds-3",
  group_col: "region",
  value_col: "revenue",
  value_cols: ["revenue"],
  agg: "mean",
  rows: [],
  total: null,
  summary: "No groups found after aggregation.",
}

// --- Component tests ----------------------------------------------------

describe("GroupStatsCard", () => {
  it("renders with testid", () => {
    render(<GroupStatsCard result={sumResult} />)
    expect(screen.getByTestId("group-stats-card")).toBeInTheDocument()
  })

  it("shows the group column and value column in the header", () => {
    render(<GroupStatsCard result={sumResult} />)
    expect(screen.getAllByText(/revenue/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/region/i).length).toBeGreaterThan(0)
  })

  it("shows aggregation label 'Total' for sum", () => {
    render(<GroupStatsCard result={sumResult} />)
    expect(screen.getAllByText(/Total/i).length).toBeGreaterThan(0)
  })

  it("shows aggregation label 'Count' for count", () => {
    render(<GroupStatsCard result={countResult} />)
    expect(screen.getAllByText(/Count/i).length).toBeGreaterThan(0)
  })

  it("renders all group rows", () => {
    render(<GroupStatsCard result={sumResult} />)
    expect(screen.getByText("West")).toBeInTheDocument()
    expect(screen.getByText("East")).toBeInTheDocument()
    expect(screen.getByText("North")).toBeInTheDocument()
  })

  it("shows the number of groups", () => {
    render(<GroupStatsCard result={sumResult} />)
    expect(screen.getAllByText(/3 groups/i).length).toBeGreaterThan(0)
  })

  it("shows summary text", () => {
    render(<GroupStatsCard result={sumResult} />)
    expect(screen.getByText(/Grouped revenue by region/i)).toBeInTheDocument()
  })

  it("renders empty state when rows is empty", () => {
    render(<GroupStatsCard result={emptyResult} />)
    expect(screen.getByTestId("group-stats-card")).toBeInTheDocument()
    expect(screen.getByText(/No groups found/i)).toBeInTheDocument()
  })

  it("shows total when agg is sum", () => {
    render(<GroupStatsCard result={sumResult} />)
    // Total 1080 should appear somewhere in the header stats
    expect(screen.getAllByText(/Total:/i).length).toBeGreaterThan(0)
  })

  it("does not show total label for count aggregation", () => {
    render(<GroupStatsCard result={countResult} />)
    // For count agg, the "Total:" label is not rendered (agg !== 'sum')
    const totalLabel = screen.queryByText(/Total:/i)
    expect(totalLabel).toBeNull()
  })
})

// --- Store action tests -------------------------------------------------

describe("attachGroupStatsToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches group_stats to the last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "show me revenue by region", timestamp: "t1" },
        { role: "assistant", content: "Here is the breakdown...", timestamp: "t2" },
      ],
    })
    const { attachGroupStatsToLastMessage } = useAppStore.getState()
    attachGroupStatsToLastMessage(sumResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].group_stats).toEqual(sumResult)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "show me revenue by region", timestamp: "t1" },
      ],
    })
    const { attachGroupStatsToLastMessage } = useAppStore.getState()
    attachGroupStatsToLastMessage(sumResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].group_stats).toBeUndefined()
  })

  it("is a no-op on empty message list", () => {
    useAppStore.setState({ messages: [] })
    const { attachGroupStatsToLastMessage } = useAppStore.getState()
    expect(() => attachGroupStatsToLastMessage(sumResult)).not.toThrow()
  })
})
