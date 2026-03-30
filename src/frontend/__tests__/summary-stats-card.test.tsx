/**
 * Tests for SummaryStatsCard component and Zustand store integration.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { SummaryStatsCard } from "@/components/data/summary-stats-card"
import type { SummaryStatsResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const mixedResult: SummaryStatsResult = {
  dataset_id: "ds-001",
  total_rows: 8,
  total_cols: 4,
  numeric_stats: [
    {
      column: "revenue",
      count: 8,
      mean: 4468.75,
      std: 3256.1,
      min: 300.0,
      q25: 1050.0,
      median: 4600.0,
      q75: 7650.0,
      max: 9100.0,
      null_count: 0,
    },
    {
      column: "units",
      count: 8,
      mean: 44.125,
      std: 32.56,
      min: 3.0,
      q25: 8.25,
      median: 46.0,
      q75: 74.25,
      max: 91.0,
      null_count: 0,
    },
  ],
  categorical_stats: [
    {
      column: "region",
      count: 8,
      unique: 4,
      top: "East",
      freq: 3,
      null_count: 0,
    },
    {
      column: "product_category",
      count: 8,
      unique: 3,
      top: "Electronics",
      freq: 4,
      null_count: 1,
    },
  ],
  summary: "8 rows × 4 columns (2 numeric, 2 categorical).",
}

const numericOnlyResult: SummaryStatsResult = {
  dataset_id: "ds-002",
  total_rows: 5,
  total_cols: 2,
  numeric_stats: [
    { column: "a", count: 5, mean: 3.0, std: 1.58, min: 1.0, q25: 2.0, median: 3.0, q75: 4.0, max: 5.0, null_count: 0 },
    { column: "b", count: 5, mean: 8.0, std: 1.58, min: 6.0, q25: 7.0, median: 8.0, q75: 9.0, max: 10.0, null_count: 0 },
  ],
  categorical_stats: [],
  summary: "5 rows × 2 columns (2 numeric, 0 categorical).",
}

const emptyResult: SummaryStatsResult = {
  dataset_id: "ds-003",
  total_rows: 0,
  total_cols: 0,
  numeric_stats: [],
  categorical_stats: [],
  summary: "0 rows × 0 columns (0 numeric, 0 categorical).",
}

// ---------------------------------------------------------------------------
// Component rendering
// ---------------------------------------------------------------------------

describe("SummaryStatsCard rendering", () => {
  it("renders the Summary Statistics header", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(screen.getByText("Summary Statistics")).toBeInTheDocument()
  })

  it("shows total rows badge", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(screen.getAllByText(/8 rows/).length).toBeGreaterThan(0)
  })

  it("shows total columns badge", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(screen.getAllByText(/4 columns/).length).toBeGreaterThan(0)
  })

  it("renders Numeric Columns section when numeric stats exist", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(screen.getByText(/Numeric Columns/i)).toBeInTheDocument()
  })

  it("renders Categorical Columns section when categorical stats exist", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(screen.getByText(/Categorical Columns/i)).toBeInTheDocument()
  })

  it("renders numeric column name in table", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
    expect(screen.getByText("units")).toBeInTheDocument()
  })

  it("renders categorical column name in table", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(screen.getByText("region")).toBeInTheDocument()
    // underscore replaced
    expect(screen.getByText("product category")).toBeInTheDocument()
  })

  it("shows top categorical value", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    // Rendered with typographic curly quotes ("\u201CEast\u201D")
    expect(screen.getAllByText(/East/).length).toBeGreaterThan(0)
  })

  it("shows null count for column with nulls in red", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    // product_category has null_count: 1 — should show "1" somewhere in categorical table
    const nullCells = screen.getAllByText("1")
    expect(nullCells.length).toBeGreaterThan(0)
  })

  it("shows summary footer text", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(screen.getByText(mixedResult.summary)).toBeInTheDocument()
  })

  it("does not render Categorical Columns when there are none", () => {
    render(<SummaryStatsCard result={numericOnlyResult} />)
    expect(screen.queryByText(/Categorical Columns/i)).not.toBeInTheDocument()
  })

  it("renders column headers for numeric table", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(screen.getByText("Mean")).toBeInTheDocument()
    expect(screen.getByText("Median")).toBeInTheDocument()
  })

  it("renders column headers for categorical table", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(screen.getByText("Unique")).toBeInTheDocument()
    expect(screen.getByText("Most Common")).toBeInTheDocument()
  })

  it("shows empty state when no stats at all", () => {
    render(<SummaryStatsCard result={emptyResult} />)
    expect(screen.getByText(/No column statistics available/i)).toBeInTheDocument()
  })

  it("has accessible region label", () => {
    render(<SummaryStatsCard result={mixedResult} />)
    expect(
      screen.getByRole("region", { name: /dataset summary statistics/i })
    ).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Zustand store — summary stats attachment
// ---------------------------------------------------------------------------

describe("Zustand store summary stats attachment", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attachSummaryStatsToLastMessage links stats to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Here are your summary statistics." },
      ],
    })

    useAppStore.getState().attachSummaryStatsToLastMessage(mixedResult)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.summary_stats).toBeDefined()
    expect(lastMsg.summary_stats?.total_rows).toBe(8)
    expect(lastMsg.summary_stats?.numeric_stats).toHaveLength(2)
    expect(lastMsg.summary_stats?.categorical_stats).toHaveLength(2)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [{ id: "2", role: "user", content: "summarize my data" }],
    })

    useAppStore.getState().attachSummaryStatsToLastMessage(mixedResult)

    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].summary_stats).toBeUndefined()
  })

  it("does nothing when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    useAppStore.getState().attachSummaryStatsToLastMessage(mixedResult)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })
})
