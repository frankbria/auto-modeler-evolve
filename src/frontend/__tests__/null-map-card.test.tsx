/**
 * Tests for NullMapCard component and Zustand store integration.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { NullMapCard } from "@/components/data/null-map-card"
import type { NullMapResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const nullMapWithMissing: NullMapResult = {
  dataset_id: "ds-001",
  total_rows: 100,
  total_columns: 4,
  columns_with_nulls: 2,
  fully_complete_columns: 2,
  overall_completeness: 87.5,
  summary: "2 of 4 columns have missing values. Overall completeness: 87.5%.",
  columns: [
    { column: "notes", null_count: 10, null_pct: 10.0, complete_pct: 90.0 },
    { column: "revenue", null_count: 3, null_pct: 3.0, complete_pct: 97.0 },
    { column: "region", null_count: 0, null_pct: 0.0, complete_pct: 100.0 },
    { column: "units", null_count: 0, null_pct: 0.0, complete_pct: 100.0 },
  ],
}

const nullMapComplete: NullMapResult = {
  dataset_id: "ds-002",
  total_rows: 50,
  total_columns: 3,
  columns_with_nulls: 0,
  fully_complete_columns: 3,
  overall_completeness: 100.0,
  summary: "All 3 columns are fully complete — no missing values!",
  columns: [
    { column: "region", null_count: 0, null_pct: 0.0, complete_pct: 100.0 },
    { column: "revenue", null_count: 0, null_pct: 0.0, complete_pct: 100.0 },
    { column: "units", null_count: 0, null_pct: 0.0, complete_pct: 100.0 },
  ],
}

// ---------------------------------------------------------------------------
// Component rendering
// ---------------------------------------------------------------------------

describe("NullMapCard rendering", () => {
  it("renders the Data Completeness header", () => {
    render(<NullMapCard result={nullMapWithMissing} />)
    expect(screen.getByText("Data Completeness")).toBeInTheDocument()
  })

  it("shows overall completeness badge", () => {
    render(<NullMapCard result={nullMapWithMissing} />)
    expect(screen.getByText("87.5% complete")).toBeInTheDocument()
  })

  it("shows columns with nulls count", () => {
    render(<NullMapCard result={nullMapWithMissing} />)
    expect(screen.getAllByText(/2 of 4 columns have missing values/).length).toBeGreaterThan(0)
  })

  it("shows total row count", () => {
    render(<NullMapCard result={nullMapWithMissing} />)
    expect(screen.getAllByText(/100/).length).toBeGreaterThan(0)
  })

  it("renders a row for each column", () => {
    render(<NullMapCard result={nullMapWithMissing} />)
    // column names with underscores replaced by spaces
    expect(screen.getByText("notes")).toBeInTheDocument()
    expect(screen.getByText("revenue")).toBeInTheDocument()
    expect(screen.getByText("region")).toBeInTheDocument()
    expect(screen.getByText("units")).toBeInTheDocument()
  })

  it("shows missing count for columns with nulls", () => {
    render(<NullMapCard result={nullMapWithMissing} />)
    expect(screen.getByText(/10 missing/)).toBeInTheDocument()
    expect(screen.getByText(/3 missing/)).toBeInTheDocument()
  })

  it("shows complete status for fully complete columns", () => {
    render(<NullMapCard result={nullMapWithMissing} />)
    const completeLabels = screen.getAllByText("complete")
    expect(completeLabels.length).toBeGreaterThan(0)
  })

  it("renders summary footer", () => {
    render(<NullMapCard result={nullMapWithMissing} />)
    expect(screen.getByText(nullMapWithMissing.summary)).toBeInTheDocument()
  })

  it("shows 100% complete badge for fully complete dataset", () => {
    render(<NullMapCard result={nullMapComplete} />)
    expect(screen.getByText("100% complete")).toBeInTheDocument()
  })

  it("shows all complete message for fully complete dataset", () => {
    render(<NullMapCard result={nullMapComplete} />)
    expect(screen.getAllByText(/All 3 columns are fully complete/).length).toBeGreaterThan(0)
  })

  it("replaces underscores in column names with spaces", () => {
    const resultWithUnderscore: NullMapResult = {
      ...nullMapWithMissing,
      columns: [
        { column: "total_revenue", null_count: 5, null_pct: 5.0, complete_pct: 95.0 },
      ],
    }
    render(<NullMapCard result={resultWithUnderscore} />)
    expect(screen.getByText("total revenue")).toBeInTheDocument()
  })

  it("has accessible region label", () => {
    render(<NullMapCard result={nullMapWithMissing} />)
    expect(screen.getByRole("region", { name: /missing values overview/i })).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Zustand store — null map attachment
// ---------------------------------------------------------------------------

describe("Zustand store null map attachment", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attachNullMapToLastMessage links null map to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Here is the missing values overview." },
      ],
    })

    useAppStore.getState().attachNullMapToLastMessage(nullMapWithMissing)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.null_map).toBeDefined()
    expect(lastMsg.null_map?.columns_with_nulls).toBe(2)
    expect(lastMsg.null_map?.overall_completeness).toBe(87.5)
    expect(lastMsg.null_map?.columns).toHaveLength(4)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [{ id: "2", role: "user", content: "show missing values" }],
    })

    useAppStore.getState().attachNullMapToLastMessage(nullMapWithMissing)

    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].null_map).toBeUndefined()
  })

  it("does nothing when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    useAppStore.getState().attachNullMapToLastMessage(nullMapWithMissing)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })

  it("attaches complete dataset null map correctly", () => {
    useAppStore.setState({
      messages: [{ id: "3", role: "assistant", content: "All complete!" }],
    })

    useAppStore.getState().attachNullMapToLastMessage(nullMapComplete)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.null_map?.columns_with_nulls).toBe(0)
    expect(lastMsg.null_map?.overall_completeness).toBe(100.0)
  })
})
