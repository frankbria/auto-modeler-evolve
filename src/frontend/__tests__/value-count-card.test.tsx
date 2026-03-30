/**
 * Tests for ValueCountCard component and Zustand store integration.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { ValueCountCard } from "@/components/data/value-count-card"
import type { ValueCountResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const regionResult: ValueCountResult = {
  dataset_id: "ds-001",
  column: "region",
  total_rows: 10,
  non_null: 10,
  null_count: 0,
  unique_count: 4,
  rows: [
    { value: "East", count: 4, pct: 40.0 },
    { value: "West", count: 3, pct: 30.0 },
    { value: "North", count: 2, pct: 20.0 },
    { value: "South", count: 1, pct: 10.0 },
  ],
  has_more: false,
  summary: "'region' has 4 unique values. Most common: 'East' (40% of non-null rows).",
}

const productResult: ValueCountResult = {
  dataset_id: "ds-002",
  column: "product_category",
  total_rows: 100,
  non_null: 95,
  null_count: 5,
  unique_count: 25,
  rows: [
    { value: "Electronics", count: 30, pct: 31.6 },
    { value: "Clothing", count: 25, pct: 26.3 },
    { value: "Food", count: 20, pct: 21.1 },
  ],
  has_more: true,
  summary: "'product_category' has 25 unique values. Most common: 'Electronics' (31.6%). 5 nulls. Showing top 20 of 25.",
}

// ---------------------------------------------------------------------------
// Component rendering
// ---------------------------------------------------------------------------

describe("ValueCountCard rendering", () => {
  it("renders the column name in the header", () => {
    render(<ValueCountCard result={regionResult} />)
    expect(screen.getAllByText(/region/).length).toBeGreaterThan(0)
  })

  it("shows unique count badge", () => {
    render(<ValueCountCard result={regionResult} />)
    expect(screen.getAllByText(/4 unique/).length).toBeGreaterThan(0)
  })

  it("shows total rows count", () => {
    render(<ValueCountCard result={regionResult} />)
    expect(screen.getAllByText(/10/).length).toBeGreaterThan(0)
  })

  it("renders a row for each value", () => {
    render(<ValueCountCard result={regionResult} />)
    // Values rendered with typographic curly quotes
    expect(screen.getAllByText(/East/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/West/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/North/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/South/).length).toBeGreaterThan(0)
  })

  it("renders count values", () => {
    render(<ValueCountCard result={regionResult} />)
    // "4" appears as the count for East
    expect(screen.getAllByText(/\b4\b/).length).toBeGreaterThan(0)
  })

  it("renders percentage values", () => {
    render(<ValueCountCard result={regionResult} />)
    expect(screen.getByText("40%")).toBeInTheDocument()
  })

  it("renders summary footer", () => {
    render(<ValueCountCard result={regionResult} />)
    expect(screen.getByText(regionResult.summary)).toBeInTheDocument()
  })

  it("shows null count badge when nulls exist", () => {
    render(<ValueCountCard result={productResult} />)
    expect(screen.getAllByText(/5 null/).length).toBeGreaterThan(0)
  })

  it("does not show null badge when no nulls", () => {
    render(<ValueCountCard result={regionResult} />)
    // No "N null" badge — summary may say "non-null" so test for badge specifically
    expect(screen.queryByText(/^\d+ null$/)).not.toBeInTheDocument()
  })

  it("shows has_more notice when truncated", () => {
    render(<ValueCountCard result={productResult} />)
    expect(screen.getAllByText(/Showing top/i).length).toBeGreaterThan(0)
  })

  it("does not show has_more notice when all values shown", () => {
    render(<ValueCountCard result={regionResult} />)
    expect(screen.queryByText(/Showing top \d+ of/i)).not.toBeInTheDocument()
  })

  it("replaces underscores in column name with spaces", () => {
    render(<ValueCountCard result={productResult} />)
    expect(screen.getAllByText(/product category/).length).toBeGreaterThan(0)
  })

  it("has accessible region label with column name", () => {
    render(<ValueCountCard result={regionResult} />)
    expect(
      screen.getByRole("region", { name: /value frequency table for region/i })
    ).toBeInTheDocument()
  })

  it("renders table headers", () => {
    render(<ValueCountCard result={regionResult} />)
    expect(screen.getByText("Value")).toBeInTheDocument()
    expect(screen.getByText("Count")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Zustand store — value counts attachment
// ---------------------------------------------------------------------------

describe("Zustand store value counts attachment", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attachValueCountsToLastMessage links result to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Here are the most common values." },
      ],
    })

    useAppStore.getState().attachValueCountsToLastMessage(regionResult)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.value_counts).toBeDefined()
    expect(lastMsg.value_counts?.column).toBe("region")
    expect(lastMsg.value_counts?.unique_count).toBe(4)
    expect(lastMsg.value_counts?.rows).toHaveLength(4)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [{ id: "2", role: "user", content: "most common values in region" }],
    })

    useAppStore.getState().attachValueCountsToLastMessage(regionResult)

    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].value_counts).toBeUndefined()
  })

  it("does nothing when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    useAppStore.getState().attachValueCountsToLastMessage(regionResult)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })

  it("attaches result with nulls correctly", () => {
    useAppStore.setState({
      messages: [{ id: "3", role: "assistant", content: "Product categories:" }],
    })

    useAppStore.getState().attachValueCountsToLastMessage(productResult)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.value_counts?.null_count).toBe(5)
    expect(lastMsg.value_counts?.has_more).toBe(true)
  })
})
