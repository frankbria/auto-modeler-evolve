/**
 * Tests for TopNCard component, store action, and API client.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"
import { TopNCard } from "@/components/data/top-n-card"
import type { TopNResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"
import { api } from "@/lib/api"

fetchMock.enableMocks()

// --- Fixtures ----------------------------------------------------------------

const topResult: TopNResult = {
  sort_col: "revenue",
  direction: "top",
  ascending: false,
  n_requested: 5,
  n_returned: 5,
  total_rows: 50,
  display_cols: ["revenue", "customer", "region"],
  rows: [
    { _rank: 1, revenue: 9100.0, customer: "Frank", region: "East" },
    { _rank: 2, revenue: 8500.0, customer: "Carol", region: "East" },
    { _rank: 3, revenue: 6300.0, customer: "Hank", region: "North" },
    { _rank: 4, revenue: 5000.0, customer: "Alice", region: "East" },
    { _rank: 5, revenue: 4200.0, customer: "Eve", region: "West" },
  ],
  summary: "Top 5 records by revenue (highest: 9,100.00, lowest in this list: 4,200.00). Showing 5 of 50 total rows.",
}

const bottomResult: TopNResult = {
  sort_col: "revenue",
  direction: "bottom",
  ascending: true,
  n_requested: 3,
  n_returned: 3,
  total_rows: 50,
  display_cols: ["revenue", "customer"],
  rows: [
    { _rank: 1, revenue: 300.0, customer: "Dave" },
    { _rank: 2, revenue: 750.0, customer: "Grace" },
    { _rank: 3, revenue: 1200.0, customer: "Bob" },
  ],
  summary: "Bottom 3 records by revenue (lowest: 300.00, highest in this list: 1,200.00). Showing 3 of 50 total rows.",
}

// --- Component tests ---------------------------------------------------------

describe("TopNCard", () => {
  it("renders the direction label for top results", () => {
    render(<TopNCard result={topResult} />)
    expect(screen.getByText(/Highest 5 by revenue/i)).toBeInTheDocument()
  })

  it("renders the direction label for bottom results", () => {
    render(<TopNCard result={bottomResult} />)
    expect(screen.getByText(/Lowest 3 by revenue/i)).toBeInTheDocument()
  })

  it("shows row count and total", () => {
    render(<TopNCard result={topResult} />)
    expect(screen.getByText(/5 of 50 rows/i)).toBeInTheDocument()
  })

  it("renders medal emoji for top 3 ranks", () => {
    render(<TopNCard result={topResult} />)
    expect(screen.getByLabelText("1st place")).toBeInTheDocument()
    expect(screen.getByLabelText("2nd place")).toBeInTheDocument()
    expect(screen.getByLabelText("3rd place")).toBeInTheDocument()
  })

  it("shows rank number (not medal) for ranks > 3", () => {
    render(<TopNCard result={topResult} />)
    expect(screen.getByText("4")).toBeInTheDocument()
    expect(screen.getByText("5")).toBeInTheDocument()
  })

  it("renders column headers", () => {
    render(<TopNCard result={topResult} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
    expect(screen.getByText("customer")).toBeInTheDocument()
    expect(screen.getByText("region")).toBeInTheDocument()
  })

  it("renders row data", () => {
    render(<TopNCard result={topResult} />)
    expect(screen.getByText("Frank")).toBeInTheDocument()
    expect(screen.getByText("Carol")).toBeInTheDocument()
  })

  it("renders summary footer", () => {
    render(<TopNCard result={topResult} />)
    expect(screen.getByText(/Top 5 records by revenue/i)).toBeInTheDocument()
  })

  it("formats large numbers with k suffix", () => {
    render(<TopNCard result={topResult} />)
    // 9100 -> "9.1k"
    expect(screen.getByText("9.1k")).toBeInTheDocument()
  })

  it("replaces underscores in column names with spaces", () => {
    const resultWithUnderscore: TopNResult = {
      ...topResult,
      sort_col: "total_revenue",
      n_returned: 1,
      display_cols: ["total_revenue", "customer"],
      rows: [{ _rank: 1, total_revenue: 1000.0, customer: "Alice" }],
      summary: "Top 1 records by total revenue...",
    }
    render(<TopNCard result={resultWithUnderscore} />)
    expect(screen.getByText("Highest 1 by total revenue")).toBeInTheDocument()
  })

  it("shows null values as em dash", () => {
    const resultWithNull: TopNResult = {
      ...bottomResult,
      rows: [{ _rank: 1, revenue: null, customer: "Dave" }],
    }
    render(<TopNCard result={resultWithNull} />)
    expect(screen.getAllByText("—").length).toBeGreaterThan(0)
  })
})

// --- Zustand store tests -----------------------------------------------------

describe("attachTopNToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches top_n to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "top 5 customers", timestamp: "" },
        { role: "assistant", content: "Here are the top 5...", timestamp: "" },
      ],
    })
    useAppStore.getState().attachTopNToLastMessage(topResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].top_n).toEqual(topResult)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "top 5?", timestamp: "" }],
    })
    useAppStore.getState().attachTopNToLastMessage(topResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].top_n).toBeUndefined()
  })
})

// --- API client tests ---------------------------------------------------------

describe("api.data.getTopN", () => {
  beforeEach(() => fetchMock.resetMocks())

  it("calls correct URL with default params", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(topResult))
    await api.data.getTopN("ds-123", "revenue")
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/data/ds-123/top-n?")
    )
    const url = (fetchMock.mock.calls[0][0] as string)
    expect(url).toContain("col=revenue")
    expect(url).toContain("n=10")
    expect(url).toContain("order=desc")
  })

  it("passes ascending order when specified", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(bottomResult))
    await api.data.getTopN("ds-123", "revenue", 3, "asc")
    const url = (fetchMock.mock.calls[0][0] as string)
    expect(url).toContain("order=asc")
    expect(url).toContain("n=3")
  })

  it("throws on error response", async () => {
    fetchMock.mockResponseOnce("Bad Request", { status: 400 })
    await expect(api.data.getTopN("ds-123", "region")).rejects.toThrow("HTTP 400")
  })
})
