/**
 * Tests for TimeWindowCard component and related store/API plumbing.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"
import { TimeWindowCard } from "@/components/data/time-window-card"
import type { TimeWindowComparison } from "@/lib/types"
import { useAppStore } from "@/lib/store"
import { api } from "@/lib/api"

fetchMock.enableMocks()

// --- Fixtures ---------------------------------------------------------------

const basicResult: TimeWindowComparison = {
  date_col: "date",
  period1: { name: "2023", start: "2023-01-01", end: "2023-12-31", row_count: 4 },
  period2: { name: "2024", start: "2024-01-01", end: "2024-12-31", row_count: 4 },
  columns: [
    {
      column: "revenue",
      p1_mean: 1050.0,
      p2_mean: 1525.0,
      pct_change: 45.2,
      direction: "up",
      notable: true,
    },
    {
      column: "units",
      p1_mean: 10.5,
      p2_mean: 15.25,
      pct_change: 45.2,
      direction: "up",
      notable: true,
    },
    {
      column: "cost",
      p1_mean: 525.0,
      p2_mean: 762.5,
      pct_change: 45.2,
      direction: "up",
      notable: true,
    },
  ],
  notable_changes: ["revenue", "units", "cost"],
  summary: "Comparing 2023 (4 rows) vs 2024 (4 rows) across 3 metrics. Biggest change: revenue increased by 45% (1050.00 → 1525.00). All tracked metrics improved in 2024.",
}

const noNotableResult: TimeWindowComparison = {
  date_col: "date",
  period1: { name: "Q1 2024", start: "2024-01-01", end: "2024-03-31", row_count: 3 },
  period2: { name: "Q2 2024", start: "2024-04-01", end: "2024-06-30", row_count: 3 },
  columns: [
    {
      column: "revenue",
      p1_mean: 1000.0,
      p2_mean: 1020.0,
      pct_change: 2.0,
      direction: "up",
      notable: false,
    },
    {
      column: "units",
      p1_mean: 10.0,
      p2_mean: 9.8,
      pct_change: -2.0,
      direction: "down",
      notable: false,
    },
  ],
  notable_changes: [],
  summary: "Comparing Q1 2024 (3 rows) vs Q2 2024 (3 rows) across 2 metrics. Q2 2024 is broadly similar to Q1 2024 — no metrics changed by more than 20%.",
}

const flatResult: TimeWindowComparison = {
  date_col: "date",
  period1: { name: "H1 2024", start: "2024-01-01", end: "2024-06-30", row_count: 5 },
  period2: { name: "H2 2024", start: "2024-07-01", end: "2024-12-31", row_count: 5 },
  columns: [
    {
      column: "revenue",
      p1_mean: 1000.0,
      p2_mean: 1000.0,
      pct_change: 0.0,
      direction: "flat",
      notable: false,
    },
  ],
  notable_changes: [],
  summary: "Comparing H1 2024 (5 rows) vs H2 2024 (5 rows) across 1 metric.",
}

// --- Component tests --------------------------------------------------------

describe("TimeWindowCard", () => {
  it("renders period names in header", () => {
    render(<TimeWindowCard result={basicResult} />)
    expect(screen.getByText("Period Comparison")).toBeInTheDocument()
    expect(screen.getByText(/2023 vs 2024/)).toBeInTheDocument()
  })

  it("renders period chips with row counts", () => {
    render(<TimeWindowCard result={basicResult} />)
    expect(screen.getAllByText("4 rows").length).toBeGreaterThan(0)
  })

  it("renders all metric rows", () => {
    render(<TimeWindowCard result={basicResult} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
    expect(screen.getByText("units")).toBeInTheDocument()
    expect(screen.getByText("cost")).toBeInTheDocument()
  })

  it("renders up/down badges in header", () => {
    render(<TimeWindowCard result={basicResult} />)
    expect(screen.getByText(/↑ 3 up/)).toBeInTheDocument()
  })

  it("renders notable changes callout when present", () => {
    render(<TimeWindowCard result={basicResult} />)
    expect(screen.getByText(/Notable changes/)).toBeInTheDocument()
    expect(screen.getAllByText(/revenue/).length).toBeGreaterThan(0)
  })

  it("does not render notable callout when no notable changes", () => {
    render(<TimeWindowCard result={noNotableResult} />)
    expect(screen.queryByText(/Notable changes/)).not.toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<TimeWindowCard result={basicResult} />)
    expect(screen.getByText(/Comparing 2023/)).toBeInTheDocument()
  })

  it("renders up arrows for increasing metrics", () => {
    render(<TimeWindowCard result={basicResult} />)
    const arrows = screen.getAllByText(/↑/)
    expect(arrows.length).toBeGreaterThan(0)
  })

  it("renders down arrows for decreasing metrics", () => {
    render(<TimeWindowCard result={noNotableResult} />)
    const down = screen.getAllByText(/↓/)
    expect(down.length).toBeGreaterThan(0)
  })

  it("renders flat indicator for unchanged metrics", () => {
    render(<TimeWindowCard result={flatResult} />)
    expect(screen.getByText(/→ 0%/)).toBeInTheDocument()
  })

  it("renders table column headers with period names", () => {
    render(<TimeWindowCard result={basicResult} />)
    // Check table headers contain period names
    const headers = screen.getAllByText("2023")
    expect(headers.length).toBeGreaterThan(0)
    const headers2 = screen.getAllByText("2024")
    expect(headers2.length).toBeGreaterThan(0)
  })

  it("renders notable row highlight for notable columns", () => {
    render(<TimeWindowCard result={basicResult} />)
    // "notable" text appears next to notable columns
    expect(screen.getAllByText("notable").length).toBeGreaterThan(0)
  })
})

// --- Store tests ------------------------------------------------------------

describe("attachTimeWindowToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches time_window_comparison to the last assistant message", () => {
    useAppStore.setState({
      messages: [{ role: "assistant", content: "Hello", timestamp: "" }],
    })
    useAppStore.getState().attachTimeWindowToLastMessage(basicResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].time_window_comparison).toEqual(basicResult)
  })

  it("does not attach to a user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "Compare 2023 vs 2024", timestamp: "" }],
    })
    useAppStore.getState().attachTimeWindowToLastMessage(basicResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].time_window_comparison).toBeUndefined()
  })

  it("does nothing on empty messages array", () => {
    useAppStore.getState().attachTimeWindowToLastMessage(basicResult)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })
})

// --- API client tests -------------------------------------------------------

describe("api.data.compareTimeWindows", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
  })

  it("calls the correct URL with all parameters", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(basicResult))
    await api.data.compareTimeWindows(
      "ds-123", "date", "2023", "2023-01-01", "2023-12-31", "2024", "2024-01-01", "2024-12-31"
    )
    const url = fetchMock.mock.calls[0][0] as string
    expect(url).toContain("/api/data/ds-123/compare-time-windows")
    expect(url).toContain("date_col=date")
    expect(url).toContain("p1_name=2023")
    expect(url).toContain("p2_name=2024")
    expect(url).toContain("p1_start=2023-01-01")
    expect(url).toContain("p2_end=2024-12-31")
  })

  it("throws on non-ok response", async () => {
    fetchMock.mockResponseOnce("error", { status: 400 })
    await expect(
      api.data.compareTimeWindows(
        "ds-123", "date", "2023", "2023-01-01", "2023-12-31", "2024", "2024-01-01", "2024-12-31"
      )
    ).rejects.toThrow("HTTP 400")
  })
})
