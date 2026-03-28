import "@testing-library/jest-dom"
import { render, screen } from "@testing-library/react"
import { RecordTableCard } from "@/components/data/record-table-card"
import type { RecordTableResult } from "@/lib/types"

const baseResult: RecordTableResult = {
  columns: ["customer", "revenue", "region"],
  rows: [
    { customer: "Alice", revenue: 5000.0, region: "East" },
    { customer: "Bob", revenue: 1200.0, region: "West" },
    { customer: "Carol", revenue: 8500.5, region: "East" },
  ],
  total_rows: 10,
  filtered_rows: 10,
  shown_rows: 3,
  filtered: false,
  condition_summary: "",
  summary: "Showing 3 of 10 rows.",
}

describe("RecordTableCard", () => {
  it("renders Data Preview header", () => {
    render(<RecordTableCard result={baseResult} />)
    expect(screen.getByText("Data Preview")).toBeInTheDocument()
  })

  it("renders column headers", () => {
    render(<RecordTableCard result={baseResult} />)
    expect(screen.getByText("customer")).toBeInTheDocument()
    expect(screen.getByText("revenue")).toBeInTheDocument()
    expect(screen.getByText("region")).toBeInTheDocument()
  })

  it("renders row values", () => {
    render(<RecordTableCard result={baseResult} />)
    expect(screen.getByText("Alice")).toBeInTheDocument()
    expect(screen.getByText("Bob")).toBeInTheDocument()
    expect(screen.getAllByText("East").length).toBeGreaterThan(0)
  })

  it("shows footer with correct row count", () => {
    render(<RecordTableCard result={baseResult} />)
    expect(screen.getByText(/Showing 3 of 10/)).toBeInTheDocument()
  })

  it("does not show filter badge when not filtered", () => {
    render(<RecordTableCard result={baseResult} />)
    expect(screen.queryByText("filtered")).not.toBeInTheDocument()
  })

  it("shows filter badge when filtered", () => {
    const filtered: RecordTableResult = {
      ...baseResult,
      filtered: true,
      condition_summary: "region = East",
      filtered_rows: 3,
      summary: "Found 3 matching rows.",
    }
    render(<RecordTableCard result={filtered} />)
    expect(screen.getByText("filtered")).toBeInTheDocument()
  })

  it("shows condition summary when filtered", () => {
    const filtered: RecordTableResult = {
      ...baseResult,
      filtered: true,
      condition_summary: "region = East",
      filtered_rows: 3,
    }
    render(<RecordTableCard result={filtered} />)
    expect(screen.getByText("region = East")).toBeInTheDocument()
  })

  it("shows empty state when no rows match", () => {
    const empty: RecordTableResult = {
      ...baseResult,
      rows: [],
      shown_rows: 0,
      filtered: true,
      filtered_rows: 0,
      condition_summary: "region = Antarctica",
      summary: "No rows match: region = Antarctica.",
    }
    render(<RecordTableCard result={empty} />)
    expect(screen.getByText(/No rows match/)).toBeInTheDocument()
  })

  it("formats null values as em dash", () => {
    const withNull: RecordTableResult = {
      ...baseResult,
      rows: [{ customer: null, revenue: 100.0, region: "East" }],
      shown_rows: 1,
    }
    render(<RecordTableCard result={withNull} />)
    const dashes = screen.getAllByText("—")
    expect(dashes.length).toBeGreaterThan(0)
  })

  it("truncates long string values", () => {
    const longStr = "A".repeat(40)
    const withLong: RecordTableResult = {
      ...baseResult,
      rows: [{ customer: longStr, revenue: 100, region: "East" }],
      shown_rows: 1,
    }
    render(<RecordTableCard result={withLong} />)
    // Should be truncated to 28 chars + "…"
    expect(screen.getByTitle(longStr)).toBeInTheDocument()
    const cell = screen.getByTitle(longStr)
    expect(cell.textContent).toMatch(/…$/)
  })

  it("renders columns count badge", () => {
    render(<RecordTableCard result={baseResult} />)
    expect(screen.getByText("3 columns")).toBeInTheDocument()
  })

  it("shows filtered row count in footer when filtered", () => {
    const filtered: RecordTableResult = {
      ...baseResult,
      filtered: true,
      filtered_rows: 3,
      condition_summary: "region = East",
    }
    render(<RecordTableCard result={filtered} />)
    expect(screen.getByText(/3 matching rows/)).toBeInTheDocument()
  })
})

// Store action test
describe("attachRecordsToLastMessage store action", () => {
  it("attaches records to the last assistant message", () => {
    const { useAppStore } = require("@/lib/store")
    const store = useAppStore.getState()

    store.setMessages([
      { role: "user", content: "show me the data" },
      { role: "assistant", content: "Here are some rows:" },
    ])

    store.attachRecordsToLastMessage(baseResult)

    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.records).toBeDefined()
    expect(last.records!.shown_rows).toBe(3)
  })

  it("does not attach to user message", () => {
    const { useAppStore } = require("@/lib/store")
    const store = useAppStore.getState()

    store.setMessages([{ role: "user", content: "show me the data" }])
    store.attachRecordsToLastMessage(baseResult)

    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.records).toBeUndefined()
  })
})

// API client test
describe("api.data.getRecords", () => {
  it("builds correct URL with n and where params", async () => {
    const fetchMock = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => baseResult,
    })
    global.fetch = fetchMock

    const { api } = require("@/lib/api")
    await api.data.getRecords("ds-1", 15, "region = East", 0)

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/data/ds-1/records")
    )
    const url: string = fetchMock.mock.calls[0][0]
    expect(url).toContain("n=15")
    expect(url).toContain("where=")
  })

  it("throws on non-ok response", async () => {
    global.fetch = jest.fn().mockResolvedValue({ ok: false, status: 404 })
    const { api } = require("@/lib/api")
    await expect(api.data.getRecords("bad-id")).rejects.toThrow("HTTP 404")
  })
})
