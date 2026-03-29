/**
 * Tests for DataExportCard component and Zustand store integration.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { DataExportCard } from "@/components/data/data-export-card"
import type { DataExportResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const exportResult: DataExportResult = {
  dataset_id: "ds-001",
  filename: "sales.csv",
  row_count: 150,
  filtered: false,
  download_url: "/api/data/ds-001/download",
}

const filteredExportResult: DataExportResult = {
  dataset_id: "ds-002",
  filename: "sales_filtered.csv",
  row_count: 42,
  filtered: true,
  download_url: "/api/data/ds-002/download",
}

// ---------------------------------------------------------------------------
// Component rendering
// ---------------------------------------------------------------------------

describe("DataExportCard rendering", () => {
  it("renders the export ready header", () => {
    render(<DataExportCard result={exportResult} />)
    expect(screen.getByText("Dataset Export Ready")).toBeInTheDocument()
  })

  it("shows filename and row count", () => {
    render(<DataExportCard result={exportResult} />)
    expect(screen.getByText(/sales\.csv/)).toBeInTheDocument()
    expect(screen.getByText(/150/)).toBeInTheDocument()
  })

  it("renders Download CSV link", () => {
    render(<DataExportCard result={exportResult} />)
    const link = screen.getByRole("link", { name: /download csv/i })
    expect(link).toBeInTheDocument()
    expect(link).toHaveAttribute("href", "/api/data/ds-001/download")
  })

  it("does not show Filtered badge when not filtered", () => {
    render(<DataExportCard result={exportResult} />)
    expect(screen.queryByText("Filtered")).not.toBeInTheDocument()
  })

  it("shows Filtered badge when filtered", () => {
    render(<DataExportCard result={filteredExportResult} />)
    expect(screen.getByText("Filtered")).toBeInTheDocument()
  })

  it("shows filtered row count and note", () => {
    render(<DataExportCard result={filteredExportResult} />)
    expect(screen.getByText(/42/)).toBeInTheDocument()
    expect(screen.getByText(/active filter applied/)).toBeInTheDocument()
  })

  it("uses singular 'row' for single row count", () => {
    const singleRow: DataExportResult = { ...exportResult, row_count: 1 }
    render(<DataExportCard result={singleRow} />)
    // "1 row" without trailing "s"
    expect(screen.getByText(/\b1 row\b/)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Zustand store — data export attachment
// ---------------------------------------------------------------------------

describe("Zustand store data export attachment", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attachDataExportToLastMessage links export to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Your data export is ready." },
      ],
    })

    useAppStore.getState().attachDataExportToLastMessage(exportResult)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.data_export).toBeDefined()
    expect(lastMsg.data_export?.filename).toBe("sales.csv")
    expect(lastMsg.data_export?.row_count).toBe(150)
    expect(lastMsg.data_export?.filtered).toBe(false)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [{ id: "2", role: "user", content: "download my data" }],
    })

    useAppStore.getState().attachDataExportToLastMessage(exportResult)

    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].data_export).toBeUndefined()
  })

  it("does nothing when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    useAppStore.getState().attachDataExportToLastMessage(exportResult)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })

  it("attachDataExportToLastMessage works with filtered result", () => {
    useAppStore.setState({
      messages: [
        { id: "3", role: "assistant", content: "Filtered export ready." },
      ],
    })

    useAppStore.getState().attachDataExportToLastMessage(filteredExportResult)

    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1]
    expect(lastMsg.data_export?.filtered).toBe(true)
    expect(lastMsg.data_export?.row_count).toBe(42)
  })
})

// ---------------------------------------------------------------------------
// API — downloadDatasetUrl helper
// ---------------------------------------------------------------------------

import { api } from "@/lib/api"

describe("api.data.downloadDatasetUrl", () => {
  it("constructs the correct URL", () => {
    const url = api.data.downloadDatasetUrl("abc-123")
    expect(url).toContain("/api/data/abc-123/download")
  })
})
