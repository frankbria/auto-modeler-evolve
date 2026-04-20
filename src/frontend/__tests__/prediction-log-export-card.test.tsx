/**
 * Tests for PredictionLogExportCard — inline chat card for downloading prediction history.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { PredictionLogExportCard } from "../components/deploy/prediction-log-export-card"
import type { PredictionLogExportResult } from "../lib/types"
import { useAppStore } from "../lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const EMPTY_RESULT: PredictionLogExportResult = {
  deployment_id: "dep-1",
  total_predictions: 0,
  download_url: "/api/deploy/dep-1/prediction-logs/export",
  first_prediction_at: null,
  last_prediction_at: null,
}

const WITH_DATA: PredictionLogExportResult = {
  deployment_id: "dep-2",
  total_predictions: 42,
  download_url: "/api/deploy/dep-2/prediction-logs/export",
  first_prediction_at: "2024-01-01T09:00:00",
  last_prediction_at: "2024-03-15T14:30:00",
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("PredictionLogExportCard", () => {
  it("has correct aria-label on card container", () => {
    render(<PredictionLogExportCard result={WITH_DATA} />)
    const card = document.querySelector('[aria-label="Prediction log export"]')
    expect(card).toBeTruthy()
  })

  it("shows heading with export icon", () => {
    render(<PredictionLogExportCard result={WITH_DATA} />)
    const headings = screen.getAllByText(/Prediction Log Export/i)
    expect(headings.length).toBeGreaterThanOrEqual(1)
  })

  it("shows total predictions count badge", () => {
    render(<PredictionLogExportCard result={WITH_DATA} />)
    expect(screen.getByText(/42 predictions/i)).toBeInTheDocument()
  })

  it("shows CSV badge", () => {
    render(<PredictionLogExportCard result={WITH_DATA} />)
    expect(screen.getByText("CSV")).toBeInTheDocument()
  })

  it("shows date range when predictions exist", () => {
    render(<PredictionLogExportCard result={WITH_DATA} />)
    // Both date labels should appear
    expect(screen.getByText(/First prediction/i)).toBeInTheDocument()
    expect(screen.getByText(/Last prediction/i)).toBeInTheDocument()
  })

  it("renders download link with correct href", () => {
    render(<PredictionLogExportCard result={WITH_DATA} />)
    const link = screen.getByRole("link", { name: /download.*prediction.*csv/i })
    expect(link).toHaveAttribute("href", WITH_DATA.download_url)
  })

  it("download link has download attribute", () => {
    render(<PredictionLogExportCard result={WITH_DATA} />)
    const link = screen.getByRole("link", { name: /download.*csv/i })
    expect(link).toHaveAttribute("download")
  })

  it("shows empty state when no predictions", () => {
    render(<PredictionLogExportCard result={EMPTY_RESULT} />)
    expect(screen.getByText(/No predictions recorded yet/i)).toBeInTheDocument()
  })

  it("does not show download link in empty state", () => {
    render(<PredictionLogExportCard result={EMPTY_RESULT} />)
    expect(screen.queryByRole("link")).not.toBeInTheDocument()
  })

  it("singular 'prediction' label for count of 1", () => {
    const single: PredictionLogExportResult = {
      ...WITH_DATA,
      total_predictions: 1,
    }
    render(<PredictionLogExportCard result={single} />)
    expect(screen.getByText(/1 prediction$/)).toBeInTheDocument()
  })

  it("shows sr-only figcaption with count for accessibility", () => {
    render(<PredictionLogExportCard result={WITH_DATA} />)
    const caption = document.querySelector("figcaption.sr-only")
    expect(caption).toBeTruthy()
    expect(caption?.textContent).toMatch(/42/)
  })

  it("shows sr-only empty state figcaption", () => {
    render(<PredictionLogExportCard result={EMPTY_RESULT} />)
    const caption = document.querySelector("figcaption.sr-only")
    expect(caption).toBeTruthy()
    expect(caption?.textContent).toMatch(/no records/i)
  })

  it("mentions spreadsheet-ready format in description", () => {
    render(<PredictionLogExportCard result={WITH_DATA} />)
    expect(screen.getByText(/spreadsheet-ready/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action test
// ---------------------------------------------------------------------------

describe("attachPredictionLogExportToLastMessage store action", () => {
  it("attaches result to last assistant message", () => {
    const store = useAppStore.getState()
    store.setMessages([
      { id: "1", role: "user", content: "export prediction history", timestamp: new Date().toISOString() },
      { id: "2", role: "assistant", content: "Here is your download link.", timestamp: new Date().toISOString() },
    ])

    store.attachPredictionLogExportToLastMessage(WITH_DATA)

    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.prediction_log_export).toEqual(WITH_DATA)
  })

  it("does not attach to user messages", () => {
    const store = useAppStore.getState()
    store.setMessages([
      { id: "1", role: "user", content: "export prediction history", timestamp: new Date().toISOString() },
    ])

    store.attachPredictionLogExportToLastMessage(WITH_DATA)

    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.prediction_log_export).toBeUndefined()
  })
})
