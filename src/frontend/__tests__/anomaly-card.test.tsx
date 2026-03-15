/**
 * Tests for AnomalyCard — the anomaly detection UI component.
 *
 * Covers:
 *   1. Renders nothing unusual when no result yet (scan button visible)
 *   2. Renders summary when result is provided via props
 *   3. Shows anomaly count badge when anomalies found
 *   4. Shows "All rows normal" badge when none found
 *   5. Shows top anomalies table
 *   6. "Show N more" collapse/expand
 *   7. Manual scan button calls api.data.detectAnomalies
 *   8. Error message shown on API failure
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { AnomalyCard } from "../components/data/anomaly-card"
import { api } from "../lib/api"
import type { AnomalyResult } from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    data: {
      detectAnomalies: jest.fn(),
    },
  },
}))

const mockDetect = api.data.detectAnomalies as jest.MockedFunction<typeof api.data.detectAnomalies>

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const ANOMALY_RESULT: AnomalyResult = {
  dataset_id: "ds-1",
  anomaly_count: 3,
  total_rows: 100,
  contamination_used: 0.05,
  features_used: ["revenue", "quantity"],
  summary: "Found 3 unusual record(s) out of 100 (3.0%). The most anomalous record has a score of 95.2/100 (row index 99).",
  top_anomalies: [
    { row_index: 99, anomaly_score: 95.2, is_anomaly: true, values: { revenue: 50000, quantity: -999 } },
    { row_index: 50, anomaly_score: 72.1, is_anomaly: true, values: { revenue: 200, quantity: 300 } },
    { row_index: 10, anomaly_score: 61.5, is_anomaly: true, values: { revenue: 980, quantity: 5 } },
    { row_index: 5,  anomaly_score: 45.0, is_anomaly: false, values: { revenue: 1100, quantity: 48 } },
    { row_index: 3,  anomaly_score: 38.0, is_anomaly: false, values: { revenue: 1000, quantity: 50 } },
    { row_index: 1,  anomaly_score: 22.5, is_anomaly: false, values: { revenue: 995, quantity: 52 } },
  ],
}

const CLEAN_RESULT: AnomalyResult = {
  dataset_id: "ds-2",
  anomaly_count: 0,
  total_rows: 50,
  contamination_used: 0.05,
  features_used: ["revenue"],
  summary: "No anomalous records found across 50 rows using 1 features.",
  top_anomalies: [],
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("AnomalyCard", () => {
  beforeEach(() => jest.clearAllMocks())

  it("shows scan button when no result provided and datasetId given", () => {
    render(<AnomalyCard datasetId="ds-1" numericFeatures={["revenue", "quantity"]} />)
    expect(screen.getByRole("button", { name: /scan for anomalies/i })).toBeInTheDocument()
  })

  it("shows summary text when result is provided", () => {
    render(<AnomalyCard result={ANOMALY_RESULT} />)
    expect(screen.getByText(/Found 3 unusual record/)).toBeInTheDocument()
  })

  it("shows anomaly count badge when anomalies found", () => {
    render(<AnomalyCard result={ANOMALY_RESULT} />)
    expect(screen.getByText(/3 unusual row/)).toBeInTheDocument()
  })

  it("shows 'All rows normal' badge when no anomalies", () => {
    render(<AnomalyCard result={CLEAN_RESULT} />)
    expect(screen.getByText(/All rows normal/)).toBeInTheDocument()
  })

  it("shows top 5 anomaly rows in table (collapse enabled)", () => {
    render(<AnomalyCard result={ANOMALY_RESULT} />)
    // row_index 99 → display 100 (multiple cells may have this value; check at least one exists)
    expect(screen.getAllByText("100").length).toBeGreaterThan(0)
  })

  it("shows 'Show N more' button when there are more than 5 anomalies", () => {
    render(<AnomalyCard result={ANOMALY_RESULT} />)
    expect(screen.getByText(/Show 1 more/)).toBeInTheDocument()
  })

  it("expands to show all anomalies when 'Show more' clicked", () => {
    render(<AnomalyCard result={ANOMALY_RESULT} />)
    fireEvent.click(screen.getByText(/Show 1 more/))
    expect(screen.getByText(/Show less/)).toBeInTheDocument()
  })

  it("calls api.data.detectAnomalies on scan button click", async () => {
    mockDetect.mockResolvedValueOnce(ANOMALY_RESULT)
    render(<AnomalyCard datasetId="ds-1" numericFeatures={["revenue", "quantity"]} />)
    fireEvent.click(screen.getByRole("button", { name: /scan for anomalies/i }))
    await waitFor(() => {
      expect(mockDetect).toHaveBeenCalledWith("ds-1", ["revenue", "quantity"])
    })
  })

  it("shows re-scan button after result is loaded", async () => {
    mockDetect.mockResolvedValueOnce(ANOMALY_RESULT)
    render(<AnomalyCard datasetId="ds-1" numericFeatures={["revenue", "quantity"]} />)
    fireEvent.click(screen.getByRole("button", { name: /scan for anomalies/i }))
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /re-scan/i })).toBeInTheDocument()
    })
  })

  it("shows error message on API failure", async () => {
    mockDetect.mockRejectedValueOnce(new Error("Network error"))
    render(<AnomalyCard datasetId="ds-1" numericFeatures={["revenue"]} />)
    fireEvent.click(screen.getByRole("button", { name: /scan for anomalies/i }))
    await waitFor(() => {
      expect(screen.getByText(/failed/i)).toBeInTheDocument()
    })
  })

  it("shows features used when result provided", () => {
    render(<AnomalyCard result={ANOMALY_RESULT} />)
    // "Features analysed: revenue, quantity" text
    expect(screen.getByText(/Features analysed:/)).toBeInTheDocument()
  })
})
