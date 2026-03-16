/**
 * Tests for CleaningCard — the conversational data cleaning UI component.
 *
 * Covers:
 * - Shows quality summary (duplicates + missing value counts)
 * - Shows suggested operation from SSE data
 * - Apply button calls api.data.clean and shows result
 * - Error state when API fails
 * - No specific op → shows help hint
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { CleaningCard } from "../components/data/cleaning-card"
import { api } from "../lib/api"
import type { CleaningSuggestion, CleanResult } from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    data: {
      clean: jest.fn(),
    },
  },
}))

const mockClean = api.data.clean as jest.MockedFunction<typeof api.data.clean>

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const SUGGESTION_WITH_OP: CleaningSuggestion = {
  dataset_id: "ds-1",
  suggested_operation: {
    operation: "fill_missing",
    column: "revenue",
    strategy: "median",
  },
  quality_summary: {
    duplicate_rows: 2,
    missing_value_columns: { revenue: 3, quantity: 1 },
    total_rows: 100,
  },
}

const SUGGESTION_NO_OP: CleaningSuggestion = {
  dataset_id: "ds-1",
  suggested_operation: null,
  quality_summary: {
    duplicate_rows: 0,
    missing_value_columns: {},
    total_rows: 100,
  },
}

const CLEAN_RESULT: CleanResult = {
  dataset_id: "ds-1",
  operation_result: {
    operation: "fill_missing",
    column: "revenue",
    strategy: "median",
    fill_value_used: "median (150)",
    before_rows: 100,
    after_rows: 100,
    modified_count: 3,
    summary: "Filled 3 missing value(s) in 'revenue' with median (150). Column is now complete.",
  },
  preview: [],
  updated_stats: {
    row_count: 100,
    column_count: 5,
    columns: [],
  },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("CleaningCard", () => {
  beforeEach(() => jest.clearAllMocks())

  it("shows duplicate count when duplicates found", () => {
    render(<CleaningCard suggestion={SUGGESTION_WITH_OP} datasetId="ds-1" />)
    expect(screen.getByText(/2/)).toBeInTheDocument()
    expect(screen.getByText(/duplicate row/)).toBeInTheDocument()
  })

  it("shows missing value columns", () => {
    render(<CleaningCard suggestion={SUGGESTION_WITH_OP} datasetId="ds-1" />)
    expect(screen.getAllByText(/revenue/).length).toBeGreaterThan(0)
  })

  it("shows 'Issues found' badge when issues exist", () => {
    render(<CleaningCard suggestion={SUGGESTION_WITH_OP} datasetId="ds-1" />)
    expect(screen.getByText(/Issues found/)).toBeInTheDocument()
  })

  it("shows 'Data looks clean' badge when no issues", () => {
    render(<CleaningCard suggestion={SUGGESTION_NO_OP} datasetId="ds-1" />)
    expect(screen.getByText(/Data looks clean/)).toBeInTheDocument()
  })

  it("shows suggested operation description", () => {
    render(<CleaningCard suggestion={SUGGESTION_WITH_OP} datasetId="ds-1" />)
    // The description text includes the column name — uniquely identifies it
    expect(screen.getByText(/Fill missing values in 'revenue'/)).toBeInTheDocument()
  })

  it("shows Apply button when suggested operation and datasetId provided", () => {
    render(<CleaningCard suggestion={SUGGESTION_WITH_OP} datasetId="ds-1" />)
    expect(screen.getByRole("button", { name: /Apply/i })).toBeInTheDocument()
  })

  it("calls api.data.clean when Apply button clicked", async () => {
    mockClean.mockResolvedValueOnce(CLEAN_RESULT)
    render(<CleaningCard suggestion={SUGGESTION_WITH_OP} datasetId="ds-1" />)
    fireEvent.click(screen.getByRole("button", { name: /Apply/i }))
    await waitFor(() => {
      expect(mockClean).toHaveBeenCalledWith("ds-1", SUGGESTION_WITH_OP.suggested_operation)
    })
  })

  it("shows success result after clean applied", async () => {
    mockClean.mockResolvedValueOnce(CLEAN_RESULT)
    render(<CleaningCard suggestion={SUGGESTION_WITH_OP} datasetId="ds-1" />)
    fireEvent.click(screen.getByRole("button", { name: /Apply/i }))
    await waitFor(() => {
      expect(screen.getByText(/Done!/)).toBeInTheDocument()
      expect(screen.getByText(/Filled 3 missing/)).toBeInTheDocument()
    })
  })

  it("shows error message when API fails", async () => {
    mockClean.mockRejectedValueOnce(new Error("Network error"))
    render(<CleaningCard suggestion={SUGGESTION_WITH_OP} datasetId="ds-1" />)
    fireEvent.click(screen.getByRole("button", { name: /Apply/i }))
    await waitFor(() => {
      expect(screen.getByText(/Cleaning operation failed/)).toBeInTheDocument()
    })
  })

  it("calls onCleaned callback after success", async () => {
    mockClean.mockResolvedValueOnce(CLEAN_RESULT)
    const onCleaned = jest.fn()
    render(<CleaningCard suggestion={SUGGESTION_WITH_OP} datasetId="ds-1" onCleaned={onCleaned} />)
    fireEvent.click(screen.getByRole("button", { name: /Apply/i }))
    await waitFor(() => {
      expect(onCleaned).toHaveBeenCalledWith(CLEAN_RESULT)
    })
  })

  it("shows hint when no specific operation detected", () => {
    render(<CleaningCard suggestion={SUGGESTION_NO_OP} datasetId="ds-1" />)
    expect(screen.getByText(/fill missing/i)).toBeInTheDocument()
  })

  it("does not show Apply button when no suggested operation", () => {
    render(<CleaningCard suggestion={SUGGESTION_NO_OP} datasetId="ds-1" />)
    expect(screen.queryByRole("button", { name: /Apply/i })).not.toBeInTheDocument()
  })
})
