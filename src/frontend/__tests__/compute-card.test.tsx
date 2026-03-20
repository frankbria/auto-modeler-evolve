/**
 * Tests for ComputeCard — the computed column suggestion component.
 *
 * Covers:
 *   1. Renders with data-testid
 *   2. Shows column name and expression
 *   3. Shows sample preview values
 *   4. Shows dtype
 *   5. Shows "Add column" apply button
 *   6. Calls api.data.computeColumn on click
 *   7. Shows success state after apply
 *   8. Shows error state on failure
 *   9. "New column" badge is visible
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { ComputeCard } from "../components/data/compute-card"
import type { ComputedColumnSuggestion } from "../lib/types"

// Mock api module
jest.mock("../lib/api", () => ({
  api: {
    data: {
      computeColumn: jest.fn(),
    },
  },
}))

import { api } from "../lib/api"
const mockComputeColumn = api.data.computeColumn as jest.MockedFunction<
  typeof api.data.computeColumn
>

const BASE_SUGGESTION: ComputedColumnSuggestion = {
  dataset_id: "ds-123",
  name: "margin",
  expression: "revenue - cost",
  sample_values: [400, 250, 700, 150, 550],
  dtype: "float64",
}

const MOCK_RESULT = {
  dataset_id: "ds-123",
  compute_result: {
    column_name: "margin",
    expression: "revenue - cost",
    dtype: "float64",
    sample_values: [400, 250, 700, 150, 550],
    row_count: 5,
    column_count: 6,
    action: "added" as const,
    summary: "Added new column 'margin' = revenue - cost. First values: 400, 250, 700.",
  },
  preview: [],
  updated_stats: { row_count: 5, column_count: 6 },
}

describe("ComputeCard", () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it("renders with data-testid", () => {
    render(<ComputeCard suggestion={BASE_SUGGESTION} />)
    expect(screen.getByTestId("compute-card")).toBeInTheDocument()
  })

  it("shows column name", () => {
    render(<ComputeCard suggestion={BASE_SUGGESTION} />)
    expect(screen.getByText("margin")).toBeInTheDocument()
  })

  it("shows expression", () => {
    render(<ComputeCard suggestion={BASE_SUGGESTION} />)
    expect(screen.getByText(/revenue - cost/i)).toBeInTheDocument()
  })

  it("shows sample values", () => {
    render(<ComputeCard suggestion={BASE_SUGGESTION} />)
    // At least one sample value should be visible
    expect(screen.getByText("400")).toBeInTheDocument()
  })

  it("shows dtype in preview label", () => {
    render(<ComputeCard suggestion={BASE_SUGGESTION} />)
    expect(screen.getByText(/float64/i)).toBeInTheDocument()
  })

  it("shows Apply button with column name", () => {
    render(<ComputeCard suggestion={BASE_SUGGESTION} />)
    expect(screen.getByRole("button", { name: /add column 'margin'/i })).toBeInTheDocument()
  })

  it("shows 'New column' badge", () => {
    render(<ComputeCard suggestion={BASE_SUGGESTION} />)
    expect(screen.getByText("New column")).toBeInTheDocument()
  })

  it("calls computeColumn with correct args on Apply", async () => {
    mockComputeColumn.mockResolvedValueOnce(MOCK_RESULT)
    render(<ComputeCard suggestion={BASE_SUGGESTION} />)

    fireEvent.click(screen.getByRole("button", { name: /add column/i }))

    await waitFor(() => {
      expect(mockComputeColumn).toHaveBeenCalledWith("ds-123", "margin", "revenue - cost")
    })
  })

  it("shows success state after apply", async () => {
    mockComputeColumn.mockResolvedValueOnce(MOCK_RESULT)
    render(<ComputeCard suggestion={BASE_SUGGESTION} />)

    fireEvent.click(screen.getByRole("button", { name: /add column/i }))

    await waitFor(() => {
      expect(screen.getByText("Column added!")).toBeInTheDocument()
    })
  })

  it("shows error state on failure", async () => {
    mockComputeColumn.mockRejectedValueOnce(new Error("API error"))
    render(<ComputeCard suggestion={BASE_SUGGESTION} />)

    fireEvent.click(screen.getByRole("button", { name: /add column/i }))

    await waitFor(() => {
      expect(screen.getByText(/failed to add/i)).toBeInTheDocument()
    })
  })

  it("calls onComputed callback after success", async () => {
    mockComputeColumn.mockResolvedValueOnce(MOCK_RESULT)
    const onComputed = jest.fn()
    render(<ComputeCard suggestion={BASE_SUGGESTION} onComputed={onComputed} />)

    fireEvent.click(screen.getByRole("button", { name: /add column/i }))

    await waitFor(() => {
      expect(onComputed).toHaveBeenCalledWith(MOCK_RESULT)
    })
  })
})
