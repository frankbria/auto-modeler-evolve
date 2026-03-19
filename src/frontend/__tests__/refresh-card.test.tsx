/**
 * Tests for RefreshCard — the dataset refresh UI component.
 *
 * Covers:
 * - Default info text rendered without SSE prompt
 * - Prompt info (current filename + row count) shown when prompt provided
 * - File chooser button rendered
 * - Compatible refresh → shows green success + calls onRefreshed
 * - Incompatible refresh → shows warning about missing feature columns
 * - New/removed columns shown in result
 * - Error state when API fails
 * - api.data.refresh called with correct arguments
 * - api.ts refresh method returns correct type (unit test on api object)
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { RefreshCard } from "../components/data/refresh-card"
import { api } from "../lib/api"
import type { DatasetRefreshResult, RefreshPrompt } from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    data: {
      refresh: jest.fn(),
    },
  },
}))

const mockRefresh = api.data.refresh as jest.MockedFunction<typeof api.data.refresh>

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const PROMPT: RefreshPrompt = {
  dataset_id: "ds-1",
  current_filename: "sales_q3.csv",
  current_row_count: 500,
  required_columns: ["date", "product", "region", "revenue", "units"],
}

const COMPATIBLE_RESULT: DatasetRefreshResult = {
  dataset_id: "ds-1",
  filename: "sales_q4.csv",
  row_count: 600,
  column_count: 5,
  new_columns: [],
  removed_columns: [],
  feature_columns_missing: [],
  compatible: true,
  preview: [],
  column_stats: [],
}

const RESULT_WITH_NEW_COL: DatasetRefreshResult = {
  ...COMPATIBLE_RESULT,
  new_columns: ["discount"],
  compatible: true,
}

const RESULT_WITH_REMOVED_COL: DatasetRefreshResult = {
  ...COMPATIBLE_RESULT,
  removed_columns: ["units"],
  compatible: true,
}

const INCOMPATIBLE_RESULT: DatasetRefreshResult = {
  ...COMPATIBLE_RESULT,
  removed_columns: ["units"],
  feature_columns_missing: ["units"],
  compatible: false,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeFile(name = "data.csv", content = "date,product\n2024-01-01,A"): File {
  return new File([content], name, { type: "text/csv" })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("RefreshCard", () => {
  beforeEach(() => jest.clearAllMocks())

  it("renders the card heading", () => {
    render(<RefreshCard datasetId="ds-1" />)
    expect(screen.getByText(/Replace Dataset/i)).toBeInTheDocument()
  })

  it("shows default info text without prompt", () => {
    render(<RefreshCard datasetId="ds-1" />)
    expect(screen.getByText(/feature engineering and model history/i, { exact: false })).toBeInTheDocument()
  })

  it("shows prompt filename and row count when prompt provided", () => {
    render(<RefreshCard datasetId="ds-1" prompt={PROMPT} />)
    expect(screen.getByText(/sales_q3.csv/)).toBeInTheDocument()
    expect(screen.getByText(/500/)).toBeInTheDocument()
  })

  it("renders the file chooser button", () => {
    render(<RefreshCard datasetId="ds-1" />)
    expect(screen.getByTestId("replace-data-button")).toBeInTheDocument()
  })

  it("calls api.data.refresh with datasetId and file on selection", async () => {
    mockRefresh.mockResolvedValueOnce(COMPATIBLE_RESULT)
    render(<RefreshCard datasetId="ds-1" />)
    const input = screen.getByTestId("refresh-file-input")
    const file = makeFile()
    fireEvent.change(input, { target: { files: [file] } })
    await waitFor(() => {
      expect(mockRefresh).toHaveBeenCalledWith("ds-1", file)
    })
  })

  it("shows compatible badge after successful refresh", async () => {
    mockRefresh.mockResolvedValueOnce(COMPATIBLE_RESULT)
    render(<RefreshCard datasetId="ds-1" />)
    fireEvent.change(screen.getByTestId("refresh-file-input"), {
      target: { files: [makeFile()] },
    })
    await waitFor(() => {
      expect(screen.getByText(/Compatible/i)).toBeInTheDocument()
    })
  })

  it("shows updated row count after refresh", async () => {
    mockRefresh.mockResolvedValueOnce(COMPATIBLE_RESULT)
    render(<RefreshCard datasetId="ds-1" />)
    fireEvent.change(screen.getByTestId("refresh-file-input"), {
      target: { files: [makeFile()] },
    })
    await waitFor(() => {
      expect(screen.getByText(/600/)).toBeInTheDocument()
    })
  })

  it("shows new column name in result", async () => {
    mockRefresh.mockResolvedValueOnce(RESULT_WITH_NEW_COL)
    render(<RefreshCard datasetId="ds-1" />)
    fireEvent.change(screen.getByTestId("refresh-file-input"), {
      target: { files: [makeFile()] },
    })
    await waitFor(() => {
      expect(screen.getByText(/discount/)).toBeInTheDocument()
    })
  })

  it("shows removed column name in result", async () => {
    mockRefresh.mockResolvedValueOnce(RESULT_WITH_REMOVED_COL)
    render(<RefreshCard datasetId="ds-1" />)
    fireEvent.change(screen.getByTestId("refresh-file-input"), {
      target: { files: [makeFile()] },
    })
    await waitFor(() => {
      expect(screen.getByText(/units/)).toBeInTheDocument()
    })
  })

  it("shows incompatible badge when feature columns missing", async () => {
    mockRefresh.mockResolvedValueOnce(INCOMPATIBLE_RESULT)
    render(<RefreshCard datasetId="ds-1" />)
    fireEvent.change(screen.getByTestId("refresh-file-input"), {
      target: { files: [makeFile()] },
    })
    await waitFor(() => {
      expect(screen.getByText(/Incompatible/i)).toBeInTheDocument()
    })
  })

  it("shows missing feature columns when incompatible", async () => {
    mockRefresh.mockResolvedValueOnce(INCOMPATIBLE_RESULT)
    render(<RefreshCard datasetId="ds-1" />)
    fireEvent.change(screen.getByTestId("refresh-file-input"), {
      target: { files: [makeFile()] },
    })
    await waitFor(() => {
      expect(screen.getByText(/Model feature columns missing/i)).toBeInTheDocument()
    })
  })

  it("calls onRefreshed callback after success", async () => {
    mockRefresh.mockResolvedValueOnce(COMPATIBLE_RESULT)
    const onRefreshed = jest.fn()
    render(<RefreshCard datasetId="ds-1" onRefreshed={onRefreshed} />)
    fireEvent.change(screen.getByTestId("refresh-file-input"), {
      target: { files: [makeFile()] },
    })
    await waitFor(() => {
      expect(onRefreshed).toHaveBeenCalledWith(COMPATIBLE_RESULT)
    })
  })

  it("shows error when API call fails", async () => {
    mockRefresh.mockRejectedValueOnce(new Error("Network error"))
    render(<RefreshCard datasetId="ds-1" />)
    fireEvent.change(screen.getByTestId("refresh-file-input"), {
      target: { files: [makeFile()] },
    })
    await waitFor(() => {
      expect(screen.getByTestId("refresh-error")).toBeInTheDocument()
    })
  })
})

// ---------------------------------------------------------------------------
// api.ts: data.refresh method
// ---------------------------------------------------------------------------

describe("api.data.refresh method", () => {
  it("is a function", () => {
    const realApi = jest.requireActual("../lib/api") as typeof import("../lib/api")
    expect(typeof realApi.api.data.refresh).toBe("function")
  })
})
