/**
 * Tests for DictionaryCard — AI-powered data dictionary component.
 *
 * Covers:
 * - Initial render: shows CTA buttons when no data loaded
 * - "Quick summary" loads GET dictionary
 * - "AI descriptions" triggers POST /dictionary
 * - Columns render with type badge and description
 * - "Show N more" expand/collapse when > 8 columns
 * - "Regenerate" button appears after data is loaded
 * - Error state when API fails
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { DictionaryCard } from "../components/data/dictionary-card"
import { api } from "../lib/api"
import type { DataDictionary } from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    data: {
      getDictionary: jest.fn(),
      generateDictionary: jest.fn(),
    },
  },
}))

const mockGet = api.data.getDictionary as jest.MockedFunction<typeof api.data.getDictionary>
const mockPost = api.data.generateDictionary as jest.MockedFunction<
  typeof api.data.generateDictionary
>

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const DICT_3_COLS: DataDictionary = {
  dataset_id: "ds-1",
  filename: "sales.csv",
  generated: false,
  columns: [
    {
      name: "revenue",
      dtype: "float64",
      col_type: "metric",
      description: "Total sales revenue.",
    },
    {
      name: "region",
      dtype: "object",
      col_type: "dimension",
      description: "Geographic region of the sale.",
    },
    {
      name: "order_date",
      dtype: "object",
      col_type: "date",
      description: "Date the order was placed.",
    },
  ],
}

const DICT_AI_GENERATED: DataDictionary = {
  ...DICT_3_COLS,
  generated: true,
}

// Build a dictionary with 10 columns to test show-more
const DICT_10_COLS: DataDictionary = {
  dataset_id: "ds-big",
  filename: "big.csv",
  generated: false,
  columns: Array.from({ length: 10 }, (_, i) => ({
    name: `col_${i}`,
    dtype: "float64",
    col_type: "metric" as const,
    description: `Description for column ${i}.`,
  })),
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderCard(datasetId = "ds-1", initialData?: DataDictionary) {
  return render(<DictionaryCard datasetId={datasetId} initialData={initialData} />)
}

// ---------------------------------------------------------------------------
// Tests: empty state
// ---------------------------------------------------------------------------

describe("DictionaryCard — empty state", () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it("renders without crashing", () => {
    renderCard()
    expect(screen.getByText("Data Dictionary")).toBeInTheDocument()
  })

  it("shows quick summary and AI descriptions buttons when no data", () => {
    renderCard()
    expect(screen.getByRole("button", { name: /quick summary/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /ai descriptions/i })).toBeInTheDocument()
  })

  it("calls getDictionary when Quick summary button clicked", async () => {
    mockGet.mockResolvedValueOnce(DICT_3_COLS)
    renderCard()
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    await waitFor(() => expect(mockGet).toHaveBeenCalledWith("ds-1"))
  })

  it("calls generateDictionary when AI descriptions button clicked", async () => {
    mockPost.mockResolvedValueOnce(DICT_AI_GENERATED)
    renderCard()
    fireEvent.click(screen.getByRole("button", { name: /ai descriptions/i }))
    await waitFor(() => expect(mockPost).toHaveBeenCalledWith("ds-1"))
  })
})

// ---------------------------------------------------------------------------
// Tests: data loaded
// ---------------------------------------------------------------------------

describe("DictionaryCard — with data", () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockGet.mockResolvedValue(DICT_3_COLS)
  })

  it("shows column names after loading", async () => {
    renderCard()
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    await waitFor(() => screen.getByText("revenue"))
    expect(screen.getByText("region")).toBeInTheDocument()
    expect(screen.getByText("order_date")).toBeInTheDocument()
  })

  it("shows column descriptions", async () => {
    renderCard()
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    await waitFor(() => screen.getByText("Total sales revenue."))
  })

  it("shows type badges (Metric, Dimension, Date)", async () => {
    renderCard()
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    // Wait for data to load first (column names render faster than badges)
    await waitFor(() => screen.getByText("revenue"))
    // Badges may have surrounding whitespace; use regex for robustness
    expect(screen.getAllByText(/^Metric$/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/^Dimension$/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/^Date$/i).length).toBeGreaterThanOrEqual(1)
  })

  it("shows AI descriptions indicator when generated=true", async () => {
    mockGet.mockResolvedValueOnce(DICT_AI_GENERATED)
    renderCard()
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    await waitFor(() => screen.getByText(/AI descriptions/i))
  })

  it("shows Regenerate button after data is loaded", async () => {
    renderCard()
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    await waitFor(() => screen.getByRole("button", { name: /regenerate/i }))
  })
})

// ---------------------------------------------------------------------------
// Tests: show more / collapse
// ---------------------------------------------------------------------------

describe("DictionaryCard — show more", () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockGet.mockResolvedValue(DICT_10_COLS)
  })

  it("shows only 8 columns initially", async () => {
    renderCard("ds-big")
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    await waitFor(() => screen.getByText("col_0"))
    // col_8 and col_9 should be hidden
    expect(screen.queryByText("col_8")).not.toBeInTheDocument()
  })

  it("shows 'Show 2 more columns' button for 10-col dataset", async () => {
    renderCard("ds-big")
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    await waitFor(() => screen.getByText(/show 2 more/i))
  })

  it("reveals all columns after clicking show more", async () => {
    renderCard("ds-big")
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    await waitFor(() => screen.getByText(/show 2 more/i))
    fireEvent.click(screen.getByText(/show 2 more/i))
    expect(screen.getByText("col_8")).toBeInTheDocument()
    expect(screen.getByText("col_9")).toBeInTheDocument()
  })

  it("collapses back after clicking show less", async () => {
    renderCard("ds-big")
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    await waitFor(() => screen.getByText(/show 2 more/i))
    fireEvent.click(screen.getByText(/show 2 more/i))
    fireEvent.click(screen.getByText(/show less/i))
    expect(screen.queryByText("col_8")).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Tests: error state
// ---------------------------------------------------------------------------

describe("DictionaryCard — error state", () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it("shows error message when getDictionary fails", async () => {
    mockGet.mockRejectedValueOnce(new Error("network error"))
    renderCard()
    fireEvent.click(screen.getByRole("button", { name: /quick summary/i }))
    await waitFor(() => screen.getByText(/failed to load/i))
  })

  it("shows error message when generateDictionary fails", async () => {
    mockPost.mockRejectedValueOnce(new Error("api error"))
    renderCard()
    fireEvent.click(screen.getByRole("button", { name: /ai descriptions/i }))
    await waitFor(() => screen.getByText(/failed to generate/i))
  })
})
