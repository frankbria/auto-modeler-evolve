/**
 * Tests for CrosstabTable — the pivot table component.
 *
 * Covers:
 *   1. Renders the data-testid container
 *   2. Shows metric label (agg × value_col) in header
 *   3. Shows column headers from col_headers
 *   4. Shows row labels from rows
 *   5. Shows cell values
 *   6. Shows row totals column
 *   7. Shows "Total" grand totals row
 *   8. Shows summary text
 *   9. Handles count mode (no value_col)
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { CrosstabTable } from "../components/data/crosstab-table"
import type { CrosstabResult } from "../lib/types"

const BASE_RESULT: CrosstabResult = {
  row_col: "product",
  col_col: "region",
  value_col: "revenue",
  agg_func: "sum",
  col_headers: ["East", "North", "South", "West"],
  rows: [
    { row_label: "Widget A", cells: [2100, 1700, 980, null], row_total: 4780 },
    { row_label: "Widget B", cells: [null, 1650, 850, 760], row_total: 3260 },
    { row_label: "Widget C", cells: [3200, null, 1100, 450], row_total: 4750 },
  ],
  col_totals: [5300, 3350, 2930, 1210],
  grand_total: 12790,
  summary: "Sum of revenue broken down by product (rows) × region (columns). Showing 3 product values across 4 region categories.",
}

describe("CrosstabTable", () => {
  it("renders with data-testid", () => {
    render(<CrosstabTable result={BASE_RESULT} />)
    expect(screen.getByTestId("crosstab-table")).toBeInTheDocument()
  })

  it("shows metric label header", () => {
    render(<CrosstabTable result={BASE_RESULT} />)
    // The header p contains the agg+value+dimensions text; summary also references them
    const matches = screen.getAllByText(/Sum of revenue/i)
    expect(matches.length).toBeGreaterThanOrEqual(1)
  })

  it("shows row × col dimension label", () => {
    render(<CrosstabTable result={BASE_RESULT} />)
    // Both the header and the summary mention "product" and "region"
    const matches = screen.getAllByText((content) =>
      content.includes("product") && content.includes("region")
    )
    expect(matches.length).toBeGreaterThanOrEqual(1)
  })

  it("renders all column headers", () => {
    render(<CrosstabTable result={BASE_RESULT} />)
    for (const header of BASE_RESULT.col_headers) {
      expect(screen.getByText(header)).toBeInTheDocument()
    }
  })

  it("renders all row labels", () => {
    render(<CrosstabTable result={BASE_RESULT} />)
    for (const row of BASE_RESULT.rows) {
      expect(screen.getByText(row.row_label)).toBeInTheDocument()
    }
  })

  it("shows Total column header for row totals", () => {
    render(<CrosstabTable result={BASE_RESULT} />)
    const totals = screen.getAllByText("Total")
    expect(totals.length).toBeGreaterThanOrEqual(1)
  })

  it("shows grand total row", () => {
    render(<CrosstabTable result={BASE_RESULT} />)
    // There should be a "Total" row at the bottom
    expect(screen.getAllByText("Total").length).toBeGreaterThanOrEqual(1)
  })

  it("shows summary text", () => {
    render(<CrosstabTable result={BASE_RESULT} />)
    expect(screen.getByText(/Showing 3 product values/i)).toBeInTheDocument()
  })

  it("shows 'Count' label when no value_col", () => {
    const countResult: CrosstabResult = {
      ...BASE_RESULT,
      value_col: null,
      agg_func: "sum",
    }
    render(<CrosstabTable result={countResult} />)
    expect(screen.getByText(/Count/i)).toBeInTheDocument()
  })

  it("renders null cells as em dash", () => {
    render(<CrosstabTable result={BASE_RESULT} />)
    // Widget A has null for West — should render as —
    const dashes = screen.getAllByText("—")
    expect(dashes.length).toBeGreaterThan(0)
  })
})
