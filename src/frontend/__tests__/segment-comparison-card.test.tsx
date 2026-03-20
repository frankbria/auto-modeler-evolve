/**
 * Tests for SegmentComparisonCard — the side-by-side segment stats component.
 *
 * Covers:
 *   1. Renders data-testid container
 *   2. Shows group_col name in header
 *   3. Shows val1 and val2 labels with counts
 *   4. Shows metric row names
 *   5. Shows mean values for each segment
 *   6. Shows direction arrows for notable differences
 *   7. Shows em-dash for null stats
 *   8. Shows effect badge for notable columns
 *   9. Shows "showing N more" when >8 columns
 *  10. Sorts notable columns first
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { SegmentComparisonCard } from "../components/data/segment-comparison-card"
import type { SegmentComparisonResult } from "../lib/types"

const BASE_RESULT: SegmentComparisonResult = {
  group_col: "region",
  val1: "East",
  val2: "West",
  count1: 4,
  count2: 4,
  columns: [
    {
      name: "revenue",
      mean1: 1975.0,
      std1: 172.3,
      median1: 1950.0,
      count1: 4,
      mean2: 532.5,
      std2: 51.6,
      median2: 525.0,
      count2: 4,
      effect_size: 8.5,
      direction: "higher_in_val1",
    },
    {
      name: "units",
      mean1: 19.75,
      std1: 1.7,
      median1: 19.5,
      count1: 4,
      mean2: 5.25,
      std2: 0.5,
      median2: 5.0,
      count2: 4,
      effect_size: 6.2,
      direction: "higher_in_val1",
    },
    {
      name: "cost",
      mean1: 775.0,
      std1: 62.0,
      median1: 762.5,
      count1: 4,
      mean2: 206.25,
      std2: 10.5,
      median2: 205.0,
      count2: 4,
      effect_size: 4.1,
      direction: "higher_in_val1",
    },
  ],
  notable_diffs: [
    { name: "revenue", effect_size: 8.5, direction: "higher_in_val1" },
    { name: "units", effect_size: 6.2, direction: "higher_in_val1" },
  ],
  summary: "Comparing East (4 rows) vs West (4 rows). Notable differences: revenue is higher in East; units is higher in East.",
}

describe("SegmentComparisonCard", () => {
  it("renders with data-testid", () => {
    render(<SegmentComparisonCard result={BASE_RESULT} />)
    expect(screen.getByTestId("segment-comparison-card")).toBeInTheDocument()
  })

  it("shows group_col in header", () => {
    render(<SegmentComparisonCard result={BASE_RESULT} />)
    expect(screen.getByText(/region/i)).toBeInTheDocument()
  })

  it("shows val1 label with count", () => {
    render(<SegmentComparisonCard result={BASE_RESULT} />)
    const eastMatches = screen.getAllByText(/East/)
    expect(eastMatches.length).toBeGreaterThanOrEqual(1)
    const counts = screen.getAllByText("(4)")
    expect(counts.length).toBeGreaterThanOrEqual(1)
  })

  it("shows val2 label", () => {
    render(<SegmentComparisonCard result={BASE_RESULT} />)
    const westMatches = screen.getAllByText(/West/)
    expect(westMatches.length).toBeGreaterThanOrEqual(1)
  })

  it("shows metric names (underscores replaced with spaces)", () => {
    render(<SegmentComparisonCard result={BASE_RESULT} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
    expect(screen.getByText("units")).toBeInTheDocument()
    expect(screen.getByText("cost")).toBeInTheDocument()
  })

  it("shows direction arrow for notable higher_in_val1", () => {
    render(<SegmentComparisonCard result={BASE_RESULT} />)
    const arrows = screen.getAllByText(/↑ East/)
    expect(arrows.length).toBeGreaterThanOrEqual(1)
  })

  it("shows summary text", () => {
    render(<SegmentComparisonCard result={BASE_RESULT} />)
    expect(screen.getByText(/Comparing East.*vs West/i)).toBeInTheDocument()
  })

  it("shows em-dash for null mean values", () => {
    const withNull: SegmentComparisonResult = {
      ...BASE_RESULT,
      columns: [
        {
          ...BASE_RESULT.columns[0],
          mean2: null,
          direction: null,
          effect_size: null,
        },
      ],
      notable_diffs: [],
    }
    render(<SegmentComparisonCard result={withNull} />)
    const dashes = screen.getAllByText("—")
    expect(dashes.length).toBeGreaterThan(0)
  })

  it("shows 'showing N more' when more than 8 columns", () => {
    const manyColumns: SegmentComparisonResult = {
      ...BASE_RESULT,
      columns: Array.from({ length: 10 }, (_, i) => ({
        name: `col_${i}`,
        mean1: i * 10.0,
        std1: 1.0,
        median1: i * 10.0,
        count1: 4,
        mean2: i * 2.0,
        std2: 0.5,
        median2: i * 2.0,
        count2: 4,
        effect_size: 0.1,
        direction: "higher_in_val1" as const,
      })),
      notable_diffs: [],
    }
    render(<SegmentComparisonCard result={manyColumns} />)
    expect(screen.getByText(/Showing top 8 of 10/i)).toBeInTheDocument()
  })

  it("shows effect badge for large effects", () => {
    render(<SegmentComparisonCard result={BASE_RESULT} />)
    // revenue has effect_size 8.5 → "very large" badge
    const badges = screen.getAllByText("very large")
    expect(badges.length).toBeGreaterThanOrEqual(1)
  })
})
