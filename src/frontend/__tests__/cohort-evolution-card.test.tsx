import React from "react"
import { render, screen } from "@testing-library/react"
import { CohortEvolutionCard } from "@/components/deploy/cohort-evolution-card"
import type { CohortEvolutionResult } from "@/lib/types"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const minimalPeriod = (
  label: string,
  dsId: string,
  rows: number,
): CohortEvolutionResult["periods"][0] => ({
  period_label: label,
  dataset_id: dsId,
  total_rows: rows,
  categorical_profile: [],
  numeric_profile: [],
})

const periodWithCats = (
  label: string,
  dsId: string,
): CohortEvolutionResult["periods"][0] => ({
  period_label: label,
  dataset_id: dsId,
  total_rows: 200,
  categorical_profile: [
    {
      column: "age_group",
      categories: [
        { value: "18-25", top_pct: 60, overall_pct: 30, ratio: 2 },
        { value: "26-35", top_pct: 30, overall_pct: 25, ratio: 1.2 },
      ],
      dominant: "18-25",
      dominant_top_pct: 60,
    },
  ],
  numeric_profile: [],
})

const risingShift: CohortEvolutionResult["shifts"][0]["categorical_shifts"][0] = {
  column: "region",
  category: "north",
  from_pct: 30,
  to_pct: 45,
  change: 15,
  direction: "rising",
}

const fallingShift: CohortEvolutionResult["shifts"][0]["categorical_shifts"][0] = {
  column: "age_group",
  category: "18-25",
  from_pct: 60,
  to_pct: 40,
  change: -20,
  direction: "falling",
}

const twoShiftResult = (overrides?: Partial<CohortEvolutionResult>): CohortEvolutionResult => ({
  n: 20,
  direction: "highest",
  target_column: "churn_score",
  problem_type: "regression",
  n_periods: 2,
  periods: [periodWithCats("Upload 1", "ds1"), periodWithCats("Upload 2", "ds2")],
  shifts: [
    {
      from_period: "Upload 1",
      to_period: "Upload 2",
      categorical_shifts: [risingShift, fallingShift],
      summary: "Region north rose 15pp between Upload 1 and Upload 2.",
    },
  ],
  summary:
    "The top-20 highest churn_score cohort shifted notably: region north rose from 30% to 45% (+15pp).",
  ...overrides,
})

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("CohortEvolutionCard", () => {
  it("renders the card container", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    expect(screen.getByTestId("cohort-evolution-card")).toBeInTheDocument()
  })

  it("displays the Cohort Evolution heading", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    expect(screen.getByText("Cohort Evolution")).toBeInTheDocument()
  })

  it("shows the number of periods badge", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    expect(screen.getByText("2 periods")).toBeInTheDocument()
  })

  it("shows the Top-N badge", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    expect(screen.getByText("Top-20 Highest")).toBeInTheDocument()
  })

  it("shows the target column badge", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    expect(screen.getByText("churn_score")).toBeInTheDocument()
  })

  it("displays the overall summary sentence", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    const matches = screen.getAllByText(/top-20 highest churn_score cohort shifted/i)
    expect(matches.length).toBeGreaterThan(0)
  })

  it("renders period nodes for each period", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    const nodes = screen.getAllByTestId("cohort-period-node")
    expect(nodes).toHaveLength(2)
  })

  it("renders shift connectors between periods", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    const connectors = screen.getAllByTestId("cohort-shift-connector")
    expect(connectors).toHaveLength(1)
  })

  it("renders shift rows for notable shifts", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    const rows = screen.getAllByTestId("cohort-shift-row")
    expect(rows.length).toBeGreaterThan(0)
  })

  it("shows rising shift with correct styling cues", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    // The rising shift row should contain the ▲ arrow
    const rows = screen.getAllByTestId("cohort-shift-row")
    const texts = rows.map((r) => r.textContent ?? "")
    expect(texts.some((t) => t.includes("▲"))).toBe(true)
  })

  it("shows falling shift with correct styling cues", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    const rows = screen.getAllByTestId("cohort-shift-row")
    const texts = rows.map((r) => r.textContent ?? "")
    expect(texts.some((t) => t.includes("▼"))).toBe(true)
  })

  it("renders period labels in period nodes", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    expect(screen.getByText("Upload 1")).toBeInTheDocument()
    expect(screen.getByText("Upload 2")).toBeInTheDocument()
  })

  it("renders row counts in period nodes", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    // 200 rows for each period
    const rowTexts = screen.getAllByText(/200 rows/i)
    expect(rowTexts.length).toBeGreaterThan(0)
  })

  it("renders bar charts for categorical profile data", () => {
    render(<CohortEvolutionCard result={twoShiftResult()} />)
    // age_group column should appear (underscores replaced with spaces)
    const ageGroupLabels = screen.getAllByText(/age group/i)
    expect(ageGroupLabels.length).toBeGreaterThan(0)
  })

  it("shows 'No categorical data' when profile is empty", () => {
    const result = twoShiftResult({
      periods: [
        minimalPeriod("Upload 1", "ds1", 100),
        minimalPeriod("Upload 2", "ds2", 100),
      ],
      shifts: [
        {
          from_period: "Upload 1",
          to_period: "Upload 2",
          categorical_shifts: [],
          summary: "No notable shifts.",
        },
      ],
    })
    render(<CohortEvolutionCard result={result} />)
    const noDataMessages = screen.getAllByText(/no categorical data/i)
    expect(noDataMessages.length).toBeGreaterThan(0)
  })

  it("shows stable message when no shifts exceed threshold", () => {
    const result = twoShiftResult({
      shifts: [
        {
          from_period: "Upload 1",
          to_period: "Upload 2",
          categorical_shifts: [],
          summary: "Composition was stable.",
        },
      ],
    })
    render(<CohortEvolutionCard result={result} />)
    expect(
      screen.getByText(/no categorical segments shifted more than 5 percentage points/i),
    ).toBeInTheDocument()
  })

  it("renders 'Lowest' label for direction=lowest", () => {
    render(<CohortEvolutionCard result={twoShiftResult({ direction: "lowest" })} />)
    expect(screen.getByText("Top-20 Lowest")).toBeInTheDocument()
  })

  it("has accessible figcaption for screen readers", () => {
    const { container } = render(<CohortEvolutionCard result={twoShiftResult()} />)
    const caption = container.querySelector("figcaption.sr-only")
    expect(caption).toBeInTheDocument()
    expect(caption?.textContent).toMatch(/cohort evolution/i)
  })

  it("caps shift rows at 6", () => {
    // Create 10 shifts — card should show at most 6
    const manyShifts = Array.from({ length: 10 }, (_, i) => ({
      column: `col_${i}`,
      category: `cat_${i}`,
      from_pct: 10,
      to_pct: 30,
      change: 20,
      direction: "rising" as const,
    }))
    const result = twoShiftResult({
      shifts: [
        {
          from_period: "Upload 1",
          to_period: "Upload 2",
          categorical_shifts: manyShifts,
          summary: "Many shifts.",
        },
      ],
    })
    render(<CohortEvolutionCard result={result} />)
    const rows = screen.getAllByTestId("cohort-shift-row")
    expect(rows.length).toBeLessThanOrEqual(6)
  })

  it("sorts shift rows by absolute change descending", () => {
    const shifts = [
      { column: "a", category: "x", from_pct: 10, to_pct: 15, change: 5, direction: "rising" as const },
      { column: "b", category: "y", from_pct: 50, to_pct: 20, change: -30, direction: "falling" as const },
      { column: "c", category: "z", from_pct: 30, to_pct: 50, change: 20, direction: "rising" as const },
    ]
    const result = twoShiftResult({
      shifts: [
        {
          from_period: "Upload 1",
          to_period: "Upload 2",
          categorical_shifts: shifts,
          summary: "B fell most.",
        },
      ],
    })
    render(<CohortEvolutionCard result={result} />)
    const rows = screen.getAllByTestId("cohort-shift-row")
    // First row should show the biggest absolute change (-30pp)
    expect(rows[0].textContent).toContain("-30pp")
  })
})
