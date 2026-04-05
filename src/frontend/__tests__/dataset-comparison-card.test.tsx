import { render, screen } from "@testing-library/react"
import { DatasetComparisonCard } from "@/components/data/dataset-comparison-card"
import type { DatasetComparisonResult } from "@/lib/types"

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

function makeResult(overrides?: Partial<DatasetComparisonResult>): DatasetComparisonResult {
  return {
    baseline_id: "ds-1",
    new_id: "ds-2",
    baseline_name: "sales_q1.csv",
    new_name: "sales_q2.csv",
    row_count_old: 1000,
    row_count_new: 1200,
    row_count_change_pct: 20.0,
    col_count_old: 5,
    col_count_new: 5,
    new_columns: [],
    dropped_columns: [],
    numeric_drifts: [],
    categorical_drifts: [],
    drift_score: 0,
    summary: "The new dataset looks very similar to the original — distributions match closely.",
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Rendering tests
// ---------------------------------------------------------------------------

test("renders card heading", () => {
  render(<DatasetComparisonCard result={makeResult()} />)
  expect(screen.getByText(/Data Comparison/i)).toBeInTheDocument()
})

test("renders summary text", () => {
  render(<DatasetComparisonCard result={makeResult()} />)
  expect(screen.getByText(/distributions match closely/i)).toBeInTheDocument()
})

test("renders baseline and new file names", () => {
  render(<DatasetComparisonCard result={makeResult()} />)
  expect(screen.getByText("sales_q1.csv")).toBeInTheDocument()
  expect(screen.getByText("sales_q2.csv")).toBeInTheDocument()
})

test("renders row counts", () => {
  render(<DatasetComparisonCard result={makeResult()} />)
  expect(screen.getByText(/1,000 rows/)).toBeInTheDocument()
  expect(screen.getByText(/1,200 rows/)).toBeInTheDocument()
})

test("shows green drift score badge for zero drift", () => {
  render(<DatasetComparisonCard result={makeResult({ drift_score: 0 })} />)
  expect(screen.getByText(/Drift score: 0\/100/i)).toBeInTheDocument()
})

test("shows yellow drift score badge for moderate drift", () => {
  render(<DatasetComparisonCard result={makeResult({ drift_score: 30 })} />)
  expect(screen.getByText(/Drift score: 30\/100/i)).toBeInTheDocument()
})

test("shows red drift score badge for high drift", () => {
  render(<DatasetComparisonCard result={makeResult({ drift_score: 70 })} />)
  expect(screen.getByText(/Drift score: 70\/100/i)).toBeInTheDocument()
})

test("renders change count badge when changes exist", () => {
  render(
    <DatasetComparisonCard
      result={makeResult({
        numeric_drifts: [
          {
            col: "revenue",
            old_mean: 100,
            new_mean: 200,
            old_std: 10,
            new_std: 20,
            pct_change: 100,
            severity: "high",
          },
        ],
      })}
    />
  )
  expect(screen.getByText(/1 change/i)).toBeInTheDocument()
})

test("renders numeric drift table with column name", () => {
  render(
    <DatasetComparisonCard
      result={makeResult({
        numeric_drifts: [
          {
            col: "revenue",
            old_mean: 100,
            new_mean: 200,
            old_std: 10,
            new_std: 20,
            pct_change: 100,
            severity: "high",
          },
        ],
      })}
    />
  )
  expect(screen.getByText("revenue")).toBeInTheDocument()
  expect(screen.getByText(/Numeric distribution shifts/i)).toBeInTheDocument()
})

test("renders High severity badge for high drift", () => {
  render(
    <DatasetComparisonCard
      result={makeResult({
        numeric_drifts: [
          {
            col: "revenue",
            old_mean: 100,
            new_mean: 200,
            old_std: 10,
            new_std: 20,
            pct_change: 100,
            severity: "high",
          },
        ],
      })}
    />
  )
  expect(screen.getByText("High")).toBeInTheDocument()
})

test("renders categorical drift section with new categories", () => {
  render(
    <DatasetComparisonCard
      result={makeResult({
        categorical_drifts: [
          {
            col: "region",
            new_categories: ["Pacific"],
            dropped_categories: [],
            top_shift_pct: 15.0,
            severity: "medium",
          },
        ],
      })}
    />
  )
  expect(screen.getByText(/Categorical changes/i)).toBeInTheDocument()
  expect(screen.getByText(/Pacific/)).toBeInTheDocument()
})

test("renders dropped categories in red", () => {
  render(
    <DatasetComparisonCard
      result={makeResult({
        categorical_drifts: [
          {
            col: "region",
            new_categories: [],
            dropped_categories: ["West"],
            top_shift_pct: 0,
            severity: "medium",
          },
        ],
      })}
    />
  )
  expect(screen.getByText(/West/)).toBeInTheDocument()
  expect(screen.getByText(/Dropped:/i)).toBeInTheDocument()
})

test("renders new columns section", () => {
  render(
    <DatasetComparisonCard
      result={makeResult({
        new_columns: ["quantity", "discount"],
        dropped_columns: [],
      })}
    />
  )
  expect(screen.getByText(/New columns:/i)).toBeInTheDocument()
  expect(screen.getByText(/quantity, discount/)).toBeInTheDocument()
})

test("renders dropped columns section", () => {
  render(
    <DatasetComparisonCard
      result={makeResult({
        new_columns: [],
        dropped_columns: ["old_col"],
      })}
    />
  )
  expect(screen.getByText(/Dropped columns:/i)).toBeInTheDocument()
  expect(screen.getByText(/old_col/)).toBeInTheDocument()
})

test("renders no-changes message for zero issues", () => {
  render(<DatasetComparisonCard result={makeResult({ drift_score: 0, summary: "identical" })} />)
  expect(screen.getByText(/No significant distribution changes detected/i)).toBeInTheDocument()
})

test("has accessible figure aria-label", () => {
  render(<DatasetComparisonCard result={makeResult()} />)
  expect(screen.getByRole("figure")).toHaveAttribute(
    "aria-label",
    "Dataset comparison report"
  )
})

test("row change percentage shown in row section", () => {
  render(<DatasetComparisonCard result={makeResult({ row_count_change_pct: 20.0 })} />)
  expect(screen.getByText(/\+20%\s*more rows/i)).toBeInTheDocument()
})

test("negative row change shown correctly", () => {
  render(
    <DatasetComparisonCard
      result={makeResult({ row_count_change_pct: -30.0, row_count_old: 1000, row_count_new: 700 })}
    />
  )
  expect(screen.getByText(/-30%\s*fewer rows/i)).toBeInTheDocument()
})
