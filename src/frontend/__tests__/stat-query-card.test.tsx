import { render, screen } from "@testing-library/react"
import { StatQueryCard } from "@/components/data/stat-query-card"
import type { StatQueryResult } from "@/lib/types"

const meanResult: StatQueryResult = {
  dataset_id: "ds-1",
  agg: "mean",
  col: "revenue",
  value: 3840.0,
  n_rows: 5,
  n_valid: 5,
  formatted_value: "3,840.00",
  label: "average",
  summary: "The average of 'revenue' is 3,840.00 (based on 5 non-null values out of 5 rows).",
}

const sumResult: StatQueryResult = {
  dataset_id: "ds-1",
  agg: "sum",
  col: "revenue",
  value: 19200.0,
  n_rows: 5,
  n_valid: 5,
  formatted_value: "19.20k",
  label: "total",
  summary: "The total of 'revenue' is 19.20k (based on 5 non-null values out of 5 rows).",
}

const countResult: StatQueryResult = {
  dataset_id: "ds-1",
  agg: "count",
  col: null,
  value: 100,
  n_rows: 100,
  formatted_value: "100",
  label: "count",
  summary: "The dataset has 100 rows.",
}

const maxResult: StatQueryResult = {
  dataset_id: "ds-1",
  agg: "max",
  col: "cost",
  value: 5000.0,
  n_rows: 5,
  n_valid: 5,
  formatted_value: "5,000.00",
  label: "maximum",
  summary: "The maximum of 'cost' is 5,000.00.",
}

describe("StatQueryCard", () => {
  it("renders formatted value prominently", () => {
    render(<StatQueryCard result={meanResult} />)
    expect(screen.getByText("3,840.00")).toBeInTheDocument()
  })

  it("shows label and column name", () => {
    render(<StatQueryCard result={meanResult} />)
    expect(screen.getAllByText(/average/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/revenue/i).length).toBeGreaterThan(0)
  })

  it("shows aggregation badge", () => {
    render(<StatQueryCard result={meanResult} />)
    expect(screen.getByText("mean")).toBeInTheDocument()
  })

  it("shows sum with k suffix", () => {
    render(<StatQueryCard result={sumResult} />)
    expect(screen.getByText("19.20k")).toBeInTheDocument()
  })

  it("shows 'total' label for sum", () => {
    render(<StatQueryCard result={sumResult} />)
    expect(screen.getAllByText(/total/i).length).toBeGreaterThan(0)
  })

  it("shows count without column name", () => {
    render(<StatQueryCard result={countResult} />)
    expect(screen.getByText("100")).toBeInTheDocument()
  })

  it("shows maximum label", () => {
    render(<StatQueryCard result={maxResult} />)
    expect(screen.getAllByText(/maximum/i).length).toBeGreaterThan(0)
  })

  it("shows summary footer", () => {
    render(<StatQueryCard result={meanResult} />)
    expect(
      screen.getByText(/The average of 'revenue' is/i)
    ).toBeInTheDocument()
  })

  it("shows valid vs total row info when different", () => {
    const partial = { ...meanResult, n_valid: 4, n_rows: 5 }
    render(<StatQueryCard result={partial} />)
    expect(screen.getByText(/4 non-null values out of 5 rows/i)).toBeInTheDocument()
  })

  it("does not show row info when all values valid", () => {
    const { container } = render(<StatQueryCard result={meanResult} />)
    // n_valid == n_rows (5 == 5) → dedicated row-info paragraph should not be rendered
    // (summary footer may mention non-null, but the dedicated <p> row-info element should be absent)
    const rowInfoPara = container.querySelector("p.text-xs.text-muted-foreground.mb-2")
    expect(rowInfoPara).not.toBeInTheDocument()
  })

  it("has accessible region label with value", () => {
    render(<StatQueryCard result={meanResult} />)
    expect(
      screen.getByRole("region", { name: /average of revenue: 3,840.00/i })
    ).toBeInTheDocument()
  })

  it("renders underscore-replaced column name", () => {
    const result = { ...meanResult, col: "total_revenue" }
    render(<StatQueryCard result={result} />)
    expect(screen.getByText(/total revenue/i)).toBeInTheDocument()
  })
})
