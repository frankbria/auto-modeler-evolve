import { render, screen } from "@testing-library/react"
import { PairCorrelationCard } from "@/components/data/pair-correlation-card"
import type { PairCorrelationResult } from "@/lib/types"

const strongPositiveResult: PairCorrelationResult = {
  dataset_id: "ds-1",
  col1: "revenue",
  col2: "cost",
  r: 0.9412,
  p_value: 0.000001,
  n: 100,
  strength: "very strong",
  direction: "positive",
  significant: "highly significant (p < 0.001)",
  interpretation: "When revenue increases, cost tends to increase strongly.",
  summary: "'revenue' and 'cost' have a very strong positive correlation (r = 0.941, n = 100).",
}

const negativeResult: PairCorrelationResult = {
  dataset_id: "ds-1",
  col1: "price",
  col2: "demand",
  r: -0.7321,
  p_value: 0.008,
  n: 50,
  strength: "strong",
  direction: "negative",
  significant: "significant (p < 0.01)",
  interpretation: "There is a strong negative relationship between these columns.",
  summary: "'price' and 'demand' have a strong negative correlation (r = -0.732, n = 50).",
}

const insufficientResult: PairCorrelationResult = {
  dataset_id: "ds-1",
  col1: "a",
  col2: "b",
  r: null,
  p_value: null,
  n: 2,
  strength: "insufficient data",
  direction: "unknown",
  significant: "insufficient data for correlation",
  summary: "Need at least 3 paired observations (only 2 found).",
}

describe("PairCorrelationCard", () => {
  it("renders column names in header", () => {
    render(<PairCorrelationCard result={strongPositiveResult} />)
    expect(screen.getAllByText(/revenue/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/cost/i).length).toBeGreaterThan(0)
  })

  it("shows strength badge", () => {
    render(<PairCorrelationCard result={strongPositiveResult} />)
    expect(screen.getAllByText(/very strong/i).length).toBeGreaterThan(0)
  })

  it("shows direction badge", () => {
    render(<PairCorrelationCard result={strongPositiveResult} />)
    expect(screen.getAllByText(/positive/i).length).toBeGreaterThan(0)
  })

  it("displays r value", () => {
    render(<PairCorrelationCard result={strongPositiveResult} />)
    expect(screen.getAllByText(/\+0\.941/).length).toBeGreaterThan(0)
  })

  it("shows negative r with minus sign", () => {
    render(<PairCorrelationCard result={negativeResult} />)
    expect(screen.getAllByText(/-0\.732/).length).toBeGreaterThan(0)
  })

  it("shows p-value", () => {
    render(<PairCorrelationCard result={strongPositiveResult} />)
    expect(screen.getByText(/p-value/i)).toBeInTheDocument()
    expect(screen.getAllByText(/< 0.001/i).length).toBeGreaterThan(0)
  })

  it("shows significance badge", () => {
    render(<PairCorrelationCard result={strongPositiveResult} />)
    expect(screen.getAllByText(/highly significant/i).length).toBeGreaterThan(0)
  })

  it("shows interpretation text", () => {
    render(<PairCorrelationCard result={strongPositiveResult} />)
    expect(
      screen.getAllByText(/When revenue increases, cost tends to increase strongly/i).length
    ).toBeGreaterThan(0)
  })

  it("shows observation count", () => {
    render(<PairCorrelationCard result={strongPositiveResult} />)
    expect(screen.getAllByText(/100/).length).toBeGreaterThan(0)
  })

  it("shows summary footer", () => {
    render(<PairCorrelationCard result={strongPositiveResult} />)
    expect(
      screen.getAllByText(/very strong positive correlation/i).length
    ).toBeGreaterThan(0)
  })

  it("handles insufficient data gracefully", () => {
    render(<PairCorrelationCard result={insufficientResult} />)
    expect(screen.getAllByText(/at least 3 paired observations/i).length).toBeGreaterThan(0)
  })

  it("has accessible region label", () => {
    render(<PairCorrelationCard result={strongPositiveResult} />)
    expect(
      screen.getByRole("region", { name: /correlation analysis between revenue and cost/i })
    ).toBeInTheDocument()
  })

  it("renders underscore-replaced column names", () => {
    const result = {
      ...strongPositiveResult,
      col1: "total_revenue",
      col2: "total_cost",
    }
    render(<PairCorrelationCard result={result} />)
    expect(screen.getByText(/total revenue/i)).toBeInTheDocument()
    expect(screen.getByText(/total cost/i)).toBeInTheDocument()
  })
})
