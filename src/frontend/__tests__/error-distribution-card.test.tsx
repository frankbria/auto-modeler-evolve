/**
 * Tests for ErrorDistributionCard — Prediction Error Distribution Analysis (Day 65).
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ErrorDistributionCard } from "@/components/chat/error-distribution-card"
import type { ErrorDistributionResult } from "@/lib/types"

// Recharts needs ResizeObserver in jsdom
class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}
global.ResizeObserver = ResizeObserverMock as unknown as typeof ResizeObserver

// ---------------------------------------------------------------------------
// Test data helpers
// ---------------------------------------------------------------------------

const REGRESSION_UNBIASED: ErrorDistributionResult = {
  problem_type: "regression",
  bins: [
    { lo: -2.0, hi: -1.0, count: 5, pct: 10.0, label: "-2.00 to -1.00" },
    { lo: -1.0, hi: 0.0, count: 20, pct: 40.0, label: "-1.00 to 0.00" },
    { lo: 0.0, hi: 1.0, count: 20, pct: 40.0, label: "0.00 to 1.00" },
    { lo: 1.0, hi: 2.0, count: 5, pct: 10.0, label: "1.00 to 2.00" },
  ],
  stats: {
    mean: 0.01,
    std: 0.8,
    mae: 0.65,
    bias_label: "unbiased",
    bias_pct: 0.5,
    within_1std_pct: 80.0,
    total: 50,
  },
  summary:
    "Residual distribution across 50 training rows: MAE = 0.650, std = 0.800. 80.0% of errors fall within ±0.800. The model has no systematic over- or under-prediction tendency.",
  algorithm: "linear_regression",
  target_col: "revenue",
}

const REGRESSION_BIASED: ErrorDistributionResult = {
  problem_type: "regression",
  bins: [
    { lo: -3.0, hi: -2.0, count: 10, pct: 20.0, label: "-3.00 to -2.00" },
    { lo: -2.0, hi: -1.0, count: 30, pct: 60.0, label: "-2.00 to -1.00" },
    { lo: -1.0, hi: 0.0, count: 10, pct: 20.0, label: "-1.00 to 0.00" },
  ],
  stats: {
    mean: -1.8,
    std: 0.7,
    mae: 1.8,
    bias_label: "under-predicts",
    bias_pct: 18.0,
    within_1std_pct: 60.0,
    total: 50,
  },
  summary: "Residual distribution across 50 training rows: MAE = 1.800, std = 0.700. Tends to under-predict.",
  algorithm: "random_forest_regressor",
  target_col: "sales",
}

const CLASSIFICATION_RESULT: ErrorDistributionResult = {
  problem_type: "classification",
  bins: [],
  class_breakdown: [
    { class: "yes", total: 40, wrong: 16, error_rate: 0.4, error_pct: 40.0 },
    { class: "no", total: 60, wrong: 6, error_rate: 0.1, error_pct: 10.0 },
  ],
  stats: {
    total: 100,
    total_wrong: 22,
    overall_error_rate: 0.22,
    overall_accuracy: 0.78,
    n_classes: 2,
  },
  summary:
    "The model is 78% accurate across 100 training rows. Hardest class: 'yes' (40% errors). Easiest class: 'no' (10% errors).",
  algorithm: "random_forest_classifier",
  target_col: "churn",
}

// ---------------------------------------------------------------------------
// Tests: Regression card
// ---------------------------------------------------------------------------

describe("ErrorDistributionCard — regression", () => {
  it("renders header with algorithm and target badges", () => {
    render(<ErrorDistributionCard result={REGRESSION_UNBIASED} />)
    expect(screen.getByText("Error Distribution")).toBeInTheDocument()
    expect(screen.getByText(/Linear Regression/i)).toBeInTheDocument()
    expect(screen.getByText(/Target: revenue/i)).toBeInTheDocument()
  })

  it("shows regression problem type badge", () => {
    render(<ErrorDistributionCard result={REGRESSION_UNBIASED} />)
    // exact lowercase DOM text; algorithm badge says "Linear Regression"
    expect(screen.getByText("regression")).toBeInTheDocument()
  })

  it("displays MAE stat", () => {
    render(<ErrorDistributionCard result={REGRESSION_UNBIASED} />)
    expect(screen.getByText("0.650")).toBeInTheDocument()
  })

  it("shows std stat", () => {
    render(<ErrorDistributionCard result={REGRESSION_UNBIASED} />)
    expect(screen.getByText("0.800")).toBeInTheDocument()
  })

  it("shows within_1std_pct", () => {
    render(<ErrorDistributionCard result={REGRESSION_UNBIASED} />)
    expect(screen.getByText("80.0%")).toBeInTheDocument()
  })

  it("shows unbiased badge for unbiased model", () => {
    render(<ErrorDistributionCard result={REGRESSION_UNBIASED} />)
    expect(screen.getByText("Unbiased")).toBeInTheDocument()
  })

  it("shows under-predicts badge for biased model", () => {
    render(<ErrorDistributionCard result={REGRESSION_BIASED} />)
    expect(screen.getByText("Tends to under-predict")).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<ErrorDistributionCard result={REGRESSION_UNBIASED} />)
    expect(screen.getByText(/MAE = 0.650/)).toBeInTheDocument()
  })

  it("renders axis labels for residual histogram", () => {
    render(<ErrorDistributionCard result={REGRESSION_UNBIASED} />)
    expect(screen.getByText(/Residual/)).toBeInTheDocument()
  })

  it("has accessible figure element", () => {
    const { container } = render(<ErrorDistributionCard result={REGRESSION_UNBIASED} />)
    const figure = container.querySelector("figure")
    expect(figure).toBeInTheDocument()
    expect(figure?.getAttribute("aria-label")).toContain("revenue")
  })
})

// ---------------------------------------------------------------------------
// Tests: Classification card
// ---------------------------------------------------------------------------

describe("ErrorDistributionCard — classification", () => {
  it("renders header with algorithm badge", () => {
    render(<ErrorDistributionCard result={CLASSIFICATION_RESULT} />)
    expect(screen.getByText("Error Distribution")).toBeInTheDocument()
  })

  it("shows overall accuracy stat", () => {
    render(<ErrorDistributionCard result={CLASSIFICATION_RESULT} />)
    expect(screen.getByText("78.0%")).toBeInTheDocument()
  })

  it("shows total wrong count", () => {
    render(<ErrorDistributionCard result={CLASSIFICATION_RESULT} />)
    expect(screen.getByText("22")).toBeInTheDocument()
  })

  it("shows n_classes stat", () => {
    render(<ErrorDistributionCard result={CLASSIFICATION_RESULT} />)
    expect(screen.getByText("2")).toBeInTheDocument()
  })

  it("shows per-class names in table", () => {
    render(<ErrorDistributionCard result={CLASSIFICATION_RESULT} />)
    expect(screen.getByText("yes")).toBeInTheDocument()
    expect(screen.getByText("no")).toBeInTheDocument()
  })

  it("shows highest error rate for hardest class", () => {
    render(<ErrorDistributionCard result={CLASSIFICATION_RESULT} />)
    expect(screen.getByText("40.0%")).toBeInTheDocument()
  })

  it("shows lowest error rate for easiest class", () => {
    render(<ErrorDistributionCard result={CLASSIFICATION_RESULT} />)
    expect(screen.getByText("10.0%")).toBeInTheDocument()
  })

  it("renders per-class total rows", () => {
    render(<ErrorDistributionCard result={CLASSIFICATION_RESULT} />)
    expect(screen.getByText("40")).toBeInTheDocument()
    expect(screen.getByText("60")).toBeInTheDocument()
  })

  it("shows sorted-highest note", () => {
    render(<ErrorDistributionCard result={CLASSIFICATION_RESULT} />)
    expect(screen.getByText(/Sorted highest error rate first/i)).toBeInTheDocument()
  })

  it("renders summary", () => {
    render(<ErrorDistributionCard result={CLASSIFICATION_RESULT} />)
    expect(screen.getByText(/78% accurate/)).toBeInTheDocument()
  })
})
