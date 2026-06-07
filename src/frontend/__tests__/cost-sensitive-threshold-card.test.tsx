import React from "react"
import { render, screen } from "@testing-library/react"
import { CostSensitiveThresholdCard } from "@/components/models/cost-sensitive-threshold-card"
import type { CostSensitiveThresholdResult } from "@/lib/types"

const mockMetricsDefault = {
  threshold: 0.5,
  tp: 45,
  fp: 20,
  tn: 130,
  fn: 5,
  expected_cost: 700.0,
  precision: 0.69,
  recall: 0.9,
  f1: 0.78,
  accuracy: 0.875,
}

const mockMetricsOptimal = {
  threshold: 0.09,
  tp: 49,
  fp: 60,
  tn: 90,
  fn: 1,
  expected_cost: 300.0,
  precision: 0.45,
  recall: 0.98,
  f1: 0.61,
  accuracy: 0.695,
}

const mockResult: CostSensitiveThresholdResult = {
  model_run_id: "run-42",
  algorithm: "random_forest_classifier",
  target_col: "churn",
  optimal_threshold: 0.09,
  fp_cost: 10.0,
  fn_cost: 100.0,
  cost_ratio: 10.0,
  default_metrics: mockMetricsDefault,
  optimal_metrics: mockMetricsOptimal,
  default_cost: 700.0,
  optimal_cost: 300.0,
  cost_savings_pct: 57.1,
  suggested_class_weight: 10.0,
  verdict: "threshold_change_recommended",
  summary:
    "Lowering the decision threshold from 50% to 9% reduces expected cost per 200 predictions from $700 to $300 (saving 57.1%).",
  n_samples: 200,
  n_positive: 50,
  prevalence: 0.25,
  positive_label: "churn",
}

describe("CostSensitiveThresholdCard", () => {
  it("renders without crashing", () => {
    render(<CostSensitiveThresholdCard result={mockResult} />)
    expect(screen.getByTestId("cost-sensitive-threshold-card")).toBeInTheDocument()
  })

  it("shows the verdict badge", () => {
    render(<CostSensitiveThresholdCard result={mockResult} />)
    const badge = screen.getByTestId("verdict-badge")
    expect(badge).toHaveTextContent("Threshold Change Recommended")
  })

  it("shows near-optimal verdict badge when appropriate", () => {
    const nearOptimal: CostSensitiveThresholdResult = {
      ...mockResult,
      verdict: "default_near_optimal",
      cost_savings_pct: 1.0,
    }
    render(<CostSensitiveThresholdCard result={nearOptimal} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Default Near-Optimal")
  })

  it("shows FP and FN costs", () => {
    render(<CostSensitiveThresholdCard result={mockResult} />)
    expect(screen.getByText("False Positive Cost")).toBeInTheDocument()
    expect(screen.getByText("False Negative Cost")).toBeInTheDocument()
    expect(screen.getByText("$10")).toBeInTheDocument()
    expect(screen.getByText("$100")).toBeInTheDocument()
  })

  it("shows optimal threshold percentage", () => {
    render(<CostSensitiveThresholdCard result={mockResult} />)
    expect(screen.getByText("Optimal Threshold")).toBeInTheDocument()
    // "9%" appears in both the header badge and the formula line
    const matches = screen.getAllByText(/9%/)
    expect(matches.length).toBeGreaterThan(0)
  })

  it("shows default and optimal metrics sections", () => {
    render(<CostSensitiveThresholdCard result={mockResult} />)
    expect(screen.getByTestId("default-metrics")).toBeInTheDocument()
    expect(screen.getByTestId("optimal-metrics")).toBeInTheDocument()
  })

  it("shows savings banner when savings are positive", () => {
    render(<CostSensitiveThresholdCard result={mockResult} />)
    expect(screen.getByTestId("savings-banner")).toBeInTheDocument()
    expect(screen.getByText(/57\.1% cost reduction/)).toBeInTheDocument()
  })

  it("hides savings banner when savings are zero", () => {
    const noSavings: CostSensitiveThresholdResult = {
      ...mockResult,
      cost_savings_pct: 0.0,
    }
    render(<CostSensitiveThresholdCard result={noSavings} />)
    expect(screen.queryByTestId("savings-banner")).not.toBeInTheDocument()
  })

  it("shows suggested class weight", () => {
    render(<CostSensitiveThresholdCard result={mockResult} />)
    expect(screen.getByText(/class_weight/)).toBeInTheDocument()
  })

  it("shows algorithm and target column", () => {
    render(<CostSensitiveThresholdCard result={mockResult} />)
    expect(screen.getByText(/random_forest_classifier/)).toBeInTheDocument()
    // "churn" appears multiple times (target_col + positive_label in hint)
    const churnMatches = screen.getAllByText(/churn/)
    expect(churnMatches.length).toBeGreaterThan(0)
  })
})
