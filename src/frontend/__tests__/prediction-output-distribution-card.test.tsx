/**
 * Tests for PredictionOutputDistributionCard component and store action.
 */

import React from "react"
import { render, screen, act } from "@testing-library/react"

import { PredictionOutputDistributionCard } from "@/components/deploy/prediction-output-distribution-card"
import type { PredictionOutputDistributionShiftResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const stableResult: PredictionOutputDistributionShiftResult = {
  deployment_id: "dep-1",
  n_training: 200,
  n_production: 80,
  verdict: "stable",
  verdict_label: "Distribution is stable",
  ks_statistic: 0.08,
  ks_p_value: 0.32,
  mean_shift: 2.1,
  mean_shift_pct: 2.1,
  std_ratio: 0.98,
  training_stats: {
    mean: 100.0,
    std: 10.0,
    min: 72.0,
    p25: 93.0,
    median: 100.0,
    p75: 107.0,
    p95: 117.0,
    max: 133.0,
  },
  production_stats: {
    mean: 102.1,
    std: 9.8,
    min: 75.0,
    p25: 95.0,
    median: 102.0,
    p75: 109.0,
    p95: 119.0,
    max: 130.0,
  },
  training_histogram: [
    { bin_start: 70.0, bin_end: 80.0, count: 5, label: "70.00–80.00" },
    { bin_start: 80.0, bin_end: 90.0, count: 20, label: "80.00–90.00" },
    { bin_start: 90.0, bin_end: 100.0, count: 60, label: "90.00–100.00" },
  ],
  production_histogram: [
    { bin_start: 70.0, bin_end: 80.0, count: 2, label: "70.00–80.00" },
    { bin_start: 80.0, bin_end: 90.0, count: 10, label: "80.00–90.00" },
    { bin_start: 90.0, bin_end: 100.0, count: 25, label: "90.00–100.00" },
  ],
  problem_type: "regression",
  summary: "The model's output distribution is stable. Production predictions (n=80) closely match the training-time distribution.",
}

const significantShiftResult: PredictionOutputDistributionShiftResult = {
  deployment_id: "dep-2",
  n_training: 300,
  n_production: 120,
  verdict: "significant_shift",
  verdict_label: "Significant distribution shift detected",
  ks_statistic: 0.72,
  ks_p_value: 0.0001,
  mean_shift: 45.0,
  mean_shift_pct: 45.0,
  std_ratio: 1.3,
  training_stats: {
    mean: 100.0,
    std: 10.0,
    min: 72.0,
    p25: 93.0,
    median: 100.0,
    p75: 107.0,
    p95: 117.0,
    max: 133.0,
  },
  production_stats: {
    mean: 145.0,
    std: 13.0,
    min: 110.0,
    p25: 135.0,
    median: 145.0,
    p75: 155.0,
    p95: 170.0,
    max: 190.0,
  },
  training_histogram: [
    { bin_start: 70.0, bin_end: 100.0, count: 200, label: "70.00–100.00" },
    { bin_start: 100.0, bin_end: 130.0, count: 100, label: "100.00–130.00" },
  ],
  production_histogram: [
    { bin_start: 70.0, bin_end: 100.0, count: 5, label: "70.00–100.00" },
    { bin_start: 100.0, bin_end: 130.0, count: 115, label: "100.00–130.00" },
  ],
  problem_type: "regression",
  summary: "Significant distribution shift detected. Production predictions are 45.0% above the training baseline on average. Retraining is strongly recommended.",
}

const noDataResult: PredictionOutputDistributionShiftResult = {
  deployment_id: "dep-3",
  n_training: 0,
  n_production: 5,
  verdict: "no_data",
  verdict_label: "Not enough production predictions",
  summary: "Only 5 production predictions recorded — need at least 10 to compare distributions.",
}

const moderateResult: PredictionOutputDistributionShiftResult = {
  deployment_id: "dep-4",
  n_training: 150,
  n_production: 60,
  verdict: "moderate_shift",
  verdict_label: "Moderate distribution shift detected",
  ks_statistic: 0.28,
  ks_p_value: 0.03,
  mean_shift: 15.0,
  mean_shift_pct: 15.0,
  std_ratio: 1.1,
  training_stats: {
    mean: 100.0,
    std: 10.0,
    min: 75.0,
    p25: 93.0,
    median: 100.0,
    p75: 107.0,
    p95: 117.0,
    max: 130.0,
  },
  production_stats: {
    mean: 115.0,
    std: 11.0,
    min: 85.0,
    p25: 107.0,
    median: 115.0,
    p75: 123.0,
    p95: 135.0,
    max: 148.0,
  },
  training_histogram: [
    { bin_start: 70.0, bin_end: 100.0, count: 100, label: "70.00–100.00" },
  ],
  production_histogram: [
    { bin_start: 70.0, bin_end: 100.0, count: 20, label: "70.00–100.00" },
  ],
  problem_type: "regression",
  summary: "Moderate distribution shift detected. Monitor closely.",
}

// ---------------------------------------------------------------------------
// Component tests
// ---------------------------------------------------------------------------

describe("PredictionOutputDistributionCard", () => {
  it("renders the component heading", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    expect(screen.getByText("Output Distribution Shift")).toBeInTheDocument()
  })

  it("renders the verdict badge for stable", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Distribution is stable")
  })

  it("renders the verdict badge for significant_shift", () => {
    render(<PredictionOutputDistributionCard result={significantShiftResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Significant distribution shift detected")
  })

  it("renders the verdict badge for moderate_shift", () => {
    render(<PredictionOutputDistributionCard result={moderateResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Moderate distribution shift detected")
  })

  it("renders the verdict badge for no_data", () => {
    render(<PredictionOutputDistributionCard result={noDataResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Not enough production predictions")
  })

  it("renders n_training and n_production for a data result", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    expect(screen.getByText(/200 training \/ 80 production predictions/)).toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    const elements = screen.getAllByText(/The model's output distribution is stable/)
    expect(elements.length).toBeGreaterThan(0)
  })

  it("shows summary text for no_data result without stats", () => {
    render(<PredictionOutputDistributionCard result={noDataResult} />)
    const elements = screen.getAllByText(/Only 5 production predictions recorded/)
    expect(elements.length).toBeGreaterThan(0)
  })

  it("renders KS statistic when data is available", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    expect(screen.getByText("KS Statistic")).toBeInTheDocument()
    expect(screen.getByText("0.080")).toBeInTheDocument()
  })

  it("renders p-value when data is available", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    expect(screen.getByText("p-value")).toBeInTheDocument()
    expect(screen.getByText("0.3200")).toBeInTheDocument()
  })

  it("renders positive mean shift with + prefix", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    expect(screen.getByText("+2.1%")).toBeInTheDocument()
  })

  it("renders negative mean shift with - prefix", () => {
    const negativeShift: PredictionOutputDistributionShiftResult = {
      ...stableResult,
      mean_shift: -5.0,
      mean_shift_pct: -5.0,
    }
    render(<PredictionOutputDistributionCard result={negativeShift} />)
    expect(screen.getByText("-5.0%")).toBeInTheDocument()
  })

  it("renders distribution comparison table headers", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    expect(screen.getByText("Distribution Comparison")).toBeInTheDocument()
    expect(screen.getByRole("columnheader", { name: /Training/ })).toBeInTheDocument()
    expect(screen.getByRole("columnheader", { name: /Production/ })).toBeInTheDocument()
  })

  it("renders Mean row in table", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    expect(screen.getByRole("cell", { name: "Mean" })).toBeInTheDocument()
  })

  it("renders histogram section for data results", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    expect(
      screen.getByText("Prediction Value Distribution (Training vs Production)")
    ).toBeInTheDocument()
  })

  it("does not render stats or histogram for no_data", () => {
    render(<PredictionOutputDistributionCard result={noDataResult} />)
    expect(screen.queryByText("Distribution Comparison")).not.toBeInTheDocument()
    expect(
      screen.queryByText("Prediction Value Distribution (Training vs Production)")
    ).not.toBeInTheDocument()
  })

  it("has figcaption for screen readers", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    const figcaption = document.querySelector("figcaption.sr-only")
    expect(figcaption).not.toBeNull()
    expect(figcaption!.textContent).toContain("Distribution is stable")
  })

  it("has data-testid on the card root", () => {
    render(<PredictionOutputDistributionCard result={significantShiftResult} />)
    expect(screen.getByTestId("prediction-output-distribution-card")).toBeInTheDocument()
  })

  it("uses emerald border for stable verdict", () => {
    render(<PredictionOutputDistributionCard result={stableResult} />)
    const card = screen.getByTestId("prediction-output-distribution-card")
    expect(card.className).toContain("emerald")
  })

  it("uses rose border for significant_shift verdict", () => {
    render(<PredictionOutputDistributionCard result={significantShiftResult} />)
    const card = screen.getByTestId("prediction-output-distribution-card")
    expect(card.className).toContain("rose")
  })

  it("uses amber border for moderate_shift verdict", () => {
    render(<PredictionOutputDistributionCard result={moderateResult} />)
    const card = screen.getByTestId("prediction-output-distribution-card")
    expect(card.className).toContain("amber")
  })
})

// ---------------------------------------------------------------------------
// Zustand store action tests
// ---------------------------------------------------------------------------

describe("attachOutputDistributionShiftToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches output_distribution_shift to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "m1", role: "assistant", content: "hello", timestamp: new Date() },
      ],
    })
    act(() => {
      useAppStore.getState().attachOutputDistributionShiftToLastMessage(stableResult)
    })
    const messages = useAppStore.getState().messages
    expect(messages[0].output_distribution_shift).toEqual(stableResult)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [
        { id: "m1", role: "user", content: "query", timestamp: new Date() },
      ],
    })
    act(() => {
      useAppStore.getState().attachOutputDistributionShiftToLastMessage(stableResult)
    })
    const messages = useAppStore.getState().messages
    expect(messages[0].output_distribution_shift).toBeUndefined()
  })

  it("does nothing when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() => {
      act(() => {
        useAppStore.getState().attachOutputDistributionShiftToLastMessage(stableResult)
      })
    }).not.toThrow()
  })
})
