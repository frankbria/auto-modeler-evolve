/**
 * Tests for ProdPerformanceCard — Training vs Production Performance Monitor (Day 64).
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ProdPerformanceCard } from "@/components/chat/prod-performance-card"
import type { ProdPerformanceResult } from "@/lib/types"

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

const NO_FEEDBACK_REGRESSION: ProdPerformanceResult = {
  deployment_id: "dep-1",
  algorithm: "linear_regression",
  target_column: "revenue",
  problem_type: "regression",
  metric_name: "MAE",
  metric_direction: "lower_is_better",
  training_value: 10.0,
  has_data: false,
  status: "no_feedback",
  summary:
    "No matched feedback yet. Record actual outcomes in the Deployment tab to compare training and production accuracy.",
}

const STABLE_REGRESSION: ProdPerformanceResult = {
  deployment_id: "dep-1",
  algorithm: "random_forest_regressor",
  target_column: "revenue",
  problem_type: "regression",
  metric_name: "MAE",
  metric_direction: "lower_is_better",
  training_value: 10.0,
  live_value: 9.5,
  degradation_pct: -5.0,
  status: "stable",
  has_data: true,
  n_feedback: 25,
  weekly_timeline: [
    { period: "2026-01-05", value: 9.8, n: 10 },
    { period: "2026-01-12", value: 9.5, n: 15 },
  ],
  summary: "Training MAE: 10.0000 | Live MAE: 9.5000 (from 25 feedback records). Production performance is stable.",
}

const WARNING_CLASSIFICATION: ProdPerformanceResult = {
  deployment_id: "dep-2",
  algorithm: "random_forest_classifier",
  target_column: "churn",
  problem_type: "classification",
  metric_name: "Accuracy",
  metric_direction: "higher_is_better",
  training_value: 0.85,
  training_pct: 85.0,
  live_value: 0.75,
  live_pct: 75.0,
  degradation_pct: 11.8,
  status: "warning",
  has_data: true,
  n_feedback: 80,
  weekly_timeline: [
    { period: "2026-01-05", value: 80.0, n: 40 },
    { period: "2026-01-12", value: 70.0, n: 40 },
  ],
  summary:
    "Training accuracy: 85.0% | Live accuracy: 75.0% (from 80 rated feedback records). Accuracy dropped 11.8% vs training.",
}

const DEGRADING_CLASSIFICATION: ProdPerformanceResult = {
  deployment_id: "dep-3",
  algorithm: "logistic_regression",
  target_column: "churn",
  problem_type: "classification",
  metric_name: "Accuracy",
  metric_direction: "higher_is_better",
  training_value: 0.85,
  training_pct: 85.0,
  live_value: 0.60,
  live_pct: 60.0,
  degradation_pct: 29.4,
  status: "degrading",
  has_data: true,
  n_feedback: 100,
  weekly_timeline: [
    { period: "2026-01-05", value: 75.0, n: 50 },
    { period: "2026-01-12", value: 60.0, n: 50 },
  ],
  summary:
    "Training accuracy: 85.0% | Live accuracy: 60.0% (from 100 rated feedback records). Accuracy dropped 29.4% vs training.",
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ProdPerformanceCard", () => {
  it("renders without crashing", () => {
    render(<ProdPerformanceCard result={NO_FEEDBACK_REGRESSION} />)
    expect(screen.getByTestId("prod-performance-card")).toBeInTheDocument()
  })

  it("shows card heading", () => {
    render(<ProdPerformanceCard result={STABLE_REGRESSION} />)
    expect(screen.getByText(/Training vs Production Performance/i)).toBeInTheDocument()
  })

  it("shows target column badge", () => {
    render(<ProdPerformanceCard result={STABLE_REGRESSION} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("shows feedback record count badge", () => {
    render(<ProdPerformanceCard result={STABLE_REGRESSION} />)
    expect(screen.getAllByText(/25 feedback record/i).length).toBeGreaterThan(0)
  })

  it("shows summary text", () => {
    render(<ProdPerformanceCard result={STABLE_REGRESSION} />)
    expect(
      screen.getByText(/Production performance is stable/i)
    ).toBeInTheDocument()
  })

  // Status badges

  it("shows Stable badge for stable status", () => {
    render(<ProdPerformanceCard result={STABLE_REGRESSION} />)
    expect(screen.getByText("Stable")).toBeInTheDocument()
  })

  it("shows Warning badge for warning status", () => {
    render(<ProdPerformanceCard result={WARNING_CLASSIFICATION} />)
    expect(screen.getByText("Warning")).toBeInTheDocument()
  })

  it("shows Degrading badge for degrading status", () => {
    render(<ProdPerformanceCard result={DEGRADING_CLASSIFICATION} />)
    expect(screen.getByText("Degrading")).toBeInTheDocument()
  })

  it("shows No Feedback badge when status is no_feedback", () => {
    render(<ProdPerformanceCard result={NO_FEEDBACK_REGRESSION} />)
    expect(screen.getByText("No Feedback")).toBeInTheDocument()
  })

  // No-feedback state

  it("shows no-feedback message when has_data is false", () => {
    render(<ProdPerformanceCard result={NO_FEEDBACK_REGRESSION} />)
    expect(screen.getByTestId("no-feedback-message")).toBeInTheDocument()
  })

  it("does not show metric boxes when has_data is false", () => {
    render(<ProdPerformanceCard result={NO_FEEDBACK_REGRESSION} />)
    expect(screen.queryByTestId("metric-box-training-mae")).not.toBeInTheDocument()
  })

  // Metric boxes

  it("shows training and live metric boxes when has_data is true", () => {
    render(<ProdPerformanceCard result={STABLE_REGRESSION} />)
    expect(screen.getByTestId("metric-box-training-mae")).toBeInTheDocument()
    expect(screen.getByTestId("metric-box-live-mae")).toBeInTheDocument()
  })

  it("shows classification training/live accuracy boxes", () => {
    render(<ProdPerformanceCard result={WARNING_CLASSIFICATION} />)
    expect(screen.getByTestId("metric-box-training-accuracy")).toBeInTheDocument()
    expect(screen.getByTestId("metric-box-live-accuracy")).toBeInTheDocument()
  })

  // Degradation badge

  it("shows degradation badge when has_data is true", () => {
    render(<ProdPerformanceCard result={STABLE_REGRESSION} />)
    expect(screen.getByTestId("degradation-badge")).toBeInTheDocument()
  })

  it("degradation badge text mentions error improvement for regression stable", () => {
    render(<ProdPerformanceCard result={STABLE_REGRESSION} />)
    const badge = screen.getByTestId("degradation-badge")
    // pct = -5.0 → error improved
    expect(badge.textContent).toMatch(/better than training/i)
  })

  it("degradation badge mentions accuracy drop for classification warning", () => {
    render(<ProdPerformanceCard result={WARNING_CLASSIFICATION} />)
    const badge = screen.getByTestId("degradation-badge")
    expect(badge.textContent).toMatch(/accuracy.*vs training/i)
  })

  // Alert callout for warning/degrading

  it("shows warning callout with role=alert for warning status", () => {
    render(<ProdPerformanceCard result={WARNING_CLASSIFICATION} />)
    const alerts = screen.getAllByRole("alert")
    expect(alerts.length).toBeGreaterThan(0)
    expect(alerts[0].textContent).toMatch(/monitor/i)
  })

  it("shows degrading callout with role=alert for degrading status", () => {
    render(<ProdPerformanceCard result={DEGRADING_CLASSIFICATION} />)
    const alerts = screen.getAllByRole("alert")
    expect(alerts.length).toBeGreaterThan(0)
    expect(alerts[0].textContent).toMatch(/retrain/i)
  })

  it("does not show alert callout for stable status", () => {
    render(<ProdPerformanceCard result={STABLE_REGRESSION} />)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  // Accessibility

  it("has aria-label on the figure", () => {
    render(<ProdPerformanceCard result={STABLE_REGRESSION} />)
    const fig = screen.getByLabelText(/training vs production performance comparison/i)
    expect(fig).toBeInTheDocument()
  })

  it("renders without metric boxes when weekly_timeline has < 2 items", () => {
    const result: ProdPerformanceResult = {
      ...STABLE_REGRESSION,
      weekly_timeline: [{ period: "2026-01-05", value: 9.5, n: 5 }],
    }
    render(<ProdPerformanceCard result={result} />)
    // Card still renders correctly — just no timeline chart
    expect(screen.getByTestId("prod-performance-card")).toBeInTheDocument()
  })
})
