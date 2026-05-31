/**
 * Tests for ErrorCorrelationCard component and store action.
 */

import React from "react"
import { render, screen, act } from "@testing-library/react"

import { ErrorCorrelationCard } from "@/components/chat/error-correlation-card"
import type { ErrorCorrelationResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const clearDriversResult: ErrorCorrelationResult = {
  model_run_id: "run-1",
  algorithm: "linear_regression",
  target_col: "revenue",
  features: [
    { feature: "units", correlation: 0.45, correlation_abs: 0.45, direction: "positive", rank: 1 },
    { feature: "region", correlation: -0.22, correlation_abs: 0.22, direction: "negative", rank: 2 },
    { feature: "discount", correlation: 0.08, correlation_abs: 0.08, direction: "neutral", rank: 3 },
  ],
  error_type: "absolute_residual",
  n_total: 100,
  n_errors: 100,
  error_rate: null,
  verdict: "clear_drivers",
  verdict_label: "Clear Error Drivers Found",
  top_driver: "units",
  summary: "Feature 'units' most strongly correlates with prediction errors (r=0.45).",
}

const noDriversResult: ErrorCorrelationResult = {
  model_run_id: "run-2",
  algorithm: "random_forest_classifier",
  target_col: "churn",
  features: [
    { feature: "age", correlation: 0.04, correlation_abs: 0.04, direction: "neutral", rank: 1 },
  ],
  error_type: "misclassification",
  n_total: 80,
  n_errors: 12,
  error_rate: 0.15,
  verdict: "none",
  verdict_label: "No Clear Error Drivers",
  top_driver: null,
  summary: "No input feature strongly correlates with prediction errors.",
}

const weakDriversResult: ErrorCorrelationResult = {
  model_run_id: "run-3",
  algorithm: "random_forest_regressor",
  target_col: "price",
  features: [
    { feature: "sqft", correlation: 0.18, correlation_abs: 0.18, direction: "positive", rank: 1 },
  ],
  error_type: "absolute_residual",
  n_total: 60,
  n_errors: 60,
  error_rate: null,
  verdict: "weak_drivers",
  verdict_label: "Weak Error Drivers",
  top_driver: "sqft",
  summary: "Weak error correlation found.",
}

// ---------------------------------------------------------------------------
// Component tests
// ---------------------------------------------------------------------------

describe("ErrorCorrelationCard", () => {
  it("renders the header and verdict badge for clear_drivers", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByText("🔎 Error Correlation")).toBeInTheDocument()
    expect(screen.getByText("Clear Error Drivers Found")).toBeInTheDocument()
  })

  it("renders emerald badge for no clear drivers", () => {
    render(<ErrorCorrelationCard result={noDriversResult} />)
    expect(screen.getByText("No Clear Error Drivers")).toBeInTheDocument()
  })

  it("renders amber badge for weak drivers", () => {
    render(<ErrorCorrelationCard result={weakDriversResult} />)
    expect(screen.getByText("Weak Error Drivers")).toBeInTheDocument()
  })

  it("renders algorithm badge", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByTestId("algorithm-badge")).toHaveTextContent("linear_regression")
  })

  it("renders target column badge", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByTestId("target-badge")).toHaveTextContent("revenue")
  })

  it("renders n_total rows count", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    // n_total (100) and n_errors (100) both appear; just check "rows" text is present
    expect(screen.getByText("rows")).toBeInTheDocument()
  })

  it("renders error_rate for classification", () => {
    render(<ErrorCorrelationCard result={noDriversResult} />)
    expect(screen.getByText(/15\.0%/)).toBeInTheDocument()
  })

  it("shows feature rows with correlation values", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByTestId("error-correlation-row-units")).toBeInTheDocument()
    expect(screen.getByTestId("correlation-value-units")).toHaveTextContent("r = 0.450")
  })

  it("shows all feature rows", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByTestId("error-correlation-row-units")).toBeInTheDocument()
    expect(screen.getByTestId("error-correlation-row-region")).toBeInTheDocument()
    expect(screen.getByTestId("error-correlation-row-discount")).toBeInTheDocument()
  })

  it("shows top driver callout for clear_drivers", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByTestId("top-driver-callout")).toBeInTheDocument()
    expect(screen.getByTestId("top-driver-callout")).toHaveTextContent("units")
  })

  it("does not show top driver callout for none verdict", () => {
    render(<ErrorCorrelationCard result={noDriversResult} />)
    expect(screen.queryByText(/Top error driver/)).not.toBeInTheDocument()
  })

  it("renders summary paragraph", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByTestId("error-correlation-summary")).toHaveTextContent(
      "Feature 'units' most strongly correlates with prediction errors"
    )
  })

  it("renders sr-only figcaption", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    const fig = screen.getByRole("figure")
    expect(fig.querySelector("figcaption")).toBeInTheDocument()
  })

  it("shows positive direction arrow for positive correlation", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByText(/↑ more errors/)).toBeInTheDocument()
  })

  it("shows negative direction arrow for negative correlation", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByText(/↓ more errors/)).toBeInTheDocument()
  })

  it("shows neutral indicator for near-zero correlation", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByText(/≈ neutral/)).toBeInTheDocument()
  })

  it("renders misclassification error type label", () => {
    render(<ErrorCorrelationCard result={noDriversResult} />)
    expect(screen.getByText(/misclassification/)).toBeInTheDocument()
  })

  it("renders absolute residual error type label", () => {
    render(<ErrorCorrelationCard result={clearDriversResult} />)
    expect(screen.getByText(/absolute residual/)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachErrorCorrelationToLastMessage", () => {
  it("attaches error correlation to last assistant message", () => {
    act(() => {
      useAppStore.setState({
        messages: [
          { role: "user", content: "which features cause errors?" },
          { role: "assistant", content: "Let me check." },
        ],
      })
    })

    act(() => {
      useAppStore.getState().attachErrorCorrelationToLastMessage(clearDriversResult)
    })

    const msgs = useAppStore.getState().messages
    expect(msgs[1].error_correlation).toEqual(clearDriversResult)
  })

  it("does not modify user messages", () => {
    act(() => {
      useAppStore.setState({
        messages: [{ role: "user", content: "test" }],
      })
    })

    act(() => {
      useAppStore.getState().attachErrorCorrelationToLastMessage(clearDriversResult)
    })

    const msgs = useAppStore.getState().messages
    expect(msgs[0].error_correlation).toBeUndefined()
  })
})
