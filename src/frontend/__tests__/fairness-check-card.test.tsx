/**
 * Tests for FairnessCheckCard component and Zustand store action.
 *
 * Covers:
 *  1.  Renders figure with correct aria-label
 *  2.  Shows "Fair" badge for fair status
 *  3.  Shows "Minor Disparity" badge for warning status
 *  4.  Shows "Bias Detected" badge for biased status
 *  5.  Shows "Insufficient Data" badge + summary for insufficient_data
 *  6.  Shows SPD value and label for classification
 *  7.  Shows DIR value and label for classification
 *  8.  Shows MAE disparity section for regression
 *  9.  Shows per-group metrics table for classification (positive_rate + accuracy)
 * 10.  Shows per-group metrics table for regression (MAE)
 * 11.  Shows role="alert" for warning status
 * 12.  Shows role="alert" for biased status
 * 13.  Does NOT render alert for fair status
 * 14.  Shows algorithm and problem_type badges
 * 15.  Shows sensitive_col badge
 * 16.  Shows target_col when present
 * 17.  sr-only figcaption is present
 * 18.  Store: attachFairnessCheckToLastMessage attaches to last assistant message
 * 19.  Store: does not attach to user message
 * 20.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { FairnessCheckCard } from "@/components/chat/fairness-check-card"
import type { FairnessCheckResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const fairClassification: FairnessCheckResult = {
  overall_status: "fair",
  sensitive_col: "gender",
  target_col: "approved",
  algorithm: "random_forest",
  problem_type: "classification",
  per_group_metrics: [
    { group: "male", count: 120, positive_rate: 0.6, accuracy: 0.82 },
    { group: "female", count: 110, positive_rate: 0.62, accuracy: 0.83 },
  ],
  spd: 0.02,
  spd_label: "fair",
  dir: 1.03,
  dir_label: "passes 4/5ths rule",
  summary: "No significant disparity detected across gender groups.",
}

const warningClassification: FairnessCheckResult = {
  ...fairClassification,
  overall_status: "warning",
  spd: 0.12,
  spd_label: "slight disparity",
  dir: 0.82,
  dir_label: "borderline",
  summary: "Minor disparity detected. Monitor as more data accumulates.",
}

const biasedClassification: FairnessCheckResult = {
  ...fairClassification,
  overall_status: "biased",
  spd: 0.35,
  spd_label: "significant disparity",
  dir: 0.55,
  dir_label: "fails 4/5ths rule",
  summary: "Significant bias detected across gender groups.",
}

const insufficientData: FairnessCheckResult = {
  overall_status: "insufficient_data",
  sensitive_col: "gender",
  target_col: "approved",
  algorithm: "random_forest",
  problem_type: "classification",
  per_group_metrics: [],
  summary: "Need at least 2 distinct groups to assess fairness.",
}

const fairRegression: FairnessCheckResult = {
  overall_status: "fair",
  sensitive_col: "age_group",
  target_col: "salary",
  algorithm: "linear_regression",
  problem_type: "regression",
  per_group_metrics: [
    { group: "young", count: 80, mae: 1200.5 },
    { group: "senior", count: 90, mae: 1180.3 },
  ],
  mae_disparity: 1.02,
  summary: "MAE is consistent across age groups.",
}

const biasedRegression: FairnessCheckResult = {
  ...fairRegression,
  overall_status: "biased",
  mae_disparity: 1.85,
  summary: "Large error gap detected across age groups.",
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("FairnessCheckCard", () => {
  it("renders figure with correct aria-label", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(
      screen.getByRole("figure", { name: /fairness analysis for gender/i }),
    ).toBeInTheDocument()
  })

  it("shows Fair badge for fair status", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(screen.getByText("Fair")).toBeInTheDocument()
  })

  it("shows Minor Disparity badge for warning status", () => {
    render(<FairnessCheckCard result={warningClassification} />)
    expect(screen.getByText("Minor Disparity")).toBeInTheDocument()
  })

  it("shows Bias Detected badge for biased status", () => {
    render(<FairnessCheckCard result={biasedClassification} />)
    expect(screen.getByText("Bias Detected")).toBeInTheDocument()
  })

  it("shows Insufficient Data badge and summary for insufficient_data", () => {
    render(<FairnessCheckCard result={insufficientData} />)
    expect(screen.getByText("Insufficient Data")).toBeInTheDocument()
    expect(screen.getByText(/need at least 2 distinct groups/i)).toBeInTheDocument()
  })

  it("shows SPD value for classification", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(screen.getByText("0.020")).toBeInTheDocument()
  })

  it("shows SPD label for classification", () => {
    render(<FairnessCheckCard result={warningClassification} />)
    expect(screen.getByText("slight disparity")).toBeInTheDocument()
  })

  it("shows DIR value for classification", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(screen.getByText("1.030")).toBeInTheDocument()
  })

  it("shows DIR label for classification", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(screen.getByText("passes 4/5ths rule")).toBeInTheDocument()
  })

  it("shows MAE disparity section for regression", () => {
    render(<FairnessCheckCard result={fairRegression} />)
    expect(screen.getByText("MAE Disparity Ratio")).toBeInTheDocument()
    expect(screen.getByText("1.02×")).toBeInTheDocument()
  })

  it("shows MAE disparity positive guidance for fair regression", () => {
    render(<FairnessCheckCard result={fairRegression} />)
    expect(screen.getByText(/error rates are consistent across groups/i)).toBeInTheDocument()
  })

  it("shows MAE disparity warning guidance for biased regression", () => {
    render(<FairnessCheckCard result={biasedRegression} />)
    expect(screen.getByText(/large error gap — consider re-balancing/i)).toBeInTheDocument()
  })

  it("shows per-group metrics table for classification", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(screen.getByRole("table", { name: /per-group fairness metrics/i })).toBeInTheDocument()
    expect(screen.getByText("male")).toBeInTheDocument()
    expect(screen.getByText("female")).toBeInTheDocument()
    expect(screen.getByText("Pos. Rate")).toBeInTheDocument()
    expect(screen.getByText("Accuracy")).toBeInTheDocument()
  })

  it("shows per-group metrics table for regression", () => {
    render(<FairnessCheckCard result={fairRegression} />)
    expect(screen.getByRole("table", { name: /per-group fairness metrics/i })).toBeInTheDocument()
    expect(screen.getByText("young")).toBeInTheDocument()
    expect(screen.getByText("senior")).toBeInTheDocument()
    expect(screen.getByText("MAE")).toBeInTheDocument()
  })

  it("shows role=alert for warning status", () => {
    render(<FairnessCheckCard result={warningClassification} />)
    expect(screen.getByRole("alert")).toBeInTheDocument()
  })

  it("shows role=alert for biased status", () => {
    render(<FairnessCheckCard result={biasedClassification} />)
    const alert = screen.getByRole("alert")
    expect(alert).toBeInTheDocument()
    expect(alert).toHaveTextContent(/collecting more balanced training data/i)
  })

  it("does NOT render alert for fair status", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("shows algorithm badge", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(screen.getByText("random_forest")).toBeInTheDocument()
  })

  it("shows problem_type badge", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(screen.getByText("classification")).toBeInTheDocument()
  })

  it("shows sensitive_col badge", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(screen.getByText("by gender")).toBeInTheDocument()
  })

  it("shows target_col when present", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(screen.getByText("approved")).toBeInTheDocument()
  })

  it("contains sr-only figcaption", () => {
    const { container } = render(<FairnessCheckCard result={fairClassification} />)
    const caption = container.querySelector("figcaption.sr-only")
    expect(caption).toBeInTheDocument()
    expect(caption?.textContent).toMatch(/fair/i)
  })

  it("shows summary text", () => {
    render(<FairnessCheckCard result={fairClassification} />)
    expect(
      screen.getByText("No significant disparity detected across gender groups."),
    ).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store tests
// ---------------------------------------------------------------------------

describe("attachFairnessCheckToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches fairness_check to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "check fairness" },
        { role: "assistant", content: "Here is the fairness analysis." },
      ],
    })
    useAppStore.getState().attachFairnessCheckToLastMessage(fairClassification)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].fairness_check).toEqual(fairClassification)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "check fairness" }],
    })
    useAppStore.getState().attachFairnessCheckToLastMessage(fairClassification)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].fairness_check).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachFairnessCheckToLastMessage(fairClassification),
    ).not.toThrow()
  })
})
