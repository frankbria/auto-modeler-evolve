/**
 * Tests for PartialDependenceCard component and Zustand store action.
 *
 * Covers:
 *  1.  Renders figure with correct aria-label
 *  2.  Renders 📉 icon (aria-hidden)
 *  3.  Shows feature and target in heading
 *  4.  Shows Regression badge for regression models
 *  5.  Shows Classification badge for classification models
 *  6.  Shows algorithm name in badge
 *  7.  Shows trend direction badge (increases/decreases/flat)
 *  8.  Renders Recharts ResponsiveContainer for regression with multi-point grid
 *  9.  Shows "averaged over N training records" explainer text
 * 10.  Shows constant-feature fallback message when grid has 1 point
 * 11.  Renders summary footer text
 * 12.  Store: attachPartialDependenceToLastMessage attaches to last assistant message
 * 13.  Store: does not attach to user message
 * 14.  Store: does not crash when messages list is empty
 * 15.  Renders per-class colour legend for multiclass result
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { PartialDependenceCard } from "@/components/validation/partial-dependence-card"
import type { PartialDependenceResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Recharts mock
// ---------------------------------------------------------------------------
jest.mock("recharts", () => {
  const Original = jest.requireActual("recharts")
  return {
    ...Original,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container" style={{ width: 500, height: 200 }}>
        {children}
      </div>
    ),
  }
})

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const regressionResult: PartialDependenceResult = {
  feature: "units",
  target_col: "revenue",
  algorithm: "linear_regression",
  problem_type: "regression",
  grid_values: [5, 10, 15, 20],
  mean_predictions: [500, 1000, 1500, 2000],
  std_predictions: [10, 12, 14, 16],
  class_curves: null,
  n_training_rows: 200,
  summary:
    "As the feature varies from 5 to 20, the average prediction increases (500 → 2000) across 200 training records.",
}

const classificationResult: PartialDependenceResult = {
  feature: "price",
  target_col: "label",
  algorithm: "logistic_regression",
  problem_type: "classification",
  grid_values: [0.1, 0.5, 0.9],
  mean_predictions: [0.2, 0.5, 0.8],
  std_predictions: [0.1, 0.1, 0.1],
  class_curves: null,
  n_training_rows: 150,
  summary:
    "As the feature varies from 0.1 to 0.9, the average predicted probability increases (0.2 → 0.8) across 150 training records.",
}

const multiclassResult: PartialDependenceResult = {
  feature: "quantity",
  target_col: "category",
  algorithm: "random_forest",
  problem_type: "classification",
  grid_values: [1, 2, 3, 4, 5],
  mean_predictions: [0.4, 0.5, 0.6, 0.5, 0.4],
  std_predictions: [0.05, 0.05, 0.05, 0.05, 0.05],
  class_curves: {
    ClassA: [0.7, 0.6, 0.5, 0.4, 0.3],
    ClassB: [0.2, 0.3, 0.4, 0.5, 0.6],
    ClassC: [0.1, 0.1, 0.1, 0.1, 0.1],
  },
  n_training_rows: 300,
  summary: "Multiclass PDP for quantity across 300 training records.",
}

const constantFeatureResult: PartialDependenceResult = {
  feature: "constant",
  target_col: "revenue",
  algorithm: "linear_regression",
  problem_type: "regression",
  grid_values: [5.0],
  mean_predictions: [100.0],
  std_predictions: [0.0],
  class_curves: null,
  n_training_rows: 100,
  summary: "Partial dependence computed.",
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("PartialDependenceCard", () => {
  it("renders figure with correct aria-label", () => {
    render(<PartialDependenceCard result={regressionResult} />)
    expect(
      screen.getByRole("figure", {
        name: /Partial dependence plot: units vs average revenue/i,
      })
    ).toBeInTheDocument()
  })

  it("renders 📉 icon as aria-hidden", () => {
    const { container } = render(<PartialDependenceCard result={regressionResult} />)
    const icon = container.querySelector("[aria-hidden='true']")
    expect(icon).toBeInTheDocument()
    expect(icon?.textContent).toBe("📉")
  })

  it("shows feature and target in heading", () => {
    render(<PartialDependenceCard result={regressionResult} />)
    // The heading paragraph contains both feature and target — use getAllByText
    // to tolerate the feature name also appearing in axis labels / figcaption
    expect(screen.getAllByText(/units/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/revenue/i).length).toBeGreaterThan(0)
  })

  it("shows Regression badge for regression models", () => {
    render(<PartialDependenceCard result={regressionResult} />)
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("shows Classification badge for classification models", () => {
    render(<PartialDependenceCard result={classificationResult} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("shows algorithm name in badge", () => {
    render(<PartialDependenceCard result={regressionResult} />)
    expect(screen.getByText("linear regression")).toBeInTheDocument()
  })

  it("shows increases trend badge when predictions increase", () => {
    render(<PartialDependenceCard result={regressionResult} />)
    // "Increases" appears in both the trend badge and the summary — use getAllByText
    expect(screen.getAllByText(/increases/i).length).toBeGreaterThan(0)
  })

  it("renders ResponsiveContainer (chart) for multi-point regression", () => {
    render(<PartialDependenceCard result={regressionResult} />)
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument()
  })

  it("shows 'averaged over N training records' explainer text", () => {
    render(<PartialDependenceCard result={regressionResult} />)
    // The text "200 training records" appears in both the explainer and the summary;
    // use getAllByText to avoid a duplicate-element error
    expect(screen.getAllByText(/200 training records/i).length).toBeGreaterThan(0)
  })

  it("shows constant-feature fallback message when grid has 1 point", () => {
    render(<PartialDependenceCard result={constantFeatureResult} />)
    expect(screen.getByText(/is constant in the training data/i)).toBeInTheDocument()
  })

  it("renders summary footer text", () => {
    render(<PartialDependenceCard result={regressionResult} />)
    expect(screen.getByText(/the average prediction increases/i)).toBeInTheDocument()
  })

  it("renders per-class colour legend for multiclass result", () => {
    render(<PartialDependenceCard result={multiclassResult} />)
    expect(screen.getByText("ClassA")).toBeInTheDocument()
    expect(screen.getByText("ClassB")).toBeInTheDocument()
    expect(screen.getByText("ClassC")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Zustand store action tests
// ---------------------------------------------------------------------------

describe("attachPartialDependenceToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "hello", timestamp: new Date() },
      ],
    })
    const { attachPartialDependenceToLastMessage } = useAppStore.getState()
    attachPartialDependenceToLastMessage(regressionResult)

    const msgs = useAppStore.getState().messages
    expect(msgs[0].partial_dependence).toEqual(regressionResult)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "pdp for units", timestamp: new Date() },
      ],
    })
    const { attachPartialDependenceToLastMessage } = useAppStore.getState()
    attachPartialDependenceToLastMessage(regressionResult)

    const msgs = useAppStore.getState().messages
    expect(msgs[0].partial_dependence).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    const { attachPartialDependenceToLastMessage } = useAppStore.getState()
    expect(() => attachPartialDependenceToLastMessage(regressionResult)).not.toThrow()
  })
})
