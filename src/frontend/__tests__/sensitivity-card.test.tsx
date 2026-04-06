/**
 * Tests for SensitivityCard component and store action.
 *
 * Covers:
 *  1.  Renders figure with correct aria-label
 *  2.  Renders 🎚️ icon (aria-hidden)
 *  3.  Shows feature and target in heading
 *  4.  Shows Regression badge for regression models
 *  5.  Shows Classification badge for classification models
 *  6.  Shows change % badge for regression
 *  7.  Shows min/max prediction badges for regression
 *  8.  Renders a Recharts ResponsiveContainer (chart) for regression with numeric curve
 *  9.  Renders prediction table for classification without numeric confidences
 * 10.  Renders summary footer text
 * 11.  Store: attachSensitivityToLastMessage attaches to last assistant message
 * 12.  Store: does not attach to user message
 * 13.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { SensitivityCard } from "@/components/deploy/sensitivity-card"
import type { SensitivityResult } from "@/lib/types"
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

const regressionResult: SensitivityResult = {
  feature: "units",
  target_column: "revenue",
  problem_type: "regression",
  values: [5, 10, 15, 20],
  predictions: [500, 1000, 1500, 2000],
  confidences: [null, null, null, null],
  min_pred: 500,
  max_pred: 2000,
  change_pct: 300.0,
  summary:
    "As units varies from 5 to 20 across 4 steps, revenue increases by 300.0% (from 500 to 2000).",
}

const classificationResult: SensitivityResult = {
  feature: "price",
  target_column: "label",
  problem_type: "classification",
  values: [10, 20, 30],
  predictions: ["cat", "dog", "cat"],
  confidences: [null, null, null],
  min_pred: null,
  max_pred: null,
  change_pct: null,
  summary:
    "As price varies from 10 to 30, the predicted class switches between: cat, dog.",
}

const classificationWithConfidence: SensitivityResult = {
  feature: "score",
  target_column: "outcome",
  problem_type: "classification",
  values: [1, 2, 3],
  predictions: ["yes", "yes", "no"],
  confidences: [0.85, 0.72, 0.61],
  min_pred: null,
  max_pred: null,
  change_pct: null,
  summary: "As score varies from 1 to 3, the predicted class switches between: yes, no.",
}

// ---------------------------------------------------------------------------
// Component tests
// ---------------------------------------------------------------------------

describe("SensitivityCard", () => {
  it("renders figure with correct aria-label", () => {
    render(<SensitivityCard result={regressionResult} />)
    const fig = screen.getByRole("figure")
    expect(fig).toHaveAttribute("aria-label", expect.stringContaining("units"))
    expect(fig).toHaveAttribute("aria-label", expect.stringContaining("revenue"))
  })

  it("renders the 🎚️ icon as aria-hidden", () => {
    render(<SensitivityCard result={regressionResult} />)
    const icon = screen.getByText("🎚️")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("shows feature name in heading", () => {
    render(<SensitivityCard result={regressionResult} />)
    const items = screen.getAllByText(/units/i)
    expect(items.length).toBeGreaterThan(0)
  })

  it("shows target column in heading", () => {
    render(<SensitivityCard result={regressionResult} />)
    const headings = screen.getAllByText(/revenue/i)
    expect(headings.length).toBeGreaterThan(0)
  })

  it("shows Regression badge for regression model", () => {
    render(<SensitivityCard result={regressionResult} />)
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("shows Classification badge for classification model", () => {
    render(<SensitivityCard result={classificationResult} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("shows change % badge for regression", () => {
    render(<SensitivityCard result={regressionResult} />)
    expect(screen.getByText(/300\.0%\s*range/i)).toBeInTheDocument()
  })

  it("does not show change % badge when change_pct is null", () => {
    render(<SensitivityCard result={classificationResult} />)
    expect(screen.queryByText(/range/i)).not.toBeInTheDocument()
  })

  it("shows min and max prediction for regression", () => {
    render(<SensitivityCard result={regressionResult} />)
    // min_pred = 500 → "500", max_pred = 2000 → "2k"
    const fiveHundreds = screen.getAllByText("500")
    expect(fiveHundreds.length).toBeGreaterThan(0)
    const twoKs = screen.getAllByText("2.0k")
    expect(twoKs.length).toBeGreaterThan(0)
  })

  it("does not show min/max for classification", () => {
    render(<SensitivityCard result={classificationResult} />)
    expect(screen.queryByText(/Min revenue/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Max revenue/i)).not.toBeInTheDocument()
  })

  it("renders a Recharts chart container for regression", () => {
    render(<SensitivityCard result={regressionResult} />)
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument()
  })

  it("renders a Recharts chart container when confidences are provided", () => {
    render(<SensitivityCard result={classificationWithConfidence} />)
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument()
  })

  it("renders prediction table for classification without confidences", () => {
    render(<SensitivityCard result={classificationResult} />)
    // Table rows contain the predicted class labels (cat appears twice, dog once)
    const catEls = screen.getAllByText("cat")
    expect(catEls.length).toBeGreaterThan(0)
    const dogEls = screen.getAllByText("dog")
    expect(dogEls.length).toBeGreaterThan(0)
  })

  it("renders the summary footer", () => {
    render(<SensitivityCard result={regressionResult} />)
    expect(screen.getByText(/units varies from 5 to 20/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachSensitivityToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        {
          role: "assistant",
          content: "Here is the sensitivity analysis.",
          timestamp: new Date().toISOString(),
        },
      ],
    })
  })

  it("attaches sensitivity to the last assistant message", () => {
    useAppStore.getState().attachSensitivityToLastMessage(regressionResult)
    const messages = useAppStore.getState().messages
    expect(messages[0].sensitivity).toEqual(regressionResult)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [
        {
          role: "user",
          content: "sensitivity analysis on units",
          timestamp: new Date().toISOString(),
        },
      ],
    })
    useAppStore.getState().attachSensitivityToLastMessage(regressionResult)
    const messages = useAppStore.getState().messages
    expect(messages[0].sensitivity).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachSensitivityToLastMessage(regressionResult)
    ).not.toThrow()
  })
})
