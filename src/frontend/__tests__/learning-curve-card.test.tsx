/**
 * Tests for LearningCurveCard component and store action.
 *
 * Covers:
 *  1.  Renders figure with correct aria-label
 *  2.  Renders 📈 icon (aria-hidden)
 *  3.  Shows "Converged" badge when converged=true
 *  4.  Shows "Still Learning" badge when converged=false
 *  5.  Shows algorithm name in header badge
 *  6.  Shows row count in header badge
 *  7.  Renders a Recharts ResponsiveContainer (chart)
 *  8.  Shows best val score box
 *  9.  Shows convergence plateau when converged + plateau_pct set
 * 10.  Does not show plateau box when converged=false
 * 11.  Renders recommendation text
 * 12.  Renders summary text
 * 13.  Shows R² metric label for regression
 * 14.  Shows accuracy metric label for classification
 * 15.  Store: attachLearningCurveToLastMessage attaches to last assistant message
 * 16.  Store: does not attach to user message
 * 17.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { LearningCurveCard } from "@/components/chat/learning-curve-card"
import type { LearningCurveResult } from "@/lib/types"
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

const convergingResult: LearningCurveResult = {
  sizes_pct: [20, 40, 60, 80, 100],
  train_scores: [0.97, 0.96, 0.95, 0.94, 0.94],
  val_scores: [0.82, 0.88, 0.90, 0.91, 0.91],
  converged: true,
  plateau_pct: 60,
  best_val_score: 0.91,
  metric_label: "R²",
  metric_key: "r2",
  n_total: 500,
  algorithm: "linear_regression",
  algorithm_name: "Linear Regression",
  recommendation: "Your model appears to have converged around 60% of your data.",
  summary: "With 500 training rows, R² = 0.910. Model has converged — more data won't help much.",
}

const stillLearningResult: LearningCurveResult = {
  sizes_pct: [20, 40, 60, 80, 100],
  train_scores: [0.85, 0.83, 0.82, 0.81, 0.80],
  val_scores: [0.60, 0.68, 0.73, 0.77, 0.80],
  converged: false,
  plateau_pct: null,
  best_val_score: 0.80,
  metric_label: "accuracy",
  metric_key: "accuracy",
  n_total: 200,
  algorithm: "logistic_regression",
  algorithm_name: "Logistic Regression",
  recommendation: "The validation accuracy is still climbing — more training data would likely improve your model.",
  summary: "With 200 training rows, accuracy = 80%. More data would likely improve accuracy.",
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("LearningCurveCard", () => {
  it("renders figure with aria-label", () => {
    render(<LearningCurveCard result={convergingResult} />)
    expect(
      screen.getByRole("figure", { name: /learning curve analysis/i })
    ).toBeInTheDocument()
  })

  it("renders 📈 icon as aria-hidden", () => {
    render(<LearningCurveCard result={convergingResult} />)
    const icon = screen.getByText("📈")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("shows Converged badge when converged=true", () => {
    render(<LearningCurveCard result={convergingResult} />)
    expect(screen.getByText("Converged")).toBeInTheDocument()
  })

  it("shows Still Learning badge when converged=false", () => {
    render(<LearningCurveCard result={stillLearningResult} />)
    expect(screen.getByText("Still Learning")).toBeInTheDocument()
  })

  it("shows algorithm name in header badge", () => {
    render(<LearningCurveCard result={convergingResult} />)
    expect(screen.getByText(/Linear Regression/)).toBeInTheDocument()
  })

  it("shows row count in header badge", () => {
    render(<LearningCurveCard result={convergingResult} />)
    expect(screen.getByText(/500 rows/)).toBeInTheDocument()
  })

  it("renders a Recharts ResponsiveContainer", () => {
    render(<LearningCurveCard result={convergingResult} />)
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument()
  })

  it("shows best val score box", () => {
    render(<LearningCurveCard result={convergingResult} />)
    // 0.910 formatted
    expect(screen.getByText("0.910")).toBeInTheDocument()
  })

  it("shows convergence plateau box when converged + plateau_pct set", () => {
    render(<LearningCurveCard result={convergingResult} />)
    expect(screen.getByText(/60% of data/)).toBeInTheDocument()
  })

  it("does not show plateau box when converged=false", () => {
    render(<LearningCurveCard result={stillLearningResult} />)
    expect(screen.queryByText(/% of data/)).not.toBeInTheDocument()
  })

  it("renders recommendation text", () => {
    render(<LearningCurveCard result={convergingResult} />)
    expect(
      screen.getByText(/converged around 60% of your data/)
    ).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<LearningCurveCard result={convergingResult} />)
    expect(
      screen.getByText(/With 500 training rows, R² = 0.910/)
    ).toBeInTheDocument()
  })

  it("shows R² metric label for regression", () => {
    render(<LearningCurveCard result={convergingResult} />)
    expect(screen.getAllByText(/R²/).length).toBeGreaterThan(0)
  })

  it("shows accuracy metric label for classification", () => {
    render(<LearningCurveCard result={stillLearningResult} />)
    expect(screen.getAllByText(/accuracy/i).length).toBeGreaterThan(0)
  })
})

// ---------------------------------------------------------------------------
// Store: attachLearningCurveToLastMessage
// ---------------------------------------------------------------------------

describe("attachLearningCurveToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "would more data help?" },
        { role: "assistant", content: "Let me check." },
      ],
    })
    useAppStore.getState().attachLearningCurveToLastMessage(convergingResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].learning_curve).toEqual(convergingResult)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "would more data help?" }],
    })
    useAppStore.getState().attachLearningCurveToLastMessage(convergingResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].learning_curve).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() => {
      useAppStore.getState().attachLearningCurveToLastMessage(convergingResult)
    }).not.toThrow()
  })
})
