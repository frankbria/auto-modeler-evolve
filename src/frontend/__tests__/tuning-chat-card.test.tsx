/**
 * Tests for TuningChatCard and Zustand store action.
 *
 * Covers:
 *  1.  Renders figure with aria-label "Hyperparameter tuning result"
 *  2.  Shows 🔧 icon (aria-hidden)
 *  3.  Shows "Hyperparameter Tuning" heading
 *  4.  Shows algorithm_name badge
 *  5.  Shows problem_type badge when present
 *  6.  Shows "Improved" badge when improved=true
 *  7.  Shows "Unchanged" badge when improved=false
 *  8.  Shows improvement_pct badge when >= 0.01
 *  9.  Renders before/after metrics table (data-testid="tuning-metrics-table")
 * 10.  Renders best params section (data-testid="tuning-best-params")
 * 11.  Renders summary text (data-testid="tuning-summary")
 * 12.  Renders figcaption footer
 * 13.  Not-tunable state: shows slate explanation, no metrics table
 * 14.  Store: attachTuneChatToLastMessage attaches to last assistant message
 * 15.  Store: does not attach to user message
 * 16.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { TuningChatCard } from "@/components/models/tuning-chat-card"
import type { TuningChatResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const improvedResult: TuningChatResult = {
  tunable: true,
  algorithm: "random_forest",
  algorithm_name: "Random Forest",
  problem_type: "regression",
  original_run_id: "run-1",
  tuned_run_id: "run-2",
  original_metrics: { r2: 0.72, rmse: 0.45 },
  tuned_metrics: { r2: 0.81, rmse: 0.38 },
  best_params: { n_estimators: 200, max_depth: 10 },
  improved: true,
  improvement_pct: 12.5,
  summary: "Tuning improved R² from 0.720 to 0.810 (+12.5%).",
}

const unchangedResult: TuningChatResult = {
  tunable: true,
  algorithm: "gradient_boosting",
  algorithm_name: "Gradient Boosting",
  problem_type: "classification",
  original_run_id: "run-3",
  tuned_run_id: "run-4",
  original_metrics: { accuracy: 0.85 },
  tuned_metrics: { accuracy: 0.85 },
  best_params: { learning_rate: 0.1 },
  improved: false,
  improvement_pct: 0,
  summary: "No improvement found. Current settings are already near-optimal.",
}

const notTunableResult: TuningChatResult = {
  tunable: false,
  algorithm: "linear_regression",
  algorithm_name: "Linear Regression",
  summary: "Linear Regression has no hyperparameters to tune.",
}

// ─── Component tests ───────────────────────────────────────────────────────

describe("TuningChatCard — improved result", () => {
  beforeEach(() => render(<TuningChatCard result={improvedResult} />))

  it("renders figure with correct aria-label", () => {
    expect(
      screen.getByRole("figure", { name: "Hyperparameter tuning result" }),
    ).toBeInTheDocument()
  })

  it("shows 🔧 icon as aria-hidden", () => {
    const icon = screen.getByText("🔧")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("shows Hyperparameter Tuning heading", () => {
    expect(screen.getByText("Hyperparameter Tuning")).toBeInTheDocument()
  })

  it("shows algorithm_name badge", () => {
    expect(screen.getByText("Random Forest")).toBeInTheDocument()
  })

  it("shows problem_type badge", () => {
    expect(screen.getByText("regression")).toBeInTheDocument()
  })

  it("shows Improved badge", () => {
    expect(screen.getByTestId("tuning-improvement-badge")).toHaveTextContent("Improved")
  })

  it("shows improvement_pct badge", () => {
    const badge = screen.getByTestId("tuning-pct-badge")
    expect(badge.textContent).toContain("+12.5%")
  })

  it("renders before/after metrics table", () => {
    expect(screen.getByTestId("tuning-metrics-table")).toBeInTheDocument()
    expect(screen.getByText("r2")).toBeInTheDocument()
    expect(screen.getByText("0.7200")).toBeInTheDocument()
    expect(screen.getByText("0.8100")).toBeInTheDocument()
  })

  it("renders best params section", () => {
    const params = screen.getByTestId("tuning-best-params")
    expect(params).toBeInTheDocument()
    expect(params.textContent).toContain("n_estimators")
    expect(params.textContent).toContain("200")
    expect(params.textContent).toContain("max_depth")
    expect(params.textContent).toContain("10")
  })

  it("renders summary text", () => {
    expect(screen.getByTestId("tuning-summary")).toHaveTextContent(
      "Tuning improved R² from 0.720 to 0.810 (+12.5%).",
    )
  })

  it("renders figcaption footer", () => {
    expect(screen.getByText(/RandomizedSearchCV/)).toBeInTheDocument()
  })
})

describe("TuningChatCard — unchanged result", () => {
  beforeEach(() => render(<TuningChatCard result={unchangedResult} />))

  it("shows Unchanged badge", () => {
    expect(screen.getByTestId("tuning-improvement-badge")).toHaveTextContent("Unchanged")
  })

  it("does not show pct badge when improvement_pct is 0", () => {
    expect(screen.queryByTestId("tuning-pct-badge")).not.toBeInTheDocument()
  })

  it("renders metrics table", () => {
    expect(screen.getByTestId("tuning-metrics-table")).toBeInTheDocument()
    expect(screen.getByText("accuracy")).toBeInTheDocument()
  })
})

describe("TuningChatCard — not tunable", () => {
  beforeEach(() => render(<TuningChatCard result={notTunableResult} />))

  it("shows algorithm_name in explanation", () => {
    // algorithm_name appears in both the badge and the explanation — check it appears at least once
    expect(screen.getAllByText("Linear Regression").length).toBeGreaterThanOrEqual(1)
  })

  it("does not render metrics table", () => {
    expect(screen.queryByTestId("tuning-metrics-table")).not.toBeInTheDocument()
  })

  it("does not render best params", () => {
    expect(screen.queryByTestId("tuning-best-params")).not.toBeInTheDocument()
  })

  it("renders summary", () => {
    expect(screen.getByTestId("tuning-summary")).toHaveTextContent(
      "Linear Regression has no hyperparameters to tune.",
    )
  })
})

// ─── Zustand store tests ───────────────────────────────────────────────────

describe("Zustand store — attachTuneChatToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "tune my model" },
        { id: "2", role: "assistant", content: "Tuning now…" },
      ],
    })
    useAppStore.getState().attachTuneChatToLastMessage(improvedResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].tune_chat).toEqual(improvedResult)
  })

  it("does not attach when last message is from user", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Hello" },
        { id: "2", role: "user", content: "tune my model" },
      ],
    })
    useAppStore.getState().attachTuneChatToLastMessage(improvedResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].tune_chat).toBeUndefined()
    expect(msgs[0].tune_chat).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    expect(() =>
      useAppStore.getState().attachTuneChatToLastMessage(improvedResult),
    ).not.toThrow()
  })
})
