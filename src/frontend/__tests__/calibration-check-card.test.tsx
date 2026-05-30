/**
 * Tests for CalibrationCheckCard component and Zustand store action.
 *
 * Covers (17 tests):
 *  1.  Renders figure with data-testid="calibration-check-card"
 *  2.  Shows verdict badge for well_calibrated
 *  3.  Shows verdict badge for moderate
 *  4.  Shows verdict badge for poorly_calibrated
 *  5.  Shows algorithm badge (human-readable)
 *  6.  Shows target column badge
 *  7.  Shows n_classes badge
 *  8.  Shows n_total badge
 *  9.  Shows ECE value
 * 10.  Shows Brier score value
 * 11.  Shows plain-English summary text
 * 12.  Renders Recharts chart when curve_points has entries
 * 13.  Per-class section visible for n_classes > 2
 * 14.  Per-class section hidden for binary (n_classes == 2)
 * 15.  sr-only figcaption contains verdict and metric text
 * 16.  Store: attachCalibrationCheckToLastMessage attaches to last assistant message
 * 17.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import CalibrationCheckCard from "@/components/models/calibration-check-card"
import type { CalibrationCheckResult } from "@/lib/types"
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

const CURVE_POINTS = [
  { mean_prob: 0.1, frac_positive: 0.08, count: 20, bin_label: "0–20%" },
  { mean_prob: 0.5, frac_positive: 0.49, count: 60, bin_label: "40–60%" },
  { mean_prob: 0.9, frac_positive: 0.91, count: 30, bin_label: "80–100%" },
]

const baseResult: CalibrationCheckResult = {
  model_run_id: "run-1",
  algorithm: "logistic_regression",
  target_col: "churn",
  curve_points: CURVE_POINTS,
  ece: 0.03,
  brier_score: 0.07,
  verdict: "well_calibrated",
  verdict_label: "Well-Calibrated",
  per_class: [
    { class_name: "no", brier_score: 0.06, ece: 0.02, n_positive: 70 },
    { class_name: "yes", brier_score: 0.08, ece: 0.04, n_positive: 30 },
  ],
  n_classes: 2,
  n_total: 100,
  summary:
    "Well-Calibrated. ECE 0.030 — the model's probability scores closely match actual outcome rates. Brier score: 0.070.",
}

const moderateResult: CalibrationCheckResult = {
  ...baseResult,
  ece: 0.09,
  brier_score: 0.15,
  verdict: "moderate",
  verdict_label: "Moderately Calibrated",
  summary: "Moderately Calibrated. ECE 0.090.",
}

const poorResult: CalibrationCheckResult = {
  ...baseResult,
  ece: 0.22,
  brier_score: 0.28,
  verdict: "poorly_calibrated",
  verdict_label: "Poorly Calibrated",
  summary: "Poorly Calibrated. ECE 0.220.",
}

const multiclassResult: CalibrationCheckResult = {
  ...baseResult,
  n_classes: 3,
  per_class: [
    { class_name: "A", brier_score: 0.05, ece: 0.02, n_positive: 40 },
    { class_name: "B", brier_score: 0.07, ece: 0.03, n_positive: 35 },
    { class_name: "C", brier_score: 0.09, ece: 0.04, n_positive: 25 },
  ],
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("CalibrationCheckCard", () => {
  it("renders figure with data-testid", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.getByTestId("calibration-check-card")).toBeInTheDocument()
  })

  it("shows Well-Calibrated verdict badge", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.getByTestId("calibration-verdict-badge")).toHaveTextContent(
      "Well-Calibrated"
    )
  })

  it("shows Moderately Calibrated verdict badge", () => {
    render(<CalibrationCheckCard result={moderateResult} />)
    expect(screen.getByTestId("calibration-verdict-badge")).toHaveTextContent(
      "Moderately Calibrated"
    )
  })

  it("shows Poorly Calibrated verdict badge", () => {
    render(<CalibrationCheckCard result={poorResult} />)
    expect(screen.getByTestId("calibration-verdict-badge")).toHaveTextContent(
      "Poorly Calibrated"
    )
  })

  it("shows human-readable algorithm name", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.getByText("Logistic Regression")).toBeInTheDocument()
  })

  it("shows target column badge", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.getByText("churn")).toBeInTheDocument()
  })

  it("shows n_classes badge", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.getByText(/2 classes/)).toBeInTheDocument()
  })

  it("shows n_total badge", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.getByText(/n=100/)).toBeInTheDocument()
  })

  it("shows ECE value", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.getByText("0.030")).toBeInTheDocument()
  })

  it("shows Brier score value", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.getByText("0.070")).toBeInTheDocument()
  })

  it("shows plain-English summary text", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.getByText(/probability scores closely match/i)).toBeInTheDocument()
  })

  it("renders recharts chart when curve_points is non-empty", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument()
  })

  it("shows per-class section for multiclass (n_classes > 2)", () => {
    render(<CalibrationCheckCard result={multiclassResult} />)
    expect(screen.getByText("Per-class calibration")).toBeInTheDocument()
    expect(screen.getByText("A")).toBeInTheDocument()
    expect(screen.getByText("B")).toBeInTheDocument()
  })

  it("hides per-class section for binary (n_classes == 2)", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    expect(screen.queryByText("Per-class calibration")).not.toBeInTheDocument()
  })

  it("sr-only figcaption contains verdict label and metric values", () => {
    render(<CalibrationCheckCard result={baseResult} />)
    const caption = document.querySelector("figcaption.sr-only")
    expect(caption?.textContent).toMatch(/Well-Calibrated/)
    expect(caption?.textContent).toMatch(/0\.030/)
  })
})

// ---------------------------------------------------------------------------
// Store tests
// ---------------------------------------------------------------------------

describe("attachCalibrationCheckToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches calibration_check to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "check calibration" },
        { role: "assistant", content: "Here you go." },
      ],
    })
    useAppStore.getState().attachCalibrationCheckToLastMessage(baseResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].calibration_check).toEqual(baseResult)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "check calibration" }],
    })
    useAppStore.getState().attachCalibrationCheckToLastMessage(baseResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].calibration_check).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachCalibrationCheckToLastMessage(baseResult)
    ).not.toThrow()
  })
})
