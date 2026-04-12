/**
 * Tests for CalibrationCheckCard component and Zustand store action.
 *
 * Covers:
 *  1.  Renders region with correct aria-label
 *  2.  Renders 🎯 icon (aria-hidden)
 *  3.  Shows "Confidence Calibration Check" heading
 *  4.  Shows "Excellent" quality badge for brier_score < 0.1
 *  5.  Shows "Good" quality badge for brier_score 0.1–0.2
 *  6.  Shows "Needs attention" quality badge for brier_score ≥ 0.2
 *  7.  Shows algorithm badge
 *  8.  Shows formatted brier score value
 *  9.  Shows plain-English summary text
 * 10.  Renders Recharts chart when calibration_curve has entries
 * 11.  Shows "No calibration curve data" fallback when curve is empty
 * 12.  Shows calibration_note when present
 * 13.  Store: attachCalibrationCheckToLastMessage attaches to last assistant message
 * 14.  Store: does not attach to user message
 * 15.  Store: does not crash when messages list is empty
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

const excellentResult: CalibrationCheckResult = {
  run_id: "run-1",
  algorithm: "logistic_regression",
  is_calibrated: true,
  brier_score: 0.075,
  calibration_quality: "excellent",
  calibration_curve: [
    { predicted: 0.1, actual: 0.08 },
    { predicted: 0.5, actual: 0.49 },
    { predicted: 0.9, actual: 0.91 },
  ],
  calibration_note: "Calibrated via isotonic regression.",
  summary:
    "Brier score 0.075 — excellent calibration. When the model says '80% confident', it's right roughly 80% of the time.",
}

const goodResult: CalibrationCheckResult = {
  ...excellentResult,
  brier_score: 0.15,
  calibration_quality: "good",
  summary: "Brier score 0.150 — good calibration.",
}

const poorResult: CalibrationCheckResult = {
  ...excellentResult,
  brier_score: 0.28,
  calibration_quality: "poor",
  summary: "Brier score 0.280 — calibration needs attention.",
}

const emptyCurveResult: CalibrationCheckResult = {
  ...excellentResult,
  calibration_curve: [],
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("CalibrationCheckCard", () => {
  it("renders region with correct aria-label", () => {
    render(<CalibrationCheckCard result={excellentResult} />)
    expect(
      screen.getByRole("region", { name: "Model calibration check" }),
    ).toBeInTheDocument()
  })

  it("renders 🎯 icon with aria-hidden", () => {
    render(<CalibrationCheckCard result={excellentResult} />)
    const icon = screen.getByText("🎯")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("shows heading text", () => {
    render(<CalibrationCheckCard result={excellentResult} />)
    expect(screen.getByText("Confidence Calibration Check")).toBeInTheDocument()
  })

  it("shows Excellent badge for brier_score < 0.1", () => {
    render(<CalibrationCheckCard result={excellentResult} />)
    expect(screen.getByText("Excellent")).toBeInTheDocument()
  })

  it("shows Good badge for brier_score 0.1–0.2", () => {
    render(<CalibrationCheckCard result={goodResult} />)
    expect(screen.getByText("Good")).toBeInTheDocument()
  })

  it("shows Needs attention badge for brier_score ≥ 0.2", () => {
    render(<CalibrationCheckCard result={poorResult} />)
    expect(screen.getByText("Needs attention")).toBeInTheDocument()
  })

  it("shows algorithm badge", () => {
    render(<CalibrationCheckCard result={excellentResult} />)
    expect(screen.getByText("logistic_regression")).toBeInTheDocument()
  })

  it("shows formatted brier score value", () => {
    render(<CalibrationCheckCard result={excellentResult} />)
    expect(screen.getByText("0.075")).toBeInTheDocument()
  })

  it("shows plain-English summary text", () => {
    render(<CalibrationCheckCard result={excellentResult} />)
    expect(screen.getByText(/excellent calibration/i)).toBeInTheDocument()
  })

  it("renders chart when calibration_curve is non-empty", () => {
    render(<CalibrationCheckCard result={excellentResult} />)
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument()
  })

  it("shows fallback message when calibration_curve is empty", () => {
    render(<CalibrationCheckCard result={emptyCurveResult} />)
    expect(
      screen.getByText(/No calibration curve data available/i),
    ).toBeInTheDocument()
  })

  it("shows calibration_note when present", () => {
    render(<CalibrationCheckCard result={excellentResult} />)
    expect(
      screen.getByText("Calibrated via isotonic regression."),
    ).toBeInTheDocument()
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
    useAppStore.getState().attachCalibrationCheckToLastMessage(excellentResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].calibration_check).toEqual(excellentResult)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "check calibration" }],
    })
    useAppStore.getState().attachCalibrationCheckToLastMessage(excellentResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].calibration_check).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachCalibrationCheckToLastMessage(excellentResult),
    ).not.toThrow()
  })
})
