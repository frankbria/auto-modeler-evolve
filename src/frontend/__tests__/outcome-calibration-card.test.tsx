import React from "react"
import { render, screen } from "@testing-library/react"
import OutcomeCalibrationCard from "@/components/deploy/outcome-calibration-card"
import { OutcomeCalibrationResult } from "@/lib/types"

const classificationResult: OutcomeCalibrationResult = {
  deployment_id: "dep-1",
  algorithm: "LogisticRegression",
  target_col: "churn",
  verdict: "well_calibrated",
  summary: "Your model is well-calibrated across 50 production outcomes. ECE = 0.030.",
  n_samples: 50,
  problem_type: "classification",
  ece: 0.03,
  overall_accuracy: 0.82,
  n_filled_buckets: 6,
  buckets: Array.from({ length: 10 }, (_, i) => ({
    lo: i * 0.1,
    hi: (i + 1) * 0.1,
    n: i < 6 ? 8 : 0,
    mean_confidence: i < 6 ? i * 0.1 + 0.05 : null,
    actual_accuracy: i < 6 ? i * 0.1 + 0.04 : null,
    deviation: i < 6 ? 0.01 : null,
  })),
}

const regressionResult: OutcomeCalibrationResult = {
  deployment_id: "dep-2",
  algorithm: "LinearRegression",
  target_col: "revenue",
  verdict: "positive_bias",
  summary: "Your model has a positive bias across 20 outcomes.",
  n_samples: 20,
  problem_type: "regression",
  mean_error: 2.5,
  median_error: 2.4,
  std_error: 0.8,
  mae: 2.5,
  rmse: 2.63,
  error_min: 1.0,
  error_max: 4.0,
  bins: Array.from({ length: 10 }, (_, i) => ({ lo: 1.0 + i * 0.3, hi: 1.3 + i * 0.3, count: i < 7 ? 3 : 0 })),
}

const noDataResult: OutcomeCalibrationResult = {
  deployment_id: "dep-3",
  algorithm: null,
  target_col: null,
  verdict: "no_data",
  summary: "Only 2 feedback record(s) available. Need at least 5.",
  n_samples: 2,
  problem_type: "classification",
}

describe("OutcomeCalibrationCard", () => {
  it("renders the card title", () => {
    render(<OutcomeCalibrationCard result={classificationResult} />)
    expect(screen.getByText(/Prediction Outcome Calibration/i)).toBeInTheDocument()
  })

  it("shows the verdict badge for well_calibrated", () => {
    render(<OutcomeCalibrationCard result={classificationResult} />)
    expect(screen.getByText("Well Calibrated")).toBeInTheDocument()
  })

  it("shows ECE for classification", () => {
    render(<OutcomeCalibrationCard result={classificationResult} />)
    expect(screen.getByText("3.0%")).toBeInTheDocument()
  })

  it("shows overall accuracy for classification", () => {
    render(<OutcomeCalibrationCard result={classificationResult} />)
    expect(screen.getByText("82.0%")).toBeInTheDocument()
  })

  it("shows n_filled_buckets annotation", () => {
    render(<OutcomeCalibrationCard result={classificationResult} />)
    expect(screen.getByText(/6 of 10 confidence buckets have data/i)).toBeInTheDocument()
  })

  it("shows algorithm badge", () => {
    render(<OutcomeCalibrationCard result={classificationResult} />)
    expect(screen.getByText("LogisticRegression")).toBeInTheDocument()
  })

  it("shows target column badge", () => {
    render(<OutcomeCalibrationCard result={classificationResult} />)
    expect(screen.getByText("churn")).toBeInTheDocument()
  })

  it("shows sample count", () => {
    render(<OutcomeCalibrationCard result={classificationResult} />)
    expect(screen.getByText(/50 outcomes/i)).toBeInTheDocument()
  })

  it("renders regression card with positive_bias verdict", () => {
    render(<OutcomeCalibrationCard result={regressionResult} />)
    expect(screen.getByText("Positive Bias")).toBeInTheDocument()
  })

  it("shows MAE for regression", () => {
    render(<OutcomeCalibrationCard result={regressionResult} />)
    expect(screen.getByText("2.500")).toBeInTheDocument()
  })

  it("shows RMSE for regression", () => {
    render(<OutcomeCalibrationCard result={regressionResult} />)
    expect(screen.getByText("2.630")).toBeInTheDocument()
  })

  it("shows positive bias value with sign", () => {
    render(<OutcomeCalibrationCard result={regressionResult} />)
    expect(screen.getByText("+2.500")).toBeInTheDocument()
  })

  it("shows std dev for regression", () => {
    render(<OutcomeCalibrationCard result={regressionResult} />)
    expect(screen.getByText("0.800")).toBeInTheDocument()
  })

  it("renders no_data card with summary", () => {
    render(<OutcomeCalibrationCard result={noDataResult} />)
    expect(screen.getByText(/Only 2 feedback/i)).toBeInTheDocument()
  })

  it("shows No Data verdict badge for no_data", () => {
    render(<OutcomeCalibrationCard result={noDataResult} />)
    expect(screen.getByText("No Data")).toBeInTheDocument()
  })

  it("shows footer for classification", () => {
    render(<OutcomeCalibrationCard result={classificationResult} />)
    expect(screen.getByText(/Built from production feedback/i)).toBeInTheDocument()
  })

  it("shows footer for regression", () => {
    render(<OutcomeCalibrationCard result={regressionResult} />)
    expect(screen.getByText(/Error = actual/i)).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<OutcomeCalibrationCard result={classificationResult} />)
    expect(screen.getByText(/well-calibrated across 50 production outcomes/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

import { useAppStore } from "@/lib/store"

describe("attachOutcomeCalibrationToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Hello", timestamp: new Date().toISOString() },
      ],
    })
  })

  it("attaches outcome_calibration to the last assistant message", () => {
    const { attachOutcomeCalibrationToLastMessage } = useAppStore.getState()
    attachOutcomeCalibrationToLastMessage(classificationResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].outcome_calibration).toEqual(classificationResult)
  })

  it("does not attach if last message is not from assistant", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "Hello", timestamp: new Date().toISOString() },
      ],
    })
    const { attachOutcomeCalibrationToLastMessage } = useAppStore.getState()
    attachOutcomeCalibrationToLastMessage(classificationResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].outcome_calibration).toBeUndefined()
  })

  it("preserves existing message fields when attaching", () => {
    const { attachOutcomeCalibrationToLastMessage } = useAppStore.getState()
    attachOutcomeCalibrationToLastMessage(regressionResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].content).toBe("Hello")
    expect(msgs[0].outcome_calibration).toEqual(regressionResult)
  })
})
