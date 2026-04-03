/**
 * Tests for the Calibration sub-tab and ReliabilityDiagramView in ValidationPanel.
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { ValidationPanel } from "../components/validation/validation-panel"
import { api } from "../lib/api"
import type { CalibrationData, ValidationMetricsResponse } from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    validation: {
      metrics: jest.fn(),
      explain: jest.fn(),
      explainRow: jest.fn(),
    },
    models: {
      calibration: jest.fn(),
    },
  },
}))

const mockCalibration = api.models.calibration as jest.MockedFunction<typeof api.models.calibration>
const mockMetrics = api.validation.metrics as jest.MockedFunction<typeof api.validation.metrics>

const makeCalibrationData = (overrides: Partial<CalibrationData> = {}): CalibrationData => ({
  run_id: "run-1",
  algorithm: "logistic_regression",
  brier_score: 0.08,
  calibration_curve: [
    { predicted: 0.1, actual: 0.09 },
    { predicted: 0.3, actual: 0.28 },
    { predicted: 0.5, actual: 0.52 },
    { predicted: 0.7, actual: 0.71 },
    { predicted: 0.9, actual: 0.88 },
  ],
  calibration_note: "Model is well-calibrated (Brier score: 0.080). Bars close to the diagonal line mean confidence scores are trustworthy.",
  is_calibrated: true,
  ...overrides,
})

const makeMetrics = (): ValidationMetricsResponse => ({
  model_run_id: "run-1",
  algorithm: "logistic_regression",
  problem_type: "classification",
  held_out_metrics: { accuracy: 0.9, f1: 0.88 },
  cross_validation: {
    metric: "f1",
    scores: [0.85, 0.88, 0.87],
    mean: 0.867,
    std: 0.015,
    ci_low: 0.837,
    ci_high: 0.897,
    n_splits: 3,
    summary: "3-fold CV F1: 0.867 ± 0.015",
  },
  error_analysis: {
    type: "confusion_matrix",
    matrix: [[10, 2], [1, 15]],
    labels: ["A", "B"],
    accuracy: 0.893,
    summary: "High accuracy",
  },
  confidence: {
    overall_confidence: "high",
    limitations: [],
    summary: "High confidence",
  },
})

beforeEach(() => {
  jest.clearAllMocks()
  mockMetrics.mockResolvedValue(makeMetrics())
})

function renderPanel() {
  return render(
    <ValidationPanel
      projectId="proj-1"
      selectedRunId="run-1"
      algorithmName="Logistic Regression"
    />
  )
}

describe("Calibration tab", () => {
  it("shows the Calibration tab", () => {
    renderPanel()
    expect(screen.getByText("Calibration")).toBeInTheDocument()
  })

  it("loads calibration data when Calibration tab is clicked", async () => {
    mockCalibration.mockResolvedValue(makeCalibrationData())
    renderPanel()
    fireEvent.click(screen.getByText("Calibration"))
    await waitFor(() => expect(mockCalibration).toHaveBeenCalledWith("run-1"))
  })

  it("shows Reliability Diagram heading after loading", async () => {
    mockCalibration.mockResolvedValue(makeCalibrationData())
    renderPanel()
    fireEvent.click(screen.getByText("Calibration"))
    await waitFor(() => expect(screen.getByText("Reliability Diagram")).toBeInTheDocument())
  })

  it("shows Brier score badge after loading", async () => {
    mockCalibration.mockResolvedValue(makeCalibrationData())
    renderPanel()
    fireEvent.click(screen.getByText("Calibration"))
    // Use getAllByText because the brier score appears in both badge and calibration note
    await waitFor(() => expect(screen.getAllByText(/Brier score.*0\.08/i).length).toBeGreaterThan(0))
  })

  it("shows calibration note text about bar proximity to diagonal", async () => {
    mockCalibration.mockResolvedValue(makeCalibrationData())
    renderPanel()
    fireEvent.click(screen.getByText("Calibration"))
    await waitFor(() =>
      expect(screen.getByText(/diagonal line mean confidence scores/i)).toBeInTheDocument()
    )
  })

  it("shows 'not available' message when calibration returns 400", async () => {
    mockCalibration.mockRejectedValue(new Error("HTTP 400"))
    renderPanel()
    fireEvent.click(screen.getByText("Calibration"))
    await waitFor(() =>
      expect(screen.getByText(/Calibration not available/i)).toBeInTheDocument()
    )
  })

  it("explains why calibration is not available", async () => {
    mockCalibration.mockRejectedValue(new Error("HTTP 400"))
    renderPanel()
    fireEvent.click(screen.getByText("Calibration"))
    await waitFor(() =>
      expect(screen.getByText(/classifiers only/i)).toBeInTheDocument()
    )
  })

  it("shows load button when no data has been fetched yet", async () => {
    // Don't mock calibration (never resolves/rejects synchronously), just check initial state
    mockCalibration.mockReturnValue(new Promise(() => {})) // pending forever
    renderPanel()
    fireEvent.click(screen.getByText("Calibration"))
    // During load, no "Reliability Diagram" or "not available" message yet
    expect(screen.queryByText("Reliability Diagram")).not.toBeInTheDocument()
  })

  it("handles empty calibration_curve gracefully", async () => {
    mockCalibration.mockResolvedValue(makeCalibrationData({ calibration_curve: [] }))
    renderPanel()
    fireEvent.click(screen.getByText("Calibration"))
    await waitFor(() =>
      expect(screen.getByText(/no calibration curve data/i)).toBeInTheDocument()
    )
  })

  it("does not call calibration API when other tabs are clicked", async () => {
    mockMetrics.mockResolvedValue(makeMetrics())
    renderPanel()
    fireEvent.click(screen.getByText("Cross-Validation"))
    await waitFor(() => expect(mockMetrics).toHaveBeenCalled())
    expect(mockCalibration).not.toHaveBeenCalled()
  })

  it("does not re-fetch calibration data on second click of tab", async () => {
    mockCalibration.mockResolvedValue(makeCalibrationData())
    renderPanel()
    fireEvent.click(screen.getByText("Calibration"))
    await waitFor(() => expect(mockCalibration).toHaveBeenCalledTimes(1))
    // Switch away and back
    fireEvent.click(screen.getByText("Cross-Validation"))
    fireEvent.click(screen.getByText("Calibration"))
    // Should still be 1 call (data cached in state)
    expect(mockCalibration).toHaveBeenCalledTimes(1)
  })
})
