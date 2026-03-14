/**
 * Tests for ModelTrainingPanel — algorithm recommendations, training start, run display.
 *
 * EventSource (SSE training stream) is stubbed globally.
 * API calls are fully mocked.
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { ModelTrainingPanel } from "../components/models/model-training-panel"
import { api } from "../lib/api"
import type { ModelRecommendation, ModelRun, ModelComparison } from "../lib/types"

// Stub EventSource globally
class MockEventSource {
  static instances: MockEventSource[] = []
  url: string
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  close = jest.fn()
  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
;(global as any).EventSource = MockEventSource

jest.mock("../lib/api", () => ({
  api: {
    models: {
      recommendations: jest.fn(),
      runs: jest.fn(),
      train: jest.fn(),
      compare: jest.fn(),
      comparisonRadar: jest.fn(),
      select: jest.fn(),
      trainingStreamUrl: jest.fn().mockReturnValue("/api/models/proj-1/training-stream"),
      downloadUrl: jest.fn().mockReturnValue("/api/models/run-1/download"),
      reportUrl: jest.fn().mockReturnValue("/api/models/run-1/report"),
    },
  },
}))

const mockRecs = api.models.recommendations as jest.MockedFunction<typeof api.models.recommendations>
const mockRuns = api.models.runs as jest.MockedFunction<typeof api.models.runs>
const mockTrain = api.models.train as jest.MockedFunction<typeof api.models.train>
const mockSelect = api.models.select as jest.MockedFunction<typeof api.models.select>
const mockCompare = api.models.compare as jest.MockedFunction<typeof api.models.compare>
const mockRadar = api.models.comparisonRadar as jest.MockedFunction<typeof api.models.comparisonRadar>

const makeRec = (overrides: Partial<ModelRecommendation> = {}): ModelRecommendation => ({
  algorithm: "linear_regression",
  name: "Linear Regression",
  description: "A simple, fast baseline",
  plain_english: "Draws the best-fit line through your data",
  best_for: "Interpretability",
  recommended_because: "Your dataset has low complexity",
  ...overrides,
})

const makeRun = (overrides: Partial<ModelRun> = {}): ModelRun => ({
  id: "run-1",
  algorithm: "linear_regression",
  status: "done",
  metrics: { r2: 0.85, mae: 0.12, rmse: 0.18, train_size: 160, test_size: 40 },
  summary: "Linear Regression achieved R² of 0.85",
  is_selected: false,
  is_deployed: false,
  created_at: "2026-01-01T00:00:00",
  training_duration_ms: 200,
  error_message: null,
  ...overrides,
})

const makeComparison = (overrides: Partial<ModelComparison> = {}): ModelComparison => ({
  project_id: "proj-1",
  problem_type: "regression",
  models: [makeRun()],
  recommendation: {
    model_run_id: "run-1",
    algorithm: "linear_regression",
    reason: "Best R²",
  },
  ...overrides,
})

const defaultRecsResponse = {
  project_id: "proj-1",
  problem_type: "regression",
  target_column: "revenue",
  n_rows: 200,
  n_features: 5,
  recommendations: [
    makeRec(),
    makeRec({ algorithm: "random_forest_regressor", name: "Random Forest" }),
  ],
}

beforeEach(() => {
  jest.clearAllMocks()
  MockEventSource.instances = []
  mockRecs.mockResolvedValue(defaultRecsResponse)
  mockRuns.mockResolvedValue({ project_id: "proj-1", runs: [] })
  mockCompare.mockResolvedValue(makeComparison())
  mockRadar.mockResolvedValue(null)
})

describe("ModelTrainingPanel — loading and error states", () => {
  it("shows loading text initially", () => {
    mockRecs.mockReturnValue(new Promise(() => {}))
    render(<ModelTrainingPanel projectId="proj-1" />)
    expect(screen.getByText(/loading model recommendations/i)).toBeInTheDocument()
  })

  it("shows error state when recommendations fail and no runs exist", async () => {
    mockRecs.mockRejectedValue(new Error("No target column"))
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/cannot load recommendations/i)).toBeInTheDocument()
    )
  })
})

describe("ModelTrainingPanel — recommendations display", () => {
  it("shows target column badge after loading", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText("revenue")).toBeInTheDocument()
    )
  })

  it("shows problem type badge", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText("regression")).toBeInTheDocument()
    )
  })

  it("shows recommendation names", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => {
      expect(screen.getByText("Linear Regression")).toBeInTheDocument()
      expect(screen.getByText("Random Forest")).toBeInTheDocument()
    })
  })

  it("enables Train button with default algo selections", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText("Linear Regression")).toBeInTheDocument()
    )
    // Button label is "Train N models" depending on selection count
    expect(screen.getByRole("button", { name: /train \d+ models/i })).not.toBeDisabled()
  })
})

describe("ModelTrainingPanel — training", () => {
  it("calls api.models.train with selected algorithms", async () => {
    // First call (mount) returns empty runs → shows algo selection + Train button
    // Second call (after train) returns pending run
    mockRuns
      .mockResolvedValueOnce({ project_id: "proj-1", runs: [] })
      .mockResolvedValueOnce({ project_id: "proj-1", runs: [makeRun({ status: "pending" })] })

    mockTrain.mockResolvedValue({
      project_id: "proj-1",
      model_run_ids: ["run-1"],
      algorithms: ["linear_regression"],
      status: "started",
      message: "Training started",
    })

    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /train \d+ models/i })).toBeInTheDocument()
    )
    fireEvent.click(screen.getByRole("button", { name: /train \d+ models/i }))
    await waitFor(() =>
      expect(mockTrain).toHaveBeenCalledWith(
        "proj-1",
        expect.arrayContaining(["linear_regression"])
      )
    )
  })
})

describe("ModelTrainingPanel — run display", () => {
  it("shows completed run summary", async () => {
    mockRuns.mockResolvedValue({
      project_id: "proj-1",
      runs: [makeRun({ summary: "R² of 0.85 — solid fit" })],
    })
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/R² of 0.85/)).toBeInTheDocument()
    )
  })

  it("shows Select button for non-selected completed runs", async () => {
    mockRuns.mockResolvedValue({
      project_id: "proj-1",
      runs: [makeRun({ is_selected: false })],
    })
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /select this model/i })).toBeInTheDocument()
    )
  })

  it("calls api.models.select on Select button click", async () => {
    const selectedRun = makeRun({ is_selected: true })
    mockRuns
      .mockResolvedValueOnce({ project_id: "proj-1", runs: [makeRun({ is_selected: false })] })
      .mockResolvedValueOnce({ project_id: "proj-1", runs: [selectedRun] })
    mockSelect.mockResolvedValue(selectedRun)

    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /select this model/i })).toBeInTheDocument()
    )
    fireEvent.click(screen.getByRole("button", { name: /select this model/i }))
    await waitFor(() =>
      expect(mockSelect).toHaveBeenCalledWith("run-1")
    )
  })

  it("shows failed run with error message", async () => {
    mockRuns.mockResolvedValue({
      project_id: "proj-1",
      runs: [
        makeRun({
          status: "failed",
          error_message: "Training diverged",
          metrics: null,
          summary: null,
        }),
      ],
    })
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/training diverged/i)).toBeInTheDocument()
    )
  })

  it("loads comparison summary when done runs exist", async () => {
    mockRuns.mockResolvedValue({
      project_id: "proj-1",
      runs: [makeRun({ status: "done" })],
    })
    mockCompare.mockResolvedValue(makeComparison())
    render(<ModelTrainingPanel projectId="proj-1" />)
    // Recommendation label should appear in the comparison panel
    await waitFor(() =>
      expect(screen.getByText(/best r²/i)).toBeInTheDocument()
    )
  })
})
