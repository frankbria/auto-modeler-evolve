/**
 * Tests for VersionHistoryCard and model history integration in ModelTrainingPanel.
 *
 * VersionHistoryCard is unexported (private to model-training-panel.tsx), so we
 * test it via the ModelTrainingPanel with a mocked api.models.history response.
 */

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { ModelTrainingPanel } from "../components/models/model-training-panel"
import { api } from "../lib/api"
import type { ModelRun, ModelVersionHistory } from "../lib/types"

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
      history: jest.fn(),
      tune: jest.fn(),
      retrain: jest.fn(),
    },
  },
}))

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mockRecs = (api as any).models.recommendations as jest.Mock
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mockRuns = (api as any).models.runs as jest.Mock
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mockCompare = (api as any).models.compare as jest.Mock
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mockRadar = (api as any).models.comparisonRadar as jest.Mock
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mockHistory = (api as any).models.history as jest.Mock

const defaultRecsResponse = {
  project_id: "proj-1",
  problem_type: "regression",
  target_column: "revenue",
  n_rows: 200,
  n_features: 5,
  recommendations: [
    {
      algorithm: "linear_regression",
      name: "Linear Regression",
      description: "Fast baseline",
      plain_english: "Draws the best-fit line",
      best_for: "Interpretability",
      recommended_because: "Simple dataset",
    },
  ],
}

const makeRun = (overrides: Partial<ModelRun> = {}): ModelRun => ({
  id: "run-1",
  algorithm: "linear_regression",
  status: "done",
  metrics: { r2: 0.85, mae: 0.12, rmse: 0.18, train_size: 160, test_size: 40 },
  summary: "R² = 0.85",
  is_selected: false,
  is_deployed: false,
  created_at: "2026-01-01T00:00:00",
  training_duration_ms: 200,
  error_message: null,
  ...overrides,
})

const makeHistory = (overrides: Partial<ModelVersionHistory> = {}): ModelVersionHistory => ({
  project_id: "proj-1",
  problem_type: "regression",
  primary_metric: "r2",
  primary_metric_label: "R²",
  runs: [],
  trend: "insufficient_data",
  trend_summary: "Not enough training runs to determine a trend yet.",
  best_metric: null,
  latest_metric: null,
  ...overrides,
})

beforeEach(() => {
  jest.clearAllMocks()
  MockEventSource.instances = []
  mockRecs.mockResolvedValue(defaultRecsResponse)
  mockRuns.mockResolvedValue({ project_id: "proj-1", runs: [] })
  mockCompare.mockResolvedValue({
    project_id: "proj-1",
    problem_type: "regression",
    models: [],
    recommendation: null,
  })
  mockRadar.mockResolvedValue(null)
  mockHistory.mockResolvedValue(makeHistory())
})

describe("VersionHistoryCard — hidden when fewer than 2 completed runs", () => {
  it("does not render history card when history has 0 completed runs", async () => {
    mockHistory.mockResolvedValue(makeHistory({ runs: [] }))
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => expect(mockHistory).toHaveBeenCalled())
    expect(screen.queryByText(/Model Version History/i)).not.toBeInTheDocument()
  })

  it("does not render history card when only 1 completed run", async () => {
    mockHistory.mockResolvedValue(
      makeHistory({ runs: [makeRun()], best_metric: 0.85, latest_metric: 0.85 })
    )
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => expect(mockHistory).toHaveBeenCalled())
    expect(screen.queryByText(/Model Version History/i)).not.toBeInTheDocument()
  })
})

describe("VersionHistoryCard — visible with 2+ completed runs", () => {
  const twoRuns: ModelRun[] = [
    makeRun({ id: "run-1", created_at: "2026-01-01T00:00:00" }),
    makeRun({
      id: "run-2",
      created_at: "2026-01-02T00:00:00",
      metrics: { r2: 0.90, mae: 0.10, rmse: 0.15, train_size: 160, test_size: 40 },
    }),
  ]

  beforeEach(() => {
    mockHistory.mockResolvedValue(
      makeHistory({
        runs: twoRuns,
        trend: "improving",
        trend_summary: "R² has improved by 5.9% over 2 training runs.",
        best_metric: 0.90,
        latest_metric: 0.90,
      })
    )
  })

  it("shows the Model Version History heading", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/Model Version History/i)).toBeInTheDocument()
    )
  })

  it("shows the trend badge", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText("Improving")).toBeInTheDocument()
    )
  })

  it("renders the trend summary text", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/R² has improved by 5\.9% over 2 training runs/i)).toBeInTheDocument()
    )
  })

  it("shows Best metric", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => expect(screen.getByText(/Best:/i)).toBeInTheDocument())
    // "0.900" may appear in stats row and table row — use getAllByText
    const elements = screen.getAllByText("0.900")
    expect(elements.length).toBeGreaterThanOrEqual(1)
  })

  it("shows Latest metric", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => expect(screen.getByText(/Latest:/i)).toBeInTheDocument())
  })

  it("shows Runs count", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => expect(screen.getByText(/Runs:/i)).toBeInTheDocument())
  })

  it("renders run table rows for completed runs", async () => {
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/Model Version History/i)).toBeInTheDocument()
    )
    // Two completed runs → table header row + two data rows
    // Row numbers "1" and "2" appear in the # column cells
    const cells = screen.getAllByRole("cell")
    const cellTexts = cells.map((c) => c.textContent?.trim())
    expect(cellTexts).toContain("1")
    expect(cellTexts).toContain("2")
  })
})

describe("VersionHistoryCard — declining trend", () => {
  it("shows Declining badge for declining trend", async () => {
    mockHistory.mockResolvedValue(
      makeHistory({
        runs: [
          makeRun({ id: "run-1", created_at: "2026-01-01T00:00:00" }),
          makeRun({
            id: "run-2",
            created_at: "2026-01-02T00:00:00",
            metrics: { r2: 0.60, mae: 0.20, rmse: 0.25, train_size: 160, test_size: 40 },
          }),
        ],
        trend: "declining",
        trend_summary: "R² has declined by 29.4% over 2 training runs.",
        best_metric: 0.85,
        latest_metric: 0.60,
      })
    )
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => expect(screen.getByText("Declining")).toBeInTheDocument())
  })
})

describe("VersionHistoryCard — stable trend", () => {
  it("shows Stable badge for stable trend", async () => {
    mockHistory.mockResolvedValue(
      makeHistory({
        runs: [
          makeRun({ id: "run-1", created_at: "2026-01-01T00:00:00" }),
          makeRun({
            id: "run-2",
            created_at: "2026-01-02T00:00:00",
            metrics: { r2: 0.851, mae: 0.119, rmse: 0.179, train_size: 160, test_size: 40 },
          }),
        ],
        trend: "stable",
        trend_summary: "R² is stable at 0.851 across 2 training runs.",
        best_metric: 0.851,
        latest_metric: 0.851,
      })
    )
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => expect(screen.getByText("Stable")).toBeInTheDocument())
  })
})

describe("VersionHistoryCard — classification accuracy display", () => {
  it("shows accuracy as percentage for classification models", async () => {
    mockRecs.mockResolvedValue({
      ...defaultRecsResponse,
      problem_type: "classification",
      target_column: "churned",
    })
    mockHistory.mockResolvedValue(
      makeHistory({
        problem_type: "classification",
        primary_metric: "accuracy",
        primary_metric_label: "Accuracy",
        runs: [
          makeRun({
            id: "run-1",
            metrics: { accuracy: 0.92, f1: 0.91, precision: 0.90, recall: 0.92 },
          }),
          makeRun({
            id: "run-2",
            created_at: "2026-01-02T00:00:00",
            metrics: { accuracy: 0.95, f1: 0.94, precision: 0.93, recall: 0.95 },
          }),
        ],
        trend: "improving",
        trend_summary: "Accuracy has improved by 3.3% over 2 training runs.",
        best_metric: 0.95,
        latest_metric: 0.95,
      })
    )
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() =>
      expect(screen.getByText(/Model Version History/i)).toBeInTheDocument()
    )
    // Best displayed as percentage — may appear in stats row and table row
    const pctElements = screen.getAllByText("95.0%")
    expect(pctElements.length).toBeGreaterThanOrEqual(1)
  })
})

describe("VersionHistoryCard — selected and deployed badges", () => {
  it("shows Current badge for selected run", async () => {
    mockHistory.mockResolvedValue(
      makeHistory({
        runs: [
          makeRun({ id: "run-1", created_at: "2026-01-01T00:00:00" }),
          makeRun({
            id: "run-2",
            created_at: "2026-01-02T00:00:00",
            is_selected: true,
            metrics: { r2: 0.90, mae: 0.10, rmse: 0.15, train_size: 160, test_size: 40 },
          }),
        ],
        trend: "improving",
        trend_summary: "Improving trend.",
        best_metric: 0.90,
        latest_metric: 0.90,
      })
    )
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => expect(screen.getByText("Current")).toBeInTheDocument())
  })

  it("shows Live badge for deployed run", async () => {
    mockHistory.mockResolvedValue(
      makeHistory({
        runs: [
          makeRun({ id: "run-1", created_at: "2026-01-01T00:00:00" }),
          makeRun({
            id: "run-2",
            created_at: "2026-01-02T00:00:00",
            is_deployed: true,
            metrics: { r2: 0.90, mae: 0.10, rmse: 0.15, train_size: 160, test_size: 40 },
          }),
        ],
        trend: "stable",
        trend_summary: "Stable.",
        best_metric: 0.90,
        latest_metric: 0.90,
      })
    )
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => expect(screen.getByText("Live")).toBeInTheDocument())
  })
})
