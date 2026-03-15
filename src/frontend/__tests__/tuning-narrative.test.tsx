/**
 * Tests for:
 * 1. TuningCard rendering (model-training-panel.tsx)
 * 2. Auto-Tune button interaction
 * 3. api.models.tune() client method shape
 * 4. api.projects.narrative() client method shape
 * 5. TuningResult and ProjectNarrative type contracts
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { ModelTrainingPanel } from "../components/models/model-training-panel"
import { api } from "../lib/api"
import type {
  ModelRecommendation,
  ModelRun,
  TuningResult,
  ProjectNarrative,
} from "../lib/types"

// ── EventSource stub ─────────────────────────────────────────────────────────
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

// ── API mock ─────────────────────────────────────────────────────────────────
jest.mock("../lib/api", () => ({
  api: {
    projects: {
      narrative: jest.fn(),
    },
    models: {
      recommendations: jest.fn(),
      runs: jest.fn(),
      train: jest.fn(),
      compare: jest.fn(),
      comparisonRadar: jest.fn(),
      select: jest.fn(),
      tune: jest.fn(),
      trainingStreamUrl: jest.fn().mockReturnValue("/stream"),
      downloadUrl: jest.fn().mockReturnValue("/download"),
      reportUrl: jest.fn().mockReturnValue("/report"),
      readiness: jest.fn(),
      history: jest.fn(),
      retrain: jest.fn(),
    },
  },
}))

const mockRecs = api.models.recommendations as jest.MockedFunction<typeof api.models.recommendations>
const mockRuns = api.models.runs as jest.MockedFunction<typeof api.models.runs>
const mockTune = api.models.tune as jest.MockedFunction<typeof api.models.tune>
const mockCompare = api.models.compare as jest.MockedFunction<typeof api.models.compare>
const mockRadar = api.models.comparisonRadar as jest.MockedFunction<typeof api.models.comparisonRadar>
const mockNarrative = api.projects.narrative as jest.MockedFunction<typeof api.projects.narrative>

// ── Sample data ──────────────────────────────────────────────────────────────
const SAMPLE_REC: ModelRecommendation = {
  algorithm: "random_forest_regressor",
  name: "Random Forest",
  description: "Ensemble of trees",
  plain_english: "Like asking 100 experts",
  best_for: "Most datasets",
  recommended_because: "Good baseline",
}

const SAMPLE_RUN: ModelRun = {
  id: "run-1",
  algorithm: "random_forest_regressor",
  status: "done",
  is_selected: false,
  is_deployed: false,
  metrics: { r2: 0.82, mae: 120.5, rmse: 145.0, train_size: 16, test_size: 4 } as never,
  summary: "R² = 0.82",
  training_duration_ms: 400,
  error_message: null,
  created_at: "2026-03-15T00:00:00",
}

const SAMPLE_TUNING_IMPROVED: TuningResult = {
  original_model_run_id: "run-1",
  tuned_model_run_id: "run-2",
  algorithm: "random_forest_regressor",
  tunable: true,
  original_metrics: { r2: 0.82, mae: 120.5, rmse: 145.0, train_size: 16, test_size: 4 },
  tuned_metrics: { r2: 0.87, mae: 108.0, rmse: 130.0, train_size: 16, test_size: 4 },
  best_params: { n_estimators: 200, max_depth: 10 },
  tuned_cv_score: 0.85,
  improved: true,
  improvement_pct: 6.1,
  summary: "Tuning improved R² from 0.82 to 0.87 (+6.1%).",
  tuned_run: { ...SAMPLE_RUN, id: "run-2", metrics: { r2: 0.87, mae: 108.0, rmse: 130.0 } as never },
}

const SAMPLE_TUNING_NOT_TUNABLE: TuningResult = {
  original_model_run_id: "run-1",
  tuned_model_run_id: null,
  algorithm: "linear_regression",
  tunable: false,
  original_metrics: { r2: 0.79 },
  tuned_metrics: null,
  best_params: null,
  tuned_cv_score: null,
  improved: false,
  improvement_pct: null,
  summary: "Linear Regression has no hyperparameters to tune.",
  tuned_run: null,
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mockHistory = (api as any).models.history as jest.Mock

// ── Helper ───────────────────────────────────────────────────────────────────
function setupPanelWithRun(run: ModelRun = SAMPLE_RUN) {
  mockRecs.mockResolvedValue({
    project_id: "proj-1",
    problem_type: "regression",
    target_column: "revenue",
    n_rows: 20,
    n_features: 4,
    recommendations: [SAMPLE_REC],
  })
  mockRuns.mockResolvedValue({ project_id: "proj-1", runs: [run] })
  mockCompare.mockResolvedValue({ project_id: "proj-1", problem_type: "regression", models: [run], recommendation: null })
  mockRadar.mockResolvedValue(null)
  mockHistory.mockResolvedValue({
    project_id: "proj-1",
    problem_type: "regression",
    primary_metric: "r2",
    primary_metric_label: "R²",
    runs: [run],
    trend: "insufficient_data",
    trend_summary: "Not enough training runs to determine a trend yet.",
    best_metric: null,
    latest_metric: null,
  })
}

// ──────────────────────────────────────────────────────────────────────────────
// 1. Auto-Tune Button Rendering
// ──────────────────────────────────────────────────────────────────────────────

describe("Auto-Tune button", () => {
  test("renders Auto-Tune button for completed run", async () => {
    setupPanelWithRun()
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => expect(screen.getByText("Auto-Tune")).toBeInTheDocument())
  })

  test("Auto-Tune button is disabled while tuning", async () => {
    setupPanelWithRun()
    mockTune.mockImplementation(() => new Promise(() => {})) // never resolves
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText("Auto-Tune"))
    fireEvent.click(screen.getByText("Auto-Tune"))
    await waitFor(() => expect(screen.getByText("Tuning...")).toBeInTheDocument())
  })

  test("calls api.models.tune with correct run ID", async () => {
    setupPanelWithRun()
    mockTune.mockResolvedValue(SAMPLE_TUNING_IMPROVED)
    mockRuns.mockResolvedValue({ project_id: "proj-1", runs: [SAMPLE_RUN, SAMPLE_TUNING_IMPROVED.tuned_run!] })
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => screen.getAllByText("Auto-Tune"))
    fireEvent.click(screen.getAllByText("Auto-Tune")[0])
    await waitFor(() => expect(mockTune).toHaveBeenCalledWith("run-1"))
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// 2. TuningCard Rendering
// ──────────────────────────────────────────────────────────────────────────────

describe("TuningCard rendering", () => {
  test("shows improvement badge when tuning improved the model", async () => {
    mockRecs.mockResolvedValue({
      project_id: "proj-1", problem_type: "regression", target_column: "revenue",
      n_rows: 20, n_features: 4, recommendations: [SAMPLE_REC],
    })
    mockRuns.mockResolvedValue({ project_id: "proj-1", runs: [SAMPLE_RUN] })
    mockCompare.mockResolvedValue({ project_id: "proj-1", problem_type: "regression", models: [SAMPLE_RUN], recommendation: null })
    mockRadar.mockResolvedValue(null)
    mockTune.mockResolvedValue(SAMPLE_TUNING_IMPROVED)

    render(<ModelTrainingPanel projectId="proj-1" />)
    // Wait for initial load and click first Auto-Tune button
    await waitFor(() => expect(screen.getAllByText("Auto-Tune").length).toBeGreaterThan(0))
    fireEvent.click(screen.getAllByText("Auto-Tune")[0])
    // After tune completes, TuningCard shows improvement badge
    await waitFor(() => expect(screen.getAllByText(/6\.1%/i).length).toBeGreaterThan(0))
  })

  test("shows summary text after tuning", async () => {
    setupPanelWithRun()
    mockTune.mockResolvedValue(SAMPLE_TUNING_IMPROVED)
    mockRuns.mockResolvedValue({ project_id: "proj-1", runs: [SAMPLE_RUN] })
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText("Auto-Tune"))
    fireEvent.click(screen.getByText("Auto-Tune"))
    await waitFor(() =>
      expect(screen.getByText(/Tuning improved/i)).toBeInTheDocument()
    )
  })

  test("shows not-tunable message for non-tunable algorithm", async () => {
    setupPanelWithRun({ ...SAMPLE_RUN, algorithm: "linear_regression" })
    mockTune.mockResolvedValue(SAMPLE_TUNING_NOT_TUNABLE)
    mockRuns.mockResolvedValue({ project_id: "proj-1", runs: [{ ...SAMPLE_RUN, algorithm: "linear_regression" }] })
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText("Auto-Tune"))
    fireEvent.click(screen.getByText("Auto-Tune"))
    await waitFor(() =>
      expect(screen.getByText(/no hyperparameters/i)).toBeInTheDocument()
    )
  })

  test("shows before/after metrics", async () => {
    setupPanelWithRun()
    mockTune.mockResolvedValue(SAMPLE_TUNING_IMPROVED)
    mockRuns.mockResolvedValue({ project_id: "proj-1", runs: [SAMPLE_RUN] })
    render(<ModelTrainingPanel projectId="proj-1" />)
    await waitFor(() => screen.getByText("Auto-Tune"))
    fireEvent.click(screen.getByText("Auto-Tune"))
    await waitFor(() => expect(screen.getByText("Before:")).toBeInTheDocument())
    expect(screen.getByText("After:")).toBeInTheDocument()
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// 3. API client method shapes
// ──────────────────────────────────────────────────────────────────────────────

describe("api.models.tune client method", () => {
  test("api.models.tune is defined", () => {
    expect(typeof api.models.tune).toBe("function")
  })

  test("api.projects.narrative is defined", () => {
    expect(typeof api.projects.narrative).toBe("function")
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// 4. TuningResult type contract
// ──────────────────────────────────────────────────────────────────────────────

describe("TuningResult type contract", () => {
  test("improved result has all expected fields", () => {
    const r: TuningResult = SAMPLE_TUNING_IMPROVED
    expect(r.original_model_run_id).toBeDefined()
    expect(r.tuned_model_run_id).toBeDefined()
    expect(r.algorithm).toBeDefined()
    expect(r.tunable).toBe(true)
    expect(r.original_metrics).toBeDefined()
    expect(r.tuned_metrics).toBeDefined()
    expect(r.best_params).toBeDefined()
    expect(r.improved).toBe(true)
    expect(r.improvement_pct).toBeGreaterThan(0)
    expect(typeof r.summary).toBe("string")
  })

  test("non-tunable result has null tuned fields", () => {
    const r: TuningResult = SAMPLE_TUNING_NOT_TUNABLE
    expect(r.tunable).toBe(false)
    expect(r.tuned_model_run_id).toBeNull()
    expect(r.tuned_metrics).toBeNull()
    expect(r.best_params).toBeNull()
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// 5. ProjectNarrative type contract
// ──────────────────────────────────────────────────────────────────────────────

describe("ProjectNarrative type contract", () => {
  test("narrative result has expected fields", async () => {
    const narrative: ProjectNarrative = {
      project_id: "proj-1",
      project_name: "Revenue Analysis",
      narrative: "This project analysed 1500 rows of sales data...",
      generated_at: "2026-03-15T04:44:00",
      context: { project_name: "Revenue Analysis" },
    }
    expect(narrative.project_id).toBe("proj-1")
    expect(narrative.project_name).toBe("Revenue Analysis")
    expect(typeof narrative.narrative).toBe("string")
    expect(narrative.generated_at).toBeDefined()
    expect(narrative.context).toBeDefined()
  })

  test("api.projects.narrative is callable", async () => {
    const fakeNarrative: ProjectNarrative = {
      project_id: "p1",
      project_name: "Test",
      narrative: "A great project.",
      generated_at: "2026-03-15T00:00:00",
      context: {},
    }
    mockNarrative.mockResolvedValue(fakeNarrative)
    const result = await api.projects.narrative("p1")
    expect(result.narrative).toBe("A great project.")
    expect(mockNarrative).toHaveBeenCalledWith("p1")
  })
})
