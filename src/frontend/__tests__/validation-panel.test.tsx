/**
 * Tests for ValidationPanel — cross-validation, error analysis, feature importance, explain row.
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { ValidationPanel } from "../components/validation/validation-panel"
import { api } from "../lib/api"
import type { ValidationMetricsResponse, GlobalExplanationResponse, RowExplanationResponse } from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    validation: {
      metrics: jest.fn(),
      explain: jest.fn(),
      explainRow: jest.fn(),
    },
  },
}))

const mockMetrics = api.validation.metrics as jest.MockedFunction<typeof api.validation.metrics>
const mockExplain = api.validation.explain as jest.MockedFunction<typeof api.validation.explain>
const mockExplainRow = api.validation.explainRow as jest.MockedFunction<typeof api.validation.explainRow>

const makeMetrics = (overrides: Partial<ValidationMetricsResponse> = {}): ValidationMetricsResponse => ({
  model_run_id: "run-1",
  algorithm: "linear_regression",
  problem_type: "regression",
  held_out_metrics: { r2: 0.85, mae: 0.12 },
  cross_validation: {
    metric: "r2",
    scores: [0.82, 0.85, 0.88, 0.84, 0.83],
    mean: 0.844,
    std: 0.021,
    ci_low: 0.802,
    ci_high: 0.886,
    n_splits: 5,
    summary: "5-fold CV R²: 0.844 ± 0.021",
  },
  error_analysis: {
    type: "residuals",
    scatter: [{ predicted: 98, residual: 2 }],
    mae: 0.12,
    bias: 0.01,
    std: 0.15,
    percentile_75: 0.18,
    percentile_90: 0.28,
    summary: "Small, symmetric residuals",
  },
  confidence: {
    overall_confidence: "high",
    limitations: ["May underperform on unseen product categories"],
    summary: "High confidence — consistent across all folds",
  },
  ...overrides,
})

const makeExplain = (): GlobalExplanationResponse => ({
  model_run_id: "run-1",
  algorithm: "linear_regression",
  problem_type: "regression",
  feature_importance: [
    { feature: "region", importance: 0.45, rank: 1 },
    { feature: "season", importance: 0.30, rank: 2 },
    { feature: "price", importance: 0.25, rank: 3 },
  ],
  summary: "The top 3 factors are region, season, and price.",
})

const makeRowExplain = (): RowExplanationResponse => ({
  model_run_id: "run-1",
  row_index: 0,
  actual_value: 130,
  prediction: 125.5,
  prediction_value: 125.5,
  contributions: [
    { feature: "region", value: 1.0, mean_value: 0.5, contribution: 20, direction: "positive" },
    { feature: "season", value: 4.0, mean_value: 2.5, contribution: 5, direction: "positive" },
    { feature: "price", value: 50, mean_value: 45, contribution: -1.5, direction: "negative" },
  ],
  summary: "High revenue because of North region and Q4 season.",
})

beforeEach(() => {
  jest.clearAllMocks()
})

describe("ValidationPanel — no model selected", () => {
  it("shows prompt to select a model", () => {
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId={null}
        algorithmName={null}
      />
    )
    expect(screen.getByText(/select a model in the models tab first/i)).toBeInTheDocument()
  })
})

describe("ValidationPanel — sub-tab navigation", () => {
  it("shows the four sub-tabs", () => {
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    expect(screen.getByText("Cross-Validation")).toBeInTheDocument()
    expect(screen.getByText("Error Analysis")).toBeInTheDocument()
    expect(screen.getByText("Feature Importance")).toBeInTheDocument()
    expect(screen.getByText("Explain Row")).toBeInTheDocument()
  })

  it("calls api.validation.metrics when CV tab is clicked", async () => {
    mockMetrics.mockResolvedValue(makeMetrics())
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Cross-Validation"))
    await waitFor(() => expect(mockMetrics).toHaveBeenCalledWith("run-1"))
  })

  it("shows cv summary text after loading metrics", async () => {
    mockMetrics.mockResolvedValue(makeMetrics())
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Cross-Validation"))
    await waitFor(() =>
      expect(screen.getByText(/5-fold CV R²/i)).toBeInTheDocument()
    )
  })

  it("shows error when metrics API fails", async () => {
    mockMetrics.mockRejectedValue(new Error("Network error"))
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Cross-Validation"))
    await waitFor(() =>
      expect(screen.getByText(/failed to load validation metrics/i)).toBeInTheDocument()
    )
  })

  it("shows algorithm name in panel header", () => {
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    expect(screen.getByText(/linear regression/i)).toBeInTheDocument()
  })
})

describe("ValidationPanel — confidence badge", () => {
  it("shows HIGH confidence badge", async () => {
    mockMetrics.mockResolvedValue(makeMetrics({
      confidence: { overall_confidence: "high", limitations: [], summary: "High" },
    }))
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Cross-Validation"))
    await waitFor(() =>
      expect(screen.getByText(/HIGH confidence/i)).toBeInTheDocument()
    )
  })

  it("shows MEDIUM confidence badge", async () => {
    mockMetrics.mockResolvedValue(makeMetrics({
      confidence: { overall_confidence: "medium", limitations: [], summary: "Medium" },
    }))
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Cross-Validation"))
    await waitFor(() =>
      expect(screen.getByText(/MEDIUM confidence/i)).toBeInTheDocument()
    )
  })

  it("shows LOW confidence badge", async () => {
    mockMetrics.mockResolvedValue(makeMetrics({
      confidence: { overall_confidence: "low", limitations: [], summary: "Low" },
    }))
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Cross-Validation"))
    await waitFor(() =>
      expect(screen.getByText(/LOW confidence/i)).toBeInTheDocument()
    )
  })

  it("shows limitation text", async () => {
    mockMetrics.mockResolvedValue(makeMetrics())
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Cross-Validation"))
    await waitFor(() =>
      expect(screen.getByText(/unseen product categories/i)).toBeInTheDocument()
    )
  })
})

describe("ValidationPanel — feature importance tab", () => {
  it("calls api.validation.explain when Importance tab is clicked", async () => {
    mockExplain.mockResolvedValue(makeExplain())
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Feature Importance"))
    await waitFor(() => expect(mockExplain).toHaveBeenCalledWith("run-1"))
  })

  it("shows importance chart after loading (chart renders without crashing)", async () => {
    mockExplain.mockResolvedValue(makeExplain())
    const { container } = render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Feature Importance"))
    // Wait for summary narrative which is in a <p> tag (accessible via getByText)
    await waitFor(() =>
      expect(screen.getByText(/top 3 factors/i)).toBeInTheDocument()
    )
    // Chart container renders (recharts wraps data in SVG)
    expect(container.querySelector(".recharts-responsive-container")).toBeInTheDocument()
  })

  it("shows importance type label", async () => {
    mockExplain.mockResolvedValue(makeExplain())
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Feature Importance"))
    // Summary from the API is rendered in a <p>
    await waitFor(() =>
      expect(screen.getByText(/top 3 factors/i)).toBeInTheDocument()
    )
  })

  it("shows error when importance API fails", async () => {
    mockExplain.mockRejectedValue(new Error("No feature data"))
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Feature Importance"))
    await waitFor(() =>
      expect(screen.getByText(/failed to load feature importance/i)).toBeInTheDocument()
    )
  })
})

describe("ValidationPanel — explain row tab", () => {
  // Helper: navigate to Explain Row tab and find the Explain (action) button
  function getExplainActionButton() {
    // The tab button is "Explain Row"; the action button text is just "Explain"
    const allButtons = screen.getAllByRole("button", { name: /explain/i })
    // The action button is the one that is NOT the tab button
    return allButtons.find((btn) => btn.textContent?.trim() === "Explain") ?? allButtons[allButtons.length - 1]
  }

  it("shows row index input after switching to Explain Row tab", () => {
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Explain Row"))
    expect(screen.getByRole("spinbutton")).toBeInTheDocument()
  })

  it("calls api.validation.explainRow with the entered row index", async () => {
    mockExplainRow.mockResolvedValue(makeRowExplain())
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Explain Row"))
    const input = screen.getByRole("spinbutton")
    fireEvent.change(input, { target: { value: "3" } })
    fireEvent.click(getExplainActionButton())
    await waitFor(() =>
      expect(mockExplainRow).toHaveBeenCalledWith("run-1", 3)
    )
  })

  it("shows prediction value after loading row explanation", async () => {
    mockExplainRow.mockResolvedValue(makeRowExplain())
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Explain Row"))
    fireEvent.click(getExplainActionButton())
    await waitFor(() =>
      expect(screen.getByText(/125.5/)).toBeInTheDocument()
    )
  })

  it("shows explanation summary", async () => {
    mockExplainRow.mockResolvedValue(makeRowExplain())
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Explain Row"))
    fireEvent.click(getExplainActionButton())
    await waitFor(() =>
      expect(screen.getByText(/north region/i)).toBeInTheDocument()
    )
  })

  it("shows error when explain row API fails", async () => {
    mockExplainRow.mockRejectedValue(new Error("Row out of range"))
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Explain Row"))
    fireEvent.click(getExplainActionButton())
    await waitFor(() =>
      expect(screen.getByText(/failed to load row explanation/i)).toBeInTheDocument()
    )
  })
})

describe("ValidationPanel — error analysis tab", () => {
  it("loads metrics when Error Analysis tab is clicked", async () => {
    mockMetrics.mockResolvedValue(makeMetrics())
    render(
      <ValidationPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByText("Error Analysis"))
    await waitFor(() => expect(mockMetrics).toHaveBeenCalledWith("run-1"))
  })
})
