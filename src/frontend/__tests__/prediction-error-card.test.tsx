/**
 * Tests for PredictionErrorCard component and related store/API plumbing.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"
import { PredictionErrorCard } from "@/components/models/prediction-error-card"
import type { PredictionErrorResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"
import { api } from "@/lib/api"

fetchMock.enableMocks()

// --- Fixtures ---------------------------------------------------------------

const regressionResult: PredictionErrorResult = {
  algorithm: "linear_regression",
  target_col: "revenue",
  problem_type: "regression",
  errors: [
    {
      actual: 2100.75,
      predicted: 1450.0,
      error: 650.75,
      abs_error: 650.75,
      rank: 1,
      features: { region: "East", units: 18 },
    },
    {
      actual: 450.25,
      predicted: 980.0,
      error: -529.75,
      abs_error: 529.75,
      rank: 2,
      features: { region: "West", units: 4 },
    },
  ],
  total_errors: 15,
  error_rate: 0.0,
  summary: "The model's worst 2 predictions have errors up to 41% of the data range (MAE across all 15 training rows = 312.50).",
}

const classificationResult: PredictionErrorResult = {
  algorithm: "decision_tree_classifier",
  target_col: "churned",
  problem_type: "classification",
  errors: [
    {
      actual: "yes",
      predicted: "no",
      error: "predicted no, actually yes",
      abs_error: null,
      rank: 1,
      features: { age: 30, income: 35000 },
    },
    {
      actual: "no",
      predicted: "yes",
      error: "predicted yes, actually no",
      abs_error: null,
      rank: 2,
      features: { age: 25, income: 45000 },
    },
  ],
  total_errors: 4,
  error_rate: 0.267,
  summary: "The model made 4 incorrect predictions out of 15 training rows (26.7% error rate).",
}

const emptyResult: PredictionErrorResult = {
  algorithm: "linear_regression",
  target_col: "revenue",
  problem_type: "regression",
  errors: [],
  total_errors: 0,
  error_rate: 0.0,
  summary: "No significant errors found.",
}

// --- Component rendering tests -----------------------------------------------

describe("PredictionErrorCard", () => {
  it("renders with data-testid for assertions", () => {
    render(<PredictionErrorCard result={regressionResult} />)
    expect(screen.getByTestId("prediction-error-card")).toBeInTheDocument()
  })

  it("renders 'Prediction Errors' header", () => {
    render(<PredictionErrorCard result={regressionResult} />)
    expect(screen.getByText("Prediction Errors")).toBeInTheDocument()
  })

  it("shows algorithm badge", () => {
    render(<PredictionErrorCard result={regressionResult} />)
    expect(screen.getByText("linear_regression")).toBeInTheDocument()
  })

  it("shows target column name", () => {
    render(<PredictionErrorCard result={regressionResult} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("renders all regression error rows", () => {
    render(<PredictionErrorCard result={regressionResult} />)
    // rank numbers
    expect(screen.getByText("1")).toBeInTheDocument()
    expect(screen.getByText("2")).toBeInTheDocument()
  })

  it("shows summary footer text", () => {
    render(<PredictionErrorCard result={regressionResult} />)
    expect(screen.getByText(/training rows/i)).toBeInTheDocument()
  })

  it("renders feature chips for rows with features", () => {
    render(<PredictionErrorCard result={regressionResult} />)
    expect(screen.getByText(/East/i)).toBeInTheDocument()
  })

  it("shows empty state message when no errors", () => {
    render(<PredictionErrorCard result={emptyResult} />)
    expect(screen.getByText(/no prediction errors found/i)).toBeInTheDocument()
  })

  it("shows classification error rate", () => {
    render(<PredictionErrorCard result={classificationResult} />)
    expect(screen.getAllByText(/27%|26\.7%|wrong/i).length).toBeGreaterThan(0)
  })

  it("shows classification problem type badge", () => {
    render(<PredictionErrorCard result={classificationResult} />)
    expect(screen.getByText("classification")).toBeInTheDocument()
  })

  it("renders classification actual and predicted class labels", () => {
    render(<PredictionErrorCard result={classificationResult} />)
    expect(screen.getAllByText("yes").length).toBeGreaterThan(0)
    expect(screen.getAllByText("no").length).toBeGreaterThan(0)
  })
})

// --- Store action tests -------------------------------------------------------

describe("attachPredictionErrorsToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches pred_errors to last assistant message", () => {
    useAppStore.setState({
      messages: [{ role: "assistant", content: "Here are the errors.", timestamp: "" }],
    })
    useAppStore.getState().attachPredictionErrorsToLastMessage(regressionResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].pred_errors).toBeDefined()
    expect(msgs[0].pred_errors?.algorithm).toBe("linear_regression")
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "Where was my model wrong?", timestamp: "" }],
    })
    useAppStore.getState().attachPredictionErrorsToLastMessage(regressionResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].pred_errors).toBeUndefined()
  })

  it("does not crash when message list is empty", () => {
    expect(() =>
      useAppStore.getState().attachPredictionErrorsToLastMessage(regressionResult)
    ).not.toThrow()
  })
})

// --- API client smoke test ---------------------------------------------------

describe("api.models.getPredictionErrors", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
  })

  it("calls correct URL with default n=10", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(regressionResult))
    await api.models.getPredictionErrors("run-abc")
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/models/run-abc/prediction-errors?n=10")
    )
  })

  it("calls correct URL with custom n", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(regressionResult))
    await api.models.getPredictionErrors("run-abc", 5)
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/models/run-abc/prediction-errors?n=5")
    )
  })

  it("throws on HTTP error", async () => {
    fetchMock.mockResponseOnce("", { status: 404 })
    await expect(api.models.getPredictionErrors("bad-run")).rejects.toThrow()
  })
})
