/**
 * Tests for PredictionOutputAnomalyCard component and store action.
 */

import React from "react"
import { render, screen, act } from "@testing-library/react"

import { PredictionOutputAnomalyCard } from "@/components/deploy/prediction-output-anomaly-card"
import type { PredictionOutputAnomalyResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const regressionFewAnomalies: PredictionOutputAnomalyResult = {
  deployment_id: "dep-1",
  n_total: 50,
  n_anomalous: 2,
  anomaly_rate: 4.0,
  verdict: "few_anomalies",
  verdict_label: "A few unusual predictions detected",
  problem_type: "regression",
  stats: { mean: 1000.0, std: 120.0, min: 750.0, max: 4500.0 },
  anomalies: [
    {
      id: "abc12345",
      prediction_value: "4500.0",
      z_score: 2.9,
      confidence: null,
      deviation: "2.9σ above mean",
      reason: "Prediction is 2.9 standard deviations above the typical value of 1000.00",
      created_at: "2026-05-31T09:00:00",
      input_summary: { region: "East", units: 500 },
    },
    {
      id: "def67890",
      prediction_value: "750.0",
      z_score: 2.6,
      confidence: null,
      deviation: "2.6σ below mean",
      reason: "Prediction is 2.6 standard deviations below the typical value of 1000.00",
      created_at: "2026-05-30T08:00:00",
      input_summary: { region: "West", units: 1 },
    },
  ],
  summary: "2 of 50 recent predictions (4.0%) are unusually extreme.",
}

const regressionNoAnomalies: PredictionOutputAnomalyResult = {
  deployment_id: "dep-2",
  n_total: 30,
  n_anomalous: 0,
  anomaly_rate: 0.0,
  verdict: "no_anomalies",
  verdict_label: "No unusual predictions detected",
  problem_type: "regression",
  stats: { mean: 500.0, std: 50.0, min: 400.0, max: 600.0 },
  anomalies: [],
  summary: "All 30 recent predictions fall within the normal range.",
}

const classificationManyAnomalies: PredictionOutputAnomalyResult = {
  deployment_id: "dep-3",
  n_total: 20,
  n_anomalous: 5,
  anomaly_rate: 25.0,
  verdict: "many_anomalies",
  verdict_label: "Several unusual predictions detected",
  problem_type: "classification",
  stats: { mean_confidence: 0.82, min_confidence: 0.3, max_confidence: 0.98 },
  anomalies: [
    {
      id: "low001",
      prediction_value: "churn",
      z_score: null,
      confidence: 31.0,
      deviation: "31% confidence",
      reason: "Model was only 31% confident — below the 55% threshold",
      created_at: "2026-05-31T10:00:00",
      input_summary: { tenure: 2, charges: 80 },
    },
  ],
  summary: "5 of 20 recent predictions (25.0%) have low model confidence.",
}

const noDataResult: PredictionOutputAnomalyResult = {
  deployment_id: "dep-4",
  n_total: 3,
  n_anomalous: 0,
  anomaly_rate: 0.0,
  verdict: "no_data",
  verdict_label: "Not enough predictions yet",
  problem_type: "regression",
  stats: {},
  anomalies: [],
  summary: "Only 3 predictions recorded — need at least 5 to detect anomalies.",
}

// ---------------------------------------------------------------------------
// Component rendering tests
// ---------------------------------------------------------------------------

describe("PredictionOutputAnomalyCard", () => {
  it("renders the card with data-testid", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    expect(screen.getByTestId("prediction-output-anomaly-card")).toBeInTheDocument()
  })

  it("renders verdict badge with correct label", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("A few unusual predictions detected")
  })

  it("renders no_anomalies verdict in emerald style", () => {
    render(<PredictionOutputAnomalyCard result={regressionNoAnomalies} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("No unusual predictions detected")
  })

  it("renders many_anomalies verdict", () => {
    render(<PredictionOutputAnomalyCard result={classificationManyAnomalies} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Several unusual predictions detected")
  })

  it("renders no_data verdict", () => {
    render(<PredictionOutputAnomalyCard result={noDataResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Not enough predictions yet")
  })

  it("renders problem type badge", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    expect(screen.getByTestId("problem-type-badge")).toHaveTextContent("Regression")
  })

  it("renders classification problem type", () => {
    render(<PredictionOutputAnomalyCard result={classificationManyAnomalies} />)
    expect(screen.getByTestId("problem-type-badge")).toHaveTextContent("Classification")
  })

  it("renders n_total badge", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    expect(screen.getByTestId("n-total-badge")).toHaveTextContent("50 predictions analyzed")
  })

  it("renders anomaly rate stat for regression", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    expect(screen.getByTestId("anomaly-rate")).toHaveTextContent("2 (4%)")
  })

  it("renders anomaly rows when anomalies exist", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    expect(screen.getByTestId("anomaly-row-0")).toBeInTheDocument()
    expect(screen.getByTestId("anomaly-row-1")).toBeInTheDocument()
  })

  it("shows prediction value in anomaly row", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    const values = screen.getAllByTestId("prediction-value")
    expect(values[0]).toHaveTextContent("4500.0")
  })

  it("shows deviation badge in anomaly row", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    const badges = screen.getAllByTestId("deviation-badge")
    expect(badges[0]).toHaveTextContent("2.9σ above mean")
  })

  it("shows input summary chips in anomaly row", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    expect(screen.getByText("region=East")).toBeInTheDocument()
  })

  it("renders no anomaly rows when verdict is no_anomalies", () => {
    render(<PredictionOutputAnomalyCard result={regressionNoAnomalies} />)
    expect(screen.queryByTestId("anomaly-row-0")).not.toBeInTheDocument()
  })

  it("shows all-in-range message when no anomalies", () => {
    render(<PredictionOutputAnomalyCard result={regressionNoAnomalies} />)
    expect(screen.getByText(/All recent predictions are within the normal range/)).toBeInTheDocument()
  })

  it("renders classification anomaly with confidence display", () => {
    render(<PredictionOutputAnomalyCard result={classificationManyAnomalies} />)
    expect(screen.getByTestId("anomaly-row-0")).toBeInTheDocument()
    expect(screen.getByTestId("deviation-badge")).toHaveTextContent("31% confidence")
  })

  it("renders summary paragraph", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    const summaries = screen.getAllByText(/2 of 50 recent predictions/)
    expect(summaries.length).toBeGreaterThanOrEqual(1)
  })

  it("renders sr-only figcaption for accessibility", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    const figcaption = document.querySelector("figcaption")
    expect(figcaption).toBeInTheDocument()
    expect(figcaption?.className).toContain("sr-only")
  })

  it("renders figure with aria-label", () => {
    render(<PredictionOutputAnomalyCard result={regressionFewAnomalies} />)
    const fig = screen.getByRole("figure", { name: /Prediction Output Anomaly Analysis/ })
    expect(fig).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Zustand store action tests
// ---------------------------------------------------------------------------

describe("attachOutputAnomaliesToLastMessage", () => {
  beforeEach(() => {
    act(() => {
      useAppStore.setState({
        messages: [{ id: "1", role: "assistant", content: "Hello", timestamp: "t" }],
      })
    })
  })

  it("attaches output_anomalies to the last assistant message", () => {
    act(() => {
      useAppStore.getState().attachOutputAnomaliesToLastMessage(regressionFewAnomalies)
    })
    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1] as import("@/lib/types").ChatMessage
    expect(lastMsg.output_anomalies).toEqual(regressionFewAnomalies)
  })

  it("does not attach to non-assistant messages", () => {
    act(() => {
      useAppStore.setState({
        messages: [{ id: "1", role: "user", content: "Question", timestamp: "t" }],
      })
      useAppStore.getState().attachOutputAnomaliesToLastMessage(regressionFewAnomalies)
    })
    const msgs = useAppStore.getState().messages
    const lastMsg = msgs[msgs.length - 1] as import("@/lib/types").ChatMessage
    expect(lastMsg.output_anomalies).toBeUndefined()
  })

  it("does not mutate existing message properties", () => {
    act(() => {
      useAppStore.getState().attachOutputAnomaliesToLastMessage(regressionNoAnomalies)
    })
    const last = useAppStore.getState().messages[0] as import("@/lib/types").ChatMessage
    expect(last.content).toBe("Hello")
    expect(last.role).toBe("assistant")
  })
})
