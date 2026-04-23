/**
 * Tests for BatchJobResultCard component and Zustand store action.
 *
 * Covers:
 *  1.  Renders empty state card when has_results is false
 *  2.  Renders teal card with aria-label when has_results is true
 *  3.  Shows "Regression" badge for regression problem_type
 *  4.  Shows "Classification" badge for classification problem_type
 *  5.  Shows record count badge
 *  6.  Shows completed_at formatted timestamp
 *  7.  Shows target column code element
 *  8.  Regression: shows avg/median/min/max stats
 *  9.  Regression: renders histogram bars
 * 10.  Classification: renders class distribution bars
 * 11.  Classification: shows avg_confidence when present
 * 12.  Classification: omits avg_confidence when null
 * 13.  Summary text appears in both states
 * 14.  sr-only figcaption is present
 * 15.  Store: attachBatchJobResultsToLastMessage attaches to last assistant message
 * 16.  Store: does not attach when last message is from user
 * 17.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { BatchJobResultCard } from "@/components/chat/batch-job-result-card"
import type { BatchJobResultsResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const emptyResult: BatchJobResultsResult = {
  has_results: false,
  summary: "No batch jobs have run yet for this deployment.",
}

const regressionResult: BatchJobResultsResult = {
  has_results: true,
  problem_type: "regression",
  target_column: "price",
  total_rows: 1500,
  completed_at: "2026-04-22T09:00:00.000Z",
  avg_prediction: 245000.5,
  median_prediction: 230000.0,
  min_prediction: 85000.0,
  max_prediction: 650000.0,
  std_prediction: 95000.0,
  histogram: [
    { bin_start: 85000, bin_end: 170000, count: 200 },
    { bin_start: 170000, bin_end: 255000, count: 600 },
    { bin_start: 255000, bin_end: 340000, count: 450 },
    { bin_start: 340000, bin_end: 425000, count: 200 },
    { bin_start: 425000, bin_end: 650000, count: 50 },
  ],
  summary: "Batch produced 1,500 predictions for price. Average: 245000.50.",
}

const classificationResult: BatchJobResultsResult = {
  has_results: true,
  problem_type: "classification",
  target_column: "churn",
  total_rows: 800,
  completed_at: "2026-04-22T14:30:00.000Z",
  top_class: "retained",
  top_pct: 72.5,
  class_distribution: [
    { class_name: "retained", count: 580, pct: 72.5 },
    { class_name: "churned", count: 220, pct: 27.5 },
  ],
  avg_confidence: 88.3,
  summary: "Batch produced 800 predictions for churn. Most common: 'retained' (72.5%).",
}

const classificationNoConfidence: BatchJobResultsResult = {
  ...classificationResult,
  avg_confidence: undefined,
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

describe("BatchJobResultCard — empty state", () => {
  it("renders when has_results is false", () => {
    render(<BatchJobResultCard result={emptyResult} />)
    expect(screen.getByText("Batch Job Results")).toBeInTheDocument()
  })

  it("shows summary text in empty state", () => {
    render(<BatchJobResultCard result={emptyResult} />)
    expect(screen.getByText(/no batch jobs have run yet/i)).toBeInTheDocument()
  })

  it("shows onboarding guidance in empty state", () => {
    render(<BatchJobResultCard result={emptyResult} />)
    expect(screen.getByText(/schedule.*batch prediction/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Active regression result
// ---------------------------------------------------------------------------

describe("BatchJobResultCard — regression", () => {
  beforeEach(() => {
    render(<BatchJobResultCard result={regressionResult} />)
  })

  it("renders card with aria-label", () => {
    expect(
      screen.getByRole("region", { name: /batch job results/i }),
    ).toBeInTheDocument()
  })

  it("shows Regression badge", () => {
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("shows record count badge", () => {
    expect(screen.getByText(/1,500 records/i)).toBeInTheDocument()
  })

  it("shows target column", () => {
    expect(screen.getByText("price")).toBeInTheDocument()
  })

  it("shows Average stat", () => {
    expect(screen.getByText("Average")).toBeInTheDocument()
  })

  it("shows Median stat", () => {
    expect(screen.getByText("Median")).toBeInTheDocument()
  })

  it("shows Min stat", () => {
    expect(screen.getByText("Min")).toBeInTheDocument()
  })

  it("shows Max stat", () => {
    expect(screen.getByText("Max")).toBeInTheDocument()
  })

  it("renders histogram bars", () => {
    expect(screen.getByLabelText(/prediction distribution histogram/i)).toBeInTheDocument()
  })

  it("shows Prediction Distribution label", () => {
    expect(screen.getByText("Prediction Distribution")).toBeInTheDocument()
  })

  it("shows summary text", () => {
    const els = screen.getAllByText(/batch produced 1,500 predictions/i)
    expect(els.length).toBeGreaterThanOrEqual(1)
  })

  it("has sr-only figcaption", () => {
    const { container } = render(<BatchJobResultCard result={regressionResult} />)
    const caption = container.querySelector("figcaption.sr-only")
    expect(caption).toBeInTheDocument()
    expect(caption?.textContent).toMatch(/1500/i)
  })
})

// ---------------------------------------------------------------------------
// Active classification result
// ---------------------------------------------------------------------------

describe("BatchJobResultCard — classification", () => {
  beforeEach(() => {
    render(<BatchJobResultCard result={classificationResult} />)
  })

  it("shows Classification badge", () => {
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("shows record count badge", () => {
    const els = screen.getAllByText(/800 records/i)
    expect(els.length).toBeGreaterThanOrEqual(1)
  })

  it("shows target column", () => {
    expect(screen.getByText("churn")).toBeInTheDocument()
  })

  it("renders class distribution bars for retained", () => {
    expect(screen.getByText("retained")).toBeInTheDocument()
    const pcts = screen.getAllByText("72.5%")
    expect(pcts.length).toBeGreaterThanOrEqual(1)
  })

  it("renders class distribution bars for churned", () => {
    expect(screen.getByText("churned")).toBeInTheDocument()
    expect(screen.getByText("27.5%")).toBeInTheDocument()
  })

  it("shows avg confidence when present", () => {
    expect(screen.getByText(/avg confidence/i)).toBeInTheDocument()
    expect(screen.getByText("88.3%")).toBeInTheDocument()
  })

  it("shows summary text", () => {
    const els = screen.getAllByText(/most common.*retained/i)
    expect(els.length).toBeGreaterThanOrEqual(1)
  })
})

describe("BatchJobResultCard — classification without confidence", () => {
  it("does not render confidence row when avg_confidence is null", () => {
    render(<BatchJobResultCard result={classificationNoConfidence} />)
    expect(screen.queryByText(/avg confidence/i)).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store: attachBatchJobResultsToLastMessage
// ---------------------------------------------------------------------------

describe("attachBatchJobResultsToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches batch_job_results to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "show batch results" },
        { role: "assistant", content: "Here are the batch results." },
      ],
    })
    useAppStore.getState().attachBatchJobResultsToLastMessage(regressionResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].batch_job_results).toEqual(regressionResult)
  })

  it("does not attach when last message is from user", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "show batch results" }],
    })
    useAppStore.getState().attachBatchJobResultsToLastMessage(regressionResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].batch_job_results).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachBatchJobResultsToLastMessage(regressionResult),
    ).not.toThrow()
  })
})
