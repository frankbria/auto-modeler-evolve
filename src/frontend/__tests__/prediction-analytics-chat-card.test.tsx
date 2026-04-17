/**
 * Tests for PredictionAnalyticsChatCard component and store action.
 *
 * Covers:
 *  1.  Renders figure with correct aria-label
 *  2.  Renders 📊 icon (aria-hidden)
 *  3.  Shows total predictions badge
 *  4.  Shows problem_type badge when present
 *  5.  Renders summary text
 *  6.  Shows 7-day stat box
 *  7.  Shows 30-day stat box
 *  8.  Shows today stat box
 *  9.  Renders sparkline chart when data has non-zero counts
 * 10.  Shows "No predictions in this period" when all counts are zero
 * 11.  Shows peak_day when provided
 * 12.  Does not show peak day section when peak_day is null
 * 13.  Renders class distribution bars for classification
 * 14.  Does not render class distribution for regression (class_counts null)
 * 15.  Shows avg_prediction for regression
 * 16.  Store: attachPredictionAnalyticsChatToLastMessage attaches to last assistant message
 * 17.  Store: does not attach to user message
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { PredictionAnalyticsChatCard } from "@/components/chat/prediction-analytics-chat-card"
import type { PredictionAnalyticsChatResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Recharts mock
// ---------------------------------------------------------------------------
jest.mock("recharts", () => {
  const Original = jest.requireActual("recharts")
  return {
    ...Original,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container" style={{ width: 500, height: 100 }}>
        {children}
      </div>
    ),
  }
})

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeDays(counts: number[]): { date: string; count: number }[] {
  const now = new Date()
  return counts.map((count, i) => {
    const d = new Date(now)
    d.setDate(d.getDate() - (13 - i))
    return { date: d.toISOString().split("T")[0], count }
  })
}

const regressionResult: PredictionAnalyticsChatResult = {
  deployment_id: "dep-1",
  total_predictions: 142,
  predictions_last_7_days: 28,
  predictions_last_30_days: 90,
  predictions_today: 5,
  predictions_by_day: makeDays([0, 2, 5, 8, 3, 1, 4, 6, 7, 2, 3, 5, 8, 5]),
  peak_day: { date: "2026-04-08", count: 8 },
  class_counts: null,
  avg_prediction: 183.7,
  problem_type: "regression",
  summary: "142 total predictions, 28 in the last 7 days, 5 today.",
}

const classificationResult: PredictionAnalyticsChatResult = {
  deployment_id: "dep-2",
  total_predictions: 60,
  predictions_last_7_days: 15,
  predictions_last_30_days: 55,
  predictions_today: 3,
  predictions_by_day: makeDays(Array(14).fill(0)),
  peak_day: null,
  class_counts: { Yes: 35, No: 25 },
  avg_prediction: null,
  problem_type: "classification",
  summary: "60 total predictions, 15 in the last 7 days, 3 today.",
}

const emptyResult: PredictionAnalyticsChatResult = {
  deployment_id: "dep-3",
  total_predictions: 0,
  predictions_last_7_days: 0,
  predictions_last_30_days: 0,
  predictions_today: 0,
  predictions_by_day: makeDays(Array(14).fill(0)),
  peak_day: null,
  class_counts: null,
  avg_prediction: null,
  problem_type: null,
  summary: "0 total predictions, 0 in the last 7 days, 0 today.",
}

// ---------------------------------------------------------------------------
// Component tests
// ---------------------------------------------------------------------------

describe("PredictionAnalyticsChatCard", () => {
  it("renders figure with correct aria-label", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(
      screen.getByRole("figure", { name: /prediction usage analytics/i })
    ).toBeInTheDocument()
  })

  it("renders the 📊 icon", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(screen.getByText("📊")).toBeInTheDocument()
  })

  it("shows total predictions badge", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(screen.getAllByText(/142/).length).toBeGreaterThan(0)
  })

  it("shows problem_type badge when present", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(screen.getByText(/regression/i)).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(
      screen.getByText(/142 total predictions/i)
    ).toBeInTheDocument()
  })

  it("shows 7-day stat box", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(screen.getByText("7-day")).toBeInTheDocument()
    expect(screen.getByText("28")).toBeInTheDocument()
  })

  it("shows 30-day stat box", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(screen.getByText("30-day")).toBeInTheDocument()
    expect(screen.getByText("90")).toBeInTheDocument()
  })

  it("shows today stat box", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(screen.getByText("Today")).toBeInTheDocument()
    expect(screen.getByText("5")).toBeInTheDocument()
  })

  it("renders sparkline chart when data has non-zero counts", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument()
  })

  it("shows no-predictions message when all counts are zero", () => {
    render(<PredictionAnalyticsChatCard result={emptyResult} />)
    expect(
      screen.getByText(/no predictions in this period/i)
    ).toBeInTheDocument()
  })

  it("shows peak_day when provided", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(screen.getByText(/peak day/i)).toBeInTheDocument()
    expect(screen.getByText("2026-04-08")).toBeInTheDocument()
  })

  it("does not show peak day section when peak_day is null", () => {
    render(<PredictionAnalyticsChatCard result={emptyResult} />)
    expect(screen.queryByText(/peak day/i)).toBeNull()
  })

  it("renders class distribution bars for classification", () => {
    render(<PredictionAnalyticsChatCard result={classificationResult} />)
    expect(
      screen.getByText(/predicted class distribution/i)
    ).toBeInTheDocument()
    expect(screen.getByText("Yes")).toBeInTheDocument()
    expect(screen.getByText("No")).toBeInTheDocument()
  })

  it("does not render class distribution when class_counts is null", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(
      screen.queryByText(/predicted class distribution/i)
    ).toBeNull()
  })

  it("shows avg_prediction for regression", () => {
    render(<PredictionAnalyticsChatCard result={regressionResult} />)
    expect(
      screen.getByText(/avg prediction/i)
    ).toBeInTheDocument()
    expect(screen.getByText("183.7")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachPredictionAnalyticsChatToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches to last assistant message", () => {
    const store = useAppStore.getState()
    store.addMessage({ id: "1", role: "user", content: "hello" })
    store.addMessage({ id: "2", role: "assistant", content: "hi" })
    store.attachPredictionAnalyticsChatToLastMessage(regressionResult)

    const msgs = useAppStore.getState().messages
    expect(msgs[1].prediction_analytics_chat).toEqual(regressionResult)
  })

  it("does not attach to user message", () => {
    const store = useAppStore.getState()
    store.addMessage({ id: "1", role: "user", content: "hello" })
    store.attachPredictionAnalyticsChatToLastMessage(regressionResult)

    const msgs = useAppStore.getState().messages
    expect(msgs[0].prediction_analytics_chat).toBeUndefined()
  })
})
