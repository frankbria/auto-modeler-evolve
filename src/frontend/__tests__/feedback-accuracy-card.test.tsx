import React from "react"
import { render, screen } from "@testing-library/react"
import { FeedbackAccuracyCard } from "@/components/deploy/feedback-accuracy-card"
import { useAppStore } from "@/lib/store"
import type { FeedbackAccuracyReportResult } from "@/lib/types"

const noDataResult: FeedbackAccuracyReportResult = {
  deployment_id: "dep1",
  status: "no_feedback",
  problem_type: "regression",
  has_data: false,
  total_feedback: 0,
  summary:
    "No feedback recorded yet. After making predictions and seeing the real outcomes, record them using the Deployment tab.",
}

const regressionResult: FeedbackAccuracyReportResult = {
  deployment_id: "dep1",
  status: "computed",
  problem_type: "regression",
  has_data: true,
  total_feedback: 5,
  paired_count: 5,
  mae: 12.5,
  pct_error: 8.3,
  avg_actual: 150.0,
  verdict: "good",
  verdict_msg: "Good accuracy — predictions are reasonably close to actual outcomes.",
  weekly_trend: [
    { week_start: "2026-01-05", mae: 15.0, sample_count: 2 },
    { week_start: "2026-01-12", mae: 10.0, sample_count: 3 },
  ],
  trend_direction: "improving",
  summary: "Based on 5 matched predictions: MAE = 12.5000 (8.3% of average actual value). Good accuracy.",
}

const classificationResult: FeedbackAccuracyReportResult = {
  deployment_id: "dep1",
  status: "computed",
  problem_type: "classification",
  has_data: true,
  total_feedback: 10,
  rated_count: 10,
  correct_count: 9,
  incorrect_count: 1,
  unknown_count: 0,
  accuracy: 0.9,
  accuracy_pct: 90.0,
  verdict: "excellent",
  verdict_msg: "Excellent — 90.0% of real-world predictions were correct.",
  weekly_trend: [
    { week_start: "2026-01-05", accuracy: 80.0, sample_count: 5 },
    { week_start: "2026-01-12", accuracy: 100.0, sample_count: 5 },
  ],
  trend_direction: "improving",
  summary: "9 of 10 rated predictions were correct (90.0% real-world accuracy). Excellent.",
}

describe("FeedbackAccuracyCard", () => {
  it("renders with correct aria-label", () => {
    render(<FeedbackAccuracyCard result={noDataResult} />)
    const regions = screen.getAllByRole("region")
    const hasLabel = regions.some((r) =>
      r.getAttribute("aria-label") !== null
    )
    expect(hasLabel).toBe(true)
  })

  it("shows no-data empty state", () => {
    render(<FeedbackAccuracyCard result={noDataResult} />)
    expect(screen.getByText(/No feedback recorded yet/i)).toBeInTheDocument()
  })

  it("shows tip text in no-feedback state", () => {
    render(<FeedbackAccuracyCard result={noDataResult} />)
    expect(screen.getByText(/Record the actual outcomes/i)).toBeInTheDocument()
  })

  it("shows Real-World Accuracy heading", () => {
    render(<FeedbackAccuracyCard result={regressionResult} />)
    const headings = screen.getAllByText(/Real-World Accuracy/i)
    expect(headings.length).toBeGreaterThan(0)
  })

  it("shows verdict badge for regression", () => {
    render(<FeedbackAccuracyCard result={regressionResult} />)
    // "✓ Good" badge text
    expect(screen.getByText(/✓ Good/i)).toBeInTheDocument()
  })

  it("shows MAE stat label for regression", () => {
    render(<FeedbackAccuracyCard result={regressionResult} />)
    // "MAE" appears as a muted label
    const maeLabels = screen.getAllByText(/^MAE$/i)
    expect(maeLabels.length).toBeGreaterThan(0)
  })

  it("shows MAE value for regression", () => {
    render(<FeedbackAccuracyCard result={regressionResult} />)
    expect(screen.getByText("12.5000")).toBeInTheDocument()
  })

  it("shows percent error stat for regression", () => {
    render(<FeedbackAccuracyCard result={regressionResult} />)
    expect(screen.getByText("8.3%")).toBeInTheDocument()
  })

  it("shows matched count for regression", () => {
    render(<FeedbackAccuracyCard result={regressionResult} />)
    expect(screen.getByText(/^Matched$/i)).toBeInTheDocument()
  })

  it("shows accuracy pct for classification", () => {
    render(<FeedbackAccuracyCard result={classificationResult} />)
    expect(screen.getByText("90%")).toBeInTheDocument()
  })

  it("shows Accuracy label for classification", () => {
    render(<FeedbackAccuracyCard result={classificationResult} />)
    expect(screen.getByText(/^Accuracy$/i)).toBeInTheDocument()
  })

  it("shows correct count for classification", () => {
    render(<FeedbackAccuracyCard result={classificationResult} />)
    expect(screen.getByText(/^Correct$/i)).toBeInTheDocument()
    expect(screen.getByText(/^Incorrect$/i)).toBeInTheDocument()
  })

  it("shows excellent verdict badge for classification", () => {
    render(<FeedbackAccuracyCard result={classificationResult} />)
    expect(screen.getByText(/✓ Excellent/i)).toBeInTheDocument()
  })

  it("shows trend direction for regression", () => {
    render(<FeedbackAccuracyCard result={regressionResult} />)
    expect(screen.getByText(/↑ Improving/i)).toBeInTheDocument()
  })

  it("shows trend chart figcaption for regression when multi-week", () => {
    render(<FeedbackAccuracyCard result={regressionResult} />)
    expect(screen.getByText(/mean absolute error/i)).toBeInTheDocument()
  })

  it("shows trend chart figcaption for classification when multi-week", () => {
    render(<FeedbackAccuracyCard result={classificationResult} />)
    // sr-only figcaption uses "accuracy" not in other elements
    expect(screen.getByText(/weekly.*accuracy/i)).toBeInTheDocument()
  })

  it("shows verdict message text", () => {
    render(<FeedbackAccuracyCard result={regressionResult} />)
    expect(
      screen.getByText(/reasonably close to actual outcomes/i)
    ).toBeInTheDocument()
  })

  it("shows summary sentence", () => {
    render(<FeedbackAccuracyCard result={regressionResult} />)
    expect(screen.getByText(/Based on 5 matched/i)).toBeInTheDocument()
  })

  it("renders feedback-only state correctly", () => {
    const feedbackOnly: FeedbackAccuracyReportResult = {
      ...noDataResult,
      status: "feedback_only",
      total_feedback: 3,
      paired_count: 0,
      summary:
        "3 actual outcomes recorded but no paired prediction logs found.",
    }
    render(<FeedbackAccuracyCard result={feedbackOnly} />)
    expect(screen.getByText(/actual outcomes recorded/i)).toBeInTheDocument()
  })
})

describe("FeedbackAccuracyCard store action", () => {
  it("attachFeedbackAccuracyReportToLastMessage attaches to last assistant message", () => {
    const store = useAppStore.getState()

    store.setMessages([
      { role: "user", content: "how accurate", id: "1" },
      { role: "assistant", content: "Checking...", id: "2" },
    ])

    store.attachFeedbackAccuracyReportToLastMessage(regressionResult)

    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.feedback_accuracy_report).toBeDefined()
    expect(last.feedback_accuracy_report.verdict).toBe("good")
  })

  it("does not attach to user messages", () => {
    const store = useAppStore.getState()

    store.setMessages([{ role: "user", content: "hello", id: "u1" }])
    store.attachFeedbackAccuracyReportToLastMessage(regressionResult)

    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.feedback_accuracy_report).toBeUndefined()
  })
})
