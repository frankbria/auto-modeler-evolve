/**
 * Tests for RankedPredictionsCard component and store action.
 *
 * Covers:
 *  1.  Renders figure with correct aria-label
 *  2.  Renders 🏆 icon (aria-hidden)
 *  3.  Shows target column in heading
 *  4.  Shows n-rows count badge
 *  5.  Shows "Highest" direction badge for highest direction
 *  6.  Shows "Lowest" direction badge for lowest direction
 *  7.  Shows Regression problem type badge
 *  8.  Shows Classification problem type badge
 *  9.  Renders table with correct number of rows (result rows)
 * 10.  Shows rank 1, 2, 3 cells
 * 11.  Shows prediction value for regression
 * 12.  Shows predicted class for classification
 * 13.  Shows feature column values in table
 * 14.  Renders summary footer text
 * 15.  Store: attachRankedPredictionsToLastMessage attaches to last assistant message
 * 16.  Store: does not attach to user message
 * 17.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { RankedPredictionsCard } from "@/components/deploy/ranked-predictions-card"
import type { RankedPredictionsResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const regressionResult: RankedPredictionsResult = {
  problem_type: "regression",
  target_column: "revenue",
  direction: "highest",
  n: 3,
  total_scored: 100,
  rows: [
    {
      rank: 1,
      row_index: 42,
      score: 1500.0,
      feature_values: { units: 50, region: "East" },
      prediction: 1500.0,
    },
    {
      rank: 2,
      row_index: 17,
      score: 1200.0,
      feature_values: { units: 40, region: "West" },
      prediction: 1200.0,
    },
    {
      rank: 3,
      row_index: 5,
      score: 1100.0,
      feature_values: { units: 35, region: "North" },
      prediction: 1100.0,
    },
  ],
  summary: "Scored all 100 rows and ranked by predicted revenue. The highest predicted value is 1500.",
  class_names: null,
}

const classificationResult: RankedPredictionsResult = {
  problem_type: "classification",
  target_column: "churned",
  direction: "highest",
  n: 2,
  total_scored: 50,
  rows: [
    {
      rank: 1,
      row_index: 10,
      score: 0.92,
      feature_values: { age: 45, balance: 500 },
      predicted_class: "yes",
      confidence: 0.92,
      probabilities: { yes: 0.92, no: 0.08 },
    },
    {
      rank: 2,
      row_index: 22,
      score: 0.85,
      feature_values: { age: 38, balance: 1200 },
      predicted_class: "yes",
      confidence: 0.85,
      probabilities: { yes: 0.85, no: 0.15 },
    },
  ],
  summary: "Scored all 50 rows by model confidence. Top result: 'yes' at 92.0% confidence.",
  class_names: ["no", "yes"],
}

const lowestResult: RankedPredictionsResult = {
  ...regressionResult,
  direction: "lowest",
  rows: regressionResult.rows.map((r) => ({ ...r, prediction: r.prediction! * 0.1 })),
}

// ---------------------------------------------------------------------------
// 1. Renders figure with correct aria-label
// ---------------------------------------------------------------------------

test("renders figure with correct aria-label (regression)", () => {
  render(<RankedPredictionsCard result={regressionResult} />)
  expect(screen.getByRole("figure", { name: /ranked predictions.*revenue/i })).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 2. Renders 🏆 icon (aria-hidden)
// ---------------------------------------------------------------------------

test("renders trophy icon aria-hidden", () => {
  const { container } = render(<RankedPredictionsCard result={regressionResult} />)
  const icon = container.querySelector("[aria-hidden]")
  expect(icon).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 3. Shows target column in heading
// ---------------------------------------------------------------------------

test("shows target column name in heading", () => {
  render(<RankedPredictionsCard result={regressionResult} />)
  expect(screen.getByRole("heading", { name: /revenue/i })).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 4. Shows n-rows count badge
// ---------------------------------------------------------------------------

test("shows count badge with n of total_scored", () => {
  render(<RankedPredictionsCard result={regressionResult} />)
  expect(screen.getByText(/3 of 100/)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 5. Shows "Highest" direction badge
// ---------------------------------------------------------------------------

test("shows Highest badge for highest direction", () => {
  render(<RankedPredictionsCard result={regressionResult} />)
  expect(screen.getByText("Highest")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 6. Shows "Lowest" direction badge
// ---------------------------------------------------------------------------

test("shows Lowest badge for lowest direction", () => {
  render(<RankedPredictionsCard result={lowestResult} />)
  expect(screen.getByText("Lowest")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 7. Shows Regression problem type badge
// ---------------------------------------------------------------------------

test("shows Regression badge for regression", () => {
  render(<RankedPredictionsCard result={regressionResult} />)
  expect(screen.getByText("Regression")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 8. Shows Classification problem type badge
// ---------------------------------------------------------------------------

test("shows Classification badge for classification", () => {
  render(<RankedPredictionsCard result={classificationResult} />)
  expect(screen.getByText("Classification")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 9. Renders table with correct number of data rows
// ---------------------------------------------------------------------------

test("renders correct number of result rows in table", () => {
  render(<RankedPredictionsCard result={regressionResult} />)
  // 3 data rows plus header row
  const rows = screen.getAllByRole("row")
  expect(rows.length).toBe(4) // header + 3 data rows
})

// ---------------------------------------------------------------------------
// 10. Shows rank 1, 2, 3
// ---------------------------------------------------------------------------

test("shows rank numbers 1, 2, 3", () => {
  render(<RankedPredictionsCard result={regressionResult} />)
  expect(screen.getByText("1")).toBeInTheDocument()
  expect(screen.getByText("2")).toBeInTheDocument()
  expect(screen.getByText("3")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 11. Shows prediction value for regression
// ---------------------------------------------------------------------------

test("shows prediction value for regression row", () => {
  render(<RankedPredictionsCard result={regressionResult} />)
  expect(screen.getByText("1.5k")).toBeInTheDocument() // 1500 formatted
})

// ---------------------------------------------------------------------------
// 12. Shows predicted class for classification
// ---------------------------------------------------------------------------

test("shows predicted class and confidence for classification", () => {
  render(<RankedPredictionsCard result={classificationResult} />)
  // "yes (92%)" — should appear for the first row
  expect(screen.getByText(/yes.*92%/i)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 13. Shows feature column values in table
// ---------------------------------------------------------------------------

test("shows feature column headers and values", () => {
  render(<RankedPredictionsCard result={regressionResult} />)
  // Column header "units" (with spaces replacing underscores)
  expect(screen.getByText("units")).toBeInTheDocument()
  // Value 50 appears in first row
  expect(screen.getByText("50")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 14. Renders summary footer text
// ---------------------------------------------------------------------------

test("renders summary footer text", () => {
  render(<RankedPredictionsCard result={regressionResult} />)
  expect(screen.getByText(/Scored all 100 rows/i)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 15. Store: attachRankedPredictionsToLastMessage attaches to last assistant message
// ---------------------------------------------------------------------------

test("store attaches ranked_predictions to last assistant message", () => {
  const store = useAppStore.getState()
  store.setMessages([
    { id: "u1", role: "user", content: "rank predictions", timestamp: new Date() },
    { id: "a1", role: "assistant", content: "Here are the top ranked rows.", timestamp: new Date() },
  ])
  store.attachRankedPredictionsToLastMessage(regressionResult)
  const messages = useAppStore.getState().messages
  expect(messages[1].ranked_predictions).toEqual(regressionResult)
})

// ---------------------------------------------------------------------------
// 16. Store: does not attach to user message
// ---------------------------------------------------------------------------

test("store does not attach ranked_predictions to user message", () => {
  const store = useAppStore.getState()
  store.setMessages([
    { id: "u1", role: "user", content: "rank predictions", timestamp: new Date() },
  ])
  store.attachRankedPredictionsToLastMessage(regressionResult)
  const messages = useAppStore.getState().messages
  expect((messages[0] as any).ranked_predictions).toBeUndefined()
})

// ---------------------------------------------------------------------------
// 17. Store: does not crash on empty messages list
// ---------------------------------------------------------------------------

test("store attach does not crash when messages is empty", () => {
  const store = useAppStore.getState()
  store.setMessages([])
  expect(() => store.attachRankedPredictionsToLastMessage(regressionResult)).not.toThrow()
})
