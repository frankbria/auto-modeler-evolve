/**
 * Tests for InteractionCard component and store action.
 *
 * Covers:
 *  1.  Renders figure with correct aria-label
 *  2.  Renders 🔬 icon (aria-hidden)
 *  3.  Shows feature1 and feature2 in heading
 *  4.  Shows target column in heading
 *  5.  Shows Regression badge for regression models
 *  6.  Shows Classification badge for classification models
 *  7.  Shows min and max prediction boxes for regression
 *  8.  Does not show min/max for classification (null values)
 *  9.  Renders the interaction grid table
 * 10.  Grid has correct number of rows (row_labels)
 * 11.  Grid has correct number of columns (col_labels + header)
 * 12.  Renders summary footer text
 * 13.  Renders color legend for regression
 * 14.  Does not render min/max boxes when min_val is null
 * 15.  Store: attachInteractionToLastMessage attaches to last assistant message
 * 16.  Store: does not attach to user message
 * 17.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { InteractionCard } from "@/components/deploy/interaction-card"
import type { InteractionResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const regressionResult: InteractionResult = {
  feature1: "units",
  feature2: "price",
  target_column: "revenue",
  problem_type: "regression",
  row_labels: ["5", "10", "15"],
  col_labels: ["10", "20", "30"],
  values: [
    [500, 600, 700],
    [1000, 1100, 1200],
    [1500, 1600, 1700],
  ],
  min_val: 500,
  max_val: 1700,
  summary:
    "Across all combinations of units and price, revenue ranges from 500 to 1700 (a 240% spread).",
}

const classificationResult: InteractionResult = {
  feature1: "units",
  feature2: "region",
  target_column: "churn",
  problem_type: "classification",
  row_labels: ["1", "5", "10"],
  col_labels: ["North", "South"],
  values: [
    ["no", "no"],
    ["no", "yes"],
    ["yes", "yes"],
  ],
  min_val: null,
  max_val: null,
  summary: "Across combinations of units and region, the model predicts 2 different classes: no, yes.",
}

// ---------------------------------------------------------------------------
// 1. aria-label
// ---------------------------------------------------------------------------

test("renders figure with correct aria-label (regression)", () => {
  render(<InteractionCard result={regressionResult} />)
  expect(
    screen.getByRole("figure", { name: /Interaction: units × price vs revenue/i })
  ).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 2. Icon aria-hidden
// ---------------------------------------------------------------------------

test("renders 🔬 icon with aria-hidden", () => {
  render(<InteractionCard result={regressionResult} />)
  const icon = screen.getByText("🔬")
  expect(icon).toHaveAttribute("aria-hidden", "true")
})

// ---------------------------------------------------------------------------
// 3. Feature names in heading
// ---------------------------------------------------------------------------

test("shows feature1 in heading", () => {
  render(<InteractionCard result={regressionResult} />)
  // "units" appears multiple times (heading + grid); just check at least once
  expect(screen.getAllByText("units").length).toBeGreaterThan(0)
})

test("shows feature2 in heading", () => {
  render(<InteractionCard result={regressionResult} />)
  // "price" appears in heading + grid col header
  expect(screen.getAllByText("price").length).toBeGreaterThan(0)
})

// ---------------------------------------------------------------------------
// 4. Target column
// ---------------------------------------------------------------------------

test("shows target column in heading", () => {
  render(<InteractionCard result={regressionResult} />)
  // "revenue" appears in heading and min/max labels
  expect(screen.getAllByText("revenue").length).toBeGreaterThan(0)
})

// ---------------------------------------------------------------------------
// 5-6. Problem type badge
// ---------------------------------------------------------------------------

test("shows Regression badge for regression problem type", () => {
  render(<InteractionCard result={regressionResult} />)
  expect(screen.getByText("Regression")).toBeInTheDocument()
})

test("shows Classification badge for classification problem type", () => {
  render(<InteractionCard result={classificationResult} />)
  expect(screen.getByText("Classification")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 7. Min / Max label boxes for regression
// ---------------------------------------------------------------------------

test("shows Min label for regression", () => {
  render(<InteractionCard result={regressionResult} />)
  expect(screen.getByText("Min revenue")).toBeInTheDocument()
})

test("shows Max label for regression", () => {
  render(<InteractionCard result={regressionResult} />)
  expect(screen.getByText("Max revenue")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 8. No min/max for classification
// ---------------------------------------------------------------------------

test("does not render min/max boxes when min_val is null", () => {
  render(<InteractionCard result={classificationResult} />)
  // Should not have Min/Max labels
  expect(screen.queryByText(/Min churn/i)).not.toBeInTheDocument()
  expect(screen.queryByText(/Max churn/i)).not.toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 9. Grid table rendered
// ---------------------------------------------------------------------------

test("renders the interaction grid table", () => {
  render(<InteractionCard result={regressionResult} />)
  expect(screen.getByTestId("interaction-grid")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 10. Grid row count
// ---------------------------------------------------------------------------

test("grid has correct number of data rows", () => {
  const { container } = render(<InteractionCard result={regressionResult} />)
  const tbody = container.querySelector("tbody")
  expect(tbody).toBeInTheDocument()
  const rows = tbody!.querySelectorAll("tr")
  expect(rows).toHaveLength(regressionResult.row_labels.length)
})

// ---------------------------------------------------------------------------
// 11. Grid column count (col_labels + header label column)
// ---------------------------------------------------------------------------

test("grid has correct number of columns in header", () => {
  const { container } = render(<InteractionCard result={regressionResult} />)
  const thead = container.querySelector("thead")
  expect(thead).toBeInTheDocument()
  const ths = thead!.querySelectorAll("th")
  // 1 corner cell + col_labels.length columns
  expect(ths).toHaveLength(1 + regressionResult.col_labels.length)
})

// ---------------------------------------------------------------------------
// 12. Summary footer
// ---------------------------------------------------------------------------

test("renders summary footer text", () => {
  render(<InteractionCard result={regressionResult} />)
  expect(
    screen.getByText(/Across all combinations of units and price/i)
  ).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 13. Color legend for regression
// ---------------------------------------------------------------------------

test("renders Low/High color legend for regression", () => {
  render(<InteractionCard result={regressionResult} />)
  expect(screen.getByText("Low")).toBeInTheDocument()
  expect(screen.getByText("High")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 14. No legend for classification
// ---------------------------------------------------------------------------

test("does not render color legend for classification", () => {
  render(<InteractionCard result={classificationResult} />)
  expect(screen.queryByText("Low")).not.toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// 15-17. Store: attachInteractionToLastMessage
// ---------------------------------------------------------------------------

const makeMsg = (role: "user" | "assistant", content: string) => ({
  role,
  content,
  timestamp: new Date().toISOString(),
})

test("store: attachInteractionToLastMessage attaches to last assistant message", () => {
  useAppStore.setState({
    messages: [makeMsg("user", "hi"), makeMsg("assistant", "hello")],
  })
  useAppStore.getState().attachInteractionToLastMessage(regressionResult)
  const msgs = useAppStore.getState().messages
  expect(msgs[msgs.length - 1].interaction).toEqual(regressionResult)
})

test("store: does not attach interaction to user message", () => {
  useAppStore.setState({
    messages: [makeMsg("user", "hi")],
  })
  useAppStore.getState().attachInteractionToLastMessage(regressionResult)
  const msgs = useAppStore.getState().messages
  expect((msgs[msgs.length - 1] as { interaction?: unknown }).interaction).toBeUndefined()
})

test("store: does not crash when messages list is empty", () => {
  useAppStore.setState({ messages: [] })
  expect(() =>
    useAppStore.getState().attachInteractionToLastMessage(regressionResult)
  ).not.toThrow()
})
