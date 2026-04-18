/**
 * Tests for ProductionInputDistributionCard component and store action.
 *
 * Covers:
 *  1.  Renders card with correct aria-label
 *  2.  Renders 📊 icon
 *  3.  Shows sample_count badge
 *  4.  Shows features count badge
 *  5.  Shows "All inputs in range" badge when no OOR values
 *  6.  Shows "out-of-range values" badge when OOR values exist
 *  7.  Renders numeric feature row with min/avg/max
 *  8.  Shows out-of-range badge on numeric feature when oor > 0
 *  9.  Renders categorical feature row with top categories
 * 10.  Shows unseen badge on categorical feature when unseen > 0
 * 11.  Shows empty state for 0 predictions
 * 12.  Renders summary text
 * 13.  Renders figcaption legend
 * 14.  Store: attachProdInputDistToLastMessage attaches to last assistant message
 * 15.  Store: does not modify user messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ProductionInputDistributionCard } from "@/components/chat/production-input-distribution-card"
import type { ProductionInputDistributionResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const numericOnlyResult: ProductionInputDistributionResult = {
  deployment_id: "dep-001",
  sample_count: 25,
  features: [
    {
      feature: "revenue",
      feature_type: "numeric",
      count: 25,
      mean: 175.5,
      min: 100.0,
      max: 300.0,
      train_min: 90.0,
      train_max: 310.0,
      train_p5: 95.0,
      train_p95: 305.0,
      out_of_range_count: 0,
      out_of_range_pct: 0.0,
    },
  ],
  summary: "25 predictions analyzed across 1 feature. All inputs are within training ranges.",
}

const withOORResult: ProductionInputDistributionResult = {
  deployment_id: "dep-002",
  sample_count: 50,
  features: [
    {
      feature: "units",
      feature_type: "numeric",
      count: 50,
      mean: 22.4,
      min: 1.0,
      max: 500.0,
      train_min: 1.0,
      train_max: 100.0,
      train_p5: 2.0,
      train_p95: 90.0,
      out_of_range_count: 5,
      out_of_range_pct: 10.0,
    },
  ],
  summary: "50 predictions analyzed across 1 feature. 5 input values outside the training distribution.",
}

const categoricalResult: ProductionInputDistributionResult = {
  deployment_id: "dep-003",
  sample_count: 30,
  features: [
    {
      feature: "region",
      feature_type: "categorical",
      count: 30,
      top_categories: [
        { value: "East", count: 15, pct: 50.0 },
        { value: "West", count: 10, pct: 33.3 },
        { value: "Unknown", count: 5, pct: 16.7 },
      ],
      n_unique: 3,
      known_categories: ["East", "West", "North"],
      unseen_count: 5,
      unseen_pct: 16.7,
    },
  ],
  summary: "30 predictions analyzed across 1 feature. 5 input values outside the training distribution.",
}

const emptyResult: ProductionInputDistributionResult = {
  deployment_id: "dep-004",
  sample_count: 0,
  features: [],
  summary: "No predictions have been made yet — input distributions will appear after users start using the model.",
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ProductionInputDistributionCard", () => {
  test("1. renders card with correct aria-label", () => {
    render(<ProductionInputDistributionCard result={numericOnlyResult} />)
    expect(screen.getByLabelText(/production input distribution/i)).toBeTruthy()
  })

  test("2. renders 📊 icon", () => {
    render(<ProductionInputDistributionCard result={numericOnlyResult} />)
    expect(screen.getByText("📊")).toBeTruthy()
  })

  test("3. shows sample_count badge", () => {
    render(<ProductionInputDistributionCard result={numericOnlyResult} />)
    expect(screen.getAllByText(/25 predictions analyzed/i).length).toBeGreaterThan(0)
  })

  test("4. shows features count badge", () => {
    render(<ProductionInputDistributionCard result={numericOnlyResult} />)
    expect(screen.getAllByText(/1 feature/i).length).toBeGreaterThan(0)
  })

  test("5. shows 'All inputs in range' badge when no OOR values", () => {
    render(<ProductionInputDistributionCard result={numericOnlyResult} />)
    expect(screen.getByText(/all inputs in range/i)).toBeTruthy()
  })

  test("6. shows 'out-of-range values' badge when OOR values exist", () => {
    render(<ProductionInputDistributionCard result={withOORResult} />)
    expect(screen.getByText(/out-of-range values/i)).toBeTruthy()
  })

  test("7. renders numeric feature row with min/avg/max labels", () => {
    render(<ProductionInputDistributionCard result={numericOnlyResult} />)
    expect(screen.getByLabelText(/Feature revenue: numeric/i)).toBeTruthy()
    expect(screen.getByText("min")).toBeTruthy()
    expect(screen.getByText("avg")).toBeTruthy()
    expect(screen.getByText("max")).toBeTruthy()
  })

  test("8. shows out-of-range badge on numeric feature when oor > 0", () => {
    render(<ProductionInputDistributionCard result={withOORResult} />)
    expect(screen.getByText(/10% out of range/i)).toBeTruthy()
  })

  test("9. renders categorical feature row with top categories", () => {
    render(<ProductionInputDistributionCard result={categoricalResult} />)
    expect(screen.getByLabelText(/Feature region: categorical/i)).toBeTruthy()
    expect(screen.getByLabelText(/East: 50%/i)).toBeTruthy()
    expect(screen.getByLabelText(/West: 33.3%/i)).toBeTruthy()
  })

  test("10. shows unseen badge on categorical feature when unseen > 0", () => {
    render(<ProductionInputDistributionCard result={categoricalResult} />)
    expect(screen.getByText(/16.7% unseen/i)).toBeTruthy()
    expect(screen.getByText(/not seen during training/i)).toBeTruthy()
  })

  test("11. shows empty state for 0 predictions", () => {
    render(<ProductionInputDistributionCard result={emptyResult} />)
    expect(screen.getByText(/no predictions have been made yet/i)).toBeTruthy()
  })

  test("12. renders summary text", () => {
    render(<ProductionInputDistributionCard result={numericOnlyResult} />)
    expect(screen.getByText(/All inputs are within training ranges/i)).toBeTruthy()
  })

  test("13. renders figcaption legend", () => {
    render(<ProductionInputDistributionCard result={numericOnlyResult} />)
    expect(screen.getByText(/amber = numeric values outside training range/i)).toBeTruthy()
  })
})

describe("Store: attachProdInputDistToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  test("14. attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "what are users sending?" },
        { role: "assistant", content: "Let me check." },
      ],
    })
    useAppStore.getState().attachProdInputDistToLastMessage(numericOnlyResult)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.prod_input_dist).toEqual(numericOnlyResult)
  })

  test("15. does not modify user messages", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "what are users sending?" }],
    })
    useAppStore.getState().attachProdInputDistToLastMessage(numericOnlyResult)
    const messages = useAppStore.getState().messages
    expect((messages[0] as { prod_input_dist?: unknown }).prod_input_dist).toBeUndefined()
  })
})
