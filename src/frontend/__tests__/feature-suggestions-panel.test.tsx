/**
 * Tests for FeatureSuggestionsPanel — the feature transformation approval UI.
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { FeatureSuggestionsPanel } from "../components/features/feature-suggestions"
import { api } from "../lib/api"
import type { FeatureSuggestion, FeatureSetResult } from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    features: {
      apply: jest.fn(),
      suggestions: jest.fn(),
      setTarget: jest.fn(),
      importance: jest.fn(),
      getSteps: jest.fn(),
      addStep: jest.fn(),
      removeStep: jest.fn(),
    },
    data: {
      listByProject: jest.fn(),
      joinKeys: jest.fn(),
      merge: jest.fn(),
    },
  },
}))

const mockApply = api.features.apply as jest.MockedFunction<typeof api.features.apply>

const makeSuggestion = (overrides: Partial<FeatureSuggestion> = {}): FeatureSuggestion => ({
  id: "sug-1",
  column: "order_date",
  transform_type: "date_decompose",
  title: "Extract Date Parts",
  description: "Split date into year, month, day-of-week",
  preview_columns: ["order_date_year", "order_date_month", "order_date_dayofweek"],
  example_values: [2024, 3, 1],
  ...overrides,
})

const makeResult = (): FeatureSetResult => ({
  feature_set_id: "fs-1",
  column_mapping: { order_date: ["order_date_year", "order_date_month", "order_date_dayofweek"] },
  new_columns: ["order_date_year", "order_date_month", "order_date_dayofweek"],
  total_columns: 12,
  preview: [],
})

beforeEach(() => {
  jest.clearAllMocks()
})

describe("FeatureSuggestionsPanel — empty state", () => {
  it("shows 'no suggestions' message when suggestions array is empty", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[]}
      />
    )
    expect(screen.getByText(/no feature suggestions available/i)).toBeInTheDocument()
  })
})

describe("FeatureSuggestionsPanel — suggestion display", () => {
  it("shows title for each suggestion", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion({ title: "Extract Date Parts" })]}
      />
    )
    expect(screen.getByText("Extract Date Parts")).toBeInTheDocument()
  })

  it("shows the transform type label badge", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion({ transform_type: "date_decompose" })]}
      />
    )
    expect(screen.getByText("Date Parts")).toBeInTheDocument()
  })

  it("shows the description text", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion()]}
      />
    )
    expect(screen.getByText(/split date into year, month, day-of-week/i)).toBeInTheDocument()
  })

  it("shows 0 of N selected initially", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion()]}
      />
    )
    expect(screen.getByText(/0 of 1 selected/i)).toBeInTheDocument()
  })

  it("Apply button is disabled when nothing is selected", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion()]}
      />
    )
    expect(screen.getByRole("button", { name: /apply/i })).toBeDisabled()
  })

  it("shows multiple suggestions", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[
          makeSuggestion({ id: "s1", title: "Split order_date", column: "order_date", transform_type: "date_decompose" }),
          makeSuggestion({ id: "s2", title: "Scale price column", column: "price", transform_type: "log_transform", preview_columns: ["price_log"] }),
          makeSuggestion({ id: "s3", title: "Encode category col", column: "category", transform_type: "one_hot", preview_columns: ["category_A"] }),
        ]}
      />
    )
    expect(screen.getByText("Split order_date")).toBeInTheDocument()
    expect(screen.getByText("Scale price column")).toBeInTheDocument()
    expect(screen.getByText("Encode category col")).toBeInTheDocument()
    // Count display uses JSX interpolation — test via normalized text
    expect(screen.getByText((_, el) => el?.textContent?.trim() === "0 of 3 selected")).toBeInTheDocument()
  })

  it("shows transform type badge for log transform", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion({ transform_type: "log_transform", preview_columns: ["price_log"] })]}
      />
    )
    expect(screen.getByText("Log Transform")).toBeInTheDocument()
  })

  it("shows one-hot encode badge", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion({ transform_type: "one_hot", preview_columns: ["cat_A", "cat_B"] })]}
      />
    )
    expect(screen.getByText("One-Hot Encode")).toBeInTheDocument()
  })
})

describe("FeatureSuggestionsPanel — approve/deselect", () => {
  it("updates selected count when a card is clicked", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion({ title: "Extract Date Parts" })]}
      />
    )
    fireEvent.click(screen.getByText("Extract Date Parts"))
    expect(screen.getByText((_, el) => el?.textContent?.trim() === "1 of 1 selected")).toBeInTheDocument()
  })

  it("enables Apply button after selecting a suggestion", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion({ title: "Extract Date Parts" })]}
      />
    )
    fireEvent.click(screen.getByText("Extract Date Parts"))
    expect(screen.getByRole("button", { name: /apply/i })).not.toBeDisabled()
  })

  it("deselects a card on second click", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion({ title: "Extract Date Parts" })]}
      />
    )
    fireEvent.click(screen.getByText("Extract Date Parts"))
    expect(screen.getByText((_, el) => el?.textContent?.trim() === "1 of 1 selected")).toBeInTheDocument()
    fireEvent.click(screen.getByText("Extract Date Parts"))
    expect(screen.getByText((_, el) => el?.textContent?.trim() === "0 of 1 selected")).toBeInTheDocument()
  })

  it("selects multiple cards independently", () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[
          makeSuggestion({ id: "s1", title: "Split date column", column: "order_date", transform_type: "date_decompose" }),
          makeSuggestion({ id: "s2", title: "Scale to log", column: "price", transform_type: "log_transform", preview_columns: ["price_log"] }),
        ]}
      />
    )
    fireEvent.click(screen.getByText("Split date column"))
    fireEvent.click(screen.getByText("Scale to log"))
    expect(screen.getByText((_, el) => el?.textContent?.trim() === "2 of 2 selected")).toBeInTheDocument()
  })
})

describe("FeatureSuggestionsPanel — apply transforms", () => {
  it("calls api.features.apply with the selected transforms", async () => {
    mockApply.mockResolvedValue(makeResult())
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion()]}
      />
    )
    fireEvent.click(screen.getByText("Extract Date Parts"))
    fireEvent.click(screen.getByRole("button", { name: /apply/i }))
    await waitFor(() =>
      expect(mockApply).toHaveBeenCalledWith("ds-1", [
        { column: "order_date", transform_type: "date_decompose" },
      ])
    )
  })

  it("calls onApplied callback with the result", async () => {
    const result = makeResult()
    mockApply.mockResolvedValue(result)
    const onApplied = jest.fn()
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion()]}
        onApplied={onApplied}
      />
    )
    fireEvent.click(screen.getByText("Extract Date Parts"))
    fireEvent.click(screen.getByRole("button", { name: /apply/i }))
    await waitFor(() => expect(onApplied).toHaveBeenCalledWith(result))
  })

  it("shows success message with new column count", async () => {
    mockApply.mockResolvedValue({
      ...makeResult(),
      new_columns: ["col_a", "col_b", "col_c"],
    })
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion()]}
      />
    )
    fireEvent.click(screen.getByText("Extract Date Parts"))
    fireEvent.click(screen.getByRole("button", { name: /apply/i }))
    await waitFor(() =>
      expect(screen.getByText(/3 new columns created/i)).toBeInTheDocument()
    )
  })

  it("shows singular 'column' for 1 new column", async () => {
    mockApply.mockResolvedValue({
      ...makeResult(),
      new_columns: ["col_a"],
    })
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion()]}
      />
    )
    fireEvent.click(screen.getByText("Extract Date Parts"))
    fireEvent.click(screen.getByRole("button", { name: /apply/i }))
    await waitFor(() =>
      expect(screen.getByText(/1 new column created/i)).toBeInTheDocument()
    )
  })

  it("shows '+N more' for more than 5 new columns", async () => {
    mockApply.mockResolvedValue({
      ...makeResult(),
      new_columns: ["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
    })
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion()]}
      />
    )
    fireEvent.click(screen.getByText("Extract Date Parts"))
    fireEvent.click(screen.getByRole("button", { name: /apply/i }))
    await waitFor(() =>
      expect(screen.getByText(/\+2 more/)).toBeInTheDocument()
    )
  })

  it("does not call api when no suggestions are approved", async () => {
    render(
      <FeatureSuggestionsPanel
        datasetId="ds-1"
        suggestions={[makeSuggestion()]}
      />
    )
    // Don't click any card — Apply button should be disabled
    const btn = screen.getByRole("button", { name: /apply/i })
    expect(btn).toBeDisabled()
    expect(mockApply).not.toHaveBeenCalled()
  })
})
