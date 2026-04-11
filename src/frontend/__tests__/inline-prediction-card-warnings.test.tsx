/**
 * Tests for InlinePredictionCard guard-rail warning display.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { InlinePredictionCard } from "../components/models/inline-prediction-card"
import type { InlinePredictionResult, GuardRailWarning } from "../lib/types"

const baseResult: InlinePredictionResult = {
  deployment_id: "dep-1",
  target_column: "revenue",
  prediction: 1500,
  provided_features: { units: 15 },
  defaults_used_count: 1,
  total_features: 2,
  summary: "Predicted revenue: 1500",
  problem_type: "regression",
}

const outOfRangeWarning: GuardRailWarning = {
  feature: "units",
  provided_value: 9999,
  severity: "out_of_range",
  message: "units: 9999 is outside the typical range (10–20); confidence may be lower",
  expected_min: 10,
  expected_max: 20,
  training_min: 8,
  training_max: 22,
}

const extremeWarning: GuardRailWarning = {
  feature: "price",
  provided_value: -500,
  severity: "extreme_outlier",
  message: "price: -500 is outside the training range (0–200); predictions may be unreliable",
  expected_min: 0,
  expected_max: 200,
  training_min: -5,
  training_max: 250,
}

const categoryWarning: GuardRailWarning = {
  feature: "region",
  provided_value: "Atlantis",
  severity: "unknown_category",
  message: "region: 'Atlantis' was not seen during training; model will use a fallback encoding",
  known_categories: ["North", "South", "East", "West"],
}

describe("InlinePredictionCard — no warnings", () => {
  it("renders aria-label for the card", () => {
    render(<InlinePredictionCard result={baseResult} />)
    expect(
      screen.getByRole("figure", { name: /inline prediction result for revenue/i })
    ).toBeInTheDocument()
  })

  it("does not show warning badge when no warnings present", () => {
    render(<InlinePredictionCard result={baseResult} />)
    expect(screen.queryByText(/warning/i)).not.toBeInTheDocument()
  })

  it("shows prediction value when no warnings", () => {
    render(<InlinePredictionCard result={baseResult} />)
    // 1500 → "1.5k" via formatValue
    expect(screen.getByText("1.5k")).toBeInTheDocument()
  })
})

describe("InlinePredictionCard — with warnings", () => {
  it("shows warning count badge when warnings present", () => {
    const result = { ...baseResult, guard_rail_warnings: [outOfRangeWarning] }
    render(<InlinePredictionCard result={result} />)
    // The badge renders "1 warning" as visible text
    expect(screen.getByText(/1 warning/)).toBeInTheDocument()
  })

  it("shows plural warnings badge for multiple warnings", () => {
    const result = {
      ...baseResult,
      guard_rail_warnings: [outOfRangeWarning, categoryWarning],
    }
    render(<InlinePredictionCard result={result} />)
    expect(screen.getByText(/2 warnings/)).toBeInTheDocument()
  })

  it("renders out_of_range warning message", () => {
    const result = { ...baseResult, guard_rail_warnings: [outOfRangeWarning] }
    render(<InlinePredictionCard result={result} />)
    expect(
      screen.getByText(/units: 9999 is outside the typical range/i)
    ).toBeInTheDocument()
  })

  it("renders extreme_outlier warning message", () => {
    const result = { ...baseResult, guard_rail_warnings: [extremeWarning] }
    render(<InlinePredictionCard result={result} />)
    expect(
      screen.getByText(/price: -500 is outside the training range/i)
    ).toBeInTheDocument()
  })

  it("renders unknown_category warning message", () => {
    const result = { ...baseResult, guard_rail_warnings: [categoryWarning] }
    render(<InlinePredictionCard result={result} />)
    expect(
      screen.getByText(/region: 'Atlantis' was not seen during training/i)
    ).toBeInTheDocument()
  })

  it("shows typical range for out_of_range warning", () => {
    const result = { ...baseResult, guard_rail_warnings: [outOfRangeWarning] }
    render(<InlinePredictionCard result={result} />)
    expect(screen.getByText(/Typical training range/i)).toBeInTheDocument()
  })

  it("shows known categories for unknown_category warning", () => {
    const result = { ...baseResult, guard_rail_warnings: [categoryWarning] }
    render(<InlinePredictionCard result={result} />)
    expect(screen.getByText(/Known values:/i)).toBeInTheDocument()
    expect(screen.getByText(/North/)).toBeInTheDocument()
  })

  it("warning severity label: Out of range", () => {
    const result = { ...baseResult, guard_rail_warnings: [outOfRangeWarning] }
    render(<InlinePredictionCard result={result} />)
    expect(screen.getByText(/Out of range:/i)).toBeInTheDocument()
  })

  it("warning severity label: Extreme outlier", () => {
    const result = { ...baseResult, guard_rail_warnings: [extremeWarning] }
    render(<InlinePredictionCard result={result} />)
    expect(screen.getByText(/Extreme outlier:/i)).toBeInTheDocument()
  })

  it("warning severity label: Unknown category", () => {
    const result = { ...baseResult, guard_rail_warnings: [categoryWarning] }
    render(<InlinePredictionCard result={result} />)
    expect(screen.getByText(/Unknown category:/i)).toBeInTheDocument()
  })

  it("still shows prediction value when warnings are present", () => {
    const result = { ...baseResult, guard_rail_warnings: [outOfRangeWarning] }
    render(<InlinePredictionCard result={result} />)
    expect(screen.getByText("1.5k")).toBeInTheDocument()
  })

  it("input validation warnings section has aria-label", () => {
    const result = { ...baseResult, guard_rail_warnings: [outOfRangeWarning] }
    render(<InlinePredictionCard result={result} />)
    expect(
      screen.getByLabelText(/input validation warnings/i)
    ).toBeInTheDocument()
  })

  it("warning role=alert on each warning row", () => {
    const result = {
      ...baseResult,
      guard_rail_warnings: [outOfRangeWarning, categoryWarning],
    }
    render(<InlinePredictionCard result={result} />)
    const alerts = screen.getAllByRole("alert")
    expect(alerts.length).toBe(2)
  })

  it("empty guard_rail_warnings array shows no warnings", () => {
    const result = { ...baseResult, guard_rail_warnings: [] }
    render(<InlinePredictionCard result={result} />)
    expect(screen.queryByText(/warning/i)).not.toBeInTheDocument()
  })
})
