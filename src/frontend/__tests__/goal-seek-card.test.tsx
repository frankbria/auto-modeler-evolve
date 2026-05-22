/**
 * Unit tests for GoalSeekCard component.
 */
import { render, screen } from "@testing-library/react"
import { GoalSeekCard } from "@/components/deploy/goal-seek-card"
import type { GoalSeekResult } from "@/lib/types"

const baseSuggestion = {
  feature: "units_sold",
  current_mean: 100,
  suggested_value: 150,
  direction: "increase" as const,
  change_pct: 50.0,
}

const baseResult: GoalSeekResult = {
  target_column: "revenue",
  problem_type: "regression",
  algorithm_plain: "Linear Regression",
  target_value: 5000,
  achieved_value: 4950,
  achieved: false,
  gap_pct: 1.0,
  suggestions: [baseSuggestion],
  fixed_features: {},
  categorical_features: {},
  n_optimized: 1,
  feasible: true,
  summary: "Increase units sold by 50% to approach your revenue target.",
}

describe("GoalSeekCard", () => {
  it("renders the card container", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("goal-seek-card")).toBeInTheDocument()
  })

  it("shows target column name in header", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("shows algorithm name in header", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByText("Linear Regression")).toBeInTheDocument()
  })

  it("shows target value", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("target-value")).toHaveTextContent("5,000")
  })

  it("shows achieved value", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("achieved-value")).toHaveTextContent("4,950")
  })

  it("shows best-effort badge when goal not achieved", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("goal-best-effort-badge")).toBeInTheDocument()
    expect(screen.queryByTestId("goal-achieved-badge")).not.toBeInTheDocument()
  })

  it("shows achieved badge when goal is met", () => {
    const result = { ...baseResult, achieved: true, gap_pct: null }
    render(<GoalSeekCard result={result} />)
    expect(screen.getByTestId("goal-achieved-badge")).toBeInTheDocument()
    expect(screen.queryByTestId("goal-best-effort-badge")).not.toBeInTheDocument()
  })

  it("shows gap indicator for unmet regression goals", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("gap-indicator")).toHaveTextContent("1%")
  })

  it("hides gap indicator when goal is achieved", () => {
    const result = { ...baseResult, achieved: true, gap_pct: null }
    render(<GoalSeekCard result={result} />)
    expect(screen.queryByTestId("gap-indicator")).not.toBeInTheDocument()
  })

  it("hides gap indicator for classification", () => {
    const result = { ...baseResult, problem_type: "classification", gap_pct: null }
    render(<GoalSeekCard result={result} />)
    expect(screen.queryByTestId("gap-indicator")).not.toBeInTheDocument()
  })

  it("renders suggestion row with feature name", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("suggestion-row-units_sold")).toBeInTheDocument()
  })

  it("renders suggestion row with underscores replaced by spaces", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByText("units sold")).toBeInTheDocument()
  })

  it("renders increase direction badge", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("direction-badge-increase")).toBeInTheDocument()
  })

  it("renders decrease direction badge", () => {
    const result = {
      ...baseResult,
      suggestions: [
        { ...baseSuggestion, direction: "decrease" as const, change_pct: 20.0 },
      ],
    }
    render(<GoalSeekCard result={result} />)
    expect(screen.getByTestId("direction-badge-decrease")).toBeInTheDocument()
  })

  it("shows suggestions list container", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("suggestions-list")).toBeInTheDocument()
  })

  it("shows no-features note when n_optimized is 0", () => {
    const result = { ...baseResult, n_optimized: 0, suggestions: [] }
    render(<GoalSeekCard result={result} />)
    expect(screen.getByTestId("no-features-note")).toBeInTheDocument()
  })

  it("hides no-features note when there are suggestions", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.queryByTestId("no-features-note")).not.toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("goal-seek-summary")).toHaveTextContent(
      "Increase units sold by 50% to approach your revenue target."
    )
  })

  it("shows feasibility note when not feasible", () => {
    const result = { ...baseResult, feasible: false }
    render(<GoalSeekCard result={result} />)
    expect(
      screen.getByText(/Optimizer did not fully converge/i)
    ).toBeInTheDocument()
  })

  it("hides feasibility note when feasible", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.queryByText(/Optimizer did not fully converge/i)).not.toBeInTheDocument()
  })

  it("shows fixed features when present", () => {
    const result = { ...baseResult, fixed_features: { price: 99.99 } }
    render(<GoalSeekCard result={result} />)
    expect(screen.getByText(/Fixed features/i)).toBeInTheDocument()
    expect(screen.getByText(/price=99\.99/)).toBeInTheDocument()
  })

  it("hides fixed features section when empty", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.queryByText(/Fixed features/i)).not.toBeInTheDocument()
  })

  it("renders multiple suggestion rows", () => {
    const result = {
      ...baseResult,
      suggestions: [
        baseSuggestion,
        { feature: "price", current_mean: 50, suggested_value: 45, direction: "decrease" as const, change_pct: 10.0 },
      ],
    }
    render(<GoalSeekCard result={result} />)
    expect(screen.getByTestId("suggestion-row-units_sold")).toBeInTheDocument()
    expect(screen.getByTestId("suggestion-row-price")).toBeInTheDocument()
  })

  it("uses emerald border class when goal achieved", () => {
    const result = { ...baseResult, achieved: true, gap_pct: null }
    const { container } = render(<GoalSeekCard result={result} />)
    expect(container.firstChild).toHaveClass("border-emerald-500/40")
  })

  it("uses amber border class when goal not achieved", () => {
    const { container } = render(<GoalSeekCard result={baseResult} />)
    expect(container.firstChild).toHaveClass("border-amber-500/40")
  })
})
