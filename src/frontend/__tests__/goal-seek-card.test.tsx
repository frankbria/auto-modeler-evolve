/**
 * Unit tests for GoalSeekCard component.
 *
 * Covers:
 * - Core result display (target, achieved, gap, summary)
 * - Suggestion rows with direction badges
 * - Fixed features display ("Previously locked:")
 * - Lock toggle per suggestion row
 * - Re-run button shown when features are locked
 * - Re-run button calls onActionClick with locked feature key=value pairs in message
 * - Re-run button hidden when no features locked or no onActionClick provided
 */
import { render, screen, fireEvent } from "@testing-library/react"
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

  it("shows previously locked features when present", () => {
    const result = { ...baseResult, fixed_features: { price: 99.99 } }
    render(<GoalSeekCard result={result} />)
    expect(screen.getByText(/Previously locked/i)).toBeInTheDocument()
    expect(screen.getByText(/price=99\.99/)).toBeInTheDocument()
  })

  it("hides previously locked section when fixed_features is empty", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.queryByText(/Previously locked/i)).not.toBeInTheDocument()
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

  // ─── Lock toggle tests ───────────────────────────────────────────────────

  it("renders a lock toggle button for each suggestion", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("lock-toggle-units_sold")).toBeInTheDocument()
  })

  it("lock toggle starts unlocked (🔓 icon)", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("lock-toggle-units_sold")).toHaveTextContent("🔓")
  })

  it("clicking lock toggle changes icon to locked (🔒)", () => {
    render(<GoalSeekCard result={baseResult} />)
    fireEvent.click(screen.getByTestId("lock-toggle-units_sold"))
    expect(screen.getByTestId("lock-toggle-units_sold")).toHaveTextContent("🔒")
  })

  it("clicking lock toggle twice returns to unlocked state", () => {
    render(<GoalSeekCard result={baseResult} />)
    const toggle = screen.getByTestId("lock-toggle-units_sold")
    fireEvent.click(toggle)
    fireEvent.click(toggle)
    expect(toggle).toHaveTextContent("🔓")
  })

  it("lock toggle has aria-pressed=false when unlocked", () => {
    render(<GoalSeekCard result={baseResult} />)
    expect(screen.getByTestId("lock-toggle-units_sold")).toHaveAttribute("aria-pressed", "false")
  })

  it("lock toggle has aria-pressed=true when locked", () => {
    render(<GoalSeekCard result={baseResult} />)
    fireEvent.click(screen.getByTestId("lock-toggle-units_sold"))
    expect(screen.getByTestId("lock-toggle-units_sold")).toHaveAttribute("aria-pressed", "true")
  })

  it("re-run button is not shown when no features are locked", () => {
    render(<GoalSeekCard result={baseResult} onActionClick={jest.fn()} />)
    expect(screen.queryByTestId("rerun-with-locked-button")).not.toBeInTheDocument()
  })

  it("re-run button appears after locking a feature", () => {
    render(<GoalSeekCard result={baseResult} onActionClick={jest.fn()} />)
    fireEvent.click(screen.getByTestId("lock-toggle-units_sold"))
    expect(screen.getByTestId("rerun-with-locked-button")).toBeInTheDocument()
  })

  it("re-run button is not shown without onActionClick prop", () => {
    render(<GoalSeekCard result={baseResult} />)
    fireEvent.click(screen.getByTestId("lock-toggle-units_sold"))
    expect(screen.queryByTestId("rerun-with-locked-button")).not.toBeInTheDocument()
  })

  it("re-run button calls onActionClick with locked feature in message", () => {
    const onActionClick = jest.fn()
    render(<GoalSeekCard result={baseResult} onActionClick={onActionClick} />)
    fireEvent.click(screen.getByTestId("lock-toggle-units_sold"))
    fireEvent.click(screen.getByTestId("rerun-with-locked-button"))
    expect(onActionClick).toHaveBeenCalledTimes(1)
    const msg: string = onActionClick.mock.calls[0][0]
    expect(msg).toContain("goal seek")
    expect(msg).toContain("revenue")
    expect(msg).toContain("units_sold=150")
  })

  it("re-run message includes the original target value", () => {
    const onActionClick = jest.fn()
    render(<GoalSeekCard result={baseResult} onActionClick={onActionClick} />)
    fireEvent.click(screen.getByTestId("lock-toggle-units_sold"))
    fireEvent.click(screen.getByTestId("rerun-with-locked-button"))
    const msg: string = onActionClick.mock.calls[0][0]
    expect(msg).toContain("5,000")
  })

  it("re-run button label shows count of locked features", () => {
    const result = {
      ...baseResult,
      suggestions: [
        baseSuggestion,
        { feature: "price", current_mean: 50, suggested_value: 45, direction: "decrease" as const, change_pct: 10.0 },
      ],
    }
    render(<GoalSeekCard result={result} onActionClick={jest.fn()} />)
    fireEvent.click(screen.getByTestId("lock-toggle-units_sold"))
    fireEvent.click(screen.getByTestId("lock-toggle-price"))
    const btn = screen.getByTestId("rerun-with-locked-button")
    expect(btn).toHaveTextContent("2 features locked")
  })

  it("re-run message includes all locked features", () => {
    const result = {
      ...baseResult,
      suggestions: [
        baseSuggestion,
        { feature: "price", current_mean: 50, suggested_value: 45, direction: "decrease" as const, change_pct: 10.0 },
      ],
    }
    const onActionClick = jest.fn()
    render(<GoalSeekCard result={result} onActionClick={onActionClick} />)
    fireEvent.click(screen.getByTestId("lock-toggle-units_sold"))
    fireEvent.click(screen.getByTestId("lock-toggle-price"))
    fireEvent.click(screen.getByTestId("rerun-with-locked-button"))
    const msg: string = onActionClick.mock.calls[0][0]
    expect(msg).toContain("units_sold=150")
    expect(msg).toContain("price=45")
  })
})
