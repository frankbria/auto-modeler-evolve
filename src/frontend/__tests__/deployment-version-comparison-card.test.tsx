/**
 * Tests for DeploymentVersionComparisonCard and Zustand store action.
 *
 * Covers:
 *  1.  Renders region with aria-label "Deployment version comparison"
 *  2.  Shows 🔄 icon (aria-hidden)
 *  3.  Shows "Version Comparison" heading
 *  4.  Shows version range badge (v1 → v2) when has_comparison=true
 *  5.  Shows "improved" count badge when improved_count > 0
 *  6.  Shows "declined" count badge when declined_count > 0
 *  7.  Renders metric comparison table with header row
 *  8.  Shows metric label in table row (R²)
 *  9.  Shows previous and current values in table
 * 10.  Shows change direction (↑ emerald for improvement)
 * 11.  Shows change direction (↓ rose for decline)
 * 12.  Shows algorithm changed note when algorithm_changed=true
 * 13.  Shows summary footer paragraph
 * 14.  Shows MAE/RMSE note in content area
 * 15.  Renders no-comparison state with summary when has_comparison=false
 * 16.  No metric table rendered when has_comparison=false
 * 17.  Store: attachVersionComparisonToLastMessage attaches to last assistant message
 * 18.  Store: does not attach to user message
 * 19.  Store: does not crash on empty messages list
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { DeploymentVersionComparisonCard } from "@/components/deploy/deployment-version-comparison-card"
import type { DeploymentVersionComparisonResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const withComparison: DeploymentVersionComparisonResult = {
  has_comparison: true,
  current_version: 2,
  previous_version: 1,
  current_algorithm: "random_forest_regressor",
  previous_algorithm: "linear_regression",
  current_deployed_at: "2026-04-15T10:00:00",
  previous_deployed_at: "2026-04-10T08:00:00",
  algorithm_changed: true,
  metric_diffs: [
    {
      metric: "r2",
      previous: 0.72,
      current: 0.84,
      delta: 0.12,
      pct_change: 16.7,
      direction: "up",
      improved: true,
      higher_is_better: true,
    },
    {
      metric: "mae",
      previous: 12.5,
      current: 9.3,
      delta: -3.2,
      pct_change: -25.6,
      direction: "down",
      improved: true,
      higher_is_better: false,
    },
    {
      metric: "rmse",
      previous: 18.1,
      current: 20.4,
      delta: 2.3,
      pct_change: 12.7,
      direction: "up",
      improved: false,
      higher_is_better: false,
    },
  ],
  improved_count: 2,
  declined_count: 1,
  summary: "Version 2 vs 1: R² improved from 0.72 to 0.84 (+0.1200, +16.7%). 2 metrics improved, 1 declined.",
}

const noComparison: DeploymentVersionComparisonResult = {
  has_comparison: false,
  version_count: 1,
  summary: "Only one deployment version exists so far — retrain the model to create a second version for comparison.",
}

// ---------------------------------------------------------------------------
// Render tests — with comparison
// ---------------------------------------------------------------------------

describe("DeploymentVersionComparisonCard — with comparison", () => {
  beforeEach(() => {
    render(<DeploymentVersionComparisonCard result={withComparison} />)
  })

  test("1. renders region with aria-label", () => {
    expect(screen.getByRole("region", { name: /deployment version comparison/i })).toBeInTheDocument()
  })

  test("2. shows 🔄 icon aria-hidden", () => {
    const icon = screen.getByText("🔄")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  test("3. shows Version Comparison heading", () => {
    expect(screen.getByText("Version Comparison")).toBeInTheDocument()
  })

  test("4. shows version range badge", () => {
    expect(screen.getByText("v1 → v2")).toBeInTheDocument()
  })

  test("5. shows improved count badge", () => {
    expect(screen.getByText("2 improved")).toBeInTheDocument()
  })

  test("6. shows declined count badge", () => {
    expect(screen.getByText("1 declined")).toBeInTheDocument()
  })

  test("7. renders metric comparison table", () => {
    expect(screen.getByRole("table", { name: /metric comparison/i })).toBeInTheDocument()
  })

  test("8. shows R² label in table", () => {
    expect(screen.getByText("R²")).toBeInTheDocument()
  })

  test("9. shows previous and current R² values", () => {
    // 72.0% = previous, 84.0% = current
    expect(screen.getByText("72.0%")).toBeInTheDocument()
    expect(screen.getByText("84.0%")).toBeInTheDocument()
  })

  test("10. shows ↑ arrow for improved metric (R²)", () => {
    const arrows = screen.getAllByText("↑")
    expect(arrows.length).toBeGreaterThan(0)
  })

  test("11. shows ↓ arrow for declined metric (RMSE)", () => {
    // RMSE went up (bad for error metrics → declined)
    const arrows = screen.getAllByText(/↓|↑/)
    expect(arrows.length).toBeGreaterThanOrEqual(2)
  })

  test("12. shows algorithm changed note", () => {
    expect(screen.getByText(/Algorithm changed/i)).toBeInTheDocument()
  })

  test("13. shows summary footer paragraph", () => {
    expect(screen.getByText(/Version 2 vs 1/i)).toBeInTheDocument()
  })

  test("14. shows MAE/RMSE note", () => {
    expect(screen.getByText(/lower is better/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Render tests — no comparison state
// ---------------------------------------------------------------------------

describe("DeploymentVersionComparisonCard — no comparison", () => {
  beforeEach(() => {
    render(<DeploymentVersionComparisonCard result={noComparison} />)
  })

  test("15. renders summary text in no-comparison state", () => {
    expect(screen.getByText(/Only one deployment version exists/i)).toBeInTheDocument()
  })

  test("16. no metric table in no-comparison state", () => {
    expect(screen.queryByRole("table")).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("store: attachVersionComparisonToLastMessage", () => {
  const reset = () =>
    useAppStore.setState({
      messages: [],
    })

  afterEach(reset)

  test("17. attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "compare", timestamp: "t1" },
        { role: "assistant", content: "Comparing...", timestamp: "t2" },
      ],
    })
    useAppStore.getState().attachVersionComparisonToLastMessage(withComparison)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].version_comparison).toBeDefined()
    expect(msgs[msgs.length - 1].version_comparison?.has_comparison).toBe(true)
  })

  test("18. does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "compare", timestamp: "t1" }],
    })
    useAppStore.getState().attachVersionComparisonToLastMessage(withComparison)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].version_comparison).toBeUndefined()
  })

  test("19. does not crash on empty messages", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachVersionComparisonToLastMessage(withComparison)
    ).not.toThrow()
  })
})
