/**
 * Tests for DriftImportanceCard and attachDriftImportanceRankingToLastMessage.
 *
 * Covers:
 *  1.  Renders container with data-testid "drift-importance-card"
 *  2.  action_required verdict shows rose border
 *  3.  attention verdict shows amber border
 *  4.  monitoring verdict shows sky border
 *  5.  clear verdict shows emerald border
 *  6.  no_importances verdict shows no-importances-message
 *  7.  verdict badge is rendered with correct label
 *  8.  feature count badge is shown when has_importances=true
 *  9.  sample count badge is shown when n_samples > 0
 * 10.  features table is rendered when has_importances=true
 * 11.  feature row renders with correct data-testid
 * 12.  priority badge is rendered per feature
 * 13.  priority alert shown when critical_count > 0
 * 14.  priority alert shown when high_count > 0
 * 15.  priority alert not shown when both counts are 0
 * 16.  summary paragraph is rendered
 * 17.  sr-only figcaption is present for accessibility
 * 18.  Store: attachDriftImportanceRankingToLastMessage attaches to last assistant message
 * 19.  Store: does not attach to user message
 * 20.  Store: does not crash with empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { DriftImportanceCard } from "@/components/deploy/drift-importance-card"
import type { DriftImportanceRankingResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeFeature(name: string, priority: string = "no_drift") {
  return {
    name,
    importance_pct: 30.0,
    rank: 1,
    drift_pct: 0.0,
    risk_score: 0,
    priority: priority as never,
    feature_type: "numeric" as never,
    drift_details: "All values within training range",
  }
}

function makeResult(
  overrides: Partial<DriftImportanceRankingResult> = {}
): DriftImportanceRankingResult {
  return {
    deployment_id: "dep-1",
    has_importances: true,
    ranked_features: [makeFeature("revenue"), makeFeature("units")],
    n_features: 2,
    n_samples: 100,
    critical_count: 0,
    high_count: 0,
    medium_count: 0,
    low_count: 0,
    no_drift_count: 2,
    verdict: "clear",
    summary: "No drift detected across 2 features.",
    ...overrides,
  }
}

function makeNoImportancesResult(): DriftImportanceRankingResult {
  return {
    deployment_id: "dep-2",
    has_importances: false,
    ranked_features: [],
    n_features: 0,
    n_samples: 50,
    critical_count: 0,
    high_count: 0,
    medium_count: 0,
    low_count: 0,
    no_drift_count: 0,
    verdict: "no_importances",
    summary: "Feature importances not available for this model type.",
  }
}

// 1. Container renders
test("renders container with data-testid", () => {
  render(<DriftImportanceCard result={makeResult()} />)
  expect(screen.getByTestId("drift-importance-card")).toBeInTheDocument()
})

// 2. action_required shows rose border
test("action_required verdict shows rose border", () => {
  const { container } = render(
    <DriftImportanceCard result={makeResult({ verdict: "action_required" })} />
  )
  expect((container.firstChild as HTMLElement).className).toContain("border-rose")
})

// 3. attention shows amber border
test("attention verdict shows amber border", () => {
  const { container } = render(
    <DriftImportanceCard result={makeResult({ verdict: "attention" })} />
  )
  expect((container.firstChild as HTMLElement).className).toContain("border-amber")
})

// 4. monitoring shows sky border
test("monitoring verdict shows sky border", () => {
  const { container } = render(
    <DriftImportanceCard result={makeResult({ verdict: "monitoring" })} />
  )
  expect((container.firstChild as HTMLElement).className).toContain("border-sky")
})

// 5. clear shows emerald border
test("clear verdict shows emerald border", () => {
  const { container } = render(<DriftImportanceCard result={makeResult({ verdict: "clear" })} />)
  expect((container.firstChild as HTMLElement).className).toContain("border-emerald")
})

// 6. no_importances shows message
test("no_importances shows no-importances-message", () => {
  render(<DriftImportanceCard result={makeNoImportancesResult()} />)
  expect(screen.getByTestId("no-importances-message")).toBeInTheDocument()
})

// 7. verdict badge
test("verdict badge renders with correct label", () => {
  render(<DriftImportanceCard result={makeResult({ verdict: "clear" })} />)
  expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Clear")
})

// 8. feature count badge when has_importances=true
test("feature count badge is shown when has_importances=true", () => {
  render(<DriftImportanceCard result={makeResult()} />)
  expect(screen.getByTestId("feature-count-badge")).toBeInTheDocument()
  expect(screen.getByTestId("feature-count-badge")).toHaveTextContent("2")
})

// 9. sample count badge
test("sample count badge is shown when n_samples > 0", () => {
  render(<DriftImportanceCard result={makeResult({ n_samples: 250 })} />)
  expect(screen.getByTestId("sample-count-badge")).toHaveTextContent("250")
})

// 10. features table rendered
test("features table is rendered when has_importances=true", () => {
  render(<DriftImportanceCard result={makeResult()} />)
  expect(screen.getByTestId("features-table")).toBeInTheDocument()
})

// 11. feature row renders
test("feature row renders with data-testid", () => {
  render(<DriftImportanceCard result={makeResult()} />)
  expect(screen.getByTestId("feature-row-revenue")).toBeInTheDocument()
  expect(screen.getByTestId("feature-row-units")).toBeInTheDocument()
})

// 12. priority badge per feature
test("priority badge is rendered per feature", () => {
  const features = [makeFeature("revenue", "critical"), makeFeature("units", "no_drift")]
  render(<DriftImportanceCard result={makeResult({ ranked_features: features })} />)
  expect(screen.getByTestId("priority-badge-revenue")).toHaveTextContent("Critical")
  expect(screen.getByTestId("priority-badge-units")).toHaveTextContent("No Drift")
})

// 13. priority alert when critical_count > 0
test("priority alert shown when critical_count > 0", () => {
  render(<DriftImportanceCard result={makeResult({ critical_count: 2 })} />)
  expect(screen.getByTestId("priority-alert")).toBeInTheDocument()
})

// 14. priority alert when high_count > 0
test("priority alert shown when high_count > 0", () => {
  render(<DriftImportanceCard result={makeResult({ high_count: 1 })} />)
  expect(screen.getByTestId("priority-alert")).toBeInTheDocument()
})

// 15. no priority alert when both are 0
test("priority alert not shown when counts are 0", () => {
  render(<DriftImportanceCard result={makeResult({ critical_count: 0, high_count: 0 })} />)
  expect(screen.queryByTestId("priority-alert")).not.toBeInTheDocument()
})

// 16. summary paragraph
test("summary paragraph is rendered", () => {
  render(<DriftImportanceCard result={makeResult({ summary: "No drift in 2 features." })} />)
  expect(screen.getByTestId("summary")).toHaveTextContent("No drift in 2 features.")
})

// 17. Accessibility figcaption
test("sr-only figcaption is present", () => {
  const { container } = render(<DriftImportanceCard result={makeResult()} />)
  const figcaption = container.querySelector("figcaption")
  expect(figcaption).toBeInTheDocument()
  expect(figcaption).toHaveClass("sr-only")
})

// 18-20. Store actions
describe("attachDriftImportanceRankingToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  test("attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "test", id: "u1" },
        { role: "assistant", content: "reply", id: "a1" },
      ],
    })
    const result = makeResult()
    useAppStore.getState().attachDriftImportanceRankingToLastMessage(result)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].drift_importance_ranking).toEqual(result)
  })

  test("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "test", id: "u1" }],
    })
    const result = makeResult()
    useAppStore.getState().attachDriftImportanceRankingToLastMessage(result)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].drift_importance_ranking).toBeUndefined()
  })

  test("does not crash with empty messages", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachDriftImportanceRankingToLastMessage(makeResult())
    ).not.toThrow()
  })
})
