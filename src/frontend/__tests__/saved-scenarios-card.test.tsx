/**
 * Tests for SavedScenariosCard component and store action.
 *
 * Covers:
 *  1.  Renders figure with correct data-testid
 *  2.  Renders 📋 icon (aria-hidden)
 *  3.  Shows scenario count badge
 *  4.  Shows target column badge when provided
 *  5.  Shows Regression badge
 *  6.  Shows Classification badge
 *  7.  Empty state message when no scenarios
 *  8.  Renders scenario rows when scenarios present
 *  9.  Shows scenario name in each row
 * 10.  Shows prediction value in each row
 * 11.  Best badge shown on best scenario (regression, multiple)
 * 12.  Worst badge shown on worst scenario (regression, multiple)
 * 13.  No best/worst badges for single scenario
 * 14.  Regression bars rendered for numeric predictions
 * 15.  Confidence shown for classification with non-null confidence
 * 16.  Range callout shown for regression with spread ≥ 2 scenarios
 * 17.  Summary paragraph rendered
 * 18.  sr-only figcaption present
 * 19.  Store: attachSavedScenariosToLastMessage attaches to last assistant message
 * 20.  Store: does not attach to user message
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import "@testing-library/jest-dom"
import { SavedScenariosCard } from "@/components/deploy/saved-scenarios-card"
import { SavedScenariosResult, SavedScenarioEntry, ChatMessage } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const makeEntry = (overrides: Partial<SavedScenarioEntry> = {}): SavedScenarioEntry => ({
  id: "id-1",
  name: "Base Case",
  inputs: { discount: 0.1 },
  prediction_value: "150",
  prediction_numeric: 150,
  confidence: null,
  created_at: "2026-06-10T12:00:00",
  ...overrides,
})

const baseResult: SavedScenariosResult = {
  deployment_id: "dep-1",
  scenarios: [],
  n_scenarios: 0,
  problem_type: "regression",
  target_column: "revenue",
  best_scenario: null,
  worst_scenario: null,
  spread: null,
  summary: "No saved scenarios yet.",
}

// 1. Renders figure with correct data-testid
test("renders figure with data-testid", () => {
  render(<SavedScenariosCard result={baseResult} />)
  expect(screen.getByTestId("saved-scenarios-card")).toBeInTheDocument()
})

// 2. 📋 icon
test("renders clipboard icon", () => {
  render(<SavedScenariosCard result={baseResult} />)
  expect(screen.getByText("📋")).toBeInTheDocument()
})

// 3. Scenario count badge — empty
test("shows 0 scenarios badge", () => {
  render(<SavedScenariosCard result={baseResult} />)
  expect(screen.getByText("0 scenarios")).toBeInTheDocument()
})

// 4. Target column badge
test("shows target column badge", () => {
  render(<SavedScenariosCard result={baseResult} />)
  expect(screen.getByText("target: revenue")).toBeInTheDocument()
})

// 5. Regression badge
test("shows Regression badge", () => {
  render(<SavedScenariosCard result={baseResult} />)
  expect(screen.getByText("Regression")).toBeInTheDocument()
})

// 6. Classification badge
test("shows Classification badge for classification", () => {
  render(<SavedScenariosCard result={{ ...baseResult, problem_type: "classification" }} />)
  expect(screen.getByText("Classification")).toBeInTheDocument()
})

// 7. Empty state
test("shows empty state message when no scenarios", () => {
  render(<SavedScenariosCard result={baseResult} />)
  expect(screen.getByText(/No saved scenarios yet\. Say/, { selector: "p" })).toBeInTheDocument()
})

// 8-9. Scenario rows and names
const withScenarios: SavedScenariosResult = {
  ...baseResult,
  scenarios: [
    makeEntry({ id: "s1", name: "Q2 Base", prediction_value: "100", prediction_numeric: 100 }),
    makeEntry({ id: "s2", name: "Q2 Optimistic", prediction_value: "200", prediction_numeric: 200 }),
  ],
  n_scenarios: 2,
  best_scenario: makeEntry({ id: "s2", name: "Q2 Optimistic", prediction_value: "200", prediction_numeric: 200 }),
  worst_scenario: makeEntry({ id: "s1", name: "Q2 Base", prediction_value: "100", prediction_numeric: 100 }),
  spread: 100,
  summary: "2 saved scenarios.",
}

test("renders scenario rows", () => {
  render(<SavedScenariosCard result={withScenarios} />)
  expect(screen.getByText("Q2 Base")).toBeInTheDocument()
  expect(screen.getByText("Q2 Optimistic")).toBeInTheDocument()
})

// 10. Prediction value displayed
test("shows prediction values", () => {
  render(<SavedScenariosCard result={withScenarios} />)
  expect(screen.getAllByText("100").length).toBeGreaterThan(0)
  expect(screen.getAllByText("200").length).toBeGreaterThan(0)
})

// 11. Best badge
test("shows Best badge on highest-prediction scenario", () => {
  render(<SavedScenariosCard result={withScenarios} />)
  expect(screen.getByText("Best")).toBeInTheDocument()
})

// 12. Worst badge
test("shows Worst badge on lowest-prediction scenario", () => {
  render(<SavedScenariosCard result={withScenarios} />)
  expect(screen.getByText("Worst")).toBeInTheDocument()
})

// 13. No best/worst for single scenario
test("no Best/Worst badges for single scenario", () => {
  const single: SavedScenariosResult = {
    ...baseResult,
    scenarios: [makeEntry()],
    n_scenarios: 1,
    best_scenario: makeEntry(),
    worst_scenario: makeEntry(),
    spread: 0,
    summary: "1 saved scenario.",
  }
  render(<SavedScenariosCard result={single} />)
  expect(screen.queryByText("Best")).not.toBeInTheDocument()
  expect(screen.queryByText("Worst")).not.toBeInTheDocument()
})

// 14. Regression bars
test("renders regression bars for numeric predictions", () => {
  render(<SavedScenariosCard result={withScenarios} />)
  const bars = screen.getAllByRole("img", { name: /Prediction bar/ })
  expect(bars.length).toBe(2)
})

// 15. Classification confidence
test("shows confidence for classification entries", () => {
  const clsResult: SavedScenariosResult = {
    ...baseResult,
    problem_type: "classification",
    scenarios: [
      makeEntry({ prediction_value: "churn", prediction_numeric: null, confidence: 0.85 }),
    ],
    n_scenarios: 1,
    summary: "1 saved scenario.",
  }
  render(<SavedScenariosCard result={clsResult} />)
  expect(screen.getByText(/confidence: 85%/)).toBeInTheDocument()
})

// 16. Range callout for regression spread
test("shows Range callout for regression with 2+ scenarios", () => {
  render(<SavedScenariosCard result={withScenarios} />)
  expect(screen.getByText(/Range/)).toBeInTheDocument()
})

// 17. Summary paragraph
test("renders summary paragraph", () => {
  render(<SavedScenariosCard result={withScenarios} />)
  expect(screen.getByText("2 saved scenarios.")).toBeInTheDocument()
})

// 18. sr-only figcaption
test("has sr-only figcaption", () => {
  render(<SavedScenariosCard result={withScenarios} />)
  const figcaption = document.querySelector(".sr-only")
  expect(figcaption).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Store action
// ---------------------------------------------------------------------------

describe("attachSavedScenariosToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  // 19. Attaches to last assistant message
  test("attaches saved_scenarios to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "compare scenarios", id: "u1" } as ChatMessage,
        { role: "assistant", content: "Here are your scenarios.", id: "a1" } as ChatMessage,
      ],
    })
    const { attachSavedScenariosToLastMessage } = useAppStore.getState()
    attachSavedScenariosToLastMessage(baseResult)

    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].saved_scenarios).toEqual(baseResult)
  })

  // 20. Does not attach to user message
  test("does not attach to user message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "compare scenarios", id: "u1" } as ChatMessage,
      ],
    })
    const { attachSavedScenariosToLastMessage } = useAppStore.getState()
    attachSavedScenariosToLastMessage(baseResult)

    const msgs = useAppStore.getState().messages
    expect((msgs[msgs.length - 1] as ChatMessage).saved_scenarios).toBeUndefined()
  })
})
