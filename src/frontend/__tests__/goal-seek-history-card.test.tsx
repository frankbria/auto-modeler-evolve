/**
 * Tests for GoalSeekHistoryCard component and goal-seek-history store action.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { GoalSeekHistoryCard } from "@/components/deploy/goal-seek-history-card"
import type { GoalSeekHistoryResult } from "@/lib/types"

// Minimal suggestion for test fixtures
const suggestion = {
  feature: "units_sold",
  current_mean: 100,
  suggested_value: 200,
  direction: "increase" as const,
  change_pct: 100,
}

const entry1 = {
  id: "rec-1",
  target_column: "revenue",
  problem_type: "regression",
  algorithm_plain: "Linear Regression",
  target_value_str: "5000000",
  achieved_value_str: "4850000",
  achieved: false,
  gap_pct: 3.0,
  suggestions: [suggestion],
  fixed_features: {},
  summary: "Try increasing units to reach $5M.",
  created_at: new Date(Date.now() - 60_000).toISOString(),
}

const entry2 = {
  id: "rec-2",
  target_column: "revenue",
  problem_type: "regression",
  algorithm_plain: "Linear Regression",
  target_value_str: "4000000",
  achieved_value_str: "4000000",
  achieved: true,
  gap_pct: null,
  suggestions: [suggestion],
  fixed_features: { region: 1 },
  summary: "Goal achieved — increase units to 180.",
  created_at: new Date(Date.now() - 3_600_000).toISOString(),
}

const twoEntries: GoalSeekHistoryResult = {
  deployment_id: "dep-1",
  count: 2,
  entries: [entry1, entry2],
}

const emptyHistory: GoalSeekHistoryResult = {
  deployment_id: "dep-1",
  count: 0,
  entries: [],
}

describe("GoalSeekHistoryCard", () => {
  it("renders the card heading", () => {
    render(<GoalSeekHistoryCard result={twoEntries} />)
    expect(screen.getByText("Goal Seek History")).toBeInTheDocument()
  })

  it("shows the scenario count badge", () => {
    render(<GoalSeekHistoryCard result={twoEntries} />)
    expect(screen.getByTestId("history-count-badge")).toHaveTextContent("2 scenarios")
  })

  it("renders all history entries", () => {
    render(<GoalSeekHistoryCard result={twoEntries} />)
    expect(screen.getByTestId("history-entry-0")).toBeInTheDocument()
    expect(screen.getByTestId("history-entry-1")).toBeInTheDocument()
  })

  it("shows achieved badge for achieved entry", () => {
    render(<GoalSeekHistoryCard result={twoEntries} />)
    expect(screen.getByTestId("history-achieved-badge-1")).toBeInTheDocument()
  })

  it("shows best-effort badge for non-achieved entry", () => {
    render(<GoalSeekHistoryCard result={twoEntries} />)
    expect(screen.getByTestId("history-best-effort-badge-0")).toBeInTheDocument()
  })

  it("displays target and achieved values for first entry", () => {
    render(<GoalSeekHistoryCard result={twoEntries} />)
    expect(screen.getByTestId("history-target-0")).toHaveTextContent("5,000,000")
    expect(screen.getByTestId("history-achieved-0")).toHaveTextContent("4,850,000")
  })

  it("displays target and achieved values for second entry", () => {
    render(<GoalSeekHistoryCard result={twoEntries} />)
    expect(screen.getByTestId("history-target-1")).toHaveTextContent("4,000,000")
    expect(screen.getByTestId("history-achieved-1")).toHaveTextContent("4,000,000")
  })

  it("shows gap indicator for non-achieved entry", () => {
    render(<GoalSeekHistoryCard result={twoEntries} />)
    const entry0 = screen.getByTestId("history-entry-0")
    expect(entry0.textContent).toContain("3%")
    expect(entry0.textContent).toContain("gap from target")
  })

  it("shows suggestion arrow and feature for first entry", () => {
    render(<GoalSeekHistoryCard result={twoEntries} />)
    const entry0 = screen.getByTestId("history-entry-0")
    expect(entry0.textContent).toContain("units sold")
    expect(entry0.textContent).toContain("200")
  })

  it("shows fixed features when present", () => {
    render(<GoalSeekHistoryCard result={twoEntries} />)
    const entry1El = screen.getByTestId("history-entry-1")
    expect(entry1El.textContent).toContain("region=1")
  })

  it("shows empty state when no history", () => {
    render(<GoalSeekHistoryCard result={emptyHistory} />)
    expect(screen.getByTestId("history-empty-state")).toBeInTheDocument()
    expect(screen.queryByTestId("history-count-badge")).not.toBeInTheDocument()
  })

  it("renders sr-only figcaption for accessibility", () => {
    const { container } = render(<GoalSeekHistoryCard result={twoEntries} />)
    const figcaption = container.querySelector("figcaption.sr-only")
    expect(figcaption).toBeInTheDocument()
    expect(figcaption?.textContent).toContain("Goal seek history")
    expect(figcaption?.textContent).toContain("2 scenarios")
  })

  it("shows empty state sr-only text when no entries", () => {
    const { container } = render(<GoalSeekHistoryCard result={emptyHistory} />)
    const figcaption = container.querySelector("figcaption.sr-only")
    expect(figcaption?.textContent).toContain("No scenarios recorded yet")
  })
})

// ---------------------------------------------------------------------------
// Zustand store action test
// ---------------------------------------------------------------------------

describe("attachGoalSeekHistoryToLastMessage store action", () => {
  it("attaches goal_seek_history to the last assistant message", () => {
    const { useAppStore } = jest.requireActual<typeof import("@/lib/store")>("@/lib/store")
    const store = useAppStore.getState()

    // Seed messages
    store.setMessages([
      { id: "1", role: "user", content: "compare my goal seek scenarios" },
      { id: "2", role: "assistant", content: "Here are your past runs:" },
    ])

    store.attachGoalSeekHistoryToLastMessage(twoEntries)

    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.goal_seek_history).toBeDefined()
    expect(last.goal_seek_history?.count).toBe(2)
    expect(last.goal_seek_history?.entries).toHaveLength(2)
  })
})
