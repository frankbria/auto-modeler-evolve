/**
 * Tests for DeploymentScorecardCard and attachDeploymentScorecardToLastMessage.
 *
 * Covers:
 *  1.  Renders container with data-testid "deployment-scorecard-card"
 *  2.  Empty result shows summary text and no entries
 *  3.  Single entry shows rank 1 medal 🥇
 *  4.  Winner entry has "Top Performer" badge
 *  5.  Non-winner entry does not have "Top Performer" badge
 *  6.  Each entry row has data-testid "scorecard-entry-{rank}"
 *  7.  Entry shows algorithm_plain and target_column
 *  8.  Entry shows composite_score
 *  9.  Production environment shows Production badge
 * 10.  Staging environment shows Staging badge
 * 11.  request_count is displayed in signal pills
 * 12.  feedback_accuracy shown when available
 * 13.  "No feedback" shown when accuracy_score is null
 * 14.  SLA p95 shown when available
 * 15.  "No SLA data" shown when sla_score is null
 * 16.  Summary text is rendered
 * 17.  scorecard-entries container present when entries exist
 * 18.  Store: attachDeploymentScorecardToLastMessage attaches to last assistant message
 * 19.  Store: does not attach to user message
 * 20.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { DeploymentScorecardCard } from "@/components/deploy/deployment-scorecard-card"
import type { DeploymentScorecardResult, DeploymentScorecardEntry } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeEntry(overrides: Partial<DeploymentScorecardEntry> = {}): DeploymentScorecardEntry {
  return {
    rank: 1,
    deployment_id: "dep-1",
    algorithm: "random_forest_regressor",
    algorithm_plain: "Random Forest",
    target_column: "revenue",
    environment: "production",
    request_count: 150,
    feedback_accuracy: 0.82,
    p95_ms: 120,
    age_days: 15,
    deployed_at: "2025-01-01T00:00:00",
    dashboard_url: "/predict/dep-1",
    composite_score: 72,
    usage_score: 50,
    sla_score: 95,
    accuracy_score: 82,
    freshness_score: 80,
    ...overrides,
  }
}

const emptyResult: DeploymentScorecardResult = {
  total: 0,
  entries: [],
  winner_id: null,
  summary: "No active deployments found for this project.",
}

const singleResult: DeploymentScorecardResult = {
  total: 1,
  entries: [makeEntry()],
  winner_id: "dep-1",
  summary: "1 deployment: Random Forest → revenue (score 72/100, 150 predictions).",
}

const multiResult: DeploymentScorecardResult = {
  total: 2,
  entries: [
    makeEntry({ rank: 1, deployment_id: "dep-1", composite_score: 72 }),
    makeEntry({
      rank: 2,
      deployment_id: "dep-2",
      algorithm_plain: "Linear Regression",
      target_column: "sales",
      environment: "staging",
      composite_score: 45,
      accuracy_score: null,
      sla_score: null,
      p95_ms: null,
      request_count: 5,
    }),
  ],
  winner_id: "dep-1",
  summary: "2 deployments ranked. Top performer: Random Forest → revenue (score 72/100).",
}

// 1. Container rendered
test("renders deployment-scorecard-card container", () => {
  render(<DeploymentScorecardCard result={singleResult} />)
  expect(screen.getByTestId("deployment-scorecard-card")).toBeInTheDocument()
})

// 2. Empty result shows summary and no entries
test("empty result shows summary text", () => {
  render(<DeploymentScorecardCard result={emptyResult} />)
  expect(screen.getByText("No active deployments found for this project.")).toBeInTheDocument()
})

test("empty result has no entry rows", () => {
  render(<DeploymentScorecardCard result={emptyResult} />)
  expect(screen.queryByTestId("scorecard-entry-1")).not.toBeInTheDocument()
})

// 3. Single entry rank 1 medal
test("rank 1 shows gold medal emoji", () => {
  render(<DeploymentScorecardCard result={singleResult} />)
  expect(screen.getByRole("img", { name: "Rank 1" })).toBeInTheDocument()
})

// 4. Winner has Top Performer badge
test("winner entry has Top Performer badge", () => {
  render(<DeploymentScorecardCard result={multiResult} />)
  expect(screen.getByText("Top Performer")).toBeInTheDocument()
})

// 5. Non-winner does not have Top Performer badge
test("non-winner entry does not have multiple Top Performer badges", () => {
  render(<DeploymentScorecardCard result={multiResult} />)
  const badges = screen.getAllByText("Top Performer")
  expect(badges).toHaveLength(1)
})

// 6. Entry row has correct data-testid
test("entry rows have scorecard-entry-{rank} testids", () => {
  render(<DeploymentScorecardCard result={multiResult} />)
  expect(screen.getByTestId("scorecard-entry-1")).toBeInTheDocument()
  expect(screen.getByTestId("scorecard-entry-2")).toBeInTheDocument()
})

// 7. Entry shows algorithm and target
test("entry shows algorithm_plain and target_column", () => {
  render(<DeploymentScorecardCard result={singleResult} />)
  expect(screen.getAllByText(/Random Forest/).length).toBeGreaterThan(0)
  expect(screen.getAllByText(/revenue/).length).toBeGreaterThan(0)
})

// 8. Entry shows composite score
test("entry shows composite score", () => {
  render(<DeploymentScorecardCard result={singleResult} />)
  expect(screen.getByText("72")).toBeInTheDocument()
})

// 9. Production environment badge
test("shows Production badge for production environment", () => {
  render(<DeploymentScorecardCard result={singleResult} />)
  expect(screen.getByText("Production")).toBeInTheDocument()
})

// 10. Staging environment badge
test("shows Staging badge for staging environment", () => {
  render(<DeploymentScorecardCard result={multiResult} />)
  expect(screen.getByText("Staging")).toBeInTheDocument()
})

// 11. request_count shown in signal pills
test("shows request count in signal pills", () => {
  render(<DeploymentScorecardCard result={singleResult} />)
  expect(screen.getByText("150 predictions")).toBeInTheDocument()
})

// 12. feedback accuracy shown when available
test("shows feedback accuracy when accuracy_score is provided", () => {
  render(<DeploymentScorecardCard result={singleResult} />)
  expect(screen.getByText("82% accurate")).toBeInTheDocument()
})

// 13. No feedback shown when null
test('shows "No feedback" when accuracy_score is null', () => {
  const result: DeploymentScorecardResult = {
    ...singleResult,
    entries: [makeEntry({ accuracy_score: null, feedback_accuracy: null })],
  }
  render(<DeploymentScorecardCard result={result} />)
  expect(screen.getByText("No feedback")).toBeInTheDocument()
})

// 14. SLA shown when available
test("shows SLA p95 when available", () => {
  render(<DeploymentScorecardCard result={singleResult} />)
  expect(screen.getByText("p95 120ms")).toBeInTheDocument()
})

// 15. No SLA data when null
test('shows "No SLA data" when sla_score is null', () => {
  const result: DeploymentScorecardResult = {
    ...singleResult,
    entries: [makeEntry({ sla_score: null, p95_ms: null })],
  }
  render(<DeploymentScorecardCard result={result} />)
  expect(screen.getByText("No SLA data")).toBeInTheDocument()
})

// 16. Summary text rendered
test("summary text is rendered", () => {
  render(<DeploymentScorecardCard result={multiResult} />)
  expect(
    screen.getByText(
      "2 deployments ranked. Top performer: Random Forest → revenue (score 72/100)."
    )
  ).toBeInTheDocument()
})

// 17. scorecard-entries container present
test("scorecard-entries container is present when entries exist", () => {
  render(<DeploymentScorecardCard result={singleResult} />)
  expect(screen.getByTestId("scorecard-entries")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Store tests
// ---------------------------------------------------------------------------

describe("attachDeploymentScorecardToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  // 18. Attaches to last assistant message
  test("attaches to the last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "hello", timestamp: new Date().toISOString() },
        { id: "2", role: "assistant", content: "hi", timestamp: new Date().toISOString() },
      ],
    })
    const { attachDeploymentScorecardToLastMessage } = useAppStore.getState()
    attachDeploymentScorecardToLastMessage(singleResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].deployment_scorecard).toEqual(singleResult)
  })

  // 19. Does not attach to user message
  test("does not attach to a user message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "hello", timestamp: new Date().toISOString() },
      ],
    })
    const { attachDeploymentScorecardToLastMessage } = useAppStore.getState()
    attachDeploymentScorecardToLastMessage(singleResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].deployment_scorecard).toBeUndefined()
  })

  // 20. Does not crash when messages empty
  test("does not crash when messages list is empty", () => {
    const { attachDeploymentScorecardToLastMessage } = useAppStore.getState()
    expect(() => attachDeploymentScorecardToLastMessage(singleResult)).not.toThrow()
  })
})
