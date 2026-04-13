/**
 * Tests for ABTestChatCard and Zustand store action.
 *
 * Covers:
 *  1.  Status view: renders region with aria-label "A/B test status"
 *  2.  Status view: shows ⚗️ icon (aria-hidden)
 *  3.  Status view: shows "A/B Test Status" heading
 *  4.  Status view: shows "Live" badge
 *  5.  Status view: shows traffic split labels (champion/challenger)
 *  6.  Status view: shows champion split percentage
 *  7.  Status view: shows challenger split percentage
 *  8.  Status view: shows champion metrics column
 *  9.  Status view: shows challenger metrics column
 * 10.  Status view: shows significance note
 * 11.  Promoted view: renders region "Challenger promoted"
 * 12.  Promoted view: shows "Promoted ✓" badge
 * 13.  Promoted view: shows URL-unchanged note
 * 14.  Ended view: renders region "A/B test ended"
 * 15.  None view: renders region "No active A/B test"
 * 16.  None view: shows guidance text
 * 17.  Store: attachABTestResultToLastMessage attaches to last assistant message
 * 18.  Store: does not attach to user message
 * 19.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ABTestChatCard } from "@/components/deploy/ab-test-chat-card"
import type { ABTestChatResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const noMetrics = {
  request_count: 0,
  avg_confidence: null,
  p95_ms: null,
  avg_prediction: null,
}

const withMetrics = {
  request_count: 45,
  avg_confidence: 0.87,
  p95_ms: 180,
  avg_prediction: 3.14,
}

const statusResult: ABTestChatResult = {
  action: "status",
  summary:
    "A/B test active: 80% champion (linear_regression) / 20% challenger (random_forest). Champion: 45 requests. Challenger: 12 requests.",
  id: "ab-1",
  champion_id: "dep-1",
  challenger_id: "dep-2",
  champion_algorithm: "linear_regression",
  challenger_algorithm: "random_forest",
  champion_split_pct: 80,
  challenger_split_pct: 20,
  is_active: true,
  champion_metrics: withMetrics,
  challenger_metrics: noMetrics,
  significance: {
    significant: false,
    p_value: null,
    note: "Need 5 more samples per variant (minimum 5)",
  },
  created_at: "2026-04-10T08:00:00",
}

const promotedResult: ABTestChatResult = {
  action: "promoted",
  summary:
    "Challenger (random_forest) promoted to champion. Your prediction URL stays the same.",
}

const endedResult: ABTestChatResult = {
  action: "ended",
  summary: "A/B test ended. Champion model remains active.",
}

const noneResult: ABTestChatResult = {
  action: "none",
  summary:
    "No active A/B test. You can start one from the Deployment panel once you have a second trained model as a challenger.",
}

// ---------------------------------------------------------------------------
// Render tests — status view
// ---------------------------------------------------------------------------

test("1. status view renders region with aria-label 'A/B test status'", () => {
  render(<ABTestChatCard result={statusResult} />)
  expect(screen.getByRole("region", { name: "A/B test status" })).toBeInTheDocument()
})

test("2. status view shows ⚗️ icon with aria-hidden", () => {
  render(<ABTestChatCard result={statusResult} />)
  const icon = screen.getByText("⚗️")
  expect(icon).toHaveAttribute("aria-hidden", "true")
})

test("3. status view shows 'A/B Test Status' heading", () => {
  render(<ABTestChatCard result={statusResult} />)
  expect(screen.getByText("A/B Test Status")).toBeInTheDocument()
})

test("4. status view shows 'Live' badge", () => {
  render(<ABTestChatCard result={statusResult} />)
  expect(screen.getByText("Live")).toBeInTheDocument()
})

test("5. status view shows champion and challenger labels", () => {
  render(<ABTestChatCard result={statusResult} />)
  expect(screen.getByText(/Champion \(linear_regression\)/)).toBeInTheDocument()
  expect(screen.getByText(/Challenger \(random_forest\)/)).toBeInTheDocument()
})

test("6. status view shows champion split percentage", () => {
  render(<ABTestChatCard result={statusResult} />)
  expect(screen.getByText("80% of traffic")).toBeInTheDocument()
})

test("7. status view shows challenger split percentage", () => {
  render(<ABTestChatCard result={statusResult} />)
  expect(screen.getByText("20% of traffic")).toBeInTheDocument()
})

test("8. status view shows champion metrics column", () => {
  render(<ABTestChatCard result={statusResult} />)
  // Champion column header
  expect(screen.getAllByText("Champion").length).toBeGreaterThan(0)
  // Champion request count
  expect(screen.getByText("45")).toBeInTheDocument()
})

test("9. status view shows challenger metrics column", () => {
  render(<ABTestChatCard result={statusResult} />)
  expect(screen.getAllByText("Challenger").length).toBeGreaterThan(0)
})

test("10. status view shows significance note", () => {
  render(<ABTestChatCard result={statusResult} />)
  expect(screen.getByText(/Need 5 more samples/)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Render tests — promoted view
// ---------------------------------------------------------------------------

test("11. promoted view renders region 'Challenger promoted'", () => {
  render(<ABTestChatCard result={promotedResult} />)
  expect(screen.getByRole("region", { name: "Challenger promoted" })).toBeInTheDocument()
})

test("12. promoted view shows 'Promoted ✓' badge", () => {
  render(<ABTestChatCard result={promotedResult} />)
  expect(screen.getByText("Promoted ✓")).toBeInTheDocument()
})

test("13. promoted view shows URL-unchanged note", () => {
  render(<ABTestChatCard result={promotedResult} />)
  expect(screen.getByText(/URL is unchanged/)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Render tests — ended view
// ---------------------------------------------------------------------------

test("14. ended view renders region 'A/B test ended'", () => {
  render(<ABTestChatCard result={endedResult} />)
  expect(screen.getByRole("region", { name: "A/B test ended" })).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Render tests — none view
// ---------------------------------------------------------------------------

test("15. none view renders region 'No active A/B test'", () => {
  render(<ABTestChatCard result={noneResult} />)
  expect(screen.getByRole("region", { name: "No active A/B test" })).toBeInTheDocument()
})

test("16. none view shows guidance text about training a second model", () => {
  render(<ABTestChatCard result={noneResult} />)
  expect(screen.getByText(/Train a second model/)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

beforeEach(() => {
  useAppStore.setState({ messages: [] })
})

test("17. store attaches to last assistant message", () => {
  useAppStore.setState({
    messages: [
      { id: "m1", role: "assistant", content: "Checking test status.", timestamp: "" },
    ],
  })
  useAppStore.getState().attachABTestResultToLastMessage(statusResult)
  const msgs = useAppStore.getState().messages
  expect(msgs[0].ab_test_result).toEqual(statusResult)
})

test("18. store does not attach to user message", () => {
  useAppStore.setState({
    messages: [
      { id: "m1", role: "user", content: "how is the A/B test?", timestamp: "" },
    ],
  })
  useAppStore.getState().attachABTestResultToLastMessage(statusResult)
  const msgs = useAppStore.getState().messages
  expect(msgs[0].ab_test_result).toBeUndefined()
})

test("19. store does not crash when messages list is empty", () => {
  useAppStore.setState({ messages: [] })
  expect(() =>
    useAppStore.getState().attachABTestResultToLastMessage(noneResult)
  ).not.toThrow()
})
