/**
 * Tests for ApiUptimeSummaryCard and attachUptimeSummaryToLastMessage.
 *
 * Covers:
 *  1.  Renders container with data-testid "api-uptime-summary-card"
 *  2.  healthy verdict shows emerald border class
 *  3.  degraded verdict shows amber border class
 *  4.  mostly_idle verdict shows slate border class
 *  5.  idle / no_data show muted border
 *  6.  verdict badge is rendered with correct label
 *  7.  n_days badge shows window size
 *  8.  total_predictions badge is rendered
 *  9.  daily grid is rendered with correct number of cells
 * 10.  active days show emerald cells
 * 11.  degraded days show amber cells
 * 12.  silent days show slate cells
 * 13.  latency stats block renders avg_latency_ms
 * 14.  latency stats block renders overall_p95_latency_ms
 * 15.  peak latency stat shows peak_latency_ms
 * 16.  summary paragraph renders result.summary
 * 17.  note text renders result.note
 * 18.  sr-only figcaption is present for accessibility
 * 19.  Store: attachUptimeSummaryToLastMessage attaches to last assistant message
 * 20.  Store: does not attach to user message
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ApiUptimeSummaryCard } from "@/components/deploy/api-uptime-summary-card"
import type { ApiUptimeSummaryResult, ApiUptimeDayStat } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeDayStats(n: number, status: "active" | "silent" | "degraded" = "active"): ApiUptimeDayStat[] {
  return Array.from({ length: n }, (_, i) => ({
    date: `2026-06-${String(i + 1).padStart(2, "0")}`,
    n_predictions: status === "silent" ? 0 : 5,
    avg_latency_ms: status === "silent" ? null : 100,
    p95_latency_ms: status === "silent" ? null : (status === "degraded" ? 3000 : 150),
    status,
  }))
}

function makeResult(overrides: Partial<ApiUptimeSummaryResult> = {}): ApiUptimeSummaryResult {
  return {
    deployment_id: "dep-1",
    n_days: 7,
    total_predictions: 35,
    n_active_days: 5,
    n_silent_days: 2,
    daily_stats: makeDayStats(7, "active"),
    overall_avg_latency_ms: 105.0,
    overall_p95_latency_ms: 145.0,
    peak_latency_ms: 300.0,
    peak_latency_date: "2026-06-03",
    degraded_days: 0,
    verdict: "healthy",
    note: "Only successful predictions are logged — silent days may indicate low traffic, not necessarily downtime.",
    summary: "Your API looks healthy: active on 5 of the last 7 days with 35 total predictions. Overall p95 latency: 145 ms.",
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

test("renders container with correct testid", () => {
  render(<ApiUptimeSummaryCard result={makeResult()} />)
  expect(screen.getByTestId("api-uptime-summary-card")).toBeInTheDocument()
})

test("healthy verdict renders emerald border", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ verdict: "healthy" })} />)
  const card = screen.getByTestId("api-uptime-summary-card")
  expect(card.className).toContain("border-emerald-500")
})

test("degraded verdict renders amber border", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ verdict: "degraded", degraded_days: 2 })} />)
  const card = screen.getByTestId("api-uptime-summary-card")
  expect(card.className).toContain("border-amber-500")
})

test("mostly_idle verdict renders slate border", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ verdict: "mostly_idle" })} />)
  const card = screen.getByTestId("api-uptime-summary-card")
  expect(card.className).toContain("border-slate-400")
})

test("idle verdict renders muted border", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ verdict: "idle", n_active_days: 0 })} />)
  const card = screen.getByTestId("api-uptime-summary-card")
  expect(card.className).toContain("border-muted")
})

test("no_data verdict renders muted border", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ verdict: "no_data", total_predictions: 0, n_active_days: 0 })} />)
  const card = screen.getByTestId("api-uptime-summary-card")
  expect(card.className).toContain("border-muted")
})

// ---------------------------------------------------------------------------
// Badges
// ---------------------------------------------------------------------------

test("verdict badge shows Healthy for healthy verdict", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ verdict: "healthy" })} />)
  expect(screen.getByTestId("uptime-verdict-badge")).toHaveTextContent("Healthy")
})

test("verdict badge shows Degraded for degraded verdict", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ verdict: "degraded", degraded_days: 1 })} />)
  expect(screen.getByTestId("uptime-verdict-badge")).toHaveTextContent("Degraded")
})

test("n_days badge shows window size", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ n_days: 7 })} />)
  expect(screen.getByTestId("uptime-n-days-badge")).toHaveTextContent("7")
})

test("total_predictions badge is rendered", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ total_predictions: 35 })} />)
  expect(screen.getByTestId("uptime-total-badge")).toHaveTextContent("35")
})

// ---------------------------------------------------------------------------
// Daily grid
// ---------------------------------------------------------------------------

test("daily grid is rendered", () => {
  render(<ApiUptimeSummaryCard result={makeResult()} />)
  expect(screen.getByTestId("uptime-daily-grid")).toBeInTheDocument()
})

test("active day cells have emerald color", () => {
  const stats = makeDayStats(7, "active")
  render(<ApiUptimeSummaryCard result={makeResult({ daily_stats: stats, n_active_days: 7, n_silent_days: 0 })} />)
  const grid = screen.getByTestId("uptime-daily-grid")
  expect(grid.innerHTML).toContain("bg-emerald-500")
})

test("degraded day cells have amber color", () => {
  const stats = makeDayStats(7, "degraded")
  render(<ApiUptimeSummaryCard result={makeResult({ daily_stats: stats, degraded_days: 7 })} />)
  const grid = screen.getByTestId("uptime-daily-grid")
  expect(grid.innerHTML).toContain("bg-amber-400")
})

test("silent day cells have slate color", () => {
  const stats = makeDayStats(7, "silent")
  render(<ApiUptimeSummaryCard result={makeResult({ daily_stats: stats, n_active_days: 0, n_silent_days: 7, total_predictions: 0, verdict: "idle", overall_avg_latency_ms: null, overall_p95_latency_ms: null, peak_latency_ms: null, peak_latency_date: null })} />)
  const grid = screen.getByTestId("uptime-daily-grid")
  expect(grid.innerHTML).toContain("bg-slate-200")
})

// ---------------------------------------------------------------------------
// Activity row
// ---------------------------------------------------------------------------

test("activity row shows active days count", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ n_active_days: 5, n_silent_days: 2 })} />)
  const row = screen.getByTestId("uptime-activity-row")
  expect(row.textContent).toContain("5")
})

test("degraded days shown when present", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ degraded_days: 2, verdict: "degraded" })} />)
  const row = screen.getByTestId("uptime-activity-row")
  expect(row.textContent).toContain("2")
})

// ---------------------------------------------------------------------------
// Latency stats
// ---------------------------------------------------------------------------

test("latency stats block renders avg_latency_ms", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ overall_avg_latency_ms: 105.0 })} />)
  const stats = screen.getByTestId("uptime-latency-stats")
  expect(stats.textContent).toContain("105")
})

test("latency stats block renders overall_p95_latency_ms", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ overall_p95_latency_ms: 145.0 })} />)
  const stats = screen.getByTestId("uptime-latency-stats")
  expect(stats.textContent).toContain("145")
})

test("peak latency stat shows peak_latency_ms", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ peak_latency_ms: 300.0, peak_latency_date: "2026-06-03" })} />)
  const stats = screen.getByTestId("uptime-latency-stats")
  expect(stats.textContent).toContain("300")
})

test("latency stats block absent when no latency data", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ overall_avg_latency_ms: null, overall_p95_latency_ms: null, peak_latency_ms: null })} />)
  expect(screen.queryByTestId("uptime-latency-stats")).not.toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Summary and note
// ---------------------------------------------------------------------------

test("summary paragraph renders result.summary", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ summary: "Test summary text for API health." })} />)
  expect(screen.getByTestId("uptime-summary")).toHaveTextContent("Test summary text for API health.")
})

test("note renders result.note", () => {
  render(<ApiUptimeSummaryCard result={makeResult({ note: "Only successful predictions are logged." })} />)
  expect(screen.getByTestId("uptime-note")).toHaveTextContent("Only successful predictions are logged.")
})

test("sr-only figcaption is present", () => {
  render(<ApiUptimeSummaryCard result={makeResult()} />)
  const fig = screen.getByTestId("api-uptime-summary-card")
  const caption = fig.querySelector("figcaption.sr-only")
  expect(caption).not.toBeNull()
})

// ---------------------------------------------------------------------------
// Store action
// ---------------------------------------------------------------------------

test("attachUptimeSummaryToLastMessage attaches to last assistant message", () => {
  const { attachUptimeSummaryToLastMessage } = useAppStore.getState()
  useAppStore.setState({
    messages: [
      { role: "user", content: "was my endpoint down?" },
      { role: "assistant", content: "Let me check." },
    ],
  })
  const result = makeResult()
  attachUptimeSummaryToLastMessage(result)
  const messages = useAppStore.getState().messages
  expect(messages[messages.length - 1].uptime_summary).toEqual(result)
})

test("attachUptimeSummaryToLastMessage does not attach to user message", () => {
  const { attachUptimeSummaryToLastMessage } = useAppStore.getState()
  useAppStore.setState({
    messages: [
      { role: "user", content: "was my endpoint down?" },
    ],
  })
  attachUptimeSummaryToLastMessage(makeResult())
  const messages = useAppStore.getState().messages
  expect(messages[0].uptime_summary).toBeUndefined()
})
