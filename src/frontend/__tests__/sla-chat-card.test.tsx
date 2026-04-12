/**
 * Tests for SlaCard (chat-inline latency card) and Zustand store action.
 *
 * Covers:
 *  1.  Renders region with aria-label "no data yet" when sample_count=0
 *  2.  Renders ⚡ icon (aria-hidden)
 *  3.  Shows "Prediction Latency" heading
 *  4.  Shows "Healthy" badge when alert=false
 *  5.  Shows "p95 > 500ms" alert badge when alert=true
 *  6.  Shows p50/p95/p99 values
 *  7.  Shows avg_ms value
 *  8.  Shows sample count text
 *  9.  Renders sparkline chart when latency_by_day has > 1 entry
 * 10.  Shows alert message paragraph with role="alert" when alert=true
 * 11.  Shows SLA target footnote text
 * 12.  Shows empty-state message when sample_count=0
 * 13.  Store: attachSlaMetricsToLastMessage attaches to last assistant message
 * 14.  Store: does not attach to user message
 * 15.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { SlaCard } from "@/components/deploy/sla-chat-card"
import type { SlaData } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Recharts mock
// ---------------------------------------------------------------------------
jest.mock("recharts", () => {
  const Original = jest.requireActual("recharts")
  return {
    ...Original,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container" style={{ width: 500, height: 200 }}>
        {children}
      </div>
    ),
  }
})

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const emptySla: SlaData = {
  deployment_id: "dep-1",
  sample_count: 0,
  p50_ms: null,
  p95_ms: null,
  p99_ms: null,
  avg_ms: null,
  alert: false,
  alert_message: null,
  latency_by_day: [],
}

const healthySla: SlaData = {
  deployment_id: "dep-1",
  sample_count: 42,
  p50_ms: 25.5,
  p95_ms: 120.3,
  p99_ms: 210.7,
  avg_ms: 45.2,
  alert: false,
  alert_message: null,
  latency_by_day: [
    { date: "2026-04-10", avg_ms: 40.0 },
    { date: "2026-04-11", avg_ms: 50.0 },
    { date: "2026-04-12", avg_ms: 45.2 },
  ],
}

const alertSla: SlaData = {
  deployment_id: "dep-1",
  sample_count: 20,
  p50_ms: 80.0,
  p95_ms: 620.5,
  p99_ms: 750.0,
  avg_ms: 100.0,
  alert: true,
  alert_message:
    "p95 latency is 620.5ms — above the 500ms target. Consider retraining with fewer features or switching to a simpler algorithm.",
  latency_by_day: [
    { date: "2026-04-11", avg_ms: 580.0 },
    { date: "2026-04-12", avg_ms: 620.5 },
  ],
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("SlaCard — empty state", () => {
  it("renders region with aria-label containing 'no data'", () => {
    render(<SlaCard sla={emptySla} />)
    expect(
      screen.getByRole("region", { name: /no data yet/i }),
    ).toBeInTheDocument()
  })

  it("shows empty state message", () => {
    render(<SlaCard sla={emptySla} />)
    expect(
      screen.getByText(/No timing data yet/i),
    ).toBeInTheDocument()
  })
})

describe("SlaCard — healthy", () => {
  it("renders region with aria-label 'healthy'", () => {
    render(<SlaCard sla={healthySla} />)
    expect(
      screen.getByRole("region", { name: /healthy/i }),
    ).toBeInTheDocument()
  })

  it("renders ⚡ icon with aria-hidden", () => {
    render(<SlaCard sla={healthySla} />)
    const icon = screen.getAllByText("⚡")[0]
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("shows Prediction Latency heading", () => {
    render(<SlaCard sla={healthySla} />)
    expect(screen.getByText("Prediction Latency")).toBeInTheDocument()
  })

  it("shows Healthy badge when alert=false", () => {
    render(<SlaCard sla={healthySla} />)
    expect(screen.getByText("Healthy")).toBeInTheDocument()
  })

  it("shows p50 value", () => {
    render(<SlaCard sla={healthySla} />)
    expect(screen.getByText("25.5ms")).toBeInTheDocument()
  })

  it("shows p95 value", () => {
    render(<SlaCard sla={healthySla} />)
    expect(screen.getByText("120.3ms")).toBeInTheDocument()
  })

  it("shows p99 value", () => {
    render(<SlaCard sla={healthySla} />)
    expect(screen.getByText("210.7ms")).toBeInTheDocument()
  })

  it("shows avg_ms value", () => {
    render(<SlaCard sla={healthySla} />)
    expect(screen.getByText("45.2ms")).toBeInTheDocument()
  })

  it("shows sample count", () => {
    render(<SlaCard sla={healthySla} />)
    expect(screen.getByText(/42 predictions/i)).toBeInTheDocument()
  })

  it("renders sparkline chart when latency_by_day has >1 entry", () => {
    render(<SlaCard sla={healthySla} />)
    expect(screen.getByTestId("responsive-container")).toBeInTheDocument()
  })

  it("shows SLA target footnote", () => {
    render(<SlaCard sla={healthySla} />)
    expect(screen.getByText(/SLA target: p95 ≤ 500ms/i)).toBeInTheDocument()
  })
})

describe("SlaCard — alert", () => {
  it("renders region with aria-label 'alert'", () => {
    render(<SlaCard sla={alertSla} />)
    expect(
      screen.getByRole("region", { name: /alert/i }),
    ).toBeInTheDocument()
  })

  it("shows p95 > 500ms badge", () => {
    render(<SlaCard sla={alertSla} />)
    const matches = screen.getAllByText(/p95.*500ms/i)
    expect(matches.length).toBeGreaterThan(0)
  })

  it("shows alert message with role=alert", () => {
    render(<SlaCard sla={alertSla} />)
    const alertEl = screen.getByRole("alert")
    expect(alertEl).toBeInTheDocument()
    expect(alertEl.textContent).toMatch(/above the 500ms target/i)
  })
})

// ---------------------------------------------------------------------------
// Store tests
// ---------------------------------------------------------------------------

describe("attachSlaMetricsToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches sla_metrics to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "show me latency" },
        { role: "assistant", content: "Here are the latency stats." },
      ],
    })
    useAppStore.getState().attachSlaMetricsToLastMessage(healthySla)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].sla_metrics).toEqual(healthySla)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "show latency" }],
    })
    useAppStore.getState().attachSlaMetricsToLastMessage(healthySla)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].sla_metrics).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachSlaMetricsToLastMessage(healthySla),
    ).not.toThrow()
  })
})
