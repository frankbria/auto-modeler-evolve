import React from "react"
import { act } from "react"
import { render, screen, renderHook } from "@testing-library/react"
import { useAppStore } from "@/lib/store"
import { MonitoringDigestCard } from "@/components/deploy/monitoring-digest-card"
import type { MonitoringDigestResult, MonitoringDigestSignal } from "@/lib/types"

function makeSignal(overrides: Partial<MonitoringDigestSignal> = {}): MonitoringDigestSignal {
  return {
    signal_key: "output_anomalies",
    signal_label: "Output Anomalies",
    verdict: "no_anomalies",
    verdict_label: "No anomalies",
    severity: "green",
    finding: "All production predictions are within normal range.",
    icon: "🔍",
    is_available: true,
    ...overrides,
  }
}

function makeResult(overrides: Partial<MonitoringDigestResult> = {}): MonitoringDigestResult {
  return {
    deployment_id: "dep-1",
    algorithm: "random_forest",
    target_col: "revenue",
    problem_type: "regression",
    signals: [
      makeSignal(),
      makeSignal({
        signal_key: "usage_activity",
        signal_label: "Usage Activity",
        verdict: "active",
        verdict_label: "Active",
        icon: "📡",
        finding: "50 predictions in the last 7 days.",
      }),
    ],
    overall_health: "healthy",
    overall_score: 100,
    signals_total: 2,
    signals_firing: 0,
    priority_actions: [],
    summary: "All 2 monitoring signals are healthy — no action needed.",
    ...overrides,
  }
}

describe("MonitoringDigestCard", () => {
  it("renders the card", () => {
    render(<MonitoringDigestCard result={makeResult()} />)
    expect(screen.getByTestId("monitoring-digest-card")).toBeInTheDocument()
  })

  it("shows overall health badge", () => {
    render(<MonitoringDigestCard result={makeResult({ overall_health: "healthy" })} />)
    expect(screen.getByTestId("overall-health-badge")).toHaveTextContent("Healthy")
  })

  it("shows warning health badge for warning state", () => {
    render(<MonitoringDigestCard result={makeResult({ overall_health: "warning" })} />)
    expect(screen.getByTestId("overall-health-badge")).toHaveTextContent("Warning")
  })

  it("shows critical health badge for critical state", () => {
    render(<MonitoringDigestCard result={makeResult({ overall_health: "critical" })} />)
    expect(screen.getByTestId("overall-health-badge")).toHaveTextContent("Critical")
  })

  it("shows watching health badge for watching state", () => {
    render(<MonitoringDigestCard result={makeResult({ overall_health: "watching" })} />)
    expect(screen.getByTestId("overall-health-badge")).toHaveTextContent("Watching")
  })

  it("renders signal rows", () => {
    render(<MonitoringDigestCard result={makeResult()} />)
    expect(screen.getByTestId("signal-row-output_anomalies")).toBeInTheDocument()
    expect(screen.getByTestId("signal-row-usage_activity")).toBeInTheDocument()
  })

  it("shows signal finding text", () => {
    render(<MonitoringDigestCard result={makeResult()} />)
    expect(screen.getByText("All production predictions are within normal range.")).toBeInTheDocument()
  })

  it("shows all signals healthy badge when signals_firing is 0", () => {
    render(<MonitoringDigestCard result={makeResult({ signals_firing: 0 })} />)
    expect(screen.getByText("All signals healthy")).toBeInTheDocument()
  })

  it("shows signals need attention badge when signals_firing > 0", () => {
    render(
      <MonitoringDigestCard
        result={makeResult({
          signals_firing: 2,
          overall_health: "warning",
        })}
      />
    )
    expect(screen.getByText(/2 signals need attention/)).toBeInTheDocument()
  })

  it("renders priority actions when present", () => {
    render(
      <MonitoringDigestCard
        result={makeResult({
          priority_actions: ["Retrain your model immediately.", "Check for distribution shift."],
          signals_firing: 1,
          overall_health: "warning",
        })}
      />
    )
    expect(screen.getByTestId("priority-actions")).toBeInTheDocument()
    expect(screen.getByText("Retrain your model immediately.")).toBeInTheDocument()
  })

  it("does not render priority actions section when list is empty", () => {
    render(<MonitoringDigestCard result={makeResult({ priority_actions: [] })} />)
    expect(screen.queryByTestId("priority-actions")).not.toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<MonitoringDigestCard result={makeResult()} />)
    expect(screen.getByTestId("digest-summary")).toHaveTextContent(
      "All 2 monitoring signals are healthy"
    )
  })

  it("shows target column", () => {
    render(<MonitoringDigestCard result={makeResult({ target_col: "revenue" })} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("renders sr-only figcaption", () => {
    render(<MonitoringDigestCard result={makeResult()} />)
    const caption = document.querySelector("figcaption.sr-only")
    expect(caption).not.toBeNull()
    expect(caption!.textContent).toContain("monitoring signal digest")
  })

  it("renders signal list with role=list", () => {
    render(<MonitoringDigestCard result={makeResult()} />)
    const list = screen.getByRole("list", { name: /monitoring signals/i })
    expect(list).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Zustand store action
// ---------------------------------------------------------------------------

describe("attachMonitoringDigestToLastMessage", () => {
  it("attaches monitoring digest to last assistant message", () => {
    const { result } = renderHook(() => useAppStore())
    act(() => {
      result.current.setMessages([
        { id: "1", role: "user", content: "test", timestamp: "2025-01-01T00:00:00Z" },
        { id: "2", role: "assistant", content: "Here is the digest.", timestamp: "2025-01-01T00:00:01Z" },
      ])
    })
    act(() => {
      result.current.attachMonitoringDigestToLastMessage(makeResult())
    })
    const msgs = result.current.messages
    const last = msgs[msgs.length - 1]
    expect(last.monitoring_digest).toBeDefined()
    expect(last.monitoring_digest?.overall_health).toBe("healthy")
  })

  it("does not attach to non-assistant message", () => {
    const { result } = renderHook(() => useAppStore())
    act(() => {
      result.current.setMessages([
        { id: "1", role: "user", content: "test", timestamp: "2025-01-01T00:00:00Z" },
      ])
    })
    act(() => {
      result.current.attachMonitoringDigestToLastMessage(makeResult())
    })
    const last = result.current.messages[result.current.messages.length - 1]
    expect(last.monitoring_digest).toBeUndefined()
  })
})
