/**
 * Tests for CanaryCard component and Zustand store action.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { CanaryCard } from "@/components/deploy/canary-card"
import type { CanaryStatusResult, CanaryComparison } from "@/lib/types"

const baseResult: CanaryStatusResult = {
  deployment_id: "dep-1",
  canary_is_active: false,
  canary_run_id: null,
  canary_traffic_pct: 0,
  canary_version_number: null,
  canary_algorithm: null,
  canary_started_at: null,
  current_version_number: 2,
  current_algorithm: "random_forest",
  available_versions: [
    { version_number: 2, algorithm: "random_forest", is_current: true },
    { version_number: 1, algorithm: "linear_regression", is_current: false },
  ],
  comparison: null,
  summary: "No active canary. Deployment has 2 version(s).",
}

const activeResult: CanaryStatusResult = {
  ...baseResult,
  canary_is_active: true,
  canary_run_id: "run-1",
  canary_traffic_pct: 10,
  canary_version_number: 1,
  canary_algorithm: "linear_regression",
  canary_started_at: "2026-06-11T00:00:00",
  summary: "Canary active: 10% of traffic routes to v1 (linear_regression).",
}

const comparison: CanaryComparison = {
  canary_run_id: "run-1",
  current_run_id: "run-2",
  traffic_pct: 10,
  control_count: 20,
  canary_count: 15,
  metric_label: "Avg prediction",
  control_avg: 100.0,
  canary_avg: 125.0,
  delta: 25.0,
  delta_pct: 25.0,
  direction: "canary_higher",
  verdict: "canary_different",
  min_samples: 10,
  has_enough_data: true,
  summary: "Canary model shows higher avg prediction: 125.000 vs control 100.000 (+25.0%).",
}

describe("CanaryCard — inactive state", () => {
  it("renders heading", () => {
    render(<CanaryCard result={baseResult} />)
    expect(screen.getByText("Canary Deployment")).toBeInTheDocument()
  })

  it("shows Inactive badge", () => {
    render(<CanaryCard result={baseResult} />)
    expect(screen.getByText("Inactive")).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<CanaryCard result={baseResult} />)
    expect(screen.getByText(/No active canary/)).toBeInTheDocument()
  })

  it("shows available versions when inactive", () => {
    render(<CanaryCard result={baseResult} />)
    expect(screen.getByText(/Available versions/)).toBeInTheDocument()
  })

  it("shows start canary hint", () => {
    render(<CanaryCard result={baseResult} />)
    expect(screen.getByText(/start a canary with v1 at 10% traffic/i)).toBeInTheDocument()
  })
})

describe("CanaryCard — active canary without comparison", () => {
  it("shows Active badge", () => {
    render(<CanaryCard result={activeResult} />)
    expect(screen.getByText("Active")).toBeInTheDocument()
  })

  it("shows traffic percentage badge", () => {
    render(<CanaryCard result={activeResult} />)
    expect(screen.getByText("10% canary traffic")).toBeInTheDocument()
  })

  it("shows canary version in grid", () => {
    render(<CanaryCard result={activeResult} />)
    expect(screen.getByText("Canary (testing)")).toBeInTheDocument()
  })

  it("shows control version in grid", () => {
    render(<CanaryCard result={activeResult} />)
    expect(screen.getByText("Control (current)")).toBeInTheDocument()
  })

  it("shows canary summary text", () => {
    render(<CanaryCard result={activeResult} />)
    expect(screen.getByText(/Canary active/)).toBeInTheDocument()
  })

  it("shows promote/cancel commands footer", () => {
    render(<CanaryCard result={activeResult} />)
    expect(screen.getByText(/Commands:/)).toBeInTheDocument()
  })
})

describe("CanaryCard — active with comparison data", () => {
  const resultWithComparison: CanaryStatusResult = {
    ...activeResult,
    comparison,
  }

  it("renders Live Comparison section", () => {
    render(<CanaryCard result={resultWithComparison} />)
    expect(screen.getByText("Live Comparison")).toBeInTheDocument()
  })

  it("shows control count", () => {
    render(<CanaryCard result={resultWithComparison} />)
    expect(screen.getByText("20 preds")).toBeInTheDocument()
  })

  it("shows canary count", () => {
    render(<CanaryCard result={resultWithComparison} />)
    expect(screen.getByText("15 preds")).toBeInTheDocument()
  })

  it("shows positive delta badge", () => {
    render(<CanaryCard result={resultWithComparison} />)
    expect(screen.getByText("+25.0%")).toBeInTheDocument()
  })

  it("shows comparison summary", () => {
    render(<CanaryCard result={resultWithComparison} />)
    expect(screen.getByText(/higher avg prediction/)).toBeInTheDocument()
  })
})

describe("CanaryCard — insufficient data comparison", () => {
  const insufficientComparison: CanaryComparison = {
    ...comparison,
    has_enough_data: false,
    verdict: "insufficient_data",
    control_count: 3,
    canary_count: 2,
    summary: "Not enough data yet to compare canary vs control.",
  }
  const resultInsufficient: CanaryStatusResult = {
    ...activeResult,
    comparison: insufficientComparison,
  }

  it("shows insufficient data message", () => {
    render(<CanaryCard result={resultInsufficient} />)
    expect(screen.getByText(/Not enough data yet/)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action
// ---------------------------------------------------------------------------

import { useAppStore } from "@/lib/store"

describe("attachCanaryStatusToLastMessage store action", () => {
  it("attaches canary_status to last assistant message", () => {
    const store = useAppStore.getState()

    store.setMessages([
      { id: "1", role: "user", content: "canary status", timestamp: new Date() },
      { id: "2", role: "assistant", content: "Let me check...", timestamp: new Date() },
    ])

    store.attachCanaryStatusToLastMessage(baseResult)

    const messages = useAppStore.getState().messages
    expect(messages[1].canary_status).toEqual(baseResult)
  })

  it("does not modify if last message is user", () => {
    const store = useAppStore.getState()

    store.setMessages([
      { id: "1", role: "user", content: "hello", timestamp: new Date() },
    ])
    store.attachCanaryStatusToLastMessage(baseResult)

    const messages = useAppStore.getState().messages
    expect(messages[0].canary_status).toBeUndefined()
  })
})
