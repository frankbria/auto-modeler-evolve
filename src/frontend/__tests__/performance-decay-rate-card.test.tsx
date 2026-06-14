import React from "react"
import { render, screen } from "@testing-library/react"
import { PerformanceDecayRateCard } from "@/components/deploy/performance-decay-rate-card"
import { PerformanceDecayResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const weeklyData = [
  { week: "2025-W01", accuracy_pct: 90.0, n_samples: 10 },
  { week: "2025-W02", accuracy_pct: 88.0, n_samples: 10 },
  { week: "2025-W03", accuracy_pct: 86.0, n_samples: 10 },
  { week: "2025-W04", accuracy_pct: 84.0, n_samples: 10 },
  { week: "2025-W05", accuracy_pct: 82.0, n_samples: 10 },
]

const degradingSlowlyResult: PerformanceDecayResult = {
  deployment_id: "dep-1",
  weekly_data: weeklyData,
  slope_pct_per_week: -2.0,
  weeks_until_threshold: 1.0,
  current_accuracy_pct: 82.0,
  threshold_pct: 80.0,
  verdict: "degrading_slowly",
  summary:
    "Accuracy is slowly declining — 2.0% per week (currently 82.0%). At this rate it will fall below 80% in approximately 1 week.",
}

const stableResult: PerformanceDecayResult = {
  deployment_id: "dep-2",
  weekly_data: weeklyData,
  slope_pct_per_week: 0.1,
  weeks_until_threshold: null,
  current_accuracy_pct: 82.0,
  threshold_pct: 80.0,
  verdict: "stable",
  summary: "Accuracy is stable at approximately 82.0% (slope: +0.10% per week). No retraining needed right now.",
}

const degradingFastResult: PerformanceDecayResult = {
  deployment_id: "dep-3",
  weekly_data: weeklyData,
  slope_pct_per_week: -5.0,
  weeks_until_threshold: 0.4,
  current_accuracy_pct: 82.0,
  threshold_pct: 80.0,
  verdict: "degrading_fast",
  summary:
    "Accuracy is dropping quickly — losing 5.0% per week (currently 82.0%). At this rate it will fall below 80% in approximately 0 weeks.",
}

const improvingResult: PerformanceDecayResult = {
  deployment_id: "dep-4",
  weekly_data: weeklyData,
  slope_pct_per_week: 2.0,
  weeks_until_threshold: null,
  current_accuracy_pct: 82.0,
  threshold_pct: 80.0,
  verdict: "improving",
  summary: "Accuracy is trending upward — gaining 2.0% per week (currently 82.0%). Your model is getting better over time.",
}

const belowThresholdResult: PerformanceDecayResult = {
  deployment_id: "dep-5",
  weekly_data: weeklyData,
  slope_pct_per_week: -1.0,
  weeks_until_threshold: 0.0,
  current_accuracy_pct: 70.0,
  threshold_pct: 80.0,
  verdict: "below_threshold",
  summary:
    "Accuracy is already below the 80% threshold (currently 70.0%). Retraining is strongly recommended.",
}

const noDataResult: PerformanceDecayResult = {
  deployment_id: "dep-6",
  weekly_data: [],
  slope_pct_per_week: null,
  weeks_until_threshold: null,
  current_accuracy_pct: null,
  threshold_pct: 80.0,
  verdict: "no_data",
  summary: "Not enough feedback to compute a decay rate — need at least 5 labeled records.",
}

// ---------------------------------------------------------------------------
// Card renders correctly for degrading_slowly
// ---------------------------------------------------------------------------

describe("PerformanceDecayRateCard — degrading_slowly", () => {
  it("renders heading", () => {
    render(<PerformanceDecayRateCard result={degradingSlowlyResult} />)
    expect(screen.getByText("Performance Decay Rate")).toBeInTheDocument()
  })

  it("renders verdict badge", () => {
    render(<PerformanceDecayRateCard result={degradingSlowlyResult} />)
    expect(screen.getByText("Degrading slowly")).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<PerformanceDecayRateCard result={degradingSlowlyResult} />)
    expect(screen.getByText(/slowly declining/i)).toBeInTheDocument()
  })

  it("renders current accuracy", () => {
    render(<PerformanceDecayRateCard result={degradingSlowlyResult} />)
    expect(screen.getByText("82.0%")).toBeInTheDocument()
  })

  it("renders slope", () => {
    render(<PerformanceDecayRateCard result={degradingSlowlyResult} />)
    expect(screen.getByText("-2.00% / week")).toBeInTheDocument()
  })

  it("renders weeks until threshold", () => {
    render(<PerformanceDecayRateCard result={degradingSlowlyResult} />)
    expect(screen.getByText("~1")).toBeInTheDocument()
  })

  it("renders week count badge", () => {
    render(<PerformanceDecayRateCard result={degradingSlowlyResult} />)
    expect(screen.getByText("5 weeks")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// stable
// ---------------------------------------------------------------------------

describe("PerformanceDecayRateCard — stable", () => {
  it("renders stable badge", () => {
    render(<PerformanceDecayRateCard result={stableResult} />)
    expect(screen.getByText("Stable")).toBeInTheDocument()
  })

  it("renders N/A for weeks until threshold when null", () => {
    render(<PerformanceDecayRateCard result={stableResult} />)
    expect(screen.getByText("N/A")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// degrading_fast — alert callout
// ---------------------------------------------------------------------------

describe("PerformanceDecayRateCard — degrading_fast", () => {
  it("renders Degrading fast badge", () => {
    render(<PerformanceDecayRateCard result={degradingFastResult} />)
    expect(screen.getByText("Degrading fast")).toBeInTheDocument()
  })

  it("renders alert callout with weeks estimate", () => {
    render(<PerformanceDecayRateCard result={degradingFastResult} />)
    const alert = screen.getByRole("alert")
    expect(alert).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// improving
// ---------------------------------------------------------------------------

describe("PerformanceDecayRateCard — improving", () => {
  it("renders Improving badge", () => {
    render(<PerformanceDecayRateCard result={improvingResult} />)
    expect(screen.getByText("Improving")).toBeInTheDocument()
  })

  it("renders positive slope", () => {
    render(<PerformanceDecayRateCard result={improvingResult} />)
    expect(screen.getByText("+2.00% / week")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// below_threshold — alert callout
// ---------------------------------------------------------------------------

describe("PerformanceDecayRateCard — below_threshold", () => {
  it("renders Below threshold badge", () => {
    render(<PerformanceDecayRateCard result={belowThresholdResult} />)
    expect(screen.getByText("Below threshold")).toBeInTheDocument()
  })

  it("renders alert with 'already below' message", () => {
    render(<PerformanceDecayRateCard result={belowThresholdResult} />)
    const alert = screen.getByRole("alert")
    expect(alert.textContent).toMatch(/already below/i)
  })

  it("renders 'Now' for weeks until threshold 0", () => {
    render(<PerformanceDecayRateCard result={belowThresholdResult} />)
    expect(screen.getByText("Now")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// no_data
// ---------------------------------------------------------------------------

describe("PerformanceDecayRateCard — no_data", () => {
  it("renders no_data summary", () => {
    render(<PerformanceDecayRateCard result={noDataResult} />)
    expect(screen.getByText(/not enough feedback/i)).toBeInTheDocument()
  })

  it("does not render stats grid when no weekly_data", () => {
    render(<PerformanceDecayRateCard result={noDataResult} />)
    expect(screen.queryByText("Current accuracy")).not.toBeInTheDocument()
  })

  it("renders No data badge", () => {
    render(<PerformanceDecayRateCard result={noDataResult} />)
    expect(screen.getByText("No data")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action
// ---------------------------------------------------------------------------

describe("attachPerformanceDecayRateToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches to the last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "hi", timestamp: new Date() },
        { id: "2", role: "assistant", content: "hello", timestamp: new Date() },
      ],
    })
    useAppStore.getState().attachPerformanceDecayRateToLastMessage(stableResult)
    const messages = useAppStore.getState().messages
    expect(messages[1].performance_decay_rate).toEqual(stableResult)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [{ id: "1", role: "user", content: "hi", timestamp: new Date() }],
    })
    useAppStore.getState().attachPerformanceDecayRateToLastMessage(stableResult)
    const messages = useAppStore.getState().messages
    expect(messages[0].performance_decay_rate).toBeUndefined()
  })

  it("does nothing when messages is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachPerformanceDecayRateToLastMessage(stableResult),
    ).not.toThrow()
  })
})
