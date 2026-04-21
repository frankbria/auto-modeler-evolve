import React from "react"
import { render, screen } from "@testing-library/react"
import { ConfidenceTrendCard } from "../components/deploy/confidence-trend-card"
import type { ConfidenceTrendResult } from "../lib/types"
import { useAppStore } from "../lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const NO_DATA: ConfidenceTrendResult = {
  deployment_id: "dep-1",
  has_data: false,
  daily_stats: [],
  overall_avg: null,
  trend_direction: "stable",
  trend_rate_per_day: null,
  peak_day: null,
  peak_value: null,
  low_day: null,
  low_value: null,
  sample_count: 0,
  summary: "",
}

const IMPROVING: ConfidenceTrendResult = {
  deployment_id: "dep-2",
  has_data: true,
  daily_stats: [
    { date: "2026-04-18", avg_confidence: 65.0, count: 3 },
    { date: "2026-04-19", avg_confidence: 75.0, count: 4 },
    { date: "2026-04-20", avg_confidence: 85.0, count: 5 },
  ],
  overall_avg: 75.0,
  trend_direction: "improving",
  trend_rate_per_day: 10.0,
  peak_day: "2026-04-20",
  peak_value: 85.0,
  low_day: "2026-04-18",
  low_value: 65.0,
  sample_count: 12,
  summary: "Confidence is improving at +10.0%/day over 3 days.",
}

const DECLINING: ConfidenceTrendResult = {
  deployment_id: "dep-3",
  has_data: true,
  daily_stats: [
    { date: "2026-04-18", avg_confidence: 90.0, count: 5 },
    { date: "2026-04-19", avg_confidence: 80.0, count: 4 },
    { date: "2026-04-20", avg_confidence: 70.0, count: 3 },
  ],
  overall_avg: 80.0,
  trend_direction: "declining",
  trend_rate_per_day: -10.0,
  peak_day: "2026-04-18",
  peak_value: 90.0,
  low_day: "2026-04-20",
  low_value: 70.0,
  sample_count: 12,
  summary: "Confidence is declining at -10.0%/day over 3 days.",
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ConfidenceTrendCard — empty state", () => {
  it("shows no-data message", () => {
    render(<ConfidenceTrendCard result={NO_DATA} />)
    expect(screen.getByText(/No confidence data yet/i)).toBeInTheDocument()
  })

  it("has region role with no-data label", () => {
    render(<ConfidenceTrendCard result={NO_DATA} />)
    expect(screen.getByRole("region", { name: /no data yet/i })).toBeInTheDocument()
  })
})

describe("ConfidenceTrendCard — heading and icon", () => {
  it("renders heading", () => {
    render(<ConfidenceTrendCard result={IMPROVING} />)
    expect(screen.getByText("Confidence Trend")).toBeInTheDocument()
  })

  it("renders icon", () => {
    render(<ConfidenceTrendCard result={IMPROVING} />)
    expect(screen.getByText("📉")).toBeInTheDocument()
  })
})

describe("ConfidenceTrendCard — direction badge", () => {
  it("shows improving badge", () => {
    render(<ConfidenceTrendCard result={IMPROVING} />)
    expect(screen.getByText("↑ Improving")).toBeInTheDocument()
  })

  it("shows declining badge", () => {
    render(<ConfidenceTrendCard result={DECLINING} />)
    expect(screen.getByText("↓ Declining")).toBeInTheDocument()
  })
})

describe("ConfidenceTrendCard — stats grid", () => {
  it("shows overall avg", () => {
    render(<ConfidenceTrendCard result={IMPROVING} />)
    expect(screen.getByText("75.0%")).toBeInTheDocument()
  })

  it("shows peak value", () => {
    render(<ConfidenceTrendCard result={IMPROVING} />)
    expect(screen.getByText("85.0%")).toBeInTheDocument()
  })

  it("shows low value", () => {
    render(<ConfidenceTrendCard result={IMPROVING} />)
    expect(screen.getByText("65.0%")).toBeInTheDocument()
  })

  it("shows rate label", () => {
    render(<ConfidenceTrendCard result={IMPROVING} />)
    expect(screen.getByText("+10.00%/day")).toBeInTheDocument()
  })

  it("shows sample count", () => {
    render(<ConfidenceTrendCard result={IMPROVING} />)
    expect(screen.getByText("12 predictions")).toBeInTheDocument()
  })
})

describe("ConfidenceTrendCard — chart", () => {
  it("renders figure with aria-label for multi-day data", () => {
    render(<ConfidenceTrendCard result={IMPROVING} />)
    expect(
      screen.getByLabelText("Daily confidence trend over 3 days")
    ).toBeInTheDocument()
  })

  it("does not render chart for single-day data", () => {
    const single: ConfidenceTrendResult = {
      ...IMPROVING,
      daily_stats: [{ date: "2026-04-20", avg_confidence: 80.0, count: 5 }],
    }
    render(<ConfidenceTrendCard result={single} />)
    expect(
      screen.queryByLabelText(/Daily confidence trend/)
    ).not.toBeInTheDocument()
  })
})

describe("ConfidenceTrendCard — summary", () => {
  it("renders summary text", () => {
    render(<ConfidenceTrendCard result={IMPROVING} />)
    expect(screen.getByText(/Confidence is improving/i)).toBeInTheDocument()
  })
})

describe("ConfidenceTrendCard — store action", () => {
  it("attachConfidenceTrendToLastMessage attaches to last assistant message", () => {
    const store = useAppStore.getState()
    store.setMessages([
      { id: "1", role: "user", content: "hello" },
      { id: "2", role: "assistant", content: "hi" },
    ])
    store.attachConfidenceTrendToLastMessage(IMPROVING)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].confidence_trend).toEqual(IMPROVING)
  })
})
