/**
 * Tests for UsagePatternCard — prediction usage pattern analysis chat card.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { UsagePatternCard } from "../components/deploy/usage-pattern-card"
import type { UsagePatternResult } from "../lib/types"
import { useAppStore } from "../lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const EMPTY_RESULT: UsagePatternResult = {
  deployment_id: "dep-1",
  hour_counts: new Array(24).fill(0),
  day_counts: new Array(7).fill(0),
  peak_hour: null,
  peak_hour_count: 0,
  peak_day: null,
  peak_day_name: null,
  peak_day_short: null,
  quiet_hours: [],
  busiest_period: null,
  total_predictions: 0,
  day_names: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
  summary: "No predictions recorded yet — usage patterns will appear once the model starts receiving requests.",
}

function makeHourCounts(peakHour: number): number[] {
  const counts = new Array(24).fill(1)
  counts[peakHour] = 20
  return counts
}

function makeDayCounts(peakDay: number): number[] {
  const counts = new Array(7).fill(2)
  counts[peakDay] = 15
  return counts
}

const WITH_DATA: UsagePatternResult = {
  deployment_id: "dep-2",
  hour_counts: makeHourCounts(9),
  day_counts: makeDayCounts(1),
  peak_hour: 9,
  peak_hour_count: 20,
  peak_day: 1,
  peak_day_name: "Tuesday",
  peak_day_short: "Tue",
  quiet_hours: [0, 1, 2, 3, 4, 5],
  busiest_period: "morning (6am–12pm)",
  total_predictions: 47,
  day_names: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
  summary: "Peak usage is at 9am UTC (20 predictions) and on Tuesdays (15 predictions). Lowest usage: 12am, 1am, 2am UTC.",
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("UsagePatternCard — aria-label", () => {
  it("has figure with aria-label", () => {
    render(<UsagePatternCard result={EMPTY_RESULT} />)
    const fig = screen.getByRole("figure")
    expect(fig).toHaveAttribute("aria-label", "Prediction usage pattern analysis")
  })
})

describe("UsagePatternCard — icon and heading", () => {
  it("renders 🕐 icon", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    expect(screen.getByText("🕐")).toBeInTheDocument()
  })

  it("renders card heading", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    expect(screen.getByText("Prediction Usage Patterns")).toBeInTheDocument()
  })
})

describe("UsagePatternCard — empty state", () => {
  it("shows no-predictions message when empty", () => {
    render(<UsagePatternCard result={EMPTY_RESULT} />)
    expect(screen.getByText(/No predictions recorded yet/i)).toBeInTheDocument()
  })

  it("does not show peak hour badge when empty", () => {
    render(<UsagePatternCard result={EMPTY_RESULT} />)
    expect(screen.queryByText(/Peak:/)).not.toBeInTheDocument()
  })
})

describe("UsagePatternCard — badges", () => {
  it("shows peak hour badge", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    expect(screen.getByText(/Peak:.*UTC/)).toBeInTheDocument()
  })

  it("shows peak day badge", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    expect(screen.getByText(/Busiest:.*Tuesdays/)).toBeInTheDocument()
  })

  it("shows total predictions badge", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    expect(screen.getByText("47 predictions")).toBeInTheDocument()
  })
})

describe("UsagePatternCard — hour/day charts", () => {
  it("renders 24 hour bars", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    const container = screen.getByLabelText("Hour-of-day prediction distribution")
    expect(container.children).toHaveLength(24)
  })

  it("renders 7 day bars", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    const container = screen.getByLabelText("Day-of-week prediction distribution")
    expect(container.children).toHaveLength(7)
  })
})

describe("UsagePatternCard — sections", () => {
  it("shows busiest period", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    expect(screen.getByText(/morning \(6am/)).toBeInTheDocument()
  })

  it("shows maintenance window suggestion when quiet hours exist", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    expect(screen.getByText(/maintenance window/i)).toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    expect(screen.getByText(/Peak usage is at 9am UTC/)).toBeInTheDocument()
  })
})

describe("UsagePatternCard — accessibility", () => {
  it("has sr-only figcaption", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    const caption = screen.getByRole("figure").querySelector("figcaption.sr-only")
    expect(caption).toBeInTheDocument()
    expect(caption?.textContent).toContain("47 total predictions")
  })

  it("hour bar aria-labels include prediction count", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    const bar = screen.getByLabelText("9am: 20 predictions")
    expect(bar).toBeInTheDocument()
  })

  it("day bar aria-labels include prediction count", () => {
    render(<UsagePatternCard result={WITH_DATA} />)
    const bar = screen.getByLabelText("Tue: 15 predictions")
    expect(bar).toBeInTheDocument()
  })
})

describe("UsagePatternCard — store action", () => {
  it("attachUsagePatternToLastMessage attaches to last assistant message", () => {
    const store = useAppStore.getState()
    store.setMessages([
      { id: "1", role: "user", content: "hello" },
      { id: "2", role: "assistant", content: "hi" },
    ])
    store.attachUsagePatternToLastMessage(WITH_DATA)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].usage_pattern).toEqual(WITH_DATA)
  })
})
