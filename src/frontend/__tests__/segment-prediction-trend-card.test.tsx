import React from "react"
import { render, screen } from "@testing-library/react"

import { SegmentPredictionTrendCard } from "@/components/deploy/segment-prediction-trend-card"
import type { SegmentPredTrendResult } from "@/lib/types"

const DAILY_STATS_EAST = [
  { date: "2026-05-20", mean: 100.0, count: 3 },
  { date: "2026-05-25", mean: 120.0, count: 4 },
  { date: "2026-05-30", mean: 140.0, count: 3 },
]

const DAILY_STATS_WEST = [
  { date: "2026-05-20", mean: 200.0, count: 2 },
  { date: "2026-05-25", mean: 180.0, count: 3 },
  { date: "2026-05-30", mean: 160.0, count: 2 },
]

const divergingResult: SegmentPredTrendResult = {
  deployment_id: "dep-1",
  segment_column: "region",
  problem_type: "regression",
  n_segments: 2,
  n_samples: 17,
  segments: [
    {
      name: "East",
      n_samples: 10,
      n_days_with_data: 3,
      direction: "trending_up",
      direction_label: "Trending Up",
      change_pct: 40.0,
      first_mean: 100.0,
      last_mean: 140.0,
      daily_stats: DAILY_STATS_EAST,
    },
    {
      name: "West",
      n_samples: 7,
      n_days_with_data: 3,
      direction: "trending_down",
      direction_label: "Trending Down",
      change_pct: -20.0,
      first_mean: 200.0,
      last_mean: 160.0,
      daily_stats: DAILY_STATS_WEST,
    },
  ],
  most_improved_segment: "East",
  most_declining_segment: "West",
  verdict: "diverging",
  summary: "Predictions are diverging across segments.",
  n_days: 30,
}

const noDataResult: SegmentPredTrendResult = {
  deployment_id: "dep-2",
  segment_column: "category",
  problem_type: "regression",
  n_segments: 0,
  n_samples: 0,
  segments: [],
  most_improved_segment: null,
  most_declining_segment: null,
  verdict: "no_data",
  summary: "Not enough prediction data.",
  n_days: 30,
}

const stableResult: SegmentPredTrendResult = {
  deployment_id: "dep-3",
  segment_column: "region",
  problem_type: "regression",
  n_segments: 1,
  n_samples: 5,
  segments: [
    {
      name: "North",
      n_samples: 5,
      n_days_with_data: 2,
      direction: "stable",
      direction_label: "Stable",
      change_pct: 0.5,
      first_mean: 100.0,
      last_mean: 100.5,
      daily_stats: [
        { date: "2026-05-25", mean: 100.0, count: 3 },
        { date: "2026-05-30", mean: 100.5, count: 2 },
      ],
    },
  ],
  most_improved_segment: null,
  most_declining_segment: null,
  verdict: "stable",
  summary: "All segments are stable.",
  n_days: 30,
}

describe("SegmentPredictionTrendCard", () => {
  it("renders the card container", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("segment-pred-trend-card")).toBeInTheDocument()
  })

  it("shows the verdict badge", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Diverging")
  })

  it("shows segment column badge", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("segment-col-badge")).toHaveTextContent("region")
  })

  it("shows n-segments badge", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("n-segments-badge")).toHaveTextContent("2 segments")
  })

  it("shows n-days badge", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("n-days-badge")).toHaveTextContent("30d window")
  })

  it("shows summary text", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("summary-text")).toHaveTextContent("Predictions are diverging")
  })

  it("shows most improved segment", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("most-improved")).toHaveTextContent("East")
  })

  it("shows most declining segment", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("most-declining")).toHaveTextContent("West")
  })

  it("renders segment rows", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("segments-list")).toBeInTheDocument()
    expect(screen.getByTestId("segment-row-East")).toBeInTheDocument()
    expect(screen.getByTestId("segment-row-West")).toBeInTheDocument()
  })

  it("shows direction badge for trending up segment", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("segment-direction-East")).toHaveTextContent("Trending Up")
  })

  it("shows direction badge for trending down segment", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("segment-direction-West")).toHaveTextContent("Trending Down")
  })

  it("shows change percentage for each segment", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("segment-change-East")).toHaveTextContent("+40.0%")
    expect(screen.getByTestId("segment-change-West")).toHaveTextContent("-20.0%")
  })

  it("shows empty state for no_data verdict", () => {
    render(<SegmentPredictionTrendCard result={noDataResult} />)
    expect(screen.getByTestId("empty-state")).toBeInTheDocument()
  })

  it("renders chart container for segments with enough data", () => {
    render(<SegmentPredictionTrendCard result={divergingResult} />)
    expect(screen.getByTestId("segment-chart-East")).toBeInTheDocument()
  })

  it("does not show most-improved when null", () => {
    render(<SegmentPredictionTrendCard result={stableResult} />)
    expect(screen.queryByTestId("most-improved")).not.toBeInTheDocument()
  })

  it("does not show most-declining when null", () => {
    render(<SegmentPredictionTrendCard result={stableResult} />)
    expect(screen.queryByTestId("most-declining")).not.toBeInTheDocument()
  })

  it("shows stable direction badge", () => {
    render(<SegmentPredictionTrendCard result={stableResult} />)
    expect(screen.getByTestId("segment-direction-North")).toHaveTextContent("Stable")
  })
})
