import React from "react"
import { render, screen } from "@testing-library/react"

import { SegmentConfidenceTrendCard } from "@/components/deploy/segment-confidence-trend-card"
import type { SegmentConfTrendResult } from "@/lib/types"

const DAILY_STATS_GOOD = [
  { date: "2026-05-20", mean: 0.92, count: 3 },
  { date: "2026-05-25", mean: 0.88, count: 4 },
  { date: "2026-05-30", mean: 0.95, count: 3 },
]

const DAILY_STATS_BAD = [
  { date: "2026-05-20", mean: 0.88, count: 2 },
  { date: "2026-05-25", mean: 0.72, count: 3 },
  { date: "2026-05-30", mean: 0.61, count: 2 },
]

const gapResult: SegmentConfTrendResult = {
  deployment_id: "dep-1",
  segment_column: "category",
  problem_type: "classification",
  metric: "confidence",
  n_segments: 2,
  n_samples: 17,
  segments: [
    {
      name: "VIP",
      n_samples: 10,
      n_days_with_data: 3,
      direction: "improving",
      direction_label: "Improving",
      change_pct: 3.3,
      first_mean: 0.92,
      last_mean: 0.95,
      daily_stats: DAILY_STATS_GOOD,
    },
    {
      name: "Basic",
      n_samples: 7,
      n_days_with_data: 3,
      direction: "deteriorating",
      direction_label: "Deteriorating",
      change_pct: -30.7,
      first_mean: 0.88,
      last_mean: 0.61,
      daily_stats: DAILY_STATS_BAD,
    },
  ],
  most_confident_segment: "VIP",
  least_confident_segment: "Basic",
  calibration_gap: true,
  verdict: "calibration_gap",
  summary: "Significant confidence gap across segments.",
  n_days: 30,
}

const noDataResult: SegmentConfTrendResult = {
  deployment_id: "dep-2",
  segment_column: "category",
  problem_type: "classification",
  metric: "confidence",
  n_segments: 0,
  n_samples: 0,
  segments: [],
  most_confident_segment: null,
  least_confident_segment: null,
  calibration_gap: false,
  verdict: "no_data",
  summary: "Not enough prediction data.",
  n_days: 30,
}

const stableResult: SegmentConfTrendResult = {
  deployment_id: "dep-3",
  segment_column: "region",
  problem_type: "classification",
  metric: "confidence",
  n_segments: 1,
  n_samples: 5,
  segments: [
    {
      name: "North",
      n_samples: 5,
      n_days_with_data: 2,
      direction: "stable",
      direction_label: "Stable",
      change_pct: 0.2,
      first_mean: 0.80,
      last_mean: 0.802,
      daily_stats: [
        { date: "2026-05-25", mean: 0.80, count: 3 },
        { date: "2026-05-30", mean: 0.802, count: 2 },
      ],
    },
  ],
  most_confident_segment: "North",
  least_confident_segment: "North",
  calibration_gap: false,
  verdict: "stable",
  summary: "All segments are stable.",
  n_days: 30,
}

const spreadResult: SegmentConfTrendResult = {
  deployment_id: "dep-4",
  segment_column: "region",
  problem_type: "regression",
  metric: "prediction_spread",
  n_segments: 1,
  n_samples: 6,
  segments: [
    {
      name: "East",
      n_samples: 6,
      n_days_with_data: 2,
      direction: "stable",
      direction_label: "Stable",
      change_pct: 0.0,
      first_mean: 5.2,
      last_mean: 5.2,
      daily_stats: [
        { date: "2026-05-25", mean: 5.2, count: 3 },
        { date: "2026-05-30", mean: 5.2, count: 3 },
      ],
    },
  ],
  most_confident_segment: "East",
  least_confident_segment: "East",
  calibration_gap: false,
  verdict: "stable",
  summary: "Prediction spread is stable.",
  n_days: 30,
}

describe("SegmentConfidenceTrendCard", () => {
  it("renders the card container", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("segment-conf-trend-card")).toBeInTheDocument()
  })

  it("shows the verdict badge", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Calibration Gap")
  })

  it("shows segment column badge", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("segment-col-badge")).toHaveTextContent("category")
  })

  it("shows n-segments badge", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("n-segments-badge")).toHaveTextContent("2 segments")
  })

  it("shows n-days badge", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("n-days-badge")).toHaveTextContent("30d window")
  })

  it("shows metric badge for confidence", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("metric-badge")).toHaveTextContent("Confidence")
  })

  it("shows metric badge for prediction spread", () => {
    render(<SegmentConfidenceTrendCard result={spreadResult} />)
    expect(screen.getByTestId("metric-badge")).toHaveTextContent("Spread")
  })

  it("shows summary text", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("summary-text")).toHaveTextContent("Significant confidence gap")
  })

  it("shows calibration gap warning when gap is true", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("calibration-gap-warning")).toBeInTheDocument()
  })

  it("does not show calibration gap warning when gap is false", () => {
    render(<SegmentConfidenceTrendCard result={stableResult} />)
    expect(screen.queryByTestId("calibration-gap-warning")).not.toBeInTheDocument()
  })

  it("shows most confident segment", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("most-confident")).toHaveTextContent("VIP")
  })

  it("shows least confident segment", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("least-confident")).toHaveTextContent("Basic")
  })

  it("hides least-confident when same as most-confident", () => {
    render(<SegmentConfidenceTrendCard result={stableResult} />)
    expect(screen.queryByTestId("least-confident")).not.toBeInTheDocument()
  })

  it("renders segment rows", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("segments-list")).toBeInTheDocument()
    expect(screen.getByTestId("segment-row-VIP")).toBeInTheDocument()
    expect(screen.getByTestId("segment-row-Basic")).toBeInTheDocument()
  })

  it("shows direction badge for improving segment", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("segment-direction-VIP")).toHaveTextContent("Improving")
  })

  it("shows direction badge for deteriorating segment", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("segment-direction-Basic")).toHaveTextContent("Deteriorating")
  })

  it("shows change percentage for each segment", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("segment-change-VIP")).toHaveTextContent("+3.3%")
    expect(screen.getByTestId("segment-change-Basic")).toHaveTextContent("-30.7%")
  })

  it("shows empty state for no_data verdict", () => {
    render(<SegmentConfidenceTrendCard result={noDataResult} />)
    expect(screen.getByTestId("empty-state")).toBeInTheDocument()
  })

  it("renders chart container for segments with enough data", () => {
    render(<SegmentConfidenceTrendCard result={gapResult} />)
    expect(screen.getByTestId("segment-chart-VIP")).toBeInTheDocument()
  })

  it("shows stable direction badge", () => {
    render(<SegmentConfidenceTrendCard result={stableResult} />)
    expect(screen.getByTestId("segment-direction-North")).toHaveTextContent("Stable")
  })
})
