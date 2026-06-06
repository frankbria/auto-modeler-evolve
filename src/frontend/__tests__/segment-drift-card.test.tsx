import React from "react"
import { render, screen } from "@testing-library/react"

import { SegmentDriftCard } from "@/components/deploy/segment-drift-card"
import type { SegmentDriftResult } from "@/lib/types"

const baseResult: SegmentDriftResult = {
  deployment_id: "dep-1",
  segment_column: "region",
  n_segments: 2,
  n_samples: 30,
  segments: [
    {
      name: "West",
      sample_count: 15,
      drift_score: 45.0,
      status: "high",
      top_drifting_features: [{ name: "units", drift_pct: 45.0, feature_type: "numeric" }],
    },
    {
      name: "East",
      sample_count: 15,
      drift_score: 5.0,
      status: "low",
      top_drifting_features: [],
    },
  ],
  max_drift_segment: "West",
  avg_drift_score: 25.0,
  high_count: 1,
  moderate_count: 0,
  low_count: 1,
  minimal_count: 0,
  verdict: "concentrated",
  summary: "Drift is concentrated in the West segment.",
}

describe("SegmentDriftCard", () => {
  it("renders the card container", () => {
    render(<SegmentDriftCard result={baseResult} />)
    expect(screen.getByTestId("segment-drift-card")).toBeInTheDocument()
  })

  it("shows the segment column badge", () => {
    render(<SegmentDriftCard result={baseResult} />)
    expect(screen.getByTestId("segment-col-badge")).toHaveTextContent("By: region")
  })

  it("shows n-segments badge", () => {
    render(<SegmentDriftCard result={baseResult} />)
    expect(screen.getByTestId("n-segments-badge")).toHaveTextContent("2 segments")
  })

  it("shows verdict badge", () => {
    render(<SegmentDriftCard result={baseResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Drift Concentrated")
  })

  it("shows summary text", () => {
    render(<SegmentDriftCard result={baseResult} />)
    expect(screen.getByTestId("summary-text")).toHaveTextContent(
      "Drift is concentrated in the West segment."
    )
  })

  it("renders segment names", () => {
    render(<SegmentDriftCard result={baseResult} />)
    const names = screen.getAllByTestId("segment-name")
    expect(names.map((n) => n.textContent)).toContain("West")
    expect(names.map((n) => n.textContent)).toContain("East")
  })

  it("renders status badges for each segment", () => {
    render(<SegmentDriftCard result={baseResult} />)
    const badges = screen.getAllByTestId("status-badge")
    expect(badges.length).toBe(2)
  })

  it("shows high status badge", () => {
    render(<SegmentDriftCard result={baseResult} />)
    const badges = screen.getAllByTestId("status-badge")
    expect(badges.some((b) => b.textContent === "high")).toBe(true)
  })

  it("shows top drifting feature names", () => {
    render(<SegmentDriftCard result={baseResult} />)
    const featureCells = screen.getAllByTestId("top-features")
    expect(featureCells[0]).toHaveTextContent("units")
  })

  it("shows status counts for high segments", () => {
    render(<SegmentDriftCard result={baseResult} />)
    expect(screen.getByTestId("status-counts")).toHaveTextContent("1 high")
  })

  it("renders segments list", () => {
    render(<SegmentDriftCard result={baseResult} />)
    expect(screen.getByTestId("segments-list")).toBeInTheDocument()
  })

  it("renders with minimal verdict", () => {
    const minResult: SegmentDriftResult = {
      ...baseResult,
      segments: [
        { name: "East", sample_count: 10, drift_score: 0.5, status: "minimal", top_drifting_features: [] },
      ],
      high_count: 0,
      moderate_count: 0,
      low_count: 0,
      minimal_count: 1,
      verdict: "minimal",
      summary: "Drift is minimal across all segments.",
    }
    render(<SegmentDriftCard result={minResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Minimal Drift")
  })

  it("renders no_data verdict gracefully", () => {
    const noData: SegmentDriftResult = {
      ...baseResult,
      segments: [],
      n_segments: 0,
      verdict: "no_data",
      summary: "No prediction logs available.",
    }
    render(<SegmentDriftCard result={noData} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("No Data")
    expect(screen.queryByTestId("segments-list")).not.toBeInTheDocument()
  })

  it("renders widespread verdict", () => {
    const widespread: SegmentDriftResult = {
      ...baseResult,
      verdict: "widespread",
      summary: "Drift is widespread.",
    }
    render(<SegmentDriftCard result={widespread} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Drift Widespread")
  })

  it("includes sr-only figcaption", () => {
    render(<SegmentDriftCard result={baseResult} />)
    const fig = screen.getByTestId("segment-drift-card")
    expect(fig.tagName).toBe("FIGURE")
    const captions = fig.querySelectorAll("figcaption")
    expect(captions.length).toBeGreaterThan(0)
  })
})
