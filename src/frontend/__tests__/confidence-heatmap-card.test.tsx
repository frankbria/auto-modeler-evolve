import React from "react"
import { render, screen } from "@testing-library/react"

import { ConfidenceHeatmapCard } from "@/components/deploy/confidence-heatmap-card"
import type { ConfidenceHeatmapResult } from "@/lib/types"

const gapResult: ConfidenceHeatmapResult = {
  deployment_id: "dep-1",
  feature_x: "age",
  feature_y: "region",
  x_labels: ["18–35", "36–55", "56+"],
  y_labels: ["East", "West"],
  cells: [
    { x_idx: 0, y_idx: 0, x_label: "18–35", y_label: "East", avg_confidence: 0.82, n_samples: 15, is_low_confidence: false },
    { x_idx: 0, y_idx: 1, x_label: "18–35", y_label: "West", avg_confidence: 0.52, n_samples: 8, is_low_confidence: true },
    { x_idx: 1, y_idx: 0, x_label: "36–55", y_label: "East", avg_confidence: 0.88, n_samples: 20, is_low_confidence: false },
    { x_idx: 1, y_idx: 1, x_label: "36–55", y_label: "West", avg_confidence: 0.79, n_samples: 12, is_low_confidence: false },
    { x_idx: 2, y_idx: 0, x_label: "56+", y_label: "East", avg_confidence: 0.91, n_samples: 18, is_low_confidence: false },
    { x_idx: 2, y_idx: 1, x_label: "56+", y_label: "West", avg_confidence: 0.85, n_samples: 10, is_low_confidence: false },
  ],
  low_confidence_zones: [
    { x_label: "18–35", y_label: "West", avg_confidence: 0.52, n_samples: 8 },
  ],
  overall_min: 0.52,
  overall_max: 0.91,
  overall_mean: 0.80,
  n_samples: 83,
  n_cells_total: 6,
  n_cells_populated: 6,
  verdict: "gaps_found",
  summary: "1 input combination shows low confidence (<65%), worst for age='18–35' + region='West' (52% avg, 8 predictions).",
}

const highResult: ConfidenceHeatmapResult = {
  deployment_id: "dep-2",
  feature_x: "age",
  feature_y: "region",
  x_labels: ["0–50", "51–100"],
  y_labels: ["East", "West"],
  cells: [
    { x_idx: 0, y_idx: 0, x_label: "0–50", y_label: "East", avg_confidence: 0.92, n_samples: 30, is_low_confidence: false },
    { x_idx: 0, y_idx: 1, x_label: "0–50", y_label: "West", avg_confidence: 0.88, n_samples: 25, is_low_confidence: false },
    { x_idx: 1, y_idx: 0, x_label: "51–100", y_label: "East", avg_confidence: 0.95, n_samples: 20, is_low_confidence: false },
    { x_idx: 1, y_idx: 1, x_label: "51–100", y_label: "West", avg_confidence: 0.90, n_samples: 15, is_low_confidence: false },
  ],
  low_confidence_zones: [],
  overall_min: 0.88,
  overall_max: 0.95,
  overall_mean: 0.91,
  n_samples: 90,
  n_cells_total: 4,
  n_cells_populated: 4,
  verdict: "uniform_high",
  summary: "Model confidence is uniformly high across all 'age' × 'region' combinations (min 88%). No weak spots detected.",
}

const noDataResult: ConfidenceHeatmapResult = {
  feature_x: "age",
  feature_y: "region",
  x_labels: [],
  y_labels: [],
  cells: [],
  low_confidence_zones: [],
  overall_min: null,
  overall_max: null,
  overall_mean: null,
  n_samples: 3,
  n_cells_total: 0,
  n_cells_populated: 0,
  verdict: "insufficient_data",
  summary: "Not enough prediction data for a confidence heatmap (only 3 predictions). Make more predictions to enable heatmap analysis.",
}

// ---------------------------------------------------------------------------
// Card structure
// ---------------------------------------------------------------------------

describe("ConfidenceHeatmapCard — gaps_found", () => {
  it("renders the card", () => {
    render(<ConfidenceHeatmapCard result={gapResult} />)
    expect(screen.getByTestId("confidence-heatmap-card")).toBeInTheDocument()
  })

  it("shows verdict badge with correct label", () => {
    render(<ConfidenceHeatmapCard result={gapResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Gaps Found")
  })

  it("shows feature pair badge", () => {
    render(<ConfidenceHeatmapCard result={gapResult} />)
    expect(screen.getByTestId("feature-pair-badge")).toHaveTextContent("age × region")
  })

  it("shows n-samples badge", () => {
    render(<ConfidenceHeatmapCard result={gapResult} />)
    expect(screen.getByTestId("n-samples-badge")).toHaveTextContent("83")
  })

  it("shows summary text", () => {
    render(<ConfidenceHeatmapCard result={gapResult} />)
    expect(screen.getByTestId("summary-text")).toHaveTextContent("1 input combination")
  })

  it("renders heatmap grid", () => {
    render(<ConfidenceHeatmapCard result={gapResult} />)
    expect(screen.getByTestId("heatmap-grid")).toBeInTheDocument()
  })

  it("renders cells with confidence values", () => {
    render(<ConfidenceHeatmapCard result={gapResult} />)
    // Cell at x=0, y=1 should show 52%
    const cell = screen.getByTestId("cell-0-1")
    expect(cell).toHaveTextContent("52%")
  })

  it("shows low confidence zones callout", () => {
    render(<ConfidenceHeatmapCard result={gapResult} />)
    expect(screen.getByTestId("low-confidence-zones")).toBeInTheDocument()
    expect(screen.getByTestId("low-confidence-zones")).toHaveTextContent("18–35")
    expect(screen.getByTestId("low-confidence-zones")).toHaveTextContent("West")
  })

  it("shows color legend", () => {
    render(<ConfidenceHeatmapCard result={gapResult} />)
    expect(screen.getByTestId("legend")).toBeInTheDocument()
  })

  it("shows stats row with min/mean/max", () => {
    render(<ConfidenceHeatmapCard result={gapResult} />)
    const stats = screen.getByTestId("stats-row")
    expect(stats).toHaveTextContent("52%")  // min
    expect(stats).toHaveTextContent("80%")  // mean
    expect(stats).toHaveTextContent("91%")  // max
  })
})

describe("ConfidenceHeatmapCard — uniform_high", () => {
  it("shows Uniformly High verdict", () => {
    render(<ConfidenceHeatmapCard result={highResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Uniformly High")
  })

  it("does not show low confidence zones callout", () => {
    render(<ConfidenceHeatmapCard result={highResult} />)
    expect(screen.queryByTestId("low-confidence-zones")).not.toBeInTheDocument()
  })

  it("shows grid with 4 cells", () => {
    render(<ConfidenceHeatmapCard result={highResult} />)
    expect(screen.getByTestId("cell-0-0")).toBeInTheDocument()
    expect(screen.getByTestId("cell-1-1")).toBeInTheDocument()
  })
})

describe("ConfidenceHeatmapCard — insufficient_data", () => {
  it("shows insufficient data verdict", () => {
    render(<ConfidenceHeatmapCard result={noDataResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Insufficient Data")
  })

  it("shows empty state message", () => {
    render(<ConfidenceHeatmapCard result={noDataResult} />)
    expect(screen.getByTestId("empty-state")).toBeInTheDocument()
    expect(screen.getByTestId("empty-state")).toHaveTextContent("Not enough")
  })

  it("does not render the grid when no cells", () => {
    render(<ConfidenceHeatmapCard result={noDataResult} />)
    expect(screen.queryByTestId("heatmap-grid")).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action
// ---------------------------------------------------------------------------

import { useAppStore } from "@/lib/store"
import { act } from "@testing-library/react"

describe("attachConfHeatmapToLastMessage", () => {
  beforeEach(() => {
    act(() => {
      useAppStore.setState({ messages: [{ role: "assistant", content: "Hi" }] })
    })
  })

  it("attaches conf_heatmap to last assistant message", () => {
    act(() => {
      useAppStore.getState().attachConfHeatmapToLastMessage(gapResult)
    })
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].conf_heatmap).toBeDefined()
    expect(msgs[msgs.length - 1].conf_heatmap?.verdict).toBe("gaps_found")
  })

  it("does not attach to user messages", () => {
    act(() => {
      useAppStore.setState({ messages: [{ role: "user", content: "Hi" }] })
    })
    act(() => {
      useAppStore.getState().attachConfHeatmapToLastMessage(gapResult)
    })
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].conf_heatmap).toBeUndefined()
  })
})
