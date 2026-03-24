/**
 * Tests for SegmentPerformanceCard component and related store/API plumbing.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"
import { SegmentPerformanceCard } from "@/components/models/segment-performance-card"
import type { SegmentPerformanceResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

fetchMock.enableMocks()

// --- Fixtures ---------------------------------------------------------------

const regressionResult: SegmentPerformanceResult = {
  group_col: "region",
  algorithm: "linear_regression",
  problem_type: "regression",
  metric_name: "R²",
  segments: [
    { name: "East", n: 4, metric: 0.91, metric_name: "R²", status: "strong", low_sample: false },
    { name: "West", n: 4, metric: 0.72, metric_name: "R²", status: "moderate", low_sample: false },
    { name: "North", n: 2, metric: 0.35, metric_name: "R²", status: "poor", low_sample: true },
  ],
  best_segment: "East",
  worst_segment: "North",
  gap: 0.56,
  summary:
    "Your model performs best on 'East' (R²=0.91) and worst on 'North' (R²=0.35). The 0.56 performance gap is significant.",
}

const classificationResult: SegmentPerformanceResult = {
  group_col: "product",
  algorithm: "random_forest",
  problem_type: "classification",
  metric_name: "Accuracy",
  segments: [
    {
      name: "Widget A",
      n: 5,
      metric: 0.9,
      metric_name: "Accuracy",
      status: "strong",
      low_sample: false,
    },
    {
      name: "Widget B",
      n: 3,
      metric: 0.67,
      metric_name: "Accuracy",
      status: "moderate",
      low_sample: false,
    },
  ],
  best_segment: "Widget A",
  worst_segment: "Widget B",
  gap: 0.23,
  summary: "Your model performs best on 'Widget A' (Accuracy=90.0%) and worst on 'Widget B' (Accuracy=67.0%).",
}

// --- Component rendering tests -----------------------------------------------

describe("SegmentPerformanceCard", () => {
  it("renders the group column name", () => {
    render(<SegmentPerformanceCard result={regressionResult} />)
    expect(screen.getByText("region")).toBeInTheDocument()
  })

  it("renders the summary text", () => {
    render(<SegmentPerformanceCard result={regressionResult} />)
    expect(screen.getByText(/performs best on/i)).toBeInTheDocument()
  })

  it("renders all segment rows", () => {
    render(<SegmentPerformanceCard result={regressionResult} />)
    expect(screen.getByText("East")).toBeInTheDocument()
    expect(screen.getByText("West")).toBeInTheDocument()
    expect(screen.getByText("North")).toBeInTheDocument()
  })

  it("shows ▲ best and ▼ lowest labels", () => {
    render(<SegmentPerformanceCard result={regressionResult} />)
    expect(screen.getByText(/▲ best/i)).toBeInTheDocument()
    expect(screen.getByText(/▼ lowest/i)).toBeInTheDocument()
  })

  it("renders regression metric values as decimals", () => {
    render(<SegmentPerformanceCard result={regressionResult} />)
    expect(screen.getByText("0.910")).toBeInTheDocument()
  })

  it("renders classification metrics as percentages", () => {
    render(<SegmentPerformanceCard result={classificationResult} />)
    expect(screen.getByText("90.0%")).toBeInTheDocument()
    expect(screen.getByText("67.0%")).toBeInTheDocument()
  })

  it("shows low-sample warning indicator", () => {
    render(<SegmentPerformanceCard result={regressionResult} />)
    // The ! indicator appears for North (low_sample=true)
    expect(screen.getByText("!")).toBeInTheDocument()
    expect(screen.getByText(/fewer than 10 rows/i)).toBeInTheDocument()
  })

  it("renders status badges", () => {
    render(<SegmentPerformanceCard result={regressionResult} />)
    expect(screen.getByText("strong")).toBeInTheDocument()
    expect(screen.getByText("moderate")).toBeInTheDocument()
    expect(screen.getByText("poor")).toBeInTheDocument()
  })

  it("renders testid for assertions", () => {
    render(<SegmentPerformanceCard result={regressionResult} />)
    expect(screen.getByTestId("segment-performance-card")).toBeInTheDocument()
  })
})

// --- Store action tests -------------------------------------------------------

describe("attachSegmentPerformanceToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches segment_performance to last assistant message", () => {
    useAppStore.setState({
      messages: [{ role: "assistant", content: "Here is the breakdown.", timestamp: "" }],
    })
    useAppStore.getState().attachSegmentPerformanceToLastMessage(regressionResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].segment_performance).toBeDefined()
    expect(msgs[0].segment_performance?.group_col).toBe("region")
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "How does my model perform?", timestamp: "" }],
    })
    useAppStore.getState().attachSegmentPerformanceToLastMessage(regressionResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].segment_performance).toBeUndefined()
  })
})

// --- API client smoke test ---------------------------------------------------

describe("api.models.getSegmentPerformance", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
  })

  it("builds the correct URL", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(regressionResult))

    const { api } = await import("@/lib/api")
    await api.models.getSegmentPerformance("run-123", "region")

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/models/run-123/segment-performance?col=region"),
    )
  })
})
