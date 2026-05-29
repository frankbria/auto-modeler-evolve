import React from "react"
import { render, screen } from "@testing-library/react"
import { ThresholdAnalysisCard } from "@/components/models/threshold-analysis-card"
import type { ThresholdAnalysisResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// Recharts uses ResizeObserver which jsdom doesn't implement
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
global.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver

function makeSweepPoint(threshold: number) {
  return {
    threshold,
    precision: 0.75 - threshold * 0.1,
    recall: 0.40 + threshold * 0.5,
    f1: 0.55,
    positive_rate: 0.30 + threshold * 0.2,
  }
}

function makeResult(
  overrides: Partial<ThresholdAnalysisResult> = {}
): ThresholdAnalysisResult {
  const sweep = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    .map(makeSweepPoint)

  return {
    model_run_id: "run-1",
    algorithm: "logistic_regression",
    target_col: "churn",
    sweep,
    current_threshold: 0.5,
    current_metrics: makeSweepPoint(0.5),
    recommendations: {
      max_f1: {
        threshold: 0.40,
        precision: 0.78,
        recall: 0.72,
        f1: 0.75,
        label: "Balanced (Max F1)",
        description: "At 40%, precision and recall are most balanced.",
      },
      high_recall: {
        threshold: 0.20,
        precision: 0.55,
        recall: 0.85,
        f1: 0.67,
        label: "High Recall (Catch More)",
        description: "At 20%, you catch 85% of actual positives.",
      },
      high_precision: {
        threshold: 0.70,
        precision: 0.88,
        recall: 0.50,
        f1: 0.64,
        label: "High Precision (Fewer False Alarms)",
        description: "At 70%, 88% of flagged cases are truly positive.",
      },
    },
    n_positive: 30,
    n_total: 100,
    prevalence: 0.3,
    positive_class: "1",
    summary: "At the default 50% threshold: 70% precision, 65% recall.",
    ...overrides,
  }
}

describe("ThresholdAnalysisCard", () => {
  it("renders without crashing", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByRole("region")).toBeInTheDocument()
  })

  it("shows the card heading", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText("Threshold Advisor")).toBeInTheDocument()
  })

  it("shows Classification badge", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("shows target column badge", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText("Target: churn")).toBeInTheDocument()
  })

  it("shows prevalence badge with n_positive/n_total", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText(/30% positive \(30\/100/)).toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    // summary text appears in both the summary paragraph and the current-threshold section label
    const matches = screen.getAllByText((content) =>
      content.includes("At the default 50% threshold")
    )
    expect(matches.length).toBeGreaterThan(0)
  })

  it("shows current threshold section", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText("At Default 50% Threshold")).toBeInTheDocument()
  })

  it("shows precision, recall, F1 at current threshold", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    // These labels appear in the current-threshold section
    expect(screen.getAllByText("Precision:").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Recall:").length).toBeGreaterThan(0)
    expect(screen.getAllByText("F1:").length).toBeGreaterThan(0)
  })

  it("shows threshold trade-off curve section", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText("Threshold Trade-off Curve")).toBeInTheDocument()
  })

  it("shows three recommendation rows", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    const recs = screen.getAllByTestId("threshold-recommendation")
    expect(recs).toHaveLength(3)
  })

  it("shows Recommended badge on max_f1 option", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText("Recommended")).toBeInTheDocument()
  })

  it("shows max_f1 recommendation label", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText("Balanced (Max F1)")).toBeInTheDocument()
  })

  it("shows high_recall recommendation label", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText("High Recall (Catch More)")).toBeInTheDocument()
  })

  it("shows high_precision recommendation label", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText("High Precision (Fewer False Alarms)")).toBeInTheDocument()
  })

  it("shows threshold % badges on recommendations", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    // max_f1 = 40%, high_recall = 20%, high_precision = 70%
    // Use getAllByText since "70%" may appear more than once (in precision display too)
    expect(screen.getAllByText("40%").length).toBeGreaterThan(0)
    expect(screen.getAllByText("20%").length).toBeGreaterThan(0)
    expect(screen.getAllByText("70%").length).toBeGreaterThan(0)
  })

  it("shows business guidance section", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByText("Which threshold is right for me?")).toBeInTheDocument()
  })

  it("shows figcaption for screen readers", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(screen.getByRole("region")).toHaveAccessibleName(
      "Classification threshold analysis"
    )
  })

  it("shows recommendation description text", () => {
    render(<ThresholdAnalysisCard result={makeResult()} />)
    expect(
      screen.getByText("At 40%, precision and recall are most balanced.")
    ).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Zustand store action
// ---------------------------------------------------------------------------

describe("attachThresholdAnalysisToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "hi", timestamp: new Date().toISOString() },
        { role: "assistant", content: "hello", timestamp: new Date().toISOString() },
      ],
    })
  })

  it("attaches threshold_analysis to last assistant message", () => {
    const result = makeResult()
    useAppStore.getState().attachThresholdAnalysisToLastMessage(result)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1] as Record<string, unknown>
    expect(last.threshold_analysis).toEqual(result)
  })

  it("does not modify user messages", () => {
    const result = makeResult()
    useAppStore.setState({
      messages: [
        { role: "user", content: "hi", timestamp: new Date().toISOString() },
      ],
    })
    useAppStore.getState().attachThresholdAnalysisToLastMessage(result)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1] as Record<string, unknown>
    expect(last.threshold_analysis).toBeUndefined()
  })

  it("preserves other message fields", () => {
    const result = makeResult()
    useAppStore.getState().attachThresholdAnalysisToLastMessage(result)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.content).toBe("hello")
    expect(last.role).toBe("assistant")
  })
})
