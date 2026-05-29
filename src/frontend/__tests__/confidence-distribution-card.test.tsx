import React from "react"
import { render, screen } from "@testing-library/react"
import { ConfidenceDistributionCard } from "@/components/models/confidence-distribution-card"
import type { ConfidenceDistributionResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// Recharts uses ResizeObserver which jsdom doesn't implement
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
global.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver

function makeResult(
  overrides: Partial<ConfidenceDistributionResult> = {}
): ConfidenceDistributionResult {
  const bins = Array.from({ length: 10 }, (_, i) => ({
    lo: i * 0.1,
    hi: (i + 1) * 0.1,
    count: i < 5 ? 5 : 15,
    pct: i < 5 ? 5.0 : 15.0,
    label: `${i * 10}–${(i + 1) * 10}%`,
  }))
  return {
    model_run_id: "run-1",
    algorithm: "logistic_regression",
    target_col: "churn",
    bins,
    n_total: 100,
    n_bins: 10,
    mean_confidence: 0.72,
    median_confidence: 0.75,
    pct_high: 65.0,
    pct_medium: 25.0,
    pct_low: 10.0,
    n_high: 65,
    n_medium: 25,
    n_low: 10,
    decisiveness: "decisive",
    decisiveness_label: "Decisive",
    per_class_mean: [
      { class_name: "No", mean_confidence: 0.85, count: 60 },
      { class_name: "Yes", mean_confidence: 0.78, count: 40 },
    ],
    summary: "The model has mean confidence of 72% across 100 training predictions.",
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Card component tests
// ---------------------------------------------------------------------------

describe("ConfidenceDistributionCard", () => {
  it("renders the header with icon", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getAllByText(/Confidence Distribution/i).length).toBeGreaterThanOrEqual(1)
  })

  it("renders the decisiveness badge for decisive model", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getByTestId("decisiveness-badge")).toHaveTextContent("Decisive")
  })

  it("renders moderate decisiveness badge", () => {
    render(
      <ConfidenceDistributionCard
        result={makeResult({ decisiveness: "moderate", decisiveness_label: "Moderate" })}
      />
    )
    expect(screen.getByTestId("decisiveness-badge")).toHaveTextContent("Moderate")
  })

  it("renders uncertain decisiveness badge", () => {
    render(
      <ConfidenceDistributionCard
        result={makeResult({ decisiveness: "uncertain", decisiveness_label: "Uncertain" })}
      />
    )
    expect(screen.getByTestId("decisiveness-badge")).toHaveTextContent("Uncertain")
  })

  it("shows confidence breakdown percentages", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    const breakdown = screen.getByTestId("confidence-breakdown")
    expect(breakdown).toBeInTheDocument()
    // Should contain percentage labels
    expect(breakdown.textContent).toContain("65%")
    expect(breakdown.textContent).toContain("25%")
    expect(breakdown.textContent).toContain("10%")
  })

  it("shows High/Medium/Low labels", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getByText(/High/)).toBeInTheDocument()
    expect(screen.getByText(/Medium/)).toBeInTheDocument()
    expect(screen.getByText(/Low/)).toBeInTheDocument()
  })

  it("shows mean and median confidence", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getAllByText(/Mean confidence/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/72\.0%/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Median/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/75\.0%/).length).toBeGreaterThanOrEqual(1)
  })

  it("shows N (total predictions)", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getAllByText(/N:/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText("100").length).toBeGreaterThanOrEqual(1)
  })

  it("renders target column name", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getByText("churn")).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getByTestId("confidence-summary")).toHaveTextContent(
      "mean confidence of 72%"
    )
  })

  it("renders per-class confidence for each class", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getByTestId("class-confidence-No")).toBeInTheDocument()
    expect(screen.getByTestId("class-confidence-Yes")).toBeInTheDocument()
  })

  it("renders progress bars for per-class confidence with ARIA attrs", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    const bars = screen.getAllByRole("progressbar")
    expect(bars.length).toBeGreaterThanOrEqual(2)
  })

  it("has a figcaption with sr-only accessibility text", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    const fig = screen.getByRole("figure")
    expect(fig.querySelector("figcaption.sr-only")).not.toBeNull()
  })

  it("renders the chart area with aria-label", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getByLabelText(/Confidence histogram/i)).toBeInTheDocument()
  })

  it("does not crash with empty per_class_mean", () => {
    render(
      <ConfidenceDistributionCard
        result={makeResult({ per_class_mean: [] })}
      />
    )
    expect(screen.getByTestId("confidence-distribution-card")).toBeInTheDocument()
  })

  it("applies the correct card test id", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getByTestId("confidence-distribution-card")).toBeInTheDocument()
  })

  it("renders algorithm name in human form", () => {
    render(<ConfidenceDistributionCard result={makeResult()} />)
    expect(screen.getByText("Logistic Regression")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachConfidenceDistributionToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches result to last assistant message", () => {
    useAppStore.setState({
      messages: [{ role: "assistant", content: "test" }],
    })
    const result = makeResult()
    useAppStore.getState().attachConfidenceDistributionToLastMessage(result)
    const { messages } = useAppStore.getState()
    expect(messages[0].confidence_distribution).toEqual(result)
  })

  it("does not modify if last message is user", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "hi" }],
    })
    const result = makeResult()
    useAppStore.getState().attachConfidenceDistributionToLastMessage(result)
    const { messages } = useAppStore.getState()
    expect(messages[0].confidence_distribution).toBeUndefined()
  })

  it("does not crash with empty messages array", () => {
    useAppStore.setState({ messages: [] })
    expect(() => {
      useAppStore.getState().attachConfidenceDistributionToLastMessage(makeResult())
    }).not.toThrow()
  })
})
