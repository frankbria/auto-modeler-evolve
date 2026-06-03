import React from "react"
import { render, screen, act } from "@testing-library/react"
import { renderHook } from "@testing-library/react"
import { ProductionThresholdOptimizerCard } from "@/components/deploy/production-threshold-optimizer-card"
import type { ProductionThresholdOptimizerResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeSweepPoint(threshold: number, f1 = 0.75): import("@/lib/types").ProductionThresholdSweepPoint {
  return {
    threshold,
    precision: f1 + 0.05,
    recall: f1 - 0.05,
    f1,
    coverage: 1 - threshold,
    n_served: Math.round((1 - threshold) * 30),
  }
}

function makeFullResult(overrides: Partial<ProductionThresholdOptimizerResult> = {}): ProductionThresholdOptimizerResult {
  const sweep = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].map(t =>
    makeSweepPoint(t, t < 0.65 ? 0.65 : 0.8)
  )
  return {
    deployment_id: "dep-1",
    algorithm: "RandomForestClassifier",
    target_col: "churn",
    verdict: "improved",
    verdict_label: "Optimal ≠ Default",
    sweep,
    optimal_threshold: 0.65,
    optimal_f1: 0.80,
    optimal_precision: 0.82,
    optimal_recall: 0.78,
    optimal_coverage: 0.60,
    current_threshold: 0.50,
    current_f1: 0.65,
    current_precision: 0.67,
    current_recall: 0.63,
    current_coverage: 0.85,
    n_feedback: 45,
    n_correct_total: 36,
    overall_accuracy: 0.80,
    summary: "Based on 45 real-world outcomes: at threshold 65%, F1=0.80 with 60% coverage.",
    ...overrides,
  }
}

describe("ProductionThresholdOptimizerCard", () => {
  it("renders the card with aria-label", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult()} />)
    expect(screen.getByRole("region")).toHaveAttribute(
      "aria-label",
      expect.stringContaining("Production threshold optimizer")
    )
  })

  it("shows the title text", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult()} />)
    expect(screen.getByText("Production Threshold Optimizer")).toBeInTheDocument()
  })

  it("shows the verdict badge for improved", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult({ verdict: "improved" })} />)
    expect(screen.getByText("⚡ Optimal ≠ Default")).toBeInTheDocument()
  })

  it("shows the verdict badge for same", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult({ verdict: "same", verdict_label: "Default Is Optimal" })} />)
    expect(screen.getByText("✓ Default Is Optimal")).toBeInTheDocument()
  })

  it("shows n_feedback in outcomes badge", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult({ n_feedback: 45 })} />)
    expect(screen.getByText("45 outcomes")).toBeInTheDocument()
  })

  it("shows algorithm badge", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult({ algorithm: "RandomForestClassifier" })} />)
    expect(screen.getByText("RandomForestClassifier")).toBeInTheDocument()
  })

  it("shows optimal F1 stat", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult({ optimal_f1: 0.803 })} />)
    expect(screen.getByText("0.803")).toBeInTheDocument()
  })

  it("shows optimal coverage as percentage", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult({ optimal_coverage: 0.60 })} />)
    expect(screen.getByText("60%")).toBeInTheDocument()
  })

  it("shows overall accuracy", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult({ overall_accuracy: 0.80 })} />)
    expect(screen.getByText("80.0%")).toBeInTheDocument()
  })

  it("shows current F1 at default threshold", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult({ current_f1: 0.650 })} />)
    expect(screen.getByText("0.650")).toBeInTheDocument()
  })

  it("shows positive F1 gain", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult({ optimal_f1: 0.80, current_f1: 0.65 })} />)
    expect(screen.getByText("+0.150")).toBeInTheDocument()
  })

  it("shows the summary text", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult({ summary: "Unique summary text." })} />)
    expect(screen.getByText("Unique summary text.")).toBeInTheDocument()
  })

  it("shows sr-only figcaption", () => {
    render(<ProductionThresholdOptimizerCard result={makeFullResult()} />)
    const caption = document.querySelector("figcaption.sr-only")
    expect(caption).not.toBeNull()
  })

  it("renders no_data state gracefully", () => {
    render(<ProductionThresholdOptimizerCard result={{
      verdict: "no_data",
      verdict_label: "Not Enough Data",
      n_feedback: 2,
      summary: "Only 2 feedback records found.",
    }} />)
    expect(screen.getByText("Only 2 feedback records found.")).toBeInTheDocument()
    expect(screen.getByText("Feedback records with labels: 2")).toBeInTheDocument()
  })

  it("renders not_applicable state gracefully", () => {
    render(<ProductionThresholdOptimizerCard result={{
      verdict: "not_applicable",
      verdict_label: "N/A",
      summary: "Threshold optimization is only applicable to classification models.",
    }} />)
    expect(screen.getByText("Threshold optimization is only applicable to classification models.")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachProductionThresholdOptimizerToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches result to last assistant message", () => {
    const { result } = renderHook(() => useAppStore())
    act(() => {
      result.current.setMessages([
        { id: "1", role: "user", content: "optimize", timestamp: "2025-01-01T00:00:00Z" },
        { id: "2", role: "assistant", content: "Here is the optimizer.", timestamp: "2025-01-01T00:00:01Z" },
      ])
    })
    act(() => {
      result.current.attachProductionThresholdOptimizerToLastMessage(makeFullResult())
    })
    const msgs = result.current.messages
    const last = msgs[msgs.length - 1]
    expect(last.production_threshold_optimizer).toBeDefined()
    expect(last.production_threshold_optimizer?.verdict).toBe("improved")
  })

  it("does not attach when last message is not assistant", () => {
    const { result } = renderHook(() => useAppStore())
    act(() => {
      result.current.setMessages([
        { id: "1", role: "user", content: "test", timestamp: "2025-01-01T00:00:00Z" },
      ])
    })
    act(() => {
      result.current.attachProductionThresholdOptimizerToLastMessage(makeFullResult())
    })
    const msgs = result.current.messages
    expect(msgs[0].production_threshold_optimizer).toBeUndefined()
  })
})
