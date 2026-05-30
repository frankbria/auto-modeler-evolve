import React from "react"
import { render, screen } from "@testing-library/react"
import { SampleSizeAdequacyCard } from "@/components/models/sample-size-adequacy-card"
import type { SampleSizeAdequacyResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeResult(
  overrides: Partial<SampleSizeAdequacyResult> = {}
): SampleSizeAdequacyResult {
  return {
    model_run_id: "run-001",
    algorithm: "linear_regression",
    target_col: "revenue",
    n_rows: 200,
    n_features: 5,
    n_classes: null,
    problem_type: "regression",
    recommended_n: 50,
    shortfall: 0,
    coverage_pct: 100,
    ratio: 0.025,
    verdict: "adequate",
    verdict_label: "Adequate",
    cv_std: null,
    cv_stable: null,
    summary: "Your 200-row dataset meets the recommended minimum of 50 rows.",
    recommendations: ["Your dataset meets the size recommendation."],
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Component rendering
// ---------------------------------------------------------------------------

describe("SampleSizeAdequacyCard", () => {
  it("renders adequate verdict with emerald badge", () => {
    render(<SampleSizeAdequacyCard result={makeResult()} />)
    expect(screen.getByTestId("sample-size-adequacy-card")).toBeTruthy()
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Adequate")
  })

  it("renders borderline verdict with amber-style badge", () => {
    render(
      <SampleSizeAdequacyCard
        result={makeResult({ verdict: "borderline", verdict_label: "Borderline", shortfall: 10, coverage_pct: 80 })}
      />
    )
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Borderline")
  })

  it("renders insufficient verdict with rose-style badge", () => {
    render(
      <SampleSizeAdequacyCard
        result={makeResult({ verdict: "insufficient", verdict_label: "Insufficient", shortfall: 30, coverage_pct: 40 })}
      />
    )
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Insufficient")
  })

  it("shows current row count", () => {
    render(<SampleSizeAdequacyCard result={makeResult({ n_rows: 150 })} />)
    expect(screen.getByTestId("current-n")).toHaveTextContent("150")
  })

  it("shows recommended row count", () => {
    render(<SampleSizeAdequacyCard result={makeResult({ recommended_n: 500 })} />)
    expect(screen.getByTestId("recommended-n")).toHaveTextContent("500")
  })

  it("shows zero shortfall as None", () => {
    render(<SampleSizeAdequacyCard result={makeResult({ shortfall: 0 })} />)
    expect(screen.getByTestId("shortfall")).toHaveTextContent("None")
  })

  it("shows positive shortfall", () => {
    render(<SampleSizeAdequacyCard result={makeResult({ shortfall: 300, verdict: "insufficient" })} />)
    expect(screen.getByTestId("shortfall")).toHaveTextContent("300")
  })

  it("shows algorithm badge", () => {
    render(<SampleSizeAdequacyCard result={makeResult({ algorithm: "random_forest" })} />)
    expect(screen.getByText("random_forest")).toBeTruthy()
  })

  it("shows problem type badge", () => {
    render(<SampleSizeAdequacyCard result={makeResult({ problem_type: "classification" })} />)
    expect(screen.getByText("classification")).toBeTruthy()
  })

  it("shows ratio badge", () => {
    render(<SampleSizeAdequacyCard result={makeResult({ ratio: 0.025, n_features: 5, n_rows: 200 })} />)
    expect(screen.getByTestId("ratio-badge")).toBeTruthy()
  })

  it("shows coverage progress bar with aria attributes", () => {
    render(<SampleSizeAdequacyCard result={makeResult({ coverage_pct: 75 })} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toBeTruthy()
    expect(bar.getAttribute("aria-valuenow")).toBe("75")
    expect(bar.getAttribute("aria-valuemin")).toBe("0")
    expect(bar.getAttribute("aria-valuemax")).toBe("100")
  })

  it("does not show cv stability row when cv_stable is null", () => {
    render(<SampleSizeAdequacyCard result={makeResult({ cv_stable: null })} />)
    expect(screen.queryByTestId("cv-stability-row")).toBeNull()
  })

  it("shows stable cv badge when cv_stable is true", () => {
    render(
      <SampleSizeAdequacyCard
        result={makeResult({ cv_stable: true, cv_std: 0.05 })}
      />
    )
    expect(screen.getByTestId("cv-stability-row")).toBeTruthy()
    expect(screen.getByTestId("cv-stability-badge")).toHaveTextContent("Stable")
  })

  it("shows unstable cv badge when cv_stable is false", () => {
    render(
      <SampleSizeAdequacyCard
        result={makeResult({ cv_stable: false, cv_std: 0.18 })}
      />
    )
    expect(screen.getByTestId("cv-stability-badge")).toHaveTextContent("Unstable")
  })

  it("shows recommendations list", () => {
    render(
      <SampleSizeAdequacyCard
        result={makeResult({ recommendations: ["Collect 50 more rows.", "Use simpler model."] })}
      />
    )
    expect(screen.getByTestId("recommendations-list")).toBeTruthy()
    expect(screen.getByText(/Collect 50 more rows/)).toBeTruthy()
  })

  it("shows summary paragraph", () => {
    render(
      <SampleSizeAdequacyCard result={makeResult({ summary: "Adequate dataset for training." })} />
    )
    expect(screen.getByTestId("sample-size-summary")).toHaveTextContent(
      "Adequate dataset for training."
    )
  })

  it("renders sr-only figcaption for accessibility", () => {
    render(<SampleSizeAdequacyCard result={makeResult()} />)
    const fig = screen.getByTestId("sample-size-adequacy-card")
    expect(fig.querySelector("figcaption")).toBeTruthy()
  })

  it("returns null when result is falsy", () => {
    // @ts-expect-error testing null guard
    const { container } = render(<SampleSizeAdequacyCard result={null} />)
    expect(container.firstChild).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachSampleSizeAdequacyToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches result to last assistant message", () => {
    useAppStore.setState({
      messages: [{ role: "assistant", content: "test" }],
    })
    const result = makeResult()
    useAppStore.getState().attachSampleSizeAdequacyToLastMessage(result)
    const { messages } = useAppStore.getState()
    expect(messages[0].sample_size_adequacy).toEqual(result)
  })

  it("does not modify if last message is user", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "hi" }],
    })
    const result = makeResult()
    useAppStore.getState().attachSampleSizeAdequacyToLastMessage(result)
    const { messages } = useAppStore.getState()
    expect(messages[0].sample_size_adequacy).toBeUndefined()
  })

  it("does not crash with empty messages array", () => {
    useAppStore.setState({ messages: [] })
    expect(() => {
      useAppStore.getState().attachSampleSizeAdequacyToLastMessage(makeResult())
    }).not.toThrow()
  })
})
