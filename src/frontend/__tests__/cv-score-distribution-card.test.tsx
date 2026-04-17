import { render, screen } from "@testing-library/react"
import { CvScoreDistributionCard } from "@/components/models/cv-score-distribution-card"
import type { CvScoreDistributionResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const stableResult: CvScoreDistributionResult = {
  algorithm: "random_forest_regressor",
  algorithm_plain: "Random Forest Regressor",
  problem_type: "regression",
  metric: "r2",
  metric_plain: "R²",
  scores: [0.821, 0.834, 0.818, 0.829, 0.826],
  mean: 0.8256,
  std: 0.006,
  ci_low: 0.8138,
  ci_high: 0.8374,
  n_splits: 5,
  consistency: "stable",
  consistency_pct: 0.7,
  summary:
    "Across 5 folds, R² = 0.826 ± 0.006 (good, consistent). This tells us the model's performance is not just a fluke on one data split.",
}

const variableResult: CvScoreDistributionResult = {
  ...stableResult,
  scores: [0.92, 0.61, 0.88, 0.55, 0.79],
  mean: 0.75,
  std: 0.15,
  ci_low: 0.456,
  ci_high: 1.044,
  consistency: "variable",
  consistency_pct: 20.0,
  summary: "Across 5 folds, R² = 0.750 ± 0.150 (good, somewhat variable).",
}

const moderateResult: CvScoreDistributionResult = {
  ...stableResult,
  scores: [0.82, 0.74, 0.79, 0.71, 0.78],
  mean: 0.768,
  std: 0.041,
  consistency: "moderate",
  consistency_pct: 5.3,
  summary: "Across 5 folds, moderate consistency.",
}

describe("CvScoreDistributionCard", () => {
  it("renders the card with title and icon", () => {
    render(<CvScoreDistributionCard result={stableResult} />)
    expect(screen.getByTestId("cv-score-distribution-card")).toBeInTheDocument()
    expect(screen.getByText("Cross-Validation Scores")).toBeInTheDocument()
    expect(screen.getByText("📊")).toBeInTheDocument()
  })

  it("shows algorithm and problem type badges", () => {
    render(<CvScoreDistributionCard result={stableResult} />)
    expect(screen.getByText("Random Forest Regressor")).toBeInTheDocument()
    expect(screen.getByText("regression")).toBeInTheDocument()
  })

  it("shows Stable consistency badge for stable result", () => {
    render(<CvScoreDistributionCard result={stableResult} />)
    expect(screen.getByTestId("consistency-badge")).toHaveTextContent("Stable")
  })

  it("shows High Variance badge for variable result", () => {
    render(<CvScoreDistributionCard result={variableResult} />)
    expect(screen.getByTestId("consistency-badge")).toHaveTextContent("High Variance")
  })

  it("shows Moderate Variance badge for moderate result", () => {
    render(<CvScoreDistributionCard result={moderateResult} />)
    expect(screen.getByTestId("consistency-badge")).toHaveTextContent("Moderate Variance")
  })

  it("renders stats row with mean, std, and CoV", () => {
    render(<CvScoreDistributionCard result={stableResult} />)
    const statsRow = screen.getByTestId("cv-stats-row")
    expect(statsRow).toBeInTheDocument()
    expect(statsRow).toHaveTextContent("Mean R²")
    expect(statsRow).toHaveTextContent("0.826")
    expect(statsRow).toHaveTextContent("±0.006")
    expect(statsRow).toHaveTextContent("0.7%")
  })

  it("renders fold bars for each score", () => {
    render(<CvScoreDistributionCard result={stableResult} />)
    const bars = screen.getAllByTestId("fold-bar")
    expect(bars).toHaveLength(5)
  })

  it("shows CI range", () => {
    render(<CvScoreDistributionCard result={stableResult} />)
    expect(screen.getByTestId("cv-ci")).toHaveTextContent("95% CI: 0.814 – 0.837")
  })

  it("renders summary text", () => {
    render(<CvScoreDistributionCard result={stableResult} />)
    expect(screen.getByTestId("cv-summary")).toHaveTextContent(
      "not just a fluke on one data split"
    )
  })

  it("renders figcaption with stability guidance for stable", () => {
    render(<CvScoreDistributionCard result={stableResult} />)
    expect(screen.getByRole("figure")).toHaveTextContent("Low variance across folds")
  })

  it("renders figcaption with instability guidance for variable", () => {
    render(<CvScoreDistributionCard result={variableResult} />)
    expect(screen.getByRole("figure")).toHaveTextContent(
      "High variance across folds"
    )
  })
})

// ─── Zustand store tests ────────────────────────────────────────────────────

describe("Zustand store — attachCvScoreDistributionToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "how consistent is my model?" },
        { id: "2", role: "assistant", content: "Running cross-validation…" },
      ],
    })
    useAppStore.getState().attachCvScoreDistributionToLastMessage(stableResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].cv_score_distribution).toEqual(stableResult)
  })

  it("does not attach when last message is from user", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "Hello" },
        { id: "2", role: "user", content: "fold scores" },
      ],
    })
    useAppStore.getState().attachCvScoreDistributionToLastMessage(stableResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].cv_score_distribution).toBeUndefined()
    expect(msgs[0].cv_score_distribution).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    expect(() =>
      useAppStore.getState().attachCvScoreDistributionToLastMessage(stableResult),
    ).not.toThrow()
  })
})
