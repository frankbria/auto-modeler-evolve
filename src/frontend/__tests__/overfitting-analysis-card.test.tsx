import { render, screen } from "@testing-library/react"
import { OverfittingAnalysisCard } from "@/components/models/overfitting-analysis-card"
import type { OverfittingAnalysisResult } from "@/lib/types"

const BASE_RESULT: OverfittingAnalysisResult = {
  run_id: "run-001",
  algorithm: "random_forest_regressor",
  target_column: "revenue",
  problem_type: "regression",
  train_score: 0.95,
  cv_mean: 0.72,
  cv_std: 0.04,
  cv_scores: [0.70, 0.73, 0.71, 0.74, 0.72],
  gap: 0.23,
  gap_pct: 24.2,
  metric_key: "r2",
  metric_label: "R²",
  verdict: "overfit",
  verdict_label: "Overfitting",
  n_rows: 200,
  n_features: 5,
  algorithm_plain: "Random Forest",
  recommendations: [
    "The model has memorized the training data. Try a simpler algorithm.",
    "Consider feature selection to remove noisy or redundant columns.",
  ],
  summary:
    "Random Forest shows a train R² of 0.950 but CV R² of only 0.720 (gap: +0.230). This suggests the model has memorized training patterns.",
}

const WELL_FIT_RESULT: OverfittingAnalysisResult = {
  ...BASE_RESULT,
  train_score: 0.80,
  cv_mean: 0.78,
  cv_std: 0.02,
  cv_scores: [0.77, 0.79, 0.78, 0.80, 0.76],
  gap: 0.02,
  gap_pct: 2.5,
  verdict: "well_fit",
  verdict_label: "Well Fitted",
  recommendations: ["The model generalizes well. Train score and validation score are close."],
  summary: "Random Forest generalizes well — train R² 0.800 vs CV R² 0.780 (gap: +0.020).",
}

const MILD_OVERFIT_RESULT: OverfittingAnalysisResult = {
  ...BASE_RESULT,
  train_score: 0.88,
  cv_mean: 0.80,
  cv_std: 0.03,
  cv_scores: [0.79, 0.81, 0.80, 0.82, 0.78],
  gap: 0.08,
  gap_pct: 9.1,
  verdict: "mild_overfit",
  verdict_label: "Mild Overfitting",
  recommendations: [
    "Slight overfitting detected. Consider regularization or feature selection.",
    "Hyperparameter tuning may help.",
  ],
  summary: "Random Forest shows mild overfitting.",
}

const UNDERFIT_RESULT: OverfittingAnalysisResult = {
  ...BASE_RESULT,
  train_score: 0.25,
  cv_mean: 0.22,
  cv_std: 0.05,
  cv_scores: [0.20, 0.23, 0.22, 0.24, 0.21],
  gap: 0.03,
  gap_pct: 12.0,
  verdict: "underfit",
  verdict_label: "Underfitting",
  recommendations: [
    "Try a more complex algorithm (e.g., Random Forest or Gradient Boosting).",
  ],
  summary: "Random Forest is underfitting.",
}

describe("OverfittingAnalysisCard", () => {
  it("renders without crashing", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getByTestId("overfitting-analysis-card")).toBeInTheDocument()
  })

  it("returns null for falsy result", () => {
    const { container } = render(
      <OverfittingAnalysisCard result={null as unknown as OverfittingAnalysisResult} />
    )
    expect(container.firstChild).toBeNull()
  })

  it("shows 'Model Fit Analysis' heading", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getByText("Model Fit Analysis")).toBeInTheDocument()
  })

  it("shows overfit verdict badge", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Overfitting")
  })

  it("shows well_fit verdict badge", () => {
    render(<OverfittingAnalysisCard result={WELL_FIT_RESULT} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Well Fitted")
  })

  it("shows mild_overfit verdict badge", () => {
    render(<OverfittingAnalysisCard result={MILD_OVERFIT_RESULT} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Mild Overfitting")
  })

  it("shows underfit verdict badge", () => {
    render(<OverfittingAnalysisCard result={UNDERFIT_RESULT} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Underfitting")
  })

  it("shows algorithm plain name", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getByText("Random Forest")).toBeInTheDocument()
  })

  it("shows metric label badge", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getAllByText("R²").length).toBeGreaterThan(0)
  })

  it("shows train score", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getByTestId("train-score")).toHaveTextContent("0.950")
  })

  it("shows cv score", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getByTestId("cv-score")).toHaveTextContent("0.720")
  })

  it("shows positive gap value", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getByTestId("gap-value")).toHaveTextContent("+0.230")
  })

  it("shows well_fit near-zero gap", () => {
    render(<OverfittingAnalysisCard result={WELL_FIT_RESULT} />)
    expect(screen.getByTestId("gap-value")).toHaveTextContent("+0.020")
  })

  it("shows gap progressbar", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toBeInTheDocument()
    expect(bar).toHaveAttribute("aria-valuenow")
  })

  it("gap progressbar has aria label", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-label", expect.stringContaining("gap"))
  })

  it("shows recommendations list", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    const list = screen.getByTestId("recommendations-list")
    expect(list).toBeInTheDocument()
    expect(list.querySelectorAll("li").length).toBe(2)
  })

  it("shows recommendation text", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(
      screen.getByText(/The model has memorized the training data/)
    ).toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getByTestId("overfitting-summary")).toBeInTheDocument()
  })

  it("shows cv fold count from cv_scores length", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getByText(/5-fold/)).toBeInTheDocument()
  })

  it("shows accuracy format for classification", () => {
    const classResult: OverfittingAnalysisResult = {
      ...BASE_RESULT,
      problem_type: "classification",
      metric_key: "accuracy",
      metric_label: "Accuracy",
      train_score: 0.92,
      cv_mean: 0.75,
    }
    render(<OverfittingAnalysisCard result={classResult} />)
    // Accuracy shown as percentage
    expect(screen.getByTestId("train-score")).toHaveTextContent("92.0%")
    expect(screen.getByTestId("cv-score")).toHaveTextContent("75.0%")
  })

  it("has accessible sr-only figcaption", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    const figcaption = screen.getByText(/Model fit analysis for/)
    expect(figcaption).toHaveClass("sr-only")
  })

  it("figcaption mentions target column", () => {
    render(<OverfittingAnalysisCard result={BASE_RESULT} />)
    expect(screen.getByText(/revenue/)).toBeInTheDocument()
  })
})
