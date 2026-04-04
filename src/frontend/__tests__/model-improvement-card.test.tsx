import { render, screen } from "@testing-library/react"
import { ModelImprovementCard } from "@/components/models/model-improvement-card"
import type { ModelImprovementResult } from "@/lib/types"

const regressionResult: ModelImprovementResult = {
  run_id: "run-1",
  project_id: "proj-1",
  algorithm: "random_forest_regressor",
  problem_type: "regression",
  primary_metric: 0.72,
  primary_metric_name: "R²",
  n_suggestions: 3,
  summary: "Your model explains 72% of variation — good predictive power. Top suggestion: remove 2 weak features.",
  suggestions: [
    {
      rank: 1,
      category: "features",
      title: "Remove 2 Weak Features",
      explanation: "Feature selection found 2 features with near-zero importance.",
      action: "feature_selection",
      difficulty: "easy",
      expected_impact: "moderate",
    },
    {
      rank: 2,
      category: "algorithm",
      title: "Try an Ensemble Model",
      explanation: "Voting or Stacking often outperforms any single algorithm.",
      action: "train_ensemble",
      difficulty: "medium",
      expected_impact: "high",
    },
    {
      rank: 3,
      category: "algorithm",
      title: "Tune Hyperparameters",
      explanation: "RandomizedSearchCV can find better settings automatically.",
      action: "hyperparameter_tuning",
      difficulty: "easy",
      expected_impact: "moderate",
    },
  ],
}

const classificationResult: ModelImprovementResult = {
  run_id: "run-2",
  project_id: "proj-1",
  algorithm: "random_forest_classifier",
  problem_type: "classification",
  primary_metric: 0.80,
  primary_metric_name: "accuracy",
  n_suggestions: 1,
  summary: "Your model achieves 80% accuracy — good performance.",
  suggestions: [
    {
      rank: 1,
      category: "reliability",
      title: "Calibrate Confidence Scores",
      explanation: "Calibration ensures confidence percentages are trustworthy.",
      action: "calibration",
      difficulty: "medium",
      expected_impact: "moderate",
    },
  ],
}

const noSuggestionsResult: ModelImprovementResult = {
  run_id: "run-3",
  project_id: "proj-1",
  algorithm: "random_forest_regressor",
  problem_type: "regression",
  primary_metric: 0.92,
  primary_metric_name: "R²",
  n_suggestions: 0,
  summary: "Your model explains 92% of variation — excellent predictive power. No obvious improvements detected.",
  suggestions: [],
}

describe("ModelImprovementCard", () => {
  it("renders header with suggestion count badge", () => {
    render(<ModelImprovementCard result={regressionResult} />)
    expect(screen.getByText(/Improvement Suggestions/i)).toBeInTheDocument()
    expect(screen.getByText(/3 suggestions/i)).toBeInTheDocument()
  })

  it("renders primary metric badge", () => {
    render(<ModelImprovementCard result={regressionResult} />)
    expect(screen.getByText(/72%.*R²/i)).toBeInTheDocument()
  })

  it("renders all suggestion titles", () => {
    render(<ModelImprovementCard result={regressionResult} />)
    // Use getAllByText because the title may also appear in the summary sentence
    expect(screen.getAllByText(/Remove 2 Weak Features/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/Try an Ensemble Model/i)).toBeInTheDocument()
    expect(screen.getByText(/Tune Hyperparameters/i)).toBeInTheDocument()
  })

  it("renders rank numbers for each suggestion", () => {
    render(<ModelImprovementCard result={regressionResult} />)
    expect(screen.getByText(/#1 Remove 2 Weak Features/i)).toBeInTheDocument()
    expect(screen.getByText(/#2 Try an Ensemble Model/i)).toBeInTheDocument()
    expect(screen.getByText(/#3 Tune Hyperparameters/i)).toBeInTheDocument()
  })

  it("renders difficulty badges", () => {
    render(<ModelImprovementCard result={regressionResult} />)
    const easyBadges = screen.getAllByText("Easy")
    expect(easyBadges.length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText("Medium")).toBeInTheDocument()
  })

  it("renders impact badges", () => {
    render(<ModelImprovementCard result={regressionResult} />)
    expect(screen.getByText("High impact")).toBeInTheDocument()
    const moderateBadges = screen.getAllByText("Moderate impact")
    expect(moderateBadges.length).toBeGreaterThanOrEqual(1)
  })

  it("renders summary text", () => {
    render(<ModelImprovementCard result={regressionResult} />)
    expect(
      screen.getByText(/Your model explains 72% of variation/i)
    ).toBeInTheDocument()
  })

  it("renders explanation text for each suggestion", () => {
    render(<ModelImprovementCard result={regressionResult} />)
    expect(
      screen.getByText(/Feature selection found 2 features with near-zero importance/i)
    ).toBeInTheDocument()
  })

  it("renders classification result correctly", () => {
    render(<ModelImprovementCard result={classificationResult} />)
    // metric badge
    expect(screen.getAllByText(/80%.*accuracy/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/1 suggestion/i)).toBeInTheDocument()
    expect(screen.getByText(/Calibrate Confidence Scores/i)).toBeInTheDocument()
  })

  it("renders no suggestions state", () => {
    render(<ModelImprovementCard result={noSuggestionsResult} />)
    expect(
      screen.getAllByText(/No obvious improvements detected/i).length
    ).toBeGreaterThan(0)
    expect(screen.getByText(/0 suggestions/i)).toBeInTheDocument()
  })

  it("renders legend text", () => {
    render(<ModelImprovementCard result={regressionResult} />)
    expect(screen.getByText(/Difficulty:/i)).toBeInTheDocument()
  })

  it("is accessible — has figure with aria-label", () => {
    const { container } = render(<ModelImprovementCard result={regressionResult} />)
    const figure = container.querySelector("figure")
    expect(figure).toBeInTheDocument()
    expect(figure?.getAttribute("aria-label")).toBe("Model improvement suggestions")
  })

  it("category icons are aria-hidden", () => {
    const { container } = render(<ModelImprovementCard result={regressionResult} />)
    const iconSpans = container.querySelectorAll("span[aria-hidden='true']")
    expect(iconSpans.length).toBeGreaterThan(0)
  })
})
