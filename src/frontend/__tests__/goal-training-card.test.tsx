import React from "react"
import { render, screen } from "@testing-library/react"
import { GoalTrainingCard } from "@/components/models/goal-training-card"
import type { GoalTrainingResult } from "@/lib/types"

const achievedResult: GoalTrainingResult = {
  project_id: "proj-1",
  target_col: "revenue",
  goal_metric: "r2",
  goal_target: 0.75,
  achieved: true,
  winner_algorithm: "random_forest_regressor",
  winner_algorithm_name: "Random Forest",
  winner_score: 0.82,
  trials: [
    { algorithm: "linear_regression", algorithm_name: "Linear Regression", score: 0.65, achieved_goal: false },
    { algorithm: "random_forest_regressor", algorithm_name: "Random Forest", score: 0.82, achieved_goal: true },
  ],
  tried_tuning: false,
  summary: "Goal achieved! Random Forest reached R² = 0.820 (target: 0.75). Tried 2 algorithms.",
}

const notAchievedResult: GoalTrainingResult = {
  project_id: "proj-2",
  target_col: "churn",
  goal_metric: "accuracy",
  goal_target: 0.95,
  achieved: false,
  winner_algorithm: "gradient_boosting_classifier",
  winner_algorithm_name: "Gradient Boosting",
  winner_score: 0.88,
  trials: [
    { algorithm: "logistic_regression", algorithm_name: "Logistic Regression", score: 0.80, achieved_goal: false },
    { algorithm: "random_forest_classifier", algorithm_name: "Random Forest", score: 0.85, achieved_goal: false },
    { algorithm: "gradient_boosting_classifier", algorithm_name: "Gradient Boosting", score: 0.88, achieved_goal: false },
    { algorithm: "gradient_boosting_classifier", algorithm_name: "Gradient Boosting (tuned)", score: 0.90, achieved_goal: false },
  ],
  tried_tuning: true,
  summary: "Best result: Gradient Boosting (tuned) reached accuracy = 90% (target was 95%). Tried 4 algorithms.",
}

describe("GoalTrainingCard", () => {
  it("renders figure with aria-label", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    expect(screen.getByRole("figure", { name: /goal-driven training result/i })).toBeInTheDocument()
  })

  it("shows 'Goal-Driven Training' heading", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    expect(screen.getByText("Goal-Driven Training")).toBeInTheDocument()
  })

  it("shows 'Goal Achieved ✓' badge when achieved", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    expect(screen.getByText("Goal Achieved ✓")).toBeInTheDocument()
  })

  it("shows 'Best Effort' badge when not achieved", () => {
    render(<GoalTrainingCard result={notAchievedResult} />)
    expect(screen.getByText("Best Effort")).toBeInTheDocument()
  })

  it("shows goal target badge for r2", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    expect(screen.getByText(/R²\s*≥\s*0\.75/)).toBeInTheDocument()
  })

  it("shows goal target badge for accuracy as percentage", () => {
    render(<GoalTrainingCard result={notAchievedResult} />)
    expect(screen.getByText(/Accuracy\s*≥\s*95%/)).toBeInTheDocument()
  })

  it("shows winner algorithm name", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    // "Random Forest" appears in winner box and trial table — both are acceptable
    expect(screen.getAllByText("Random Forest").length).toBeGreaterThan(0)
  })

  it("shows winner score for r2 in winner box", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    // Winner box contains "R² = 0.820"
    const matches = screen.getAllByText(/R²\s*=\s*0\.820/)
    expect(matches.length).toBeGreaterThan(0)
  })

  it("shows winner score for accuracy as percentage", () => {
    render(<GoalTrainingCard result={notAchievedResult} />)
    // 0.88 → 88.0%
    const matches = screen.getAllByText(/88\.0%/)
    expect(matches.length).toBeGreaterThan(0)
  })

  it("renders trials table with algorithm names", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    expect(screen.getByText("Linear Regression")).toBeInTheDocument()
  })

  it("shows checkmarks for achieved trials", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    const checks = screen.getAllByLabelText(/goal achieved/i)
    expect(checks.length).toBeGreaterThan(0)
  })

  it("shows cross for non-achieved trials", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    const crosses = screen.getAllByLabelText(/goal not achieved/i)
    expect(crosses.length).toBeGreaterThan(0)
  })

  it("shows tuning note when tried_tuning is true", () => {
    render(<GoalTrainingCard result={notAchievedResult} />)
    expect(screen.getByText(/hyperparameter tuning was also attempted/i)).toBeInTheDocument()
  })

  it("does not show tuning note when tried_tuning is false", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    expect(screen.queryByText(/hyperparameter tuning was also attempted/i)).not.toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<GoalTrainingCard result={achievedResult} />)
    expect(screen.getByText(/Goal achieved!/i)).toBeInTheDocument()
  })

  it("shows all trial algorithm names", () => {
    render(<GoalTrainingCard result={notAchievedResult} />)
    expect(screen.getByText("Logistic Regression")).toBeInTheDocument()
    expect(screen.getByText("Gradient Boosting (tuned)")).toBeInTheDocument()
  })
})
