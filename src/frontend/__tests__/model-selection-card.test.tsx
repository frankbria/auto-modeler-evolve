import { render, screen } from "@testing-library/react"
import { ModelSelectionCard } from "@/components/models/model-selection-card"
import type { ModelSelectionResult } from "@/lib/types"

const makeRun = (
  id: string,
  algo: string,
  algoPlain: string,
  score: number,
  rank: number,
  opts: Partial<{
    is_selected: boolean
    is_deployed: boolean
    primary_metric: number
  }> = {}
) => ({
  run_id: id,
  algorithm: algo,
  algorithm_plain: algoPlain,
  score,
  primary_metric: opts.primary_metric ?? 0.8,
  primary_metric_name: "R²",
  component_scores: {
    accuracy: score,
    explainability: score * 0.8,
    stability: score * 0.9,
    speed: score * 0.7,
  },
  why: `${algoPlain} is a capable model.`,
  is_selected: opts.is_selected ?? false,
  is_deployed: opts.is_deployed ?? false,
  rank,
})

const multiRunResult: ModelSelectionResult = {
  project_id: "proj-1",
  criteria: "balanced",
  criteria_description: "Best overall — balances accuracy, explainability, and stability",
  winner: makeRun("r1", "random_forest_regressor", "Random Forest", 0.75, 1, {
    is_selected: true,
    primary_metric: 0.88,
  }),
  ranked_runs: [
    makeRun("r1", "random_forest_regressor", "Random Forest", 0.75, 1, {
      is_selected: true,
      primary_metric: 0.88,
    }),
    makeRun("r2", "linear_regression", "Linear Regression", 0.60, 2, {
      primary_metric: 0.72,
    }),
  ],
  summary: "Random Forest scores best overall (75/100) — a strong balance.",
  n_runs: 2,
}

const singleRunResult: ModelSelectionResult = {
  project_id: "proj-1",
  criteria: "accuracy",
  criteria_description: "Highest accuracy — the most predictively powerful model",
  winner: makeRun("r1", "xgboost_regressor", "XGBoost", 0.91, 1, {
    primary_metric: 0.91,
  }),
  ranked_runs: [
    makeRun("r1", "xgboost_regressor", "XGBoost", 0.91, 1, { primary_metric: 0.91 }),
  ],
  summary: "XGBoost achieves the highest R² of 91%.",
  n_runs: 1,
}

const noWinnerResult: ModelSelectionResult = {
  project_id: "proj-1",
  criteria: "balanced",
  criteria_description: "Best overall",
  winner: null,
  ranked_runs: [],
  summary: "No completed model runs to compare.",
  n_runs: 0,
}

// ---------------------------------------------------------------------------
// Render tests
// ---------------------------------------------------------------------------

test("renders accessible figure element", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getByRole("figure")).toBeInTheDocument()
})

test("shows Model Selection Recommendation heading", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getByText(/Model Selection Recommendation/i)).toBeInTheDocument()
})

test("shows criteria badge", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getAllByText(/Best Overall/i).length).toBeGreaterThanOrEqual(1)
})

test("shows accuracy criteria badge for accuracy criteria", () => {
  render(<ModelSelectionCard result={singleRunResult} />)
  expect(screen.getByText(/Highest Accuracy/i)).toBeInTheDocument()
})

test("shows run count badge", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getByText("2 models compared")).toBeInTheDocument()
})

test("shows singular run count", () => {
  render(<ModelSelectionCard result={singleRunResult} />)
  expect(screen.getByText("1 model compared")).toBeInTheDocument()
})

test("shows winner algorithm name in recommendation box", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  // Multiple "Random Forest" elements exist (winner box + ranked list) — at least one present
  expect(screen.getAllByText("Random Forest").length).toBeGreaterThanOrEqual(1)
})

test("shows winner why text", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getAllByText(/Random Forest is a capable model/i).length).toBeGreaterThanOrEqual(1)
})

test("shows summary sentence", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getByText(/Random Forest scores best overall/i)).toBeInTheDocument()
})

test("shows Recommended label above winner name", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getByText(/Recommended/i)).toBeInTheDocument()
})

test("shows all ranked runs when more than one", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getByText(/All models ranked/i)).toBeInTheDocument()
  expect(screen.getAllByRole("listitem").length).toBe(2)
})

test("does not show ranked list when only one run", () => {
  render(<ModelSelectionCard result={singleRunResult} />)
  expect(screen.queryByText(/All models ranked/i)).not.toBeInTheDocument()
})

test("shows trophy emoji for rank 1", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getByLabelText(/Rank 1/i)).toHaveTextContent("🏆")
})

test("shows is_selected badge", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getByText("Currently selected")).toBeInTheDocument()
})

test("shows is_deployed badge", () => {
  const withDeployed: ModelSelectionResult = {
    ...multiRunResult,
    ranked_runs: [
      makeRun("r1", "random_forest_regressor", "Random Forest", 0.75, 1, {
        is_deployed: true,
        primary_metric: 0.88,
      }),
      makeRun("r2", "linear_regression", "Linear Regression", 0.60, 2, {
        primary_metric: 0.72,
      }),
    ],
    winner: makeRun("r1", "random_forest_regressor", "Random Forest", 0.75, 1, {
      is_deployed: true,
      primary_metric: 0.88,
    }),
  }
  render(<ModelSelectionCard result={withDeployed} />)
  expect(screen.getByText("Deployed")).toBeInTheDocument()
})

test("shows component score bars for winner", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getByText("Accuracy")).toBeInTheDocument()
  expect(screen.getByText("Explainability")).toBeInTheDocument()
  expect(screen.getByText("Stability")).toBeInTheDocument()
  expect(screen.getByText("Speed")).toBeInTheDocument()
})

test("no-winner state shows empty message", () => {
  render(<ModelSelectionCard result={noWinnerResult} />)
  expect(screen.getByText(/No completed model runs to compare/i)).toBeInTheDocument()
})

test("aria-label on figure is descriptive", () => {
  render(<ModelSelectionCard result={multiRunResult} />)
  expect(screen.getByRole("figure")).toHaveAttribute(
    "aria-label",
    "Model selection advisor"
  )
})
