import React from "react"
import { render, screen } from "@testing-library/react"
import { ModelComparisonSummaryCard } from "@/components/chat/model-comparison-summary-card"
import type { ModelComparisonSummaryResult, ModelComparisonRunSummary } from "@/lib/types"

function makeRun(overrides: Partial<ModelComparisonRunSummary> = {}): ModelComparisonRunSummary {
  return {
    run_id: "run-1",
    algorithm: "random_forest_regressor",
    algorithm_plain: "Random Forest",
    primary_metric: 0.88,
    primary_metric_name: "R²",
    primary_metric_pct: 88.0,
    cv_mean: 0.86,
    cv_std: 0.03,
    explainability_rank: 6,
    explainability_label: "Medium",
    speed_rank: 5,
    speed_label: "Fast",
    is_selected: false,
    is_deployed: false,
    secondary: { mae: 30.5 },
    ...overrides,
  }
}

function makeResult(overrides: Partial<ModelComparisonSummaryResult> = {}): ModelComparisonSummaryResult {
  const winner = makeRun()
  return {
    n_runs: 2,
    winner,
    runs_compared: [
      winner,
      makeRun({
        run_id: "run-2",
        algorithm: "linear_regression",
        algorithm_plain: "Linear Regression",
        primary_metric: 0.72,
        primary_metric_pct: 72.0,
        explainability_rank: 1,
        explainability_label: "Very high",
        speed_rank: 1,
        speed_label: "Very fast",
        cv_mean: null,
        cv_std: null,
        secondary: {},
      }),
    ],
    trade_offs: ["Random Forest edges out Linear Regression by 16 percentage points."],
    narrative: "Random Forest leads with 88.0% R². Linear Regression is the most interpretable alternative.",
    summary: "Random Forest is the top performer across 2 runs.",
    problem_type: "regression",
    only_one_run: false,
    ...overrides,
  }
}

describe("ModelComparisonSummaryCard", () => {
  it("renders nothing when n_runs is 0", () => {
    const { container } = render(
      <ModelComparisonSummaryCard result={makeResult({ n_runs: 0, winner: null, runs_compared: [] })} />
    )
    expect(container.firstChild).toBeNull()
  })

  it("shows header with run count and problem type", () => {
    render(<ModelComparisonSummaryCard result={makeResult()} />)
    expect(screen.getByText(/2 trained models · regression/i)).toBeInTheDocument()
  })

  it("renders narrative text", () => {
    render(<ModelComparisonSummaryCard result={makeResult()} />)
    expect(screen.getAllByText(/Random Forest leads with 88/)[0]).toBeInTheDocument()
  })

  it("renders comparison table with algorithm names", () => {
    render(<ModelComparisonSummaryCard result={makeResult()} />)
    expect(screen.getByText("Random Forest")).toBeInTheDocument()
    expect(screen.getByText("Linear Regression")).toBeInTheDocument()
  })

  it("shows winner checkmark on first row", () => {
    render(<ModelComparisonSummaryCard result={makeResult()} />)
    expect(screen.getByLabelText("Winner")).toBeInTheDocument()
  })

  it("shows metric percentages in table", () => {
    render(<ModelComparisonSummaryCard result={makeResult()} />)
    expect(screen.getByText("88.0%")).toBeInTheDocument()
    expect(screen.getByText("72.0%")).toBeInTheDocument()
  })

  it("shows CV values for runs that have them", () => {
    render(<ModelComparisonSummaryCard result={makeResult()} />)
    expect(screen.getByText(/86\.0%.*±3\.0/)).toBeInTheDocument()
  })

  it("shows dash for runs without CV", () => {
    render(<ModelComparisonSummaryCard result={makeResult()} />)
    expect(screen.getByText("—")).toBeInTheDocument()
  })

  it("renders trade-offs section", () => {
    render(<ModelComparisonSummaryCard result={makeResult()} />)
    expect(screen.getByText("Key trade-offs")).toBeInTheDocument()
    expect(screen.getAllByText(/edges out/i)[0]).toBeInTheDocument()
  })

  it("omits trade-offs section when list is empty", () => {
    render(<ModelComparisonSummaryCard result={makeResult({ trade_offs: [] })} />)
    expect(screen.queryByText("Key trade-offs")).not.toBeInTheDocument()
  })

  it("renders summary footer", () => {
    render(<ModelComparisonSummaryCard result={makeResult()} />)
    expect(screen.getByText(/top performer across 2 runs/i)).toBeInTheDocument()
  })

  it("shows (selected) badge when is_selected", () => {
    const winner = makeRun({ is_selected: true })
    render(<ModelComparisonSummaryCard result={makeResult({ winner, runs_compared: [winner, makeRun({ run_id: "run-2" })] })} />)
    expect(screen.getByText("(selected)")).toBeInTheDocument()
  })

  it("shows (live) badge when is_deployed", () => {
    const winner = makeRun({ is_deployed: true })
    render(<ModelComparisonSummaryCard result={makeResult({ winner, runs_compared: [winner, makeRun({ run_id: "run-2" })] })} />)
    expect(screen.getByText("(live)")).toBeInTheDocument()
  })

  it("renders single-run result", () => {
    const winner = makeRun()
    render(
      <ModelComparisonSummaryCard
        result={makeResult({
          n_runs: 1,
          runs_compared: [winner],
          trade_offs: [],
          narrative: "Random Forest is the only trained model.",
          summary: "Train more models to enable comparison.",
          only_one_run: true,
        })}
      />
    )
    expect(screen.getByText(/1 trained model/i)).toBeInTheDocument()
    expect(screen.getByText(/Train more/i)).toBeInTheDocument()
  })

  it("renders classification result", () => {
    const winner = makeRun({ primary_metric_name: "accuracy", primary_metric_pct: 91.0 })
    render(
      <ModelComparisonSummaryCard
        result={makeResult({
          winner,
          runs_compared: [winner],
          problem_type: "classification",
        })}
      />
    )
    expect(screen.getByText(/classification/i)).toBeInTheDocument()
    // "accuracy" appears as both the metric badge and table header — just check presence
    expect(screen.getAllByText("accuracy").length).toBeGreaterThan(0)
  })
})
