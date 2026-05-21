/**
 * Tests for CrossProjectComparisonCard component.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { CrossProjectComparisonCard } from "@/components/chat/cross-project-comparison-card"
import type {
  CrossProjectComparisonResult,
  CrossProjectComparisonRow,
} from "@/lib/types"

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const makeRow = (
  overrides: Partial<CrossProjectComparisonRow> = {},
): CrossProjectComparisonRow => ({
  project_id: "proj-1",
  name: "Revenue Forecast",
  target_column: "revenue",
  algorithm: "random_forest",
  algorithm_plain: "Random Forest",
  problem_type: "regression",
  metric_name: "r2",
  metric_value: 0.88,
  performance_score: 88,
  has_deployment: false,
  prediction_count: 0,
  rank: 1,
  ...overrides,
})

const COMPARISON_EMPTY: CrossProjectComparisonResult = {
  n_projects: 0,
  n_with_models: 0,
  winner: null,
  projects_compared: [],
  insights: [],
  summary: "No trained models found across your projects.",
}

const COMPARISON_SINGLE: CrossProjectComparisonResult = {
  n_projects: 1,
  n_with_models: 1,
  winner: makeRow(),
  projects_compared: [makeRow()],
  insights: [],
  summary: "Compared 1 project with trained models. Top performer: 'Revenue Forecast'.",
}

const COMPARISON_MULTI: CrossProjectComparisonResult = {
  n_projects: 3,
  n_with_models: 2,
  winner: makeRow({ name: "Revenue Forecast", performance_score: 88, rank: 1 }),
  projects_compared: [
    makeRow({ name: "Revenue Forecast", performance_score: 88, rank: 1 }),
    makeRow({
      project_id: "proj-2",
      name: "Churn Model",
      algorithm: "gradient_boosting",
      algorithm_plain: "Gradient Boosting",
      problem_type: "classification",
      metric_name: "accuracy",
      metric_value: 0.72,
      performance_score: 72,
      has_deployment: true,
      prediction_count: 45,
      rank: 2,
    }),
  ],
  insights: [
    "'Revenue Forecast' edges out 'Churn Model' (88 vs 72 score).",
    "'Churn Model' has no live deployment yet.",
  ],
  summary: "Compared 2 projects with trained models. Top performer: 'Revenue Forecast' (Random Forest, 88% r2).",
}

// ---------------------------------------------------------------------------
// Rendering tests
// ---------------------------------------------------------------------------

describe("CrossProjectComparisonCard", () => {
  it("renders as a region with accessible label", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_EMPTY} />)
    expect(
      screen.getByRole("region", { name: /cross-project model comparison/i }),
    ).toBeInTheDocument()
  })

  it("renders the heading", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_EMPTY} />)
    expect(screen.getByText("Cross-Project Comparison")).toBeInTheDocument()
  })

  it("renders n_with_models badge", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_SINGLE} />)
    expect(screen.getByText(/1 model compared/i)).toBeInTheDocument()
  })

  it("renders plural badge when multiple models", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_MULTI} />)
    expect(screen.getByText(/2 models compared/i)).toBeInTheDocument()
  })

  it("renders 'without model' badge when some projects lack models", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_MULTI} />)
    // n_projects(3) - n_with_models(2) = 1 without model
    expect(screen.getByText(/1 without model/i)).toBeInTheDocument()
  })

  it("renders the summary text", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_SINGLE} />)
    expect(
      screen.getByText(/Compared 1 project with trained models/i),
    ).toBeInTheDocument()
  })

  it("renders empty state when no models", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_EMPTY} />)
    expect(screen.getByText(/Train models in your projects/i)).toBeInTheDocument()
  })

  it("renders top performer winner section", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_SINGLE} />)
    expect(screen.getByText("Top Performer")).toBeInTheDocument()
  })

  it("renders winner name in highlight box", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_SINGLE} />)
    // Name appears in both winner highlight and project row
    expect(screen.getAllByText("Revenue Forecast").length).toBeGreaterThanOrEqual(1)
  })

  it("renders winner score badge", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_SINGLE} />)
    // Score appears in both winner badge and project row
    expect(screen.getAllByText("88/100").length).toBeGreaterThanOrEqual(1)
  })

  it("does not render winner section when no winner", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_EMPTY} />)
    expect(screen.queryByText("Top Performer")).not.toBeInTheDocument()
  })

  it("renders all ranked project rows", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_MULTI} />)
    // "Revenue Forecast" appears in both the winner highlight and the project row
    expect(screen.getAllByText("Revenue Forecast").length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText("Churn Model")).toBeInTheDocument()
  })

  it("renders the score progress bar with aria attributes", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_SINGLE} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-valuenow", "88")
    expect(bar).toHaveAttribute("aria-valuemin", "0")
    expect(bar).toHaveAttribute("aria-valuemax", "100")
  })

  it("renders Live badge for deployed project", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_MULTI} />)
    expect(screen.getByText(/● Live/i)).toBeInTheDocument()
  })

  it("renders Not deployed badge for undeployed project", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_MULTI} />)
    expect(screen.getAllByText(/Not deployed/i).length).toBeGreaterThanOrEqual(1)
  })

  it("renders prediction count when > 0", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_MULTI} />)
    expect(screen.getByText(/45 pred/i)).toBeInTheDocument()
  })

  it("does not render prediction count when 0", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_SINGLE} />)
    // prediction count display is "{n} pred" — /\d+ pred/ avoids matching "predicting"
    expect(screen.queryByText(/\d+ pred/)).not.toBeInTheDocument()
  })

  it("renders problem type badge (Regression)", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_SINGLE} />)
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("renders problem type badge (Classification)", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_MULTI} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("renders rank medal for top 3", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_MULTI} />)
    expect(screen.getAllByText("🥇").length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText("🥈")).toBeInTheDocument()
  })

  it("renders insights section when insights present", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_MULTI} />)
    expect(screen.getByText("Insights")).toBeInTheDocument()
    expect(
      screen.getByText(/'Revenue Forecast' edges out/),
    ).toBeInTheDocument()
  })

  it("does not render insights section when empty", () => {
    render(<CrossProjectComparisonCard result={COMPARISON_SINGLE} />)
    expect(screen.queryByText("Insights")).not.toBeInTheDocument()
  })

  it("renders sr-only figcaption", () => {
    const { container } = render(
      <CrossProjectComparisonCard result={COMPARISON_SINGLE} />,
    )
    const caption = container.querySelector("figcaption.sr-only")
    expect(caption).toBeInTheDocument()
    expect(caption?.textContent).toMatch(/cross-project model comparison/i)
  })
})
