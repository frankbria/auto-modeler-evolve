/**
 * Tests for PortfolioCard component.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { PortfolioCard } from "@/components/chat/portfolio-card"
import type { PortfolioResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const PORTFOLIO_EMPTY: PortfolioResult = {
  total_projects: 0,
  active_deployments: 0,
  total_predictions: 0,
  best_performer: null,
  projects: [],
  summary: "No projects found. Create a project and upload some data to get started.",
}

const PORTFOLIO_SINGLE: PortfolioResult = {
  total_projects: 1,
  active_deployments: 1,
  total_predictions: 42,
  best_performer: {
    project_id: "proj-1",
    name: "Sales Forecast",
    metric_name: "r2",
    metric_value: 0.87,
    algorithm: "random_forest",
    problem_type: "regression",
    target_column: "revenue",
  },
  projects: [
    {
      project_id: "proj-1",
      name: "Sales Forecast",
      dataset_filename: "sales_data.csv",
      row_count: 500,
      model_count: 2,
      best_algorithm: "random_forest",
      best_metric_name: "r2",
      best_metric_value: 0.87,
      best_problem_type: "regression",
      best_target_column: "revenue",
      has_deployment: true,
      prediction_count: 42,
      last_activity_at: "2026-04-10T12:00:00",
    },
  ],
  summary: "You have 1 project. 1 live prediction API. 42 total predictions made. best model: Sales Forecast (Random Forest, 87% r2).",
}

const PORTFOLIO_MULTI: PortfolioResult = {
  total_projects: 3,
  active_deployments: 2,
  total_predictions: 150,
  best_performer: {
    project_id: "proj-2",
    name: "Churn Model",
    metric_name: "accuracy",
    metric_value: 0.92,
    algorithm: "gradient_boosting",
    problem_type: "classification",
    target_column: "churn",
  },
  projects: [
    {
      project_id: "proj-1",
      name: "Sales Forecast",
      dataset_filename: "sales.csv",
      row_count: 500,
      model_count: 2,
      best_algorithm: "random_forest",
      best_metric_name: "r2",
      best_metric_value: 0.75,
      best_problem_type: "regression",
      best_target_column: "revenue",
      has_deployment: true,
      prediction_count: 100,
      last_activity_at: "2026-04-10T12:00:00",
    },
    {
      project_id: "proj-2",
      name: "Churn Model",
      dataset_filename: "customers.csv",
      row_count: 1200,
      model_count: 3,
      best_algorithm: "gradient_boosting",
      best_metric_name: "accuracy",
      best_metric_value: 0.92,
      best_problem_type: "classification",
      best_target_column: "churn",
      has_deployment: true,
      prediction_count: 50,
      last_activity_at: "2026-04-09T08:00:00",
    },
    {
      project_id: "proj-3",
      name: "Demand Forecast",
      dataset_filename: "demand.csv",
      row_count: 300,
      model_count: 0,
      best_algorithm: null,
      best_metric_name: null,
      best_metric_value: null,
      best_problem_type: null,
      best_target_column: null,
      has_deployment: false,
      prediction_count: 0,
      last_activity_at: "2026-04-08T10:00:00",
    },
  ],
  summary: "You have 3 projects. 2 live prediction APIs. 150 total predictions made.",
}

describe("PortfolioCard", () => {
  it("renders with aria-label region", () => {
    render(<PortfolioCard result={PORTFOLIO_EMPTY} />)
    expect(screen.getByRole("region")).toBeInTheDocument()
  })

  it("renders portfolio heading", () => {
    render(<PortfolioCard result={PORTFOLIO_EMPTY} />)
    expect(screen.getByText("Model Portfolio")).toBeInTheDocument()
  })

  it("renders 0 projects badge", () => {
    render(<PortfolioCard result={PORTFOLIO_EMPTY} />)
    expect(screen.getByText(/0 projects/i)).toBeInTheDocument()
  })

  it("renders empty state message", () => {
    render(<PortfolioCard result={PORTFOLIO_EMPTY} />)
    expect(screen.getByText(/No projects yet/i)).toBeInTheDocument()
  })

  it("renders total projects badge for single project", () => {
    render(<PortfolioCard result={PORTFOLIO_SINGLE} />)
    expect(screen.getByText(/1 project$/)).toBeInTheDocument()
  })

  it("renders active deployments badge", () => {
    render(<PortfolioCard result={PORTFOLIO_SINGLE} />)
    expect(screen.getByText(/1 deployed/i)).toBeInTheDocument()
  })

  it("renders total predictions badge", () => {
    render(<PortfolioCard result={PORTFOLIO_SINGLE} />)
    // May appear in both header badge and project row
    const elements = screen.getAllByText(/42 prediction/i)
    expect(elements.length).toBeGreaterThanOrEqual(1)
  })

  it("renders best performer section with trophy icon", () => {
    render(<PortfolioCard result={PORTFOLIO_SINGLE} />)
    expect(screen.getByText("Best Performer")).toBeInTheDocument()
  })

  it("renders best performer project name", () => {
    render(<PortfolioCard result={PORTFOLIO_SINGLE} />)
    // Name appears in best performer box and project list
    const elements = screen.getAllByText(/Sales Forecast/)
    expect(elements.length).toBeGreaterThanOrEqual(1)
  })

  it("renders best performer metric percentage", () => {
    render(<PortfolioCard result={PORTFOLIO_SINGLE} />)
    // May appear in best performer box and project row badge
    const elements = screen.getAllByText(/87%/)
    expect(elements.length).toBeGreaterThanOrEqual(1)
  })

  it("renders per-project rows for multi-project portfolio", () => {
    render(<PortfolioCard result={PORTFOLIO_MULTI} />)
    // Each name may appear multiple times (best performer + project row)
    expect(screen.getAllByText(/Sales Forecast/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/Churn Model/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText("Demand Forecast").length).toBeGreaterThanOrEqual(1)
  })

  it("shows Live badge for deployed project", () => {
    render(<PortfolioCard result={PORTFOLIO_MULTI} />)
    const liveBadges = screen.getAllByText(/● Live/)
    expect(liveBadges.length).toBeGreaterThanOrEqual(1)
  })

  it("shows No model badge for unmodeled project", () => {
    render(<PortfolioCard result={PORTFOLIO_MULTI} />)
    expect(screen.getByText("No model")).toBeInTheDocument()
  })

  it("renders 3 projects badge", () => {
    render(<PortfolioCard result={PORTFOLIO_MULTI} />)
    // May appear in both header and summary text
    const elements = screen.getAllByText(/3 projects/)
    expect(elements.length).toBeGreaterThanOrEqual(1)
  })
})

// ---------------------------------------------------------------------------
// Store action
// ---------------------------------------------------------------------------

describe("attachPortfolioToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "show all my models", timestamp: "t1" },
        { role: "assistant", content: "Here is your portfolio.", timestamp: "t2" },
      ],
    })
  })

  it("attaches portfolio to last assistant message", () => {
    const { attachPortfolioToLastMessage } = useAppStore.getState()
    attachPortfolioToLastMessage(PORTFOLIO_SINGLE)
    const msgs = useAppStore.getState().messages
    expect((msgs[1] as { portfolio?: PortfolioResult }).portfolio).toBeDefined()
    expect((msgs[1] as { portfolio?: PortfolioResult }).portfolio?.total_projects).toBe(1)
  })

  it("does not attach portfolio to user message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "show all my models", timestamp: "t1" },
      ],
    })
    const { attachPortfolioToLastMessage } = useAppStore.getState()
    attachPortfolioToLastMessage(PORTFOLIO_SINGLE)
    const msgs = useAppStore.getState().messages
    expect((msgs[0] as { portfolio?: PortfolioResult }).portfolio).toBeUndefined()
  })
})
