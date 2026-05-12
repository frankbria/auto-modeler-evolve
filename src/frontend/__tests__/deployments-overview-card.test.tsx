/**
 * Tests for DeploymentsOverviewCard component — Multi-Deployment Status Overview (Day 62).
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { DeploymentsOverviewCard } from "@/components/chat/deployments-overview-card"
import type { DeploymentsOverviewResult, DeploymentStatusRow } from "@/lib/types"

// ---------------------------------------------------------------------------
// Test data helpers
// ---------------------------------------------------------------------------

function makeDeployment(overrides: Partial<DeploymentStatusRow> = {}): DeploymentStatusRow {
  return {
    deployment_id: "dep-1",
    project_id: "proj-1",
    project_name: "Sales Project",
    name: "Sales Model",
    algorithm: "random_forest_regressor",
    algorithm_plain: "Random Forest",
    target_column: "revenue",
    environment: "staging",
    health_score: 80,
    status: "healthy",
    top_issue: null,
    recommendation: null,
    request_count: 150,
    predictions_last_7d: 42,
    predictions_today: 5,
    last_predicted_at_iso: null,
    api_key_enabled: false,
    rate_limit_rpm: null,
    monthly_quota: null,
    dashboard_url: null,
    endpoint_path: null,
    ...overrides,
  }
}

const EMPTY_RESULT: DeploymentsOverviewResult = {
  total_deployments: 0,
  production_count: 0,
  staging_count: 0,
  total_predictions: 0,
  avg_health_score: 0,
  healthy_count: 0,
  warning_count: 0,
  critical_count: 0,
  deployments: [],
  summary: "No active deployments found. Deploy a trained model to create a live prediction endpoint.",
}

const SINGLE_HEALTHY: DeploymentsOverviewResult = {
  total_deployments: 1,
  production_count: 0,
  staging_count: 1,
  total_predictions: 150,
  avg_health_score: 80,
  healthy_count: 1,
  warning_count: 0,
  critical_count: 0,
  deployments: [makeDeployment()],
  summary: "You have 1 active deployment. 150 total predictions served. All deployments healthy.",
}

const MIXED_RESULT: DeploymentsOverviewResult = {
  total_deployments: 3,
  production_count: 1,
  staging_count: 2,
  total_predictions: 5000,
  avg_health_score: 65,
  healthy_count: 1,
  warning_count: 1,
  critical_count: 1,
  deployments: [
    makeDeployment({ deployment_id: "d1", environment: "production", status: "healthy", health_score: 90 }),
    makeDeployment({ deployment_id: "d2", environment: "staging", status: "warning", health_score: 55, top_issue: "Model is 45 days old" }),
    makeDeployment({ deployment_id: "d3", environment: "staging", status: "critical", health_score: 20, top_issue: "No predictions in 90 days" }),
  ],
  summary: "You have 3 active deployments. 1 in production, 2 in staging. 5,000 total predictions served. 1 deployment needs attention.",
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

describe("DeploymentsOverviewCard — empty state", () => {
  it("renders without crashing", () => {
    render(<DeploymentsOverviewCard result={EMPTY_RESULT} />)
  })

  it("shows Active Deployments heading", () => {
    render(<DeploymentsOverviewCard result={EMPTY_RESULT} />)
    expect(screen.getByText("Active Deployments")).toBeInTheDocument()
  })

  it("shows 0 live badge", () => {
    render(<DeploymentsOverviewCard result={EMPTY_RESULT} />)
    expect(screen.getByText("0 live")).toBeInTheDocument()
  })

  it("shows empty state summary text", () => {
    render(<DeploymentsOverviewCard result={EMPTY_RESULT} />)
    expect(screen.getAllByText(/No active deployments found/i).length).toBeGreaterThan(0)
  })

  it("renders no deployment rows", () => {
    render(<DeploymentsOverviewCard result={EMPTY_RESULT} />)
    expect(screen.queryAllByTestId("deployment-row")).toHaveLength(0)
  })

  it("has accessible region label", () => {
    render(<DeploymentsOverviewCard result={EMPTY_RESULT} />)
    expect(screen.getByRole("region", { name: /No active deployments/i })).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Single healthy deployment
// ---------------------------------------------------------------------------

describe("DeploymentsOverviewCard — single healthy deployment", () => {
  it("shows 1 live badge", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getByText("1 live")).toBeInTheDocument()
  })

  it("shows healthy count badge", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getByTestId("healthy-badge")).toBeInTheDocument()
    expect(screen.getByTestId("healthy-badge")).toHaveTextContent("1 healthy")
  })

  it("does not show warning or critical badges", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.queryByTestId("warning-badge")).not.toBeInTheDocument()
    expect(screen.queryByTestId("critical-badge")).not.toBeInTheDocument()
  })

  it("renders one deployment row", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getAllByTestId("deployment-row")).toHaveLength(1)
  })

  it("shows algorithm → target column", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getByText(/Random Forest.*revenue/)).toBeInTheDocument()
  })

  it("shows project name", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getByText("Sales Project")).toBeInTheDocument()
  })

  it("shows staging environment badge", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getByText("Staging")).toBeInTheDocument()
  })

  it("shows Healthy status badge", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getByText("Healthy")).toBeInTheDocument()
  })

  it("shows total predictions stat", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getAllByText("total predictions").length).toBeGreaterThan(0)
  })

  it("shows health score percentage", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getByText("80% health")).toBeInTheDocument()
  })

  it("shows avg health score in stats row", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getByText("avg health")).toBeInTheDocument()
    expect(screen.getByText("80%")).toBeInTheDocument()
  })

  it("renders health bar with correct aria attributes", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-valuenow", "80")
    expect(bar).toHaveAttribute("aria-valuemin", "0")
    expect(bar).toHaveAttribute("aria-valuemax", "100")
  })

  it("shows summary text", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getByText(/all deployments healthy/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Mixed statuses (healthy + warning + critical)
// ---------------------------------------------------------------------------

describe("DeploymentsOverviewCard — mixed statuses", () => {
  it("shows correct live count", () => {
    render(<DeploymentsOverviewCard result={MIXED_RESULT} />)
    expect(screen.getByText("3 live")).toBeInTheDocument()
  })

  it("shows production count badge", () => {
    render(<DeploymentsOverviewCard result={MIXED_RESULT} />)
    expect(screen.getByText("1 production")).toBeInTheDocument()
  })

  it("shows all three status badges", () => {
    render(<DeploymentsOverviewCard result={MIXED_RESULT} />)
    expect(screen.getByTestId("healthy-badge")).toHaveTextContent("1 healthy")
    expect(screen.getByTestId("warning-badge")).toHaveTextContent("1 warning")
    expect(screen.getByTestId("critical-badge")).toHaveTextContent("1 critical")
  })

  it("renders all three deployment rows", () => {
    render(<DeploymentsOverviewCard result={MIXED_RESULT} />)
    expect(screen.getAllByTestId("deployment-row")).toHaveLength(3)
  })

  it("shows Warning status badge", () => {
    render(<DeploymentsOverviewCard result={MIXED_RESULT} />)
    expect(screen.getByText("Warning")).toBeInTheDocument()
  })

  it("shows Critical status badge", () => {
    render(<DeploymentsOverviewCard result={MIXED_RESULT} />)
    expect(screen.getByText("Critical")).toBeInTheDocument()
  })

  it("shows Production environment badge", () => {
    render(<DeploymentsOverviewCard result={MIXED_RESULT} />)
    expect(screen.getByText("Production")).toBeInTheDocument()
  })

  it("shows top_issue for warning deployment", () => {
    render(<DeploymentsOverviewCard result={MIXED_RESULT} />)
    expect(screen.getByText("Model is 45 days old")).toBeInTheDocument()
  })

  it("shows top_issue for critical deployment", () => {
    render(<DeploymentsOverviewCard result={MIXED_RESULT} />)
    expect(screen.getByText("No predictions in 90 days")).toBeInTheDocument()
  })

  it("has region accessible label for main overview", () => {
    render(<DeploymentsOverviewCard result={MIXED_RESULT} />)
    expect(screen.getByRole("region", { name: /Deployment status overview/i })).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// API key protected badge
// ---------------------------------------------------------------------------

describe("DeploymentsOverviewCard — API key protected", () => {
  it("shows Protected badge when api_key_enabled", () => {
    const result: DeploymentsOverviewResult = {
      ...SINGLE_HEALTHY,
      deployments: [makeDeployment({ api_key_enabled: true })],
    }
    render(<DeploymentsOverviewCard result={result} />)
    expect(screen.getByText("Protected")).toBeInTheDocument()
  })

  it("does not show Protected badge when api_key_enabled is false", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.queryByText("Protected")).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Predictions today
// ---------------------------------------------------------------------------

describe("DeploymentsOverviewCard — predictions today", () => {
  it("shows today count when > 0", () => {
    render(<DeploymentsOverviewCard result={SINGLE_HEALTHY} />)
    expect(screen.getByText("5")).toBeInTheDocument()
    expect(screen.getByText("today")).toBeInTheDocument()
  })

  it("hides today count when 0", () => {
    const result = {
      ...SINGLE_HEALTHY,
      deployments: [makeDeployment({ predictions_today: 0 })],
    }
    render(<DeploymentsOverviewCard result={result} />)
    expect(screen.queryByText("today")).not.toBeInTheDocument()
  })
})
