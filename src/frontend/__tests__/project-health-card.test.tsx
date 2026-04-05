import { render, screen, fireEvent } from "@testing-library/react"
import { ProjectHealthCard } from "@/components/chat/project-health-card"
import type { ProjectHealthSummary, DeploymentHealthItem } from "@/lib/types"

const freshItem: DeploymentHealthItem = {
  deployment_id: "dep-1",
  name: "Linear Regression → sales",
  algorithm_plain: "Linear Regression",
  target_column: "sales",
  environment: "production",
  health_score: 92,
  status: "healthy",
  top_issue: null,
  recommendation: null,
  age_score: 100,
  usage_score: 80,
}

const staleItem: DeploymentHealthItem = {
  deployment_id: "dep-2",
  name: "Random Forest → revenue",
  algorithm_plain: "Random Forest",
  target_column: "revenue",
  environment: "staging",
  health_score: 28,
  status: "critical",
  top_issue: "Model is 200 days old — patterns in your data may have changed.",
  recommendation: "Retrain with your most recent data to keep predictions accurate.",
  age_score: 20,
  usage_score: 40,
}

const warningItem: DeploymentHealthItem = {
  deployment_id: "dep-3",
  name: "XGBoost → churn",
  algorithm_plain: "XGBoost",
  target_column: "churn",
  environment: "staging",
  health_score: 60,
  status: "warning",
  top_issue: "No predictions in the last 45 days.",
  recommendation: "Check if the prediction URL is still being used by your team.",
  age_score: 80,
  usage_score: 30,
}

const healthySummary: ProjectHealthSummary = {
  project_id: "proj-1",
  total: 1,
  healthy: 1,
  warning: 0,
  critical: 0,
  alerts: [],
  all_items: [freshItem],
  overall_status: "healthy",
  summary: "All 1 deployed model is healthy.",
}

const criticalSummary: ProjectHealthSummary = {
  project_id: "proj-1",
  total: 2,
  healthy: 1,
  warning: 0,
  critical: 1,
  alerts: [staleItem],
  all_items: [freshItem, staleItem],
  overall_status: "critical",
  summary: "1 of 2 deployed models needs attention.",
}

const mixedSummary: ProjectHealthSummary = {
  project_id: "proj-1",
  total: 2,
  healthy: 0,
  warning: 1,
  critical: 1,
  alerts: [staleItem, warningItem],
  all_items: [staleItem, warningItem],
  overall_status: "critical",
  summary: "2 of 2 deployed models need attention.",
}

describe("ProjectHealthCard — rendering", () => {
  it("renders accessible figure with aria-label", () => {
    render(<ProjectHealthCard summary={criticalSummary} />)
    expect(screen.getByRole("figure", { name: /model health summary/i })).toBeInTheDocument()
  })

  it("shows 'Model Health: Healthy' heading for healthy summary", () => {
    render(<ProjectHealthCard summary={healthySummary} />)
    expect(screen.getByText(/Model Health: Healthy/i)).toBeInTheDocument()
  })

  it("shows 'Model Health: Action Required' for critical summary", () => {
    render(<ProjectHealthCard summary={criticalSummary} />)
    expect(screen.getByText(/Model Health: Action Required/i)).toBeInTheDocument()
  })

  it("shows 'Model Health: Needs Attention' for warning summary", () => {
    const warnSummary: ProjectHealthSummary = {
      ...healthySummary,
      total: 1,
      healthy: 0,
      warning: 1,
      critical: 0,
      alerts: [warningItem],
      all_items: [warningItem],
      overall_status: "warning",
      summary: "1 of 1 deployed models needs attention.",
    }
    render(<ProjectHealthCard summary={warnSummary} />)
    expect(screen.getByText(/Model Health: Needs Attention/i)).toBeInTheDocument()
  })

  it("shows deployment count badge", () => {
    render(<ProjectHealthCard summary={criticalSummary} />)
    expect(screen.getByText("2 deployments")).toBeInTheDocument()
  })

  it("shows healthy badge when there are healthy deployments", () => {
    render(<ProjectHealthCard summary={criticalSummary} />)
    expect(screen.getByText("1 healthy")).toBeInTheDocument()
  })

  it("shows critical badge when there are critical deployments", () => {
    render(<ProjectHealthCard summary={criticalSummary} />)
    expect(screen.getByText("1 critical")).toBeInTheDocument()
  })

  it("shows top_issue text for alert items", () => {
    render(<ProjectHealthCard summary={criticalSummary} />)
    expect(screen.getByText(/200 days old/i)).toBeInTheDocument()
  })

  it("shows recommendation text for alert items", () => {
    render(<ProjectHealthCard summary={criticalSummary} />)
    expect(screen.getByText(/Retrain with your most recent data/i)).toBeInTheDocument()
  })

  it("renders CTA buttons when onSwitchTab is provided", () => {
    const onSwitchTab = jest.fn()
    render(<ProjectHealthCard summary={criticalSummary} onSwitchTab={onSwitchTab} />)
    expect(screen.getByText("View Deployment")).toBeInTheDocument()
    expect(screen.getByText("Retrain Model")).toBeInTheDocument()
  })

  it("calls onSwitchTab('deploy') when View Deployment is clicked", () => {
    const onSwitchTab = jest.fn()
    render(<ProjectHealthCard summary={criticalSummary} onSwitchTab={onSwitchTab} />)
    fireEvent.click(screen.getByText("View Deployment"))
    expect(onSwitchTab).toHaveBeenCalledWith("deploy")
  })

  it("calls onSwitchTab('models') when Retrain Model is clicked", () => {
    const onSwitchTab = jest.fn()
    render(<ProjectHealthCard summary={criticalSummary} onSwitchTab={onSwitchTab} />)
    fireEvent.click(screen.getByText("Retrain Model"))
    expect(onSwitchTab).toHaveBeenCalledWith("models")
  })

  it("returns null for zero deployments", () => {
    const emptySummary: ProjectHealthSummary = {
      ...healthySummary,
      total: 0,
      all_items: [],
      alerts: [],
    }
    const { container } = render(<ProjectHealthCard summary={emptySummary} />)
    expect(container.firstChild).toBeNull()
  })

  it("shows multiple alert rows for mixed critical/warning summary", () => {
    render(<ProjectHealthCard summary={mixedSummary} />)
    // Both alert items should have their deployment names displayed
    expect(screen.getByText("Random Forest → revenue")).toBeInTheDocument()
    expect(screen.getByText("XGBoost → churn")).toBeInTheDocument()
  })
})
