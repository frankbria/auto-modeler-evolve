/**
 * Tests for DeploymentPanel — the model deployment UI in the Deploy tab.
 *
 * Covers three render states:
 *   1. No model selected — prompt to select first
 *   2. Model selected but not deployed — "Ready to deploy" + Deploy button
 *   3. Model deployed — live dashboard URL, API endpoint, Undeploy button
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { DeploymentPanel } from "../components/deploy/deployment-panel"
import { api } from "../lib/api"
import type { Deployment } from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    deploy: {
      deploy: jest.fn(),
      undeploy: jest.fn(),
      analytics: jest.fn(),
      drift: jest.fn(),
      feedbackAccuracy: jest.fn().mockResolvedValue({
        status: "no_feedback",
        total_feedback: 0,
        message: "No feedback yet.",
        problem_type: "regression",
      }),
      submitFeedback: jest.fn(),
      health: jest.fn().mockResolvedValue(null),
    },
    models: {
      readiness: jest.fn(),
      retrain: jest.fn(),
    },
  },
}))

const mockDeploy = api.deploy.deploy as jest.MockedFunction<typeof api.deploy.deploy>
const mockUndeploy = api.deploy.undeploy as jest.MockedFunction<typeof api.deploy.undeploy>
const mockAnalytics = api.deploy.analytics as jest.MockedFunction<typeof api.deploy.analytics>
const mockDrift = api.deploy.drift as jest.MockedFunction<typeof api.deploy.drift>
const mockReadiness = api.models.readiness as jest.MockedFunction<typeof api.models.readiness>

const makeDeployment = (overrides: Partial<Deployment> = {}): Deployment => ({
  id: "dep-1",
  model_run_id: "run-1",
  project_id: "proj-1",
  endpoint_path: "/api/predict/dep-1",
  dashboard_url: "/predict/dep-1",
  is_active: true,
  request_count: 42,
  algorithm: "Random Forest",
  problem_type: "regression",
  feature_names: ["region", "price"],
  target_column: "revenue",
  metrics: { r2: 0.85 },
  created_at: "2026-01-01T00:00:00",
  last_predicted_at: "2026-01-02T12:00:00",
  ...overrides,
})

const makeAnalytics = (): import("../lib/types").DeploymentAnalytics => ({
  deployment_id: "dep-1",
  total_predictions: 0,
  predictions_by_day: [],
  prediction_distribution: [],
  recent_avg: null,
  class_counts: null,
  problem_type: "regression",
})

const makeReadiness = (overrides: Partial<import("../lib/types").ModelReadiness> = {}): import("../lib/types").ModelReadiness => ({
  model_run_id: "run-1",
  algorithm: "Random Forest",
  score: 80,
  verdict: "ready",
  summary: "Your model scores 80/100. Ready to deploy.",
  problem_type: "regression",
  checks: [
    { id: "training_complete", label: "Training completed", passed: true, weight: 10 },
    { id: "sufficient_data", label: "Sufficient data (20 rows)", passed: false, weight: 20 },
    { id: "model_accuracy", label: "R² = 0.85", passed: true, weight: 30 },
    { id: "features", label: "3 features used", passed: true, weight: 15 },
    { id: "data_quality", label: "Data quality (0.0% missing)", passed: true, weight: 15 },
    { id: "model_selected", label: "Marked as preferred model", passed: true, weight: 10 },
  ],
  ...overrides,
})

beforeEach(() => {
  jest.clearAllMocks()
  // Default: readiness, analytics, and drift reject silently (component handles gracefully)
  mockReadiness.mockRejectedValue(new Error("not ready"))
  mockAnalytics.mockRejectedValue(new Error("no analytics"))
  mockDrift.mockRejectedValue(new Error("no drift"))
})

describe("DeploymentPanel — no model selected", () => {
  it("shows a prompt to select a model when selectedRunId is null", () => {
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId={null}
        algorithmName={null}
      />
    )
    expect(screen.getByText(/select a model in the/i)).toBeInTheDocument()
  })
})

describe("DeploymentPanel — ready to deploy", () => {
  it("shows Deploy button when a run is selected but not yet deployed", () => {
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    expect(screen.getByRole("button", { name: /deploy model/i })).toBeInTheDocument()
  })

  it("shows algorithm name in description", () => {
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Gradient Boosting"
      />
    )
    expect(screen.getByText(/gradient boosting/i)).toBeInTheDocument()
  })

  it("shows 'selected model' when algorithmName is null", () => {
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName={null}
      />
    )
    expect(screen.getByText(/selected model/i)).toBeInTheDocument()
  })

  it("calls api.deploy.deploy on button click", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() => expect(mockDeploy).toHaveBeenCalledWith("run-1"))
  })

  it("calls onDeployed callback after successful deployment", async () => {
    const deployment = makeDeployment()
    mockDeploy.mockResolvedValueOnce(deployment)
    const onDeployed = jest.fn()
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
        onDeployed={onDeployed}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() => expect(onDeployed).toHaveBeenCalledWith(deployment))
  })

  it("shows error message when deployment fails", async () => {
    mockDeploy.mockRejectedValueOnce(new Error("Server error"))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText(/deployment failed/i)).toBeInTheDocument()
    )
  })

  it("disables Deploy button while deploying", async () => {
    mockDeploy.mockReturnValueOnce(new Promise(() => {}))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /deploying/i })).toBeDisabled()
    )
  })
})

describe("DeploymentPanel — deployed state", () => {
  it("shows 'Model deployed' status after successful deploy", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText(/model deployed/i)).toBeInTheDocument()
    )
  })

  it("shows algorithm badge in deployed view", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment({ algorithm: "XGBoost" }))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="XGBoost"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText("XGBoost")).toBeInTheDocument()
    )
  })

  it("shows request count in deployed view", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment({ request_count: 99 }))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText(/requests: 99/i)).toBeInTheDocument()
    )
  })

  it("shows Undeploy button in deployed view", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /undeploy/i })).toBeInTheDocument()
    )
  })

  it("calls api.deploy.undeploy and returns to ready state", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    // undeploy returns a Response — resolve with anything since the component just awaits it
    mockUndeploy.mockResolvedValueOnce(undefined as unknown as Response)
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /undeploy/i })).toBeInTheDocument()
    )
    fireEvent.click(screen.getByRole("button", { name: /undeploy/i }))
    await waitFor(() => expect(mockUndeploy).toHaveBeenCalledWith("dep-1"))
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /deploy model/i })).toBeInTheDocument()
    )
  })

  it("shows 'Copy link' button in deployed view", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /copy link/i })).toBeInTheDocument()
    )
  })

  it("calls clipboard.writeText when Copy link is clicked", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /copy link/i })).toBeInTheDocument()
    )
    fireEvent.click(screen.getByRole("button", { name: /copy link/i }))
    expect(navigator.clipboard.writeText).toHaveBeenCalled()
  })

  it("shows 'Unknown' algorithm badge when algorithm is null", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment({ algorithm: null }))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName={null}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText("Unknown")).toBeInTheDocument()
    )
  })

  it("does not show last_predicted_at when it is null", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment({ last_predicted_at: null }))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText(/model deployed/i)).toBeInTheDocument()
    )
    expect(screen.queryByText(/last used/i)).not.toBeInTheDocument()
  })
})

describe("DeploymentPanel — readiness card", () => {
  it("shows readiness card when readiness data is loaded", async () => {
    mockReadiness.mockResolvedValueOnce(makeReadiness())
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    await waitFor(() =>
      expect(screen.getByText(/model readiness/i)).toBeInTheDocument()
    )
  })

  it("shows readiness score", async () => {
    mockReadiness.mockResolvedValueOnce(makeReadiness({ score: 80 }))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    await waitFor(() =>
      expect(screen.getByText("80")).toBeInTheDocument()
    )
  })

  it("shows Ready to deploy badge for passing models", async () => {
    mockReadiness.mockResolvedValueOnce(makeReadiness({ verdict: "ready" }))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    await waitFor(() =>
      expect(screen.getByText(/ready to deploy/i)).toBeInTheDocument()
    )
  })

  it("shows Needs attention badge for borderline models", async () => {
    mockReadiness.mockResolvedValueOnce(makeReadiness({ verdict: "needs_attention" }))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    await waitFor(() =>
      expect(screen.getByText(/needs attention/i)).toBeInTheDocument()
    )
  })

  it("shows Not ready badge for poor models", async () => {
    mockReadiness.mockResolvedValueOnce(makeReadiness({ verdict: "not_ready" }))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    await waitFor(() =>
      expect(screen.getByText(/not ready/i)).toBeInTheDocument()
    )
  })

  it("shows check labels in readiness card", async () => {
    mockReadiness.mockResolvedValueOnce(makeReadiness())
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    await waitFor(() =>
      expect(screen.getByText(/training completed/i)).toBeInTheDocument()
    )
  })

  it("does not show readiness card when readiness fetch fails", async () => {
    mockReadiness.mockRejectedValue(new Error("model not done"))
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    // Small wait to ensure no crash
    await new Promise((r) => setTimeout(r, 50))
    expect(screen.queryByText(/model readiness/i)).not.toBeInTheDocument()
  })

  it("does not show readiness when no run selected", () => {
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId={null}
        algorithmName={null}
      />
    )
    expect(mockReadiness).not.toHaveBeenCalled()
  })
})

describe("DeploymentPanel — analytics card", () => {
  it("shows analytics card after deployment", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    mockAnalytics.mockResolvedValueOnce(makeAnalytics())
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText(/model deployed/i)).toBeInTheDocument()
    )
    await waitFor(() =>
      expect(screen.getByText(/usage analytics/i)).toBeInTheDocument()
    )
  })

  it("shows total prediction count in analytics", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    mockAnalytics.mockResolvedValueOnce({ ...makeAnalytics(), total_predictions: 17 })
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText(/total predictions/i)).toBeInTheDocument()
    )
    await waitFor(() =>
      expect(screen.getByText("17")).toBeInTheDocument()
    )
  })

  it("shows 'no predictions yet' when no data", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    mockAnalytics.mockResolvedValueOnce(makeAnalytics())
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText(/model deployed/i)).toBeInTheDocument()
    )
    await waitFor(() =>
      expect(screen.getByText(/no predictions yet/i)).toBeInTheDocument()
    )
  })

  it("shows average prediction value when available", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    mockAnalytics.mockResolvedValueOnce({
      ...makeAnalytics(),
      total_predictions: 5,
      recent_avg: 142.5,
      predictions_by_day: [{ date: "2026-03-15", count: 5 }],
    })
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText(/avg prediction/i)).toBeInTheDocument()
    )
  })
})

describe("DeploymentPanel — drift card", () => {
  it("shows DriftCard with stable status after deployment", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    mockAnalytics.mockResolvedValueOnce(makeAnalytics())
    mockDrift.mockResolvedValueOnce({
      deployment_id: "dep-1",
      status: "stable",
      drift_score: 5,
      explanation: "Prediction values are stable.",
      baseline_stats: null,
      recent_stats: null,
      baseline_dist: null,
      recent_dist: null,
      problem_type: "regression",
    })
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText(/prediction drift/i)).toBeInTheDocument()
    )
    await waitFor(() =>
      expect(screen.getByText("Stable")).toBeInTheDocument()
    )
  })

  it("shows significant drift badge when drift is high", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    mockAnalytics.mockResolvedValueOnce(makeAnalytics())
    mockDrift.mockResolvedValueOnce({
      deployment_id: "dep-1",
      status: "significant_drift",
      drift_score: 85,
      explanation: "Significant drift detected. Consider retraining.",
      baseline_stats: { mean: 1000, std: 50, count: 20 },
      recent_stats: { mean: 1500, std: 80, count: 20 },
      baseline_dist: null,
      recent_dist: null,
      problem_type: "regression",
    })
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText("Significant drift")).toBeInTheDocument()
    )
    expect(screen.getByText("Baseline")).toBeInTheDocument()
    expect(screen.getByText("Recent")).toBeInTheDocument()
  })

  it("shows insufficient data message when not enough predictions", async () => {
    mockDeploy.mockResolvedValueOnce(makeDeployment())
    mockAnalytics.mockResolvedValueOnce(makeAnalytics())
    mockDrift.mockResolvedValueOnce({
      deployment_id: "dep-1",
      status: "insufficient_data",
      drift_score: null,
      explanation: "Need at least 40 predictions (currently 0).",
      baseline_stats: null,
      recent_stats: null,
      baseline_dist: null,
      recent_dist: null,
      problem_type: "regression",
    })
    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Random Forest"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText("Insufficient data")).toBeInTheDocument()
    )
  })
})
