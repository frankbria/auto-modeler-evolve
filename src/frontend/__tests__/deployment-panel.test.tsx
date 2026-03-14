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
    },
  },
}))

const mockDeploy = api.deploy.deploy as jest.MockedFunction<typeof api.deploy.deploy>
const mockUndeploy = api.deploy.undeploy as jest.MockedFunction<typeof api.deploy.undeploy>

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

beforeEach(() => {
  jest.clearAllMocks()
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
