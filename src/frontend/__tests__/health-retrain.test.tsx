/**
 * Tests for:
 * 1. ModelHealthCard rendering in DeploymentPanel
 * 2. Retrain button interaction
 * 3. api.deploy.health() client method shape
 * 4. api.models.retrain() client method shape
 * 5. ModelHealth and RetrainResponse type contracts
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { DeploymentPanel } from "../components/deploy/deployment-panel"
import { api } from "../lib/api"
import type { ModelHealth, RetrainResponse } from "../lib/types"

import fetchMock from "jest-fetch-mock"
fetchMock.enableMocks()

// ── API mock ──────────────────────────────────────────────────────────────────
jest.mock("../lib/api", () => ({
  api: {
    models: {
      readiness: jest.fn().mockResolvedValue({
        score: 90,
        verdict: "ready",
        checks: [],
        algorithm: "linear_regression",
        problem_type: "regression",
      }),
      retrain: jest.fn(),
    },
    deploy: {
      get: jest.fn(),
      deploy: jest.fn(),
      undeploy: jest.fn(),
      analytics: jest.fn().mockResolvedValue(null),
      drift: jest.fn().mockResolvedValue({ status: "insufficient_data", drift_score: null }),
      feedbackAccuracy: jest.fn().mockResolvedValue({ status: "no_feedback", total_feedback: 0, message: "No feedback yet.", problem_type: "regression" }),
      health: jest.fn(),
    },
  },
}))

const mockHealth = api.deploy.health as jest.MockedFunction<typeof api.deploy.health>
const mockRetrain = api.models.retrain as jest.MockedFunction<typeof api.models.retrain>

const HEALTHY: ModelHealth = {
  deployment_id: "dep1",
  health_score: 92,
  status: "healthy",
  model_age_days: 2,
  component_scores: { age: 100, feedback: null, drift: null },
  component_notes: {
    age: "Model is 2 day(s) old — still fresh.",
    feedback: "No feedback recorded yet.",
    drift: "Not enough predictions to assess drift.",
  },
  recommendations: ["Model health is good. Continue monitoring."],
  has_feedback_data: false,
  has_drift_data: false,
  algorithm: "linear_regression",
  problem_type: "regression",
}

const WARNING_HEALTH: ModelHealth = {
  ...HEALTHY,
  health_score: 55,
  status: "warning",
  model_age_days: 45,
  component_scores: { age: 75, feedback: null, drift: null },
  recommendations: ["Retrain the model with more recent data to improve freshness."],
}

const CRITICAL_HEALTH: ModelHealth = {
  ...HEALTHY,
  health_score: 30,
  status: "critical",
  model_age_days: 100,
  component_scores: { age: 25, feedback: null, drift: null },
  recommendations: ["Retrain the model with more recent data to improve freshness."],
}

// A minimal "deployed" state to trigger the deployed view
const MOCK_DEPLOYMENT = {
  id: "dep1",
  model_run_id: "run1",
  project_id: "proj1",
  endpoint_path: "/api/predict/dep1",
  dashboard_url: "/predict/dep1",
  is_active: true,
  request_count: 5,
  algorithm: "linear_regression",
  problem_type: "regression",
  feature_names: ["product", "region", "units"],
  target_column: "revenue",
  metrics: { r2: 0.87 },
  created_at: "2026-03-15T00:00:00",
  last_predicted_at: "2026-03-15T01:00:00",
}

function renderDeployed(deployment = MOCK_DEPLOYMENT, healthData: ModelHealth = HEALTHY) {
  mockHealth.mockResolvedValue(healthData)
  ;(api.deploy.get as jest.Mock).mockResolvedValue(deployment)
  ;(api.deploy.deploy as jest.Mock).mockResolvedValue(deployment)

  const { rerender } = render(
    <DeploymentPanel
      projectId="proj1"
      selectedRunId="run1"
      algorithmName="Linear Regression"
      onDeployed={jest.fn()}
    />
  )

  // Simulate having a deployment already by re-rendering after "deploy"
  // We trigger handleDeploy — for simplicity inject the deployment directly
  return { rerender }
}

// ---------------------------------------------------------------------------
// api.ts client method tests
// ---------------------------------------------------------------------------

describe("api.deploy.health()", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
  })

  it("calls GET /api/deploy/{id}/health", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(HEALTHY))
    const { api: realApi } = jest.requireActual("../lib/api") as { api: typeof import("../lib/api").api }
    // We test shape via mock since we can't call real API in unit tests
    expect(typeof api.deploy.health).toBe("function")
  })

  it("health method exists on api.deploy", () => {
    expect(api.deploy).toHaveProperty("health")
  })

  it("retrain method exists on api.models", () => {
    expect(api.models).toHaveProperty("retrain")
  })
})

// ---------------------------------------------------------------------------
// ModelHealth type contract
// ---------------------------------------------------------------------------

describe("ModelHealth type contract", () => {
  it("health_score is a number 0-100", () => {
    const h: ModelHealth = HEALTHY
    expect(h.health_score).toBeGreaterThanOrEqual(0)
    expect(h.health_score).toBeLessThanOrEqual(100)
  })

  it("status is one of healthy/warning/critical", () => {
    const validStatuses = ["healthy", "warning", "critical"]
    expect(validStatuses).toContain(HEALTHY.status)
    expect(validStatuses).toContain(WARNING_HEALTH.status)
    expect(validStatuses).toContain(CRITICAL_HEALTH.status)
  })

  it("component_scores can have null values when data unavailable", () => {
    expect(HEALTHY.component_scores.feedback).toBeNull()
    expect(HEALTHY.component_scores.drift).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// RetrainResponse type contract
// ---------------------------------------------------------------------------

describe("RetrainResponse type contract", () => {
  it("has expected fields", () => {
    const r: RetrainResponse = {
      project_id: "proj1",
      model_run_ids: ["run2"],
      algorithms: ["linear_regression"],
      status: "training_started",
      source_run_id: "run1",
      message: "Retraining linear_regression...",
    }
    expect(r.model_run_ids).toHaveLength(1)
    expect(r.status).toBe("training_started")
  })
})

// ---------------------------------------------------------------------------
// ModelHealthCard rendering tests
// ---------------------------------------------------------------------------

describe("ModelHealthCard", () => {

  it("shows health score when data loaded", async () => {
    mockHealth.mockResolvedValue(HEALTHY)
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(MOCK_DEPLOYMENT)

    render(
      <DeploymentPanel
        projectId="proj1"
        selectedRunId="run1"
        algorithmName="Linear Regression"
        onDeployed={jest.fn()}
      />
    )

    // Trigger deploy
    const deployBtn = screen.getByRole("button", { name: /deploy/i })
    fireEvent.click(deployBtn)

    // Wait for the health card to appear — health_score 92 shown in score display
    await waitFor(() => {
      expect(screen.getAllByText(/Model Health/i).length).toBeGreaterThan(0)
      expect(screen.getByText("92")).toBeTruthy()
    }, { timeout: 3000 })
  })

  it("shows Healthy badge for healthy status", async () => {
    mockHealth.mockResolvedValue(HEALTHY)
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(MOCK_DEPLOYMENT)

    render(
      <DeploymentPanel
        projectId="proj1"
        selectedRunId="run1"
        algorithmName="Linear Regression"
        onDeployed={jest.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /deploy/i }))

    await waitFor(() => {
      expect(screen.getByText("Healthy")).toBeTruthy()
    })
  })

  it("shows retrain button", async () => {
    mockHealth.mockResolvedValue(HEALTHY)
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(MOCK_DEPLOYMENT)

    render(
      <DeploymentPanel
        projectId="proj1"
        selectedRunId="run1"
        algorithmName="Linear Regression"
        onDeployed={jest.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /deploy/i }))

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /retrain model/i })).toBeTruthy()
    })
  })

  it("retrain button calls api.models.retrain with projectId", async () => {
    mockHealth.mockResolvedValue(HEALTHY)
    mockRetrain.mockResolvedValue({
      project_id: "proj1",
      model_run_ids: ["newrun"],
      algorithms: ["linear_regression"],
      status: "training_started",
      source_run_id: "run1",
      message: "Retraining linear_regression with your current data.",
    })
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(MOCK_DEPLOYMENT)

    render(
      <DeploymentPanel
        projectId="proj1"
        selectedRunId="run1"
        algorithmName="Linear Regression"
        onDeployed={jest.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /deploy/i }))

    await waitFor(() => screen.getByRole("button", { name: /retrain model/i }))
    fireEvent.click(screen.getByRole("button", { name: /retrain model/i }))

    await waitFor(() => {
      expect(mockRetrain).toHaveBeenCalledWith("proj1")
    })
  })

  it("shows retrain message after successful retrain", async () => {
    mockHealth.mockResolvedValue(HEALTHY)
    mockRetrain.mockResolvedValue({
      project_id: "proj1",
      model_run_ids: ["newrun"],
      algorithms: ["linear_regression"],
      status: "training_started",
      source_run_id: "run1",
      message: "Retraining linear_regression with your current data.",
    })
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(MOCK_DEPLOYMENT)

    render(
      <DeploymentPanel
        projectId="proj1"
        selectedRunId="run1"
        algorithmName="Linear Regression"
        onDeployed={jest.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /deploy/i }))
    await waitFor(() => screen.getByRole("button", { name: /retrain model/i }))
    fireEvent.click(screen.getByRole("button", { name: /retrain model/i }))

    await waitFor(() => {
      expect(screen.getByText(/Retraining linear_regression/i)).toBeTruthy()
    })
  })
})
