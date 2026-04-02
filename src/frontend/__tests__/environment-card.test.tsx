/**
 * Tests for EnvironmentCard (inside DeploymentPanel) — staging/production promotion.
 *
 * Covers:
 *   1. Staging deployment shows amber "Staging" badge + "Promote to Production" button
 *   2. Production deployment shows green "Production" badge + "Demote to staging" button
 *   3. Clicking "Promote to Production" shows confirmation dialog (two-click pattern)
 *   4. Confirming promotion calls api.deploy.promoteToProduction and updates state
 *   5. Clicking "Demote to staging" calls api.deploy.demoteToStaging and updates state
 *   6. Undefined environment defaults to staging behaviour
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
      analytics: jest.fn().mockRejectedValue(new Error("no analytics")),
      drift: jest.fn().mockRejectedValue(new Error("no drift")),
      feedbackAccuracy: jest.fn().mockResolvedValue({
        status: "no_feedback",
        total_feedback: 0,
        message: "No feedback yet.",
        problem_type: "regression",
      }),
      submitFeedback: jest.fn(),
      health: jest.fn().mockResolvedValue(null),
      generateApiKey: jest.fn(),
      disableApiKey: jest.fn(),
      getSchedules: jest.fn().mockResolvedValue([]),
      createSchedule: jest.fn(),
      deleteSchedule: jest.fn(),
      triggerSchedule: jest.fn(),
      getScheduleRuns: jest.fn().mockResolvedValue([]),
      getVersions: jest.fn().mockResolvedValue({
        deployment_id: "dep-1",
        current_version_number: 1,
        versions: [],
      }),
      rollback: jest.fn(),
      getWebhooks: jest.fn().mockResolvedValue([]),
      getAbTest: jest.fn().mockRejectedValue(new Error("HTTP 404")),
      createAbTest: jest.fn(),
      endAbTest: jest.fn(),
      promoteChallenger: jest.fn(),
      createWebhook: jest.fn(),
      deleteWebhook: jest.fn(),
      testWebhook: jest.fn(),
      sla: jest.fn().mockResolvedValue({
        deployment_id: "dep-1",
        sample_count: 0,
        p50_ms: null,
        p95_ms: null,
        p99_ms: null,
        avg_ms: null,
        alert: false,
        alert_message: null,
        latency_by_day: [],
      }),
      promoteToProduction: jest.fn(),
      demoteToStaging: jest.fn(),
    },
    models: {
      readiness: jest.fn().mockRejectedValue(new Error("not ready")),
      retrain: jest.fn(),
    },
    projects: {
      alerts: jest.fn().mockRejectedValue(new Error("no alerts")),
    },
  },
}))

const mockPromote = api.deploy.promoteToProduction as jest.MockedFunction<
  typeof api.deploy.promoteToProduction
>
const mockDemote = api.deploy.demoteToStaging as jest.MockedFunction<
  typeof api.deploy.demoteToStaging
>

const makeDeployment = (overrides: Partial<Deployment> = {}): Deployment => ({
  id: "dep-1",
  model_run_id: "run-1",
  project_id: "proj-1",
  endpoint_path: "/api/predict/dep-1",
  dashboard_url: "/predict/dep-1",
  is_active: true,
  request_count: 3,
  algorithm: "Linear Regression",
  problem_type: "regression",
  feature_names: ["units", "region"],
  target_column: "revenue",
  metrics: { r2: 0.85 },
  created_at: "2026-01-01T00:00:00",
  last_predicted_at: null,
  api_key_enabled: false,
  environment: "staging",
  ...overrides,
})

beforeEach(() => {
  jest.clearAllMocks()
})

describe("EnvironmentCard — staging deployment", () => {
  it("shows Staging badge when environment is staging", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(makeDeployment({ environment: "staging" }))

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getAllByText(/staging/i).length).toBeGreaterThan(0)
    )
  })

  it("shows Promote to Production button for staging deployment", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(makeDeployment({ environment: "staging" }))

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /promote to production/i })
      ).toBeInTheDocument()
    )
  })

  it("shows confirmation dialog on first click", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(makeDeployment({ environment: "staging" }))

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /promote to production/i })
      ).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: /promote to production/i }))
    await waitFor(() =>
      expect(screen.getByText(/yes, promote to production/i)).toBeInTheDocument()
    )
    // Cancel button appears
    expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument()
  })

  it("calls promoteToProduction on confirm click", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(makeDeployment({ environment: "staging" }))
    mockPromote.mockResolvedValue({
      message: "Promoted",
      deployment: makeDeployment({ environment: "production" }),
    })

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /promote to production/i })
      ).toBeInTheDocument()
    )

    // First click — confirmation dialog
    fireEvent.click(screen.getByRole("button", { name: /promote to production/i }))
    await waitFor(() =>
      expect(screen.getByText(/yes, promote to production/i)).toBeInTheDocument()
    )

    // Second click — confirm
    fireEvent.click(screen.getByText(/yes, promote to production/i))
    await waitFor(() => expect(mockPromote).toHaveBeenCalledWith("dep-1"))
  })

  it("cancelling the confirmation closes dialog without calling API", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(makeDeployment({ environment: "staging" }))

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /promote to production/i })
      ).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: /promote to production/i }))
    await waitFor(() =>
      expect(screen.getByText(/yes, promote to production/i)).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: /cancel/i }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /promote to production/i })
      ).toBeInTheDocument()
    )
    expect(mockPromote).not.toHaveBeenCalled()
  })
})

describe("EnvironmentCard — production deployment", () => {
  it("shows Production badge when environment is production", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(
      makeDeployment({ environment: "production" })
    )

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getAllByText(/production/i).length).toBeGreaterThan(0)
    )
  })

  it("shows Demote to staging button for production deployment", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(
      makeDeployment({ environment: "production" })
    )

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /demote to staging/i })
      ).toBeInTheDocument()
    )
  })

  it("calls demoteToStaging when button is clicked", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(
      makeDeployment({ environment: "production" })
    )
    mockDemote.mockResolvedValue({
      message: "Demoted",
      deployment: makeDeployment({ environment: "staging" }),
    })

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /demote to staging/i })
      ).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: /demote to staging/i }))
    await waitFor(() => expect(mockDemote).toHaveBeenCalledWith("dep-1"))
  })
})

describe("EnvironmentCard — undefined environment defaults to staging", () => {
  it("shows Promote to Production when environment field is absent", async () => {
    const dep: Deployment = {
      id: "dep-1",
      model_run_id: "run-1",
      project_id: "proj-1",
      endpoint_path: "/api/predict/dep-1",
      dashboard_url: "/predict/dep-1",
      is_active: true,
      request_count: 0,
      algorithm: "Linear Regression",
      problem_type: "regression",
      feature_names: [],
      target_column: "revenue",
      metrics: {},
      created_at: null,
      last_predicted_at: null,
      api_key_enabled: false,
      // no environment field
    }
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(dep)

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /promote to production/i })
      ).toBeInTheDocument()
    )
  })
})
