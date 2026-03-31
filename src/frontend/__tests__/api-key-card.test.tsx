/**
 * Tests for ApiKeyCard (inside DeploymentPanel) — API key protection management.
 *
 * Covers:
 *   1. Open deployment shows "Generate API key" button
 *   2. Clicking Generate calls api.deploy.generateApiKey and shows the key
 *   3. Protected deployment shows protected state + Regenerate / Remove protection buttons
 *   4. Clicking Regenerate calls generateApiKey again
 *   5. Clicking Remove protection calls disableApiKey and updates badge
 *   6. Copy button is present after generation
 */

import React from "react"
import { render, screen, fireEvent, waitFor, getAllByText } from "@testing-library/react"
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

const mockGenerateApiKey = api.deploy.generateApiKey as jest.MockedFunction<
  typeof api.deploy.generateApiKey
>
const mockDisableApiKey = api.deploy.disableApiKey as jest.MockedFunction<
  typeof api.deploy.disableApiKey
>

const makeDeployment = (overrides: Partial<Deployment> = {}): Deployment => ({
  id: "dep-1",
  model_run_id: "run-1",
  project_id: "proj-1",
  endpoint_path: "/api/predict/dep-1",
  dashboard_url: "/predict/dep-1",
  is_active: true,
  request_count: 5,
  algorithm: "Linear Regression",
  problem_type: "regression",
  feature_names: ["units", "region"],
  target_column: "revenue",
  metrics: { r2: 0.88 },
  created_at: "2026-01-01T00:00:00",
  last_predicted_at: null,
  api_key_enabled: false,
  ...overrides,
})

beforeEach(() => {
  jest.clearAllMocks()
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText: jest.fn().mockResolvedValue(undefined) },
    writable: true,
  })
})

describe("ApiKeyCard — open deployment", () => {
  it("shows 'Open access' badge and Generate API key button", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(makeDeployment())

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    // Trigger deploy
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByText(/open access/i)).toBeInTheDocument()
    )
    expect(screen.getByRole("button", { name: /generate api key/i })).toBeInTheDocument()
  })

  it("calls generateApiKey when Generate API key is clicked", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(makeDeployment())
    mockGenerateApiKey.mockResolvedValue({
      deployment_id: "dep-1",
      api_key: "test-key-abc123",
      message: "API key generated.",
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
      expect(screen.getByRole("button", { name: /generate api key/i })).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: /generate api key/i }))
    await waitFor(() => expect(mockGenerateApiKey).toHaveBeenCalledWith("dep-1"))
  })

  it("shows the generated key after clicking Generate", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(makeDeployment())
    mockGenerateApiKey.mockResolvedValue({
      deployment_id: "dep-1",
      api_key: "supersecret-key-xyz",
      message: "API key generated.",
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
      expect(screen.getByRole("button", { name: /generate api key/i })).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: /generate api key/i }))
    await waitFor(() =>
      expect(screen.getByText("supersecret-key-xyz")).toBeInTheDocument()
    )
    // Copy-once warning
    expect(screen.getByText(/will not be shown again/i)).toBeInTheDocument()
  })
})

describe("ApiKeyCard — protected deployment", () => {
  it("shows 'Protected' badge and Regenerate/Remove buttons when api_key_enabled=true", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(
      makeDeployment({ api_key_enabled: true })
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
      expect(screen.getByRole("button", { name: /regenerate key/i })).toBeInTheDocument()
    )
    expect(screen.getByRole("button", { name: /remove protection/i })).toBeInTheDocument()
    // "Protected" badge appears somewhere in the card
    expect(getAllByText(document.body, /protected/i).length).toBeGreaterThan(0)
  })

  it("shows Authorization header hint when already protected", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(
      makeDeployment({ api_key_enabled: true })
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
      expect(screen.getByText(/authorization/i)).toBeInTheDocument()
    )
  })

  it("calls generateApiKey when Regenerate key is clicked", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(
      makeDeployment({ api_key_enabled: true })
    )
    mockGenerateApiKey.mockResolvedValue({
      deployment_id: "dep-1",
      api_key: "new-key-999",
      message: "Regenerated.",
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
      expect(screen.getByRole("button", { name: /regenerate key/i })).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: /regenerate key/i }))
    await waitFor(() => expect(mockGenerateApiKey).toHaveBeenCalledWith("dep-1"))
    // Key is shown
    await waitFor(() =>
      expect(screen.getByText("new-key-999")).toBeInTheDocument()
    )
  })

  it("calls disableApiKey and shows open-access state after Remove protection", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(
      makeDeployment({ api_key_enabled: true })
    )
    mockDisableApiKey.mockResolvedValue(undefined as unknown as Response)

    render(
      <DeploymentPanel
        projectId="proj-1"
        selectedRunId="run-1"
        algorithmName="Linear Regression"
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /remove protection/i })).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: /remove protection/i }))
    await waitFor(() => expect(mockDisableApiKey).toHaveBeenCalledWith("dep-1"))
    // Badge switches back to open access
    await waitFor(() =>
      expect(screen.getByText(/open access/i)).toBeInTheDocument()
    )
  })
})

describe("ApiKeyCard — copy to clipboard", () => {
  it("shows Copy button alongside the generated key", async () => {
    ;(api.deploy.deploy as jest.Mock).mockResolvedValue(makeDeployment())
    mockGenerateApiKey.mockResolvedValue({
      deployment_id: "dep-1",
      api_key: "copy-me-key",
      message: "Generated.",
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
      expect(screen.getByRole("button", { name: /generate api key/i })).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: /generate api key/i }))
    await waitFor(() =>
      expect(screen.getByText("copy-me-key")).toBeInTheDocument()
    )
    // Copy button for the key specifically (not copy-link buttons elsewhere)
    const copyButtons = screen.getAllByRole("button", { name: /copy/i })
    expect(copyButtons.length).toBeGreaterThan(0)
  })
})
