/**
 * Tests for SlaMonitorCard — prediction latency monitoring UI.
 *
 * Covers:
 * - Empty state when sample_count = 0
 * - Shows p50/p95/p99 labels when data is present
 * - Shows "Healthy" badge when alert is false
 * - Shows "p95 > 500ms" alert badge when alert is true
 * - Shows alert message text when alert fires
 * - Renders latency sparkbar (aria-label) when latency_by_day is populated
 * - Shows sample count sentence
 * - DeploymentPanel fetches SLA after deploy and renders SlaMonitorCard
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { DeploymentPanel } from "../components/deploy/deployment-panel"
import { api } from "../lib/api"
import type { Deployment, SlaData } from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    deploy: {
      deploy: jest.fn(),
      undeploy: jest.fn(),
      analytics: jest.fn(),
      drift: jest.fn(),
      sla: jest.fn(),
      feedbackAccuracy: jest.fn().mockResolvedValue({
        status: "no_feedback",
        total_feedback: 0,
        message: "No feedback yet.",
        problem_type: "regression",
      }),
      submitFeedback: jest.fn(),
      health: jest.fn().mockResolvedValue(null),
      getSchedules: jest.fn().mockResolvedValue([]),
      createSchedule: jest.fn(),
      deleteSchedule: jest.fn(),
      triggerSchedule: jest.fn(),
      getScheduleRuns: jest.fn().mockResolvedValue([]),
      getVersions: jest.fn().mockResolvedValue({ deployment_id: "dep-1", current_version_number: 1, versions: [] }),
      rollback: jest.fn(),
      getWebhooks: jest.fn().mockResolvedValue([]),
      createWebhook: jest.fn(),
      deleteWebhook: jest.fn(),
      testWebhook: jest.fn(),
    },
    models: {
      readiness: jest.fn().mockRejectedValue(new Error("no readiness")),
      retrain: jest.fn(),
    },
  },
}))

const mockDeploy = api.deploy.deploy as jest.MockedFunction<typeof api.deploy.deploy>
const mockSla = api.deploy.sla as jest.MockedFunction<typeof api.deploy.sla>
const mockAnalytics = api.deploy.analytics as jest.MockedFunction<typeof api.deploy.analytics>
const mockDrift = api.deploy.drift as jest.MockedFunction<typeof api.deploy.drift>

const makeDeployment = (overrides: Partial<Deployment> = {}): Deployment => ({
  id: "dep-1",
  model_run_id: "run-1",
  project_id: "proj-1",
  endpoint_path: "/api/predict/dep-1",
  dashboard_url: "/predict/dep-1",
  is_active: true,
  request_count: 10,
  algorithm: "Linear Regression",
  problem_type: "regression",
  feature_names: ["region", "units"],
  target_column: "revenue",
  metrics: { r2: 0.9 },
  created_at: "2026-01-01T00:00:00",
  last_predicted_at: null,
  ...overrides,
})

const makeAnalytics = (): import("../lib/types").DeploymentAnalytics => ({
  deployment_id: "dep-1",
  total_predictions: 10,
  predictions_by_day: [],
  prediction_distribution: [],
  recent_avg: null,
  class_counts: null,
  problem_type: "regression",
})

const makeEmptySla = (): SlaData => ({
  deployment_id: "dep-1",
  sample_count: 0,
  p50_ms: null,
  p95_ms: null,
  p99_ms: null,
  avg_ms: null,
  alert: false,
  alert_message: null,
  latency_by_day: [],
})

const makeHealthySla = (): SlaData => ({
  deployment_id: "dep-1",
  sample_count: 25,
  p50_ms: 12.5,
  p95_ms: 48.3,
  p99_ms: 72.1,
  avg_ms: 18.4,
  alert: false,
  alert_message: null,
  latency_by_day: [
    { date: "2026-03-25", avg_ms: 15.2 },
    { date: "2026-03-26", avg_ms: 22.8 },
  ],
})

const makeAlertSla = (): SlaData => ({
  deployment_id: "dep-1",
  sample_count: 10,
  p50_ms: 450.0,
  p95_ms: 620.0,
  p99_ms: 890.0,
  avg_ms: 480.0,
  alert: true,
  alert_message:
    "p95 latency is 620.0ms — above the 500ms target. Consider retraining with fewer features or switching to a simpler algorithm.",
  latency_by_day: [{ date: "2026-03-26", avg_ms: 480.0 }],
})

async function deployAndWait(slaData: SlaData) {
  mockAnalytics.mockResolvedValue(makeAnalytics())
  mockDrift.mockRejectedValue(new Error("no drift"))
  mockSla.mockResolvedValue(slaData)
  mockDeploy.mockResolvedValueOnce(makeDeployment())

  render(
    <DeploymentPanel
      projectId="proj-1"
      selectedRunId="run-1"
      algorithmName="Linear Regression"
    />
  )

  fireEvent.click(screen.getByRole("button", { name: /deploy model/i }))
  // Wait for deploy to complete and SLA to load
  await waitFor(() => expect(mockSla).toHaveBeenCalledWith("dep-1"))
}

beforeEach(() => {
  jest.clearAllMocks()
  mockAnalytics.mockRejectedValue(new Error("no analytics"))
  mockDrift.mockRejectedValue(new Error("no drift"))
  mockSla.mockRejectedValue(new Error("no sla"))
})

describe("SlaMonitorCard — empty state (no timed predictions)", () => {
  it("shows no-timing-data message", async () => {
    await deployAndWait(makeEmptySla())
    await waitFor(() => {
      expect(screen.getAllByText(/No timing data yet/i).length).toBeGreaterThan(0)
    })
  })
})

describe("SlaMonitorCard — healthy latency", () => {
  it("displays p50/p95/p99 labels", async () => {
    await deployAndWait(makeHealthySla())
    await waitFor(() => {
      expect(screen.getByText("p50")).toBeInTheDocument()
      expect(screen.getByText("p95")).toBeInTheDocument()
      expect(screen.getByText("p99")).toBeInTheDocument()
    })
  })

  it("displays the Healthy badge", async () => {
    await deployAndWait(makeHealthySla())
    await waitFor(() => {
      expect(screen.getByText("Healthy")).toBeInTheDocument()
    })
  })

  it("does not show the alert badge when healthy", async () => {
    await deployAndWait(makeHealthySla())
    await waitFor(() => {
      expect(screen.getByText("p50")).toBeInTheDocument()
    })
    expect(screen.queryByText(/p95.*500ms/i)).not.toBeInTheDocument()
  })

  it("shows the sample count sentence", async () => {
    await deployAndWait(makeHealthySla())
    await waitFor(() => {
      expect(screen.getByText(/Based on 25 timed predictions/i)).toBeInTheDocument()
    })
  })

  it("renders the latency sparkbar", async () => {
    await deployAndWait(makeHealthySla())
    await waitFor(() => {
      expect(
        screen.getByRole("img", { name: /Avg latency last 7 days/i })
      ).toBeInTheDocument()
    })
  })

  it("shows avg latency value", async () => {
    await deployAndWait(makeHealthySla())
    await waitFor(() => {
      expect(screen.getAllByText(/18\.4ms/i).length).toBeGreaterThan(0)
    })
  })
})

describe("SlaMonitorCard — alert state (p95 > 500ms)", () => {
  it("shows the p95 > 500ms alert badge", async () => {
    await deployAndWait(makeAlertSla())
    await waitFor(() => {
      expect(screen.getAllByText(/p95.*500ms/i).length).toBeGreaterThan(0)
    })
  })

  it("shows the alert message text", async () => {
    await deployAndWait(makeAlertSla())
    await waitFor(() => {
      expect(screen.getAllByText(/above the 500ms target/i).length).toBeGreaterThan(0)
    })
  })

  it("does not show Healthy badge when alert is active", async () => {
    await deployAndWait(makeAlertSla())
    await waitFor(() => {
      expect(screen.getAllByText(/p95.*500ms/i).length).toBeGreaterThan(0)
    })
    expect(screen.queryByText("Healthy")).not.toBeInTheDocument()
  })
})

describe("DeploymentPanel SLA integration", () => {
  it("calls api.deploy.sla with the deployment id after deploy", async () => {
    await deployAndWait(makeEmptySla())
    expect(mockSla).toHaveBeenCalledWith("dep-1")
  })
})
