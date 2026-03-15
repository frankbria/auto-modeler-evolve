/**
 * Tests for AlertsCard — the model monitoring alerts UI component.
 *
 * Day 4 (10:00): Phase 8 Track B — proactive system-wide health alerts.
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { AlertsCard } from "../components/deploy/deployment-panel"
import { api } from "../lib/api"
import type { ProjectAlerts } from "../lib/types"

jest.mock("../lib/api", () => ({
  api: {
    projects: {
      alerts: jest.fn(),
    },
  },
}))

const mockAlerts = api.projects.alerts as jest.MockedFunction<typeof api.projects.alerts>

const makeAlerts = (overrides: Partial<ProjectAlerts> = {}): ProjectAlerts => ({
  project_id: "proj-1",
  alert_count: 0,
  critical_count: 0,
  warning_count: 0,
  alerts: [],
  ...overrides,
})

describe("AlertsCard", () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it("renders a 'Check for Alerts' button initially when no externalAlerts", () => {
    render(<AlertsCard projectId="proj-1" />)
    expect(screen.getByRole("button", { name: /check for alerts/i })).toBeInTheDocument()
  })

  it("calls api.projects.alerts when button is clicked", async () => {
    mockAlerts.mockResolvedValueOnce(makeAlerts())
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    await waitFor(() => expect(mockAlerts).toHaveBeenCalledWith("proj-1"))
  })

  it("shows loading state while fetching", async () => {
    let resolve: (v: ProjectAlerts) => void
    mockAlerts.mockReturnValueOnce(new Promise((r) => { resolve = r }))
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    expect(screen.getByRole("button", { name: /scanning/i })).toBeInTheDocument()
    resolve!(makeAlerts())
  })

  it("shows 'All clear' badge when no alerts", async () => {
    mockAlerts.mockResolvedValueOnce(makeAlerts())
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    await waitFor(() => expect(screen.getByText(/all clear/i)).toBeInTheDocument())
  })

  it("shows 'No issues detected' text when no alerts", async () => {
    mockAlerts.mockResolvedValueOnce(makeAlerts())
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    await waitFor(() =>
      expect(screen.getByText(/no issues detected/i)).toBeInTheDocument()
    )
  })

  it("renders warning alert correctly", async () => {
    const alerts = makeAlerts({
      alert_count: 1,
      warning_count: 1,
      alerts: [
        {
          deployment_id: "dep-1",
          algorithm: "Random Forest",
          severity: "warning",
          type: "no_predictions",
          message: "'Random Forest' has been deployed 2 day(s) with no predictions.",
          recommendation: "Share the dashboard link.",
        },
      ],
    })
    mockAlerts.mockResolvedValueOnce(alerts)
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    await waitFor(() => {
      expect(screen.getByText(/Random Forest/)).toBeInTheDocument()
      expect(screen.getByText(/no predictions/i)).toBeInTheDocument()
    })
  })

  it("renders critical alert with 'Critical' badge", async () => {
    const alerts = makeAlerts({
      alert_count: 1,
      critical_count: 1,
      alerts: [
        {
          deployment_id: "dep-1",
          algorithm: "XGBoost",
          severity: "critical",
          type: "stale_model",
          message: "Model 'XGBoost' is 95 days old.",
          recommendation: "Retrain immediately.",
        },
      ],
    })
    mockAlerts.mockResolvedValueOnce(alerts)
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    await waitFor(() => {
      expect(screen.getAllByText(/critical/i).length).toBeGreaterThan(0)
    })
  })

  it("shows warning count badge", async () => {
    const alerts = makeAlerts({
      alert_count: 2,
      warning_count: 2,
      alerts: [
        {
          deployment_id: "dep-1",
          algorithm: "RF",
          severity: "warning",
          type: "no_predictions",
          message: "No predictions",
          recommendation: "Share link",
        },
        {
          deployment_id: "dep-2",
          algorithm: "LR",
          severity: "warning",
          type: "stale_model",
          message: "Stale model",
          recommendation: "Retrain",
        },
      ],
    })
    mockAlerts.mockResolvedValueOnce(alerts)
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    await waitFor(() => {
      expect(screen.getByText(/2 warning/i)).toBeInTheDocument()
    })
  })

  it("renders externalAlerts immediately without clicking load button", async () => {
    const externalAlerts = makeAlerts({
      alert_count: 1,
      critical_count: 1,
      alerts: [
        {
          deployment_id: "dep-1",
          algorithm: "GBT",
          severity: "critical",
          type: "drift_detected",
          message: "Drift score 85/100",
          recommendation: "Retrain now",
        },
      ],
    })
    render(<AlertsCard projectId="proj-1" externalAlerts={externalAlerts} />)
    await waitFor(() => {
      expect(screen.getByText(/Drift score 85\/100/)).toBeInTheDocument()
    })
  })

  it("shows alert recommendation text", async () => {
    const alerts = makeAlerts({
      alert_count: 1,
      warning_count: 1,
      alerts: [
        {
          deployment_id: "dep-1",
          algorithm: "LR",
          severity: "warning",
          type: "no_predictions",
          message: "No predictions yet",
          recommendation: "Share the prediction dashboard link to get started.",
        },
      ],
    })
    mockAlerts.mockResolvedValueOnce(alerts)
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    await waitFor(() =>
      expect(screen.getByText(/Share the prediction dashboard link/)).toBeInTheDocument()
    )
  })

  it("shows 'Show N more' link when more than 2 alerts exist", async () => {
    const alerts = makeAlerts({
      alert_count: 3,
      warning_count: 3,
      alerts: Array.from({ length: 3 }, (_, i) => ({
        deployment_id: `dep-${i}`,
        algorithm: `Model${i}`,
        severity: "warning" as const,
        type: "no_predictions" as const,
        message: `Alert ${i}`,
        recommendation: "Fix it",
      })),
    })
    mockAlerts.mockResolvedValueOnce(alerts)
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    await waitFor(() =>
      expect(screen.getByText(/show 1 more/i)).toBeInTheDocument()
    )
  })

  it("expands to show all alerts when 'Show more' is clicked", async () => {
    const alerts = makeAlerts({
      alert_count: 3,
      warning_count: 3,
      alerts: Array.from({ length: 3 }, (_, i) => ({
        deployment_id: `dep-${i}`,
        algorithm: `Model${i}`,
        severity: "warning" as const,
        type: "no_predictions" as const,
        message: `Alert message ${i}`,
        recommendation: "Fix it",
      })),
    })
    mockAlerts.mockResolvedValueOnce(alerts)
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    await waitFor(() => screen.getByText(/show 1 more/i))
    fireEvent.click(screen.getByText(/show 1 more/i))
    await waitFor(() =>
      expect(screen.getByText("Alert message 2")).toBeInTheDocument()
    )
  })

  it("refresh alerts button calls api again", async () => {
    mockAlerts
      .mockResolvedValueOnce(makeAlerts())
      .mockResolvedValueOnce(makeAlerts({ alert_count: 1, warning_count: 1, alerts: [
        { deployment_id: "d", algorithm: "RF", severity: "warning", type: "stale_model", message: "Stale", recommendation: "Retrain" }
      ]}))
    render(<AlertsCard projectId="proj-1" />)
    fireEvent.click(screen.getByRole("button", { name: /check for alerts/i }))
    await waitFor(() => screen.getByText(/all clear/i))
    fireEvent.click(screen.getByRole("button", { name: /refresh alerts/i }))
    await waitFor(() => expect(mockAlerts).toHaveBeenCalledTimes(2))
  })
})
