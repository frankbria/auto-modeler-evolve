/**
 * Tests for ABTestCard component.
 *
 * Covers:
 * - "No active test" idle state with Start button
 * - Start A/B test form (input, slider, create)
 * - Active test display (split bar, variant metrics, significance)
 * - End test action
 * - Promote challenger with confirmation flow
 * - Error handling
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { ABTestCard } from "@/components/deploy/ab-test-card"
import { api } from "@/lib/api"
import type { ABTest } from "@/lib/types"

jest.mock("@/lib/api")

const DEPLOYMENT_ID = "dep-champion"

const mockActiveTest: ABTest = {
  id: "ab-test-1",
  champion_id: DEPLOYMENT_ID,
  challenger_id: "dep-challenger",
  champion_algorithm: "Linear Regression",
  challenger_algorithm: "Ridge Regression",
  champion_split_pct: 80,
  challenger_split_pct: 20,
  is_active: true,
  auto_promote: false,
  created_at: "2024-04-01T10:00:00",
  ended_at: null,
  winner: null,
  champion_metrics: {
    request_count: 40,
    avg_confidence: 0.87,
    p95_ms: 45.0,
    avg_prediction: 312.5,
  },
  challenger_metrics: {
    request_count: 10,
    avg_confidence: 0.91,
    p95_ms: 38.0,
    avg_prediction: 318.0,
  },
  significance: {
    significant: false,
    p_value: null,
    note: "Need 5 more samples per variant (minimum 5)",
  },
}

const mockSignificantTest: ABTest = {
  ...mockActiveTest,
  significance: {
    significant: true,
    p_value: 0.0312,
    note: "Mann-Whitney U test (α=0.05)",
  },
}

describe("ABTestCard", () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  // -------------------------------------------------------------------------
  // Idle state (no active test)
  // -------------------------------------------------------------------------

  it("shows Start A/B Test button when no active test", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockRejectedValue(new Error("HTTP 404"))
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("Start A/B Test")).toBeInTheDocument()
    )
  })

  it("shows description when no active test", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockRejectedValue(new Error("HTTP 404"))
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText(/Split live prediction traffic/)).toBeInTheDocument()
    )
  })

  // -------------------------------------------------------------------------
  // Form display
  // -------------------------------------------------------------------------

  it("shows form after clicking Start A/B Test", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockRejectedValue(new Error("HTTP 404"))
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => screen.getByText("Start A/B Test"))
    fireEvent.click(screen.getByText("Start A/B Test"))
    expect(screen.getByTestId("challenger-id-input")).toBeInTheDocument()
    expect(screen.getByText("Start Test")).toBeInTheDocument()
  })

  it("shows champion/challenger split labels in form", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockRejectedValue(new Error("HTTP 404"))
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => screen.getByText("Start A/B Test"))
    fireEvent.click(screen.getByText("Start A/B Test"))
    expect(screen.getByText(/Champion traffic share/)).toBeInTheDocument()
  })

  it("cancels form and hides it", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockRejectedValue(new Error("HTTP 404"))
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => screen.getByText("Start A/B Test"))
    fireEvent.click(screen.getByText("Start A/B Test"))
    fireEvent.click(screen.getByText("Cancel"))
    expect(screen.queryByTestId("challenger-id-input")).not.toBeInTheDocument()
  })

  it("calls createAbTest with correct args when form submitted", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockRejectedValue(new Error("HTTP 404"))
    ;(api.deploy.createAbTest as jest.Mock).mockResolvedValue(mockActiveTest)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => screen.getByText("Start A/B Test"))
    fireEvent.click(screen.getByText("Start A/B Test"))

    fireEvent.change(screen.getByTestId("challenger-id-input"), {
      target: { value: "dep-challenger" },
    })
    fireEvent.click(screen.getByText("Start Test"))
    await waitFor(() =>
      expect(api.deploy.createAbTest).toHaveBeenCalledWith(
        DEPLOYMENT_ID,
        "dep-challenger",
        80
      )
    )
  })

  it("shows error when createAbTest fails", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockRejectedValue(new Error("HTTP 404"))
    ;(api.deploy.createAbTest as jest.Mock).mockRejectedValue(
      new Error("HTTP 400")
    )
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => screen.getByText("Start A/B Test"))
    fireEvent.click(screen.getByText("Start A/B Test"))
    fireEvent.change(screen.getByTestId("challenger-id-input"), {
      target: { value: "dep-challenger" },
    })
    fireEvent.click(screen.getByText("Start Test"))
    await waitFor(() =>
      expect(screen.getByRole("alert")).toBeInTheDocument()
    )
  })

  // -------------------------------------------------------------------------
  // Active test display
  // -------------------------------------------------------------------------

  it("shows algorithm names for champion and challenger", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockActiveTest)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => {
      expect(screen.getByText(/Linear Regression/)).toBeInTheDocument()
      expect(screen.getByText(/Ridge Regression/)).toBeInTheDocument()
    })
  })

  it("shows Live badge when test is active", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockActiveTest)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("Live")).toBeInTheDocument()
    )
  })

  it("shows champion and challenger request counts", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockActiveTest)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => {
      expect(screen.getByText("40")).toBeInTheDocument()  // champion requests
      expect(screen.getByText("10")).toBeInTheDocument()  // challenger requests
    })
  })

  it("shows p95 latency for both variants", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockActiveTest)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => {
      expect(screen.getByText("45ms")).toBeInTheDocument()
      expect(screen.getByText("38ms")).toBeInTheDocument()
    })
  })

  it("shows significance note when no p-value", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockActiveTest)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(
        screen.getByText(/Need 5 more samples/)
      ).toBeInTheDocument()
    )
  })

  it("shows significant badge when test is statistically significant", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockSignificantTest)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(
        screen.getByText("Statistically significant")
      ).toBeInTheDocument()
    )
  })

  it("shows not-significant badge when test is not significant", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue({
      ...mockActiveTest,
      significance: { significant: false, p_value: 0.32, note: "Mann-Whitney U test (α=0.05)" },
    })
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("Not yet significant")).toBeInTheDocument()
    )
  })

  it("shows p-value when available", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockSignificantTest)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText(/p = 0.0312/)).toBeInTheDocument()
    )
  })

  // -------------------------------------------------------------------------
  // End test
  // -------------------------------------------------------------------------

  it("calls endAbTest and clears test on End Test click", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockActiveTest)
    ;(api.deploy.endAbTest as jest.Mock).mockResolvedValue(undefined)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => screen.getByText("End Test"))
    fireEvent.click(screen.getByText("End Test"))
    await waitFor(() =>
      expect(api.deploy.endAbTest).toHaveBeenCalledWith(DEPLOYMENT_ID)
    )
    // After ending, should show Start A/B Test again
    await waitFor(() =>
      expect(screen.getByText("Start A/B Test")).toBeInTheDocument()
    )
  })

  // -------------------------------------------------------------------------
  // Promote challenger
  // -------------------------------------------------------------------------

  it("shows confirmation callout after clicking Promote Challenger", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockActiveTest)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => screen.getByText("Promote Challenger"))
    fireEvent.click(screen.getByText("Promote Challenger"))
    expect(screen.getByText(/Promote challenger\?/)).toBeInTheDocument()
    expect(screen.getByText("Yes, promote")).toBeInTheDocument()
  })

  it("cancels promote and hides confirmation", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockActiveTest)
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => screen.getByText("Promote Challenger"))
    fireEvent.click(screen.getByText("Promote Challenger"))
    fireEvent.click(screen.getByText("Cancel"))
    expect(screen.queryByText(/Promote challenger\?/)).not.toBeInTheDocument()
    expect(screen.getByText("Promote Challenger")).toBeInTheDocument()
  })

  it("calls promoteChallenger and clears test on Yes promote", async () => {
    ;(api.deploy.getAbTest as jest.Mock).mockResolvedValue(mockActiveTest)
    ;(api.deploy.promoteChallenger as jest.Mock).mockResolvedValue({
      message: "Challenger promoted to champion.",
      deployment: {},
    })
    const onPromoted = jest.fn()
    render(<ABTestCard deploymentId={DEPLOYMENT_ID} onPromoted={onPromoted} />)
    await waitFor(() => screen.getByText("Promote Challenger"))
    fireEvent.click(screen.getByText("Promote Challenger"))
    fireEvent.click(screen.getByText("Yes, promote"))
    await waitFor(() =>
      expect(api.deploy.promoteChallenger).toHaveBeenCalledWith(DEPLOYMENT_ID)
    )
    expect(onPromoted).toHaveBeenCalled()
    await waitFor(() =>
      expect(screen.getByText("Start A/B Test")).toBeInTheDocument()
    )
  })
})
