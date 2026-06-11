import React from "react"
import { render, screen, within } from "@testing-library/react"
import { DeploymentHealthScorecardCard } from "../components/deploy/deployment-health-scorecard-card"
import type { DeploymentHealthScorecardResult, DeploymentHealthSignal } from "../lib/types"
import { useAppStore } from "../lib/store"

function makeSignal(
  key: string,
  label: string,
  severity: "green" | "amber" | "red" | "gray",
  score: number,
  value: number | null,
  value_label: string,
  finding: string,
  icon = "⚡",
  unit = ""
): DeploymentHealthSignal {
  return {
    signal_key: key,
    signal_label: label,
    icon,
    severity,
    score,
    value,
    value_label,
    finding,
    is_available: severity !== "gray",
    unit,
  }
}

function makeResult(overrides?: Partial<DeploymentHealthScorecardResult>): DeploymentHealthScorecardResult {
  const signals: DeploymentHealthSignal[] = [
    makeSignal("latency", "Response Latency", "green", 100, 80, "p95 80 ms", "Fast response.", "⚡", "ms"),
    makeSignal("confidence", "Prediction Confidence", "green", 90, 88, "88% avg", "High confidence.", "🎯", "%"),
    makeSignal("activity", "Usage Activity", "green", 75, 25, "25 predictions / 7d", "Steady activity.", "📊", "predictions"),
    makeSignal("accuracy", "Feedback Accuracy", "green", 95, 92, "92% (20 labeled)", "High accuracy.", "✅", "%"),
    makeSignal("model_age", "Model Freshness", "green", 100, 5, "5d old", "Fresh model.", "📅", "days"),
  ]
  return {
    deployment_id: "dep-1",
    algorithm: "random_forest_regressor",
    target_column: "revenue",
    signals,
    overall_score: 92,
    overall_grade: "A",
    overall_health: "healthy",
    n_signals_healthy: 5,
    n_signals_warning: 0,
    n_signals_critical: 0,
    canary_info: null,
    summary: "Your deployment is healthy (grade A, score 92/100). 5 of 5 signals are green.",
    ...overrides,
  }
}

describe("DeploymentHealthScorecardCard", () => {
  describe("rendering", () => {
    it("renders the card with testid", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      expect(screen.getByTestId("deployment-health-scorecard-card")).toBeInTheDocument()
    })

    it("renders the title text", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      expect(screen.getByText("Deployment Health Scorecard")).toBeInTheDocument()
    })

    it("renders the grade badge with correct grade", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      const badge = screen.getByTestId("grade-badge")
      expect(badge).toHaveTextContent("A")
    })

    it("renders grade B correctly", () => {
      render(<DeploymentHealthScorecardCard result={makeResult({ overall_grade: "B", overall_score: 75 })} />)
      expect(screen.getByTestId("grade-badge")).toHaveTextContent("B")
    })

    it("renders grade F correctly", () => {
      render(<DeploymentHealthScorecardCard result={makeResult({ overall_grade: "F", overall_score: 20, overall_health: "critical" })} />)
      expect(screen.getByTestId("grade-badge")).toHaveTextContent("F")
    })

    it("renders the health badge", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      expect(screen.getByTestId("health-badge")).toHaveTextContent("Healthy")
    })

    it("renders 'Watching' health badge", () => {
      render(<DeploymentHealthScorecardCard result={makeResult({ overall_health: "watching" })} />)
      expect(screen.getByTestId("health-badge")).toHaveTextContent("Watching")
    })

    it("renders 'Warning' health badge", () => {
      render(<DeploymentHealthScorecardCard result={makeResult({ overall_health: "warning" })} />)
      expect(screen.getByTestId("health-badge")).toHaveTextContent("Warning")
    })

    it("renders 'Critical' health badge", () => {
      render(<DeploymentHealthScorecardCard result={makeResult({ overall_health: "critical" })} />)
      expect(screen.getByTestId("health-badge")).toHaveTextContent("Critical")
    })

    it("renders score badge", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      expect(screen.getByText("Score 92/100")).toBeInTheDocument()
    })

    it("renders 'All signals healthy' when no warning/critical", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      expect(screen.getByText("All signals healthy")).toBeInTheDocument()
    })

    it("renders warning count badge when there are warnings", () => {
      render(
        <DeploymentHealthScorecardCard
          result={makeResult({ n_signals_warning: 2, n_signals_healthy: 3 })}
        />
      )
      expect(screen.getByText("2 warning")).toBeInTheDocument()
    })

    it("renders critical count badge when there are critical signals", () => {
      render(
        <DeploymentHealthScorecardCard
          result={makeResult({ n_signals_critical: 1, n_signals_healthy: 4, overall_health: "warning" })}
        />
      )
      expect(screen.getByText("1 critical")).toBeInTheDocument()
    })
  })

  describe("signal rows", () => {
    it("renders all 5 signal rows", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      expect(screen.getByTestId("signal-row-latency")).toBeInTheDocument()
      expect(screen.getByTestId("signal-row-confidence")).toBeInTheDocument()
      expect(screen.getByTestId("signal-row-activity")).toBeInTheDocument()
      expect(screen.getByTestId("signal-row-accuracy")).toBeInTheDocument()
      expect(screen.getByTestId("signal-row-model_age")).toBeInTheDocument()
    })

    it("renders latency signal with correct label and value", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      const row = screen.getByTestId("signal-row-latency")
      expect(within(row).getByText("Response Latency")).toBeInTheDocument()
      expect(within(row).getByText("p95 80 ms")).toBeInTheDocument()
    })

    it("renders finding text for each signal", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      expect(screen.getByText("Fast response.")).toBeInTheDocument()
    })

    it("renders icon for signal", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      const row = screen.getByTestId("signal-row-latency")
      expect(within(row).getByText("⚡")).toBeInTheDocument()
    })
  })

  describe("canary info", () => {
    it("renders canary banner when canary is active", () => {
      const result = makeResult({
        canary_info: {
          is_active: true,
          traffic_pct: 15,
          finding: "Canary is active — 15% of predictions routed to test version.",
        },
      })
      render(<DeploymentHealthScorecardCard result={result} />)
      expect(screen.getByText(/Canary Active/)).toBeInTheDocument()
      expect(screen.getByText(/15%/)).toBeInTheDocument()
    })

    it("does not render canary banner when canary is inactive", () => {
      render(<DeploymentHealthScorecardCard result={makeResult({ canary_info: null })} />)
      expect(screen.queryByText(/Canary Active/)).not.toBeInTheDocument()
    })
  })

  describe("summary", () => {
    it("renders summary text", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      const summary = screen.getByTestId("scorecard-summary")
      expect(summary).toHaveTextContent("Your deployment is healthy")
    })
  })

  describe("algorithm and target", () => {
    it("renders algorithm and target column", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      expect(screen.getByText(/random forest regressor.*revenue/i)).toBeInTheDocument()
    })
  })

  describe("accessibility", () => {
    it("has sr-only figcaption", () => {
      const { container } = render(<DeploymentHealthScorecardCard result={makeResult()} />)
      const figcaption = container.querySelector("figcaption.sr-only")
      expect(figcaption).toBeTruthy()
      expect(figcaption?.textContent).toContain("A")
    })

    it("has role=region", () => {
      render(<DeploymentHealthScorecardCard result={makeResult()} />)
      expect(screen.getByRole("region")).toBeInTheDocument()
    })
  })
})

describe("DeploymentHealthScorecardCard store action", () => {
  it("attachDeploymentHealthScorecardToLastMessage sets scorecard on last assistant message", () => {
    const store = useAppStore.getState()

    store.setMessages([
      { role: "user", content: "health?", id: "u1" },
      { role: "assistant", content: "Here you go", id: "a1" },
    ])

    const scorecard = makeResult()
    store.attachDeploymentHealthScorecardToLastMessage(scorecard)

    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.deployment_health_scorecard).toEqual(scorecard)
    expect(last.deployment_health_scorecard?.overall_grade).toBe("A")
  })
})
