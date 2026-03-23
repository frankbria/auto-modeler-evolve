/**
 * Tests for DeployedCard component and attachDeployedToLastMessage store action.
 */
import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { DeployedCard } from "@/components/deploy/deployed-card"
import type { DeployedResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Fixtures -----------------------------------------------------------

const regressionDeployment: DeployedResult = {
  id: "dep-1",
  model_run_id: "run-1",
  project_id: "proj-1",
  endpoint_path: "/api/predict/dep-1",
  dashboard_url: "/predict/dep-1",
  is_active: true,
  algorithm: "linear_regression",
  problem_type: "regression",
  target_column: "revenue",
  feature_names: ["region", "units", "cost"],
  metrics: { r2: 0.85, mae: 120.5 },
  created_at: "2026-03-22T20:00:00",
}

const classificationDeployment: DeployedResult = {
  id: "dep-2",
  model_run_id: "run-2",
  project_id: "proj-1",
  endpoint_path: "/api/predict/dep-2",
  dashboard_url: "/predict/dep-2",
  is_active: true,
  algorithm: "random_forest_classifier",
  problem_type: "classification",
  target_column: "churned",
  feature_names: ["age", "tenure", "spend"],
  metrics: { accuracy: 0.92 },
  created_at: "2026-03-22T20:00:00",
}

// --- Component rendering tests ------------------------------------------

describe("DeployedCard", () => {
  it("renders with testid", () => {
    render(<DeployedCard result={regressionDeployment} />)
    expect(screen.getByTestId("deployed-card")).toBeInTheDocument()
  })

  it("shows Model Deployed header", () => {
    render(<DeployedCard result={regressionDeployment} />)
    expect(screen.getByText("Model Deployed")).toBeInTheDocument()
  })

  it("shows Regression badge for regression models", () => {
    render(<DeployedCard result={regressionDeployment} />)
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("shows Classification badge for classification models", () => {
    render(<DeployedCard result={classificationDeployment} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("shows algorithm label (human-readable)", () => {
    render(<DeployedCard result={regressionDeployment} />)
    expect(screen.getByText(/Linear Regression/)).toBeInTheDocument()
  })

  it("shows random forest label for classifier", () => {
    render(<DeployedCard result={classificationDeployment} />)
    expect(screen.getByText(/Random Forest/)).toBeInTheDocument()
  })

  it("shows target column in code chip", () => {
    render(<DeployedCard result={regressionDeployment} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("shows target column for classification", () => {
    render(<DeployedCard result={classificationDeployment} />)
    expect(screen.getByText("churned")).toBeInTheDocument()
  })

  it("shows primary metric R² for regression", () => {
    render(<DeployedCard result={regressionDeployment} />)
    expect(screen.getByText(/R²\s*0\.850/)).toBeInTheDocument()
  })

  it("shows primary metric Accuracy for classification", () => {
    render(<DeployedCard result={classificationDeployment} />)
    expect(screen.getByText(/Accuracy\s*92\.0%/)).toBeInTheDocument()
  })

  it("renders dashboard link with correct href", () => {
    render(<DeployedCard result={regressionDeployment} />)
    const link = screen.getByTestId("dashboard-link")
    expect(link).toHaveAttribute("href", "/predict/dep-1")
  })

  it("renders API endpoint URL", () => {
    render(<DeployedCard result={regressionDeployment} />)
    expect(
      screen.getByText(/localhost:8000\/api\/predict\/dep-1/)
    ).toBeInTheDocument()
  })

  it("has Copy button for endpoint", () => {
    render(<DeployedCard result={regressionDeployment} />)
    expect(screen.getByTestId("copy-endpoint-btn")).toBeInTheDocument()
  })

  it("copy button shows Copied! after click (clipboard mock)", async () => {
    // Mock clipboard
    Object.assign(navigator, {
      clipboard: { writeText: jest.fn().mockResolvedValue(undefined) },
    })

    render(<DeployedCard result={regressionDeployment} />)
    const btn = screen.getByTestId("copy-endpoint-btn")
    fireEvent.click(btn)

    // Immediately after click the text changes
    expect(await screen.findByText("Copied!")).toBeInTheDocument()
  })

  it("shows no metric label when metrics are empty", () => {
    const noMetrics = { ...regressionDeployment, metrics: {} }
    const { container } = render(<DeployedCard result={noMetrics} />)
    // Should still render without crashing
    expect(container.querySelector('[data-testid="deployed-card"]')).toBeInTheDocument()
  })
})

// --- Store action tests -------------------------------------------------

describe("attachDeployedToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "deploy my model", timestamp: "t1" },
        {
          role: "assistant",
          content: "Deploying now...",
          timestamp: "t2",
        },
      ],
    })
  })

  it("attaches deployed result to last assistant message", () => {
    const { attachDeployedToLastMessage } = useAppStore.getState()
    attachDeployedToLastMessage(regressionDeployment)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.deployed).toEqual(regressionDeployment)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "deploy my model", timestamp: "t1" },
      ],
    })
    const { attachDeployedToLastMessage } = useAppStore.getState()
    // Should not crash, just skip attachment if last is user
    attachDeployedToLastMessage(regressionDeployment)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    // user message has no deployed field
    expect(last.deployed).toBeUndefined()
  })

  it("preserves other message fields when attaching deployed", () => {
    const { attachDeployedToLastMessage } = useAppStore.getState()
    attachDeployedToLastMessage(regressionDeployment)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.content).toBe("Deploying now...")
    expect(last.role).toBe("assistant")
  })
})
