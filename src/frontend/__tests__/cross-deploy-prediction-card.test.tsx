/**
 * Tests for CrossDeployPredictionCard component and cross_deploy_prediction store action.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { CrossDeployPredictionCard } from "@/components/deploy/cross-deploy-prediction-card"
import type { CrossDeployPredictionResult } from "@/lib/types"

// ── Fixtures ──────────────────────────────────────────────────────────────────

const regressionResult: CrossDeployPredictionResult = {
  target_column: "revenue",
  problem_type: "regression",
  n_deployments: 2,
  provided_features: ["units", "region"],
  defaults_used: 1,
  results: [
    {
      deployment_id: "dep-1",
      algorithm: "linear_regression",
      algorithm_plain: "Linear Regression",
      deployed_at: new Date(Date.now() - 86_400_000).toISOString(),
      environment: "staging",
      prediction: 4500.0,
      problem_type: "regression",
      target_column: "revenue",
      confidence_interval: { lower: 4100.0, upper: 4900.0 },
      error: null,
    },
    {
      deployment_id: "dep-2",
      algorithm: "ridge_regression",
      algorithm_plain: "Ridge Regression",
      deployed_at: new Date(Date.now() - 3_600_000).toISOString(),
      environment: "production",
      prediction: 5500.0,
      problem_type: "regression",
      target_column: "revenue",
      confidence_interval: { lower: 5100.0, upper: 5900.0 },
      error: null,
    },
  ],
  summary:
    "Comparing 2 deployed models on revenue: predictions range from 4500.00 to 5500.00. Highest: Ridge Regression.",
}

const classificationResult: CrossDeployPredictionResult = {
  target_column: "churn",
  problem_type: "classification",
  n_deployments: 2,
  provided_features: ["tenure", "monthly_charges"],
  defaults_used: 0,
  results: [
    {
      deployment_id: "dep-a",
      algorithm: "logistic_regression",
      algorithm_plain: "Logistic Regression",
      deployed_at: null,
      environment: "staging",
      prediction: "churn",
      problem_type: "classification",
      target_column: "churn",
      probabilities: { churn: 0.72, no_churn: 0.28 },
      confidence: 0.72,
      error: null,
    },
    {
      deployment_id: "dep-b",
      algorithm: "random_forest",
      algorithm_plain: "Random Forest",
      deployed_at: null,
      environment: "production",
      prediction: "no_churn",
      problem_type: "classification",
      target_column: "churn",
      probabilities: { churn: 0.35, no_churn: 0.65 },
      confidence: 0.65,
      error: null,
    },
  ],
  summary:
    "Comparing 2 deployed models on churn: predictions vary — Logistic Regression: churn, Random Forest: no_churn",
}

const withError: CrossDeployPredictionResult = {
  ...regressionResult,
  results: [
    ...regressionResult.results,
    {
      deployment_id: "dep-bad",
      algorithm: "xgboost",
      algorithm_plain: "Xgboost",
      deployed_at: null,
      environment: "staging",
      prediction: 0,
      problem_type: "regression",
      target_column: "revenue",
      error: "Model file not found",
    },
  ],
  n_deployments: 3,
}

// ── Rendering tests ───────────────────────────────────────────────────────────

describe("CrossDeployPredictionCard", () => {
  it("renders the card title", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    expect(screen.getByRole("region")).toBeInTheDocument()
  })

  it("shows Model Comparison heading", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    expect(screen.getByText("Model Comparison")).toBeInTheDocument()
  })

  it("shows count badge with number of models", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    expect(screen.getByTestId("cross-deploy-count-badge")).toHaveTextContent("2 models")
  })

  it("shows count badge as singular when n_deployments=1", () => {
    const single: CrossDeployPredictionResult = {
      ...regressionResult,
      n_deployments: 1,
      results: [regressionResult.results[0]],
    }
    render(<CrossDeployPredictionCard result={single} />)
    expect(screen.getByTestId("cross-deploy-count-badge")).toHaveTextContent("1 model")
  })

  it("shows target column in badge", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    expect(screen.getByText("Target: revenue")).toBeInTheDocument()
  })

  it("shows Regression problem type badge", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("shows Classification badge for classification results", () => {
    render(<CrossDeployPredictionCard result={classificationResult} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("renders one row per deployment", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    const rows = screen.getAllByTestId("cross-deploy-row")
    expect(rows).toHaveLength(2)
  })

  it("shows algorithm name in each row", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    const algos = screen.getAllByTestId("cross-deploy-algorithm")
    expect(algos[0]).toHaveTextContent("Linear Regression")
    expect(algos[1]).toHaveTextContent("Ridge Regression")
  })

  it("shows prediction values", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    const preds = screen.getAllByTestId("cross-deploy-prediction")
    expect(preds[0]).toHaveTextContent("4,500")
    expect(preds[1]).toHaveTextContent("5,500")
  })

  it("shows 🏆 winner trophy for highest regression prediction", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    // Ridge Regression (5500) should be winner
    expect(screen.getByLabelText("highest prediction")).toBeInTheDocument()
  })

  it("shows production badge for production environment", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    expect(screen.getByText("Production")).toBeInTheDocument()
    expect(screen.getByText("Staging")).toBeInTheDocument()
  })

  it("shows provided features", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    expect(screen.getByTestId("cross-deploy-features")).toBeInTheDocument()
    expect(screen.getByText("units")).toBeInTheDocument()
    expect(screen.getByText("region")).toBeInTheDocument()
  })

  it("shows defaults used note when defaults > 0", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    expect(screen.getByText(/1 using training-data averages/)).toBeInTheDocument()
  })

  it("does not show defaults note when defaults_used is 0", () => {
    render(<CrossDeployPredictionCard result={classificationResult} />)
    expect(
      screen.queryByText(/using training-data averages/)
    ).not.toBeInTheDocument()
  })

  it("shows error text for failed deployments", () => {
    render(<CrossDeployPredictionCard result={withError} />)
    expect(screen.getByTestId("cross-deploy-error")).toHaveTextContent("Error")
  })

  it("shows summary text", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    expect(screen.getByTestId("cross-deploy-summary")).toHaveTextContent(
      "Comparing 2 deployed models"
    )
  })

  it("renders sr-only figcaption for accessibility", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    const figcaption = document.querySelector("figcaption.sr-only")
    expect(figcaption).toBeInTheDocument()
    expect(figcaption?.textContent).toContain("revenue")
  })

  it("has correct aria-label on region container", () => {
    render(<CrossDeployPredictionCard result={regressionResult} />)
    expect(
      screen.getByRole("region", {
        name: /cross-deployment prediction comparison/i,
      })
    ).toBeInTheDocument()
  })
})

// ── Store action test ─────────────────────────────────────────────────────────

describe("attachCrossDeployPredictionToLastMessage store action", () => {
  it("attaches cross_deploy_prediction to the last assistant message", () => {
    const { useAppStore } = jest.requireActual<
      typeof import("@/lib/store")
    >("@/lib/store")

    const store = useAppStore.getState()
    store.addMessage({ role: "user", content: "compare my models" })
    store.addMessage({ role: "assistant", content: "Comparing..." })

    store.attachCrossDeployPredictionToLastMessage(regressionResult)

    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.cross_deploy_prediction).toBeDefined()
    expect(last.cross_deploy_prediction?.target_column).toBe("revenue")
    expect(last.cross_deploy_prediction?.n_deployments).toBe(2)
  })

  it("does not attach to user message", () => {
    const { useAppStore } = jest.requireActual<
      typeof import("@/lib/store")
    >("@/lib/store")

    const store = useAppStore.getState()
    store.addMessage({ role: "user", content: "compare my models" })
    // No assistant message added

    store.attachCrossDeployPredictionToLastMessage(regressionResult)

    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    // Last message is user — no attachment
    expect(last.cross_deploy_prediction).toBeUndefined()
  })
})
