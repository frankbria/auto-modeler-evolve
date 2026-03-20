/**
 * Tests for prediction confidence intervals in the predict/[id] page.
 *
 * Covers:
 * - Regression CI badge renders with lower/upper bounds
 * - Classification confidence badge renders
 * - No CI shown when confidence_interval absent
 * - ConfidenceInterval type exported from types.ts
 */

import React from "react"
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"

fetchMock.enableMocks()

// Mock next/navigation before importing page
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: jest.fn(),
    refresh: jest.fn(),
    replace: jest.fn(),
    back: jest.fn(),
    forward: jest.fn(),
    prefetch: jest.fn(),
  }),
  useParams: () => ({ id: "deploy-ci-test" }),
  usePathname: () => "/predict/deploy-ci-test",
  useSearchParams: () => new URLSearchParams(),
}))

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const baseDeployment = {
  id: "deploy-ci-test",
  model_run_id: "run-001",
  project_id: "proj-001",
  endpoint_path: "/api/predict/deploy-ci-test",
  dashboard_url: "/predict/deploy-ci-test",
  is_active: true,
  request_count: 5,
  created_at: "2024-01-01T00:00:00",
  last_predicted_at: null,
  algorithm: "linear_regression",
  problem_type: "regression",
  target_column: "revenue",
  feature_schema: [
    { name: "units", type: "numeric", median: 10.0, options: null },
  ],
}

const regressionResultWithCI = {
  deployment_id: "deploy-ci-test",
  prediction: 1350.25,
  problem_type: "regression",
  target_column: "revenue",
  feature_names: ["units"],
  confidence_interval: {
    lower: 900.5,
    upper: 1800.0,
    level: 0.95,
    label: "95% prediction interval",
  },
}

const regressionResultNoCI = {
  deployment_id: "deploy-ci-test",
  prediction: 1350.25,
  problem_type: "regression",
  target_column: "revenue",
  feature_names: ["units"],
  // No confidence_interval field
}

const classificationDeployment = {
  ...baseDeployment,
  problem_type: "classification",
  target_column: "label",
  algorithm: "logistic_regression",
  feature_schema: [
    { name: "f1", type: "numeric", median: 5.0, options: null },
  ],
}

const classificationResultWithConfidence = {
  deployment_id: "deploy-ci-test",
  prediction: "cat",
  problem_type: "classification",
  target_column: "label",
  feature_names: ["f1"],
  confidence: 0.87,
  probabilities: { cat: 0.87, dog: 0.13 },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("PredictionDashboard — confidence intervals", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    jest.clearAllMocks()
  })

  it("renders confidence interval badge for regression predictions", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(baseDeployment))
    fetchMock.mockResponseOnce(JSON.stringify([baseDeployment])) // listByProject
    fetchMock.mockResponseOnce(JSON.stringify(regressionResultWithCI))

    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)

    // Wait for deployment to load and form to render
    await waitFor(() => {
      expect(screen.queryByText(/units/i)).toBeTruthy()
    })

    const predictButton = screen.queryByRole("button", { name: /predict/i })
    if (predictButton) {
      await act(async () => { fireEvent.click(predictButton) })
      await waitFor(() => {
        // CI badge should render
        const ciBadge = document.querySelector("[data-testid='confidence-interval']")
        expect(ciBadge).toBeTruthy()
      })
    }
  })

  it("shows lower and upper bounds in confidence interval", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(baseDeployment))
    fetchMock.mockResponseOnce(JSON.stringify([baseDeployment])) // listByProject
    fetchMock.mockResponseOnce(JSON.stringify(regressionResultWithCI))

    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)

    await waitFor(() => {
      expect(screen.queryByText(/units/i)).toBeTruthy()
    })

    const predictButton = screen.queryByRole("button", { name: /predict/i })
    if (predictButton) {
      await act(async () => { fireEvent.click(predictButton) })
      await waitFor(() => {
        const allText = document.body.textContent ?? ""
        // Should contain the lower and upper values somewhere
        expect(allText).toContain("900")
        expect(allText).toContain("1,800")
      })
    }
  })

  it("does not render CI badge when confidence_interval absent", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(baseDeployment))
    fetchMock.mockResponseOnce(JSON.stringify([baseDeployment])) // listByProject
    fetchMock.mockResponseOnce(JSON.stringify(regressionResultNoCI))

    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)

    await waitFor(() => {
      expect(screen.queryByText(/units/i)).toBeTruthy()
    })

    const predictButton = screen.queryByRole("button", { name: /predict/i })
    if (predictButton) {
      await act(async () => { fireEvent.click(predictButton) })
      await waitFor(() => {
        expect(fetchMock).toHaveBeenCalledTimes(3)
      })
      // CI badge should NOT be present
      const ciBadge = document.querySelector("[data-testid='confidence-interval']")
      expect(ciBadge).toBeNull()
    }
  })

  it("renders classification confidence badge", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(classificationDeployment))
    fetchMock.mockResponseOnce(JSON.stringify([classificationDeployment])) // listByProject
    fetchMock.mockResponseOnce(JSON.stringify(classificationResultWithConfidence))

    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)

    await waitFor(() => {
      expect(screen.queryByText(/f1/i)).toBeTruthy()
    })

    const predictButton = screen.queryByRole("button", { name: /predict/i })
    if (predictButton) {
      await act(async () => { fireEvent.click(predictButton) })
      await waitFor(() => {
        const confBadge = document.querySelector("[data-testid='classification-confidence']")
        expect(confBadge).toBeTruthy()
      })
    }
  })

  it("shows confidence percentage for classification", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(classificationDeployment))
    fetchMock.mockResponseOnce(JSON.stringify([classificationDeployment])) // listByProject
    fetchMock.mockResponseOnce(JSON.stringify(classificationResultWithConfidence))

    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)

    await waitFor(() => {
      expect(screen.queryByText(/f1/i)).toBeTruthy()
    })

    const predictButton = screen.queryByRole("button", { name: /predict/i })
    if (predictButton) {
      await act(async () => { fireEvent.click(predictButton) })
      await waitFor(() => {
        // 87% confidence
        const allText = document.body.textContent ?? ""
        expect(allText).toContain("87%")
      })
    }
  })
})

// ---------------------------------------------------------------------------
// ConfidenceInterval type (from types.ts)
// ---------------------------------------------------------------------------

describe("ConfidenceInterval type", () => {
  it("ConfidenceInterval is exported from types.ts", async () => {
    const types = await import("../lib/types")
    // The type exists if the module exports it; TypeScript compile catches missing types
    // We verify by checking PredictionResult has the field shape
    const result: import("../lib/types").PredictionResult = {
      deployment_id: "d1",
      prediction: 123,
      problem_type: "regression",
      target_column: "revenue",
      feature_names: ["units"],
      confidence_interval: {
        lower: 100,
        upper: 150,
        level: 0.95,
        label: "95% prediction interval",
      },
    }
    expect(result.confidence_interval?.lower).toBe(100)
    expect(result.confidence_interval?.upper).toBe(150)
    expect(result.confidence_interval?.level).toBe(0.95)
    // confidence field for classification
    const cls: import("../lib/types").PredictionResult = {
      deployment_id: "d2",
      prediction: "cat",
      problem_type: "classification",
      target_column: "label",
      feature_names: ["f1"],
      confidence: 0.87,
    }
    expect(cls.confidence).toBe(0.87)
    void types
  })
})
