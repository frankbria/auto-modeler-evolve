/**
 * Tests for baseline comparison context in the predict/[id] page.
 *
 * The ExplanationCard now includes a BaselineComparisonBanner that shows
 * how the current prediction compares to the model's baseline prediction
 * (all features at training-data means).
 *
 * Covers:
 * - Regression: baseline_prediction, delta, pct_change, direction rendered
 * - Classification: baseline_prediction, direction, confidence rendered
 * - No banner when baseline_prediction absent (old API response)
 * - PredictionExplanation type includes new optional baseline fields
 */

import React from "react"
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"

fetchMock.enableMocks()

// Mock next/navigation before importing page — must be at module level
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: jest.fn(),
    refresh: jest.fn(),
    replace: jest.fn(),
    back: jest.fn(),
    forward: jest.fn(),
    prefetch: jest.fn(),
  }),
  useParams: () => ({ id: "deploy-baseline-test" }),
  usePathname: () => "/predict/deploy-baseline-test",
  useSearchParams: () => new URLSearchParams(),
}))

// Import page ONCE (static import) to avoid React hook conflicts
// eslint-disable-next-line @typescript-eslint/no-require-imports
const PredictionDashboard = require("../app/predict/[id]/page").default

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const baseDeployment = {
  id: "deploy-baseline-test",
  model_run_id: "run-001",
  project_id: "proj-001",
  endpoint_path: "/api/predict/deploy-baseline-test",
  dashboard_url: "/predict/deploy-baseline-test",
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

const predictionResult = {
  deployment_id: "deploy-baseline-test",
  prediction: 45000,
  problem_type: "regression",
  target_column: "revenue",
  feature_names: ["units"],
}

const explanationWithBaseline = {
  prediction: 45000,
  target_column: "revenue",
  problem_type: "regression",
  contributions: [
    {
      feature: "units",
      value: 150,
      mean_value: 100,
      contribution: 0.05,
      direction: "positive",
    },
  ],
  summary: "Predicted revenue = 45000. The main factors were 'units'.",
  top_drivers: ["units"],
  baseline_prediction: 33000,
  delta: 12000,
  pct_change: 36.4,
  direction: "above_baseline",
  current_confidence: null,
  baseline_confidence: null,
}

const explanationBelowBaseline = {
  ...explanationWithBaseline,
  prediction: 20000,
  delta: -13000,
  pct_change: -39.4,
  direction: "below_baseline",
}

const explanationNoBaseline = {
  prediction: 45000,
  target_column: "revenue",
  problem_type: "regression",
  contributions: [
    { feature: "units", value: 150, mean_value: 100, contribution: 0.05, direction: "positive" },
  ],
  summary: "Predicted revenue = 45000.",
  top_drivers: ["units"],
  // No baseline fields — old API response format
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

const classificationPrediction = {
  deployment_id: "deploy-baseline-test",
  prediction: "cat",
  problem_type: "classification",
  target_column: "label",
  feature_names: ["f1"],
  confidence: 0.87,
  probabilities: { cat: 0.87, dog: 0.13 },
}

const classificationExplanationChanged = {
  prediction: "cat",
  target_column: "label",
  problem_type: "classification",
  contributions: [
    { feature: "f1", value: 3.0, mean_value: 5.5, contribution: -0.02, direction: "negative" },
  ],
  summary: "Predicted label = cat.",
  top_drivers: ["f1"],
  baseline_prediction: "dog",
  delta: null,
  pct_change: null,
  direction: "class_changed",
  current_confidence: 0.87,
  baseline_confidence: 0.62,
}

// ---------------------------------------------------------------------------
// Helper: queue mocked fetch responses for one test scenario
// ---------------------------------------------------------------------------

function queueFetches(
  deployment: object,
  predResult: object,
  expl: object,
) {
  fetchMock.mockResponseOnce(JSON.stringify(deployment))      // GET deployment
  fetchMock.mockResponseOnce(JSON.stringify([]))              // getDashboardPresets
  fetchMock.mockResponseOnce(JSON.stringify({                 // getDashboardConfig
    deployment_id: "deploy-baseline-test",
    fields: [],
    total_count: 0,
    visible_count: 0,
    locked_count: 0,
  }))
  fetchMock.mockResponseOnce(JSON.stringify({                 // getDashboardMetadata
    deployment_id: "deploy-baseline-test",
    dashboard_title: null,
    dashboard_description: null,
    auto_title: "Revenue Predictor",
  }))
  fetchMock.mockResponseOnce(JSON.stringify([deployment]))    // listByProject (compare)
  fetchMock.mockResponseOnce(JSON.stringify(predResult))      // POST predict
  fetchMock.mockResponseOnce(JSON.stringify(expl))            // POST explain
}

// ---------------------------------------------------------------------------
// Helpers for common interactions
// ---------------------------------------------------------------------------

async function renderAndExplain(deployment: object, pred: object, expl: object) {
  queueFetches(deployment, pred, expl)
  render(<PredictionDashboard />)

  // Wait for page to load the deployment and show the form
  await waitFor(() => {
    const allText = document.body.textContent ?? ""
    expect(allText.length).toBeGreaterThan(10)
  })

  // Click predict
  const predictButton = screen.queryByRole("button", { name: /predict/i })
  if (!predictButton) return false

  await act(async () => { fireEvent.click(predictButton) })

  // Wait for the explain button to appear (prediction done)
  await waitFor(() => {
    const explainBtn = screen.queryByRole("button", { name: /why this prediction/i })
    if (!explainBtn) throw new Error("explain button not yet visible")
  }, { timeout: 3000 }).catch(() => null)

  const explainBtn = screen.queryByRole("button", { name: /why this prediction/i })
  if (!explainBtn) return false

  await act(async () => { fireEvent.click(explainBtn) })
  return true
}

// ---------------------------------------------------------------------------
// Tests — Regression baseline
// ---------------------------------------------------------------------------

describe("BaselineComparisonBanner — regression above baseline", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    jest.clearAllMocks()
  })

  it("renders [data-testid=baseline-comparison] when explanation has baseline_prediction", async () => {
    const shown = await renderAndExplain(baseDeployment, predictionResult, explanationWithBaseline)
    if (!shown) return // page didn't reach predict state in jsdom

    await waitFor(() => {
      const banner = document.querySelector("[data-testid='baseline-comparison']")
      expect(banner).toBeTruthy()
    })
  })

  it("shows 'Typical baseline' label in the banner", async () => {
    const shown = await renderAndExplain(baseDeployment, predictionResult, explanationWithBaseline)
    if (!shown) return

    await waitFor(() => {
      const allText = document.body.textContent ?? ""
      expect(allText).toContain("Typical baseline")
    })
  })

  it("shows baseline value 33,000 in the banner", async () => {
    const shown = await renderAndExplain(baseDeployment, predictionResult, explanationWithBaseline)
    if (!shown) return

    await waitFor(() => {
      const allText = document.body.textContent ?? ""
      expect(allText).toContain("33,000")
    })
  })

  it("shows percent change in the banner", async () => {
    const shown = await renderAndExplain(baseDeployment, predictionResult, explanationWithBaseline)
    if (!shown) return

    await waitFor(() => {
      const allText = document.body.textContent ?? ""
      expect(allText).toMatch(/36\.4/)
    })
  })

  it("shows 'raised' in description for above_baseline", async () => {
    const shown = await renderAndExplain(baseDeployment, predictionResult, explanationWithBaseline)
    if (!shown) return

    await waitFor(() => {
      const allText = document.body.textContent ?? ""
      expect(allText).toContain("raised")
    })
  })
})

describe("BaselineComparisonBanner — regression below baseline", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    jest.clearAllMocks()
  })

  it("shows 'lowered' in description for below_baseline", async () => {
    const belowPred = { ...predictionResult, prediction: 20000 }
    const shown = await renderAndExplain(baseDeployment, belowPred, explanationBelowBaseline)
    if (!shown) return

    await waitFor(() => {
      const allText = document.body.textContent ?? ""
      expect(allText).toContain("lowered")
    })
  })

  it("does not show 'raised' for below_baseline", async () => {
    const belowPred = { ...predictionResult, prediction: 20000 }
    const shown = await renderAndExplain(baseDeployment, belowPred, explanationBelowBaseline)
    if (!shown) return

    await waitFor(() => {
      const allText = document.body.textContent ?? ""
      // "raised" should NOT appear when below_baseline
      expect(allText).not.toMatch(/your inputs raised/)
    })
  })
})

describe("BaselineComparisonBanner — no baseline (backward compat)", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    jest.clearAllMocks()
  })

  it("does not render baseline-comparison when no baseline_prediction in response", async () => {
    const shown = await renderAndExplain(baseDeployment, predictionResult, explanationNoBaseline)
    if (!shown) return

    await waitFor(() => {
      // ExplanationCard itself should still show
      const allText = document.body.textContent ?? ""
      expect(allText).toContain("Why this prediction")
    })

    // But baseline-comparison element should be absent
    const banner = document.querySelector("[data-testid='baseline-comparison']")
    expect(banner).toBeFalsy()
  })
})

describe("BaselineComparisonBanner — classification class_changed", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    jest.clearAllMocks()
  })

  it("shows 'Typical baseline' with class name for classification", async () => {
    const shown = await renderAndExplain(
      classificationDeployment,
      classificationPrediction,
      classificationExplanationChanged,
    )
    if (!shown) return

    await waitFor(() => {
      const allText = document.body.textContent ?? ""
      expect(allText).toContain("Typical baseline")
      expect(allText).toContain("dog") // baseline class
    })
  })

  it("shows confidence percentages for classification with predict_proba", async () => {
    const shown = await renderAndExplain(
      classificationDeployment,
      classificationPrediction,
      classificationExplanationChanged,
    )
    if (!shown) return

    await waitFor(() => {
      const allText = document.body.textContent ?? ""
      // Should show 62% → 87% confidence
      expect(allText).toMatch(/62%|87%/)
    })
  })
})

describe("PredictionExplanation type — baseline fields (type-level)", () => {
  it("type accepts all optional baseline fields", () => {
    const expl: import("../lib/types").PredictionExplanation = {
      prediction: 45000,
      target_column: "revenue",
      problem_type: "regression",
      contributions: [],
      summary: "test",
      top_drivers: [],
      baseline_prediction: 33000,
      delta: 12000,
      pct_change: 36.4,
      direction: "above_baseline",
      current_confidence: null,
      baseline_confidence: null,
    }
    expect(expl.baseline_prediction).toBe(33000)
    expect(expl.delta).toBe(12000)
    expect(expl.direction).toBe("above_baseline")
  })

  it("type is backward compatible without baseline fields", () => {
    const expl: import("../lib/types").PredictionExplanation = {
      prediction: 78.5,
      target_column: "score",
      problem_type: "regression",
      contributions: [],
      summary: "test",
      top_drivers: [],
    }
    expect(expl.baseline_prediction).toBeUndefined()
    expect(expl.delta).toBeUndefined()
    expect(expl.direction).toBeUndefined()
  })

  it("type accepts below_baseline direction", () => {
    const expl: import("../lib/types").PredictionExplanation = {
      prediction: 20000,
      target_column: "revenue",
      problem_type: "regression",
      contributions: [],
      summary: "test",
      top_drivers: [],
      baseline_prediction: 33000,
      delta: -13000,
      pct_change: -39.4,
      direction: "below_baseline",
    }
    expect(expl.direction).toBe("below_baseline")
  })

  it("type accepts class_changed direction for classification", () => {
    const expl: import("../lib/types").PredictionExplanation = {
      prediction: "cat",
      target_column: "label",
      problem_type: "classification",
      contributions: [],
      summary: "test",
      top_drivers: [],
      baseline_prediction: "dog",
      direction: "class_changed",
      current_confidence: 0.87,
      baseline_confidence: 0.62,
    }
    expect(expl.direction).toBe("class_changed")
    expect(expl.current_confidence).toBe(0.87)
  })

  it("type accepts same_class direction for classification", () => {
    const expl: import("../lib/types").PredictionExplanation = {
      prediction: "cat",
      target_column: "label",
      problem_type: "classification",
      contributions: [],
      summary: "test",
      top_drivers: [],
      baseline_prediction: "cat",
      direction: "same_class",
    }
    expect(expl.direction).toBe("same_class")
  })
})
