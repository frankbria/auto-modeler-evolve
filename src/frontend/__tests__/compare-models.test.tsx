/**
 * Tests for CompareModelsCard on the predict/[id] page.
 *
 * Covers:
 * - Card hidden when no other deployments in the project
 * - Card visible when other deployments exist
 * - Toggle button shows count of available versions
 * - Panel expands with dropdown and Compare button on toggle click
 * - Comparison results rendered after clicking Compare
 * - api.deploy.listByProject calls with correct project_id
 * - api.deploy.compareModels POSTs correct body and throws on error
 */

import React from "react"
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"

fetchMock.enableMocks()

jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: jest.fn(),
    refresh: jest.fn(),
    replace: jest.fn(),
    back: jest.fn(),
    forward: jest.fn(),
    prefetch: jest.fn(),
  }),
  useParams: () => ({ id: "deploy-current" }),
  usePathname: () => "/predict/deploy-current",
  useSearchParams: () => new URLSearchParams(),
}))

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const currentDeployment = {
  id: "deploy-current",
  model_run_id: "run-001",
  project_id: "proj-001",
  endpoint_path: "/api/predict/deploy-current",
  dashboard_url: "/predict/deploy-current",
  is_active: true,
  request_count: 10,
  created_at: "2024-01-15T10:00:00",
  last_predicted_at: null,
  algorithm: "linear_regression",
  problem_type: "regression",
  target_column: "revenue",
  feature_schema: [
    { name: "units", type: "numeric", median: 10.0, options: null },
  ],
  feature_names: ["units"],
  metrics: {},
}

const olderDeployment = {
  id: "deploy-old",
  model_run_id: "run-000",
  project_id: "proj-001",
  endpoint_path: "/api/predict/deploy-old",
  dashboard_url: "/predict/deploy-old",
  is_active: true,
  request_count: 25,
  created_at: "2024-01-01T08:00:00",
  last_predicted_at: null,
  algorithm: "random_forest_regressor",
  problem_type: "regression",
  target_column: "revenue",
  feature_schema: [
    { name: "units", type: "numeric", median: 10.0, options: null },
  ],
  feature_names: ["units"],
  metrics: {},
}

const comparisonResponse = {
  results: [
    {
      deployment_id: "deploy-current",
      algorithm: "linear_regression",
      trained_at: "2024-01-15T10:00:00",
      prediction: 1200.5,
      problem_type: "regression",
      target_column: "revenue",
      error: null,
    },
    {
      deployment_id: "deploy-old",
      algorithm: "random_forest_regressor",
      trained_at: "2024-01-01T08:00:00",
      prediction: 1150.75,
      problem_type: "regression",
      target_column: "revenue",
      error: null,
    },
  ],
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Render the predict page, mocking GET /api/deploy/deploy-current and GET /api/deployments?project_id=proj-001 */
async function renderWithDeployments(projectDeployments: object[]) {
  fetchMock.mockResponseOnce(JSON.stringify(currentDeployment))
  fetchMock.mockResponseOnce(JSON.stringify(projectDeployments))

  const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
  render(<PredictionDashboard />)

  // Wait for page to finish loading
  await waitFor(() => {
    expect(screen.queryByText(/units/i)).toBeTruthy()
  })
}

// ---------------------------------------------------------------------------
// Tests — CompareModelsCard visibility
// ---------------------------------------------------------------------------

describe("CompareModelsCard — visibility", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    jest.clearAllMocks()
  })

  it("does not render compare card when no other deployments exist", async () => {
    await renderWithDeployments([currentDeployment])
    expect(screen.queryByTestId("compare-models-card")).not.toBeInTheDocument()
  })

  it("renders compare card when other deployments exist in the same project", async () => {
    await renderWithDeployments([currentDeployment, olderDeployment])
    await waitFor(() => {
      expect(screen.getByTestId("compare-models-card")).toBeInTheDocument()
    })
  })

  it("shows toggle button text including available version count", async () => {
    await renderWithDeployments([currentDeployment, olderDeployment])
    await waitFor(() => {
      expect(screen.getByTestId("compare-toggle")).toBeInTheDocument()
    })
    expect(screen.getByTestId("compare-toggle").textContent).toContain("1")
  })
})

// ---------------------------------------------------------------------------
// Tests — CompareModelsCard interaction
// ---------------------------------------------------------------------------

describe("CompareModelsCard — interaction", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    jest.clearAllMocks()
  })

  it("expands panel with dropdown and Compare button when toggle clicked", async () => {
    await renderWithDeployments([currentDeployment, olderDeployment])
    await waitFor(() => screen.getByTestId("compare-toggle"))

    fireEvent.click(screen.getByTestId("compare-toggle"))

    expect(screen.getByTestId("compare-select")).toBeInTheDocument()
    expect(screen.getByTestId("compare-button")).toBeInTheDocument()
  })

  it("renders comparison results table after clicking Compare", async () => {
    await renderWithDeployments([currentDeployment, olderDeployment])
    await waitFor(() => screen.getByTestId("compare-toggle"))

    fireEvent.click(screen.getByTestId("compare-toggle"))

    fetchMock.mockResponseOnce(JSON.stringify(comparisonResponse))
    await act(async () => {
      fireEvent.click(screen.getByTestId("compare-button"))
    })

    await waitFor(() => {
      const predictions = screen.getAllByTestId("compare-prediction")
      expect(predictions.length).toBe(2)
    })
  })

  it("shows algorithm names from comparison results", async () => {
    await renderWithDeployments([currentDeployment, olderDeployment])
    await waitFor(() => screen.getByTestId("compare-toggle"))

    fireEvent.click(screen.getByTestId("compare-toggle"))

    fetchMock.mockResponseOnce(JSON.stringify(comparisonResponse))
    await act(async () => {
      fireEvent.click(screen.getByTestId("compare-button"))
    })

    await waitFor(() => {
      // linear_regression appears in both the page badge AND the table
      expect(screen.getAllByText("linear_regression").length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText("random_forest_regressor").length).toBeGreaterThanOrEqual(1)
    })
  })

  it("collapses the panel when toggle clicked again", async () => {
    await renderWithDeployments([currentDeployment, olderDeployment])
    await waitFor(() => screen.getByTestId("compare-toggle"))

    fireEvent.click(screen.getByTestId("compare-toggle")) // open
    expect(screen.getByTestId("compare-select")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("compare-toggle")) // close
    expect(screen.queryByTestId("compare-select")).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Tests — API client methods
// ---------------------------------------------------------------------------

describe("api.deploy.listByProject", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
  })

  it("calls the correct URL with project_id query param", async () => {
    fetchMock.mockResponseOnce(JSON.stringify([]))
    const { api } = await import("../lib/api")
    await api.deploy.listByProject("my-project-id")

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("project_id=my-project-id")
    )
  })
})

describe("api.deploy.compareModels", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
  })

  it("POSTs deployment_ids and features to /api/predict/compare", async () => {
    fetchMock.mockResponseOnce(JSON.stringify({ results: [] }))
    const { api } = await import("../lib/api")
    await api.deploy.compareModels(["dep-1", "dep-2"], { units: 10 })

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/predict/compare"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          deployment_ids: ["dep-1", "dep-2"],
          features: { units: 10 },
        }),
      })
    )
  })

  it("throws when the API returns an error status", async () => {
    fetchMock.mockResponseOnce("Internal error", { status: 500 })
    const { api } = await import("../lib/api")
    await expect(
      api.deploy.compareModels(["dep-1", "dep-2"], {})
    ).rejects.toThrow("HTTP 500")
  })
})
