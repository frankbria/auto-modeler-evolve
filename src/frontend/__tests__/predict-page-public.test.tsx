/**
 * Security regression test for the PUBLIC prediction page (app/predict/[id]).
 *
 * The public page is reachable by anyone holding a share link. It must load
 * ONLY the single shared deployment via the public surface and must never call
 * an owner-scoped management endpoint (e.g. GET /api/deployments?project_id=…),
 * which would leak sibling deployment IDs/metadata (IDOR, issue #25).
 */

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
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
  feature_schema: [{ name: "units", type: "numeric", median: 10.0, options: null }],
  feature_names: ["units"],
  metrics: {},
}

describe("public predict page — no owner-scoped calls (IDOR #25)", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    jest.clearAllMocks()
  })

  it("loads only the single deployment and never calls an owner-scoped endpoint", async () => {
    // The public-surface loads the page makes; intentionally NO deployments list.
    fetchMock.mockResponse(async (req) => {
      if (req.url.includes("/api/deploy/deploy-current")) {
        return JSON.stringify(currentDeployment)
      }
      // presets / dashboard config / metadata — benign empties
      return JSON.stringify({})
    })

    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)

    await waitFor(() => {
      expect(screen.queryByText(/units/i)).toBeTruthy()
    })

    const calledUrls = fetchMock.mock.calls.map((c) => String(c[0]))
    // No owner-scoped deployment enumeration.
    expect(calledUrls.some((u) => u.includes("/api/deployments"))).toBe(false)
    expect(calledUrls.some((u) => u.includes("project_id="))).toBe(false)
    // The compare-versions UI (which enumerated siblings) must be gone.
    expect(screen.queryByTestId("compare-models-card")).not.toBeInTheDocument()
  })
})
