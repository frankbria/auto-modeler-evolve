/**
 * Tests for the cross-deployment comparison API client methods.
 *
 * NOTE: The "Compare Model Versions" UI was removed from the PUBLIC predict
 * page because it enumerated sibling deployments via an owner-scoped endpoint,
 * leaking sibling deployment metadata (IDOR, issue #25 — see
 * predict-page-public.test.tsx). The underlying API client methods remain (they
 * map to real, auth-gated backend endpoints) and are covered here; comparison
 * is an owner-only operation that belongs behind auth, not on the public page.
 */

import fetchMock from "jest-fetch-mock"

fetchMock.enableMocks()

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
