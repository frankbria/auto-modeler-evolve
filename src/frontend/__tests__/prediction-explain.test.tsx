/**
 * Tests for prediction explanation feature:
 * - api.deploy.explain() client method
 * - PredictionExplanation type shape
 * - predict/[id] page with explanation toggle
 */

import { api } from "../lib/api"

// ---------------------------------------------------------------------------
// Mock fetch
// ---------------------------------------------------------------------------

const mockFetch = jest.fn()
global.fetch = mockFetch as unknown as typeof fetch

beforeEach(() => {
  mockFetch.mockReset()
})

// ---------------------------------------------------------------------------
// api.deploy.explain()
// ---------------------------------------------------------------------------

describe("api.deploy.explain", () => {
  const deploymentId = "dep-abc-123"
  const inputs = { age: 40, income: 55000 }

  const sampleExplanation = {
    prediction: 78.5,
    target_column: "score",
    problem_type: "regression",
    contributions: [
      {
        feature: "income",
        value: 55000,
        mean_value: 52000,
        contribution: 0.0123,
        direction: "positive",
      },
      {
        feature: "age",
        value: 40,
        mean_value: 38,
        contribution: 0.0045,
        direction: "positive",
      },
    ],
    summary: "Predicted score = 78.5. The main factors were 'income' and 'age'.",
    top_drivers: ["income", "age"],
  }

  it("calls correct endpoint with POST method", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => sampleExplanation,
    })

    await api.deploy.explain(deploymentId, inputs)

    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining(`/api/predict/${deploymentId}/explain`),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
      })
    )
  })

  it("returns PredictionExplanation shape", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => sampleExplanation,
    })

    const result = await api.deploy.explain(deploymentId, inputs)

    expect(result.prediction).toBe(78.5)
    expect(result.target_column).toBe("score")
    expect(result.problem_type).toBe("regression")
    expect(result.contributions).toHaveLength(2)
    expect(result.summary).toContain("score")
    expect(result.top_drivers).toEqual(["income", "age"])
  })

  it("throws on non-ok response", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
    })

    await expect(api.deploy.explain(deploymentId, inputs)).rejects.toThrow("HTTP 404")
  })

  it("contribution entries have required fields", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => sampleExplanation,
    })

    const result = await api.deploy.explain(deploymentId, inputs)
    const c = result.contributions[0]
    expect(c).toHaveProperty("feature")
    expect(c).toHaveProperty("value")
    expect(c).toHaveProperty("mean_value")
    expect(c).toHaveProperty("contribution")
    expect(c.direction).toMatch(/^(positive|negative)$/)
  })

  it("handles empty contributions gracefully", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        ...sampleExplanation,
        contributions: [],
        top_drivers: [],
      }),
    })

    const result = await api.deploy.explain(deploymentId, inputs)
    expect(result.contributions).toHaveLength(0)
    expect(result.top_drivers).toHaveLength(0)
  })

  it("works with classification prediction", async () => {
    const classificationExpl = {
      prediction: "cat",
      target_column: "label",
      problem_type: "classification",
      contributions: [
        { feature: "x1", value: 3.0, mean_value: 5.5, contribution: -0.02, direction: "negative" },
        { feature: "x2", value: 4.0, mean_value: 6.5, contribution: -0.018, direction: "negative" },
      ],
      summary: "Predicted label = cat. The prediction was primarily driven by 'x1'.",
      top_drivers: ["x1"],
    }

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => classificationExpl,
    })

    const result = await api.deploy.explain("dep-clf", { x1: 3.0, x2: 4.0 })
    expect(result.problem_type).toBe("classification")
    expect(result.prediction).toBe("cat")
  })
})
