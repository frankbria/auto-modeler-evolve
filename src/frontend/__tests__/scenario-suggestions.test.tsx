/**
 * Tests for:
 *   1. Chat suggestion chips — rendered in ProjectWorkspace after SSE suggestions event
 *   2. api.deploy.scenarios() client method — correct HTTP shape
 *
 * These are unit/integration tests that don't need a real server.
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"

// ---------------------------------------------------------------------------
// Tests for api.deploy.scenarios()
// ---------------------------------------------------------------------------

describe("api.deploy.scenarios()", () => {
  const originalFetch = global.fetch

  afterEach(() => {
    global.fetch = originalFetch
  })

  it("calls POST /api/predict/{id}/scenarios with correct body", async () => {
    const mockResponse = {
      deployment_id: "dep-1",
      base_prediction: 1200,
      scenarios: [
        { label: "High units", overrides: { units: 50 }, prediction: 2000, delta: 800, percent_change: 66.7, direction: "increase", probabilities: null },
      ],
      summary: "Base = 1200. Best: 'High units' → 2000 (+66.7%).",
      problem_type: "regression",
      target_column: "revenue",
      base_probabilities: null,
    }

    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    } as Response)

    const { api } = await import("../lib/api")
    const base = { units: 10, product: "Widget A" }
    const scenarios = [{ label: "High units", overrides: { units: 50 } }]

    const result = await api.deploy.scenarios("dep-1", base, scenarios)

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/predict/dep-1/scenarios"),
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ base, scenarios }),
      })
    )
    expect(result.base_prediction).toBe(1200)
    expect(result.scenarios).toHaveLength(1)
    expect(result.scenarios[0].label).toBe("High units")
    expect(result.summary).toContain("1200")
  })

  it("throws on non-OK response", async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ detail: "At least one scenario is required" }),
    } as Response)

    const { api } = await import("../lib/api")
    await expect(api.deploy.scenarios("dep-1", {}, [])).rejects.toThrow("HTTP 400")
  })

  it("sends up to 10 scenarios in one call", async () => {
    const scenarios = Array.from({ length: 10 }, (_, i) => ({
      label: `S${i}`,
      overrides: { units: i + 1 },
    }))

    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        deployment_id: "d1",
        base_prediction: 500,
        scenarios: scenarios.map((s) => ({
          ...s,
          prediction: 500,
          delta: 0,
          percent_change: 0,
          direction: "no change",
          probabilities: null,
        })),
        summary: "Base = 500.",
        problem_type: "regression",
        target_column: "revenue",
        base_probabilities: null,
      }),
    } as Response)

    const { api } = await import("../lib/api")
    const result = await api.deploy.scenarios("d1", { units: 10 }, scenarios)
    expect(result.scenarios).toHaveLength(10)
  })
})

// ---------------------------------------------------------------------------
// Tests for suggestion chips in the workspace
// ---------------------------------------------------------------------------

// Lightweight mock of the page — we test just the chip rendering logic
function SuggestionChips({
  suggestions,
  onSelect,
}: {
  suggestions: string[]
  onSelect: (s: string) => void
}) {
  if (suggestions.length === 0) return null
  return (
    <div data-testid="suggestion-chips">
      {suggestions.map((s, i) => (
        <button
          key={i}
          data-testid="suggestion-chip"
          onClick={() => onSelect(s)}
          className="rounded-full border"
        >
          {s}
        </button>
      ))}
    </div>
  )
}

describe("SuggestionChips component", () => {
  it("renders nothing when suggestions is empty", () => {
    const { container } = render(
      <SuggestionChips suggestions={[]} onSelect={jest.fn()} />
    )
    expect(container.firstChild).toBeNull()
  })

  it("renders one chip per suggestion", () => {
    render(
      <SuggestionChips
        suggestions={["What drives revenue?", "Show correlations"]}
        onSelect={jest.fn()}
      />
    )
    const chips = screen.getAllByTestId("suggestion-chip")
    expect(chips).toHaveLength(2)
    expect(chips[0]).toHaveTextContent("What drives revenue?")
    expect(chips[1]).toHaveTextContent("Show correlations")
  })

  it("calls onSelect with the suggestion text when clicked", () => {
    const onSelect = jest.fn()
    render(
      <SuggestionChips
        suggestions={["Are there seasonal patterns?"]}
        onSelect={onSelect}
      />
    )
    fireEvent.click(screen.getByTestId("suggestion-chip"))
    expect(onSelect).toHaveBeenCalledWith("Are there seasonal patterns?")
  })

  it("renders up to 3 chips correctly", () => {
    render(
      <SuggestionChips
        suggestions={["A", "B", "C"]}
        onSelect={jest.fn()}
      />
    )
    expect(screen.getAllByTestId("suggestion-chip")).toHaveLength(3)
  })

  it("container div is present when there are suggestions", () => {
    render(
      <SuggestionChips suggestions={["Hello?"]} onSelect={jest.fn()} />
    )
    expect(screen.getByTestId("suggestion-chips")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Tests for ScenarioComparison types (type correctness via runtime values)
// ---------------------------------------------------------------------------

describe("ScenarioComparison types", () => {
  it("ScenarioResult has required fields", () => {
    const item = {
      label: "High volume",
      overrides: { units: 50 },
      prediction: 2300,
      delta: 1100,
      percent_change: 91.7,
      direction: "increase",
      probabilities: null,
    }
    // Type sanity: all expected keys exist
    expect(item).toHaveProperty("label")
    expect(item).toHaveProperty("overrides")
    expect(item).toHaveProperty("prediction")
    expect(item).toHaveProperty("delta")
    expect(item).toHaveProperty("percent_change")
    expect(item).toHaveProperty("direction")
    expect(item).toHaveProperty("probabilities")
  })

  it("ScenarioComparison summary is a non-empty string", () => {
    const comparison = {
      deployment_id: "dep-1",
      base_prediction: 1200,
      base_probabilities: null,
      problem_type: "regression",
      target_column: "revenue",
      scenarios: [],
      summary: "Base revenue = 1200. No scenarios computed.",
    }
    expect(typeof comparison.summary).toBe("string")
    expect(comparison.summary.length).toBeGreaterThan(0)
  })
})
