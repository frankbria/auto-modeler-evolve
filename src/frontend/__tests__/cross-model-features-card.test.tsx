import React from "react"
import { render, screen } from "@testing-library/react"
import { CrossModelFeaturesCard } from "@/components/chat/cross-model-features-card"
import type { CrossModelFeatureResult, CrossModelFeatureEntry } from "@/lib/types"

function makeEntry(overrides: Partial<CrossModelFeatureEntry> = {}): CrossModelFeatureEntry {
  return {
    feature: "age",
    mean_importance: 0.40,
    n_models_with_data: 2,
    agreement_count: 2,
    consistency: "high",
    per_model: [
      { algorithm_plain: "Random Forest", importance: 0.40, rank: 1 },
      { algorithm_plain: "Linear Regression", importance: 0.40, rank: 1 },
    ],
    ...overrides,
  }
}

function makeResult(overrides: Partial<CrossModelFeatureResult> = {}): CrossModelFeatureResult {
  return {
    n_models: 2,
    features: [
      makeEntry({ feature: "age", mean_importance: 0.40 }),
      makeEntry({ feature: "income", mean_importance: 0.35, consistency: "medium" }),
      makeEntry({ feature: "score", mean_importance: 0.25, consistency: "variable", agreement_count: 1 }),
    ],
    consensus_features: ["age", "income"],
    top_feature: "age",
    summary: "Across 2 models, 'age' and 'income' consistently rank in the top 5.",
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Null / empty guards
// ---------------------------------------------------------------------------

test("renders nothing when result is falsy", () => {
  // @ts-expect-error testing null case
  const { container } = render(<CrossModelFeaturesCard result={null} />)
  expect(container.firstChild).toBeNull()
})

test("renders nothing when n_models is 0", () => {
  const { container } = render(
    <CrossModelFeaturesCard result={makeResult({ n_models: 0, features: [] })} />
  )
  expect(container.firstChild).toBeNull()
})

// ---------------------------------------------------------------------------
// Basic rendering
// ---------------------------------------------------------------------------

test("renders card heading", () => {
  render(<CrossModelFeaturesCard result={makeResult()} />)
  expect(screen.getByText("Feature Importance Across Models")).toBeInTheDocument()
})

test("renders model count in subheading", () => {
  render(<CrossModelFeaturesCard result={makeResult()} />)
  expect(screen.getByText(/2 trained models/i)).toBeInTheDocument()
})

test("renders summary text", () => {
  render(<CrossModelFeaturesCard result={makeResult()} />)
  expect(screen.getAllByText(/age.*income.*consistently rank/i).length).toBeGreaterThan(0)
})

// ---------------------------------------------------------------------------
// Feature table
// ---------------------------------------------------------------------------

test("renders all feature rows", () => {
  render(<CrossModelFeaturesCard result={makeResult()} />)
  // features appear at least once each (may appear in consensus chips too)
  expect(screen.getAllByText("age").length).toBeGreaterThan(0)
  expect(screen.getAllByText("income").length).toBeGreaterThan(0)
  expect(screen.getByText("score")).toBeInTheDocument()
})

test("renders Consistent badge for high consistency feature", () => {
  render(<CrossModelFeaturesCard result={makeResult()} />)
  expect(screen.getAllByText("Consistent").length).toBeGreaterThan(0)
})

test("renders Moderate badge for medium consistency feature", () => {
  render(<CrossModelFeaturesCard result={makeResult()} />)
  expect(screen.getByText("Moderate")).toBeInTheDocument()
})

test("renders Variable badge for variable consistency feature", () => {
  render(<CrossModelFeaturesCard result={makeResult()} />)
  expect(screen.getByText("Variable")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Consensus callout
// ---------------------------------------------------------------------------

test("renders consensus callout when consensus features exist", () => {
  render(<CrossModelFeaturesCard result={makeResult()} />)
  expect(screen.getByText(/All models agree/i)).toBeInTheDocument()
})

test("renders consensus feature chips", () => {
  render(<CrossModelFeaturesCard result={makeResult()} />)
  const chips = screen.getAllByText("age")
  expect(chips.length).toBeGreaterThanOrEqual(1)
})

test("does not render consensus callout when no consensus features", () => {
  render(<CrossModelFeaturesCard result={makeResult({ consensus_features: [] })} />)
  expect(screen.queryByText(/All models agree/i)).toBeNull()
})

// ---------------------------------------------------------------------------
// Accessibility
// ---------------------------------------------------------------------------

test("figure has accessible aria-label", () => {
  render(<CrossModelFeaturesCard result={makeResult()} />)
  const fig = screen.getByRole("figure", { name: /cross-model feature importance/i })
  expect(fig).toBeInTheDocument()
})

test("figcaption contains summary text for screen readers", () => {
  const result = makeResult()
  const { container } = render(<CrossModelFeaturesCard result={result} />)
  const caption = container.querySelector("figcaption")
  expect(caption?.textContent).toContain(result.summary)
})

// ---------------------------------------------------------------------------
// Single model edge case
// ---------------------------------------------------------------------------

test("renders single model variant", () => {
  const result = makeResult({
    n_models: 1,
    consensus_features: [],
    summary: "Across your 1 trained model, 'age' is the most important feature.",
  })
  render(<CrossModelFeaturesCard result={result} />)
  expect(screen.getAllByText(/1 trained model/i).length).toBeGreaterThan(0)
})
