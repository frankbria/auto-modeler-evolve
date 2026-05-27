import { render, screen } from "@testing-library/react"
import { FeatureEngineeringImpactCard } from "@/components/models/feature-engineering-impact-card"
import type { FeatureEngineeringImpactResult } from "@/lib/types"

const mixedResult: FeatureEngineeringImpactResult = {
  run_id: "run-1",
  algorithm: "random_forest_classifier",
  target_column: "churn",
  n_features: 4,
  n_engineered: 2,
  n_original: 2,
  groups: [
    {
      name: "Original",
      total_importance: 0.6,
      features: [
        { name: "age", importance: 0.4, rank: 1, source: null },
        { name: "score", importance: 0.2, rank: 3, source: null },
      ],
    },
    {
      name: "Engineered",
      total_importance: 0.4,
      features: [
        { name: "income_log", importance: 0.3, rank: 2, source: "income" },
        { name: "income_bin", importance: 0.1, rank: 4, source: "income" },
      ],
    },
  ],
  engineered_columns: ["income_log", "income_bin"],
  original_columns: ["age", "score"],
  verdict: "Feature engineering added value — engineered features contribute 40% of total model importance.",
  has_engineering: true,
}

const noEngineeringResult: FeatureEngineeringImpactResult = {
  run_id: "run-2",
  algorithm: "linear_regression",
  target_column: "revenue",
  n_features: 2,
  n_engineered: 0,
  n_original: 2,
  groups: [
    {
      name: "Original",
      total_importance: 1.0,
      features: [
        { name: "age", importance: 0.7, rank: 1, source: null },
        { name: "income", importance: 0.3, rank: 2, source: null },
      ],
    },
    {
      name: "Engineered",
      total_importance: 0.0,
      features: [],
    },
  ],
  engineered_columns: [],
  original_columns: ["age", "income"],
  verdict: "No feature engineering was applied. All model inputs are original columns.",
  has_engineering: false,
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

describe("FeatureEngineeringImpactCard", () => {
  it("renders the card container", () => {
    render(<FeatureEngineeringImpactCard result={mixedResult} />)
    expect(screen.getByTestId("feature-engineering-impact-card")).toBeInTheDocument()
  })

  it("shows the algorithm badge", () => {
    render(<FeatureEngineeringImpactCard result={mixedResult} />)
    expect(screen.getByText("random_forest_classifier")).toBeInTheDocument()
  })

  it("shows the verdict text", () => {
    render(<FeatureEngineeringImpactCard result={mixedResult} />)
    expect(screen.getByText(/added value/)).toBeInTheDocument()
  })

  it("shows engineered feature badge count", () => {
    render(<FeatureEngineeringImpactCard result={mixedResult} />)
    expect(screen.getByText(/2 engineered features/)).toBeInTheDocument()
  })

  it("shows Original and Engineered group headings", () => {
    render(<FeatureEngineeringImpactCard result={mixedResult} />)
    expect(screen.getAllByText("Original").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Engineered").length).toBeGreaterThan(0)
  })

  it("shows feature names from both groups", () => {
    render(<FeatureEngineeringImpactCard result={mixedResult} />)
    expect(screen.getByText(/age/)).toBeInTheDocument()
    expect(screen.getByText(/income_log/)).toBeInTheDocument()
  })

  it("shows source attribution for engineered features", () => {
    render(<FeatureEngineeringImpactCard result={mixedResult} />)
    expect(screen.getAllByText(/income/).length).toBeGreaterThan(0)
  })

  it("shows target column", () => {
    render(<FeatureEngineeringImpactCard result={mixedResult} />)
    expect(screen.getByText(/churn/)).toBeInTheDocument()
  })

  // ---------------------------------------------------------------------------
  // No engineering case
  // ---------------------------------------------------------------------------

  it("shows no-engineering badge when has_engineering is false", () => {
    render(<FeatureEngineeringImpactCard result={noEngineeringResult} />)
    expect(screen.getByText("No engineering applied")).toBeInTheDocument()
  })

  it("shows no-engineering verdict", () => {
    render(<FeatureEngineeringImpactCard result={noEngineeringResult} />)
    expect(screen.getByText(/No feature engineering was applied/)).toBeInTheDocument()
  })

  it("does not render empty Engineered group section", () => {
    render(<FeatureEngineeringImpactCard result={noEngineeringResult} />)
    // Engineered section should not appear since it has no features
    const card = screen.getByTestId("feature-engineering-impact-card")
    expect(card).toBeInTheDocument()
  })

  // ---------------------------------------------------------------------------
  // Null guard
  // ---------------------------------------------------------------------------

  it("returns null when result is falsy", () => {
    // @ts-expect-error — testing null guard
    const { container } = render(<FeatureEngineeringImpactCard result={null} />)
    expect(container.firstChild).toBeNull()
  })
})
