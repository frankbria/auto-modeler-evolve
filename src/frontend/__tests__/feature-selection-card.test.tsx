import { render, screen, fireEvent } from "@testing-library/react"
import { FeatureSelectionCard } from "@/components/models/feature-selection-card"
import type { FeatureSelectionResult } from "@/lib/types"

const rfResult: FeatureSelectionResult = {
  run_id: "run-1",
  algorithm: "random_forest_regressor",
  target_column: "revenue",
  n_features: 4,
  feature_importances: [
    { name: "f1", importance: 0.6, rank: 1, is_weak: false },
    { name: "f2", importance: 0.25, rank: 2, is_weak: false },
    { name: "f3", importance: 0.1, rank: 3, is_weak: true },
    { name: "f4", importance: 0.05, rank: 4, is_weak: true },
  ],
  weak_features: ["f3", "f4"],
  threshold: 0.12,
  method: "feature_importances",
  has_importances: true,
  n_weak: 2,
  explanation:
    "2 features have near-zero feature importance (bottom 20%). Removing them may reduce noise.",
}

const linResult: FeatureSelectionResult = {
  run_id: "run-2",
  algorithm: "linear_regression",
  target_column: "revenue",
  n_features: 3,
  feature_importances: [
    { name: "a", importance: 0.7, rank: 1, is_weak: false },
    { name: "b", importance: 0.2, rank: 2, is_weak: false },
    { name: "c", importance: 0.1, rank: 3, is_weak: true },
  ],
  weak_features: ["c"],
  threshold: 0.15,
  method: "coefficients",
  has_importances: true,
  n_weak: 1,
  explanation: "1 feature has near-zero coefficient magnitude (bottom 20%).",
}

const noImportanceResult: FeatureSelectionResult = {
  run_id: "run-3",
  algorithm: "mlp_regressor",
  target_column: "revenue",
  n_features: 2,
  feature_importances: [
    { name: "x", importance: null, rank: 1, is_weak: false },
    { name: "y", importance: null, rank: 2, is_weak: false },
  ],
  weak_features: [],
  threshold: null,
  method: "not_available",
  has_importances: false,
  n_weak: 0,
  explanation: "Feature importances are not available for this model type.",
}

// -------------------------------------------------------
// Chat card (result prop)
// -------------------------------------------------------

describe("FeatureSelectionCard — chat card mode", () => {
  it("renders header with 🎯 icon", () => {
    render(<FeatureSelectionCard result={rfResult} />)
    expect(screen.getByText("🎯")).toBeInTheDocument()
    expect(screen.getByText("Feature Importance")).toBeInTheDocument()
  })

  it("shows algorithm badge", () => {
    render(<FeatureSelectionCard result={rfResult} />)
    expect(screen.getByText("random_forest_regressor")).toBeInTheDocument()
  })

  it("shows weak feature count badge when n_weak > 0", () => {
    render(<FeatureSelectionCard result={rfResult} />)
    expect(screen.getByText(/2 weak features/i)).toBeInTheDocument()
  })

  it("shows 'All features contributing' badge when n_weak = 0", () => {
    const zeroWeak = { ...rfResult, n_weak: 0, weak_features: [], explanation: "All features are contributing." }
    render(<FeatureSelectionCard result={zeroWeak} />)
    expect(screen.getByText(/All features contributing/i)).toBeInTheDocument()
  })

  it("renders explanation text", () => {
    render(<FeatureSelectionCard result={rfResult} />)
    expect(screen.getByText(/Removing them may reduce noise/i)).toBeInTheDocument()
  })

  it("renders all feature names", () => {
    render(<FeatureSelectionCard result={rfResult} />)
    expect(screen.getAllByText("f1").length).toBeGreaterThan(0)
    expect(screen.getAllByText("f2").length).toBeGreaterThan(0)
    expect(screen.getAllByText("f3").length).toBeGreaterThan(0)
    expect(screen.getAllByText("f4").length).toBeGreaterThan(0)
  })

  it("marks weak features with ↓ weak indicator", () => {
    render(<FeatureSelectionCard result={rfResult} />)
    const weakIndicators = screen.getAllByText(/↓ weak/i)
    expect(weakIndicators.length).toBe(2)
  })

  it("shows rank numbers", () => {
    render(<FeatureSelectionCard result={rfResult} />)
    expect(screen.getByText("#1")).toBeInTheDocument()
    expect(screen.getByText("#4")).toBeInTheDocument()
  })

  it("shows percentage values", () => {
    render(<FeatureSelectionCard result={rfResult} />)
    expect(screen.getByText("60.0%")).toBeInTheDocument()
    expect(screen.getByText("5.0%")).toBeInTheDocument()
  })

  it("shows method note in footer", () => {
    render(<FeatureSelectionCard result={rfResult} />)
    expect(screen.getByText(/feature importance from random_forest_regressor/i)).toBeInTheDocument()
  })

  it("shows coefficient method note for linear model", () => {
    render(<FeatureSelectionCard result={linResult} />)
    expect(screen.getAllByText(/coefficient magnitude/i).length).toBeGreaterThan(0)
  })

  it("does not show importance bars when has_importances is false", () => {
    render(<FeatureSelectionCard result={noImportanceResult} />)
    expect(screen.queryByText("#1")).not.toBeInTheDocument()
  })
})

// -------------------------------------------------------
// Panel card (data prop with interactive checkboxes)
// -------------------------------------------------------

describe("FeatureSelectionCard — panel mode", () => {
  it("renders checkboxes for each feature", () => {
    const onChange = jest.fn()
    render(
      <FeatureSelectionCard
        data={rfResult}
        excludedFeatures={[]}
        onExcludedFeaturesChange={onChange}
      />
    )
    const checkboxes = screen.getAllByRole("checkbox")
    expect(checkboxes.length).toBe(4)
  })

  it("shows 'Exclude N weak features' button when weak features exist", () => {
    const onChange = jest.fn()
    render(
      <FeatureSelectionCard
        data={rfResult}
        excludedFeatures={[]}
        onExcludedFeaturesChange={onChange}
      />
    )
    expect(screen.getByText(/Exclude 2 weak features on retrain/i)).toBeInTheDocument()
  })

  it("calls onExcludedFeaturesChange with weak_features when button clicked", () => {
    const onChange = jest.fn()
    render(
      <FeatureSelectionCard
        data={rfResult}
        excludedFeatures={[]}
        onExcludedFeaturesChange={onChange}
      />
    )
    fireEvent.click(screen.getByText(/Exclude 2 weak features on retrain/i))
    expect(onChange).toHaveBeenCalledWith(["f3", "f4"])
  })

  it("shows excluded count when features are checked", () => {
    const onChange = jest.fn()
    render(
      <FeatureSelectionCard
        data={rfResult}
        excludedFeatures={["f3"]}
        onExcludedFeaturesChange={onChange}
      />
    )
    expect(screen.getByText(/1 feature will be excluded on next retrain/i)).toBeInTheDocument()
  })

  it("shows Clear button when features are excluded", () => {
    const onChange = jest.fn()
    render(
      <FeatureSelectionCard
        data={rfResult}
        excludedFeatures={["f3", "f4"]}
        onExcludedFeaturesChange={onChange}
      />
    )
    expect(screen.getByText("Clear")).toBeInTheDocument()
  })

  it("clicking Clear calls onExcludedFeaturesChange with empty array", () => {
    const onChange = jest.fn()
    render(
      <FeatureSelectionCard
        data={rfResult}
        excludedFeatures={["f3", "f4"]}
        onExcludedFeaturesChange={onChange}
      />
    )
    fireEvent.click(screen.getByText("Clear"))
    expect(onChange).toHaveBeenCalledWith([])
  })

  it("clicking checkbox toggles feature exclusion", () => {
    const onChange = jest.fn()
    render(
      <FeatureSelectionCard
        data={rfResult}
        excludedFeatures={[]}
        onExcludedFeaturesChange={onChange}
      />
    )
    const checkboxes = screen.getAllByRole("checkbox")
    // Click the first checkbox (f1 — rank 1, not weak)
    fireEvent.click(checkboxes[0])
    expect(onChange).toHaveBeenCalled()
  })

  it("shows 'All features contributing' message when no weak features", () => {
    const noWeak = { ...rfResult, n_weak: 0, weak_features: [], explanation: "All features contributing." }
    const onChange = jest.fn()
    render(
      <FeatureSelectionCard
        data={noWeak}
        excludedFeatures={[]}
        onExcludedFeaturesChange={onChange}
      />
    )
    expect(screen.getAllByText(/All features contributing/i).length).toBeGreaterThan(0)
  })

  it("uses amber border styling", () => {
    const onChange = jest.fn()
    const { container } = render(
      <FeatureSelectionCard
        data={rfResult}
        excludedFeatures={[]}
        onExcludedFeaturesChange={onChange}
      />
    )
    const card = container.firstChild as HTMLElement
    expect(card.className).toMatch(/border-amber/)
  })
})
