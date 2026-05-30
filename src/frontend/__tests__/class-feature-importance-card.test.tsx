import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { ClassFeatureImportanceCard } from "@/components/models/class-feature-importance-card"
import type { ClassFeatureImportanceResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeFeature(overrides = {}) {
  return {
    feature: "tenure",
    global_importance: 0.3,
    mean_for_class: 12.5,
    global_mean: 8.0,
    deviation_pct: 56.25,
    weighted_deviation: 0.169,
    direction: "above_average" as const,
    ...overrides,
  }
}

function makeClass(label: string, overrides = {}) {
  return {
    class_label: label,
    sample_count: 40,
    sample_pct: 40.0,
    features: [
      makeFeature({ feature: "tenure" }),
      makeFeature({ feature: "monthly_charges", direction: "below_average" as const, deviation_pct: -30 }),
    ],
    top_feature: "tenure",
    summary: `For predictions of '${label}' (40 of 100 records): 'tenure' is the most distinguishing feature.`,
    ...overrides,
  }
}

function makeResult(
  overrides: Partial<ClassFeatureImportanceResult> = {}
): ClassFeatureImportanceResult {
  return {
    model_run_id: "run-001",
    algorithm: "random_forest_classifier",
    target_col: "churn",
    classes: [makeClass("churn"), makeClass("not_churn", { sample_count: 60, sample_pct: 60.0 })],
    n_classes: 2,
    n_samples: 100,
    feature_names: ["tenure", "monthly_charges"],
    summary: "Across 2 classes, 'tenure' is the top distinguishing feature.",
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Component rendering
// ---------------------------------------------------------------------------

describe("ClassFeatureImportanceCard", () => {
  it("renders the card container", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    expect(screen.getByTestId("class-feature-importance-card")).toBeTruthy()
  })

  it("shows the target column badge", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    // "churn" appears in target badge and class labels — ensure at least one match
    expect(screen.getAllByText("churn").length).toBeGreaterThan(0)
  })

  it("shows the algorithm badge", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    expect(screen.getByText("random_forest_classifier")).toBeTruthy()
  })

  it("shows n_classes badge", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    expect(screen.getByText("2 classes")).toBeTruthy()
  })

  it("renders class panel for each class", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    expect(screen.getByTestId("class-panel-churn")).toBeTruthy()
    expect(screen.getByTestId("class-panel-not_churn")).toBeTruthy()
  })

  it("shows class labels in panels", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    expect(screen.getByTestId("class-label-churn")).toHaveTextContent("churn")
    expect(screen.getByTestId("class-label-not_churn")).toHaveTextContent("not_churn")
  })

  it("first class panel is expanded by default", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    // The first class features should be visible
    const featuresDiv = screen.getByTestId("class-features-churn")
    expect(featuresDiv).toBeTruthy()
  })

  it("second class panel is collapsed by default", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    expect(screen.queryByTestId("class-features-not_churn")).toBeNull()
  })

  it("clicking collapsed panel expands it", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    // Click the not_churn panel toggle button
    const panel = screen.getByTestId("class-panel-not_churn")
    fireEvent.click(panel.querySelector("button")!)
    expect(screen.getByTestId("class-features-not_churn")).toBeTruthy()
  })

  it("clicking open panel collapses it", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    // churn is initially open, click to collapse
    const panel = screen.getByTestId("class-panel-churn")
    fireEvent.click(panel.querySelector("button")!)
    expect(screen.queryByTestId("class-features-churn")).toBeNull()
  })

  it("renders feature rows when panel is open", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    // churn panel is open by default
    expect(screen.getByTestId("feat-row-tenure")).toBeTruthy()
    expect(screen.getByTestId("feat-row-monthly_charges")).toBeTruthy()
  })

  it("shows direction label for above_average feature", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    expect(screen.getByTestId("feat-direction-tenure")).toHaveTextContent("above avg")
  })

  it("shows direction label for below_average feature", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    expect(screen.getByTestId("feat-direction-monthly_charges")).toHaveTextContent("below avg")
  })

  it("renders summary text", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    expect(screen.getByTestId("class-feature-importance-summary")).toHaveTextContent(
      "Across 2 classes"
    )
  })

  it("returns null for falsy result", () => {
    const { container } = render(<ClassFeatureImportanceCard result={null as unknown as ClassFeatureImportanceResult} />)
    expect(container.firstChild).toBeNull()
  })

  it("handles single class gracefully", () => {
    render(
      <ClassFeatureImportanceCard
        result={makeResult({ classes: [makeClass("only_class")], n_classes: 1 })}
      />
    )
    expect(screen.getByTestId("class-panel-only_class")).toBeTruthy()
  })

  it("shows sample count and percentage in panel header", () => {
    render(<ClassFeatureImportanceCard result={makeResult()} />)
    expect(screen.getByTestId("class-panel-churn")).toHaveTextContent("40")
    expect(screen.getByTestId("class-panel-churn")).toHaveTextContent("40%")
  })
})

// ---------------------------------------------------------------------------
// Zustand store action
// ---------------------------------------------------------------------------

describe("attachClassFeatureImportanceToLastMessage store action", () => {
  it("attaches to last assistant message", () => {
    const store = useAppStore.getState()
    store.setMessages([
      { id: "1", role: "user", content: "what features drive churn?", timestamp: new Date() },
      { id: "2", role: "assistant", content: "Here is the breakdown.", timestamp: new Date() },
    ])
    const result = makeResult()
    useAppStore.getState().attachClassFeatureImportanceToLastMessage(result)
    const messages = useAppStore.getState().messages
    expect(messages[1].class_feature_importance).toBeDefined()
    expect(messages[1].class_feature_importance!.n_classes).toBe(2)
  })

  it("does not attach to user message", () => {
    const store = useAppStore.getState()
    store.setMessages([
      { id: "1", role: "user", content: "hello", timestamp: new Date() },
    ])
    useAppStore.getState().attachClassFeatureImportanceToLastMessage(makeResult())
    const messages = useAppStore.getState().messages
    expect((messages[0] as Record<string, unknown>).class_feature_importance).toBeUndefined()
  })

  it("does not modify when messages array is empty", () => {
    const store = useAppStore.getState()
    store.setMessages([])
    expect(() =>
      useAppStore.getState().attachClassFeatureImportanceToLastMessage(makeResult())
    ).not.toThrow()
  })
})
