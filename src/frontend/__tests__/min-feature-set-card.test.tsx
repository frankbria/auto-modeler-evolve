import React from "react"
import { render, screen } from "@testing-library/react"
import { MinFeatureSetCard } from "@/components/models/min-feature-set-card"
import type { MinFeatureSetResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const base: MinFeatureSetResult = {
  model_run_id: "run-1",
  algorithm: "linear_regression",
  target_col: "revenue",
  n_original: 5,
  n_minimal: 2,
  features_dropped: 3,
  features_retained: ["feat_0", "feat_1"],
  features_dropped_list: ["feat_2", "feat_3", "feat_4"],
  baseline_score: 0.92,
  minimal_score: 0.91,
  score_loss: 0.01,
  can_simplify: true,
  reduction_pct: 60.0,
  metric: "r2",
  summary: "Your model can be simplified from 5 to 2 features (dropping 3, a 60% reduction) while keeping R² at 0.910 (vs 0.920).",
  features: [
    { feature: "feat_0", importance: 0.5, kept: true, rank: 1 },
    { feature: "feat_1", importance: 0.3, kept: true, rank: 2 },
    { feature: "feat_2", importance: 0.1, kept: false, rank: 3 },
    { feature: "feat_3", importance: 0.07, kept: false, rank: 4 },
    { feature: "feat_4", importance: 0.03, kept: false, rank: 5 },
  ],
}

describe("MinFeatureSetCard", () => {
  it("renders heading", () => {
    render(<MinFeatureSetCard result={base} />)
    const headings = screen.getAllByText(/Minimum Viable Feature Set/i)
    expect(headings.length).toBeGreaterThan(0)
  })

  it("shows algorithm badge", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getByText("linear_regression")).toBeInTheDocument()
  })

  it("shows target column badge", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getByText(/target: revenue/i)).toBeInTheDocument()
  })

  it("shows droppable count badge when can_simplify", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getByText(/3 droppable/i)).toBeInTheDocument()
  })

  it("shows all-needed badge when cannot simplify", () => {
    const r = { ...base, can_simplify: false, features_dropped: 0, features_dropped_list: [], features_retained: ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"], n_minimal: 5 }
    render(<MinFeatureSetCard result={r} />)
    expect(screen.getByText(/all features needed/i)).toBeInTheDocument()
  })

  it("renders baseline score", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getByText("0.920")).toBeInTheDocument()
  })

  it("renders minimal score", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getByText("0.910")).toBeInTheDocument()
  })

  it("renders score loss", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getByText(/0\.010/)).toBeInTheDocument()
  })

  it("shows R² label for regression", () => {
    render(<MinFeatureSetCard result={base} />)
    const labels = screen.getAllByText(/R²/)
    expect(labels.length).toBeGreaterThan(0)
  })

  it("shows Accuracy label for classification", () => {
    const r = { ...base, metric: "accuracy", baseline_score: 0.88, minimal_score: 0.87 }
    render(<MinFeatureSetCard result={r} />)
    const labels = screen.getAllByText(/Accuracy/i)
    expect(labels.length).toBeGreaterThan(0)
  })

  it("renders all feature names", () => {
    render(<MinFeatureSetCard result={base} />)
    base.features.forEach((f) => {
      const els = screen.getAllByText(f.feature)
      expect(els.length).toBeGreaterThan(0)
    })
  })

  it("shows retained section when can_simplify", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getAllByText(/Retained/i).length).toBeGreaterThan(0)
  })

  it("shows can drop section when can_simplify", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getByText(/Can drop/i)).toBeInTheDocument()
  })

  it("hides retained/dropped sections when cannot simplify", () => {
    const r = { ...base, can_simplify: false, features_dropped: 0, features_dropped_list: [], features_retained: base.features.map(f => f.feature), n_minimal: 5 }
    render(<MinFeatureSetCard result={r} />)
    expect(screen.queryByText(/Can drop/i)).not.toBeInTheDocument()
  })

  it("shows reduction pct when can_simplify", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getAllByText(/60%/).length).toBeGreaterThan(0)
  })

  it("shows summary text", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getByText(/simplified from 5 to 2/i)).toBeInTheDocument()
  })

  it("has region aria label", () => {
    render(<MinFeatureSetCard result={base} />)
    expect(screen.getByRole("region")).toBeInTheDocument()
  })

  it("has figcaption sr-only", () => {
    const { container } = render(<MinFeatureSetCard result={base} />)
    const caption = container.querySelector("figcaption")
    expect(caption).toBeInTheDocument()
    expect(caption!.className).toContain("sr-only")
  })
})

// Store action test
describe("attachMinFeatureSetToLastMessage store action", () => {
  it("attaches result to last assistant message", () => {
    const { addMessage, attachMinFeatureSetToLastMessage } = useAppStore.getState()
    addMessage({ role: "assistant", content: "test" })
    attachMinFeatureSetToLastMessage(base)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.min_feature_set).toBeDefined()
    expect(last.min_feature_set!.n_original).toBe(5)
  })

  it("does not attach to user message", () => {
    const { addMessage, attachMinFeatureSetToLastMessage } = useAppStore.getState()
    addMessage({ role: "user", content: "test" })
    attachMinFeatureSetToLastMessage(base)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.min_feature_set).toBeUndefined()
  })
})
