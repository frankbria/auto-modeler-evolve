import React from "react"
import { render, screen } from "@testing-library/react"
import { FeatureSweepCard } from "@/components/deploy/feature-sweep-card"
import type { FeatureSweepResult } from "@/lib/types"

const baseResult: FeatureSweepResult = {
  deployment_id: "dep-1",
  direction: "maximize",
  target_column: "revenue",
  problem_type: "regression",
  n_features: 3,
  features: [
    {
      feature_name: "units",
      feature_type: "numeric",
      best_value: 20,
      worst_value: 1,
      best_prediction: 100,
      worst_prediction: 5,
      delta: 95,
      rank: 1,
    },
    {
      feature_name: "region",
      feature_type: "categorical",
      best_value: "East",
      worst_value: "West",
      best_prediction: 80,
      worst_prediction: 30,
      delta: 50,
      rank: 2,
    },
    {
      feature_name: "price",
      feature_type: "numeric",
      best_value: 100,
      worst_value: 10,
      best_prediction: 90,
      worst_prediction: 85,
      delta: 5,
      rank: 3,
    },
  ],
  optimal_config: { units: 20, region: "East", price: 100 },
  optimal_prediction: 120,
  summary: "Feature sweep across 3 features. 'units' has the largest prediction impact.",
}

describe("FeatureSweepCard", () => {
  it("renders the card title", () => {
    render(<FeatureSweepCard result={baseResult} />)
    expect(screen.getByText("Feature Impact Sweep")).toBeInTheDocument()
  })

  it("shows Maximize badge for direction=maximize", () => {
    render(<FeatureSweepCard result={baseResult} />)
    expect(screen.getByText("Maximize")).toBeInTheDocument()
  })

  it("shows Minimize badge for direction=minimize", () => {
    render(<FeatureSweepCard result={{ ...baseResult, direction: "minimize" }} />)
    expect(screen.getByText("Minimize")).toBeInTheDocument()
  })

  it("shows n_features badge", () => {
    render(<FeatureSweepCard result={baseResult} />)
    expect(screen.getByText("3 features")).toBeInTheDocument()
  })

  it("shows target column badge", () => {
    render(<FeatureSweepCard result={baseResult} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("shows optimal prediction value", () => {
    render(<FeatureSweepCard result={baseResult} />)
    expect(screen.getByText("120")).toBeInTheDocument()
  })

  it("shows feature names in impact ranking", () => {
    render(<FeatureSweepCard result={baseResult} />)
    expect(screen.getByText("units")).toBeInTheDocument()
    expect(screen.getByText("region")).toBeInTheDocument()
    expect(screen.getByText("price")).toBeInTheDocument()
  })

  it("shows feature type badges", () => {
    render(<FeatureSweepCard result={baseResult} />)
    const numericBadges = screen.getAllByText("numeric")
    expect(numericBadges.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText("categorical")).toBeInTheDocument()
  })

  it("shows rank badge 1 for top feature", () => {
    render(<FeatureSweepCard result={baseResult} />)
    expect(screen.getAllByText("1").length).toBeGreaterThanOrEqual(1)
  })

  it("shows optimal config feature badges", () => {
    render(<FeatureSweepCard result={baseResult} />)
    expect(screen.getByText("units=20")).toBeInTheDocument()
    expect(screen.getByText("region=East")).toBeInTheDocument()
  })

  it("shows classification percentage for classification problem", () => {
    const clsResult: FeatureSweepResult = {
      ...baseResult,
      problem_type: "classification",
      optimal_prediction: 0.87,
      features: [
        {
          ...baseResult.features[0],
          best_prediction: 0.9,
          worst_prediction: 0.4,
          delta: 0.5,
        },
      ],
    }
    render(<FeatureSweepCard result={clsResult} />)
    // optimal prediction as percentage
    expect(screen.getByText("87.0%")).toBeInTheDocument()
  })

  it("shows empty state message when no features", () => {
    const emptyResult: FeatureSweepResult = {
      ...baseResult,
      n_features: 0,
      features: [],
      summary: "Feature sweep could not be computed.",
    }
    render(<FeatureSweepCard result={emptyResult} />)
    expect(
      screen.getByText("Feature sweep could not be computed.")
    ).toBeInTheDocument()
  })

  it("truncates optimal config to 8 entries and shows +N more", () => {
    const manyFeaturesResult: FeatureSweepResult = {
      ...baseResult,
      optimal_config: {
        f1: 1, f2: 2, f3: 3, f4: 4, f5: 5, f6: 6, f7: 7, f8: 8, f9: 9,
      },
    }
    render(<FeatureSweepCard result={manyFeaturesResult} />)
    expect(screen.getByText("+1 more")).toBeInTheDocument()
  })

  it("renders the footer disclaimer text", () => {
    render(<FeatureSweepCard result={baseResult} />)
    expect(
      screen.getByText(/Each feature is swept independently/)
    ).toBeInTheDocument()
  })

  it("shows Optimal Maximum label for maximize", () => {
    render(<FeatureSweepCard result={baseResult} />)
    expect(screen.getByText("Optimal Maximum Prediction")).toBeInTheDocument()
  })

  it("shows Optimal Minimum label for minimize", () => {
    render(<FeatureSweepCard result={{ ...baseResult, direction: "minimize" }} />)
    expect(screen.getByText("Optimal Minimum Prediction")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action
// ---------------------------------------------------------------------------

describe("attachFeatureSweepToLastMessage store action", () => {
  it("attaches feature_sweep to last assistant message", () => {
    const { useAppStore } = require("@/lib/store")
    const store = useAppStore.getState()
    store.messages = [
      { role: "user", content: "sweep features" },
      { role: "assistant", content: "Here is the analysis." },
    ]
    store.attachFeatureSweepToLastMessage(baseResult)
    const last = useAppStore.getState().messages[1]
    expect(last.feature_sweep).toEqual(baseResult)
  })

  it("does not attach to a user message", () => {
    const { useAppStore } = require("@/lib/store")
    const store = useAppStore.getState()
    store.messages = [
      { role: "user", content: "sweep features" },
    ]
    store.attachFeatureSweepToLastMessage(baseResult)
    const last = useAppStore.getState().messages[0]
    expect(last.feature_sweep).toBeUndefined()
  })

  it("does nothing when messages list is empty", () => {
    const { useAppStore } = require("@/lib/store")
    const store = useAppStore.getState()
    store.messages = []
    expect(() => store.attachFeatureSweepToLastMessage(baseResult)).not.toThrow()
  })
})
