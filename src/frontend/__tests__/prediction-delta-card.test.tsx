/**
 * Tests for PredictionDeltaCard component and prediction_delta store action.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { PredictionDeltaCard } from "@/components/deploy/prediction-delta-card"
import type { PredictionDeltaResult } from "@/lib/types"

// ── Fixtures ──────────────────────────────────────────────────────────────────

const regressionUp: PredictionDeltaResult = {
  old_prediction: 4500.0,
  new_prediction: 5250.0,
  delta: 750.0,
  pct_change: 16.7,
  direction: "up",
  problem_type: "regression",
  target_column: "revenue",
  provided_features: { units: 150, region: 1 },
  feature_delta: [
    {
      feature: "units",
      old_contribution: 0.12,
      new_contribution: 0.41,
      contribution_delta: 0.29,
      direction: "increased",
    },
    {
      feature: "region",
      old_contribution: 0.08,
      new_contribution: 0.05,
      contribution_delta: -0.03,
      direction: "decreased",
    },
    {
      feature: "discount",
      old_contribution: 0.05,
      new_contribution: 0.05,
      contribution_delta: 0.0,
      direction: "stable",
    },
  ],
  top_drivers: ["units ↑", "region ↓"],
  summary:
    "The revenue prediction increased from 4,500 to 5,250 (+750, +16.7%). Top drivers of change: units ↑, region ↓.",
  current_version: 2,
  previous_version: 1,
  inputs_from_message: true,
}

const regressionDown: PredictionDeltaResult = {
  ...regressionUp,
  old_prediction: 5250.0,
  new_prediction: 4200.0,
  delta: -1050.0,
  pct_change: -20.0,
  direction: "down",
  summary: "The revenue prediction decreased from 5,250 to 4,200 (-1050, -20.0%).",
}

const regressionUnchanged: PredictionDeltaResult = {
  ...regressionUp,
  delta: 0.0,
  pct_change: 0.0,
  direction: "unchanged",
  old_prediction: 5000.0,
  new_prediction: 5000.0,
  top_drivers: [],
  feature_delta: [
    {
      feature: "units",
      old_contribution: 0.12,
      new_contribution: 0.12,
      contribution_delta: 0.0,
      direction: "stable",
    },
  ],
  summary: "The revenue prediction remained unchanged at 5000 after retraining.",
}

const classificationChanged: PredictionDeltaResult = {
  old_prediction: "no_churn",
  new_prediction: "churn",
  delta: 0.0,
  pct_change: 0.0,
  direction: "changed",
  problem_type: "classification",
  target_column: "churn",
  provided_features: { tenure: 12, monthly_charges: 89 },
  feature_delta: [
    {
      feature: "monthly_charges",
      old_contribution: 0.18,
      new_contribution: 0.45,
      contribution_delta: 0.27,
      direction: "increased",
    },
  ],
  top_drivers: ["monthly charges ↑"],
  summary:
    "The predicted class changed from 'no_churn' to 'churn' after retraining. Top feature shifts: monthly charges ↑.",
  current_version: 3,
  previous_version: 2,
  inputs_from_message: false,
}

// ── Rendering tests ───────────────────────────────────────────────────────────

describe("PredictionDeltaCard", () => {
  it("renders the card region with correct aria-label", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(
      screen.getByRole("region", { name: /prediction delta analysis/i })
    ).toBeInTheDocument()
  })

  it("shows the card title", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByText("Why Did the Prediction Change?")).toBeInTheDocument()
  })

  it("shows version badge with previous → current version", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByTestId("version-badge")).toHaveTextContent("v1 → v2")
  })

  it("shows correct version numbers for classification result", () => {
    render(<PredictionDeltaCard result={classificationChanged} />)
    expect(screen.getByTestId("version-badge")).toHaveTextContent("v2 → v3")
  })

  it("shows direction badge as Increased when direction=up", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByTestId("direction-badge")).toHaveTextContent("Increased")
  })

  it("shows direction badge as Decreased when direction=down", () => {
    render(<PredictionDeltaCard result={regressionDown} />)
    expect(screen.getByTestId("direction-badge")).toHaveTextContent("Decreased")
  })

  it("shows direction badge as Unchanged when direction=unchanged", () => {
    render(<PredictionDeltaCard result={regressionUnchanged} />)
    expect(screen.getByTestId("direction-badge")).toHaveTextContent("Unchanged")
  })

  it("shows direction badge as Class changed for classification change", () => {
    render(<PredictionDeltaCard result={classificationChanged} />)
    expect(screen.getByTestId("direction-badge")).toHaveTextContent("Class changed")
  })

  it("shows target column in badge", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByText("Target: revenue")).toBeInTheDocument()
  })

  it("shows old prediction value", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByTestId("old-prediction")).toHaveTextContent("4,500")
  })

  it("shows new prediction value", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByTestId("new-prediction")).toHaveTextContent("5,250")
  })

  it("shows delta with sign and percentage for regression up", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByTestId("delta-display")).toHaveTextContent("+750")
    expect(screen.getByTestId("delta-display")).toHaveTextContent("+16.7%")
  })

  it("shows negative delta for regression down", () => {
    render(<PredictionDeltaCard result={regressionDown} />)
    expect(screen.getByTestId("delta-display")).toHaveTextContent("-1,050")
    expect(screen.getByTestId("delta-display")).toHaveTextContent("-20.0%")
  })

  it("does not show delta-display when direction=unchanged", () => {
    render(<PredictionDeltaCard result={regressionUnchanged} />)
    expect(screen.queryByTestId("delta-display")).not.toBeInTheDocument()
  })

  it("renders feature delta rows", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    const rows = screen.getAllByTestId("feature-delta-row")
    expect(rows.length).toBeGreaterThanOrEqual(1)
  })

  it("shows feature names in delta rows", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByText("units")).toBeInTheDocument()
    expect(screen.getByText("region")).toBeInTheDocument()
  })

  it("shows top drivers chips", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByTestId("top-drivers")).toBeInTheDocument()
    expect(screen.getByText("units ↑")).toBeInTheDocument()
    expect(screen.getByText("region ↓")).toBeInTheDocument()
  })

  it("does not render top drivers section when top_drivers is empty", () => {
    render(<PredictionDeltaCard result={regressionUnchanged} />)
    expect(screen.queryByTestId("top-drivers")).not.toBeInTheDocument()
  })

  it("shows inputs from message label when inputs_from_message=true", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByTestId("inputs-summary")).toHaveTextContent(
      "Inputs from your message:"
    )
  })

  it("shows training averages label when inputs_from_message=false", () => {
    render(<PredictionDeltaCard result={classificationChanged} />)
    expect(screen.getByTestId("inputs-summary")).toHaveTextContent(
      "Using training-data averages:"
    )
  })

  it("shows summary text", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByTestId("prediction-delta-summary")).toHaveTextContent(
      "revenue prediction increased"
    )
  })

  it("renders sr-only figcaption for accessibility", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    const figcaption = document.querySelector("figcaption.sr-only")
    expect(figcaption).toBeInTheDocument()
    expect(figcaption?.textContent).toContain("version 1")
    expect(figcaption?.textContent).toContain("to 2")
  })

  it("renders classification predictions as strings", () => {
    render(<PredictionDeltaCard result={classificationChanged} />)
    expect(screen.getByTestId("old-prediction")).toHaveTextContent("no_churn")
    expect(screen.getByTestId("new-prediction")).toHaveTextContent("churn")
  })

  it("shows testid=prediction-delta-card on root element", () => {
    render(<PredictionDeltaCard result={regressionUp} />)
    expect(screen.getByTestId("prediction-delta-card")).toBeInTheDocument()
  })
})

// ── Store action test ─────────────────────────────────────────────────────────

describe("attachPredictionDeltaToLastMessage store action", () => {
  it("attaches prediction_delta to the last assistant message", () => {
    const { useAppStore } = jest.requireActual<
      typeof import("@/lib/store")
    >("@/lib/store")

    const store = useAppStore.getState()
    store.addMessage({ role: "user", content: "why did prediction change?" })
    store.addMessage({ role: "assistant", content: "Let me explain..." })

    store.attachPredictionDeltaToLastMessage(regressionUp)

    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.prediction_delta).toBeDefined()
    expect(last.prediction_delta?.target_column).toBe("revenue")
    expect(last.prediction_delta?.current_version).toBe(2)
    expect(last.prediction_delta?.previous_version).toBe(1)
  })

  it("does not attach prediction_delta to user message", () => {
    const { useAppStore } = jest.requireActual<
      typeof import("@/lib/store")
    >("@/lib/store")

    const store = useAppStore.getState()
    store.addMessage({ role: "user", content: "why did prediction change?" })
    // No assistant message

    store.attachPredictionDeltaToLastMessage(regressionUp)

    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.prediction_delta).toBeUndefined()
  })
})
