/**
 * Tests for ModelCardView component and attachModelCardToLastMessage store action.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { ModelCardView } from "@/components/models/model-card-view"
import type { ModelCard } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Fixtures ---------------------------------------------------------------

const regressionCard: ModelCard = {
  project_id: "proj-1",
  model_run_id: "run-1",
  algorithm: "linear_regression",
  algorithm_name: "Linear Regression",
  problem_type: "regression",
  target_col: "revenue",
  row_count: 200,
  feature_count: 4,
  metric: {
    name: "R²",
    value: 0.85,
    display: "85.0%",
    plain_english: "R² = 85.0% — good — explains most patterns in your data",
  },
  top_features: [
    { feature: "units", importance: 0.45, rank: 1 },
    { feature: "cost", importance: 0.35, rank: 2 },
    { feature: "region", importance: 0.20, rank: 3 },
  ],
  limitations: ["Predictions reflect patterns in training data only"],
  summary:
    "Your Linear Regression model predicts 'revenue' with 85.0% R². R² = 85.0% — good — explains most patterns in your data.",
  is_selected: true,
  is_deployed: false,
}

const classificationCard: ModelCard = {
  project_id: "proj-2",
  model_run_id: "run-2",
  algorithm: "random_forest_classifier",
  algorithm_name: "Random Forest",
  problem_type: "classification",
  target_col: "churned",
  row_count: 50,
  feature_count: 3,
  metric: {
    name: "Accuracy",
    value: 0.92,
    display: "92.0%",
    plain_english: "Predicts correctly about 9 out of 10 times",
  },
  top_features: [],
  limitations: ["Trained on only 50 rows — more data would improve reliability"],
  summary: "Your Random Forest model predicts 'churned' with 92.0% Accuracy.",
  is_selected: true,
  is_deployed: true,
}

// --- Rendering tests --------------------------------------------------------

describe("ModelCardView", () => {
  it("renders with testid", () => {
    render(<ModelCardView card={regressionCard} />)
    expect(screen.getByTestId("model-card-view")).toBeInTheDocument()
  })

  it("shows Model Explained header", () => {
    render(<ModelCardView card={regressionCard} />)
    expect(screen.getByText("Model Explained")).toBeInTheDocument()
  })

  it("shows algorithm name", () => {
    render(<ModelCardView card={regressionCard} />)
    expect(screen.getByText("Linear Regression")).toBeInTheDocument()
  })

  it("shows regression problem type badge", () => {
    render(<ModelCardView card={regressionCard} />)
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("shows classification problem type badge", () => {
    render(<ModelCardView card={classificationCard} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("shows Live badge when deployed", () => {
    render(<ModelCardView card={classificationCard} />)
    expect(screen.getByText("Live")).toBeInTheDocument()
  })

  it("does not show Live badge when not deployed", () => {
    render(<ModelCardView card={regressionCard} />)
    expect(screen.queryByText("Live")).not.toBeInTheDocument()
  })

  it("shows metric display value", () => {
    render(<ModelCardView card={regressionCard} />)
    expect(screen.getByText("85.0%")).toBeInTheDocument()
  })

  it("shows metric plain English explanation", () => {
    render(<ModelCardView card={regressionCard} />)
    const elements = screen.getAllByText(/explains most patterns in your data/i)
    expect(elements.length).toBeGreaterThan(0)
  })

  it("shows feature importance bars", () => {
    render(<ModelCardView card={regressionCard} />)
    const bars = screen.getAllByTestId("importance-bar")
    expect(bars).toHaveLength(3)
  })

  it("does not render feature bars when top_features is empty", () => {
    render(<ModelCardView card={classificationCard} />)
    expect(screen.queryByTestId("importance-bar")).not.toBeInTheDocument()
  })

  it("shows limitation text", () => {
    render(<ModelCardView card={classificationCard} />)
    expect(screen.getByText(/50 rows/i)).toBeInTheDocument()
  })

  it("shows target column", () => {
    render(<ModelCardView card={regressionCard} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("shows row count", () => {
    render(<ModelCardView card={regressionCard} />)
    expect(screen.getByText("200")).toBeInTheDocument()
  })
})

// --- Store action tests ----------------------------------------------------

describe("attachModelCardToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        {
          role: "assistant",
          content: "Let me explain your model.",
          timestamp: "2026-03-23T04:00:00",
        },
      ],
    })
  })

  it("attaches model_card to last assistant message", () => {
    const { attachModelCardToLastMessage } = useAppStore.getState()
    attachModelCardToLastMessage(regressionCard)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].model_card).toEqual(regressionCard)
  })

  it("does not modify non-assistant messages", () => {
    useAppStore.setState({
      messages: [
        {
          role: "user",
          content: "explain my model",
          timestamp: "2026-03-23T04:00:00",
        },
      ],
    })
    const { attachModelCardToLastMessage } = useAppStore.getState()
    attachModelCardToLastMessage(regressionCard)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].model_card).toBeUndefined()
  })
})
