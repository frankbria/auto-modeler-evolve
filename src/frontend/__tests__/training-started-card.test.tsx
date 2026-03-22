/**
 * Tests for TrainingStartedCard component and attachTrainingStartedToLastMessage store action.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { TrainingStartedCard } from "@/components/models/training-started-card"
import type { TrainingStartedResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Fixtures -----------------------------------------------------------

const trainingStarted: TrainingStartedResult = {
  project_id: "proj-1",
  target_column: "revenue",
  problem_type: "regression",
  algorithms: ["linear_regression", "random_forest", "gradient_boosting"],
  run_count: 3,
  status: "started",
}

const classificationResult: TrainingStartedResult = {
  project_id: "proj-2",
  target_column: "churn",
  problem_type: "classification",
  algorithms: ["logistic_regression", "random_forest_classifier"],
  run_count: 2,
  status: "started",
}

// --- Component tests ----------------------------------------------------

describe("TrainingStartedCard", () => {
  it("renders with testid", () => {
    render(<TrainingStartedCard result={trainingStarted} />)
    expect(screen.getByTestId("training-started-card")).toBeInTheDocument()
  })

  it("shows 'Training Started' heading", () => {
    render(<TrainingStartedCard result={trainingStarted} />)
    expect(screen.getByText(/Training Started/i)).toBeInTheDocument()
  })

  it("shows the target column name", () => {
    render(<TrainingStartedCard result={trainingStarted} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("shows run count", () => {
    render(<TrainingStartedCard result={trainingStarted} />)
    expect(screen.getByText(/3/)).toBeInTheDocument()
  })

  it("shows regression problem type badge", () => {
    render(<TrainingStartedCard result={trainingStarted} />)
    expect(screen.getAllByText(/Regression/i).length).toBeGreaterThanOrEqual(1)
  })

  it("shows classification problem type badge", () => {
    render(<TrainingStartedCard result={classificationResult} />)
    expect(screen.getAllByText(/Classification/i).length).toBeGreaterThanOrEqual(1)
  })

  it("shows Models tab hint text", () => {
    render(<TrainingStartedCard result={trainingStarted} />)
    expect(screen.getByText(/Models tab/i)).toBeInTheDocument()
  })

  it("renders algorithm labels", () => {
    render(<TrainingStartedCard result={trainingStarted} />)
    // All 3 algorithms should appear as individual chips
    expect(screen.getAllByText(/Linear Regression|Random Forest|Gradient Boosting/).length).toBe(3)
  })

  it("handles single model run count", () => {
    const single = { ...trainingStarted, run_count: 1, algorithms: ["linear_regression"] }
    render(<TrainingStartedCard result={single} />)
    expect(screen.getByText(/1/)).toBeInTheDocument()
    // Should say "model" not "models"
    expect(screen.getByText(/model to predict/i)).toBeInTheDocument()
  })
})

// --- Store action tests -------------------------------------------------

describe("attachTrainingStartedToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches training_started to the last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "train a model to predict revenue", timestamp: "t1" },
        { role: "assistant", content: "Starting training now!", timestamp: "t2" },
      ],
    })
    const { attachTrainingStartedToLastMessage } = useAppStore.getState()
    attachTrainingStartedToLastMessage(trainingStarted)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].training_started).toEqual(trainingStarted)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "train a model", timestamp: "t1" },
      ],
    })
    const { attachTrainingStartedToLastMessage } = useAppStore.getState()
    attachTrainingStartedToLastMessage(trainingStarted)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].training_started).toBeUndefined()
  })

  it("is a no-op on empty message list", () => {
    useAppStore.setState({ messages: [] })
    const { attachTrainingStartedToLastMessage } = useAppStore.getState()
    expect(() => attachTrainingStartedToLastMessage(trainingStarted)).not.toThrow()
  })
})
