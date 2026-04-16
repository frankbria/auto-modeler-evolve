/**
 * Tests for EnsembleRecommendationCard and Zustand store action.
 *
 * Covers:
 *  1.  Renders figure with aria-label "Ensemble model recommendation"
 *  2.  Shows 🧩 icon (aria-hidden)
 *  3.  Shows "Ensemble Models" heading
 *  4.  Shows problem type badge (regression)
 *  5.  Shows current best score badge when present
 *  6.  Shows current best algorithm badge when present
 *  7.  Renders explanation callout containing "ensemble model" text
 *  8.  Renders recommendation summary (data-testid="ensemble-summary")
 *  9.  Renders voting option row (data-testid="ensemble-option-voting")
 * 10.  Renders stacking option row (data-testid="ensemble-option-stacking")
 * 11.  Shows "Recommended" badge on the recommended option
 * 12.  Shows "Easy" badge on voting option
 * 13.  Shows "Medium" badge on stacking option
 * 14.  Renders plain_english text for each option
 * 15.  Store: attachEnsembleRecommendationToLastMessage attaches to last assistant message
 * 16.  Store: does not attach to user message
 * 17.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { EnsembleRecommendationCard } from "@/components/models/ensemble-recommendation-card"
import type { EnsembleRecommendationResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const regressionResult: EnsembleRecommendationResult = {
  problem_type: "regression",
  best_current_algorithm: "random_forest_regressor",
  best_current_score: 0.84,
  metric_name: "r2",
  dataset_size: 300,
  recommended_algorithm: "stacking_regressor",
  recommended_name: "Stacking Ensemble",
  options: [
    {
      algorithm: "voting_regressor",
      name: "Voting Ensemble",
      ensemble_type: "voting",
      description: "Averages predictions from multiple models.",
      plain_english:
        "Gets a second opinion from three different models and averages their answers.",
      best_for: "Quick accuracy improvement",
      complexity: "easy",
      is_recommended: false,
    },
    {
      algorithm: "stacking_regressor",
      name: "Stacking Ensemble",
      ensemble_type: "stacking",
      description: "Uses a meta-learner to weight multiple models.",
      plain_english:
        "Trains three models, then trains a fourth model to learn the optimal combination.",
      best_for: "Maximum accuracy; needs 200+ rows",
      complexity: "medium",
      is_recommended: true,
    },
  ],
  summary:
    "Stacking ensemble recommended for your regression problem (current best R²: 0.840). Ensembles typically improve accuracy by 1-5%.",
}

const classificationResult: EnsembleRecommendationResult = {
  problem_type: "classification",
  best_current_algorithm: "random_forest_classifier",
  best_current_score: 0.91,
  metric_name: "accuracy",
  dataset_size: 100,
  recommended_algorithm: "voting_classifier",
  recommended_name: "Voting Classifier",
  options: [
    {
      algorithm: "voting_classifier",
      name: "Voting Classifier",
      ensemble_type: "voting",
      description: "Combines models by majority vote.",
      plain_english: "Asks three models to vote and picks the majority answer.",
      best_for: "Quick improvement; easy to explain",
      complexity: "easy",
      is_recommended: true,
    },
    {
      algorithm: "stacking_classifier",
      name: "Stacking Classifier",
      ensemble_type: "stacking",
      description: "Uses a meta-learner to combine models.",
      plain_english:
        "Trains three models, then trains a fourth model to optimally combine them.",
      best_for: "Maximum accuracy; needs 200+ rows",
      complexity: "medium",
      is_recommended: false,
    },
  ],
  summary:
    "Voting Classifier recommended for your classification problem. Dataset is small; voting trains faster.",
}

describe("EnsembleRecommendationCard", () => {
  it("renders figure with correct aria-label", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    expect(
      screen.getByRole("figure", { name: "Ensemble model recommendation" })
    ).toBeInTheDocument()
  })

  it("shows 🧩 icon as aria-hidden", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    const icon = screen.getByText("🧩")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("shows Ensemble Models heading", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    expect(screen.getByRole("heading", { name: /Ensemble Models/i })).toBeInTheDocument()
  })

  it("shows problem type badge (regression)", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    expect(screen.getByText("regression")).toBeInTheDocument()
  })

  it("shows problem type badge (classification)", () => {
    render(<EnsembleRecommendationCard result={classificationResult} />)
    expect(screen.getByText("classification")).toBeInTheDocument()
  })

  it("shows current best score badge", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    expect(screen.getByText(/Current best: R2 0\.840/i)).toBeInTheDocument()
  })

  it("shows current best algorithm badge", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    expect(screen.getByText(/Random Forest Regressor/i)).toBeInTheDocument()
  })

  it("renders explanation callout about ensemble models", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    expect(screen.getByText(/What is an ensemble model/i)).toBeInTheDocument()
  })

  it("renders recommendation summary via data-testid", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    const summary = screen.getByTestId("ensemble-summary")
    expect(summary).toBeInTheDocument()
    expect(summary.textContent).toContain("Stacking Ensemble")
  })

  it("renders voting option row", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    expect(screen.getByTestId("ensemble-option-voting")).toBeInTheDocument()
  })

  it("renders stacking option row", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    expect(screen.getByTestId("ensemble-option-stacking")).toBeInTheDocument()
  })

  it("shows Recommended badge on the recommended option (stacking)", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    const stackingRow = screen.getByTestId("ensemble-option-stacking")
    expect(stackingRow).toHaveTextContent("Recommended")
  })

  it("shows Easy badge on voting option", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    const votingRow = screen.getByTestId("ensemble-option-voting")
    expect(votingRow).toHaveTextContent("Easy")
  })

  it("shows Medium badge on stacking option", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    const stackingRow = screen.getByTestId("ensemble-option-stacking")
    expect(stackingRow).toHaveTextContent("Medium")
  })

  it("renders plain_english text for voting option", () => {
    render(<EnsembleRecommendationCard result={regressionResult} />)
    expect(
      screen.getByText(/Gets a second opinion from three different models/i)
    ).toBeInTheDocument()
  })

  it("shows Recommended on voting option when classification small dataset", () => {
    render(<EnsembleRecommendationCard result={classificationResult} />)
    const votingRow = screen.getByTestId("ensemble-option-voting")
    expect(votingRow).toHaveTextContent("Recommended")
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachEnsembleRecommendationToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "Should I use an ensemble?", timestamp: "" },
        { role: "assistant", content: "Let me check.", timestamp: "" },
      ],
    })
    useAppStore.getState().attachEnsembleRecommendationToLastMessage(regressionResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].ensemble_recommendation).toEqual(regressionResult)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "Should I use an ensemble?", timestamp: "" },
      ],
    })
    useAppStore.getState().attachEnsembleRecommendationToLastMessage(regressionResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].ensemble_recommendation).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    expect(() => {
      useAppStore.getState().attachEnsembleRecommendationToLastMessage(regressionResult)
    }).not.toThrow()
  })
})
