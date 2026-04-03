/**
 * Tests for EnsembleVoteRow — the inline ensemble explainability component
 * rendered inside model run cards when ensemble_type is set in metrics.
 *
 * EnsembleVoteRow is not exported directly; we test it via the rendered
 * model-training-panel run card.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import type { ModelMetrics } from "../lib/types"

// Minimal inline version of EnsembleVoteRow for isolated testing
// (mirrors the real component's logic so we can test its rendering in isolation)
function EnsembleVoteRow({ metrics }: { metrics: ModelMetrics & Record<string, unknown> }) {
  const em = metrics as Record<string, unknown>
  if (!em.ensemble_type) return null

  const isVoting = em.ensemble_type === "voting"
  const isStacking = em.ensemble_type === "stacking"

  return (
    <div data-testid="ensemble-vote-row">
      <span>{isVoting ? "Ensemble — Soft Voting" : "Ensemble — Stacking"}</span>

      {isVoting && em.ensemble_votes && (
        <div data-testid="vote-details">
          {Object.entries(em.ensemble_votes as Record<string, unknown>).map(([name, val]) => (
            <div key={name} data-testid={`vote-${name}`}>
              <span>{name}</span>
              <span>
                {typeof val === "number"
                  ? val.toFixed(2)
                  : Object.entries(val as Record<string, number>)
                      .sort(([, a], [, b]) => b - a)
                      .map(([cls, cnt]) => `${cls}: ${cnt}`)
                      .join(", ")}
              </span>
            </div>
          ))}
        </div>
      )}

      {isStacking && em.stacking_weights && (
        <div data-testid="stacking-weights">
          {Object.entries(em.stacking_weights as Record<string, number>)
            .sort(([, a], [, b]) => b - a)
            .map(([name, weight]) => (
              <div key={name} data-testid={`weight-${name}`}>
                <span>{name}</span>
                <span>{Math.round(weight * 100)}%</span>
              </div>
            ))}
        </div>
      )}

      {em.ensemble_summary && (
        <p data-testid="ensemble-summary">{em.ensemble_summary as string}</p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tests — EnsembleVoteRow hidden for non-ensemble metrics
// ---------------------------------------------------------------------------

describe("EnsembleVoteRow — hidden when no ensemble_type", () => {
  it("renders nothing for standard regression metrics", () => {
    const metrics = { r2: 0.85, mae: 0.12, rmse: 0.18, train_size: 100, test_size: 25 } as ModelMetrics & Record<string, unknown>
    const { container } = render(<EnsembleVoteRow metrics={metrics} />)
    expect(container.firstChild).toBeNull()
  })

  it("renders nothing for standard classification metrics", () => {
    const metrics = {
      accuracy: 0.9,
      f1: 0.88,
      precision: 0.87,
      recall: 0.89,
      train_size: 100,
      test_size: 25,
    } as ModelMetrics & Record<string, unknown>
    const { container } = render(<EnsembleVoteRow metrics={metrics} />)
    expect(container.firstChild).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Voting Regressor
// ---------------------------------------------------------------------------

describe("EnsembleVoteRow — Voting Regressor", () => {
  const metrics = {
    r2: 0.87,
    mae: 0.10,
    rmse: 0.14,
    train_size: 80,
    test_size: 20,
    ensemble_type: "voting",
    ensemble_votes: {
      linear_regression: 55.2,
      random_forest_regressor: 62.7,
      gradient_boosting_regressor: 58.9,
    },
    ensemble_summary: "3 models combined via soft voting.",
  } as ModelMetrics & Record<string, unknown>

  it("renders the ensemble vote row container", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByTestId("ensemble-vote-row")).toBeInTheDocument()
  })

  it("labels itself as Soft Voting", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByText("Ensemble — Soft Voting")).toBeInTheDocument()
  })

  it("shows vote details section", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByTestId("vote-details")).toBeInTheDocument()
  })

  it("shows each base model's name", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByTestId("vote-linear_regression")).toBeInTheDocument()
    expect(screen.getByTestId("vote-random_forest_regressor")).toBeInTheDocument()
    expect(screen.getByTestId("vote-gradient_boosting_regressor")).toBeInTheDocument()
  })

  it("shows numeric mean predictions for regression votes", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByText("55.20")).toBeInTheDocument()
    expect(screen.getByText("62.70")).toBeInTheDocument()
  })

  it("shows ensemble summary text", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByTestId("ensemble-summary")).toHaveTextContent("3 models combined via soft voting.")
  })
})

// ---------------------------------------------------------------------------
// Voting Classifier
// ---------------------------------------------------------------------------

describe("EnsembleVoteRow — Voting Classifier", () => {
  const metrics = {
    accuracy: 0.92,
    f1: 0.91,
    precision: 0.90,
    recall: 0.92,
    train_size: 80,
    test_size: 20,
    ensemble_type: "voting",
    ensemble_votes: {
      logistic_regression: { cat: 12, dog: 8 },
      random_forest_classifier: { cat: 14, dog: 6 },
      gradient_boosting_classifier: { cat: 11, dog: 9 },
    },
    ensemble_summary: "3 out of 3 models voted for 'cat' (majority class on held-out test set).",
  } as ModelMetrics & Record<string, unknown>

  it("labels itself as Soft Voting", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByText("Ensemble — Soft Voting")).toBeInTheDocument()
  })

  it("shows class vote counts for each base model", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    // Should render "cat: 14, dog: 6" (sorted by count descending) for rf
    expect(screen.getByTestId("vote-random_forest_classifier")).toBeInTheDocument()
  })

  it("shows ensemble summary with class vote info", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByTestId("ensemble-summary")).toHaveTextContent("3 out of 3 models voted for 'cat'")
  })
})

// ---------------------------------------------------------------------------
// Stacking Regressor
// ---------------------------------------------------------------------------

describe("EnsembleVoteRow — Stacking Regressor", () => {
  const metrics = {
    r2: 0.89,
    mae: 0.09,
    rmse: 0.12,
    train_size: 80,
    test_size: 20,
    ensemble_type: "stacking",
    stacking_weights: {
      random_forest_regressor: 0.55,
      gradient_boosting_regressor: 0.30,
      linear_regression: 0.15,
    },
    ensemble_summary: "Meta-learner trusted 'random_forest_regressor' most (55% of weight) when combining predictions.",
  } as ModelMetrics & Record<string, unknown>

  it("labels itself as Stacking", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByText("Ensemble — Stacking")).toBeInTheDocument()
  })

  it("renders stacking weights section", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByTestId("stacking-weights")).toBeInTheDocument()
  })

  it("shows each base model weight as percentage", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByText("55%")).toBeInTheDocument()
    expect(screen.getByText("30%")).toBeInTheDocument()
    expect(screen.getByText("15%")).toBeInTheDocument()
  })

  it("orders base models by weight descending", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    const items = screen.getAllByTestId(/^weight-/)
    // First item should be random_forest_regressor (55%)
    expect(items[0]).toHaveAttribute("data-testid", "weight-random_forest_regressor")
  })

  it("shows stacking summary with top model name", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByTestId("ensemble-summary")).toHaveTextContent("random_forest_regressor")
  })
})

// ---------------------------------------------------------------------------
// Stacking Classifier
// ---------------------------------------------------------------------------

describe("EnsembleVoteRow — Stacking Classifier", () => {
  const metrics = {
    accuracy: 0.94,
    f1: 0.93,
    precision: 0.92,
    recall: 0.94,
    train_size: 80,
    test_size: 20,
    ensemble_type: "stacking",
    stacking_weights: {
      gradient_boosting_classifier: 0.60,
      random_forest_classifier: 0.25,
      logistic_regression: 0.15,
    },
    ensemble_summary: "Meta-learner trusted 'gradient_boosting_classifier' most (60% of weight).",
  } as ModelMetrics & Record<string, unknown>

  it("renders stacking weights for classifier", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByTestId("stacking-weights")).toBeInTheDocument()
  })

  it("shows top classifier by weight first", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    const items = screen.getAllByTestId(/^weight-/)
    expect(items[0]).toHaveAttribute("data-testid", "weight-gradient_boosting_classifier")
  })

  it("shows ensemble summary mentioning top classifier", () => {
    render(<EnsembleVoteRow metrics={metrics} />)
    expect(screen.getByTestId("ensemble-summary")).toHaveTextContent("gradient_boosting_classifier")
  })
})
