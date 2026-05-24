/**
 * Tests for proactive ensemble auto-suggest rendering in EnsembleRecommendationCard.
 *
 * Covers:
 *  1.  When auto_suggested=true, renders the auto-suggest banner (data-testid)
 *  2.  Banner has role="note" for a11y
 *  3.  Banner contains "AutoModeler" text
 *  4.  When auto_suggested=true, heading reads "Accuracy Below Target" variant
 *  5.  When auto_suggested is absent, renders standard "Ensemble Models" heading
 *  6.  When auto_suggested is false, renders standard "Ensemble Models" heading
 *  7.  When auto_suggested=true, banner emoji 💡 is aria-hidden
 *  8.  Normal card content still renders when auto_suggested=true (options, summary)
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { EnsembleRecommendationCard } from "@/components/models/ensemble-recommendation-card"
import type { EnsembleRecommendationResult } from "@/lib/types"

const BASE: EnsembleRecommendationResult = {
  problem_type: "regression",
  best_current_algorithm: "random_forest_regressor",
  best_current_score: 0.68,
  metric_name: "r2",
  dataset_size: 300,
  recommended_algorithm: "stacking_regressor",
  recommended_name: "Stacking Ensemble",
  options: [
    {
      algorithm: "voting_regressor",
      name: "Voting Ensemble",
      ensemble_type: "voting",
      description: "Averages predictions.",
      plain_english: "Gets a second opinion from three different models.",
      best_for: "Quick improvement",
      complexity: "easy",
      is_recommended: false,
    },
    {
      algorithm: "stacking_regressor",
      name: "Stacking Ensemble",
      ensemble_type: "stacking",
      description: "Uses a meta-learner.",
      plain_english: "Trains a meta-model on top of base models.",
      best_for: "Maximum accuracy",
      complexity: "medium",
      is_recommended: true,
    },
  ],
  summary:
    "Your best model score is below target. Stacking ensemble recommended.",
}

describe("EnsembleRecommendationCard — auto_suggested=true", () => {
  it("renders the auto-suggest banner", () => {
    render(<EnsembleRecommendationCard result={{ ...BASE, auto_suggested: true }} />)
    expect(screen.getByTestId("ensemble-auto-suggest-banner")).toBeInTheDocument()
  })

  it("banner has role=note", () => {
    render(<EnsembleRecommendationCard result={{ ...BASE, auto_suggested: true }} />)
    expect(screen.getByRole("note")).toBeInTheDocument()
  })

  it("banner contains 'AutoModeler' text", () => {
    render(<EnsembleRecommendationCard result={{ ...BASE, auto_suggested: true }} />)
    expect(screen.getByRole("note")).toHaveTextContent(/AutoModeler/i)
  })

  it("shows accuracy-below-target heading variant", () => {
    render(<EnsembleRecommendationCard result={{ ...BASE, auto_suggested: true }} />)
    expect(
      screen.getByRole("heading", { name: /Accuracy Below Target/i })
    ).toBeInTheDocument()
  })

  it("banner emoji 💡 is aria-hidden", () => {
    render(<EnsembleRecommendationCard result={{ ...BASE, auto_suggested: true }} />)
    const emoji = screen.getByText("💡")
    expect(emoji).toHaveAttribute("aria-hidden", "true")
  })

  it("still renders both option rows when auto_suggested", () => {
    render(<EnsembleRecommendationCard result={{ ...BASE, auto_suggested: true }} />)
    expect(screen.getByTestId("ensemble-option-voting")).toBeInTheDocument()
    expect(screen.getByTestId("ensemble-option-stacking")).toBeInTheDocument()
  })

  it("still renders ensemble summary when auto_suggested", () => {
    render(<EnsembleRecommendationCard result={{ ...BASE, auto_suggested: true }} />)
    expect(screen.getByTestId("ensemble-summary")).toBeInTheDocument()
  })
})

describe("EnsembleRecommendationCard — auto_suggested absent or false", () => {
  it("does not render banner when auto_suggested is absent", () => {
    render(<EnsembleRecommendationCard result={BASE} />)
    expect(
      screen.queryByTestId("ensemble-auto-suggest-banner")
    ).not.toBeInTheDocument()
  })

  it("shows standard heading when auto_suggested is absent", () => {
    render(<EnsembleRecommendationCard result={BASE} />)
    expect(
      screen.getByRole("heading", { name: /Ensemble Models/i })
    ).toBeInTheDocument()
  })

  it("does not render banner when auto_suggested=false", () => {
    render(<EnsembleRecommendationCard result={{ ...BASE, auto_suggested: false }} />)
    expect(
      screen.queryByTestId("ensemble-auto-suggest-banner")
    ).not.toBeInTheDocument()
  })

  it("shows standard heading when auto_suggested=false", () => {
    render(<EnsembleRecommendationCard result={{ ...BASE, auto_suggested: false }} />)
    expect(
      screen.getByRole("heading", { name: /Ensemble Models/i })
    ).toBeInTheDocument()
  })
})
