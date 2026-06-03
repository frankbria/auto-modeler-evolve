/**
 * Tests for DeploymentPredictionDistributionCard and Zustand store action.
 *
 * Covers:
 *  1.  Renders region with aria-label "Deployment prediction distribution comparison"
 *  2.  Shows 📊 icon (aria-hidden)
 *  3.  Shows "Deployment Prediction Comparison" heading
 *  4.  Shows verdict badge for current_higher
 *  5.  Shows verdict badge for current_lower
 *  6.  Shows "similar" badge for similar verdict
 *  7.  Shows problem_type badge (Regression)
 *  8.  Shows n_baseline / n_current badge
 *  9.  Shows algorithm labels for baseline and current
 * 10.  Shows target_col label
 * 11.  Renders stat grids (mean, median, std) for regression
 * 12.  Shows mean shift highlight row for regression
 * 13.  Shows "Previous deployment" and "Current deployment" labels for regression
 * 14.  Renders class shifts table for classification
 * 15.  Shows verdict badge for distribution_shifted
 * 16.  Renders no_data state with summary text
 * 17.  Store: attachDeployPredDistCompareToLastMessage attaches to last assistant message
 * 18.  Store: does not attach to user message
 * 19.  Store: does not crash on empty messages list
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { DeploymentPredictionDistributionCard } from "@/components/deploy/deployment-prediction-distribution-card"
import type { DeploymentPredictionComparisonResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const regressionResult: DeploymentPredictionComparisonResult = {
  verdict: "current_higher",
  verdict_label: "New deployment predicts higher values (+20.0%)",
  problem_type: "regression",
  n_baseline: 50,
  n_current: 60,
  summary: "Current deployment mean is 20% higher than baseline.",
  current_algorithm: "random_forest_regressor",
  baseline_algorithm: "linear_regression",
  target_col: "revenue",
  mean_shift: 100.0,
  mean_shift_pct: 20.0,
  baseline_stats: { mean: 500.0, median: 490.0, std: 50.0, min: 400.0, max: 620.0, p25: 460.0, p75: 540.0, n: 50 },
  current_stats: { mean: 600.0, median: 595.0, std: 55.0, min: 480.0, max: 720.0, p25: 550.0, p75: 650.0, n: 60 },
}

const classificationResult: DeploymentPredictionComparisonResult = {
  verdict: "distribution_shifted",
  verdict_label: "Class 'churn' increased by 15.0 percentage points",
  problem_type: "classification",
  n_baseline: 40,
  n_current: 45,
  summary: "Class distribution shifted — churn class up 15pp.",
  current_algorithm: "gradient_boosting_classifier",
  baseline_algorithm: "logistic_regression",
  target_col: "churn",
  class_shifts: [
    { class_label: "churn", baseline_pct: 20.0, current_pct: 35.0, shift_pct: 15.0 },
    { class_label: "no_churn", baseline_pct: 80.0, current_pct: 65.0, shift_pct: -15.0 },
  ],
  n_classes: 2,
}

const noDataResult: DeploymentPredictionComparisonResult = {
  verdict: "no_data",
  verdict_label: "No data",
  problem_type: "regression",
  n_baseline: 2,
  n_current: 3,
  summary: "Not enough predictions to compare: 3 current, 2 baseline. Need at least 5 in each.",
}

const lowerResult: DeploymentPredictionComparisonResult = {
  verdict: "current_lower",
  verdict_label: "New deployment predicts lower values (-12.0%)",
  problem_type: "regression",
  n_baseline: 10,
  n_current: 10,
  summary: "Current is 12% lower.",
  mean_shift: -12.0,
  mean_shift_pct: -12.0,
  baseline_stats: { mean: 100.0, median: 100.0, std: 5.0, min: 90.0, max: 110.0, p25: 95.0, p75: 105.0, n: 10 },
  current_stats: { mean: 88.0, median: 88.0, std: 5.0, min: 80.0, max: 96.0, p25: 84.0, p75: 92.0, n: 10 },
}

describe("DeploymentPredictionDistributionCard", () => {
  it("1. renders region with aria-label", () => {
    render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    expect(screen.getByRole("region", { name: /deployment prediction distribution comparison/i })).toBeInTheDocument()
  })

  it("2. shows 📊 icon aria-hidden", () => {
    const { container } = render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    const icon = container.querySelector("[aria-hidden='true']")
    expect(icon).toBeTruthy()
    expect(icon?.textContent).toContain("📊")
  })

  it("3. shows Deployment Prediction Comparison heading", () => {
    render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    expect(screen.getByText(/Deployment Prediction Comparison/i)).toBeInTheDocument()
  })

  it("4. shows verdict badge for current_higher", () => {
    render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    const els = screen.getAllByText(/predicts higher values/i)
    expect(els.length).toBeGreaterThanOrEqual(1)
  })

  it("5. shows verdict badge for current_lower", () => {
    render(<DeploymentPredictionDistributionCard result={lowerResult} />)
    const els = screen.getAllByText(/predicts lower values/i)
    expect(els.length).toBeGreaterThanOrEqual(1)
  })

  it("6. shows similar badge for similar verdict", () => {
    render(<DeploymentPredictionDistributionCard result={{ ...regressionResult, verdict: "similar", verdict_label: "Similar prediction levels" }} />)
    const els = screen.getAllByText(/similar prediction levels/i)
    expect(els.length).toBeGreaterThanOrEqual(1)
  })

  it("7. shows Regression problem type badge", () => {
    render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("8. shows n_baseline / n_current badge", () => {
    render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    expect(screen.getByText(/50 baseline \/ 60 current/)).toBeInTheDocument()
  })

  it("9. shows algorithm labels", () => {
    render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    expect(screen.getByText(/random_forest_regressor/i)).toBeInTheDocument()
    expect(screen.getByText(/linear_regression/i)).toBeInTheDocument()
  })

  it("10. shows target_col label", () => {
    render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    expect(screen.getByText(/revenue/i)).toBeInTheDocument()
  })

  it("11. renders stat grids with mean label", () => {
    render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    const meanLabels = screen.getAllByText("Mean")
    expect(meanLabels.length).toBeGreaterThanOrEqual(2)
  })

  it("12. shows mean shift highlight", () => {
    render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    expect(screen.getByText(/Mean shift/i)).toBeInTheDocument()
  })

  it("13. shows Previous and Current deployment labels", () => {
    render(<DeploymentPredictionDistributionCard result={regressionResult} />)
    expect(screen.getByText("Previous deployment")).toBeInTheDocument()
    expect(screen.getByText("Current deployment")).toBeInTheDocument()
  })

  it("14. renders class shifts table for classification", () => {
    render(<DeploymentPredictionDistributionCard result={classificationResult} />)
    expect(screen.getByRole("table", { name: /class distribution comparison/i })).toBeInTheDocument()
    expect(screen.getByText("churn")).toBeInTheDocument()
    expect(screen.getByText("no_churn")).toBeInTheDocument()
  })

  it("15. shows distribution_shifted verdict badge", () => {
    render(<DeploymentPredictionDistributionCard result={classificationResult} />)
    const els = screen.getAllByText(/Class 'churn' increased/i)
    expect(els.length).toBeGreaterThanOrEqual(1)
  })

  it("16. renders no_data state with summary text", () => {
    render(<DeploymentPredictionDistributionCard result={noDataResult} />)
    const els = screen.getAllByText(/Not enough predictions to compare/i)
    expect(els.length).toBeGreaterThanOrEqual(1)
  })

  describe("store action", () => {
    beforeEach(() => {
      useAppStore.setState({
        messages: [
          { role: "user", content: "hi" },
          { role: "assistant", content: "hello" },
        ],
      })
    })

    it("17. attaches to last assistant message", () => {
      const { attachDeployPredDistCompareToLastMessage } = useAppStore.getState()
      attachDeployPredDistCompareToLastMessage(regressionResult)
      const messages = useAppStore.getState().messages
      const last = messages[messages.length - 1]
      expect(last.role).toBe("assistant")
      expect(last.deploy_pred_dist_compare).toBeDefined()
      expect(last.deploy_pred_dist_compare?.verdict).toBe("current_higher")
    })

    it("18. does not attach to user message", () => {
      useAppStore.setState({ messages: [{ role: "user", content: "hi" }] })
      const { attachDeployPredDistCompareToLastMessage } = useAppStore.getState()
      attachDeployPredDistCompareToLastMessage(regressionResult)
      const messages = useAppStore.getState().messages
      expect(messages[0].deploy_pred_dist_compare).toBeUndefined()
    })

    it("19. does not crash on empty messages list", () => {
      useAppStore.setState({ messages: [] })
      const { attachDeployPredDistCompareToLastMessage } = useAppStore.getState()
      expect(() => attachDeployPredDistCompareToLastMessage(regressionResult)).not.toThrow()
    })
  })
})
