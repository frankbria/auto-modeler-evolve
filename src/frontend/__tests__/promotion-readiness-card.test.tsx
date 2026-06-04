/**
 * Tests for PromotionReadinessCard (model promotion readiness checklist card).
 *
 * Covers:
 *  1.  Renders region with data-testid "promotion-readiness-card"
 *  2.  Shows verdict badge "Ready to Deploy" when all gates pass
 *  3.  Shows verdict badge "Deploy with Caution" when warnings but no fails
 *  4.  Shows verdict badge "Not Ready for Production" when any gate fails
 *  5.  Shows emerald border for "ready" verdict
 *  6.  Shows amber border for "ready_with_warnings" verdict
 *  7.  Shows rose border for "not_ready" verdict
 *  8.  Shows passed/warn/fail count summary row
 *  9.  Renders gate list with data-testid "gate-list"
 * 10.  Each gate row has data-testid "gate-row-{gate}"
 * 11.  Renders blocking issues callout when fail_count > 0
 * 12.  Does not render blocking issues callout when fail_count = 0
 * 13.  Shows summary paragraph
 * 14.  Shows "Deploy my model" action button when ready and onActionClick provided
 * 15.  Action button calls onActionClick with "deploy my model"
 * 16.  Does not show action button when not_ready
 * 17.  Shows algorithm badge
 * 18.  Shows problem_type badge
 * 19.  Shows primary metric badge
 * 20.  Store: attachPromotionReadinessToLastMessage attaches to last assistant message
 * 21.  Store: does not attach to user message
 * 22.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { PromotionReadinessCard } from "@/components/models/promotion-readiness-card"
import type { PromotionReadinessResult, PromotionReadinessGate } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const passGate = (gate: string, label: string): PromotionReadinessGate => ({
  gate,
  label,
  status: "pass",
  detail: `${label} passed.`,
  recommendation: "",
})

const warnGate = (gate: string, label: string): PromotionReadinessGate => ({
  gate,
  label,
  status: "warn",
  detail: `${label} borderline.`,
  recommendation: "Review this gate.",
})

const failGate = (gate: string, label: string): PromotionReadinessGate => ({
  gate,
  label,
  status: "fail",
  detail: `${label} failed.`,
  recommendation: "Fix this gate before deploying.",
})

const readyResult: PromotionReadinessResult = {
  run_id: "run-1",
  project_id: "proj-1",
  overall_verdict: "ready",
  verdict_label: "Ready to Deploy",
  verdict_color: "emerald",
  passed_count: 5,
  warn_count: 0,
  fail_count: 0,
  total_count: 5,
  gates: [
    passGate("model_quality", "Model Quality"),
    passGate("cv_stability", "Cross-Validation Stability"),
    passGate("overfitting", "Overfitting Risk"),
    passGate("data_volume", "Training Data Volume"),
    passGate("sample_feature_ratio", "Sample-to-Feature Ratio"),
  ],
  blocking_issues: [],
  warnings: [],
  algorithm: "random_forest_regressor",
  problem_type: "regression",
  primary_metric: 0.85,
  primary_metric_name: "R²",
  n_rows: 500,
  n_features: 10,
  summary: "All 5 readiness gates passed. Your model is ready for production.",
}

const warnResult: PromotionReadinessResult = {
  ...readyResult,
  overall_verdict: "ready_with_warnings",
  verdict_label: "Deploy with Caution",
  verdict_color: "amber",
  passed_count: 4,
  warn_count: 1,
  fail_count: 0,
  gates: [
    passGate("model_quality", "Model Quality"),
    warnGate("cv_stability", "Cross-Validation Stability"),
    passGate("overfitting", "Overfitting Risk"),
    passGate("data_volume", "Training Data Volume"),
    passGate("sample_feature_ratio", "Sample-to-Feature Ratio"),
  ],
  warnings: ["Cross-Validation Stability"],
  summary: "4 of 5 gates passed with 1 warning. Review before deploying.",
}

const failResult: PromotionReadinessResult = {
  ...readyResult,
  overall_verdict: "not_ready",
  verdict_label: "Not Ready for Production",
  verdict_color: "rose",
  passed_count: 3,
  warn_count: 1,
  fail_count: 1,
  gates: [
    passGate("model_quality", "Model Quality"),
    failGate("cv_stability", "Cross-Validation Stability"),
    warnGate("overfitting", "Overfitting Risk"),
    passGate("data_volume", "Training Data Volume"),
    passGate("sample_feature_ratio", "Sample-to-Feature Ratio"),
  ],
  blocking_issues: ["Cross-Validation Stability"],
  warnings: ["Overfitting Risk"],
  summary: "1 blocking issue found. Resolve before promoting to production.",
}

// ─── Card rendering tests ──────────────────────────────────────────────────────

describe("PromotionReadinessCard", () => {
  test("1. renders region with data-testid promotion-readiness-card", () => {
    render(<PromotionReadinessCard result={readyResult} />)
    expect(screen.getByTestId("promotion-readiness-card")).toBeInTheDocument()
  })

  test("2. shows 'Ready to Deploy' verdict badge", () => {
    render(<PromotionReadinessCard result={readyResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Ready to Deploy")
  })

  test("3. shows 'Deploy with Caution' verdict badge for warnings", () => {
    render(<PromotionReadinessCard result={warnResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Deploy with Caution")
  })

  test("4. shows 'Not Ready for Production' verdict badge for failures", () => {
    render(<PromotionReadinessCard result={failResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("Not Ready for Production")
  })

  test("5. applies emerald border class for ready verdict", () => {
    const { container } = render(<PromotionReadinessCard result={readyResult} />)
    const card = container.querySelector("[data-testid='promotion-readiness-card']")
    expect(card?.className).toContain("emerald")
  })

  test("6. applies amber border class for ready_with_warnings verdict", () => {
    const { container } = render(<PromotionReadinessCard result={warnResult} />)
    const card = container.querySelector("[data-testid='promotion-readiness-card']")
    expect(card?.className).toContain("amber")
  })

  test("7. applies rose border class for not_ready verdict", () => {
    const { container } = render(<PromotionReadinessCard result={failResult} />)
    const card = container.querySelector("[data-testid='promotion-readiness-card']")
    expect(card?.className).toContain("rose")
  })

  test("8. shows passed/warn/fail count summary", () => {
    render(<PromotionReadinessCard result={warnResult} />)
    expect(screen.getByText("4")).toBeInTheDocument()
    expect(screen.getByText("1")).toBeInTheDocument()
    expect(screen.getByText("of 5 gates")).toBeInTheDocument()
  })

  test("9. renders gate list", () => {
    render(<PromotionReadinessCard result={readyResult} />)
    expect(screen.getByTestId("gate-list")).toBeInTheDocument()
  })

  test("10. each gate row has correct data-testid", () => {
    render(<PromotionReadinessCard result={readyResult} />)
    expect(screen.getByTestId("gate-row-model_quality")).toBeInTheDocument()
    expect(screen.getByTestId("gate-row-cv_stability")).toBeInTheDocument()
    expect(screen.getByTestId("gate-row-overfitting")).toBeInTheDocument()
  })

  test("11. renders blocking issues callout when fail_count > 0", () => {
    render(<PromotionReadinessCard result={failResult} />)
    const callout = screen.getByTestId("blocking-issues")
    expect(callout).toBeInTheDocument()
    expect(screen.getByText("Blocking issues (1)")).toBeInTheDocument()
    // The callout itself contains the blocking issue label in a list item
    expect(callout.querySelector("li")?.textContent).toContain("Cross-Validation Stability")
  })

  test("12. does not render blocking issues when fail_count = 0", () => {
    render(<PromotionReadinessCard result={readyResult} />)
    expect(screen.queryByTestId("blocking-issues")).not.toBeInTheDocument()
  })

  test("13. shows summary paragraph", () => {
    render(<PromotionReadinessCard result={readyResult} />)
    expect(screen.getByText(/All 5 readiness gates passed/)).toBeInTheDocument()
  })

  test("14. shows deploy action button when ready and onActionClick provided", () => {
    const fn = jest.fn()
    render(<PromotionReadinessCard result={readyResult} onActionClick={fn} />)
    expect(screen.getByTestId("deploy-action-button")).toBeInTheDocument()
  })

  test("15. action button calls onActionClick with 'deploy my model'", () => {
    const fn = jest.fn()
    render(<PromotionReadinessCard result={readyResult} onActionClick={fn} />)
    fireEvent.click(screen.getByTestId("deploy-action-button"))
    expect(fn).toHaveBeenCalledWith("deploy my model")
  })

  test("16. does not show deploy button for not_ready without onActionClick", () => {
    render(<PromotionReadinessCard result={failResult} />)
    expect(screen.queryByTestId("deploy-action-button")).not.toBeInTheDocument()
  })

  test("17. shows algorithm badge", () => {
    render(<PromotionReadinessCard result={readyResult} />)
    expect(screen.getByText("random_forest_regressor")).toBeInTheDocument()
  })

  test("18. shows problem_type badge", () => {
    render(<PromotionReadinessCard result={readyResult} />)
    expect(screen.getByText("regression")).toBeInTheDocument()
  })

  test("19. shows primary metric badge", () => {
    render(<PromotionReadinessCard result={readyResult} />)
    expect(screen.getByText(/R².*0\.850/)).toBeInTheDocument()
  })

  test("shows warn_with_warnings action button when ready_with_warnings", () => {
    const fn = jest.fn()
    render(<PromotionReadinessCard result={warnResult} onActionClick={fn} />)
    expect(screen.getByTestId("deploy-action-button")).toBeInTheDocument()
  })

  test("gate row shows pass detail text", () => {
    render(<PromotionReadinessCard result={readyResult} />)
    expect(screen.getByText("Model Quality passed.")).toBeInTheDocument()
  })

  test("gate row shows warn recommendation text", () => {
    render(<PromotionReadinessCard result={warnResult} />)
    expect(screen.getByText(/Review this gate/)).toBeInTheDocument()
  })

  test("gate row shows fail recommendation text", () => {
    render(<PromotionReadinessCard result={failResult} />)
    expect(screen.getByText(/Fix this gate before deploying/)).toBeInTheDocument()
  })
})

// ─── Store tests ──────────────────────────────────────────────────────────────

describe("attachPromotionReadinessToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  test("20. attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "run readiness check", timestamp: "t1" },
        { role: "assistant", content: "Here's the checklist:", timestamp: "t2" },
      ],
    })
    useAppStore.getState().attachPromotionReadinessToLastMessage(readyResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].promotion_readiness).toEqual(readyResult)
  })

  test("21. does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "hi", timestamp: "t1" }],
    })
    useAppStore.getState().attachPromotionReadinessToLastMessage(readyResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].promotion_readiness).toBeUndefined()
  })

  test("22. does not crash when messages is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachPromotionReadinessToLastMessage(readyResult)
    ).not.toThrow()
  })
})
