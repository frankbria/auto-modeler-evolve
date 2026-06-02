import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { RetrainingReadinessCard } from "@/components/deploy/retraining-readiness-card"
import type { RetrainingReadinessResult } from "@/lib/types"

function makeData(overrides: Partial<RetrainingReadinessResult> = {}): RetrainingReadinessResult {
  return {
    deployment_id: "dep-1",
    algorithm: "random_forest",
    target_col: "revenue",
    problem_type: "regression",
    n_logs_checked: 50,
    n_feedback: 0,
    score: 25,
    verdict: "stable",
    verdict_label: "No action needed",
    signals: [
      {
        name: "model_age",
        label: "Model Age",
        value_str: "20 days",
        score_contribution: 0,
        is_firing: false,
        detail: "Trained 20 days ago (fresh)",
      },
    ],
    signals_available: 1,
    signals_firing: 0,
    top_reason: null,
    recommendations: ["Model is performing well. Keep collecting feedback to maintain visibility."],
    summary: "All 1 available monitoring signal(s) are within normal range.",
    ...overrides,
  }
}

describe("RetrainingReadinessCard", () => {
  it("renders the card root", () => {
    render(<RetrainingReadinessCard data={makeData()} />)
    expect(screen.getByTestId("retraining-readiness-card")).toBeInTheDocument()
  })

  it("shows verdict badge", () => {
    render(<RetrainingReadinessCard data={makeData()} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("No action needed")
  })

  it("shows score bar with correct aria attributes", () => {
    render(<RetrainingReadinessCard data={makeData({ score: 40 })} />)
    const bar = screen.getByTestId("score-bar")
    expect(bar).toHaveAttribute("aria-valuenow", "40")
    expect(bar).toHaveAttribute("aria-valuemax", "100")
  })

  it("shows summary text", () => {
    render(<RetrainingReadinessCard data={makeData()} />)
    expect(screen.getByTestId("readiness-summary")).toBeInTheDocument()
  })

  it("renders signal rows", () => {
    render(<RetrainingReadinessCard data={makeData()} />)
    expect(screen.getByTestId("signal-row-model_age")).toBeInTheDocument()
  })

  it("renders multiple signals", () => {
    const data = makeData({
      signals: [
        { name: "model_age", label: "Model Age", value_str: "100 days", score_contribution: 25, is_firing: true, detail: "Old" },
        { name: "anomaly_rate", label: "Anomalies", value_str: "20%", score_contribution: 30, is_firing: true, detail: "High" },
      ],
      signals_available: 2,
      signals_firing: 2,
    })
    render(<RetrainingReadinessCard data={data} />)
    expect(screen.getByTestId("signal-row-model_age")).toBeInTheDocument()
    expect(screen.getByTestId("signal-row-anomaly_rate")).toBeInTheDocument()
  })

  it("shows retrain CTA for retrain_now verdict", () => {
    const onClick = jest.fn()
    render(
      <RetrainingReadinessCard
        data={makeData({ verdict: "retrain_now", verdict_label: "Retrain urgently", score: 85 })}
        onActionClick={onClick}
      />
    )
    expect(screen.getByTestId("retrain-cta-button")).toBeInTheDocument()
  })

  it("does not show retrain CTA for stable verdict", () => {
    render(<RetrainingReadinessCard data={makeData({ verdict: "stable" })} />)
    expect(screen.queryByTestId("retrain-cta-button")).not.toBeInTheDocument()
  })

  it("fires onActionClick with correct text when CTA clicked", () => {
    const onClick = jest.fn()
    render(
      <RetrainingReadinessCard
        data={makeData({ verdict: "retrain_now", score: 90 })}
        onActionClick={onClick}
      />
    )
    fireEvent.click(screen.getByTestId("retrain-cta-button"))
    expect(onClick).toHaveBeenCalledWith("retrain my model")
  })

  it("shows recommendations list", () => {
    render(
      <RetrainingReadinessCard
        data={makeData({ recommendations: ["Retrain on fresh data.", "Review feedback."] })}
      />
    )
    expect(screen.getByText("Retrain on fresh data.")).toBeInTheDocument()
    expect(screen.getByText("Review feedback.")).toBeInTheDocument()
  })

  it("shows firing signal indicator with !", () => {
    const data = makeData({
      signals: [
        { name: "model_age", label: "Model Age", value_str: "200 days", score_contribution: 40, is_firing: true, detail: "Very old" },
      ],
    })
    render(<RetrainingReadinessCard data={data} />)
    const row = screen.getByTestId("signal-row-model_age")
    expect(row).toBeInTheDocument()
  })

  it("shows empty-state message when no signals", () => {
    render(
      <RetrainingReadinessCard
        data={makeData({ signals: [], signals_available: 0, signals_firing: 0 })}
      />
    )
    expect(
      screen.getByText(/No monitoring data available yet/i)
    ).toBeInTheDocument()
  })

  it("shows signals_firing count in badge", () => {
    render(
      <RetrainingReadinessCard
        data={makeData({ signals_firing: 2, signals_available: 3 })}
      />
    )
    expect(screen.getByText("2/3 signals firing")).toBeInTheDocument()
  })

  it("renders sr-only figcaption", () => {
    render(<RetrainingReadinessCard data={makeData()} />)
    const fig = screen.getByRole("figure", { hidden: true })
    expect(fig).toBeInTheDocument()
  })

  it("uses rose border for retrain_now", () => {
    render(
      <RetrainingReadinessCard
        data={makeData({ verdict: "retrain_now", score: 85 })}
      />
    )
    const card = screen.getByTestId("retraining-readiness-card")
    expect(card.className).toContain("rose")
  })

  it("uses emerald border for stable", () => {
    render(<RetrainingReadinessCard data={makeData({ verdict: "stable" })} />)
    const card = screen.getByTestId("retraining-readiness-card")
    expect(card.className).toContain("emerald")
  })
})

// ---------------------------------------------------------------------------
// Store action test
// ---------------------------------------------------------------------------

import { act } from "react"
import { renderHook } from "@testing-library/react"
import { useAppStore } from "@/lib/store"

describe("attachRetrainingReadinessToLastMessage store action", () => {
  it("attaches retraining_readiness to last assistant message", () => {
    const { result } = renderHook(() => useAppStore())

    act(() => {
      result.current.setMessages([
        { role: "user", content: "should I retrain?", timestamp: new Date().toISOString() },
        { role: "assistant", content: "Let me check.", timestamp: new Date().toISOString() },
      ])
    })

    const readinessData = makeData({ score: 70, verdict: "retrain_soon" })

    act(() => {
      result.current.attachRetrainingReadinessToLastMessage(readinessData)
    })

    const msgs = result.current.messages
    expect(msgs[msgs.length - 1].retraining_readiness).toBeDefined()
    expect(msgs[msgs.length - 1].retraining_readiness?.score).toBe(70)
  })

  it("does not attach to user messages", () => {
    const { result } = renderHook(() => useAppStore())

    act(() => {
      result.current.setMessages([
        { role: "user", content: "should I retrain?", timestamp: new Date().toISOString() },
      ])
    })

    act(() => {
      result.current.attachRetrainingReadinessToLastMessage(makeData())
    })

    expect(result.current.messages[0].retraining_readiness).toBeUndefined()
  })
})
