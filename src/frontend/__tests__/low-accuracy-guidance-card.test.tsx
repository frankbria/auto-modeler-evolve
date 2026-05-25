import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { LowAccuracyGuidanceCard } from "@/components/models/low-accuracy-guidance-card"
import { useAppStore } from "@/lib/store"
import type { LowAccuracyGuidanceResult } from "@/lib/types"

const baseTips = [
  {
    icon: "📥",
    title: "Collect more data",
    description: "Your dataset has 50 rows. More data helps.",
    action: "How do I add more data?",
  },
  {
    icon: "🔧",
    title: "Engineer better features",
    description: "Features matter more than algorithms.",
    action: "What new features would help?",
  },
  {
    icon: "🎯",
    title: "Review your target column",
    description: "A R² of 40.0% is very low — check your target.",
    action: "Analyze my target column",
  },
]

const baseResult: LowAccuracyGuidanceResult = {
  problem_type: "regression",
  best_score: 0.55,
  metric_name: "R²",
  score_label: "55.0%",
  is_very_low: false,
  tips: baseTips,
  summary:
    "Your best model (including ensembles) achieves R² 55.0%. Here are data-level improvements to try.",
  n_tips: 3,
}

describe("LowAccuracyGuidanceCard", () => {
  it("renders the card heading", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    expect(screen.getByText("Improve Your Model Accuracy")).toBeInTheDocument()
  })

  it("renders the score badge", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    const badge = screen.getByTestId("guidance-score-badge")
    expect(badge).toHaveTextContent("R² 55.0%")
  })

  it("does not show very-low badge when is_very_low is false", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    expect(
      screen.queryByTestId("guidance-very-low-badge")
    ).not.toBeInTheDocument()
  })

  it("shows very-low badge when is_very_low is true", () => {
    render(
      <LowAccuracyGuidanceCard result={{ ...baseResult, is_very_low: true }} />
    )
    expect(screen.getByTestId("guidance-very-low-badge")).toBeInTheDocument()
  })

  it("renders the summary text", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    expect(screen.getByText(baseResult.summary)).toBeInTheDocument()
  })

  it("renders all tip rows", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    for (let i = 0; i < baseTips.length; i++) {
      expect(screen.getByTestId(`guidance-tip-${i}`)).toBeInTheDocument()
    }
  })

  it("renders tip titles", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    expect(screen.getByText("Collect more data")).toBeInTheDocument()
    expect(screen.getByText("Engineer better features")).toBeInTheDocument()
    expect(screen.getByText("Review your target column")).toBeInTheDocument()
  })

  it("renders tip descriptions", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    expect(
      screen.getByText("Your dataset has 50 rows. More data helps.")
    ).toBeInTheDocument()
  })

  it("renders 'Try this →' buttons when onActionClick is provided", () => {
    render(
      <LowAccuracyGuidanceCard result={baseResult} onActionClick={() => {}} />
    )
    const buttons = screen.getAllByRole("button")
    expect(buttons.length).toBeGreaterThanOrEqual(baseTips.length)
  })

  it("does not render action buttons when onActionClick is absent", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    expect(screen.queryAllByRole("button")).toHaveLength(0)
  })

  it("calls onActionClick with the correct action string", () => {
    const handler = jest.fn()
    render(<LowAccuracyGuidanceCard result={baseResult} onActionClick={handler} />)
    fireEvent.click(screen.getByTestId("guidance-action-0"))
    expect(handler).toHaveBeenCalledWith("How do I add more data?")
  })

  it("renders the accessibility figcaption", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    expect(
      screen.getByText(/Low accuracy guidance card/i, { selector: ".sr-only" })
    ).toBeInTheDocument()
  })

  it("has aria-label on the figure", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    expect(screen.getByRole("figure")).toHaveAttribute(
      "aria-label",
      "Low accuracy guidance"
    )
  })

  it("renders classification metric name correctly", () => {
    render(
      <LowAccuracyGuidanceCard
        result={{
          ...baseResult,
          problem_type: "classification",
          metric_name: "accuracy",
          score_label: "68.0%",
        }}
      />
    )
    expect(screen.getByTestId("guidance-score-badge")).toHaveTextContent(
      "accuracy 68.0%"
    )
  })

  it("renders the test id on the card", () => {
    render(<LowAccuracyGuidanceCard result={baseResult} />)
    expect(
      screen.getByTestId("low-accuracy-guidance-card")
    ).toBeInTheDocument()
  })
})

describe("LowAccuracyGuidanceCard — Zustand store action", () => {
  it("attachLowAccuracyGuidanceToLastMessage adds guidance to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "assistant", content: "text", timestamp: new Date() },
      ],
    })

    useAppStore.getState().attachLowAccuracyGuidanceToLastMessage(baseResult)

    const messages = useAppStore.getState().messages
    expect(messages[messages.length - 1].low_accuracy_guidance).toEqual(
      baseResult
    )
  })
})
