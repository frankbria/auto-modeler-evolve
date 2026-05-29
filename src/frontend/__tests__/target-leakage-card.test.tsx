import React from "react"
import { render, screen } from "@testing-library/react"
import { TargetLeakageCard } from "@/components/models/target-leakage-card"
import type { TargetLeakageResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeResult(overrides: Partial<TargetLeakageResult> = {}): TargetLeakageResult {
  return {
    dataset_id: "ds-1",
    leaky_features: [],
    n_checked: 5,
    verdict: "none",
    verdict_label: "No Leakage Detected",
    high_threshold: 0.9,
    moderate_threshold: 0.75,
    target_col: "revenue",
    summary: "None of the 5 features show suspicious correlation with 'revenue'.",
    ...overrides,
  }
}

const highRiskFeature = {
  feature: "final_payment",
  correlation: 0.97,
  correlation_abs: 0.97,
  risk_level: "high" as const,
  risk_label: "High Risk",
  reason: "'final_payment' has a 97% Pearson correlation with 'revenue'.",
}

const moderateRiskFeature = {
  feature: "discount_rate",
  correlation: 0.81,
  correlation_abs: 0.81,
  risk_level: "moderate" as const,
  risk_label: "Moderate Risk",
  reason: "'discount_rate' has a 81% Pearson correlation with 'revenue'.",
}

describe("TargetLeakageCard", () => {
  it("renders the card container", () => {
    render(<TargetLeakageCard result={makeResult()} />)
    expect(screen.getByTestId("target-leakage-card")).toBeInTheDocument()
  })

  it("shows the verdict badge for none verdict", () => {
    render(<TargetLeakageCard result={makeResult()} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("No Leakage Detected")
  })

  it("shows features-checked badge", () => {
    render(<TargetLeakageCard result={makeResult({ n_checked: 8 })} />)
    expect(screen.getByTestId("features-checked-badge")).toHaveTextContent("8 features checked")
  })

  it("shows no-leakage message when verdict is none", () => {
    render(<TargetLeakageCard result={makeResult()} />)
    expect(screen.getByTestId("no-leakage-message")).toBeInTheDocument()
  })

  it("does not show no-leakage message when there are leaky features", () => {
    render(
      <TargetLeakageCard
        result={makeResult({
          verdict: "severe",
          verdict_label: "Likely Leakage",
          leaky_features: [highRiskFeature],
        })}
      />
    )
    expect(screen.queryByTestId("no-leakage-message")).not.toBeInTheDocument()
  })

  it("shows leaky feature rows when features exist", () => {
    render(
      <TargetLeakageCard
        result={makeResult({
          verdict: "severe",
          verdict_label: "Likely Leakage",
          leaky_features: [highRiskFeature],
        })}
      />
    )
    const rows = screen.getAllByTestId("leaky-feature-row")
    expect(rows).toHaveLength(1)
  })

  it("displays the feature name in the row", () => {
    render(
      <TargetLeakageCard
        result={makeResult({
          verdict: "severe",
          leaky_features: [highRiskFeature],
        })}
      />
    )
    expect(screen.getByTestId("leaky-feature-name")).toHaveTextContent("final_payment")
  })

  it("displays risk badge", () => {
    render(
      <TargetLeakageCard
        result={makeResult({
          verdict: "severe",
          leaky_features: [highRiskFeature],
        })}
      />
    )
    expect(screen.getByTestId("risk-badge")).toHaveTextContent("High Risk")
  })

  it("shows moderate-risk badge for moderate risk feature", () => {
    render(
      <TargetLeakageCard
        result={makeResult({
          verdict: "warning",
          verdict_label: "Possible Leakage",
          leaky_features: [moderateRiskFeature],
        })}
      />
    )
    expect(screen.getByTestId("risk-badge")).toHaveTextContent("Moderate Risk")
  })

  it("shows leaky-count badge when features exist", () => {
    render(
      <TargetLeakageCard
        result={makeResult({
          verdict: "severe",
          leaky_features: [highRiskFeature, moderateRiskFeature],
        })}
      />
    )
    expect(screen.getByTestId("leaky-count-badge")).toHaveTextContent("2 suspicious")
  })

  it("does not show leaky-count badge when no leaky features", () => {
    render(<TargetLeakageCard result={makeResult()} />)
    expect(screen.queryByTestId("leaky-count-badge")).not.toBeInTheDocument()
  })

  it("shows severe alert when verdict is severe", () => {
    render(
      <TargetLeakageCard
        result={makeResult({
          verdict: "severe",
          leaky_features: [highRiskFeature],
        })}
      />
    )
    expect(screen.getByTestId("severe-alert")).toBeInTheDocument()
  })

  it("does not show severe alert for warning verdict", () => {
    render(
      <TargetLeakageCard
        result={makeResult({
          verdict: "warning",
          leaky_features: [moderateRiskFeature],
        })}
      />
    )
    expect(screen.queryByTestId("severe-alert")).not.toBeInTheDocument()
  })

  it("shows recommendation box when leaky features present", () => {
    render(
      <TargetLeakageCard
        result={makeResult({
          verdict: "severe",
          leaky_features: [highRiskFeature],
        })}
      />
    )
    expect(screen.getByTestId("leakage-recommendation")).toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<TargetLeakageCard result={makeResult({ summary: "All clear!" })} />)
    expect(screen.getByTestId("leakage-summary")).toHaveTextContent("All clear!")
  })

  it("shows target column label", () => {
    render(<TargetLeakageCard result={makeResult({ target_col: "churn" })} />)
    expect(screen.getByText("churn")).toBeInTheDocument()
  })

  it("returns null when result is falsy", () => {
    const { container } = render(<TargetLeakageCard result={null as unknown as TargetLeakageResult} />)
    expect(container).toBeEmptyDOMElement()
  })

  it("renders multiple leaky features", () => {
    render(
      <TargetLeakageCard
        result={makeResult({
          verdict: "severe",
          leaky_features: [highRiskFeature, moderateRiskFeature],
        })}
      />
    )
    expect(screen.getAllByTestId("leaky-feature-row")).toHaveLength(2)
  })
})

// Store action test
describe("attachTargetLeakageToLastMessage store action", () => {
  it("attaches target_leakage to last assistant message", () => {
    const { attachTargetLeakageToLastMessage, setMessages } = useAppStore.getState()
    setMessages([
      { role: "user", content: "check leakage" },
      { role: "assistant", content: "Let me check..." },
    ])
    const result = makeResult({ verdict: "none" })
    attachTargetLeakageToLastMessage(result)
    const msgs = useAppStore.getState().messages
    expect((msgs[msgs.length - 1] as Record<string, unknown>).target_leakage).toEqual(result)
  })

  it("does not attach to user message", () => {
    const { attachTargetLeakageToLastMessage, setMessages } = useAppStore.getState()
    setMessages([{ role: "user", content: "check" }])
    attachTargetLeakageToLastMessage(makeResult())
    const msgs = useAppStore.getState().messages
    expect((msgs[0] as Record<string, unknown>).target_leakage).toBeUndefined()
  })

  it("handles empty messages gracefully", () => {
    const { attachTargetLeakageToLastMessage, setMessages } = useAppStore.getState()
    setMessages([])
    expect(() => attachTargetLeakageToLastMessage(makeResult())).not.toThrow()
  })
})
