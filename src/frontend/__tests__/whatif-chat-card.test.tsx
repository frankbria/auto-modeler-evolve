/**
 * Tests for WhatIfChatCard component, store action, and types.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { WhatIfChatCard } from "@/components/deploy/whatif-chat-card"
import type { WhatIfChatResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Fixtures -----------------------------------------------------------------

const regressionResult: WhatIfChatResult = {
  deployment_id: "dep-1",
  changed_feature: "units",
  original_feature_value: 10,
  new_feature_value: 20,
  original_prediction: 1200.5,
  modified_prediction: 2100.0,
  delta: 899.5,
  percent_change: 74.9,
  direction: "increase",
  summary:
    "Changing units from 10 to 20 would increase the prediction from 1200.5 to 2100.0 (+74.9%).",
  problem_type: "regression",
  target_column: "revenue",
}

const classificationResult: WhatIfChatResult = {
  deployment_id: "dep-2",
  changed_feature: "region",
  original_feature_value: "East",
  new_feature_value: "West",
  original_prediction: "cat",
  modified_prediction: "dog",
  delta: null,
  percent_change: null,
  direction: null,
  summary:
    "Changing region from East to West changes the prediction from 'cat' to 'dog'.",
  problem_type: "classification",
  target_column: "label",
  original_probabilities: { cat: 0.75, dog: 0.25 },
  modified_probabilities: { cat: 0.3, dog: 0.7 },
}

const noChangeResult: WhatIfChatResult = {
  deployment_id: "dep-3",
  changed_feature: "units",
  original_feature_value: 10,
  new_feature_value: 10,
  original_prediction: 1200.5,
  modified_prediction: 1200.5,
  delta: 0,
  percent_change: 0,
  direction: "no change",
  summary: "Changing units from 10 to 10 has no effect on the prediction (1200.5).",
  problem_type: "regression",
  target_column: "revenue",
}

const decreaseResult: WhatIfChatResult = {
  deployment_id: "dep-4",
  changed_feature: "units",
  original_feature_value: 20,
  new_feature_value: 5,
  original_prediction: 2100.0,
  modified_prediction: 500.0,
  delta: -1600.0,
  percent_change: -76.2,
  direction: "decrease",
  summary:
    "Changing units from 20 to 5 would decrease the prediction from 2100.0 to 500.0 (-76.2%).",
  problem_type: "regression",
  target_column: "revenue",
}

// --- Component tests ----------------------------------------------------------

describe("WhatIfChatCard", () => {
  it("renders the what-if analysis header", () => {
    render(<WhatIfChatCard result={regressionResult} />)
    expect(screen.getByText(/What-If Analysis/i)).toBeInTheDocument()
  })

  it("renders the changed feature name in the hypothesis row", () => {
    render(<WhatIfChatCard result={regressionResult} />)
    // Feature name appears in the Hypothetical Change section (capitalize class)
    const featureLabels = screen.getAllByText(/units/i)
    expect(featureLabels.length).toBeGreaterThan(0)
  })

  it("renders original and new feature values", () => {
    render(<WhatIfChatCard result={regressionResult} />)
    expect(screen.getByText("10")).toBeInTheDocument()
    expect(screen.getByText("20")).toBeInTheDocument()
  })

  it("renders original and modified predictions", () => {
    render(<WhatIfChatCard result={regressionResult} />)
    expect(screen.getByText("1.2k")).toBeInTheDocument()
    expect(screen.getByText("2.1k")).toBeInTheDocument()
  })

  it("renders increase delta badge with upward arrow", () => {
    render(<WhatIfChatCard result={regressionResult} />)
    const badge = screen.getByText(/↑/)
    expect(badge).toBeInTheDocument()
    expect(badge.textContent).toContain("+74.9%")
  })

  it("renders decrease delta badge with downward arrow", () => {
    render(<WhatIfChatCard result={decreaseResult} />)
    const badge = screen.getByText(/↓/)
    expect(badge).toBeInTheDocument()
    expect(badge.textContent).toContain("-76.2%")
  })

  it("renders no delta badge when direction is null", () => {
    render(<WhatIfChatCard result={classificationResult} />)
    expect(screen.queryByText(/↑/)).not.toBeInTheDocument()
    expect(screen.queryByText(/↓/)).not.toBeInTheDocument()
  })

  it("renders the summary footer", () => {
    render(<WhatIfChatCard result={regressionResult} />)
    expect(
      screen.getByText(/Changing units from 10 to 20 would increase/i)
    ).toBeInTheDocument()
  })

  it("renders Regression badge for regression model", () => {
    render(<WhatIfChatCard result={regressionResult} />)
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("renders Classification badge for classification model", () => {
    render(<WhatIfChatCard result={classificationResult} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("renders original and modified probabilities for classification", () => {
    render(<WhatIfChatCard result={classificationResult} />)
    expect(screen.getByText(/Original probabilities/i)).toBeInTheDocument()
    expect(screen.getByText(/Modified probabilities/i)).toBeInTheDocument()
    expect(screen.getByText(/cat: 75\.0%/i)).toBeInTheDocument()
    expect(screen.getByText(/dog: 70\.0%/i)).toBeInTheDocument()
  })

  it("does not render probabilities for regression", () => {
    render(<WhatIfChatCard result={regressionResult} />)
    expect(screen.queryByText(/probabilities/i)).not.toBeInTheDocument()
  })

  it("renders → arrow in feature change row", () => {
    render(<WhatIfChatCard result={regressionResult} />)
    expect(screen.getByText("→")).toBeInTheDocument()
  })

  it("has data-testid for the card container", () => {
    render(<WhatIfChatCard result={regressionResult} />)
    expect(screen.getByTestId("whatif-chat-card")).toBeInTheDocument()
  })
})

// --- Store action tests -------------------------------------------------------

describe("attachWhatIfChatToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        {
          role: "assistant",
          content: "Here is the what-if analysis.",
          timestamp: new Date().toISOString(),
        },
      ],
    })
  })

  it("attaches whatif_chat_result to the last assistant message", () => {
    useAppStore.getState().attachWhatIfChatToLastMessage(regressionResult)
    const messages = useAppStore.getState().messages
    expect(messages[0].whatif_chat_result).toEqual(regressionResult)
  })

  it("does not attach to user messages", () => {
    useAppStore.setState({
      messages: [
        {
          role: "user",
          content: "what if units was 20?",
          timestamp: new Date().toISOString(),
        },
      ],
    })
    useAppStore.getState().attachWhatIfChatToLastMessage(regressionResult)
    const messages = useAppStore.getState().messages
    expect(messages[0].whatif_chat_result).toBeUndefined()
  })

  it("does not attach when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    // Should not throw
    expect(() =>
      useAppStore.getState().attachWhatIfChatToLastMessage(regressionResult)
    ).not.toThrow()
  })
})
