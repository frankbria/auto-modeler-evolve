/**
 * Tests for ClassImbalanceChatCard and Zustand store action.
 *
 * Covers:
 *  1.  Imbalanced: renders region with aria-label "Class imbalance check"
 *  2.  Imbalanced: renders ⚠️ icon (aria-hidden)
 *  3.  Imbalanced: renders "Class Imbalance Detected" heading
 *  4.  Imbalanced: renders minority % badge
 *  5.  Imbalanced: renders class distribution bars
 *  6.  Imbalanced: renders explanation text
 *  7.  Imbalanced: renders recommended strategy label
 *  8.  Balanced: renders "Balanced Classes" heading
 *  9.  Balanced: renders "No action needed" badge
 * 10.  Regression: renders "Regression — N/A" badge
 * 11.  Imbalanced: onSwitchTab button navigates to models tab
 * 12.  Store: attachClassImbalanceCheckToLastMessage attaches to last assistant message
 * 13.  Store: does not attach to user message
 * 14.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { ClassImbalanceChatCard } from "@/components/models/class-imbalance-chat-card"
import type { ClassImbalanceResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const imbalancedData: ClassImbalanceResult = {
  project_id: "proj-1",
  problem_type: "classification",
  target_column: "churn",
  is_imbalanced: true,
  class_distribution: [
    { class: "0", count: 90, ratio: 0.9 },
    { class: "1", count: 10, ratio: 0.1 },
  ],
  minority_class: "1",
  minority_ratio: 0.1,
  recommended_strategy: "class_weight",
  explanation: "Your data has a class imbalance: only 10% of rows belong to '1'.",
}

const balancedData: ClassImbalanceResult = {
  project_id: "proj-1",
  problem_type: "classification",
  target_column: "churn",
  is_imbalanced: false,
  class_distribution: [
    { class: "0", count: 50, ratio: 0.5 },
    { class: "1", count: 50, ratio: 0.5 },
  ],
  minority_class: null,
  minority_ratio: 0.5,
  recommended_strategy: "none",
  explanation: "Your target classes are roughly balanced — no special handling needed.",
}

const regressionData: ClassImbalanceResult = {
  project_id: "proj-1",
  problem_type: "regression",
  target_column: "revenue",
  is_imbalanced: false,
  class_distribution: [],
  minority_class: null,
  minority_ratio: null,
  recommended_strategy: "none",
  explanation: "Class imbalance only applies to classification problems.",
}

// ---------------------------------------------------------------------------
// Rendering tests — imbalanced case
// ---------------------------------------------------------------------------

describe("ClassImbalanceChatCard — imbalanced", () => {
  it("renders region with aria-label", () => {
    render(<ClassImbalanceChatCard data={imbalancedData} />)
    expect(screen.getByRole("region", { name: "Class imbalance check" })).toBeInTheDocument()
  })

  it("renders warning icon aria-hidden", () => {
    render(<ClassImbalanceChatCard data={imbalancedData} />)
    const icon = screen.getByText("⚠️")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("renders Class Imbalance Detected heading", () => {
    render(<ClassImbalanceChatCard data={imbalancedData} />)
    expect(screen.getByText("Class Imbalance Detected")).toBeInTheDocument()
  })

  it("renders minority percent badge", () => {
    render(<ClassImbalanceChatCard data={imbalancedData} />)
    expect(screen.getByText("10% minority")).toBeInTheDocument()
  })

  it("renders class distribution rows", () => {
    render(<ClassImbalanceChatCard data={imbalancedData} />)
    expect(screen.getByRole("list", { name: "Class distribution" })).toBeInTheDocument()
    const items = screen.getAllByRole("listitem")
    expect(items.length).toBeGreaterThanOrEqual(2)
  })

  it("renders explanation text", () => {
    render(<ClassImbalanceChatCard data={imbalancedData} />)
    expect(screen.getByText(/only 10% of rows belong to '1'/)).toBeInTheDocument()
  })

  it("renders recommended strategy label", () => {
    render(<ClassImbalanceChatCard data={imbalancedData} />)
    expect(screen.getByText("Class Weighting")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Rendering tests — balanced case
// ---------------------------------------------------------------------------

describe("ClassImbalanceChatCard — balanced", () => {
  it("renders Balanced Classes heading", () => {
    render(<ClassImbalanceChatCard data={balancedData} />)
    expect(screen.getByText("Balanced Classes")).toBeInTheDocument()
  })

  it("renders No action needed badge", () => {
    render(<ClassImbalanceChatCard data={balancedData} />)
    expect(screen.getByText("No action needed")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Rendering tests — regression case
// ---------------------------------------------------------------------------

describe("ClassImbalanceChatCard — regression", () => {
  it("renders Regression — N/A badge", () => {
    render(<ClassImbalanceChatCard data={regressionData} />)
    expect(screen.getByText("Regression — N/A")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Interaction test
// ---------------------------------------------------------------------------

describe("ClassImbalanceChatCard — interaction", () => {
  it("calls onSwitchTab with 'models' when CTA button clicked", () => {
    const onSwitch = jest.fn()
    render(<ClassImbalanceChatCard data={imbalancedData} onSwitchTab={onSwitch} />)
    const btn = screen.getByText(/Go to Models tab/)
    fireEvent.click(btn)
    expect(onSwitch).toHaveBeenCalledWith("models")
  })
})

// ---------------------------------------------------------------------------
// Zustand store tests
// ---------------------------------------------------------------------------

describe("Zustand store — attachClassImbalanceCheckToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches class_imbalance_check to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "is my data imbalanced?", timestamp: "t1" },
        { role: "assistant", content: "Let me check.", timestamp: "t2" },
      ],
    })
    useAppStore.getState().attachClassImbalanceCheckToLastMessage(imbalancedData)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].class_imbalance_check).toEqual(imbalancedData)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "is my data imbalanced?", timestamp: "t1" },
      ],
    })
    useAppStore.getState().attachClassImbalanceCheckToLastMessage(imbalancedData)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].class_imbalance_check).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachClassImbalanceCheckToLastMessage(imbalancedData)
    ).not.toThrow()
  })
})
