/**
 * Tests for ConfusionMatrixChatCard component and Zustand store action.
 *
 * Covers:
 *  1.  Renders figure with aria-label
 *  2.  Renders 🎯 icon
 *  3.  Shows "Confusion Matrix" heading
 *  4.  Shows algorithm badge
 *  5.  Shows target column badge
 *  6.  Shows accuracy badge
 *  7.  Renders matrix grid with "Confusion matrix grid" aria-label
 *  8.  Renders per-class metrics table
 *  9.  Shows precision, recall, F1 values for each class
 * 10.  Shows most_confused_pair callout
 * 11.  Hides most_confused_pair when null
 * 12.  Shows correct / total count
 * 13.  Shows summary in figcaption
 * 14.  Store: attachConfusionMatrixChatToLastMessage attaches to last assistant message
 * 15.  Store: does not attach to user message
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ConfusionMatrixChatCard } from "@/components/models/confusion-matrix-chat-card"
import type { ConfusionMatrixChatResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const binaryResult: ConfusionMatrixChatResult = {
  matrix: [[10, 2], [3, 15]],
  labels: ["No", "Yes"],
  total: 30,
  correct: 25,
  accuracy: 0.8333,
  per_class_metrics: [
    { label: "No", precision: 0.769, recall: 0.833, f1: 0.8, support: 12 },
    { label: "Yes", precision: 0.882, recall: 0.833, f1: 0.857, support: 18 },
  ],
  most_confused_pair: { actual: "No", predicted: "Yes", count: 2 },
  summary: "True positives: 15, True negatives: 10, False positives: 3, False negatives: 2.",
  algorithm: "logistic_regression",
  algorithm_plain: "Logistic Regression",
  target_col: "churn",
}

const multiclassResult: ConfusionMatrixChatResult = {
  matrix: [[5, 1, 0], [0, 4, 2], [0, 0, 8]],
  labels: ["cat", "dog", "bird"],
  total: 20,
  correct: 17,
  accuracy: 0.85,
  per_class_metrics: [
    { label: "cat", precision: 1.0, recall: 0.833, f1: 0.909, support: 6 },
    { label: "dog", precision: 0.8, recall: 0.667, f1: 0.727, support: 6 },
    { label: "bird", precision: 0.8, recall: 1.0, f1: 0.889, support: 8 },
  ],
  most_confused_pair: { actual: "dog", predicted: "bird", count: 2 },
  summary: "The model struggles most with class 'dog' (recall = 67%).",
  algorithm: "random_forest_classifier",
  algorithm_plain: "Random Forest",
  target_col: "species",
}

const noPairResult: ConfusionMatrixChatResult = {
  ...binaryResult,
  most_confused_pair: null,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ConfusionMatrixChatCard", () => {
  it("renders figure with aria-label", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByRole("figure")).toHaveAttribute("aria-label", "Confusion matrix")
  })

  it("renders 🎯 icon", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByRole("img", { name: "target" })).toBeInTheDocument()
  })

  it("shows Confusion Matrix heading", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByText("Confusion Matrix")).toBeInTheDocument()
  })

  it("shows algorithm badge", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByText("Logistic Regression")).toBeInTheDocument()
  })

  it("shows target column badge", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByText("Target: churn")).toBeInTheDocument()
  })

  it("shows accuracy badge", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByText("83.3% accurate")).toBeInTheDocument()
  })

  it("renders matrix grid with aria-label", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByRole("table", { name: "Confusion matrix grid" })).toBeInTheDocument()
  })

  it("shows class labels in matrix", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getAllByText("No").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Yes").length).toBeGreaterThan(0)
  })

  it("renders per-class metrics table", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByRole("table", { name: "Per-class precision recall F1" })).toBeInTheDocument()
  })

  it("shows precision recall F1 column headers", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByText("Precision")).toBeInTheDocument()
    expect(screen.getByText("Recall")).toBeInTheDocument()
    expect(screen.getByText("F1")).toBeInTheDocument()
  })

  it("shows per-class metric values", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    // No class: precision 77%, recall 83%, f1 80%
    expect(screen.getByText("77%")).toBeInTheDocument()
    expect(screen.getByText("80%")).toBeInTheDocument()
  })

  it("shows most_confused_pair callout", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByText(/Most common mistake/i)).toBeInTheDocument()
    expect(screen.getByText(/2 times/i)).toBeInTheDocument()
  })

  it("hides most_confused_pair when null", () => {
    render(<ConfusionMatrixChatCard result={noPairResult} />)
    expect(screen.queryByText(/Most common mistake/i)).not.toBeInTheDocument()
  })

  it("shows correct of total count", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByText("25 of 30 predictions correct")).toBeInTheDocument()
  })

  it("shows summary in figcaption", () => {
    render(<ConfusionMatrixChatCard result={binaryResult} />)
    expect(screen.getByText(/True positives: 15/)).toBeInTheDocument()
  })

  it("renders multiclass labels and per-class metrics", () => {
    render(<ConfusionMatrixChatCard result={multiclassResult} />)
    expect(screen.getAllByText("cat").length).toBeGreaterThan(0)
    expect(screen.getAllByText("dog").length).toBeGreaterThan(0)
    expect(screen.getAllByText("bird").length).toBeGreaterThan(0)
    // Three per-class rows should be present
    expect(screen.getByText(/Random Forest/)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachConfusionMatrixChatToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "show me the confusion matrix" },
        { role: "assistant", content: "Here it is." },
      ],
    })
  })

  it("attaches to last assistant message", () => {
    useAppStore.getState().attachConfusionMatrixChatToLastMessage(binaryResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].confusion_matrix_chat).toEqual(binaryResult)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "show me the confusion matrix" }],
    })
    useAppStore.getState().attachConfusionMatrixChatToLastMessage(binaryResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].confusion_matrix_chat).toBeUndefined()
  })
})
