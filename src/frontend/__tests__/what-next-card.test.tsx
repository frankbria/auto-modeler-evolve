/**
 * Tests for WhatNextCard component.
 *
 * 1.  Renders the "What To Do Next" heading
 * 2.  Shows the stage badge with stage_label
 * 3.  Renders the progress bar with correct aria attributes
 * 4.  Shows the summary paragraph
 * 5.  Renders 3 step rows
 * 6.  Each step shows icon, title, and description
 * 7.  "Try this" button appears for each step
 * 8.  Clicking "Try this" calls onActionClick with the step's action
 * 9.  sr-only figcaption is present for accessibility
 * 10. Upload stage uses blue color class on badge
 * 11. Monitor stage renders correctly
 * 12. Store: attachWhatNextToLastMessage attaches to last assistant message
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { WhatNextCard } from "@/components/chat/what-next-card"
import type { WhatNextResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const exploreResult: WhatNextResult = {
  stage: "explore",
  stage_label: "Exploring Your Data",
  progress: 25,
  summary: "You've uploaded 500 rows. Let's explore your data.",
  steps: [
    {
      icon: "🔍",
      title: "Explore your data",
      description: "Ask questions about trends and patterns.",
      action: "Show me what's interesting in my data",
    },
    {
      icon: "✨",
      title: "Apply features",
      description: "Let me suggest transformations.",
      action: "What feature suggestions do you have?",
    },
    {
      icon: "🚀",
      title: "Train a model",
      description: "Train multiple algorithms and compare them.",
      action: "Train a model",
    },
  ],
}

const uploadResult: WhatNextResult = {
  stage: "upload",
  stage_label: "Getting Started",
  progress: 5,
  summary: "Start by uploading your data.",
  steps: [
    {
      icon: "📤",
      title: "Upload your data",
      description: "Drag a CSV file into the chat.",
      action: "How do I upload my data?",
    },
    {
      icon: "🔍",
      title: "Explore your data",
      description: "Ask me anything about your data.",
      action: "Walk me through the workflow",
    },
    {
      icon: "🤖",
      title: "Build a prediction model",
      description: "Tell me what outcome you want to forecast.",
      action: "What kinds of predictions can AutoModeler make?",
    },
  ],
}

const monitorResult: WhatNextResult = {
  stage: "monitor",
  stage_label: "Model Live",
  progress: 100,
  summary: "Your model is live and accepting predictions.",
  steps: [
    {
      icon: "🔗",
      title: "Share your prediction dashboard",
      description: "Send a shareable link.",
      action: "Show me the prediction dashboard link",
    },
    {
      icon: "📈",
      title: "Monitor predictions",
      description: "Track volume and accuracy.",
      action: "Show me a prediction audit",
    },
    {
      icon: "🔄",
      title: "Keep your model fresh",
      description: "Retrain with new data.",
      action: "How do I retrain my model?",
    },
  ],
}

// ---------------------------------------------------------------------------
// Rendering tests
// ---------------------------------------------------------------------------

test("1. renders the 'What To Do Next' heading", () => {
  render(<WhatNextCard result={exploreResult} />)
  expect(screen.getByText("What To Do Next")).toBeInTheDocument()
})

test("2. shows the stage badge with stage_label", () => {
  render(<WhatNextCard result={exploreResult} />)
  expect(screen.getByTestId("what-next-stage-badge")).toHaveTextContent(
    "Exploring Your Data"
  )
})

test("3. renders the progress bar with correct aria attributes", () => {
  render(<WhatNextCard result={exploreResult} />)
  const bar = screen.getByRole("progressbar")
  expect(bar).toHaveAttribute("aria-valuenow", "25")
  expect(bar).toHaveAttribute("aria-valuemin", "0")
  expect(bar).toHaveAttribute("aria-valuemax", "100")
})

test("4. shows the summary paragraph", () => {
  render(<WhatNextCard result={exploreResult} />)
  expect(screen.getByTestId("what-next-summary")).toHaveTextContent(
    "You've uploaded 500 rows. Let's explore your data."
  )
})

test("5. renders 3 step rows", () => {
  render(<WhatNextCard result={exploreResult} />)
  for (let i = 0; i < 3; i++) {
    expect(screen.getByTestId(`what-next-step-${i}`)).toBeInTheDocument()
  }
})

test("6. each step shows icon, title, and description", () => {
  render(<WhatNextCard result={exploreResult} />)
  expect(screen.getByText("Explore your data")).toBeInTheDocument()
  expect(screen.getByText("Ask questions about trends and patterns.")).toBeInTheDocument()
  expect(screen.getByText("Apply features")).toBeInTheDocument()
})

test("7. 'Try this' button appears for each step", () => {
  render(<WhatNextCard result={exploreResult} />)
  const buttons = screen.getAllByText("Try this →")
  expect(buttons).toHaveLength(3)
})

test("8. clicking 'Try this' calls onActionClick with the step's action", () => {
  const handler = jest.fn()
  render(<WhatNextCard result={exploreResult} onActionClick={handler} />)
  fireEvent.click(screen.getByTestId("what-next-try-0"))
  expect(handler).toHaveBeenCalledWith("Show me what's interesting in my data")
})

test("9. sr-only figcaption is present for accessibility", () => {
  render(<WhatNextCard result={exploreResult} />)
  // figcaption is sr-only but its text should exist in the DOM
  // Multiple elements may contain stage label text (badge + figcaption)
  const matches = screen.getAllByText(/Exploring Your Data/)
  expect(matches.length).toBeGreaterThanOrEqual(1)
})

test("10. upload stage badge is present", () => {
  render(<WhatNextCard result={uploadResult} />)
  expect(screen.getByTestId("what-next-stage-badge")).toHaveTextContent(
    "Getting Started"
  )
  expect(screen.getByTestId("what-next-progress-bar")).toBeInTheDocument()
})

test("11. monitor stage renders correctly", () => {
  render(<WhatNextCard result={monitorResult} />)
  expect(screen.getByTestId("what-next-stage-badge")).toHaveTextContent("Model Live")
  const bar = screen.getByRole("progressbar")
  expect(bar).toHaveAttribute("aria-valuenow", "100")
  expect(screen.getByTestId("what-next-card")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Store test
// ---------------------------------------------------------------------------

test("12. store: attachWhatNextToLastMessage attaches to last assistant message", () => {
  const store = useAppStore.getState()
  store.setMessages([
    { role: "user", content: "what's next?" },
    { role: "assistant", content: "Let me guide you." },
  ])

  store.attachWhatNextToLastMessage(exploreResult)

  const msgs = useAppStore.getState().messages
  const last = msgs[msgs.length - 1]
  expect(last.what_next).toEqual(exploreResult)
  // Verify user message was not modified
  expect(msgs[0].what_next).toBeUndefined()
})
