/**
 * Tests for MilestoneCard component.
 *
 * 1.  Renders the milestone title
 * 2.  Renders the milestone icon
 * 3.  Renders the subtitle badge
 * 4.  Renders the progress bar with correct aria attributes
 * 5.  Renders the summary text
 * 6.  Renders action chips
 * 7.  Clicking an action chip calls onActionClick with the prompt
 * 8.  sr-only figcaption is present for accessibility
 * 9.  Upload milestone uses emerald color class
 * 10. Train milestone uses amber color class
 * 11. Deploy milestone uses violet color class
 * 12. Store: attachMilestoneToLastMessage attaches to last assistant message
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { MilestoneCard } from "@/components/chat/milestone-card"
import type { MilestoneResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const uploadMilestone: MilestoneResult = {
  milestone_type: "upload",
  icon: "🎉",
  title: "Your data is loaded!",
  subtitle: "Step 1 of 4 — Explore",
  summary: "You've uploaded 1,250 rows and 8 columns. Explore your data.",
  progress: 20,
  actions: [
    { label: "Explore my data", prompt: "Show me what's interesting in my data" },
    { label: "Check data quality", prompt: "Is my data ready for modeling?" },
  ],
}

const trainMilestone: MilestoneResult = {
  milestone_type: "train",
  icon: "🎯",
  title: "First model trained!",
  subtitle: "Step 3 of 4 — Validate",
  summary: "Your Random Forest is trained with 87.5% accuracy.",
  progress: 65,
  actions: [
    { label: "Validate my model", prompt: "Show me my model's accuracy" },
    { label: "Deploy for predictions", prompt: "Deploy my model" },
  ],
}

const deployMilestone: MilestoneResult = {
  milestone_type: "deploy",
  icon: "🚀",
  title: "Your model is live!",
  subtitle: "Step 4 of 4 — Monitor",
  summary: "Your prediction endpoint is active.",
  progress: 100,
  actions: [
    { label: "Share the dashboard", prompt: "Show me the prediction dashboard link" },
    { label: "Monitor performance", prompt: "Show me a prediction audit" },
  ],
}

// 1. Renders the milestone title
test("renders the milestone title", () => {
  render(<MilestoneCard result={uploadMilestone} />)
  expect(screen.getByTestId("milestone-title")).toHaveTextContent(
    "Your data is loaded!"
  )
})

// 2. Renders the milestone icon
test("renders the milestone icon", () => {
  render(<MilestoneCard result={uploadMilestone} />)
  expect(screen.getByTestId("milestone-icon")).toHaveTextContent("🎉")
})

// 3. Renders the subtitle badge
test("renders the subtitle badge", () => {
  render(<MilestoneCard result={uploadMilestone} />)
  expect(screen.getByTestId("milestone-subtitle")).toHaveTextContent(
    "Step 1 of 4 — Explore"
  )
})

// 4. Renders progress bar with correct aria
test("renders progress bar with correct aria attributes", () => {
  render(<MilestoneCard result={uploadMilestone} />)
  const bar = screen.getByTestId("milestone-progress")
  expect(bar).toHaveAttribute("role", "progressbar")
  expect(bar).toHaveAttribute("aria-valuenow", "20")
  expect(bar).toHaveAttribute("aria-valuemin", "0")
  expect(bar).toHaveAttribute("aria-valuemax", "100")
})

// 5. Renders the summary text
test("renders the summary text", () => {
  render(<MilestoneCard result={uploadMilestone} />)
  expect(screen.getByTestId("milestone-summary")).toHaveTextContent(
    "You've uploaded 1,250 rows"
  )
})

// 6. Renders action chips
test("renders 2 action chips", () => {
  render(<MilestoneCard result={uploadMilestone} />)
  expect(screen.getByTestId("milestone-action-0")).toBeInTheDocument()
  expect(screen.getByTestId("milestone-action-1")).toBeInTheDocument()
})

// 7. Clicking action chip calls onActionClick with the prompt
test("clicking action chip calls onActionClick with correct prompt", () => {
  const onAction = jest.fn()
  render(<MilestoneCard result={uploadMilestone} onActionClick={onAction} />)
  fireEvent.click(screen.getByTestId("milestone-action-0"))
  expect(onAction).toHaveBeenCalledWith("Show me what's interesting in my data")
})

// 8. sr-only figcaption for accessibility
test("renders sr-only figcaption with title and summary", () => {
  const { container } = render(<MilestoneCard result={uploadMilestone} />)
  const figcaption = container.querySelector("figcaption")
  expect(figcaption).toBeInTheDocument()
  expect(figcaption).toHaveClass("sr-only")
  expect(figcaption?.textContent).toContain("Your data is loaded!")
})

// 9. Upload milestone uses emerald border
test("upload milestone card has emerald border class", () => {
  render(<MilestoneCard result={uploadMilestone} />)
  const card = screen.getByTestId("milestone-card")
  expect(card.className).toContain("border-emerald")
})

// 10. Train milestone uses amber border
test("train milestone card has amber border class", () => {
  render(<MilestoneCard result={trainMilestone} />)
  const card = screen.getByTestId("milestone-card")
  expect(card.className).toContain("border-amber")
})

// 11. Deploy milestone uses violet border
test("deploy milestone card has violet border class", () => {
  render(<MilestoneCard result={deployMilestone} />)
  const card = screen.getByTestId("milestone-card")
  expect(card.className).toContain("border-violet")
})

// 12. Store: attachMilestoneToLastMessage attaches to last assistant message
test("attachMilestoneToLastMessage attaches to last assistant message", () => {
  const store = useAppStore.getState()
  store.setMessages([
    { role: "user", content: "hello", timestamp: "" },
    { role: "assistant", content: "Hi!", timestamp: "" },
  ])
  store.attachMilestoneToLastMessage(uploadMilestone)
  const last = useAppStore.getState().messages.at(-1)
  expect(last?.milestone).toEqual(uploadMilestone)
})
