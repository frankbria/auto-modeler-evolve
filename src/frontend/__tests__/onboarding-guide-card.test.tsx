/**
 * Tests for OnboardingGuideCard component and store action.
 *
 * Covers:
 *  1.  Renders figure with correct aria-label
 *  2.  Renders 🧭 icon (aria-hidden)
 *  3.  Shows "Getting Started Guide" heading for incomplete state
 *  4.  Shows "You're all set!" heading when complete
 *  5.  Renders completion percentage badge
 *  6.  Renders progress bar with correct aria-valuenow
 *  7.  Renders all 6 step titles
 *  8.  Current step description is visible
 *  9.  Hint text is shown for the current step
 * 10.  CTA button fires onSwitchTab callback
 * 11.  No CTA button when suggested_tab is null
 * 12.  Completed steps show checkmark character
 * 13.  Complete state shows celebration message
 * 14.  Store: attachOnboardingGuideToLastMessage attaches to last assistant message
 * 15.  Store: does not attach to user message
 * 16.  Store: empty messages list handled without crash
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { OnboardingGuideCard } from "@/components/chat/onboarding-guide-card"
import type { OnboardingGuideResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const STEP_NAMES = ["upload", "explore", "target", "train", "validate", "deploy"]
const STEP_TITLES = [
  "Upload your data",
  "Explore your data",
  "Set a prediction target",
  "Train a model",
  "Review your results",
  "Deploy your model",
]

function makeGuide(overrides: Partial<OnboardingGuideResult> = {}): OnboardingGuideResult {
  const steps = STEP_NAMES.map((name, i) => ({
    name,
    title: STEP_TITLES[i],
    description: `Description for step ${i + 1}`,
    hint: `Hint for step ${i + 1}`,
    suggested_action: `Action ${i + 1}`,
    suggested_tab: i > 0 ? "data" : null,
    icon: "🔧",
    is_done: i < 2,      // steps 0+1 done
    is_current: i === 2, // step 2 is current
  }))

  return {
    step_index: 2,
    total_steps: 6,
    completion_pct: 33,
    steps,
    current_step: steps[2],
    is_complete: false,
    summary: "2 of 6 steps complete — next: set a prediction target.",
    ...overrides,
  }
}

function makeCompleteGuide(): OnboardingGuideResult {
  const steps = STEP_NAMES.map((name, i) => ({
    name,
    title: STEP_TITLES[i],
    description: `Description for step ${i + 1}`,
    hint: `Hint for step ${i + 1}`,
    suggested_action: `Action ${i + 1}`,
    suggested_tab: "data",
    icon: "🔧",
    is_done: true,
    is_current: false,
  }))

  return {
    step_index: 6,
    total_steps: 6,
    completion_pct: 100,
    steps,
    current_step: null,
    is_complete: true,
    summary: "All steps complete! Your model is deployed and ready to use.",
  }
}

// ---------------------------------------------------------------------------
// Component tests
// ---------------------------------------------------------------------------

describe("OnboardingGuideCard", () => {
  it("renders figure with aria-label", () => {
    render(<OnboardingGuideCard guide={makeGuide()} />)
    expect(screen.getByRole("figure", { name: /guided onboarding wizard/i })).toBeInTheDocument()
  })

  it("renders 🧭 icon as aria-hidden", () => {
    const { container } = render(<OnboardingGuideCard guide={makeGuide()} />)
    const icon = container.querySelector("[aria-hidden='true']")
    expect(icon).toBeInTheDocument()
    expect(icon?.textContent).toBe("🧭")
  })

  it("shows 'Getting Started Guide' heading for incomplete state", () => {
    render(<OnboardingGuideCard guide={makeGuide()} />)
    expect(screen.getByText("Getting Started Guide")).toBeInTheDocument()
  })

  it("shows 'You're all set!' heading when complete", () => {
    render(<OnboardingGuideCard guide={makeCompleteGuide()} />)
    expect(screen.getByText(/you're all set/i)).toBeInTheDocument()
  })

  it("shows completion percentage badge", () => {
    render(<OnboardingGuideCard guide={makeGuide()} />)
    expect(screen.getByText("33%")).toBeInTheDocument()
  })

  it("renders progress bar with aria-valuenow", () => {
    render(<OnboardingGuideCard guide={makeGuide()} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-valuenow", "33")
  })

  it("renders all 6 step titles", () => {
    render(<OnboardingGuideCard guide={makeGuide()} />)
    for (const title of STEP_TITLES) {
      expect(screen.getByText(title)).toBeInTheDocument()
    }
  })

  it("shows current step description", () => {
    const guide = makeGuide()
    render(<OnboardingGuideCard guide={guide} />)
    expect(screen.getByText("Description for step 3")).toBeInTheDocument()
  })

  it("shows hint text for current step", () => {
    render(<OnboardingGuideCard guide={makeGuide()} />)
    expect(screen.getByText(/💡 Hint for step 3/)).toBeInTheDocument()
  })

  it("CTA button fires onSwitchTab callback", () => {
    const onSwitchTab = jest.fn()
    render(<OnboardingGuideCard guide={makeGuide()} onSwitchTab={onSwitchTab} />)
    const btn = screen.getByRole("button", { name: /action 3/i })
    fireEvent.click(btn)
    expect(onSwitchTab).toHaveBeenCalledWith("data")
  })

  it("no CTA button when suggested_tab is null", () => {
    // Make step 0 current — suggested_tab is null in our fixture
    const guide = makeGuide()
    guide.steps[0].is_current = true
    guide.steps[0].suggested_tab = null
    guide.steps[2].is_current = false
    guide.current_step = guide.steps[0]
    guide.step_index = 0

    const onSwitchTab = jest.fn()
    render(<OnboardingGuideCard guide={guide} onSwitchTab={onSwitchTab} />)
    // Buttons may exist for other steps but the step-0 CTA should not appear
    const buttons = screen.queryAllByRole("button")
    // None of the buttons should call onSwitchTab because step 0 has no tab
    buttons.forEach((btn) => fireEvent.click(btn))
    expect(onSwitchTab).not.toHaveBeenCalled()
  })

  it("completed steps show checkmark character", () => {
    render(<OnboardingGuideCard guide={makeGuide()} />)
    // Steps 0+1 are done → their icons are "✓"
    const checkmarks = screen.getAllByText("✓")
    expect(checkmarks.length).toBeGreaterThanOrEqual(2)
  })

  it("complete state shows celebration message", () => {
    render(<OnboardingGuideCard guide={makeCompleteGuide()} />)
    expect(screen.getByText(/deployed and ready to share/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachOnboardingGuideToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches onboarding_guide to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "help me" },
        { id: "2", role: "assistant", content: "Sure!" },
      ],
    })
    const guide = makeGuide()
    useAppStore.getState().attachOnboardingGuideToLastMessage(guide)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].onboarding_guide).toEqual(guide)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ id: "1", role: "user", content: "help me" }],
    })
    const guide = makeGuide()
    useAppStore.getState().attachOnboardingGuideToLastMessage(guide)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].onboarding_guide).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachOnboardingGuideToLastMessage(makeGuide())
    ).not.toThrow()
  })
})
