/**
 * Tests for DegradationRetrainCard and attachDegradationRetrainConfigToLastMessage.
 *
 * Covers:
 *  1.  Renders container with aria-label "Degradation-triggered auto-retrain configuration"
 *  2.  Enabled badge shown when degradation_retrain_enabled=true
 *  3.  Disabled badge shown when degradation_retrain_enabled=false
 *  4.  Retrain icon rendered (🔁)
 *  5.  Summary text rendered
 *  6.  Threshold tile rendered when enabled
 *  7.  Threshold tile not rendered when disabled
 *  8.  Threshold percentage displayed correctly
 *  9.  How-it-works info block rendered when enabled
 * 10.  Last triggered timestamp rendered when present
 * 11.  "No auto-retraining triggered yet" shown when enabled + no last triggered
 * 12.  Footer help text rendered
 * 13.  "Degradation-Triggered Auto-Retrain" heading rendered
 * 14.  Violet border applied when enabled and never triggered
 * 15.  Amber border applied when triggered
 * 16.  Store: attachDegradationRetrainConfigToLastMessage attaches to last assistant message
 * 17.  Store: does not attach to user message
 * 18.  Store: does not crash with empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { DegradationRetrainCard } from "@/components/deploy/degradation-retrain-card"
import type { DegradationRetrainConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeConfig(
  overrides: Partial<DegradationRetrainConfig> = {}
): DegradationRetrainConfig {
  return {
    deployment_id: "dep-dr-1",
    degradation_retrain_enabled: true,
    accuracy_threshold_pct: 75.0,
    degradation_retrain_last_triggered_at: null,
    cooldown_hours: 24,
    min_feedback_required: 10,
    summary:
      "Degradation retraining enabled at 75.0% accuracy threshold. The scheduler automatically starts a new training run when feedback accuracy drops below this level.",
    ...overrides,
  }
}

// ---- 1. Container aria-label ------------------------------------------------
test("renders container with correct aria-label", () => {
  render(<DegradationRetrainCard data={makeConfig()} />)
  expect(
    screen.getByRole("region", {
      name: "Degradation-triggered auto-retrain configuration",
    })
  ).toBeInTheDocument()
})

// ---- 2. Enabled badge --------------------------------------------------------
test("shows Enabled badge when degradation_retrain_enabled=true", () => {
  render(<DegradationRetrainCard data={makeConfig({ degradation_retrain_enabled: true })} />)
  expect(screen.getByText("Enabled")).toBeInTheDocument()
})

// ---- 3. Disabled badge -------------------------------------------------------
test("shows Disabled badge when degradation_retrain_enabled=false", () => {
  render(
    <DegradationRetrainCard
      data={makeConfig({
        degradation_retrain_enabled: false,
        accuracy_threshold_pct: null,
      })}
    />
  )
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

// ---- 4. Retrain icon ---------------------------------------------------------
test("renders retrain icon", () => {
  render(<DegradationRetrainCard data={makeConfig()} />)
  expect(screen.getByText("🔁")).toBeInTheDocument()
})

// ---- 5. Summary text ---------------------------------------------------------
test("renders summary text", () => {
  render(<DegradationRetrainCard data={makeConfig()} />)
  expect(
    screen.getByText(/Degradation retraining enabled at 75\.0% accuracy threshold/i)
  ).toBeInTheDocument()
})

// ---- 6. Threshold tile shown when enabled ------------------------------------
test("renders threshold tile when enabled", () => {
  render(
    <DegradationRetrainCard
      data={makeConfig({ degradation_retrain_enabled: true, accuracy_threshold_pct: 75.0 })}
    />
  )
  expect(screen.getByText("Retraining threshold")).toBeInTheDocument()
})

// ---- 7. Threshold tile not shown when disabled --------------------------------
test("does not render threshold tile when disabled", () => {
  render(
    <DegradationRetrainCard
      data={makeConfig({ degradation_retrain_enabled: false, accuracy_threshold_pct: null })}
    />
  )
  expect(screen.queryByText("Retraining threshold")).not.toBeInTheDocument()
})

// ---- 8. Threshold percentage displayed -----------------------------------------
test("displays threshold percentage correctly", () => {
  render(<DegradationRetrainCard data={makeConfig({ accuracy_threshold_pct: 80.5 })} />)
  expect(screen.getAllByText(/80\.5%/).length).toBeGreaterThan(0)
})

// ---- 9. How-it-works block shown when enabled ----------------------------------
test("renders how-it-works info block when enabled", () => {
  render(<DegradationRetrainCard data={makeConfig({ degradation_retrain_enabled: true })} />)
  expect(screen.getByText("How it works")).toBeInTheDocument()
})

// ---- 10. Last triggered timestamp ----------------------------------------------
test("shows last triggered timestamp when present", () => {
  render(
    <DegradationRetrainCard
      data={makeConfig({ degradation_retrain_last_triggered_at: "2026-05-01T12:00:00Z" })}
    />
  )
  expect(screen.getAllByText(/Last triggered/i).length).toBeGreaterThan(0)
})

// ---- 11. "No auto-retraining triggered yet" message ---------------------------
test("shows no-triggered message when enabled and no last triggered", () => {
  render(
    <DegradationRetrainCard
      data={makeConfig({
        degradation_retrain_enabled: true,
        degradation_retrain_last_triggered_at: null,
      })}
    />
  )
  expect(
    screen.getByText(/No auto-retraining triggered yet/i)
  ).toBeInTheDocument()
})

// ---- 12. Footer help text -------------------------------------------------------
test("renders footer help text", () => {
  render(<DegradationRetrainCard data={makeConfig()} />)
  expect(screen.getByText(/Distinct from auto-rollback/i)).toBeInTheDocument()
})

// ---- 13. Heading text -----------------------------------------------------------
test("renders heading with correct text", () => {
  render(<DegradationRetrainCard data={makeConfig()} />)
  expect(
    screen.getByText("Degradation-Triggered Auto-Retrain")
  ).toBeInTheDocument()
})

// ---- 14. Violet border when enabled + never triggered -------------------------
test("applies violet border when enabled and never triggered", () => {
  const { container } = render(
    <DegradationRetrainCard data={makeConfig({ degradation_retrain_last_triggered_at: null })} />
  )
  const card = container.firstChild as HTMLElement
  expect(card.className).toContain("border-violet-300")
})

// ---- 15. Amber border when triggered -------------------------------------------
test("applies amber border when triggered", () => {
  const { container } = render(
    <DegradationRetrainCard
      data={makeConfig({ degradation_retrain_last_triggered_at: "2026-05-01T12:00:00Z" })}
    />
  )
  const card = container.firstChild as HTMLElement
  expect(card.className).toContain("border-amber-300")
})

// ---- 16-18. Store action tests -------------------------------------------------
describe("store: attachDegradationRetrainConfigToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  const config = makeConfig()

  test("attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "enable degradation retraining" },
        { role: "assistant", content: "Done." },
      ],
    })
    useAppStore.getState().attachDegradationRetrainConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(
      (msgs[1] as Record<string, unknown>).degradation_retrain_config
    ).toEqual(config)
  })

  test("does not attach when last message is from user", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "enable degradation retraining" }],
    })
    useAppStore.getState().attachDegradationRetrainConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(
      (msgs[0] as Record<string, unknown>).degradation_retrain_config
    ).toBeUndefined()
  })

  test("does not crash with empty messages array", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachDegradationRetrainConfigToLastMessage(config)
    ).not.toThrow()
  })
})
