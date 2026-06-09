/**
 * Tests for AutoRollbackCard and attachAutoRollbackConfigToLastMessage.
 *
 * Covers:
 *  1.  Renders container with aria-label "Auto-rollback configuration"
 *  2.  Enabled badge shown when auto_rollback_enabled=true
 *  3.  Disabled badge shown when auto_rollback_enabled=false
 *  4.  Rollback icon rendered (🔄)
 *  5.  Summary text rendered
 *  6.  Threshold tile rendered when enabled
 *  7.  Threshold tile not rendered when disabled
 *  8.  Threshold percentage displayed correctly
 *  9.  How-it-works info block rendered when enabled
 * 10.  Last triggered timestamp rendered when present
 * 11.  "No auto-rollbacks triggered yet" shown when enabled + no last triggered
 * 12.  Footer help text rendered
 * 13.  "Accuracy-Triggered Auto-Rollback" heading rendered
 * 14.  Emerald border applied when enabled and never triggered
 * 15.  Amber border applied when triggered
 * 16.  Store: attachAutoRollbackConfigToLastMessage attaches to last assistant message
 * 17.  Store: does not attach to user message
 * 18.  Store: does not crash with empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { AutoRollbackCard } from "@/components/deploy/auto-rollback-card"
import type { AutoRollbackConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeConfig(
  overrides: Partial<AutoRollbackConfig> = {}
): AutoRollbackConfig {
  return {
    deployment_id: "dep-arb-1",
    auto_rollback_enabled: true,
    accuracy_threshold_pct: 80.0,
    auto_rollback_triggered_at: null,
    cooldown_hours: 24,
    min_feedback_required: 10,
    summary:
      "Auto-rollback enabled at 80.0% accuracy threshold. The deployment rolls back automatically when feedback accuracy drops below this level.",
    ...overrides,
  }
}

// ---- 1. Container aria-label ------------------------------------------------
test("renders container with correct aria-label", () => {
  render(<AutoRollbackCard data={makeConfig()} />)
  expect(screen.getByRole("region", { name: "Auto-rollback configuration" })).toBeInTheDocument()
})

// ---- 2. Enabled badge --------------------------------------------------------
test("shows Enabled badge when auto_rollback_enabled=true", () => {
  render(<AutoRollbackCard data={makeConfig({ auto_rollback_enabled: true })} />)
  expect(screen.getByText("Enabled")).toBeInTheDocument()
})

// ---- 3. Disabled badge -------------------------------------------------------
test("shows Disabled badge when auto_rollback_enabled=false", () => {
  render(
    <AutoRollbackCard
      data={makeConfig({ auto_rollback_enabled: false, accuracy_threshold_pct: null })}
    />
  )
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

// ---- 4. Rollback icon --------------------------------------------------------
test("renders rollback icon", () => {
  render(<AutoRollbackCard data={makeConfig()} />)
  expect(screen.getByText("🔄")).toBeInTheDocument()
})

// ---- 5. Summary text ---------------------------------------------------------
test("renders summary text", () => {
  render(<AutoRollbackCard data={makeConfig()} />)
  expect(
    screen.getByText(/Auto-rollback enabled at 80\.0% accuracy threshold/i)
  ).toBeInTheDocument()
})

// ---- 6. Threshold tile shown when enabled ------------------------------------
test("renders threshold tile when enabled", () => {
  render(<AutoRollbackCard data={makeConfig({ auto_rollback_enabled: true, accuracy_threshold_pct: 80.0 })} />)
  expect(screen.getByText("Rollback threshold")).toBeInTheDocument()
})

// ---- 7. Threshold tile not shown when disabled --------------------------------
test("does not render threshold tile when disabled", () => {
  render(
    <AutoRollbackCard
      data={makeConfig({ auto_rollback_enabled: false, accuracy_threshold_pct: null })}
    />
  )
  expect(screen.queryByText("Rollback threshold")).not.toBeInTheDocument()
})

// ---- 8. Threshold percentage displayed -----------------------------------------
test("displays threshold percentage correctly", () => {
  render(<AutoRollbackCard data={makeConfig({ accuracy_threshold_pct: 75.5 })} />)
  expect(screen.getAllByText(/75\.5%/).length).toBeGreaterThan(0)
})

// ---- 9. How-it-works block shown when enabled ----------------------------------
test("renders how-it-works info block when enabled", () => {
  render(<AutoRollbackCard data={makeConfig({ auto_rollback_enabled: true })} />)
  expect(screen.getByText("How it works")).toBeInTheDocument()
})

// ---- 10. Last triggered timestamp ----------------------------------------------
test("shows last triggered timestamp when present", () => {
  render(
    <AutoRollbackCard
      data={makeConfig({ auto_rollback_triggered_at: "2026-05-01T12:00:00Z" })}
    />
  )
  expect(screen.getAllByText(/Last triggered/i).length).toBeGreaterThan(0)
})

// ---- 11. "No auto-rollbacks triggered yet" message ----------------------------
test("shows no-triggered message when enabled and no last triggered", () => {
  render(
    <AutoRollbackCard
      data={makeConfig({
        auto_rollback_enabled: true,
        auto_rollback_triggered_at: null,
      })}
    />
  )
  expect(
    screen.getByText(/No auto-rollbacks triggered yet/i)
  ).toBeInTheDocument()
})

// ---- 12. Footer help text -------------------------------------------------------
test("renders footer help text", () => {
  render(<AutoRollbackCard data={makeConfig()} />)
  expect(screen.getByText(/Distinct from manual rollback/i)).toBeInTheDocument()
})

// ---- 13. Heading text -----------------------------------------------------------
test("renders heading with correct text", () => {
  render(<AutoRollbackCard data={makeConfig()} />)
  expect(
    screen.getByText("Accuracy-Triggered Auto-Rollback")
  ).toBeInTheDocument()
})

// ---- 14. Emerald border when enabled + never triggered -------------------------
test("applies emerald border when enabled and never triggered", () => {
  const { container } = render(
    <AutoRollbackCard data={makeConfig({ auto_rollback_triggered_at: null })} />
  )
  const card = container.firstChild as HTMLElement
  expect(card.className).toContain("border-emerald-300")
})

// ---- 15. Amber border when triggered -------------------------------------------
test("applies amber border when triggered", () => {
  const { container } = render(
    <AutoRollbackCard
      data={makeConfig({ auto_rollback_triggered_at: "2026-05-01T12:00:00Z" })}
    />
  )
  const card = container.firstChild as HTMLElement
  expect(card.className).toContain("border-amber-300")
})

// ---- 16-18. Store action tests -------------------------------------------------
describe("store: attachAutoRollbackConfigToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  const config = makeConfig()

  test("attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "enable auto-rollback" },
        { role: "assistant", content: "Done." },
      ],
    })
    useAppStore.getState().attachAutoRollbackConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect((msgs[1] as any).auto_rollback_config).toEqual(config)
  })

  test("does not attach when last message is from user", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "enable auto-rollback" }],
    })
    useAppStore.getState().attachAutoRollbackConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect((msgs[0] as any).auto_rollback_config).toBeUndefined()
  })

  test("does not crash with empty messages array", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachAutoRollbackConfigToLastMessage(config)
    ).not.toThrow()
  })
})
