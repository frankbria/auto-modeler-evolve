/**
 * Tests for PredValueAlertCard and attachPredValueAlertConfigToLastMessage.
 *
 * Covers:
 *  1.  Renders container with aria-label "Prediction value trend alert configuration"
 *  2.  Enabled badge shown when pred_value_alert_enabled=true
 *  3.  Disabled badge shown when pred_value_alert_enabled=false
 *  4.  Chart icon rendered (📈)
 *  5.  Summary text rendered
 *  6.  Threshold tile rendered when enabled
 *  7.  Threshold tile not rendered when disabled
 *  8.  Alert percentage displayed correctly
 *  9.  How-it-works info block rendered when enabled
 * 10.  Last fired timestamp rendered when present
 * 11.  "No alerts fired yet" shown when enabled + no last fired
 * 12.  Footer help text rendered
 * 13.  "Prediction Value Trend Alert" heading rendered
 * 14.  Emerald border applied when enabled and never fired
 * 15.  Amber border applied when last fired set
 * 16.  Store: attachPredValueAlertConfigToLastMessage attaches to last assistant message
 * 17.  Store: does not attach to user message
 * 18.  Store: does not crash with empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { PredValueAlertCard } from "@/components/deploy/pred-value-alert-card"
import type { PredValueAlertConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeConfig(
  overrides: Partial<PredValueAlertConfig> = {}
): PredValueAlertConfig {
  return {
    deployment_id: "dep-pva-1",
    pred_value_alert_enabled: true,
    alert_pct: 15.0,
    pred_value_alert_last_fired_at: null,
    cooldown_hours: 24,
    summary:
      "Prediction value trend alert enabled: webhook fires when rolling mean output shifts more than 15%.",
    ...overrides,
  }
}

// ---- 1. Container aria-label ------------------------------------------------
test("renders container with correct aria-label", () => {
  render(<PredValueAlertCard data={makeConfig()} />)
  expect(
    screen.getByRole("region", { name: "Prediction value trend alert configuration" })
  ).toBeInTheDocument()
})

// ---- 2. Enabled badge --------------------------------------------------------
test("shows Enabled badge when pred_value_alert_enabled=true", () => {
  render(<PredValueAlertCard data={makeConfig({ pred_value_alert_enabled: true })} />)
  expect(screen.getByText("Enabled")).toBeInTheDocument()
})

// ---- 3. Disabled badge -------------------------------------------------------
test("shows Disabled badge when pred_value_alert_enabled=false", () => {
  render(
    <PredValueAlertCard
      data={makeConfig({ pred_value_alert_enabled: false, alert_pct: null })}
    />
  )
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

// ---- 4. Chart icon -----------------------------------------------------------
test("renders chart icon", () => {
  render(<PredValueAlertCard data={makeConfig()} />)
  expect(screen.getByText("📈")).toBeInTheDocument()
})

// ---- 5. Summary text ---------------------------------------------------------
test("renders summary text", () => {
  render(<PredValueAlertCard data={makeConfig()} />)
  expect(
    screen.getByText(/webhook fires when rolling mean output shifts more than 15%/i)
  ).toBeInTheDocument()
})

// ---- 6. Threshold tile when enabled -----------------------------------------
test("renders threshold tile when enabled", () => {
  render(<PredValueAlertCard data={makeConfig()} />)
  expect(screen.getAllByText(/Shift threshold/i).length).toBeGreaterThan(0)
})

// ---- 7. Threshold tile hidden when disabled ----------------------------------
test("does not render threshold tile when disabled", () => {
  render(
    <PredValueAlertCard
      data={makeConfig({ pred_value_alert_enabled: false, alert_pct: null })}
    />
  )
  expect(screen.queryByText(/Shift threshold/i)).not.toBeInTheDocument()
})

// ---- 8. Alert percentage displayed ------------------------------------------
test("displays alert percentage correctly", () => {
  render(<PredValueAlertCard data={makeConfig({ alert_pct: 25.0 })} />)
  expect(screen.getAllByText(/25%/i).length).toBeGreaterThan(0)
})

// ---- 9. How-it-works info block ---------------------------------------------
test("renders how-it-works info block when enabled", () => {
  render(<PredValueAlertCard data={makeConfig()} />)
  expect(screen.getByText("How it works")).toBeInTheDocument()
})

// ---- 10. Last fired timestamp -----------------------------------------------
test("renders last fired timestamp when present", () => {
  render(
    <PredValueAlertCard
      data={makeConfig({ pred_value_alert_last_fired_at: "2026-06-01T12:00:00" })}
    />
  )
  expect(screen.getAllByText(/Last fired:/i).length).toBeGreaterThan(0)
})

// ---- 11. "No alerts fired yet" when no last fired ---------------------------
test("shows no-alerts-yet message when enabled and never fired", () => {
  render(<PredValueAlertCard data={makeConfig({ pred_value_alert_last_fired_at: null })} />)
  expect(screen.getByText(/No alerts fired yet/i)).toBeInTheDocument()
})

// ---- 12. Footer help text ---------------------------------------------------
test("renders footer help text", () => {
  render(<PredValueAlertCard data={makeConfig()} />)
  expect(
    screen.getByText(/Distinct from accuracy alerts/i)
  ).toBeInTheDocument()
})

// ---- 13. Heading rendered ---------------------------------------------------
test("renders Prediction Value Trend Alert heading", () => {
  render(<PredValueAlertCard data={makeConfig()} />)
  expect(screen.getByText("Prediction Value Trend Alert")).toBeInTheDocument()
})

// ---- 14. Emerald border when enabled + no fired -----------------------------
test("applies emerald border when enabled and never fired", () => {
  const { container } = render(<PredValueAlertCard data={makeConfig()} />)
  expect(container.firstChild).toHaveClass("border-emerald-300")
})

// ---- 15. Amber border when last fired set -----------------------------------
test("applies amber border when last fired is set", () => {
  const { container } = render(
    <PredValueAlertCard
      data={makeConfig({ pred_value_alert_last_fired_at: "2026-06-01T12:00:00" })}
    />
  )
  expect(container.firstChild).toHaveClass("border-amber-300")
})

// ---- 16. Store: attaches to last assistant message --------------------------
test("attachPredValueAlertConfigToLastMessage attaches to last assistant message", () => {
  const store = useAppStore.getState()
  store.setMessages([
    { id: "1", role: "user", content: "hello" },
    { id: "2", role: "assistant", content: "hi" },
  ])
  store.attachPredValueAlertConfigToLastMessage(makeConfig())
  const msgs = useAppStore.getState().messages
  expect((msgs[1] as any).pred_value_alert_config).toBeDefined()
  expect((msgs[1] as any).pred_value_alert_config.alert_pct).toBe(15.0)
})

// ---- 17. Store: does not attach to user message -----------------------------
test("attachPredValueAlertConfigToLastMessage does not attach to user message", () => {
  const store = useAppStore.getState()
  store.setMessages([{ id: "1", role: "user", content: "hello" }])
  store.attachPredValueAlertConfigToLastMessage(makeConfig())
  const msgs = useAppStore.getState().messages
  expect((msgs[0] as any).pred_value_alert_config).toBeUndefined()
})

// ---- 18. Store: does not crash with empty messages -------------------------
test("attachPredValueAlertConfigToLastMessage does not crash with empty messages", () => {
  const store = useAppStore.getState()
  store.setMessages([])
  expect(() =>
    store.attachPredValueAlertConfigToLastMessage(makeConfig())
  ).not.toThrow()
})
