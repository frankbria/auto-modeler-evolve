/**
 * Tests for QuotaAlertCard (chat-inline quota alert config card) and Zustand store action.
 *
 * Covers:
 *  1.  Renders region with aria-label "Quota alert card"
 *  2.  Shows 🔔 icon (aria-hidden)
 *  3.  Shows "Quota Alert" heading
 *  4.  Shows "Alert at X%" badge when quota_alert_enabled=true
 *  5.  Shows "Disabled" badge when quota_alert_enabled=false
 *  6.  Shows threshold explanation row when enabled
 *  7.  Shows current usage fraction when monthly_quota is set
 *  8.  Shows pct_used value
 *  9.  Shows usage bar (progressbar role)
 * 10.  Shows "remaining" text
 * 11.  Shows summary text
 * 12.  Shows help text footer
 * 13.  Store: attachQuotaAlertConfigToLastMessage attaches to last assistant message
 * 14.  Store: does not attach to user message
 * 15.  Store: does not crash when messages list is empty
 * 16.  No usage bar when monthly_quota is null
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { QuotaAlertCard } from "@/components/deploy/quota-alert-card"
import type { QuotaAlertConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const enabledConfig: QuotaAlertConfig = {
  deployment_id: "dep-1",
  quota_alert_enabled: true,
  quota_alert_threshold_pct: 80,
  monthly_quota: 1000,
  used_this_month: 650,
  pct_used: 65.0,
  summary: "Quota alert set at 80% of 1,000 predictions (currently at 65.0%).",
}

const disabledConfig: QuotaAlertConfig = {
  deployment_id: "dep-2",
  quota_alert_enabled: false,
  quota_alert_threshold_pct: null,
  monthly_quota: null,
  used_this_month: 0,
  pct_used: null,
  summary: "Quota alerts are disabled.",
}

// ---------------------------------------------------------------------------
// Tests — QuotaAlertCard rendering
// ---------------------------------------------------------------------------

test("1. renders region with aria-label", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  expect(screen.getByRole("region", { name: "Quota alert card" })).toBeInTheDocument()
})

test("2. shows bell icon with aria-hidden", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  const icon = screen.getByText("🔔")
  expect(icon).toHaveAttribute("aria-hidden", "true")
})

test("3. shows Quota Alert heading", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  expect(screen.getByText("Quota Alert")).toBeInTheDocument()
})

test("4. shows Alert at N% badge when enabled", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  expect(screen.getByText("Alert at 80%")).toBeInTheDocument()
})

test("5. shows Disabled badge when not enabled", () => {
  render(<QuotaAlertCard config={disabledConfig} />)
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

test("6. shows threshold explanation row when enabled", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  expect(screen.queryAllByText(/80%/).length).toBeGreaterThan(0)
  expect(screen.getByText(/webhook notification/)).toBeInTheDocument()
})

test("7. shows current usage fraction when monthly_quota set", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  // usage: 650 / 1,000 — multiple elements may contain these, so use queryAllByText
  expect(screen.queryAllByText(/650/).length).toBeGreaterThan(0)
  expect(screen.queryAllByText(/1,000/).length).toBeGreaterThan(0)
})

test("8. shows pct_used value", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  expect(screen.queryAllByText(/65%/).length).toBeGreaterThan(0)
})

test("9. shows usage bar with progressbar role", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  expect(screen.getByRole("progressbar")).toBeInTheDocument()
})

test("10. shows remaining text", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  expect(screen.getByText(/350 remaining/)).toBeInTheDocument()
})

test("11. shows summary text", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  expect(
    screen.getByText(/Quota alert set at 80% of 1,000 predictions/)
  ).toBeInTheDocument()
})

test("12. shows help text footer", () => {
  render(<QuotaAlertCard config={enabledConfig} />)
  expect(screen.getByText(/alert me when I hit 80%/)).toBeInTheDocument()
})

test("16. no usage bar when monthly_quota is null", () => {
  render(<QuotaAlertCard config={disabledConfig} />)
  expect(screen.queryByRole("progressbar")).not.toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Tests — Zustand store actions
// ---------------------------------------------------------------------------

beforeEach(() => {
  useAppStore.setState({ messages: [] })
})

test("13. store: attachQuotaAlertConfigToLastMessage attaches to last assistant message", () => {
  useAppStore.setState({
    messages: [
      { id: "1", role: "user", content: "alert at 80%" },
      { id: "2", role: "assistant", content: "Sure!" },
    ],
  })
  useAppStore.getState().attachQuotaAlertConfigToLastMessage(enabledConfig)
  const msgs = useAppStore.getState().messages
  expect(msgs[1].quota_alert_config).toEqual(enabledConfig)
})

test("14. store: does not attach to user message", () => {
  useAppStore.setState({
    messages: [{ id: "1", role: "user", content: "alert at 80%" }],
  })
  useAppStore.getState().attachQuotaAlertConfigToLastMessage(enabledConfig)
  const msgs = useAppStore.getState().messages
  expect(msgs[0].quota_alert_config).toBeUndefined()
})

test("15. store: does not crash when messages list is empty", () => {
  useAppStore.setState({ messages: [] })
  expect(() =>
    useAppStore.getState().attachQuotaAlertConfigToLastMessage(enabledConfig)
  ).not.toThrow()
})
