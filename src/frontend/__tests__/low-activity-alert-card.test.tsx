/**
 * Tests for LowActivityAlertCard and attachLowActivityAlertConfigToLastMessage.
 *
 * Covers:
 *  1.  Renders container with aria-label "Low-activity alert card"
 *  2.  Enabled badge shown when low_activity_alert_enabled=true
 *  3.  Disabled badge shown when low_activity_alert_enabled=false
 *  4.  Muted-bell icon rendered
 *  5.  Summary text rendered
 *  6.  Threshold info rendered when enabled with threshold
 *  7.  Threshold info not rendered when disabled
 *  8.  Last fired timestamp rendered when present
 *  9.  "No alerts fired yet" message shown when enabled + no last fired
 * 10.  "No alerts fired yet" not shown when disabled
 * 11.  Footer help text rendered
 * 12.  "Low-Activity Alert" heading text rendered
 * 13.  Sky border applied to container
 * 14.  Store: attachLowActivityAlertConfigToLastMessage attaches to last assistant message
 * 15.  Store: does not attach to user message
 * 16.  Store: does not crash with empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { LowActivityAlertCard } from "@/components/deploy/low-activity-alert-card"
import type { LowActivityAlertConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeConfig(
  overrides: Partial<LowActivityAlertConfig> = {}
): LowActivityAlertConfig {
  return {
    deployment_id: "dep-1",
    low_activity_alert_enabled: true,
    threshold_per_day: 10,
    low_activity_alert_last_fired_at: null,
    cooldown_hours: 24,
    summary: "Low-activity alert enabled: webhook fires when daily predictions drop below 10.",
    ...overrides,
  }
}

// 1. Container renders
test("renders container with aria-label", () => {
  render(<LowActivityAlertCard config={makeConfig()} />)
  expect(
    screen.getByRole("region", { name: "Low-activity alert card" })
  ).toBeInTheDocument()
})

// 2. Enabled badge
test("enabled badge shown when low_activity_alert_enabled=true", () => {
  render(<LowActivityAlertCard config={makeConfig({ low_activity_alert_enabled: true })} />)
  expect(screen.getByText("Enabled")).toBeInTheDocument()
})

// 3. Disabled badge
test("disabled badge shown when low_activity_alert_enabled=false", () => {
  render(
    <LowActivityAlertCard
      config={makeConfig({
        low_activity_alert_enabled: false,
        threshold_per_day: null,
        summary: "Low-activity alert is not configured.",
      })}
    />
  )
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

// 4. Muted-bell icon
test("muted-bell icon rendered", () => {
  render(<LowActivityAlertCard config={makeConfig()} />)
  expect(screen.getByText("🔕")).toBeInTheDocument()
})

// 5. Summary text
test("summary text is rendered", () => {
  render(
    <LowActivityAlertCard
      config={makeConfig({ summary: "Alert fires when under threshold." })}
    />
  )
  expect(screen.getByText("Alert fires when under threshold.")).toBeInTheDocument()
})

// 6. Threshold info rendered when enabled
test("threshold info rendered when enabled", () => {
  render(<LowActivityAlertCard config={makeConfig({ low_activity_alert_enabled: true, threshold_per_day: 10 })} />)
  expect(screen.getAllByText(/10/).length).toBeGreaterThan(0)
  expect(screen.getAllByText(/Cooldown/).length).toBeGreaterThan(0)
})

// 7. Threshold info not rendered when disabled
test("threshold info not rendered when disabled", () => {
  render(
    <LowActivityAlertCard
      config={makeConfig({
        low_activity_alert_enabled: false,
        threshold_per_day: null,
        summary: "Low-activity alert is not configured.",
      })}
    />
  )
  expect(screen.queryByText(/Cooldown/)).not.toBeInTheDocument()
})

// 8. Last fired timestamp
test("last fired timestamp rendered when present", () => {
  render(
    <LowActivityAlertCard
      config={makeConfig({
        low_activity_alert_enabled: true,
        low_activity_alert_last_fired_at: "2026-06-04T08:00:00",
      })}
    />
  )
  expect(screen.getByText("Last alert fired")).toBeInTheDocument()
})

// 9. "No alerts fired yet" when enabled + no last fired
test("no alerts fired yet message shown when enabled + no last fired", () => {
  render(
    <LowActivityAlertCard
      config={makeConfig({
        low_activity_alert_enabled: true,
        low_activity_alert_last_fired_at: null,
      })}
    />
  )
  expect(screen.getByText(/No alerts fired yet/)).toBeInTheDocument()
})

// 10. "No alerts fired yet" not shown when disabled
test("no alerts fired yet not shown when disabled", () => {
  render(
    <LowActivityAlertCard
      config={makeConfig({
        low_activity_alert_enabled: false,
        threshold_per_day: null,
        low_activity_alert_last_fired_at: null,
        summary: "Low-activity alert is not configured.",
      })}
    />
  )
  expect(screen.queryByText(/No alerts fired yet/)).not.toBeInTheDocument()
})

// 11. Footer help text
test("footer help text rendered", () => {
  render(<LowActivityAlertCard config={makeConfig()} />)
  expect(
    screen.getByText(/alert me if fewer than/i)
  ).toBeInTheDocument()
})

// 12. Heading text
test("Low-Activity Alert heading rendered", () => {
  render(<LowActivityAlertCard config={makeConfig()} />)
  expect(screen.getByText("Low-Activity Alert")).toBeInTheDocument()
})

// 13. Sky border
test("sky border applied to container", () => {
  const { container } = render(<LowActivityAlertCard config={makeConfig()} />)
  expect((container.firstChild as HTMLElement).className).toContain("border-sky")
})

// 14-16. Store actions
describe("attachLowActivityAlertConfigToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  test("attaches to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "test", id: "u1" },
        { role: "assistant", content: "reply", id: "a1" },
      ],
    })
    const config = makeConfig()
    useAppStore.getState().attachLowActivityAlertConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].low_activity_alert_config).toEqual(config)
  })

  test("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "test", id: "u1" }],
    })
    const config = makeConfig()
    useAppStore.getState().attachLowActivityAlertConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].low_activity_alert_config).toBeUndefined()
  })

  test("does not crash with empty messages", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachLowActivityAlertConfigToLastMessage(makeConfig())
    ).not.toThrow()
  })
})
