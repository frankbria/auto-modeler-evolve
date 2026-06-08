/**
 * Tests for LatencyAlertCard and attachLatencyAlertConfigToLastMessage.
 *
 * Covers:
 *  1.  Renders container with aria-label "Latency alert card"
 *  2.  Enabled badge shown when latency_alert_enabled=true
 *  3.  Disabled badge shown when latency_alert_enabled=false
 *  4.  Timer icon rendered
 *  5.  Summary text rendered
 *  6.  Threshold info rendered when enabled with threshold
 *  7.  Threshold info not rendered when disabled
 *  8.  Last fired timestamp rendered when present
 *  9.  "No alerts fired yet" message shown when enabled + no last fired
 * 10.  "No alerts fired yet" not shown when disabled
 * 11.  Footer help text rendered
 * 12.  "Prediction Latency Alert" heading text rendered
 * 13.  Orange border applied to container
 * 14.  Store: attachLatencyAlertConfigToLastMessage attaches to last assistant message
 * 15.  Store: does not attach to user message
 * 16.  Store: does not crash with empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { LatencyAlertCard } from "@/components/deploy/latency-alert-card"
import type { LatencyAlertConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeConfig(
  overrides: Partial<LatencyAlertConfig> = {}
): LatencyAlertConfig {
  return {
    deployment_id: "dep-1",
    latency_alert_enabled: true,
    threshold_ms: 500,
    latency_alert_last_fired_at: null,
    cooldown_hours: 1,
    summary: "Latency alert enabled: webhook fires when p95 latency exceeds 500 ms.",
    ...overrides,
  }
}

// 1. Container renders
test("renders container with aria-label", () => {
  render(<LatencyAlertCard config={makeConfig()} />)
  expect(
    screen.getByRole("region", { name: "Latency alert card" })
  ).toBeInTheDocument()
})

// 2. Enabled badge
test("enabled badge shown when latency_alert_enabled=true", () => {
  render(<LatencyAlertCard config={makeConfig({ latency_alert_enabled: true })} />)
  expect(screen.getByText("Enabled")).toBeInTheDocument()
})

// 3. Disabled badge
test("disabled badge shown when latency_alert_enabled=false", () => {
  render(
    <LatencyAlertCard
      config={makeConfig({
        latency_alert_enabled: false,
        threshold_ms: null,
        summary: "Latency alert is not configured.",
      })}
    />
  )
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

// 4. Timer icon
test("timer icon rendered", () => {
  render(<LatencyAlertCard config={makeConfig()} />)
  expect(screen.getByText("⏱")).toBeInTheDocument()
})

// 5. Summary text
test("summary text is rendered", () => {
  render(
    <LatencyAlertCard
      config={makeConfig({ summary: "Alert fires when p95 is too slow." })}
    />
  )
  expect(screen.getByText("Alert fires when p95 is too slow.")).toBeInTheDocument()
})

// 6. Threshold info rendered when enabled
test("threshold info rendered when enabled", () => {
  render(<LatencyAlertCard config={makeConfig({ latency_alert_enabled: true, threshold_ms: 500 })} />)
  expect(screen.getAllByText(/500/).length).toBeGreaterThan(0)
  expect(screen.getAllByText(/Cooldown/).length).toBeGreaterThan(0)
})

// 7. Threshold info not rendered when disabled
test("threshold info not rendered when disabled", () => {
  render(
    <LatencyAlertCard
      config={makeConfig({
        latency_alert_enabled: false,
        threshold_ms: null,
        summary: "Latency alert is not configured.",
      })}
    />
  )
  expect(screen.queryByText(/Cooldown/)).not.toBeInTheDocument()
})

// 8. Last fired timestamp
test("last fired timestamp rendered when present", () => {
  render(
    <LatencyAlertCard
      config={makeConfig({
        latency_alert_enabled: true,
        latency_alert_last_fired_at: "2026-06-04T08:00:00",
      })}
    />
  )
  expect(screen.getByText("Last alert fired")).toBeInTheDocument()
})

// 9. "No alerts fired yet" when enabled + no last fired
test("no alerts fired yet message shown when enabled + no last fired", () => {
  render(
    <LatencyAlertCard
      config={makeConfig({
        latency_alert_enabled: true,
        latency_alert_last_fired_at: null,
      })}
    />
  )
  expect(screen.getByText(/No alerts fired yet/)).toBeInTheDocument()
})

// 10. "No alerts fired yet" not shown when disabled
test("no alerts fired yet not shown when disabled", () => {
  render(
    <LatencyAlertCard
      config={makeConfig({
        latency_alert_enabled: false,
        threshold_ms: null,
        latency_alert_last_fired_at: null,
        summary: "Latency alert is not configured.",
      })}
    />
  )
  expect(screen.queryByText(/No alerts fired yet/)).not.toBeInTheDocument()
})

// 11. Footer help text
test("footer help text rendered", () => {
  render(<LatencyAlertCard config={makeConfig()} />)
  expect(
    screen.getByText(/alert me if predictions take more than/i)
  ).toBeInTheDocument()
})

// 12. Heading text
test("Prediction Latency Alert heading rendered", () => {
  render(<LatencyAlertCard config={makeConfig()} />)
  expect(screen.getByText("Prediction Latency Alert")).toBeInTheDocument()
})

// 13. Orange border
test("orange border applied to container", () => {
  const { container } = render(<LatencyAlertCard config={makeConfig()} />)
  expect((container.firstChild as HTMLElement).className).toContain("border-orange")
})

// 14-16. Store actions
describe("attachLatencyAlertConfigToLastMessage store action", () => {
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
    useAppStore.getState().attachLatencyAlertConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].latency_alert_config).toEqual(config)
  })

  test("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "test", id: "u1" }],
    })
    const config = makeConfig()
    useAppStore.getState().attachLatencyAlertConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].latency_alert_config).toBeUndefined()
  })

  test("does not crash with empty messages", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachLatencyAlertConfigToLastMessage(makeConfig())
    ).not.toThrow()
  })
})
