/**
 * Tests for FeatureDriftAlertCard and attachFeatureDriftAlertConfigToLastMessage.
 *
 * Covers:
 *  1.  Renders container with aria-label "Feature drift alert card"
 *  2.  Enabled badge shown when feature_drift_alert_enabled=true
 *  3.  Disabled badge shown when feature_drift_alert_enabled=false
 *  4.  Bell icon rendered
 *  5.  Summary text rendered
 *  6.  Cooldown info rendered when enabled
 *  7.  Cooldown info not rendered when disabled
 *  8.  Last fired timestamp rendered when present
 *  9.  "No alerts fired yet" message shown when enabled + no last fired
 * 10.  "No alerts fired yet" not shown when disabled
 * 11.  Footer help text rendered
 * 12.  "Feature Drift Alert" heading text rendered
 * 13.  Sky border applied to container
 * 14.  Store: attachFeatureDriftAlertConfigToLastMessage attaches to last assistant message
 * 15.  Store: does not attach to user message
 * 16.  Store: does not crash with empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { FeatureDriftAlertCard } from "@/components/deploy/feature-drift-alert-card"
import type { FeatureDriftAlertConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeConfig(
  overrides: Partial<FeatureDriftAlertConfig> = {}
): FeatureDriftAlertConfig {
  return {
    deployment_id: "dep-1",
    feature_drift_alert_enabled: true,
    feature_drift_alert_last_fired_at: null,
    cooldown_hours: 24,
    summary:
      "Feature drift alerting is enabled. A webhook fires (max once per 24 hours) when critical-priority drifted features are detected.",
    ...overrides,
  }
}

// 1. Container renders
test("renders container with aria-label", () => {
  render(<FeatureDriftAlertCard config={makeConfig()} />)
  expect(
    screen.getByRole("region", { name: "Feature drift alert card" })
  ).toBeInTheDocument()
})

// 2. Enabled badge
test("enabled badge shown when feature_drift_alert_enabled=true", () => {
  render(<FeatureDriftAlertCard config={makeConfig({ feature_drift_alert_enabled: true })} />)
  expect(screen.getByText("Enabled")).toBeInTheDocument()
})

// 3. Disabled badge
test("disabled badge shown when feature_drift_alert_enabled=false", () => {
  render(
    <FeatureDriftAlertCard
      config={makeConfig({
        feature_drift_alert_enabled: false,
        summary: "Feature drift alerting is disabled.",
      })}
    />
  )
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

// 4. Bell icon
test("bell icon rendered", () => {
  render(<FeatureDriftAlertCard config={makeConfig()} />)
  expect(screen.getByText("🔔")).toBeInTheDocument()
})

// 5. Summary text
test("summary text is rendered", () => {
  render(
    <FeatureDriftAlertCard
      config={makeConfig({ summary: "Alerting active for this deployment." })}
    />
  )
  expect(
    screen.getByText("Alerting active for this deployment.")
  ).toBeInTheDocument()
})

// 6. Cooldown info rendered when enabled
test("cooldown info rendered when enabled", () => {
  render(<FeatureDriftAlertCard config={makeConfig({ feature_drift_alert_enabled: true })} />)
  expect(screen.getAllByText(/Cooldown/).length).toBeGreaterThan(0)
  expect(screen.getAllByText(/24/).length).toBeGreaterThan(0)
})

// 7. Cooldown info not rendered when disabled
test("cooldown info not rendered when disabled", () => {
  render(
    <FeatureDriftAlertCard
      config={makeConfig({
        feature_drift_alert_enabled: false,
        summary: "Feature drift alerting is disabled.",
      })}
    />
  )
  expect(screen.queryByText(/Cooldown/)).not.toBeInTheDocument()
})

// 8. Last fired timestamp rendered when present
test("last fired timestamp rendered when present", () => {
  render(
    <FeatureDriftAlertCard
      config={makeConfig({
        feature_drift_alert_enabled: true,
        feature_drift_alert_last_fired_at: "2026-06-04T10:30:00",
      })}
    />
  )
  expect(screen.getByText("Last alert fired")).toBeInTheDocument()
})

// 9. "No alerts fired yet" when enabled + no last fired
test("no alerts fired yet message shown when enabled + no last fired", () => {
  render(
    <FeatureDriftAlertCard
      config={makeConfig({
        feature_drift_alert_enabled: true,
        feature_drift_alert_last_fired_at: null,
      })}
    />
  )
  expect(
    screen.getByText(/No alerts fired yet/)
  ).toBeInTheDocument()
})

// 10. "No alerts fired yet" not shown when disabled
test("no alerts fired yet not shown when disabled", () => {
  render(
    <FeatureDriftAlertCard
      config={makeConfig({
        feature_drift_alert_enabled: false,
        feature_drift_alert_last_fired_at: null,
        summary: "Feature drift alerting is disabled.",
      })}
    />
  )
  expect(screen.queryByText(/No alerts fired yet/)).not.toBeInTheDocument()
})

// 11. Footer help text
test("footer help text rendered", () => {
  render(<FeatureDriftAlertCard config={makeConfig()} />)
  expect(
    screen.getByText(/enable feature drift alerts/i)
  ).toBeInTheDocument()
})

// 12. Heading text
test("Feature Drift Alert heading text rendered", () => {
  render(<FeatureDriftAlertCard config={makeConfig()} />)
  expect(screen.getByText("Feature Drift Alert")).toBeInTheDocument()
})

// 13. Sky border
test("sky border applied to container", () => {
  const { container } = render(<FeatureDriftAlertCard config={makeConfig()} />)
  expect((container.firstChild as HTMLElement).className).toContain("border-sky")
})

// 14-16. Store actions
describe("attachFeatureDriftAlertConfigToLastMessage store action", () => {
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
    useAppStore.getState().attachFeatureDriftAlertConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].feature_drift_alert_config).toEqual(config)
  })

  test("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "test", id: "u1" }],
    })
    const config = makeConfig()
    useAppStore.getState().attachFeatureDriftAlertConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].feature_drift_alert_config).toBeUndefined()
  })

  test("does not crash with empty messages", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachFeatureDriftAlertConfigToLastMessage(makeConfig())
    ).not.toThrow()
  })
})
