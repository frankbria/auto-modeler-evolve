/**
 * Tests for InputDistDriftAlertCard and attachInputDistDriftAlertConfigToLastMessage.
 *
 * Covers:
 *  1.  Renders container with aria-label "Input distribution drift alert card"
 *  2.  Enabled badge shown when input_dist_drift_alert_enabled=true
 *  3.  Disabled badge shown when input_dist_drift_alert_enabled=false
 *  4.  Wave icon rendered
 *  5.  Summary text rendered
 *  6.  Threshold label shows "Medium (≥15% OOR)" for medium threshold
 *  7.  Threshold label shows "High (≥30% OOR)" for high threshold
 *  8.  Cooldown info rendered when enabled
 *  9.  Cooldown info not rendered when disabled
 * 10.  Last fired timestamp rendered when present
 * 11.  "No alerts fired yet" message shown when enabled + no last fired
 * 12.  "No alerts fired yet" not shown when disabled
 * 13.  Footer help text rendered
 * 14.  "Input Distribution Drift Alert" heading text rendered
 * 15.  Cyan border applied to container
 * 16.  Store: attachInputDistDriftAlertConfigToLastMessage attaches to last assistant message
 * 17.  Store: does not attach to user message
 * 18.  Store: does not crash with empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { InputDistDriftAlertCard } from "@/components/deploy/input-dist-drift-alert-card"
import type { InputDistDriftAlertConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeConfig(
  overrides: Partial<InputDistDriftAlertConfig> = {}
): InputDistDriftAlertConfig {
  return {
    deployment_id: "dep-1",
    input_dist_drift_alert_enabled: true,
    input_dist_drift_severity_threshold: "medium",
    input_dist_drift_alert_last_fired_at: null,
    cooldown_hours: 24,
    summary:
      "Input distribution drift alerting is enabled at 'medium' severity. A webhook fires (max once per 24 hours) when live prediction inputs diverge from the training distribution.",
    ...overrides,
  }
}

// 1. Container renders
test("renders container with aria-label", () => {
  render(<InputDistDriftAlertCard config={makeConfig()} />)
  expect(
    screen.getByRole("region", { name: "Input distribution drift alert card" })
  ).toBeInTheDocument()
})

// 2. Enabled badge
test("enabled badge shown when input_dist_drift_alert_enabled=true", () => {
  render(<InputDistDriftAlertCard config={makeConfig({ input_dist_drift_alert_enabled: true })} />)
  expect(screen.getByText("Enabled")).toBeInTheDocument()
})

// 3. Disabled badge
test("disabled badge shown when input_dist_drift_alert_enabled=false", () => {
  render(
    <InputDistDriftAlertCard
      config={makeConfig({
        input_dist_drift_alert_enabled: false,
        summary: "Input distribution drift alerting is disabled.",
      })}
    />
  )
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

// 4. Wave icon
test("wave icon rendered", () => {
  render(<InputDistDriftAlertCard config={makeConfig()} />)
  expect(screen.getByText("🌊")).toBeInTheDocument()
})

// 5. Summary text
test("summary text is rendered", () => {
  render(
    <InputDistDriftAlertCard
      config={makeConfig({ summary: "Input drift alerting active." })}
    />
  )
  expect(screen.getByText("Input drift alerting active.")).toBeInTheDocument()
})

// 6. Medium threshold label
test("renders medium threshold label", () => {
  render(
    <InputDistDriftAlertCard
      config={makeConfig({ input_dist_drift_severity_threshold: "medium", input_dist_drift_alert_enabled: true })}
    />
  )
  expect(screen.getByText("Medium (≥15% OOR)")).toBeInTheDocument()
})

// 7. High threshold label
test("renders high threshold label", () => {
  render(
    <InputDistDriftAlertCard
      config={makeConfig({ input_dist_drift_severity_threshold: "high", input_dist_drift_alert_enabled: true })}
    />
  )
  expect(screen.getByText("High (≥30% OOR)")).toBeInTheDocument()
})

// 8. Cooldown info rendered when enabled
test("cooldown info rendered when enabled", () => {
  render(<InputDistDriftAlertCard config={makeConfig({ input_dist_drift_alert_enabled: true })} />)
  expect(screen.getAllByText(/Cooldown/).length).toBeGreaterThan(0)
  expect(screen.getAllByText(/24/).length).toBeGreaterThan(0)
})

// 9. Cooldown info not rendered when disabled
test("cooldown info not rendered when disabled", () => {
  render(
    <InputDistDriftAlertCard
      config={makeConfig({
        input_dist_drift_alert_enabled: false,
        summary: "Input distribution drift alerting is disabled.",
      })}
    />
  )
  expect(screen.queryByText(/Cooldown/)).not.toBeInTheDocument()
})

// 10. Last fired timestamp rendered when present
test("last fired timestamp rendered when present", () => {
  render(
    <InputDistDriftAlertCard
      config={makeConfig({
        input_dist_drift_alert_enabled: true,
        input_dist_drift_alert_last_fired_at: "2026-06-04T10:30:00",
      })}
    />
  )
  expect(screen.getByText("Last alert fired")).toBeInTheDocument()
})

// 11. "No alerts fired yet" when enabled + no last fired
test("no alerts fired yet message shown when enabled + no last fired", () => {
  render(
    <InputDistDriftAlertCard
      config={makeConfig({
        input_dist_drift_alert_enabled: true,
        input_dist_drift_alert_last_fired_at: null,
      })}
    />
  )
  expect(screen.getByText(/No alerts fired yet/)).toBeInTheDocument()
})

// 12. "No alerts fired yet" not shown when disabled
test("no alerts fired yet not shown when disabled", () => {
  render(
    <InputDistDriftAlertCard
      config={makeConfig({
        input_dist_drift_alert_enabled: false,
        input_dist_drift_alert_last_fired_at: null,
        summary: "Input distribution drift alerting is disabled.",
      })}
    />
  )
  expect(screen.queryByText(/No alerts fired yet/)).not.toBeInTheDocument()
})

// 13. Footer help text
test("footer help text rendered", () => {
  render(<InputDistDriftAlertCard config={makeConfig()} />)
  expect(
    screen.getByText(/enable input distribution drift alerts/i)
  ).toBeInTheDocument()
})

// 14. Heading text
test("Input Distribution Drift Alert heading text rendered", () => {
  render(<InputDistDriftAlertCard config={makeConfig()} />)
  expect(screen.getByText("Input Distribution Drift Alert")).toBeInTheDocument()
})

// 15. Cyan border
test("cyan border applied to container", () => {
  const { container } = render(<InputDistDriftAlertCard config={makeConfig()} />)
  expect((container.firstChild as HTMLElement).className).toContain("border-cyan")
})

// 16-18. Store actions
describe("attachInputDistDriftAlertConfigToLastMessage store action", () => {
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
    useAppStore.getState().attachInputDistDriftAlertConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].input_dist_drift_alert_config).toEqual(config)
  })

  test("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "test", id: "u1" }],
    })
    const config = makeConfig()
    useAppStore.getState().attachInputDistDriftAlertConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect((msgs[msgs.length - 1] as any).input_dist_drift_alert_config).toBeUndefined()
  })

  test("does not crash with empty messages", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachInputDistDriftAlertConfigToLastMessage(makeConfig())
    ).not.toThrow()
  })
})
