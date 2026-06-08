/**
 * Tests for HighActivityBurstCard and attachHighActivityBurstConfigToLastMessage.
 *
 * Covers:
 *  1.  Renders container with aria-label "High-activity burst alert card"
 *  2.  Enabled badge shown when high_activity_burst_enabled=true
 *  3.  Disabled badge shown when high_activity_burst_enabled=false
 *  4.  Chart-up icon rendered
 *  5.  Summary text rendered
 *  6.  Threshold info rendered when enabled with threshold
 *  7.  Threshold info not rendered when disabled
 *  8.  Last fired timestamp rendered when present
 *  9.  "No alerts fired yet" message shown when enabled + no last fired
 * 10.  "No alerts fired yet" not shown when disabled
 * 11.  Footer help text rendered
 * 12.  "High-Activity Burst Alert" heading text rendered
 * 13.  Amber border applied to container
 * 14.  Store: attachHighActivityBurstConfigToLastMessage attaches to last assistant message
 * 15.  Store: does not attach to user message
 * 16.  Store: does not crash with empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { HighActivityBurstCard } from "@/components/deploy/high-activity-burst-card"
import type { HighActivityBurstConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeConfig(
  overrides: Partial<HighActivityBurstConfig> = {}
): HighActivityBurstConfig {
  return {
    deployment_id: "dep-1",
    high_activity_burst_enabled: true,
    threshold_per_hour: 100,
    high_activity_burst_last_fired_at: null,
    cooldown_hours: 1,
    summary: "High-activity burst alert enabled: webhook fires when hourly predictions exceed 100.",
    ...overrides,
  }
}

// 1. Container renders
test("renders container with aria-label", () => {
  render(<HighActivityBurstCard config={makeConfig()} />)
  expect(
    screen.getByRole("region", { name: "High-activity burst alert card" })
  ).toBeInTheDocument()
})

// 2. Enabled badge
test("enabled badge shown when high_activity_burst_enabled=true", () => {
  render(<HighActivityBurstCard config={makeConfig({ high_activity_burst_enabled: true })} />)
  expect(screen.getByText("Enabled")).toBeInTheDocument()
})

// 3. Disabled badge
test("disabled badge shown when high_activity_burst_enabled=false", () => {
  render(
    <HighActivityBurstCard
      config={makeConfig({
        high_activity_burst_enabled: false,
        threshold_per_hour: null,
        summary: "High-activity burst alert is not configured.",
      })}
    />
  )
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

// 4. Chart-up icon
test("chart-up icon rendered", () => {
  render(<HighActivityBurstCard config={makeConfig()} />)
  expect(screen.getByText("📈")).toBeInTheDocument()
})

// 5. Summary text
test("summary text is rendered", () => {
  render(
    <HighActivityBurstCard
      config={makeConfig({ summary: "Alert fires when burst exceeds threshold." })}
    />
  )
  expect(screen.getByText("Alert fires when burst exceeds threshold.")).toBeInTheDocument()
})

// 6. Threshold info rendered when enabled
test("threshold info rendered when enabled", () => {
  render(<HighActivityBurstCard config={makeConfig({ high_activity_burst_enabled: true, threshold_per_hour: 100 })} />)
  expect(screen.getAllByText(/100/).length).toBeGreaterThan(0)
  expect(screen.getAllByText(/Cooldown/).length).toBeGreaterThan(0)
})

// 7. Threshold info not rendered when disabled
test("threshold info not rendered when disabled", () => {
  render(
    <HighActivityBurstCard
      config={makeConfig({
        high_activity_burst_enabled: false,
        threshold_per_hour: null,
        summary: "High-activity burst alert is not configured.",
      })}
    />
  )
  expect(screen.queryByText(/Cooldown/)).not.toBeInTheDocument()
})

// 8. Last fired timestamp
test("last fired timestamp rendered when present", () => {
  render(
    <HighActivityBurstCard
      config={makeConfig({
        high_activity_burst_enabled: true,
        high_activity_burst_last_fired_at: "2026-06-04T08:00:00",
      })}
    />
  )
  expect(screen.getByText("Last alert fired")).toBeInTheDocument()
})

// 9. "No alerts fired yet" when enabled + no last fired
test("no alerts fired yet message shown when enabled + no last fired", () => {
  render(
    <HighActivityBurstCard
      config={makeConfig({
        high_activity_burst_enabled: true,
        high_activity_burst_last_fired_at: null,
      })}
    />
  )
  expect(screen.getByText(/No alerts fired yet/)).toBeInTheDocument()
})

// 10. "No alerts fired yet" not shown when disabled
test("no alerts fired yet not shown when disabled", () => {
  render(
    <HighActivityBurstCard
      config={makeConfig({
        high_activity_burst_enabled: false,
        threshold_per_hour: null,
        high_activity_burst_last_fired_at: null,
        summary: "High-activity burst alert is not configured.",
      })}
    />
  )
  expect(screen.queryByText(/No alerts fired yet/)).not.toBeInTheDocument()
})

// 11. Footer help text
test("footer help text rendered", () => {
  render(<HighActivityBurstCard config={makeConfig()} />)
  expect(
    screen.getByText(/alert me if more than/i)
  ).toBeInTheDocument()
})

// 12. Heading text
test("High-Activity Burst Alert heading rendered", () => {
  render(<HighActivityBurstCard config={makeConfig()} />)
  expect(screen.getByText("High-Activity Burst Alert")).toBeInTheDocument()
})

// 13. Amber border
test("amber border applied to container", () => {
  const { container } = render(<HighActivityBurstCard config={makeConfig()} />)
  expect((container.firstChild as HTMLElement).className).toContain("border-amber")
})

// 14-16. Store actions
describe("attachHighActivityBurstConfigToLastMessage store action", () => {
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
    useAppStore.getState().attachHighActivityBurstConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].high_activity_burst_config).toEqual(config)
  })

  test("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "test", id: "u1" }],
    })
    const config = makeConfig()
    useAppStore.getState().attachHighActivityBurstConfigToLastMessage(config)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].high_activity_burst_config).toBeUndefined()
  })

  test("does not crash with empty messages", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachHighActivityBurstConfigToLastMessage(makeConfig())
    ).not.toThrow()
  })
})
