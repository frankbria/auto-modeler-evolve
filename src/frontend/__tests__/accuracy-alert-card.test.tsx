/**
 * Tests for AccuracyAlertCard (chat-inline accuracy degradation alert config card).
 *
 * Covers:
 *  1.  Renders region with aria-label "Accuracy alert card"
 *  2.  Shows 🎯 icon (aria-hidden)
 *  3.  Shows "Accuracy Alert" heading
 *  4.  Shows "Alert at X%" badge when enabled (classification)
 *  5.  Shows "Disabled" badge when not enabled
 *  6.  Shows "Alert fired" badge when accuracy_alert_fired=true
 *  7.  Shows threshold explanation row when enabled
 *  8.  Shows current metric value for classification (as %)
 *  9.  Shows current metric value for regression
 * 10.  Shows red badge when metric breaches threshold (classification)
 * 11.  Shows green badge when metric is healthy
 * 12.  Shows "Below threshold" text on breach
 * 13.  Shows "no feedback data" message when current_metric is null
 * 14.  Shows summary text
 * 15.  Shows help text footer
 * 16.  Store: attachAccuracyAlertConfigToLastMessage attaches to last assistant message
 * 17.  Store: does not attach to user message
 * 18.  Store: does not crash when messages list is empty
 * 19.  Regression threshold displays as pct error label
 * 20.  Classification threshold normalized display (decimal → %)
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { AccuracyAlertCard } from "@/components/deploy/accuracy-alert-card"
import type { AccuracyAlertConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const enabledClassConfig: AccuracyAlertConfig = {
  deployment_id: "dep-1",
  accuracy_alert_enabled: true,
  accuracy_alert_threshold: 0.8,
  accuracy_alert_fired: false,
  problem_type: "classification",
  metric_label: "accuracy",
  current_metric: 0.72,
  n_feedback: 25,
  summary: "Accuracy alert set at 80%. Current accuracy: 72.0% (25 feedback records).",
}

const enabledRegrConfig: AccuracyAlertConfig = {
  deployment_id: "dep-2",
  accuracy_alert_enabled: true,
  accuracy_alert_threshold: 20.0,
  accuracy_alert_fired: false,
  problem_type: "regression",
  metric_label: "error rate",
  current_metric: 12.5,
  n_feedback: 10,
  summary: "Accuracy alert set at 20% error threshold. Current error: 12.5%.",
}

const disabledConfig: AccuracyAlertConfig = {
  deployment_id: "dep-3",
  accuracy_alert_enabled: false,
  accuracy_alert_threshold: null,
  accuracy_alert_fired: false,
  problem_type: "classification",
  metric_label: "accuracy",
  current_metric: null,
  n_feedback: 0,
  summary: "Accuracy alerts are disabled.",
}

const firedConfig: AccuracyAlertConfig = {
  ...enabledClassConfig,
  accuracy_alert_fired: true,
  current_metric: 0.65,
}

const healthyConfig: AccuracyAlertConfig = {
  ...enabledClassConfig,
  current_metric: 0.92,
}

const noFeedbackConfig: AccuracyAlertConfig = {
  ...enabledClassConfig,
  current_metric: null,
  n_feedback: 0,
  summary: "Accuracy alert configured but no feedback yet.",
}

// ---------------------------------------------------------------------------
// Tests — AccuracyAlertCard rendering
// ---------------------------------------------------------------------------

test("1. renders region with aria-label", () => {
  render(<AccuracyAlertCard config={enabledClassConfig} />)
  expect(screen.getByRole("region", { name: "Accuracy alert card" })).toBeInTheDocument()
})

test("2. shows target icon with aria-hidden", () => {
  render(<AccuracyAlertCard config={enabledClassConfig} />)
  const icon = screen.getByText("🎯")
  expect(icon).toHaveAttribute("aria-hidden", "true")
})

test("3. shows Accuracy Alert heading", () => {
  render(<AccuracyAlertCard config={enabledClassConfig} />)
  expect(screen.getByText("Accuracy Alert")).toBeInTheDocument()
})

test("4. shows threshold badge when enabled (classification)", () => {
  render(<AccuracyAlertCard config={enabledClassConfig} />)
  expect(screen.getByText(/Alert at 80% accuracy/)).toBeInTheDocument()
})

test("5. shows Disabled badge when not enabled", () => {
  render(<AccuracyAlertCard config={disabledConfig} />)
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

test("6. shows fired badge when accuracy_alert_fired=true", () => {
  render(<AccuracyAlertCard config={firedConfig} />)
  expect(screen.getByText(/Alert fired/)).toBeInTheDocument()
})

test("7. shows threshold explanation row when enabled", () => {
  render(<AccuracyAlertCard config={enabledClassConfig} />)
  expect(screen.getByText(/webhook notification/)).toBeInTheDocument()
  expect(screen.queryAllByText(/drops below/).length).toBeGreaterThan(0)
})

test("8. shows current metric value for classification as %", () => {
  render(<AccuracyAlertCard config={enabledClassConfig} />)
  expect(screen.getByText("72.0%")).toBeInTheDocument()
})

test("9. shows current metric value for regression", () => {
  render(<AccuracyAlertCard config={enabledRegrConfig} />)
  expect(screen.getByText("12.5%")).toBeInTheDocument()
})

test("10. shows red badge when metric breaches threshold (classification)", () => {
  // current_metric 0.72 < threshold 0.8 → breach
  render(<AccuracyAlertCard config={enabledClassConfig} />)
  const badge = screen.getByText("72.0%")
  expect(badge.className).toMatch(/red/)
})

test("11. shows green badge when metric is healthy", () => {
  render(<AccuracyAlertCard config={healthyConfig} />)
  const badge = screen.getByText("92.0%")
  expect(badge.className).toMatch(/emerald/)
})

test("12. shows Below threshold text on breach", () => {
  render(<AccuracyAlertCard config={enabledClassConfig} />)
  expect(screen.getByText("Below threshold")).toBeInTheDocument()
})

test("13. shows no feedback message when current_metric is null", () => {
  render(<AccuracyAlertCard config={noFeedbackConfig} />)
  expect(screen.getByText(/No feedback data yet/)).toBeInTheDocument()
})

test("14. shows summary text", () => {
  render(<AccuracyAlertCard config={enabledClassConfig} />)
  expect(
    screen.getByText(/Accuracy alert set at 80%/)
  ).toBeInTheDocument()
})

test("15. shows help text footer", () => {
  render(<AccuracyAlertCard config={enabledClassConfig} />)
  expect(screen.getByText(/alert me when accuracy drops below/)).toBeInTheDocument()
})

test("19. regression threshold shows as error label", () => {
  render(<AccuracyAlertCard config={enabledRegrConfig} />)
  expect(screen.getByText(/Alert at 20% error/)).toBeInTheDocument()
})

test("20. regression threshold displays exceeds language", () => {
  render(<AccuracyAlertCard config={enabledRegrConfig} />)
  expect(screen.getByText(/exceeds/)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Tests — Zustand store actions
// ---------------------------------------------------------------------------

beforeEach(() => {
  useAppStore.setState({ messages: [] })
})

test("16. store: attachAccuracyAlertConfigToLastMessage attaches to last assistant message", () => {
  useAppStore.setState({
    messages: [
      { id: "1", role: "user", content: "alert when accuracy drops below 80%" },
      { id: "2", role: "assistant", content: "Sure!" },
    ],
  })
  useAppStore.getState().attachAccuracyAlertConfigToLastMessage(enabledClassConfig)
  const msgs = useAppStore.getState().messages
  expect(msgs[1].accuracy_alert_config).toEqual(enabledClassConfig)
})

test("17. store: does not attach to user message", () => {
  useAppStore.setState({
    messages: [{ id: "1", role: "user", content: "set accuracy alert" }],
  })
  useAppStore.getState().attachAccuracyAlertConfigToLastMessage(enabledClassConfig)
  const msgs = useAppStore.getState().messages
  expect(msgs[0].accuracy_alert_config).toBeUndefined()
})

test("18. store: does not crash when messages list is empty", () => {
  useAppStore.setState({ messages: [] })
  expect(() =>
    useAppStore.getState().attachAccuracyAlertConfigToLastMessage(enabledClassConfig)
  ).not.toThrow()
})
