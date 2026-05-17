/**
 * Tests for ConfidenceThresholdCard (chat-inline confidence threshold config card).
 *
 * Covers:
 *  1.  Renders region with aria-label "Confidence threshold card"
 *  2.  Shows 🎯 icon (aria-hidden)
 *  3.  Shows "Confidence Threshold" heading
 *  4.  Shows "Min X% confidence" badge when enabled
 *  5.  Shows "Disabled" badge when threshold not enabled
 *  6.  Shows amber warning when threshold enabled
 *  7.  Shows below_count_30d when activity and threshold enabled
 *  8.  Shows "All recent predictions met the confidence threshold" when 0 below
 *  9.  Shows "No predictions in the last 30 days yet" when no activity
 * 10.  Shows summary text
 * 11.  Shows help text footer
 * 12.  Shows below_pct as percentage when available
 * 13.  Store: attachConfidenceThresholdConfigToLastMessage attaches to last assistant message
 * 14.  Store: does not attach to user message
 * 15.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ConfidenceThresholdCard } from "@/components/deploy/confidence-threshold-card"
import type { ConfidenceThresholdConfig } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const enabledConfig: ConfidenceThresholdConfig = {
  deployment_id: "dep-1",
  confidence_threshold: 0.8,
  threshold_enabled: true,
  below_threshold_count_30d: 3,
  total_predictions_30d: 50,
  below_threshold_pct: 6.0,
  summary: "Confidence threshold set to 80%. 3 of 50 recent predictions were below threshold.",
}

const enabledNoBelow: ConfidenceThresholdConfig = {
  deployment_id: "dep-2",
  confidence_threshold: 0.7,
  threshold_enabled: true,
  below_threshold_count_30d: 0,
  total_predictions_30d: 20,
  below_threshold_pct: 0.0,
  summary: "Confidence threshold set to 70%. All recent predictions passed.",
}

const disabledConfig: ConfidenceThresholdConfig = {
  deployment_id: "dep-3",
  confidence_threshold: null,
  threshold_enabled: false,
  below_threshold_count_30d: 0,
  total_predictions_30d: 0,
  below_threshold_pct: null,
  summary: "No confidence threshold configured.",
}

const noActivityConfig: ConfidenceThresholdConfig = {
  deployment_id: "dep-4",
  confidence_threshold: 0.75,
  threshold_enabled: true,
  below_threshold_count_30d: 0,
  total_predictions_30d: 0,
  below_threshold_pct: null,
  summary: "Confidence threshold set to 75%. No predictions in the last 30 days.",
}

// ---------------------------------------------------------------------------
// Rendering tests
// ---------------------------------------------------------------------------

test("1. renders region with correct aria-label", () => {
  render(<ConfidenceThresholdCard config={enabledConfig} />)
  expect(screen.getByRole("region", { name: "Confidence threshold card" })).toBeInTheDocument()
})

test("2. shows 🎯 icon", () => {
  render(<ConfidenceThresholdCard config={enabledConfig} />)
  expect(screen.getByText("🎯")).toBeInTheDocument()
})

test("3. shows Confidence Threshold heading", () => {
  render(<ConfidenceThresholdCard config={enabledConfig} />)
  expect(screen.getByText("Confidence Threshold")).toBeInTheDocument()
})

test("4. shows Min X% confidence badge when enabled", () => {
  render(<ConfidenceThresholdCard config={enabledConfig} />)
  expect(screen.getByText("Min 80% confidence")).toBeInTheDocument()
})

test("5. shows Disabled badge when not enabled", () => {
  render(<ConfidenceThresholdCard config={disabledConfig} />)
  expect(screen.getByText("Disabled")).toBeInTheDocument()
})

test("6. shows amber warning paragraph when threshold enabled", () => {
  render(<ConfidenceThresholdCard config={enabledConfig} />)
  expect(screen.getByText(/Predictions with model confidence below/)).toBeInTheDocument()
})

test("7. shows below count and pct when activity and threshold enabled", () => {
  render(<ConfidenceThresholdCard config={enabledConfig} />)
  expect(screen.getByText(/3.*\(6\.0%\)/)).toBeInTheDocument()
})

test("8. shows all-passed message when below_count is 0", () => {
  render(<ConfidenceThresholdCard config={enabledNoBelow} />)
  expect(
    screen.getByText("All recent predictions met the confidence threshold.")
  ).toBeInTheDocument()
})

test("9. shows no-predictions message when total_30d is 0", () => {
  render(<ConfidenceThresholdCard config={noActivityConfig} />)
  expect(screen.getByText("No predictions in the last 30 days yet.")).toBeInTheDocument()
})

test("10. shows summary text", () => {
  render(<ConfidenceThresholdCard config={enabledConfig} />)
  expect(screen.getByText(enabledConfig.summary)).toBeInTheDocument()
})

test("11. shows help text footer", () => {
  render(<ConfidenceThresholdCard config={enabledConfig} />)
  expect(screen.getByText(/set confidence threshold to/i)).toBeInTheDocument()
})

test("12. shows below_pct as percentage when available", () => {
  render(<ConfidenceThresholdCard config={enabledConfig} />)
  expect(screen.getByText(/6\.0%/)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Store tests
// ---------------------------------------------------------------------------

function resetStore() {
  useAppStore.setState({ messages: [] })
}

test("13. attachConfidenceThresholdConfigToLastMessage attaches to last assistant message", () => {
  resetStore()
  useAppStore.setState({
    messages: [
      { role: "user", content: "hi" },
      { role: "assistant", content: "response" },
    ],
  })
  useAppStore.getState().attachConfidenceThresholdConfigToLastMessage(enabledConfig)
  const msgs = useAppStore.getState().messages
  expect(msgs[msgs.length - 1].confidence_threshold_config).toEqual(enabledConfig)
})

test("14. does not attach to user message", () => {
  resetStore()
  useAppStore.setState({
    messages: [{ role: "user", content: "hi" }],
  })
  useAppStore.getState().attachConfidenceThresholdConfigToLastMessage(enabledConfig)
  const msgs = useAppStore.getState().messages
  expect(msgs[msgs.length - 1].confidence_threshold_config).toBeUndefined()
})

test("15. does not crash when messages list is empty", () => {
  resetStore()
  expect(() => {
    useAppStore.getState().attachConfidenceThresholdConfigToLastMessage(enabledConfig)
  }).not.toThrow()
})
