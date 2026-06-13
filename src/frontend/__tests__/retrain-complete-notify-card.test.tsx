/**
 * Tests for RetrainCompleteNotifyCard and attachRetrainCompleteNotifyToLastMessage.
 *
 * Covers:
 *  1.  Renders container with data-testid "retrain-complete-notify-card"
 *  2.  Renders heading "Retrain Completion Notification"
 *  3.  Renders notification icon (🔔)
 *  4.  "Webhook registered" badge shown when has_notification=true
 *  5.  "No webhook" badge shown when has_notification=false
 *  6.  Summary text rendered
 *  7.  Last completed retrain section shown when last_completed_retrain is set
 *  8.  Algorithm name shown in last completed retrain
 *  9.  Training duration shown in last completed retrain
 * 10.  Primary metric value shown in last completed retrain
 * 11.  "No completed model runs found" shown when last_completed_retrain is null
 * 12.  Webhook URL list shown when has_notification=true
 * 13.  How-it-works info block rendered
 * 14.  Footer "Distinct from" note rendered
 * 15.  Emerald border applied when has webhooks + last run
 * 16.  Amber border applied when no webhooks
 * 17.  Violet border applied when has webhooks but no last run
 * 18.  Store: attachRetrainCompleteNotifyToLastMessage attaches to last assistant message
 * 19.  Store: does not attach to user message
 * 20.  Store: does not crash with empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { RetrainCompleteNotifyCard } from "@/components/deploy/retrain-complete-notify-card"
import type { RetrainCompleteNotifyResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeResult(
  overrides: Partial<RetrainCompleteNotifyResult> = {}
): RetrainCompleteNotifyResult {
  return {
    deployment_id: "dep-rcn-1",
    has_notification: false,
    retrain_complete_webhooks: [],
    last_completed_retrain: null,
    summary:
      "No retrain_complete webhooks registered. Register a webhook with event type 'retrain_complete' to be notified when auto-retraining finishes.",
    ...overrides,
  }
}

const lastRun = {
  run_id: "run-123",
  algorithm: "random_forest",
  status: "done",
  primary_metric: "accuracy",
  primary_metric_value: 0.9234,
  training_duration_ms: 4500,
  completed_at: "2026-06-12T10:30:00Z",
}

// ---- 1. Container testid -----------------------------------------------------
test("renders container with data-testid", () => {
  render(<RetrainCompleteNotifyCard data={makeResult()} />)
  expect(screen.getByTestId("retrain-complete-notify-card")).toBeInTheDocument()
})

// ---- 2. Heading --------------------------------------------------------------
test("renders heading 'Retrain Completion Notification'", () => {
  render(<RetrainCompleteNotifyCard data={makeResult()} />)
  expect(
    screen.getByText("Retrain Completion Notification")
  ).toBeInTheDocument()
})

// ---- 3. Icon -----------------------------------------------------------------
test("renders notification icon", () => {
  render(<RetrainCompleteNotifyCard data={makeResult()} />)
  expect(screen.getByText("🔔")).toBeInTheDocument()
})

// ---- 4. "Webhook registered" badge -------------------------------------------
test("shows 'Webhook registered' badge when has_notification=true", () => {
  render(
    <RetrainCompleteNotifyCard
      data={makeResult({
        has_notification: true,
        retrain_complete_webhooks: ["https://hooks.example.com/notify"],
      })}
    />
  )
  expect(screen.getByText("Webhook registered")).toBeInTheDocument()
})

// ---- 5. "No webhook" badge ---------------------------------------------------
test("shows 'No webhook' badge when has_notification=false", () => {
  render(<RetrainCompleteNotifyCard data={makeResult()} />)
  expect(screen.getByText("No webhook")).toBeInTheDocument()
})

// ---- 6. Summary text ---------------------------------------------------------
test("renders summary text", () => {
  const data = makeResult()
  render(<RetrainCompleteNotifyCard data={data} />)
  expect(screen.getByText(data.summary)).toBeInTheDocument()
})

// ---- 7. Last completed retrain section shown ---------------------------------
test("shows last completed retrain section when present", () => {
  render(
    <RetrainCompleteNotifyCard
      data={makeResult({ last_completed_retrain: lastRun })}
    />
  )
  expect(screen.getByText("Last completed retrain")).toBeInTheDocument()
})

// ---- 8. Algorithm name -------------------------------------------------------
test("shows algorithm name in last completed retrain", () => {
  render(
    <RetrainCompleteNotifyCard
      data={makeResult({ last_completed_retrain: lastRun })}
    />
  )
  expect(screen.getByTestId("last-algorithm")).toHaveTextContent("random_forest")
})

// ---- 9. Training duration ----------------------------------------------------
test("shows training duration in last completed retrain", () => {
  render(
    <RetrainCompleteNotifyCard
      data={makeResult({ last_completed_retrain: lastRun })}
    />
  )
  // 4500ms = 4.5s
  expect(screen.getByTestId("training-duration")).toHaveTextContent("4.5s")
})

// ---- 10. Primary metric value -----------------------------------------------
test("shows primary metric value in last completed retrain", () => {
  render(
    <RetrainCompleteNotifyCard
      data={makeResult({ last_completed_retrain: lastRun })}
    />
  )
  expect(screen.getByTestId("primary-metric-value")).toHaveTextContent("0.9234")
})

// ---- 11. "No completed model runs" when null ---------------------------------
test("shows 'No completed model runs' when last_completed_retrain is null", () => {
  render(<RetrainCompleteNotifyCard data={makeResult()} />)
  expect(
    screen.getByText(/No completed model runs found yet/i)
  ).toBeInTheDocument()
})

// ---- 12. Webhook URL list ----------------------------------------------------
test("shows webhook URLs when has_notification=true", () => {
  render(
    <RetrainCompleteNotifyCard
      data={makeResult({
        has_notification: true,
        retrain_complete_webhooks: [
          "https://hooks.slack.com/services/T/B/x",
          "https://zapier.com/hooks/catch/1",
        ],
      })}
    />
  )
  expect(screen.getByTestId("webhook-url-0")).toBeInTheDocument()
  expect(screen.getByTestId("webhook-url-1")).toBeInTheDocument()
})

// ---- 13. How-it-works info block --------------------------------------------
test("renders how-it-works info block", () => {
  render(<RetrainCompleteNotifyCard data={makeResult()} />)
  expect(
    screen.getByText("How retrain completion notifications work")
  ).toBeInTheDocument()
})

// ---- 14. Footer note --------------------------------------------------------
test("renders footer 'Distinct from' note", () => {
  render(<RetrainCompleteNotifyCard data={makeResult()} />)
  expect(screen.getByText(/Distinct from/i)).toBeInTheDocument()
  expect(screen.getByText(/degradation_retrain/i)).toBeInTheDocument()
})

// ---- 15. Emerald border when has webhooks + last run ------------------------
test("applies emerald border when has_notification=true and last run present", () => {
  const { container } = render(
    <RetrainCompleteNotifyCard
      data={makeResult({
        has_notification: true,
        retrain_complete_webhooks: ["https://hooks.example.com"],
        last_completed_retrain: lastRun,
      })}
    />
  )
  expect(container.firstChild).toHaveClass("border-emerald-300")
})

// ---- 16. Amber border when no webhooks --------------------------------------
test("applies amber border when has_notification=false", () => {
  const { container } = render(
    <RetrainCompleteNotifyCard data={makeResult({ has_notification: false })} />
  )
  expect(container.firstChild).toHaveClass("border-amber-300")
})

// ---- 17. Violet border when has webhooks but no last run --------------------
test("applies violet border when has_notification=true but no last run", () => {
  const { container } = render(
    <RetrainCompleteNotifyCard
      data={makeResult({
        has_notification: true,
        retrain_complete_webhooks: ["https://hooks.example.com"],
        last_completed_retrain: null,
      })}
    />
  )
  expect(container.firstChild).toHaveClass("border-violet-300")
})

// ---- 18. Store: attach to last assistant message ----------------------------
test("attachRetrainCompleteNotifyToLastMessage attaches to last assistant message", () => {
  const store = useAppStore.getState()
  store.setMessages([
    { role: "user", content: "notify me when done" },
    { role: "assistant", content: "Checking..." },
  ])

  const data = makeResult()
  useAppStore.getState().attachRetrainCompleteNotifyToLastMessage(data)

  const messages = useAppStore.getState().messages
  const last = messages[messages.length - 1]
  expect(last.retrain_complete_notify).toEqual(data)
})

// ---- 19. Store: does not attach to user message -----------------------------
test("attachRetrainCompleteNotifyToLastMessage does not modify user message", () => {
  const store = useAppStore.getState()
  store.setMessages([{ role: "user", content: "done?" }])

  useAppStore.getState().attachRetrainCompleteNotifyToLastMessage(makeResult())

  const last = useAppStore.getState().messages[0]
  expect(last.retrain_complete_notify).toBeUndefined()
})

// ---- 20. Store: does not crash with empty messages --------------------------
test("attachRetrainCompleteNotifyToLastMessage does not crash on empty messages", () => {
  const store = useAppStore.getState()
  store.setMessages([])
  expect(() =>
    useAppStore.getState().attachRetrainCompleteNotifyToLastMessage(makeResult())
  ).not.toThrow()
})
