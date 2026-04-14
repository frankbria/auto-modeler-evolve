/**
 * Tests for WebhookHistoryCard and Zustand store action.
 *
 * Covers:
 *  1.  Renders region with aria-label "Webhook event history"
 *  2.  Shows 🔔 icon (aria-hidden)
 *  3.  Shows "Webhook Event History" heading
 *  4.  Shows event count badge
 *  5.  Shows summary text
 *  6.  Empty state: renders "No webhook events recorded yet" paragraph
 *  7.  With events: renders event rows
 *  8.  Event row: shows event type badge label
 *  9.  Event row: shows status code OK badge (success)
 * 10.  Event row: shows failed status badge (status_code=0)
 * 11.  Event row: shows webhook URL (truncated)
 * 12.  Footer help text is rendered
 * 13.  Store: attachWebhookHistoryToLastMessage attaches to last assistant message
 * 14.  Store: does not attach to user message
 * 15.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { WebhookHistoryCard } from "@/components/deploy/webhook-history-card"
import type { WebhookHistoryResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const emptyHistory: WebhookHistoryResult = {
  total: 0,
  events: [],
  summary: "No webhook events have fired for this deployment yet.",
}

const withEvents: WebhookHistoryResult = {
  total: 2,
  events: [
    {
      id: "evt-1",
      webhook_id: "hook-1",
      webhook_url: "https://example.com/hook",
      event_type: "batch_complete",
      fired_at: "2026-04-13T10:00:00",
      status_code: 200,
      success: true,
    },
    {
      id: "evt-2",
      webhook_id: "hook-1",
      webhook_url: "https://example.com/hook",
      event_type: "drift_detected",
      fired_at: "2026-04-12T08:30:00",
      status_code: 0,
      success: false,
    },
  ],
  summary: "2 recent webhook events (1 successful). Event types seen: batch_complete, drift_detected.",
}

// Reset store before each test
beforeEach(() => {
  useAppStore.setState({ messages: [] })
})

describe("WebhookHistoryCard", () => {
  it("renders region with aria-label 'Webhook event history'", () => {
    render(<WebhookHistoryCard data={emptyHistory} />)
    expect(screen.getByRole("region", { name: /webhook event history/i })).toBeTruthy()
  })

  it("shows 🔔 icon with aria-hidden", () => {
    render(<WebhookHistoryCard data={emptyHistory} />)
    const icon = screen.getByText("🔔")
    expect(icon.getAttribute("aria-hidden")).toBe("true")
  })

  it("shows 'Webhook Event History' heading", () => {
    render(<WebhookHistoryCard data={emptyHistory} />)
    expect(screen.getByText("Webhook Event History")).toBeTruthy()
  })

  it("shows event count badge (0 events)", () => {
    render(<WebhookHistoryCard data={emptyHistory} />)
    expect(screen.getByText("0 events")).toBeTruthy()
  })

  it("shows summary text", () => {
    render(<WebhookHistoryCard data={emptyHistory} />)
    expect(screen.getByText(/No webhook events have fired/i)).toBeTruthy()
  })

  it("renders empty state paragraph when no events", () => {
    render(<WebhookHistoryCard data={emptyHistory} />)
    expect(screen.getByText(/No webhook events recorded yet/i)).toBeTruthy()
  })

  it("renders event rows when events exist", () => {
    render(<WebhookHistoryCard data={withEvents} />)
    expect(screen.getByText("Batch Complete")).toBeTruthy()
    expect(screen.getByText("Drift Detected")).toBeTruthy()
  })

  it("shows event type badge label for batch_complete", () => {
    render(<WebhookHistoryCard data={withEvents} />)
    expect(screen.getByText("Batch Complete")).toBeTruthy()
  })

  it("shows OK badge for successful event", () => {
    render(<WebhookHistoryCard data={withEvents} />)
    expect(screen.getByText("200 OK")).toBeTruthy()
  })

  it("shows Error badge for status_code=0 (failed event)", () => {
    render(<WebhookHistoryCard data={withEvents} />)
    expect(screen.getByText("Error")).toBeTruthy()
  })

  it("shows webhook URL in event row", () => {
    render(<WebhookHistoryCard data={withEvents} />)
    const urls = screen.getAllByText("https://example.com/hook")
    expect(urls.length).toBeGreaterThan(0)
  })

  it("renders footer help text", () => {
    render(<WebhookHistoryCard data={emptyHistory} />)
    expect(screen.getByText(/Showing up to 10 most recent events/i)).toBeTruthy()
  })
})

describe("Store: attachWebhookHistoryToLastMessage", () => {
  it("attaches webhook_history to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "what webhooks fired?" },
        { id: "2", role: "assistant", content: "Let me check..." },
      ],
    })
    const { attachWebhookHistoryToLastMessage } = useAppStore.getState()
    attachWebhookHistoryToLastMessage(emptyHistory)
    const msgs = useAppStore.getState().messages
    expect((msgs[1] as { webhook_history?: WebhookHistoryResult }).webhook_history).toEqual(emptyHistory)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ id: "1", role: "user", content: "what webhooks fired?" }],
    })
    const { attachWebhookHistoryToLastMessage } = useAppStore.getState()
    attachWebhookHistoryToLastMessage(emptyHistory)
    const msgs = useAppStore.getState().messages
    expect((msgs[0] as { webhook_history?: WebhookHistoryResult }).webhook_history).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    const { attachWebhookHistoryToLastMessage } = useAppStore.getState()
    expect(() => attachWebhookHistoryToLastMessage(emptyHistory)).not.toThrow()
  })
})
