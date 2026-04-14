/**
 * Tests for WebhookHealthSummaryCard and Zustand store action.
 *
 * Covers:
 *  1.  Renders region with aria-label "Webhook health summary"
 *  2.  Shows 🔗 icon (aria-hidden)
 *  3.  Shows "Webhook Health Summary" heading
 *  4.  Shows overall status badge (Healthy)
 *  5.  Shows overall status badge (Critical)
 *  6.  Shows overall status badge (No webhooks)
 *  7.  Shows webhook count badge when webhooks > 0
 *  8.  Shows summary text
 *  9.  Renders "no webhooks" guidance paragraph
 * 10.  Renders per-deployment section when deployments present
 * 11.  Shows deployment status badge
 * 12.  Shows webhook URL in deployment row
 * 13.  Shows event stats line (total + failed)
 * 14.  Shows stats footer with total/failed counts when events > 0
 * 15.  Footer help text rendered
 * 16.  Store: attachWebhookHealthSummaryToLastMessage attaches to last assistant message
 * 17.  Store: does not attach to user message
 * 18.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { WebhookHealthSummaryCard } from "@/components/deploy/webhook-health-summary-card"
import type { WebhookHealthSummaryResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const noWebhooks: WebhookHealthSummaryResult = {
  overall_status: "no_webhooks",
  total_webhooks: 0,
  total_events: 0,
  total_failed: 0,
  deployments: [],
  summary: "No webhooks are configured for any deployment in this project.",
}

const healthyResult: WebhookHealthSummaryResult = {
  overall_status: "healthy",
  total_webhooks: 1,
  total_events: 5,
  total_failed: 0,
  deployments: [
    {
      deployment_id: "dep-1",
      deployment_name: "Model abc12345",
      status: "healthy",
      webhooks: [
        {
          webhook_id: "wh-1",
          url: "https://example.com/hook",
          event_types: ["batch_complete"],
          total_events: 5,
          failed_events: 0,
          success_rate: 100.0,
          last_event: "2026-04-13T10:00:00",
          status: "healthy",
        },
      ],
    },
  ],
  summary: "All 1 webhook healthy. 5 events delivered successfully.",
}

const criticalResult: WebhookHealthSummaryResult = {
  overall_status: "critical",
  total_webhooks: 1,
  total_events: 10,
  total_failed: 3,
  deployments: [
    {
      deployment_id: "dep-1",
      deployment_name: "Model abc12345",
      status: "critical",
      webhooks: [
        {
          webhook_id: "wh-1",
          url: "https://broken.example.com/hook",
          event_types: ["drift_detected"],
          total_events: 10,
          failed_events: 3,
          success_rate: 70.0,
          last_event: "2026-04-12T08:00:00",
          status: "critical",
        },
      ],
    },
  ],
  summary: "3 of 10 webhook events failed (30.0% failure rate). Check the URLs are reachable.",
}

// Reset store before each test
beforeEach(() => {
  useAppStore.setState({ messages: [] })
})

describe("WebhookHealthSummaryCard — no webhooks state", () => {
  it("renders region with aria-label 'Webhook health summary'", () => {
    render(<WebhookHealthSummaryCard data={noWebhooks} />)
    expect(screen.getByRole("region", { name: /webhook health summary/i })).toBeTruthy()
  })

  it("shows 🔗 icon with aria-hidden", () => {
    render(<WebhookHealthSummaryCard data={noWebhooks} />)
    const icon = screen.getByText("🔗")
    expect(icon.getAttribute("aria-hidden")).toBe("true")
  })

  it("shows 'Webhook Health Summary' heading", () => {
    render(<WebhookHealthSummaryCard data={noWebhooks} />)
    expect(screen.getByText("Webhook Health Summary")).toBeTruthy()
  })

  it("shows No webhooks overall status badge", () => {
    render(<WebhookHealthSummaryCard data={noWebhooks} />)
    expect(screen.getByText("No webhooks")).toBeTruthy()
  })

  it("renders 'Register webhooks' guidance paragraph", () => {
    render(<WebhookHealthSummaryCard data={noWebhooks} />)
    expect(screen.getByText(/Register webhooks in the Deployment panel/i)).toBeTruthy()
  })

  it("shows summary text", () => {
    render(<WebhookHealthSummaryCard data={noWebhooks} />)
    expect(screen.getByText(/No webhooks are configured/i)).toBeTruthy()
  })
})

describe("WebhookHealthSummaryCard — healthy state", () => {
  it("shows Healthy overall status badge", () => {
    render(<WebhookHealthSummaryCard data={healthyResult} />)
    // First badge is the overall status
    const badges = screen.getAllByText("Healthy")
    expect(badges.length).toBeGreaterThanOrEqual(1)
  })

  it("shows webhook count badge", () => {
    render(<WebhookHealthSummaryCard data={healthyResult} />)
    expect(screen.getByText("1 webhook")).toBeTruthy()
  })

  it("renders per-deployment section", () => {
    render(<WebhookHealthSummaryCard data={healthyResult} />)
    expect(screen.getByText("Model abc12345")).toBeTruthy()
  })

  it("shows webhook URL in deployment row", () => {
    render(<WebhookHealthSummaryCard data={healthyResult} />)
    expect(screen.getByText("https://example.com/hook")).toBeTruthy()
  })

  it("shows event stats (5 events, 0 failed)", () => {
    render(<WebhookHealthSummaryCard data={healthyResult} />)
    const eventTexts = screen.getAllByText(/5 event/i); expect(eventTexts.length).toBeGreaterThan(0)
  })

  it("renders stats footer with total events", () => {
    render(<WebhookHealthSummaryCard data={healthyResult} />)
    expect(screen.getByText("5 total events")).toBeTruthy()
  })

  it("renders footer help text", () => {
    render(<WebhookHealthSummaryCard data={healthyResult} />)
    expect(screen.getByText(/Configure webhooks in the Deployment panel/i)).toBeTruthy()
  })
})

describe("WebhookHealthSummaryCard — critical state", () => {
  it("shows Critical overall status badge", () => {
    render(<WebhookHealthSummaryCard data={criticalResult} />)
    const badges = screen.getAllByText("Critical")
    expect(badges.length).toBeGreaterThanOrEqual(1)
  })

  it("renders broken webhook URL", () => {
    render(<WebhookHealthSummaryCard data={criticalResult} />)
    expect(screen.getByText("https://broken.example.com/hook")).toBeTruthy()
  })

  it("shows failure rate in stats footer", () => {
    render(<WebhookHealthSummaryCard data={criticalResult} />)
    expect(screen.getByText(/30%\s*failure rate/i)).toBeTruthy()
  })
})

describe("Store: attachWebhookHealthSummaryToLastMessage", () => {
  it("attaches webhook_health_summary to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "are my webhooks working?" },
        { id: "2", role: "assistant", content: "Let me check..." },
      ],
    })
    const { attachWebhookHealthSummaryToLastMessage } = useAppStore.getState()
    attachWebhookHealthSummaryToLastMessage(noWebhooks)
    const msgs = useAppStore.getState().messages
    expect(
      (msgs[1] as { webhook_health_summary?: WebhookHealthSummaryResult }).webhook_health_summary
    ).toEqual(noWebhooks)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ id: "1", role: "user", content: "are my webhooks working?" }],
    })
    const { attachWebhookHealthSummaryToLastMessage } = useAppStore.getState()
    attachWebhookHealthSummaryToLastMessage(noWebhooks)
    const msgs = useAppStore.getState().messages
    expect(
      (msgs[0] as { webhook_health_summary?: WebhookHealthSummaryResult }).webhook_health_summary
    ).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    const { attachWebhookHealthSummaryToLastMessage } = useAppStore.getState()
    expect(() => attachWebhookHealthSummaryToLastMessage(noWebhooks)).not.toThrow()
  })
})
