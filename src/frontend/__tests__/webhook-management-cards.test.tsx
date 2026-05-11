/**
 * Tests for webhook management chat card components (Day 61).
 */

import React from "react"
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react"
import { WebhookRegisteredCard } from "@/components/chat/webhook-registered-card"
import { WebhookListChatCard } from "@/components/chat/webhook-list-chat-card"
import { WebhookRemovedChatCard } from "@/components/chat/webhook-removed-chat-card"
import { WebhookTestChatCard } from "@/components/chat/webhook-test-chat-card"
import type {
  WebhookRegisteredInfo,
  WebhookListChatResult,
  WebhookRemovedChatInfo,
  WebhookTestChatResult,
} from "@/lib/types"

// ---------------------------------------------------------------------------
// WebhookRegisteredCard
// ---------------------------------------------------------------------------

const REGISTERED_INFO: WebhookRegisteredInfo = {
  id: "wh-1",
  url: "https://example.com/hook",
  event_types: ["batch_complete", "drift_detected"],
  secret: "whsec_abc123xyz",
  deployment_id: "dep-1",
  summary: "Webhook registered at https://example.com/hook.",
}

describe("WebhookRegisteredCard", () => {
  it("renders heading and url", () => {
    render(<WebhookRegisteredCard info={REGISTERED_INFO} />)
    expect(screen.getByText("Webhook Registered")).toBeInTheDocument()
    expect(screen.getByText("https://example.com/hook")).toBeInTheDocument()
  })

  it("shows Active badge", () => {
    render(<WebhookRegisteredCard info={REGISTERED_INFO} />)
    expect(screen.getByText("Active")).toBeInTheDocument()
  })

  it("renders event type badges", () => {
    render(<WebhookRegisteredCard info={REGISTERED_INFO} />)
    expect(screen.getByText("Batch Complete")).toBeInTheDocument()
    expect(screen.getByText("Drift Detected")).toBeInTheDocument()
  })

  it("displays the signing secret", () => {
    render(<WebhookRegisteredCard info={REGISTERED_INFO} />)
    expect(screen.getByTestId("webhook-secret")).toHaveTextContent("whsec_abc123xyz")
  })

  it("shows 'shown once' warning", () => {
    render(<WebhookRegisteredCard info={REGISTERED_INFO} />)
    expect(screen.getByText(/shown once/i)).toBeInTheDocument()
  })

  it("copy button triggers clipboard write", async () => {
    const writeText = jest.fn().mockResolvedValue(undefined)
    Object.assign(navigator, { clipboard: { writeText } })
    render(<WebhookRegisteredCard info={REGISTERED_INFO} />)
    fireEvent.click(screen.getByRole("button", { name: /copy signing secret/i }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith("whsec_abc123xyz"))
  })

  it("copy button shows 'Copied!' feedback then reverts", async () => {
    jest.useFakeTimers()
    const writeText = jest.fn().mockResolvedValue(undefined)
    Object.assign(navigator, { clipboard: { writeText } })
    render(<WebhookRegisteredCard info={REGISTERED_INFO} />)
    fireEvent.click(screen.getByRole("button", { name: /copy signing secret/i }))
    await waitFor(() => expect(screen.getByText("Copied!")).toBeInTheDocument())
    act(() => jest.advanceTimersByTime(2100))
    await waitFor(() => expect(screen.getByText("Copy")).toBeInTheDocument())
    jest.useRealTimers()
  })

  it("renders summary in sr-only figcaption", () => {
    const { container } = render(<WebhookRegisteredCard info={REGISTERED_INFO} />)
    const caption = container.querySelector("figcaption")
    expect(caption).toHaveTextContent("Webhook registered at https://example.com/hook")
  })
})

// ---------------------------------------------------------------------------
// WebhookListChatCard
// ---------------------------------------------------------------------------

const LIST_RESULT_EMPTY: WebhookListChatResult = {
  webhooks: [],
  total: 0,
  deployment_id: "dep-1",
  summary: "No webhooks registered.",
}

const LIST_RESULT_WITH_HOOKS: WebhookListChatResult = {
  webhooks: [
    {
      id: "wh-1",
      url: "https://example.com/hook",
      event_types: ["batch_complete"],
      created_at: "2024-01-01T00:00:00",
      last_fired_at: null,
      last_status_code: 200,
    },
    {
      id: "wh-2",
      url: "https://other.io/webhook",
      event_types: ["drift_detected", "health_degraded"],
      created_at: "2024-01-02T00:00:00",
      last_fired_at: null,
      last_status_code: null,
    },
  ],
  total: 2,
  deployment_id: "dep-1",
  summary: "2 webhooks registered.",
}

describe("WebhookListChatCard", () => {
  it("shows 'Active Webhooks' heading", () => {
    render(<WebhookListChatCard result={LIST_RESULT_EMPTY} />)
    expect(screen.getByText("Active Webhooks")).toBeInTheDocument()
  })

  it("shows count badge = 0 when empty", () => {
    render(<WebhookListChatCard result={LIST_RESULT_EMPTY} />)
    expect(screen.getByText("0")).toBeInTheDocument()
  })

  it("shows empty state message when no webhooks", () => {
    render(<WebhookListChatCard result={LIST_RESULT_EMPTY} />)
    expect(screen.getAllByText(/no webhooks registered/i).length).toBeGreaterThan(0)
  })

  it("shows count badge = 2 with hooks", () => {
    render(<WebhookListChatCard result={LIST_RESULT_WITH_HOOKS} />)
    expect(screen.getByText("2")).toBeInTheDocument()
  })

  it("renders both webhook URLs", () => {
    render(<WebhookListChatCard result={LIST_RESULT_WITH_HOOKS} />)
    expect(screen.getByText("https://example.com/hook")).toBeInTheDocument()
    expect(screen.getByText("https://other.io/webhook")).toBeInTheDocument()
  })

  it("renders event type badges for each hook", () => {
    render(<WebhookListChatCard result={LIST_RESULT_WITH_HOOKS} />)
    expect(screen.getByText("Batch Complete")).toBeInTheDocument()
    expect(screen.getByText("Drift Detected")).toBeInTheDocument()
    expect(screen.getByText("Health Degraded")).toBeInTheDocument()
  })

  it("shows HTTP 200 status badge for first hook", () => {
    render(<WebhookListChatCard result={LIST_RESULT_WITH_HOOKS} />)
    expect(screen.getByText("200")).toBeInTheDocument()
  })

  it("shows Last fired: never when last_fired_at is null", () => {
    render(<WebhookListChatCard result={LIST_RESULT_WITH_HOOKS} />)
    const neverItems = screen.getAllByText(/last fired: never/i)
    expect(neverItems.length).toBeGreaterThan(0)
  })

  it("renders summary text", () => {
    render(<WebhookListChatCard result={LIST_RESULT_WITH_HOOKS} />)
    expect(screen.getAllByText("2 webhooks registered.").length).toBeGreaterThan(0)
  })
})

// ---------------------------------------------------------------------------
// WebhookRemovedChatCard
// ---------------------------------------------------------------------------

const REMOVED_NONE: WebhookRemovedChatInfo = {
  removed: [],
  deployment_id: "dep-1",
  summary: "No matching webhooks found.",
}

const REMOVED_ONE: WebhookRemovedChatInfo = {
  removed: ["https://example.com/hook"],
  deployment_id: "dep-1",
  summary: "Removed 1 webhook.",
}

describe("WebhookRemovedChatCard", () => {
  it("shows 'Webhook Removed' heading", () => {
    render(<WebhookRemovedChatCard info={REMOVED_NONE} />)
    expect(screen.getByText("Webhook Removed")).toBeInTheDocument()
  })

  it("shows '0 removed' badge when none removed", () => {
    render(<WebhookRemovedChatCard info={REMOVED_NONE} />)
    expect(screen.getByTestId("removed-count-badge")).toHaveTextContent("0 removed")
  })

  it("shows empty state message when none removed", () => {
    render(<WebhookRemovedChatCard info={REMOVED_NONE} />)
    expect(screen.getByText(/no matching webhooks were found/i)).toBeInTheDocument()
  })

  it("shows '1 removed' badge when one removed", () => {
    render(<WebhookRemovedChatCard info={REMOVED_ONE} />)
    expect(screen.getByTestId("removed-count-badge")).toHaveTextContent("1 removed")
  })

  it("renders the removed URL", () => {
    render(<WebhookRemovedChatCard info={REMOVED_ONE} />)
    expect(screen.getByText("https://example.com/hook")).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<WebhookRemovedChatCard info={REMOVED_ONE} />)
    expect(screen.getAllByText("Removed 1 webhook.").length).toBeGreaterThan(0)
  })
})

// ---------------------------------------------------------------------------
// WebhookTestChatCard
// ---------------------------------------------------------------------------

const TEST_NO_WEBHOOK: WebhookTestChatResult = {
  url: null,
  status_code: null,
  success: false,
  deployment_id: "dep-1",
  summary: "No webhooks registered.",
}

const TEST_SUCCESS: WebhookTestChatResult = {
  url: "https://example.com/hook",
  status_code: 200,
  success: true,
  deployment_id: "dep-1",
  summary: "Webhook test succeeded with HTTP 200.",
}

const TEST_FAILURE: WebhookTestChatResult = {
  url: "https://example.com/hook",
  status_code: 500,
  success: false,
  deployment_id: "dep-1",
  summary: "Webhook test failed with HTTP 500.",
}

describe("WebhookTestChatCard", () => {
  it("shows 'Webhook Test' heading", () => {
    render(<WebhookTestChatCard result={TEST_NO_WEBHOOK} />)
    expect(screen.getByText("Webhook Test")).toBeInTheDocument()
  })

  it("shows no-webhook empty state", () => {
    render(<WebhookTestChatCard result={TEST_NO_WEBHOOK} />)
    expect(screen.getAllByText(/no webhooks registered/i).length).toBeGreaterThan(0)
  })

  it("does not show status badge when no webhook", () => {
    render(<WebhookTestChatCard result={TEST_NO_WEBHOOK} />)
    expect(screen.queryByTestId("test-status-badge")).not.toBeInTheDocument()
  })

  it("shows Success badge on success", () => {
    render(<WebhookTestChatCard result={TEST_SUCCESS} />)
    expect(screen.getByTestId("test-status-badge")).toHaveTextContent("Success")
  })

  it("shows URL and HTTP 200 status on success", () => {
    render(<WebhookTestChatCard result={TEST_SUCCESS} />)
    expect(screen.getByText("https://example.com/hook")).toBeInTheDocument()
    expect(screen.getByText("200")).toBeInTheDocument()
  })

  it("shows Failed badge on failure", () => {
    render(<WebhookTestChatCard result={TEST_FAILURE} />)
    expect(screen.getByTestId("test-status-badge")).toHaveTextContent("Failed")
  })

  it("shows error guidance message on failure", () => {
    render(<WebhookTestChatCard result={TEST_FAILURE} />)
    expect(screen.getByText(/publicly accessible/i)).toBeInTheDocument()
  })

  it("shows HTTP 500 badge on failure", () => {
    render(<WebhookTestChatCard result={TEST_FAILURE} />)
    expect(screen.getByText("500")).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<WebhookTestChatCard result={TEST_SUCCESS} />)
    expect(screen.getAllByText("Webhook test succeeded with HTTP 200.").length).toBeGreaterThan(0)
  })
})
