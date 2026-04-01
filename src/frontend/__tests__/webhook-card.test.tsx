/**
 * Tests for WebhookCard component.
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { WebhookCard } from "@/components/deploy/webhook-card"
import { api } from "@/lib/api"
import type { WebhookConfig, WebhookTestResult } from "@/lib/types"

jest.mock("@/lib/api")

const DEPLOYMENT_ID = "dep-test-1"

const mockWebhook: WebhookConfig = {
  id: "wh-1",
  deployment_id: DEPLOYMENT_ID,
  url: "https://example.com/hook",
  event_types: ["batch_complete", "drift_detected"],
  is_active: true,
  created_at: "2024-04-01T10:00:00",
  last_fired_at: null,
  last_status_code: null,
}

describe("WebhookCard", () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it("renders nothing while loading, then shows add button", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([])
    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("+ Add webhook")).toBeInTheDocument()
    )
  })

  it("shows registered webhook URL", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([mockWebhook])
    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("https://example.com/hook")).toBeInTheDocument()
    )
  })

  it("shows event type badges for each registered event", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([mockWebhook])
    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => {
      expect(screen.getByText("batch complete")).toBeInTheDocument()
      expect(screen.getByText("drift detected")).toBeInTheDocument()
    })
  })

  it("shows webhook count badge", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([mockWebhook])
    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("1 registered")).toBeInTheDocument()
    )
  })

  it("shows add form when + Add webhook is clicked", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([])
    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("+ Add webhook")).toBeInTheDocument()
    )
    fireEvent.click(screen.getByText("+ Add webhook"))
    expect(
      screen.getByPlaceholderText("https://your-server.com/webhook")
    ).toBeInTheDocument()
    expect(screen.getByText("Save webhook")).toBeInTheDocument()
  })

  it("creates webhook and shows secret callout once", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([])
    const created: WebhookConfig = {
      ...mockWebhook,
      id: "wh-new",
      secret: "a".repeat(64),
    }
    ;(api.deploy.createWebhook as jest.Mock).mockResolvedValue(created)

    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("+ Add webhook")).toBeInTheDocument()
    )
    fireEvent.click(screen.getByText("+ Add webhook"))
    fireEvent.change(
      screen.getByPlaceholderText("https://your-server.com/webhook"),
      { target: { value: "https://example.com/hook" } }
    )
    fireEvent.click(screen.getByText("Save webhook"))

    await waitFor(() =>
      expect(
        screen.getByText(/Save this secret/)
      ).toBeInTheDocument()
    )
    expect(screen.getByText("a".repeat(64))).toBeInTheDocument()
  })

  it("removes webhook from list on delete", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([mockWebhook])
    ;(api.deploy.deleteWebhook as jest.Mock).mockResolvedValue(undefined)

    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("https://example.com/hook")).toBeInTheDocument()
    )
    fireEvent.click(screen.getByText("Remove"))
    await waitFor(() =>
      expect(
        screen.queryByText("https://example.com/hook")
      ).not.toBeInTheDocument()
    )
    expect(api.deploy.deleteWebhook).toHaveBeenCalledWith(DEPLOYMENT_ID, "wh-1")
  })

  it("shows OK result after successful test dispatch", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([mockWebhook])
    const testResult: WebhookTestResult = {
      webhook_id: "wh-1",
      url: "https://example.com/hook",
      status_code: 200,
      success: true,
    }
    ;(api.deploy.testWebhook as jest.Mock).mockResolvedValue(testResult)

    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => expect(screen.getByText("Test")).toBeInTheDocument())
    fireEvent.click(screen.getByText("Test"))
    await waitFor(() =>
      expect(screen.getByText("OK (HTTP 200)")).toBeInTheDocument()
    )
  })

  it("shows failure result after failed test dispatch", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([mockWebhook])
    const testResult: WebhookTestResult = {
      webhook_id: "wh-1",
      url: "https://example.com/hook",
      status_code: 404,
      success: false,
    }
    ;(api.deploy.testWebhook as jest.Mock).mockResolvedValue(testResult)

    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() => expect(screen.getByText("Test")).toBeInTheDocument())
    fireEvent.click(screen.getByText("Test"))
    await waitFor(() =>
      expect(screen.getByText("Failed (HTTP 404)")).toBeInTheDocument()
    )
  })

  it("shows last fired timestamp when available", async () => {
    const hook: WebhookConfig = {
      ...mockWebhook,
      last_fired_at: "2024-04-01T15:30:00",
      last_status_code: 200,
    }
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([hook])
    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText(/Last fired:/)).toBeInTheDocument()
    )
    expect(screen.getByText(/HTTP 200/)).toBeInTheDocument()
  })

  it("cancel button hides add form", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([])
    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("+ Add webhook")).toBeInTheDocument()
    )
    fireEvent.click(screen.getByText("+ Add webhook"))
    expect(screen.getByText("Save webhook")).toBeInTheDocument()
    fireEvent.click(screen.getByText("Cancel"))
    expect(screen.queryByText("Save webhook")).not.toBeInTheDocument()
    expect(screen.getByText("+ Add webhook")).toBeInTheDocument()
  })

  it("renders signature header explanation in card", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([])
    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText(/X-AutoModeler-Signature/)).toBeInTheDocument()
    )
  })

  it("Save webhook button disabled when URL empty", async () => {
    ;(api.deploy.getWebhooks as jest.Mock).mockResolvedValue([])
    render(<WebhookCard deploymentId={DEPLOYMENT_ID} />)
    await waitFor(() =>
      expect(screen.getByText("+ Add webhook")).toBeInTheDocument()
    )
    fireEvent.click(screen.getByText("+ Add webhook"))
    const saveBtn = screen.getByText("Save webhook")
    expect(saveBtn).toBeDisabled()
  })
})
