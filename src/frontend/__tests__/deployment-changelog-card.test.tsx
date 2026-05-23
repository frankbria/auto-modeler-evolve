/**
 * Tests for DeploymentChangelogCard component and deployment-changelog store action.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { DeploymentChangelogCard } from "@/components/deploy/deployment-changelog-card"
import type { DeploymentChangelogResult } from "@/lib/types"

// ── Fixtures ──────────────────────────────────────────────────────────────────

const deployEntry = {
  id: "entry-1",
  change_type: "deployed",
  description: "Deployment created: Linear Regression predicting revenue",
  created_at: new Date(Date.now() - 86_400_000).toISOString(), // 1 day ago
  relative_time: "1d ago",
}

const apiKeyEntry = {
  id: "entry-2",
  change_type: "api_key_added",
  description: "API key authentication enabled — endpoint now requires Bearer token",
  created_at: new Date(Date.now() - 3_600_000).toISOString(), // 1 hour ago
  relative_time: "1h ago",
}

const redeployEntry = {
  id: "entry-3",
  change_type: "redeployed",
  description: "Model updated to version 2: Random Forest (target: revenue)",
  created_at: new Date(Date.now() - 60_000).toISOString(), // 1 min ago
  relative_time: "1m ago",
}

const withEntries: DeploymentChangelogResult = {
  deployment_id: "dep-abc",
  count: 3,
  entries: [redeployEntry, apiKeyEntry, deployEntry],
}

const emptyChangelog: DeploymentChangelogResult = {
  deployment_id: "dep-abc",
  count: 0,
  entries: [],
}

// ── Rendering tests ───────────────────────────────────────────────────────────

describe("DeploymentChangelogCard", () => {
  it("renders the card title", () => {
    render(<DeploymentChangelogCard result={withEntries} />)
    expect(screen.getByTestId("changelog-card-title")).toHaveTextContent(
      "Deployment Changelog"
    )
  })

  it("shows the entry count badge", () => {
    render(<DeploymentChangelogCard result={withEntries} />)
    expect(screen.getByTestId("changelog-count-badge")).toHaveTextContent("3 events")
  })

  it("shows singular 'event' for count=1", () => {
    const single: DeploymentChangelogResult = {
      deployment_id: "dep-1",
      count: 1,
      entries: [deployEntry],
    }
    render(<DeploymentChangelogCard result={single} />)
    expect(screen.getByTestId("changelog-count-badge")).toHaveTextContent("1 event")
  })

  it("renders entry for deployed change_type", () => {
    render(<DeploymentChangelogCard result={withEntries} />)
    const entry = screen.getByTestId("changelog-entry-deployed")
    expect(entry).toBeInTheDocument()
  })

  it("renders entry for api_key_added change_type", () => {
    render(<DeploymentChangelogCard result={withEntries} />)
    expect(screen.getByTestId("changelog-entry-api_key_added")).toBeInTheDocument()
  })

  it("renders entry for redeployed change_type", () => {
    render(<DeploymentChangelogCard result={withEntries} />)
    expect(screen.getByTestId("changelog-entry-redeployed")).toBeInTheDocument()
  })

  it("shows entry description text", () => {
    render(<DeploymentChangelogCard result={withEntries} />)
    expect(
      screen.getByText(/Deployment created: Linear Regression predicting revenue/)
    ).toBeInTheDocument()
  })

  it("shows relative_time for entries", () => {
    render(<DeploymentChangelogCard result={withEntries} />)
    expect(screen.getByText("1d ago")).toBeInTheDocument()
  })

  it("renders empty state when no entries", () => {
    render(<DeploymentChangelogCard result={emptyChangelog} />)
    expect(screen.getByTestId("changelog-empty-state")).toBeInTheDocument()
  })

  it("empty state shows count badge of 0", () => {
    render(<DeploymentChangelogCard result={emptyChangelog} />)
    expect(screen.getByTestId("changelog-count-badge")).toHaveTextContent("0 events")
  })

  it("renders sr-only figcaption for screen readers", () => {
    const { container } = render(<DeploymentChangelogCard result={withEntries} />)
    const figcaption = container.querySelector("figcaption.sr-only")
    expect(figcaption).toBeInTheDocument()
    expect(figcaption?.textContent).toMatch(/Deployment changelog with 3 events/)
  })

  it("sr-only figcaption mentions most recent change", () => {
    const { container } = render(<DeploymentChangelogCard result={withEntries} />)
    const figcaption = container.querySelector("figcaption.sr-only")
    expect(figcaption?.textContent).toMatch(/redeployed/)
  })

  it("renders list with correct aria-label", () => {
    render(<DeploymentChangelogCard result={withEntries} />)
    expect(
      screen.getByRole("list", { name: "Deployment change history" })
    ).toBeInTheDocument()
  })

  it("each entry is a listitem", () => {
    render(<DeploymentChangelogCard result={withEntries} />)
    const items = screen.getAllByRole("listitem")
    expect(items.length).toBe(3)
  })
})

// ── Zustand store action ──────────────────────────────────────────────────────

describe("attachDeploymentChangelogToLastMessage store action", () => {
  it("attaches deployment_changelog to the last assistant message", () => {
    const { useAppStore } = jest.requireActual<typeof import("@/lib/store")>("@/lib/store")
    const store = useAppStore.getState()

    store.setMessages([
      { id: "1", role: "user", content: "show my changelog" },
      { id: "2", role: "assistant", content: "Here is the changelog." },
    ])

    store.attachDeploymentChangelogToLastMessage(withEntries)

    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.deployment_changelog).toBeDefined()
    expect(last.deployment_changelog?.count).toBe(3)
    expect(last.deployment_changelog?.entries).toHaveLength(3)
  })

  it("does not attach to user messages", () => {
    const { useAppStore } = jest.requireActual<typeof import("@/lib/store")>("@/lib/store")
    const store = useAppStore.getState()

    store.setMessages([
      { id: "1", role: "user", content: "hello" },
    ])

    store.attachDeploymentChangelogToLastMessage(withEntries)

    const msgs = useAppStore.getState().messages
    expect((msgs[0] as Record<string, unknown>).deployment_changelog).toBeUndefined()
  })
})
