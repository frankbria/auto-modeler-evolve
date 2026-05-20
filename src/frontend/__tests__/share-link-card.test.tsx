import React from "react"
import { render, screen } from "@testing-library/react"
import { ShareLinkCard } from "@/components/deploy/share-link-card"
import type { ShareLinkResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const base: ShareLinkResult = {
  deployment_id: "dep-xyz",
  dashboard_url: "/predict/dep-xyz",
  prefilled_url: "/predict/dep-xyz?units=100&region=North",
  feature_values: { units: "100", region: "North" },
  feature_count: 2,
  title: "Revenue Predictor",
  summary: "Pre-filled link for 'Revenue Predictor' with 2 values pre-loaded.",
}

const baseEmpty: ShareLinkResult = {
  deployment_id: "dep-abc",
  dashboard_url: "/predict/dep-abc",
  prefilled_url: "/predict/dep-abc",
  feature_values: {},
  feature_count: 0,
  title: "Sales Predictor",
  summary: "Pre-filled link for 'Sales Predictor' (opens at default values).",
}

describe("ShareLinkCard", () => {
  it("renders the card container with accessible label", () => {
    render(<ShareLinkCard result={base} />)
    expect(screen.getByRole("region", { name: /share link card/i })).toBeInTheDocument()
  })

  it("shows the 🔗 icon", () => {
    render(<ShareLinkCard result={base} />)
    expect(screen.getByText("🔗")).toBeInTheDocument()
  })

  it("shows 'Pre-filled Scenario Link' heading", () => {
    render(<ShareLinkCard result={base} />)
    expect(screen.getByTestId("share-link-heading")).toHaveTextContent(
      "Pre-filled Scenario Link"
    )
  })

  it("displays the dashboard title", () => {
    render(<ShareLinkCard result={base} />)
    expect(screen.getByTestId("share-link-title")).toHaveTextContent("Revenue Predictor")
  })

  it("shows feature count badge with values", () => {
    render(<ShareLinkCard result={base} />)
    expect(screen.getByTestId("feature-count-badge")).toHaveTextContent("2 values pre-loaded")
  })

  it("shows 'Opens at defaults' badge when no feature values", () => {
    render(<ShareLinkCard result={baseEmpty} />)
    expect(screen.getByTestId("feature-count-badge")).toHaveTextContent("Opens at defaults")
  })

  it("renders feature chips for each pre-filled value", () => {
    render(<ShareLinkCard result={base} />)
    expect(screen.getByTestId("feature-chip-units")).toBeInTheDocument()
    expect(screen.getByTestId("feature-chip-region")).toBeInTheDocument()
  })

  it("shows feature values list", () => {
    render(<ShareLinkCard result={base} />)
    expect(screen.getByTestId("feature-values-list")).toBeInTheDocument()
  })

  it("omits feature values list when empty", () => {
    render(<ShareLinkCard result={baseEmpty} />)
    expect(screen.queryByTestId("feature-values-list")).not.toBeInTheDocument()
  })

  it("displays the prefilled URL", () => {
    render(<ShareLinkCard result={base} />)
    const urlEl = screen.getByTestId("share-link-url")
    expect(urlEl.textContent).toContain("/predict/dep-xyz")
    expect(urlEl.textContent).toContain("units=100")
    expect(urlEl.textContent).toContain("region=North")
  })

  it("shows the copy button with accessible label", () => {
    render(<ShareLinkCard result={base} />)
    expect(
      screen.getByRole("button", { name: /copy share link to clipboard/i })
    ).toBeInTheDocument()
  })

  it("copy button starts as 'Copy'", () => {
    render(<ShareLinkCard result={base} />)
    expect(screen.getByTestId("copy-share-link-button")).toHaveTextContent("Copy")
  })

  it("shows the summary text when provided", () => {
    render(<ShareLinkCard result={base} />)
    expect(screen.getByTestId("share-link-summary")).toHaveTextContent("Revenue Predictor")
  })

  it("omits summary when not provided", () => {
    render(<ShareLinkCard result={{ ...base, summary: undefined }} />)
    expect(screen.queryByTestId("share-link-summary")).not.toBeInTheDocument()
  })

  it("shows usage instructions", () => {
    render(<ShareLinkCard result={base} />)
    const instructions = screen.getByTestId("share-link-instructions")
    expect(instructions).toBeInTheDocument()
    expect(instructions.textContent).toMatch(/how to use/i)
  })

  it("shows 'Open dashboard →' footer link", () => {
    render(<ShareLinkCard result={base} />)
    expect(screen.getByTestId("open-dashboard-link")).toBeInTheDocument()
  })
})

describe("ShareLinkCard Zustand store", () => {
  it("attachShareLinkToLastMessage attaches to the last assistant message", () => {
    const store = useAppStore.getState()
    store.setMessages([
      { role: "user", content: "give me a share link" },
      { role: "assistant", content: "Here is your pre-filled link." },
    ])
    store.attachShareLinkToLastMessage(base)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].share_link).toEqual(base)
  })

  it("does not attach when last message is user", () => {
    const store = useAppStore.getState()
    store.setMessages([{ role: "user", content: "give me a share link" }])
    store.attachShareLinkToLastMessage(base)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].share_link).toBeUndefined()
  })
})
