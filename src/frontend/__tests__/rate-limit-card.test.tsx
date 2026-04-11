/**
 * Tests for RateLimitCard component and rate-limit Zustand store action.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { RateLimitCard } from "@/components/deploy/rate-limit-card"
import type { RateLimitInfo } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const NO_LIMITS: RateLimitInfo = {
  deployment_id: "dep-1",
  rate_limit_rpm: null,
  rate_limit_enabled: false,
  monthly_quota: null,
  quota_enabled: false,
  used_this_month: 0,
  remaining: null,
  pct_used: null,
  summary: "No rate limits configured — endpoint is open access.",
}

const RPM_ONLY: RateLimitInfo = {
  deployment_id: "dep-1",
  rate_limit_rpm: 60,
  rate_limit_enabled: true,
  monthly_quota: null,
  quota_enabled: false,
  used_this_month: 0,
  remaining: null,
  pct_used: null,
  summary: "Rate limits active: 60 requests/minute.",
}

const QUOTA_ONLY: RateLimitInfo = {
  deployment_id: "dep-1",
  rate_limit_rpm: null,
  rate_limit_enabled: false,
  monthly_quota: 1000,
  quota_enabled: true,
  used_this_month: 450,
  remaining: 550,
  pct_used: 45.0,
  summary: "Rate limits active: 1000 predictions/month.",
}

const BOTH_LIMITS: RateLimitInfo = {
  deployment_id: "dep-1",
  rate_limit_rpm: 30,
  rate_limit_enabled: true,
  monthly_quota: 500,
  quota_enabled: true,
  used_this_month: 450,
  remaining: 50,
  pct_used: 90.0,
  summary: "Rate limits active: 30 requests/minute and 500 predictions/month.",
}

describe("RateLimitCard", () => {
  it("renders the rate limit card region", () => {
    render(<RateLimitCard info={NO_LIMITS} />)
    expect(screen.getByRole("region", { name: "Rate limit card" })).toBeInTheDocument()
  })

  it("shows the heading", () => {
    render(<RateLimitCard info={NO_LIMITS} />)
    expect(screen.getByText("Rate Limits & Quotas")).toBeInTheDocument()
  })

  it("shows 'No limits' badge when neither limit is set", () => {
    render(<RateLimitCard info={NO_LIMITS} />)
    expect(screen.getByText("No limits")).toBeInTheDocument()
  })

  it("shows 'Active' badge when rate limit is enabled", () => {
    render(<RateLimitCard info={RPM_ONLY} />)
    expect(screen.getByText("Active")).toBeInTheDocument()
  })

  it("shows rpm value when set", () => {
    render(<RateLimitCard info={RPM_ONLY} />)
    expect(screen.getByText("60 req/min")).toBeInTheDocument()
  })

  it("shows 'Unlimited' for rpm when not set", () => {
    render(<RateLimitCard info={QUOTA_ONLY} />)
    // Multiple "Unlimited" spans may exist
    const unlimitedEls = screen.getAllByText("Unlimited")
    expect(unlimitedEls.length).toBeGreaterThan(0)
  })

  it("shows quota usage fraction when quota is set", () => {
    render(<RateLimitCard info={QUOTA_ONLY} />)
    expect(screen.getByText("450 / 1,000")).toBeInTheDocument()
  })

  it("shows percentage used for quota", () => {
    render(<RateLimitCard info={QUOTA_ONLY} />)
    expect(screen.getByText("45% used")).toBeInTheDocument()
  })

  it("shows remaining predictions", () => {
    render(<RateLimitCard info={QUOTA_ONLY} />)
    expect(screen.getByText("550 remaining")).toBeInTheDocument()
  })

  it("renders progress bar with correct aria attributes", () => {
    render(<RateLimitCard info={QUOTA_ONLY} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-valuenow", "45")
  })

  it("shows both rpm and quota when both are configured", () => {
    render(<RateLimitCard info={BOTH_LIMITS} />)
    expect(screen.getByText("30 req/min")).toBeInTheDocument()
    expect(screen.getByText("450 / 500")).toBeInTheDocument()
  })

  it("renders the summary text", () => {
    render(<RateLimitCard info={BOTH_LIMITS} />)
    expect(screen.getByText(BOTH_LIMITS.summary)).toBeInTheDocument()
  })

  it("shows help text with how to update limits", () => {
    render(<RateLimitCard info={NO_LIMITS} />)
    expect(screen.getByText(/To update:/)).toBeInTheDocument()
  })

  it("decorative emoji is aria-hidden", () => {
    render(<RateLimitCard info={NO_LIMITS} />)
    const emojis = document.querySelectorAll('[aria-hidden="true"]')
    expect(emojis.length).toBeGreaterThan(0)
  })
})

// ---------------------------------------------------------------------------
// Zustand store action
// ---------------------------------------------------------------------------

describe("attachRateLimitToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches rate_limit to the last assistant message", () => {
    useAppStore.setState({
      messages: [{ role: "assistant", content: "hello" }],
    })
    useAppStore.getState().attachRateLimitToLastMessage(NO_LIMITS)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].rate_limit).toEqual(NO_LIMITS)
  })

  it("does not attach to a user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "set rate limit" }],
    })
    useAppStore.getState().attachRateLimitToLastMessage(RPM_ONLY)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].rate_limit).toBeUndefined()
  })

  it("does not attach when messages list is empty", () => {
    useAppStore.getState().attachRateLimitToLastMessage(QUOTA_ONLY)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })
})
