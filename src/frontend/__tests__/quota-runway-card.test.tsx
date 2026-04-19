/**
 * Tests for QuotaRunwayCard — quota runway / capacity planning chat card.
 *
 * Covers:
 *   1. Unlimited state (no monthly quota)
 *   2. Quota set, usage within limits
 *   3. Quota at risk (projected to exhaust)
 *   4. Progress bar percentage
 *   5. Rate limit RPM display
 *   6. Required ARIA attributes
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { QuotaRunwayCard } from "../components/deploy/quota-runway-card"
import type { QuotaRunwayResult } from "../lib/types"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const UNLIMITED: QuotaRunwayResult = {
  deployment_id: "dep-1",
  has_quota: false,
  monthly_quota: null,
  used_this_month: 42,
  remaining: null,
  avg_per_day: 3.0,
  days_left_at_rate: null,
  est_month_total: 90,
  days_remaining_in_month: 12,
  rate_limit_rpm: null,
  will_exhaust: false,
}

const WITHIN_QUOTA: QuotaRunwayResult = {
  deployment_id: "dep-2",
  has_quota: true,
  monthly_quota: 1000,
  used_this_month: 300,
  remaining: 700,
  avg_per_day: 10.0,
  days_left_at_rate: 70.0,
  est_month_total: 420,
  days_remaining_in_month: 12,
  rate_limit_rpm: null,
  will_exhaust: false,
}

const EXHAUSTED: QuotaRunwayResult = {
  deployment_id: "dep-3",
  has_quota: true,
  monthly_quota: 400,
  used_this_month: 300,
  remaining: 100,
  avg_per_day: 30.0,
  days_left_at_rate: 3.3,
  est_month_total: 660,
  days_remaining_in_month: 12,
  rate_limit_rpm: 60,
  will_exhaust: true,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("QuotaRunwayCard — unlimited state", () => {
  it("shows Unlimited badge", () => {
    render(<QuotaRunwayCard result={UNLIMITED} />)
    expect(screen.getByText("Unlimited")).toBeInTheDocument()
  })

  it("shows unlimited predictions message", () => {
    render(<QuotaRunwayCard result={UNLIMITED} />)
    expect(screen.getByText(/No monthly quota configured/i)).toBeInTheDocument()
  })

  it("shows used this month count", () => {
    render(<QuotaRunwayCard result={UNLIMITED} />)
    expect(screen.getByText("42")).toBeInTheDocument()
  })

  it("shows avg predictions/day", () => {
    render(<QuotaRunwayCard result={UNLIMITED} />)
    expect(screen.getByText("3")).toBeInTheDocument()
  })

  it("does not render progress bar", () => {
    render(<QuotaRunwayCard result={UNLIMITED} />)
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument()
  })
})

describe("QuotaRunwayCard — within quota", () => {
  it("shows 'Quota Set' badge", () => {
    render(<QuotaRunwayCard result={WITHIN_QUOTA} />)
    expect(screen.getByText("Quota Set")).toBeInTheDocument()
  })

  it("renders usage progress bar", () => {
    render(<QuotaRunwayCard result={WITHIN_QUOTA} />)
    expect(screen.getByRole("progressbar")).toBeInTheDocument()
  })

  it("shows correct usage percentage", () => {
    render(<QuotaRunwayCard result={WITHIN_QUOTA} />)
    expect(screen.getByText("30% used this month")).toBeInTheDocument()
  })

  it("shows remaining count", () => {
    render(<QuotaRunwayCard result={WITHIN_QUOTA} />)
    expect(screen.getByText("700")).toBeInTheDocument()
  })

  it("shows days remaining in month", () => {
    render(<QuotaRunwayCard result={WITHIN_QUOTA} />)
    expect(screen.getByText("12")).toBeInTheDocument()
  })

  it("shows projected month total in the rate message", () => {
    render(<QuotaRunwayCard result={WITHIN_QUOTA} />)
    expect(screen.getByText(/420/)).toBeInTheDocument()
  })

  it("shows quota-safe message, not alert", () => {
    render(<QuotaRunwayCard result={WITHIN_QUOTA} />)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(screen.getAllByText(/Quota lasts/i).length).toBeGreaterThan(0)
  })
})

describe("QuotaRunwayCard — quota at risk", () => {
  it("shows 'Quota At Risk' badge", () => {
    render(<QuotaRunwayCard result={EXHAUSTED} />)
    expect(screen.getByText("Quota At Risk")).toBeInTheDocument()
  })

  it("shows alert role when will_exhaust is true", () => {
    render(<QuotaRunwayCard result={EXHAUSTED} />)
    expect(screen.getByRole("alert")).toBeInTheDocument()
  })

  it("shows warning text", () => {
    render(<QuotaRunwayCard result={EXHAUSTED} />)
    expect(screen.getAllByText(/Quota at risk/i).length).toBeGreaterThan(0)
  })

  it("shows rate limit RPM when set", () => {
    render(<QuotaRunwayCard result={EXHAUSTED} />)
    expect(screen.getByText(/60 req\/min/i)).toBeInTheDocument()
  })

  it("computes hourly prediction capacity from RPM", () => {
    render(<QuotaRunwayCard result={EXHAUSTED} />)
    expect(screen.getByText(/3,600/)).toBeInTheDocument()
  })
})

describe("QuotaRunwayCard — accessibility", () => {
  it("has aria-label on the figure", () => {
    render(<QuotaRunwayCard result={WITHIN_QUOTA} />)
    expect(screen.getByRole("figure", { name: /quota runway/i })).toBeInTheDocument()
  })

  it("progressbar has aria-valuenow", () => {
    render(<QuotaRunwayCard result={WITHIN_QUOTA} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-valuenow", "30")
  })

  it("has screen-reader caption", () => {
    const { container } = render(<QuotaRunwayCard result={WITHIN_QUOTA} />)
    const caption = container.querySelector("figcaption.sr-only")
    expect(caption).not.toBeNull()
  })
})
