/**
 * Tests for CostEstimateCard — deployment cost / capacity estimate chat card.
 *
 * Covers:
 *   1. Unlimited state (no monthly quota)
 *   2. Within-quota state
 *   3. Exceeds-quota state (alert shown)
 *   4. Daily capacity display when rate limit set
 *   5. Recommended RPM display
 *   6. Accessibility (aria-label, progressbar, sr-only caption)
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { CostEstimateCard } from "../components/deploy/cost-estimate-card"
import type { CostEstimateResult } from "../lib/types"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const NO_QUOTA: CostEstimateResult = {
  deployment_id: "dep-1",
  n_predictions: 1000,
  monthly_quota: null,
  used_this_month: 50,
  quota_pct: null,
  within_quota: null,
  current_rpm: null,
  daily_capacity: null,
  avg_per_day: 5.0,
  days_needed: 200,
  recommended_rpm: 1,
}

const WITHIN_QUOTA: CostEstimateResult = {
  deployment_id: "dep-2",
  n_predictions: 500,
  monthly_quota: 2000,
  used_this_month: 200,
  quota_pct: 27.8,
  within_quota: true,
  current_rpm: 60,
  daily_capacity: 86400,
  avg_per_day: 10.0,
  days_needed: 50,
  recommended_rpm: 1,
}

const EXCEEDS_QUOTA: CostEstimateResult = {
  deployment_id: "dep-3",
  n_predictions: 5000,
  monthly_quota: 2000,
  used_this_month: 1800,
  quota_pct: 250.0,
  within_quota: false,
  current_rpm: null,
  daily_capacity: null,
  avg_per_day: 20.0,
  days_needed: 250,
  recommended_rpm: 6,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("CostEstimateCard — unlimited (no quota)", () => {
  it("shows Unlimited badge", () => {
    render(<CostEstimateCard result={NO_QUOTA} />)
    expect(screen.getByText("Unlimited")).toBeInTheDocument()
  })

  it("shows no monthly quota message", () => {
    render(<CostEstimateCard result={NO_QUOTA} />)
    expect(screen.getAllByText(/No monthly quota configured/i).length).toBeGreaterThan(0)
  })

  it("shows n_predictions in badge", () => {
    render(<CostEstimateCard result={NO_QUOTA} />)
    expect(screen.getAllByText(/1,000 predictions/).length).toBeGreaterThan(0)
  })

  it("shows recommended RPM section", () => {
    render(<CostEstimateCard result={NO_QUOTA} />)
    expect(screen.getByText(/Recommended rate limit/i)).toBeInTheDocument()
  })

  it("does not show quota exceeded alert", () => {
    render(<CostEstimateCard result={NO_QUOTA} />)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })
})

describe("CostEstimateCard — within quota", () => {
  it("shows 'Fits in quota' badge", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    expect(screen.getByText("Fits in quota")).toBeInTheDocument()
  })

  it("shows quota percentage", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    expect(screen.getAllByText(/27\.8%/).length).toBeGreaterThan(0)
  })

  it("shows monthly quota value", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    expect(screen.getByText(/2,000/)).toBeInTheDocument()
  })

  it("renders capacity progress bar", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    expect(screen.getByRole("progressbar")).toBeInTheDocument()
  })

  it("shows daily capacity when rate limit set", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    expect(screen.getByText("86,400")).toBeInTheDocument()
  })

  it("shows current RPM label", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    expect(screen.getByText(/at 60 RPM/i)).toBeInTheDocument()
  })

  it("shows days needed value", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    expect(screen.getByText("50")).toBeInTheDocument()
  })
})

describe("CostEstimateCard — exceeds quota", () => {
  it("shows 'Exceeds remaining quota' badge", () => {
    render(<CostEstimateCard result={EXCEEDS_QUOTA} />)
    expect(screen.getByText("Exceeds remaining quota")).toBeInTheDocument()
  })

  it("shows alert role when within_quota is false", () => {
    render(<CostEstimateCard result={EXCEEDS_QUOTA} />)
    expect(screen.getByRole("alert")).toBeInTheDocument()
  })

  it("shows warning text about exceeding quota", () => {
    render(<CostEstimateCard result={EXCEEDS_QUOTA} />)
    expect(screen.getByText(/exceeds your remaining quota/i)).toBeInTheDocument()
  })

  it("shows recommended RPM", () => {
    render(<CostEstimateCard result={EXCEEDS_QUOTA} />)
    expect(screen.getByText("6 RPM")).toBeInTheDocument()
  })
})

describe("CostEstimateCard — accessibility", () => {
  it("has aria-label on the figure", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    expect(
      screen.getByRole("figure", { name: /prediction capacity estimate/i })
    ).toBeInTheDocument()
  })

  it("progressbar has aria-valuenow attribute", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-valuenow")
  })

  it("has screen-reader-only caption", () => {
    const { container } = render(<CostEstimateCard result={WITHIN_QUOTA} />)
    const caption = container.querySelector("figcaption.sr-only")
    expect(caption).not.toBeNull()
  })

  it("emoji icon is aria-hidden", () => {
    const { container } = render(<CostEstimateCard result={NO_QUOTA} />)
    const emoji = container.querySelector('[aria-hidden="true"]')
    expect(emoji).not.toBeNull()
  })
})

describe("CostEstimateCard — recommended rate limit", () => {
  it("shows the recommended RPM value", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    expect(screen.getByText("1 RPM")).toBeInTheDocument()
  })

  it("shows 30-days spread explanation", () => {
    render(<CostEstimateCard result={WITHIN_QUOTA} />)
    expect(screen.getByText(/spread.*30 days/i)).toBeInTheDocument()
  })
})
