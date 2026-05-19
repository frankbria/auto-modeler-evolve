import React from "react"
import { render, screen } from "@testing-library/react"
import { DashboardConfigCard } from "@/components/deploy/dashboard-config-card"
import type { DashboardConfigResult } from "@/lib/types"

const base: DashboardConfigResult = {
  action: "updated",
  visible_count: 3,
  locked_count: 1,
  total_count: 4,
  changes: [
    { feature_name: "units", is_visible: true, is_locked: false, locked_value: null },
    { feature_name: "region", is_visible: false, is_locked: false, locked_value: null },
    { feature_name: "price", is_visible: true, is_locked: true, locked_value: "99.9" },
    { feature_name: "channel", is_visible: true, is_locked: false, locked_value: null },
  ],
  summary: "Updated 4 fields.",
}

describe("DashboardConfigCard", () => {
  it("renders card with testid", () => {
    render(<DashboardConfigCard config={base} />)
    expect(screen.getByTestId("dashboard-config-card")).toBeInTheDocument()
  })

  it("shows updated heading for action=updated", () => {
    render(<DashboardConfigCard config={base} />)
    expect(screen.getByText("Dashboard Configured")).toBeInTheDocument()
  })

  it("shows reset heading for action=reset", () => {
    render(<DashboardConfigCard config={{ ...base, action: "reset" }} />)
    expect(screen.getByText("Dashboard Reset")).toBeInTheDocument()
  })

  it("shows status heading for action=status", () => {
    render(<DashboardConfigCard config={{ ...base, action: "status" }} />)
    expect(screen.getByText("Dashboard Config")).toBeInTheDocument()
  })

  it("shows visible/total badge", () => {
    render(<DashboardConfigCard config={base} />)
    expect(screen.getByText("3/4 visible")).toBeInTheDocument()
  })

  it("shows locked count badge when locked_count > 0", () => {
    render(<DashboardConfigCard config={base} />)
    expect(screen.getByText("1 locked")).toBeInTheDocument()
  })

  it("does not show locked badge when locked_count is 0", () => {
    render(<DashboardConfigCard config={{ ...base, locked_count: 0 }} />)
    expect(screen.queryByText(/locked/)).not.toBeInTheDocument()
  })

  it("renders a row for each change", () => {
    render(<DashboardConfigCard config={base} />)
    expect(screen.getByTestId("field-row-units")).toBeInTheDocument()
    expect(screen.getByTestId("field-row-region")).toBeInTheDocument()
    expect(screen.getByTestId("field-row-price")).toBeInTheDocument()
    expect(screen.getByTestId("field-row-channel")).toBeInTheDocument()
  })

  it("shows Hidden badge for non-visible fields", () => {
    render(<DashboardConfigCard config={base} />)
    const regionRow = screen.getByTestId("field-row-region")
    expect(regionRow).toHaveTextContent("Hidden")
  })

  it("shows Locked badge with value for locked fields", () => {
    render(<DashboardConfigCard config={base} />)
    const priceRow = screen.getByTestId("field-row-price")
    expect(priceRow).toHaveTextContent("Locked = 99.9")
  })

  it("shows Visible badge for visible unlocked fields", () => {
    render(<DashboardConfigCard config={base} />)
    const unitsRow = screen.getByTestId("field-row-units")
    expect(unitsRow).toHaveTextContent("Visible")
  })

  it("shows summary text", () => {
    render(<DashboardConfigCard config={base} />)
    expect(screen.getByText("Updated 4 fields.")).toBeInTheDocument()
  })

  it("renders no change rows when changes is empty", () => {
    render(<DashboardConfigCard config={{ ...base, changes: [] }} />)
    // Field rows should not exist
    expect(screen.queryByTestId(/field-row-/)).not.toBeInTheDocument()
  })

  it("shows reset footer text for action=reset", () => {
    render(<DashboardConfigCard config={{ ...base, action: "reset" }} />)
    expect(screen.getByText(/All fields are now visible/)).toBeInTheDocument()
  })

  it("shows status footer text for action=status", () => {
    render(<DashboardConfigCard config={{ ...base, action: "status" }} />)
    expect(screen.getByText(/Say 'hide X/)).toBeInTheDocument()
  })

  it("shows updated footer text for action=updated", () => {
    render(<DashboardConfigCard config={base} />)
    expect(screen.getByText(/Changes are reflected immediately/)).toBeInTheDocument()
  })

  it("shows labeled heading for action=labeled", () => {
    render(<DashboardConfigCard config={{ ...base, action: "labeled" }} />)
    expect(screen.getByText("Field Labeled")).toBeInTheDocument()
  })

  it("shows labeled footer text for action=labeled", () => {
    render(<DashboardConfigCard config={{ ...base, action: "labeled" }} />)
    expect(screen.getByText(/new label is shown on the shared prediction URL/)).toBeInTheDocument()
  })

  it("shows labeled_count badge when labeled_count > 0", () => {
    render(<DashboardConfigCard config={{ ...base, labeled_count: 2 }} />)
    expect(screen.getByText("2 labeled")).toBeInTheDocument()
  })

  it("does not show labeled badge when labeled_count is 0", () => {
    render(<DashboardConfigCard config={{ ...base, labeled_count: 0 }} />)
    expect(screen.queryByText(/labeled/)).not.toBeInTheDocument()
  })

  it("shows display_label badge in field row", () => {
    const withLabel: DashboardConfigResult = {
      ...base,
      action: "labeled",
      labeled_count: 1,
      changes: [
        { feature_name: "units", is_visible: true, is_locked: false, locked_value: null, display_label: "Monthly Units Sold" },
      ],
    }
    render(<DashboardConfigCard config={withLabel} />)
    expect(screen.getByTestId("field-row-units")).toHaveTextContent('→ "Monthly Units Sold"')
  })

  it("suppresses Visible badge when display_label is set", () => {
    const withLabel: DashboardConfigResult = {
      ...base,
      action: "labeled",
      labeled_count: 1,
      changes: [
        { feature_name: "units", is_visible: true, is_locked: false, locked_value: null, display_label: "Monthly Units Sold" },
      ],
    }
    render(<DashboardConfigCard config={withLabel} />)
    const row = screen.getByTestId("field-row-units")
    expect(row).not.toHaveTextContent("Visible")
  })
})
