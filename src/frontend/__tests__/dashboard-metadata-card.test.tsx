import React from "react"
import { render, screen } from "@testing-library/react"
import { DashboardMetadataCard } from "@/components/deploy/dashboard-metadata-card"
import type { DashboardMetadataResult } from "@/lib/types"

const base: DashboardMetadataResult = {
  action: "title_set",
  deployment_id: "dep-1",
  dashboard_title: "Q2 Revenue Forecast",
  dashboard_description: null,
  auto_title: "Revenue Predictor",
  summary: "Dashboard title set to 'Q2 Revenue Forecast'.",
}

describe("DashboardMetadataCard", () => {
  it("renders the card container", () => {
    render(<DashboardMetadataCard result={base} />)
    expect(screen.getByRole("region", { name: /dashboard metadata card/i })).toBeInTheDocument()
  })

  it("shows 'Dashboard Title Set' heading for title_set action", () => {
    render(<DashboardMetadataCard result={base} />)
    expect(screen.getByText("Dashboard Title Set")).toBeInTheDocument()
  })

  it("shows 'Dashboard Description Set' heading for description_set action", () => {
    render(<DashboardMetadataCard result={{ ...base, action: "description_set", dashboard_title: null, dashboard_description: "For finance team." }} />)
    expect(screen.getByText("Dashboard Description Set")).toBeInTheDocument()
  })

  it("shows 'Dashboard Branding Updated' heading for both_set action", () => {
    render(<DashboardMetadataCard result={{ ...base, action: "both_set", dashboard_description: "Finance use only." }} />)
    expect(screen.getByText("Dashboard Branding Updated")).toBeInTheDocument()
  })

  it("shows 'Dashboard Title Cleared' heading for cleared action", () => {
    render(<DashboardMetadataCard result={{ ...base, action: "cleared", dashboard_title: null }} />)
    expect(screen.getByText("Dashboard Title Cleared")).toBeInTheDocument()
  })

  it("shows 'Dashboard Title Status' heading for status action", () => {
    render(<DashboardMetadataCard result={{ ...base, action: "status" }} />)
    expect(screen.getByText("Dashboard Title Status")).toBeInTheDocument()
  })

  it("displays the custom title when set", () => {
    render(<DashboardMetadataCard result={base} />)
    expect(screen.getByTestId("dashboard-title-display")).toHaveTextContent("Q2 Revenue Forecast")
  })

  it("falls back to auto_title when dashboard_title is null", () => {
    render(<DashboardMetadataCard result={{ ...base, action: "cleared", dashboard_title: null }} />)
    expect(screen.getByTestId("dashboard-title-auto")).toHaveTextContent("Revenue Predictor")
    expect(screen.getByTestId("dashboard-title-auto")).toHaveTextContent("auto-generated")
  })

  it("displays description when set", () => {
    render(<DashboardMetadataCard result={{ ...base, action: "both_set", dashboard_description: "For finance team." }} />)
    expect(screen.getByTestId("dashboard-description-display")).toHaveTextContent("For finance team.")
  })

  it("shows 'not set' when description is null and action is status", () => {
    render(<DashboardMetadataCard result={{ ...base, action: "status", dashboard_title: null, dashboard_description: null }} />)
    expect(screen.getByText("not set")).toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<DashboardMetadataCard result={base} />)
    expect(screen.getByText("Dashboard title set to 'Q2 Revenue Forecast'.")).toBeInTheDocument()
  })

  it("shows 'Updated' badge for non-status/non-cleared actions", () => {
    render(<DashboardMetadataCard result={base} />)
    expect(screen.getByText("Updated")).toBeInTheDocument()
  })

  it("does not show 'Updated' badge for status action", () => {
    render(<DashboardMetadataCard result={{ ...base, action: "status" }} />)
    expect(screen.queryByText("Updated")).not.toBeInTheDocument()
  })

  it("does not show 'Updated' badge for cleared action", () => {
    render(<DashboardMetadataCard result={{ ...base, action: "cleared", dashboard_title: null }} />)
    expect(screen.queryByText("Updated")).not.toBeInTheDocument()
  })

  it("shows status footer hint for status action", () => {
    render(<DashboardMetadataCard result={{ ...base, action: "status" }} />)
    expect(screen.getByText(/To change, say/)).toBeInTheDocument()
  })

  it("shows update footer hint for non-status actions", () => {
    render(<DashboardMetadataCard result={base} />)
    expect(screen.getByText(/Changes are reflected immediately/)).toBeInTheDocument()
  })
})
