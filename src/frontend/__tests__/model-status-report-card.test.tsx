import React from "react"
import { render, screen } from "@testing-library/react"
import { ModelStatusReportCard } from "@/components/deploy/model-status-report-card"
import { useAppStore } from "@/lib/store"
import type { ModelStatusReportInfo } from "@/lib/types"

const baseInfo: ModelStatusReportInfo = {
  deployment_id: "dep-123",
  project_name: "Sales Forecast",
  algorithm_plain: "Random Forest",
  problem_type: "regression",
  target_column: "revenue",
  health_score: 82,
  health_status: "healthy",
  predictions_today: 15,
  predictions_7d: 104,
  predictions_30d: 450,
  predictions_total: 1200,
  sla_ok: true,
  p95_ms: 95,
  download_url: "/api/deploy/dep-123/status-report",
  summary: "Model health: healthy (82/100). 104 predictions in the last 7 days. SLA: p95 95 ms — within target.",
}

describe("ModelStatusReportCard", () => {
  it("renders the card with correct label", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    expect(screen.getByText("Model Status Report")).toBeInTheDocument()
  })

  it("shows the algorithm", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    expect(screen.getByText("Random Forest")).toBeInTheDocument()
  })

  it("shows health score", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    const el = screen.getByTestId("health-score")
    expect(el).toHaveTextContent("82/100")
    expect(el).toHaveTextContent("healthy")
  })

  it("shows predictions today", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    const el = screen.getByTestId("predictions-today")
    expect(el).toHaveTextContent("15")
  })

  it("shows predictions 7 days", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    const el = screen.getByTestId("predictions-7d")
    expect(el).toHaveTextContent("104")
  })

  it("shows predictions 30 days", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    const el = screen.getByTestId("predictions-30d")
    expect(el).toHaveTextContent("450")
  })

  it("shows total predictions", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    const el = screen.getByTestId("predictions-total")
    expect(el).toHaveTextContent("1,200")
  })

  it("shows within SLA badge when sla_ok is true", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    expect(screen.getByTestId("sla-badge")).toHaveTextContent("Within SLA")
  })

  it("shows SLA alert badge when sla_ok is false", () => {
    render(<ModelStatusReportCard info={{ ...baseInfo, sla_ok: false, p95_ms: 750 }} />)
    expect(screen.getByTestId("sla-badge")).toHaveTextContent("SLA Alert")
  })

  it("shows p95 latency when present", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    const el = screen.getByTestId("sla-latency")
    expect(el).toHaveTextContent("95 ms")
  })

  it("hides latency row when p95_ms is null", () => {
    render(<ModelStatusReportCard info={{ ...baseInfo, p95_ms: null }} />)
    expect(screen.queryByTestId("sla-latency")).not.toBeInTheDocument()
  })

  it("shows the summary text", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    const el = screen.getByTestId("report-summary")
    expect(el).toHaveTextContent("104 predictions")
  })

  it("has a download link pointing to backend", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    const link = screen.getByTestId("download-report-link")
    expect(link).toHaveAttribute("href", expect.stringContaining("/api/deploy/dep-123/status-report"))
    expect(link).toHaveAttribute("download")
  })

  it("has accessible progressbar for health score", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-valuenow", "82")
    expect(bar).toHaveAttribute("aria-valuemin", "0")
    expect(bar).toHaveAttribute("aria-valuemax", "100")
  })

  it("has sr-only figcaption with accessible description", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    const figcaption = screen.getByText(/Model status report for Sales Forecast/)
    expect(figcaption).toHaveClass("sr-only")
  })

  it("uses rose border for critical health", () => {
    const { container } = render(
      <ModelStatusReportCard info={{ ...baseInfo, health_score: 25, health_status: "critical" }} />
    )
    const fig = container.querySelector("figure")
    expect(fig?.className).toContain("rose")
  })

  it("uses amber border for warning health", () => {
    const { container } = render(
      <ModelStatusReportCard info={{ ...baseInfo, health_score: 60, health_status: "warning" }} />
    )
    const fig = container.querySelector("figure")
    expect(fig?.className).toContain("amber")
  })

  it("uses emerald border for healthy", () => {
    const { container } = render(
      <ModelStatusReportCard info={{ ...baseInfo, health_score: 82, health_status: "healthy" }} />
    )
    const fig = container.querySelector("figure")
    expect(fig?.className).toContain("emerald")
  })

  it("has data-testid on the card root", () => {
    render(<ModelStatusReportCard info={baseInfo} />)
    expect(screen.getByTestId("model-status-report-card")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Zustand store action test
// ---------------------------------------------------------------------------
describe("attachModelStatusReportToLastMessage store action", () => {
  it("attaches model_status_report to the last assistant message", () => {
    const store = useAppStore.getState()

    // Add a mock assistant message
    store.addMessage({ role: "assistant", content: "Here is your report." })
    store.attachModelStatusReportToLastMessage(baseInfo)

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const messages = (useAppStore.getState() as any).messages
    const last = messages[messages.length - 1]
    expect(last.model_status_report).toBeDefined()
    expect(last.model_status_report?.health_score).toBe(82)
    expect(last.model_status_report?.predictions_7d).toBe(104)
  })
})
