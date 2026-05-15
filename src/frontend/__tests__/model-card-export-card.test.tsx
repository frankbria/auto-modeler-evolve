import { render, screen } from "@testing-library/react"
import { ModelCardExportCard } from "@/components/chat/model-card-export-card"
import type { ModelCardExportInfo } from "@/lib/types"

const baseInfo: ModelCardExportInfo = {
  run_id: "run-123",
  project_name: "Sales Forecast",
  algorithm: "random_forest",
  algorithm_plain: "Random Forest",
  problem_type: "classification",
  target_column: "churned",
  metric_name: "f1",
  metric_value: 0.84,
  metric_display: "0.84",
  feature_count: 5,
  row_count: 1000,
  trained_at: "2026-05-15T12:00:00",
  download_url: "/api/models/run-123/export-model-card",
  summary: "A well-performing Random Forest classifier.",
}

describe("ModelCardExportCard — rendering", () => {
  it("renders the card heading", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    expect(screen.getByText("Model Card Ready")).toBeInTheDocument()
  })

  it("shows algorithm badge", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    expect(screen.getByText("Random Forest")).toBeInTheDocument()
  })

  it("shows problem type badge", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    expect(screen.getByText("classification")).toBeInTheDocument()
  })

  it("shows target column badge", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    expect(screen.getByText(/target:.*churned/i)).toBeInTheDocument()
  })

  it("shows metric name and value", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    expect(screen.getByText(/F1/i)).toBeInTheDocument()
    expect(screen.getByText(/0\.84/)).toBeInTheDocument()
  })

  it("shows feature count", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    expect(screen.getByText("Features:")).toBeInTheDocument()
    expect(screen.getByText("5")).toBeInTheDocument()
  })

  it("shows row count formatted with commas", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    expect(screen.getByText("Rows:")).toBeInTheDocument()
    expect(screen.getByText("1,000")).toBeInTheDocument()
  })

  it("shows trained date", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    expect(screen.getByText(/Trained/i)).toBeInTheDocument()
  })

  it("renders a download link", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    const link = screen.getByRole("link", { name: /download/i })
    expect(link).toBeInTheDocument()
  })

  it("download link contains the export URL", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    const link = screen.getByRole("link", { name: /download/i })
    expect(link.getAttribute("href")).toContain("/api/models/run-123/export-model-card")
  })

  it("has download attribute on the link", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    const link = screen.getByRole("link", { name: /download/i })
    expect(link).toHaveAttribute("download")
  })

  it("has accessible figure role with aria-label", () => {
    render(<ModelCardExportCard info={baseInfo} />)
    expect(
      screen.getByRole("figure", { name: /model card export/i })
    ).toBeInTheDocument()
  })

  it("renders without trained_at crashing", () => {
    render(<ModelCardExportCard info={{ ...baseInfo, trained_at: "" }} />)
    expect(screen.getByText("Model Card Ready")).toBeInTheDocument()
  })
})
