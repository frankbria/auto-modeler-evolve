/**
 * Tests for ReportReadyCard component and attachReportToLastMessage store action.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { ReportReadyCard } from "@/components/models/report-ready-card"
import type { ReportReady } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Fixtures -----------------------------------------------------------

const regressionReport: ReportReady = {
  model_run_id: "run-abc",
  algorithm: "linear_regression",
  problem_type: "regression",
  metric_name: "r2",
  metric_value: 0.874,
  download_url: "/api/models/run-abc/report",
}

const classificationReport: ReportReady = {
  model_run_id: "run-xyz",
  algorithm: "random_forest_classifier",
  problem_type: "classification",
  metric_name: "accuracy",
  metric_value: 0.921,
  download_url: "/api/models/run-xyz/report",
}

const nullMetricReport: ReportReady = {
  model_run_id: "run-null",
  algorithm: "gradient_boosting",
  problem_type: "regression",
  metric_name: "r2",
  metric_value: null,
  download_url: "/api/models/run-null/report",
}

// --- Component rendering tests ------------------------------------------

describe("ReportReadyCard", () => {
  it("renders with testid", () => {
    render(<ReportReadyCard result={regressionReport} />)
    expect(screen.getByTestId("report-ready-card")).toBeInTheDocument()
  })

  it("shows PDF Report Ready header", () => {
    render(<ReportReadyCard result={regressionReport} />)
    expect(screen.getByText("PDF Report Ready")).toBeInTheDocument()
  })

  it("shows Regression badge for regression models", () => {
    render(<ReportReadyCard result={regressionReport} />)
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("shows Classification badge for classification models", () => {
    render(<ReportReadyCard result={classificationReport} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("shows human-readable algorithm name", () => {
    render(<ReportReadyCard result={regressionReport} />)
    expect(screen.getByText(/Linear Regression/)).toBeInTheDocument()
  })

  it("shows Random Forest for classifier", () => {
    render(<ReportReadyCard result={classificationReport} />)
    expect(screen.getByText(/Random Forest/)).toBeInTheDocument()
  })

  it("shows R² metric value for regression", () => {
    render(<ReportReadyCard result={regressionReport} />)
    expect(screen.getByText(/R²\s*0\.874/)).toBeInTheDocument()
  })

  it("shows Accuracy metric for classification", () => {
    render(<ReportReadyCard result={classificationReport} />)
    expect(screen.getByText(/Accuracy\s*92\.1%/)).toBeInTheDocument()
  })

  it("renders download button link", () => {
    render(<ReportReadyCard result={regressionReport} />)
    const btn = screen.getByTestId("download-report-btn")
    expect(btn).toBeInTheDocument()
    expect(btn).toHaveAttribute("target", "_blank")
  })

  it("download button href includes the download_url path", () => {
    render(<ReportReadyCard result={regressionReport} />)
    const btn = screen.getByTestId("download-report-btn")
    expect(btn.getAttribute("href")).toContain("/api/models/run-abc/report")
  })

  it("shows Download PDF Report button text", () => {
    render(<ReportReadyCard result={regressionReport} />)
    expect(screen.getByText(/Download PDF Report/)).toBeInTheDocument()
  })

  it("renders without crashing when metric_value is null", () => {
    render(<ReportReadyCard result={nullMetricReport} />)
    expect(screen.getByTestId("report-ready-card")).toBeInTheDocument()
  })

  it("shows Gradient Boosting algorithm label", () => {
    render(<ReportReadyCard result={nullMetricReport} />)
    expect(screen.getByText(/Gradient Boosting/)).toBeInTheDocument()
  })

  it("mentions metrics/feature importance in description", () => {
    render(<ReportReadyCard result={regressionReport} />)
    expect(screen.getByText(/metrics.*feature importance|feature importance.*metrics/i)).toBeInTheDocument()
  })
})

// --- Store action tests -------------------------------------------------

describe("attachReportToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "generate a report", timestamp: "t1" },
        {
          role: "assistant",
          content: "Your report is ready!",
          timestamp: "t2",
        },
      ],
    })
  })

  it("attaches report_ready to last assistant message", () => {
    const { attachReportToLastMessage } = useAppStore.getState()
    attachReportToLastMessage(regressionReport)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.report_ready).toEqual(regressionReport)
  })

  it("preserves other message fields when attaching report", () => {
    const { attachReportToLastMessage } = useAppStore.getState()
    attachReportToLastMessage(regressionReport)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.content).toBe("Your report is ready!")
    expect(last.role).toBe("assistant")
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "generate a report", timestamp: "t1" },
      ],
    })
    const { attachReportToLastMessage } = useAppStore.getState()
    attachReportToLastMessage(regressionReport)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.report_ready).toBeUndefined()
  })
})
