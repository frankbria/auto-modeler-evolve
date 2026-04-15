/**
 * Tests for ServiceExportChatCard and Zustand attachServiceExportToLastMessage action.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import "@testing-library/jest-dom"
import { ServiceExportChatCard } from "../components/deploy/service-export-chat-card"
import { useAppStore } from "../lib/store"
import type { ServiceExportChatResult } from "../lib/types"

const fullResult: ServiceExportChatResult = {
  deployment_id: "dep-001",
  algorithm: "random_forest_regressor",
  target_column: "revenue",
  problem_type: "regression",
  feature_count: 5,
  download_url: "/api/deploy/dep-001/export",
  included_files: [
    "server.py",
    "model_pipeline.joblib",
    "model.joblib",
    "requirements.txt",
    "README.md",
  ],
}

const classificationResult: ServiceExportChatResult = {
  deployment_id: "dep-002",
  algorithm: "logistic_regression",
  target_column: "churn",
  problem_type: "classification",
  feature_count: 3,
  download_url: "/api/deploy/dep-002/export",
  included_files: [
    "server.py",
    "model_pipeline.joblib",
    "model.joblib",
    "requirements.txt",
    "README.md",
  ],
}

describe("ServiceExportChatCard", () => {
  it("renders the card region with aria-label", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    expect(
      screen.getByRole("region", { name: /model service export/i })
    ).toBeInTheDocument()
  })

  it("renders the package emoji (aria-hidden)", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    const icon = screen.getByText("📦")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("renders 'Model Package Ready' heading", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    expect(screen.getByText("Model Package Ready")).toBeInTheDocument()
  })

  it("renders ZIP download badge", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    expect(screen.getByText("ZIP download")).toBeInTheDocument()
  })

  it("renders problem type badge", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    expect(screen.getByText("regression")).toBeInTheDocument()
  })

  it("renders algorithm name in description", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    // Algorithm gets formatted: random_forest_regressor → Random Forest Regressor
    // It appears in the description span
    const spans = screen.getAllByText(/Random Forest Regressor/i)
    expect(spans.length).toBeGreaterThan(0)
  })

  it("renders target column in description", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("renders included files list", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    const filesList = screen.getByTestId("included-files")
    expect(filesList).toBeInTheDocument()
    expect(filesList.textContent).toContain("server.py")
    expect(filesList.textContent).toContain("model.joblib")
    expect(filesList.textContent).toContain("requirements.txt")
    expect(filesList.textContent).toContain("README.md")
  })

  it("renders quickstart code block", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    const block = screen.getByTestId("quickstart-block")
    expect(block.textContent).toContain("uvicorn server:app")
    expect(block.textContent).toContain("pip install -r requirements.txt")
  })

  it("renders feature count", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    expect(screen.getByText(/5/)).toBeInTheDocument()
    expect(screen.getByText(/features included/i)).toBeInTheDocument()
  })

  it("renders download link with correct href", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    const link = screen.getByTestId("service-export-download-link")
    expect(link).toHaveAttribute("href", "/api/deploy/dep-001/export")
  })

  it("renders download link with aria-label", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    const link = screen.getByTestId("service-export-download-link")
    const label = link.getAttribute("aria-label") ?? ""
    expect(label.toLowerCase()).toContain("download")
    expect(label.toLowerCase()).toContain("model service as zip")
  })

  it("renders download link with download attribute", () => {
    render(<ServiceExportChatCard result={fullResult} />)
    const link = screen.getByTestId("service-export-download-link")
    expect(link).toHaveAttribute("download")
  })

  it("renders classification result correctly", () => {
    render(<ServiceExportChatCard result={classificationResult} />)
    expect(screen.getByText("classification")).toBeInTheDocument()
    expect(screen.getByText("churn")).toBeInTheDocument()
    const spans = screen.getAllByText(/Logistic Regression/i)
    expect(spans.length).toBeGreaterThan(0)
  })

  it("singular 'feature' when feature_count is 1", () => {
    const singleFeature: ServiceExportChatResult = {
      ...fullResult,
      feature_count: 1,
    }
    render(<ServiceExportChatCard result={singleFeature} />)
    expect(screen.getByText(/1/)).toBeInTheDocument()
    // Should say "feature included" not "features included"
    expect(screen.getByText(/feature included/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Zustand store action
// ---------------------------------------------------------------------------

describe("attachServiceExportToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [],
    })
  })

  it("attaches service_export to last assistant message", () => {
    const { addMessage, attachServiceExportToLastMessage } =
      useAppStore.getState()
    addMessage({ role: "user", content: "package my model" })
    addMessage({ role: "assistant", content: "Your model is packaged." })
    attachServiceExportToLastMessage(fullResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].service_export).toEqual(fullResult)
  })

  it("does NOT attach to a user message", () => {
    const { addMessage, attachServiceExportToLastMessage } =
      useAppStore.getState()
    addMessage({ role: "user", content: "package my model" })
    attachServiceExportToLastMessage(fullResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].service_export).toBeUndefined()
  })

  it("does not crash on empty message list", () => {
    const { attachServiceExportToLastMessage } = useAppStore.getState()
    expect(() => attachServiceExportToLastMessage(fullResult)).not.toThrow()
  })
})
