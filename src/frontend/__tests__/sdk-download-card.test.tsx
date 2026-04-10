/**
 * Tests for SdkDownloadCard component.
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { SdkDownloadCard } from "@/components/deploy/sdk-download-card"
import type { SdkDownloadInfo } from "@/lib/types"

const SDK_INFO: SdkDownloadInfo = {
  deployment_id: "dep-abc",
  target_column: "revenue",
  algorithm: "linear_regression",
  problem_type: "regression",
  python_url: "/api/deploy/dep-abc/sdk?language=python",
  javascript_url: "/api/deploy/dep-abc/sdk?language=javascript",
  class_name: "RevenuePredictor",
}

const SDK_INFO_CLF: SdkDownloadInfo = {
  deployment_id: "dep-xyz",
  target_column: "churn",
  algorithm: "random_forest_classifier",
  problem_type: "classification",
  python_url: "/api/deploy/dep-xyz/sdk?language=python",
  javascript_url: "/api/deploy/dep-xyz/sdk?language=javascript",
  class_name: "ChurnPredictor",
}

describe("SdkDownloadCard", () => {
  it("renders with aria-label", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    expect(screen.getByRole("region", { hidden: true }) || screen.getByLabelText("SDK download card")).toBeTruthy()
  })

  it("renders SDK download heading", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    expect(screen.getByText("Developer SDK")).toBeInTheDocument()
  })

  it("renders Python SDK download link", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    const pythonLink = screen.getByRole("link", { name: /download python sdk/i })
    expect(pythonLink).toBeInTheDocument()
    expect(pythonLink).toHaveAttribute("href", "/api/deploy/dep-abc/sdk?language=python")
  })

  it("renders JavaScript SDK download link", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    const jsLink = screen.getByRole("link", { name: /download javascript sdk/i })
    expect(jsLink).toBeInTheDocument()
    expect(jsLink).toHaveAttribute("href", "/api/deploy/dep-abc/sdk?language=javascript")
  })

  it("download links have download attribute", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    const links = screen.getAllByRole("link")
    links.forEach((link) => expect(link).toHaveAttribute("download"))
  })

  it("shows target column name in description", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    expect(screen.getAllByText(/revenue/i).length).toBeGreaterThan(0)
  })

  it("shows class name in footer", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    expect(screen.getByText("RevenuePredictor")).toBeInTheDocument()
  })

  it("shows Regression badge for regression model", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    expect(screen.getByText("Regression")).toBeInTheDocument()
  })

  it("shows Classification badge for classification model", () => {
    render(<SdkDownloadCard info={SDK_INFO_CLF} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("shows algorithm label", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    expect(screen.getByText("Linear Regression")).toBeInTheDocument()
  })

  it("shows Python usage code preview", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    expect(screen.getByText(/Python usage/i)).toBeInTheDocument()
  })

  it("shows JavaScript usage code preview", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    expect(screen.getByText(/JavaScript usage/i)).toBeInTheDocument()
  })

  it("shows predictBatch mention in footer", () => {
    render(<SdkDownloadCard info={SDK_INFO} />)
    expect(screen.getByText(/predictBatch/)).toBeInTheDocument()
  })

  it("store action attaches sdk_download to last message", () => {
    // Verify the store action signature via type checking (runtime integration)
    const { useAppStore } = require("@/lib/store")
    const store = useAppStore.getState()
    expect(typeof store.attachSdkDownloadToLastMessage).toBe("function")
  })

  it("store action sets sdk_download on last assistant message", () => {
    const { useAppStore } = require("@/lib/store")
    useAppStore.setState({
      messages: [{ role: "assistant", content: "Hello", timestamp: "" }],
    })
    useAppStore.getState().attachSdkDownloadToLastMessage(SDK_INFO)
    const messages = useAppStore.getState().messages
    expect(messages[messages.length - 1].sdk_download).toEqual(SDK_INFO)
  })

  it("store action does not attach when last message is user", () => {
    const { useAppStore } = require("@/lib/store")
    useAppStore.setState({
      messages: [{ role: "user", content: "Hi", timestamp: "" }],
    })
    useAppStore.getState().attachSdkDownloadToLastMessage(SDK_INFO)
    const messages = useAppStore.getState().messages
    expect(messages[messages.length - 1].sdk_download).toBeUndefined()
  })
})
