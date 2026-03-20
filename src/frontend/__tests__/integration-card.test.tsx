/**
 * Tests for IntegrationCard — developer API integration snippets component.
 *
 * Covers:
 *   1. Renders with data-testid
 *   2. Shows "Developer Integration" title
 *   3. Shows "API snippets" badge
 *   4. Shows "Show code" toggle button
 *   5. Content hidden by default (not expanded)
 *   6. Loads snippets when expanded
 *   7. Shows endpoint URL after load
 *   8. Renders curl / python / javascript tabs
 *   9. Switches active tab when clicked
 *   10. Code block updates per active tab
 *   11. Copy button is visible
 *   12. "Hide" button collapses the panel
 *   13. Shows batch note
 *   14. Shows OpenAPI docs URL
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { IntegrationCard } from "../components/deploy/integration-card"
import type { IntegrationSnippets } from "../lib/types"

// Mock clipboard API
Object.defineProperty(navigator, "clipboard", {
  value: { writeText: jest.fn().mockResolvedValue(undefined) },
  configurable: true,
})

jest.mock("../lib/api", () => ({
  api: {
    deploy: {
      getIntegration: jest.fn(),
    },
  },
}))

import { api } from "../lib/api"
const mockGetIntegration = api.deploy.getIntegration as jest.MockedFunction<
  typeof api.deploy.getIntegration
>

const MOCK_SNIPPETS: IntegrationSnippets = {
  deployment_id: "dep-abc",
  endpoint_url: "http://localhost:8000/api/predict/dep-abc",
  problem_type: "regression",
  target_column: "revenue",
  algorithm: "linear_regression",
  example_input: { region: "value", units: 1.0 },
  curl: "curl -X POST 'http://localhost:8000/api/predict/dep-abc' -H 'Content-Type: application/json' -d '{\"region\": \"value\"}'",
  python: 'import requests\nurl = "http://localhost:8000/api/predict/dep-abc"\ndata = {"region": "value"}\nresponse = requests.post(url, json=data)\nresult = response.json()\nprint(f"Prediction: {result[\'prediction\']}")',
  javascript: "const response = await fetch('http://localhost:8000/api/predict/dep-abc', {\n  method: 'POST',\n});\nconst result = await response.json();\nconsole.log('Prediction:', result.prediction);",
  openapi_url: "http://localhost:8000/docs",
  batch_url: "http://localhost:8000/api/predict/dep-abc/batch",
  batch_note: "For bulk predictions, POST a CSV file to the batch endpoint.",
}

describe("IntegrationCard", () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockGetIntegration.mockResolvedValue(MOCK_SNIPPETS)
  })

  it("renders with data-testid", () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    expect(screen.getByTestId("integration-card")).toBeInTheDocument()
  })

  it("shows Developer Integration title", () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    expect(screen.getByText("Developer Integration")).toBeInTheDocument()
  })

  it("shows API snippets badge", () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    expect(screen.getByText("API snippets")).toBeInTheDocument()
  })

  it("shows Show code toggle button", () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    expect(screen.getByTestId("integration-toggle")).toHaveTextContent("Show code")
  })

  it("content is hidden by default", () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    expect(screen.queryByTestId("integration-content")).not.toBeInTheDocument()
  })

  it("loads snippets and shows content when expanded", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => {
      expect(screen.getByTestId("integration-content")).toBeInTheDocument()
    })
    expect(mockGetIntegration).toHaveBeenCalledWith("dep-abc")
  })

  it("shows endpoint URL after load", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => {
      expect(screen.getByTestId("endpoint-url")).toHaveTextContent(
        "http://localhost:8000/api/predict/dep-abc"
      )
    })
  })

  it("renders curl tab by default", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => {
      expect(screen.getByTestId("tab-curl")).toBeInTheDocument()
    })
    expect(screen.getByTestId("code-block")).toHaveTextContent("curl")
  })

  it("renders python and javascript tabs", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => {
      expect(screen.getByTestId("tab-python")).toBeInTheDocument()
      expect(screen.getByTestId("tab-javascript")).toBeInTheDocument()
    })
  })

  it("switches to python tab when clicked", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => screen.getByTestId("tab-python"))
    fireEvent.click(screen.getByTestId("tab-python"))
    expect(screen.getByTestId("code-block")).toHaveTextContent("import requests")
  })

  it("switches to javascript tab when clicked", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => screen.getByTestId("tab-javascript"))
    fireEvent.click(screen.getByTestId("tab-javascript"))
    expect(screen.getByTestId("code-block")).toHaveTextContent("fetch")
  })

  it("copy button is visible", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => {
      expect(screen.getByTestId("copy-button")).toBeInTheDocument()
    })
  })

  it("hide button collapses the panel", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => screen.getByTestId("integration-content"))
    expect(screen.getByTestId("integration-toggle")).toHaveTextContent("Hide")
    fireEvent.click(screen.getByTestId("integration-toggle"))
    expect(screen.queryByTestId("integration-content")).not.toBeInTheDocument()
  })

  it("shows batch note", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => {
      expect(screen.getByText(/bulk predictions/i)).toBeInTheDocument()
    })
  })

  it("shows openapi docs URL", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => {
      expect(screen.getByText("http://localhost:8000/docs")).toBeInTheDocument()
    })
  })

  it("shows target column and algorithm info", async () => {
    render(<IntegrationCard deploymentId="dep-abc" />)
    fireEvent.click(screen.getByTestId("integration-toggle"))
    await waitFor(() => {
      expect(screen.getByText("revenue")).toBeInTheDocument()
      expect(screen.getByText(/linear regression/i)).toBeInTheDocument()
    })
  })
})
