/**
 * Tests for ExportServiceCard — self-contained prediction service download.
 *
 * Covers:
 *   1. Renders with data-testid
 *   2. Shows "Export as Service" title
 *   3. Shows "ZIP download" badge
 *   4. Shows description text
 *   5. Lists the 5 included files
 *   6. Shows quick-start code snippet
 *   7. Shows target column and algorithm when provided
 *   8. Download button is present and enabled
 *   9. Clicking download triggers fetch and blob download
 *   10. Shows "Preparing download…" while downloading
 *   11. Error shown when fetch fails
 *   12. Renders without algorithm/target (optional props)
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { ExportServiceCard } from "../components/deploy/export-service-card"

// Mock URL.createObjectURL and URL.revokeObjectURL
const mockCreateObjectURL = jest.fn().mockReturnValue("blob:fake-url")
const mockRevokeObjectURL = jest.fn()
Object.defineProperty(URL, "createObjectURL", { value: mockCreateObjectURL, configurable: true })
Object.defineProperty(URL, "revokeObjectURL", { value: mockRevokeObjectURL, configurable: true })

// Mock document.createElement to intercept link click
const originalCreateElement = document.createElement.bind(document)
let mockAnchor: HTMLAnchorElement | null = null
jest.spyOn(document, "createElement").mockImplementation((tag: string) => {
  if (tag === "a") {
    mockAnchor = originalCreateElement("a") as HTMLAnchorElement
    jest.spyOn(mockAnchor, "click").mockImplementation(() => {})
    return mockAnchor
  }
  return originalCreateElement(tag)
})

const mockBlob = new Blob(["fake zip content"], { type: "application/zip" })

function mockFetchSuccess() {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    blob: () => Promise.resolve(mockBlob),
    headers: {
      get: (name: string) => {
        if (name === "content-disposition")
          return 'attachment; filename="automodeler_revenue_linear_regression.zip"'
        return null
      },
    },
  } as unknown as Response)
}

function mockFetchError() {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status: 500,
    blob: () => Promise.reject(new Error("server error")),
    headers: { get: () => null },
  } as unknown as Response)
}

beforeEach(() => {
  jest.clearAllMocks()
  mockCreateObjectURL.mockReturnValue("blob:fake-url")
})

describe("ExportServiceCard", () => {
  it("renders with data-testid", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    expect(screen.getByTestId("export-service-card")).toBeInTheDocument()
  })

  it("shows Export as Service title", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    expect(screen.getByText(/Export as Service/i)).toBeInTheDocument()
  })

  it("shows ZIP download badge", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    expect(screen.getByText("ZIP download")).toBeInTheDocument()
  })

  it("shows description text about FastAPI service", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    expect(screen.getByText(/standalone FastAPI service/i)).toBeInTheDocument()
  })

  it("lists server.py in included files", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    expect(screen.getByText("server.py")).toBeInTheDocument()
  })

  it("lists model_pipeline.joblib in included files", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    expect(screen.getByText("model_pipeline.joblib")).toBeInTheDocument()
  })

  it("lists model.joblib in included files", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    expect(screen.getByText("model.joblib")).toBeInTheDocument()
  })

  it("lists requirements.txt in included files", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    expect(screen.getByText("requirements.txt")).toBeInTheDocument()
  })

  it("lists README.md in included files", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    expect(screen.getByText("README.md")).toBeInTheDocument()
  })

  it("shows quick-start uvicorn command", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    expect(screen.getByText(/uvicorn server:app/i)).toBeInTheDocument()
  })

  it("shows target column when provided", () => {
    render(
      <ExportServiceCard deploymentId="dep-1" targetColumn="revenue" algorithm="linear_regression" />
    )
    expect(screen.getByTestId("export-target-column")).toHaveTextContent("revenue")
  })

  it("shows algorithm when provided", () => {
    render(
      <ExportServiceCard deploymentId="dep-1" targetColumn="revenue" algorithm="linear_regression" />
    )
    expect(screen.getAllByText(/linear regression/i).length).toBeGreaterThan(0)
  })

  it("download button is present and enabled by default", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    const btn = screen.getByTestId("export-download-button")
    expect(btn).toBeInTheDocument()
    expect(btn).not.toBeDisabled()
  })

  it("shows 'Preparing download…' while fetch is in progress", async () => {
    // Never-resolving fetch to keep loading state visible
    global.fetch = jest.fn().mockReturnValue(new Promise(() => {}))
    render(<ExportServiceCard deploymentId="dep-1" />)
    fireEvent.click(screen.getByTestId("export-download-button"))
    await waitFor(() => {
      expect(screen.getByTestId("export-download-button")).toHaveTextContent(
        "Preparing download…"
      )
    })
  })

  it("initiates blob download on successful fetch", async () => {
    mockFetchSuccess()
    render(<ExportServiceCard deploymentId="dep-1" />)
    fireEvent.click(screen.getByTestId("export-download-button"))
    await waitFor(() => {
      expect(mockCreateObjectURL).toHaveBeenCalled()
      expect(mockAnchor?.click).toHaveBeenCalled()
    })
  })

  it("shows error message when fetch fails", async () => {
    mockFetchError()
    render(<ExportServiceCard deploymentId="dep-1" />)
    fireEvent.click(screen.getByTestId("export-download-button"))
    await waitFor(() => {
      expect(screen.getByTestId("export-error")).toHaveTextContent(
        "Export failed. Please try again."
      )
    })
  })

  it("renders without algorithm and targetColumn (optional props)", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    // Should not crash and still show the card
    expect(screen.getByTestId("export-service-card")).toBeInTheDocument()
  })

  it("has accessible aria-label on download button", () => {
    render(<ExportServiceCard deploymentId="dep-1" />)
    const btn = screen.getByTestId("export-download-button")
    expect(btn).toHaveAttribute("aria-label")
  })
})
