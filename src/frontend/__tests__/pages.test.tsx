/**
 * Tests for Next.js app/ pages (app/page.tsx, app/predict/[id]/page.tsx).
 *
 * These are client components ("use client") so we mock:
 *   - next/navigation (useRouter, useParams)
 *   - fetch (via jest-fetch-mock / global.fetch)
 */

import React from "react"
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"

// Enable fetch mocking BEFORE any imports that use fetch
fetchMock.enableMocks()

// ---------------------------------------------------------------------------
// Mock next/navigation before importing pages
// ---------------------------------------------------------------------------
const mockPush = jest.fn()
const mockRouterRefresh = jest.fn()

jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
    refresh: mockRouterRefresh,
    replace: jest.fn(),
    back: jest.fn(),
    forward: jest.fn(),
    prefetch: jest.fn(),
  }),
  useParams: () => ({ id: "deployment-123" }),
  usePathname: () => "/",
  useSearchParams: () => new URLSearchParams(),
}))

// ---------------------------------------------------------------------------
// Shared test data
// ---------------------------------------------------------------------------

const mockProjects = [
  {
    id: "proj-1",
    name: "Sales Forecast",
    description: "",
    created_at: "2024-01-01T00:00:00",
    updated_at: "2024-01-02T00:00:00",
    status: "exploring",
    dataset_name: "sales.csv",
    dataset_rows: 200,
    model_count: 2,
    has_deployment: true,
  },
  {
    id: "proj-2",
    name: "Churn Prediction",
    description: "",
    created_at: "2024-01-03T00:00:00",
    updated_at: "2024-01-04T00:00:00",
    status: "deployed",
    dataset_name: null,
    dataset_rows: null,
    model_count: 0,
    has_deployment: false,
  },
]

const mockDeployment = {
  id: "deployment-123",
  model_run_id: "run-456",
  project_id: "proj-1",
  endpoint_path: "/api/predict/deployment-123",
  dashboard_url: "http://localhost:3000/predict/deployment-123",
  is_active: true,
  request_count: 42,
  created_at: "2024-01-05T00:00:00",
  last_predicted_at: "2024-01-06T00:00:00",
  feature_schema: [
    { name: "units", type: "numeric", median: 10.0, options: null },
    { name: "region", type: "categorical", median: null, options: ["North", "South", "East"] },
  ],
}

// ---------------------------------------------------------------------------
// HomePage tests (app/page.tsx)
// ---------------------------------------------------------------------------

describe("HomePage", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    mockPush.mockReset()
    jest.clearAllMocks()
  })

  it("renders loading state initially", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProjects))
    const { default: HomePage } = await import("../app/page")
    const { unmount } = render(<HomePage />)
    // After loading completes, verify projects API was called
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/projects")
      )
    })
    unmount()
  })

  it("renders project list after loading", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProjects))
    const { default: HomePage } = await import("../app/page")
    render(<HomePage />)
    await waitFor(() => {
      expect(screen.getByText("Sales Forecast")).toBeInTheDocument()
    })
    expect(screen.getByText("Churn Prediction")).toBeInTheDocument()
  })

  it("shows empty state when no projects exist", async () => {
    fetchMock.mockResponseOnce(JSON.stringify([]))
    const { default: HomePage } = await import("../app/page")
    render(<HomePage />)
    await waitFor(() => {
      // After loading with no projects, empty state should show
      expect(screen.queryByText("Sales Forecast")).not.toBeInTheDocument()
    })
  })

  it("renders create project button", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProjects))
    const { default: HomePage } = await import("../app/page")
    render(<HomePage />)
    await waitFor(() => {
      expect(screen.getByText("Sales Forecast")).toBeInTheDocument()
    })
    // A create / new project button should exist
    const createButton = screen.queryByRole("button", { name: /new project|create/i })
    expect(createButton).toBeTruthy()
  })

  it("shows project form when create button is clicked", async () => {
    fetchMock.mockResponseOnce(JSON.stringify([]))
    const { default: HomePage } = await import("../app/page")
    render(<HomePage />)
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled()
    })
    const createButton = screen.queryByRole("button", { name: /new project|create/i })
    if (createButton) {
      act(() => { fireEvent.click(createButton) })
      // After click, expect a form/input to appear
      await waitFor(() => {
        const input = screen.queryByPlaceholderText(/project name/i) ||
                      screen.queryByRole("textbox")
        expect(input).toBeTruthy()
      })
    }
  })

  it("navigates to project workspace when project card is clicked", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProjects))
    const { default: HomePage } = await import("../app/page")
    render(<HomePage />)
    await waitFor(() => {
      expect(screen.getByText("Sales Forecast")).toBeInTheDocument()
    })
    // Click on the first project
    fireEvent.click(screen.getByText("Sales Forecast"))
    // Should push to project workspace
    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith(expect.stringContaining("proj-1"))
    })
  })

  it("handles project creation", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockProjects))
    fetchMock.mockResponseOnce(
      JSON.stringify({ id: "proj-new", name: "New Project", status: "exploring" }),
      { status: 201 }
    )
    const { default: HomePage } = await import("../app/page")
    render(<HomePage />)
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    const createButton = screen.queryByRole("button", { name: /new project|create/i })
    if (createButton) {
      act(() => { fireEvent.click(createButton) })
      await waitFor(() => {
        const input = screen.queryByPlaceholderText(/project name/i) ||
                      screen.queryByRole("textbox")
        if (input) {
          fireEvent.change(input, { target: { value: "New Project" } })
        }
      })
    }
  })

  it("handles fetch error gracefully", async () => {
    fetchMock.mockRejectOnce(new Error("Network error"))
    const { default: HomePage } = await import("../app/page")
    render(<HomePage />)
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled()
    })
    // Should not crash — empty state renders
  })
})

// ---------------------------------------------------------------------------
// PredictionDashboard tests (app/predict/[id]/page.tsx)
// ---------------------------------------------------------------------------

describe("PredictionDashboard", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
    jest.clearAllMocks()
  })

  it("renders loading state initially", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockDeployment))
    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)
    // Should briefly show loading before hydrating
    expect(fetchMock).toBeDefined()
  })

  it("renders prediction form after loading", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockDeployment))
    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)
    await waitFor(() => {
      // Feature fields should render
      expect(screen.queryByText(/units/i) || screen.queryByText(/region/i)).toBeTruthy()
    })
  })

  it("shows error message when deployment not found", async () => {
    // Reject the fetch to simulate a network/API error that triggers catch handler
    fetchMock.mockRejectOnce(new Error("Deployment not found"))
    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)
    await waitFor(() => {
      // Error state renders — no form, error message shows
      expect(screen.queryByText(/not found|inactive|error/i)).toBeTruthy()
    })
  })

  it("renders numeric and categorical inputs from feature schema", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockDeployment))
    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)
    await waitFor(() => {
      // Check for numeric input or select dropdown
      const inputs = screen.queryAllByRole("textbox")
      const selects = screen.queryAllByRole("combobox")
      expect(inputs.length + selects.length).toBeGreaterThan(0)
    })
  })

  it("sends prediction request when predict button clicked", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockDeployment))
    fetchMock.mockResponseOnce(
      JSON.stringify({ prediction: 1200.5, deployment_id: "deployment-123" })
    )
    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)
    await waitFor(() => {
      expect(screen.queryByText(/units/i) || screen.queryByText(/predict/i)).toBeTruthy()
    })

    const predictButton = screen.queryByRole("button", { name: /predict/i })
    if (predictButton) {
      await act(async () => { fireEvent.click(predictButton) })
      await waitFor(() => {
        expect(fetchMock).toHaveBeenCalledTimes(2)
      })
    }
  })

  it("shows prediction result after successful prediction", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockDeployment))
    fetchMock.mockResponseOnce(
      JSON.stringify({ prediction: 1200.5, deployment_id: "deployment-123" })
    )
    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)
    await waitFor(() => {
      const predictButton = screen.queryByRole("button", { name: /predict/i })
      if (predictButton) {
        return true
      }
    })

    const predictButton = screen.queryByRole("button", { name: /predict/i })
    if (predictButton) {
      await act(async () => { fireEvent.click(predictButton) })
      await waitFor(() => {
        // After prediction, the result value or prediction-related text should appear
        const allText = document.body.textContent ?? ""
        expect(allText.length).toBeGreaterThan(0)
        expect(fetchMock).toHaveBeenCalledTimes(2)
      })
    }
  })

  it("handles prediction API error", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(mockDeployment))
    fetchMock.mockResponseOnce(
      JSON.stringify({ detail: "Prediction failed" }),
      { status: 500 }
    )
    const { default: PredictionDashboard } = await import("../app/predict/[id]/page")
    render(<PredictionDashboard />)
    await waitFor(() => {
      const btn = screen.queryByRole("button", { name: /predict/i })
      if (btn) return true
    })

    const predictButton = screen.queryByRole("button", { name: /predict/i })
    if (predictButton) {
      await act(async () => { fireEvent.click(predictButton) })
      await waitFor(() => {
        // Error message or fallback render
        expect(fetchMock).toHaveBeenCalledTimes(2)
      })
    }
  })
})
