import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { AutoRetrainCard } from "@/components/models/auto-retrain-card"
import type { AutoRetrainResult } from "@/lib/types"

// Mock fetch globally
global.fetch = jest.fn()

const enabledResult: AutoRetrainResult = {
  project_id: "proj-1",
  enabled: true,
  selected_algorithm: "random_forest_regressor",
  has_selected_model: true,
}

const disabledResult: AutoRetrainResult = {
  project_id: "proj-1",
  enabled: false,
  selected_algorithm: null,
  has_selected_model: false,
}

const enabledNoModel: AutoRetrainResult = {
  project_id: "proj-1",
  enabled: true,
  selected_algorithm: null,
  has_selected_model: false,
}

beforeEach(() => {
  jest.clearAllMocks()
})

describe("AutoRetrainCard — enabled state", () => {
  it("shows Enabled badge", () => {
    render(<AutoRetrainCard result={enabledResult} />)
    expect(screen.getByText("Enabled")).toBeInTheDocument()
  })

  it("shows selected algorithm", () => {
    render(<AutoRetrainCard result={enabledResult} />)
    expect(screen.getByText(/random forest regressor/i)).toBeInTheDocument()
  })

  it("shows Disable button", () => {
    render(<AutoRetrainCard result={enabledResult} />)
    expect(
      screen.getByRole("button", { name: /disable auto-retrain/i })
    ).toBeInTheDocument()
  })
})

describe("AutoRetrainCard — disabled state", () => {
  it("shows Disabled badge", () => {
    render(<AutoRetrainCard result={disabledResult} />)
    expect(screen.getByText("Disabled")).toBeInTheDocument()
  })

  it("shows Enable button", () => {
    render(<AutoRetrainCard result={disabledResult} />)
    expect(
      screen.getByRole("button", { name: /enable auto-retrain/i })
    ).toBeInTheDocument()
  })

  it("shows description of feature", () => {
    render(<AutoRetrainCard result={disabledResult} />)
    expect(screen.getByText(/background retrain/i)).toBeInTheDocument()
  })
})

describe("AutoRetrainCard — enabled without selected model", () => {
  it("shows warning when no model selected", () => {
    render(<AutoRetrainCard result={enabledNoModel} />)
    expect(screen.getByText(/no selected model/i)).toBeInTheDocument()
  })
})

describe("AutoRetrainCard — toggle interaction", () => {
  it("calls fetch PUT when toggle button clicked", async () => {
    ;(global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true })

    render(<AutoRetrainCard result={disabledResult} />)
    fireEvent.click(screen.getByRole("button", { name: /enable auto-retrain/i }))

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "/api/projects/proj-1/auto-retrain",
        expect.objectContaining({ method: "PUT" })
      )
    })
  })

  it("calls onToggle callback after successful toggle", async () => {
    ;(global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true })
    const onToggle = jest.fn()

    render(<AutoRetrainCard result={disabledResult} onToggle={onToggle} />)
    fireEvent.click(screen.getByRole("button", { name: /enable auto-retrain/i }))

    await waitFor(() => {
      expect(onToggle).toHaveBeenCalledWith(true)
    })
  })

  it("does not call onToggle if fetch fails", async () => {
    ;(global.fetch as jest.Mock).mockResolvedValueOnce({ ok: false })
    const onToggle = jest.fn()

    render(<AutoRetrainCard result={disabledResult} onToggle={onToggle} />)
    fireEvent.click(screen.getByRole("button", { name: /enable auto-retrain/i }))

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled()
    })
    expect(onToggle).not.toHaveBeenCalled()
  })
})
