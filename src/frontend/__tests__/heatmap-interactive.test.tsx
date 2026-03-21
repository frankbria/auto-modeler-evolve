/**
 * Tests for the interactive HeatmapChart cell click behavior.
 * The heatmap is rendered via ChartMessage with chart_type="heatmap".
 */
import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { ChartMessage } from "@/components/chat/chart-message"
import type { ChartSpec } from "@/lib/types"

// Sample correlation heatmap spec (2×2 matrix: region, revenue)
const heatmapSpec: ChartSpec = {
  chart_type: "heatmap",
  title: "Correlation Matrix",
  data: [
    { row: "revenue", revenue: 1.0, units: 0.85 },
    { row: "units", revenue: 0.85, units: 1.0 },
  ],
  x_key: "row",
  y_keys: ["revenue", "units"],
  x_label: "",
  y_label: "",
}

describe("HeatmapChart interactivity", () => {
  it("renders column headers", () => {
    render(<ChartMessage spec={heatmapSpec} />)
    // Check that column headers appear (truncated to 6 chars)
    expect(screen.getAllByTitle(/revenue/i).length).toBeGreaterThan(0)
  })

  it("renders row labels", () => {
    render(<ChartMessage spec={heatmapSpec} />)
    // Row labels appear in the data grid
    expect(screen.getAllByTitle(/revenue × units/i).length).toBeGreaterThan(0)
  })

  it("renders the correlation matrix title", () => {
    render(<ChartMessage spec={heatmapSpec} />)
    expect(screen.getByText("Correlation Matrix")).toBeInTheDocument()
  })

  it("shows selected cell info after clicking a cell", () => {
    render(<ChartMessage spec={heatmapSpec} />)
    // Click on the revenue×units cell (value 0.85)
    const cell = screen.getByTitle("revenue × units: 0.850")
    fireEvent.click(cell)
    // The selected tooltip should appear with the exact value
    expect(screen.getByText(/0\.850/)).toBeInTheDocument()
    expect(screen.getByText(/revenue × units/i)).toBeInTheDocument()
  })

  it("deselects cell on second click", () => {
    render(<ChartMessage spec={heatmapSpec} />)
    const cell = screen.getByTitle("revenue × units: 0.850")
    fireEvent.click(cell)
    // tooltip is shown
    expect(screen.getByText(/0\.850/)).toBeInTheDocument()
    // Click again to deselect
    fireEvent.click(cell)
    expect(screen.queryByText(/0\.850/)).not.toBeInTheDocument()
  })

  it("clears selection when close button is clicked", () => {
    render(<ChartMessage spec={heatmapSpec} />)
    const cell = screen.getByTitle("revenue × units: 0.850")
    fireEvent.click(cell)
    // Find and click the close button (✕)
    const closeBtn = screen.getByText("✕")
    fireEvent.click(closeBtn)
    expect(screen.queryByText(/0\.850/)).not.toBeInTheDocument()
  })

  it("shows 'r =' label in the selected tooltip", () => {
    render(<ChartMessage spec={heatmapSpec} />)
    const cell = screen.getByTitle("revenue × units: 0.850")
    fireEvent.click(cell)
    expect(screen.getByText("r =")).toBeInTheDocument()
  })

  it("renders the color-scale legend", () => {
    render(<ChartMessage spec={heatmapSpec} />)
    expect(screen.getByText("−1")).toBeInTheDocument()
    expect(screen.getByText("+1")).toBeInTheDocument()
  })
})
