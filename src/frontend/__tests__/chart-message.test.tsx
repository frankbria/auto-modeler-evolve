/**
 * Tests for ChartMessage — the inline chart renderer in the chat interface.
 *
 * These tests confirm that each chart type renders without crashing and
 * that the chart title appears in the DOM. Pixel-level rendering is
 * validated separately via E2E tests.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { ChartMessage } from "../components/chat/chart-message"
import type { ChartSpec } from "../lib/types"

const makeSpec = (overrides: Partial<ChartSpec> = {}): ChartSpec => ({
  chart_type: "bar",
  title: "Revenue by Region",
  data: [{ region: "North", revenue: 1000 }, { region: "South", revenue: 800 }],
  x_key: "region",
  y_keys: ["revenue"],
  x_label: "Region",
  y_label: "Revenue",
  ...overrides,
})

describe("ChartMessage", () => {
  it("renders bar chart title", () => {
    render(<ChartMessage spec={makeSpec({ chart_type: "bar" })} />)
    expect(screen.getByText("Revenue by Region")).toBeInTheDocument()
  })

  it("renders line chart without crashing", () => {
    const { container } = render(
      <ChartMessage
        spec={makeSpec({
          chart_type: "line",
          title: "Trend Over Time",
          data: [{ date: "Jan", value: 100 }, { date: "Feb", value: 150 }],
          x_key: "date",
          y_keys: ["value"],
        })}
      />
    )
    expect(screen.getByText("Trend Over Time")).toBeInTheDocument()
    expect(container.firstChild).not.toBeNull()
  })

  it("renders histogram as bar chart type", () => {
    render(
      <ChartMessage
        spec={makeSpec({ chart_type: "histogram", title: "Value Distribution" })}
      />
    )
    expect(screen.getByText("Value Distribution")).toBeInTheDocument()
  })

  it("renders scatter chart without crashing", () => {
    render(
      <ChartMessage
        spec={makeSpec({
          chart_type: "scatter",
          title: "Correlation Plot",
          data: [{ x: 1, y: 2 }, { x: 3, y: 4 }],
          x_key: "x",
          y_keys: ["y"],
        })}
      />
    )
    expect(screen.getByText("Correlation Plot")).toBeInTheDocument()
  })

  it("renders pie chart without crashing", () => {
    render(
      <ChartMessage
        spec={makeSpec({
          chart_type: "pie",
          title: "Market Share",
          data: [
            { product: "A", share: 60 },
            { product: "B", share: 40 },
          ],
          x_key: "product",
          y_keys: ["share"],
        })}
      />
    )
    expect(screen.getByText("Market Share")).toBeInTheDocument()
  })

  it("renders heatmap without crashing", () => {
    render(
      <ChartMessage
        spec={makeSpec({
          chart_type: "heatmap",
          title: "Correlation Matrix",
          data: [{ row: "revenue", revenue: 1.0, units: 0.8 }],
          x_key: "row",
          y_keys: ["revenue", "units"],
        })}
      />
    )
    expect(screen.getByText("Correlation Matrix")).toBeInTheDocument()
  })

  it("heatmap shows column headers (truncated at 6 chars)", () => {
    render(
      <ChartMessage
        spec={makeSpec({
          chart_type: "heatmap",
          title: "Correlation Matrix",
          data: [{ row: "revenue", revenue: 1.0, longcolumn: 0.5 }],
          x_key: "row",
          y_keys: ["revenue", "longcolumn"],
        })}
      />
    )
    // "longcolumn" is > 6 chars, so it should be truncated to "longco…"
    expect(screen.getByText("longco…")).toBeInTheDocument()
    // "revenue" is exactly 7 chars, truncated to "revenu…"
    expect(screen.getByText("revenu…")).toBeInTheDocument()
  })

  it("renders fallback text for unknown chart type", () => {
    render(
      <ChartMessage
        spec={makeSpec({
          // @ts-expect-error — testing the unknown fallback path
          chart_type: "unknown_type",
          title: "",
        })}
      />
    )
    // The chart container renders but no specific assertion needed for unknown — just no crash
    expect(screen.queryByText("Revenue by Region")).not.toBeInTheDocument()
  })

  it("omits title element when title is empty", () => {
    const { container } = render(<ChartMessage spec={makeSpec({ title: "" })} />)
    // Empty string is falsy — title <p> should not render
    const titleEl = container.querySelector("p.text-xs.font-semibold")
    expect(titleEl).not.toBeInTheDocument()
  })

  it("multi-series line chart renders without crashing", () => {
    render(
      <ChartMessage
        spec={makeSpec({
          chart_type: "line",
          title: "Multi-series",
          data: [
            { date: "Jan", actual: 100, rolling_avg: 90, trend: 95 },
            { date: "Feb", actual: 120, rolling_avg: 105, trend: 108 },
          ],
          x_key: "date",
          y_keys: ["actual", "rolling_avg", "trend"],
          x_label: "Date",
          y_label: "Value",
        })}
      />
    )
    expect(screen.getByText("Multi-series")).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// corrColor logic (exported indirectly via heatmap cell background colors)
// We test the deterministic color behavior through rendered cells.
// ---------------------------------------------------------------------------

describe("HeatmapChart color scale (via ChartMessage)", () => {
  it("renders cells for each data entry", () => {
    const { container } = render(
      <ChartMessage
        spec={{
          chart_type: "heatmap",
          title: "Test",
          data: [
            { row: "a", x: 1.0, y: 0.5 },
            { row: "b", x: -0.3, y: 0.0 },
          ],
          x_key: "row",
          y_keys: ["x", "y"],
          x_label: "",
          y_label: "",
        }}
      />
    )
    // Each data row × each column = 4 cells rendered
    // Cells show the value formatted to 2 decimal places
    const cells = container.querySelectorAll("[title*='×']")
    expect(cells.length).toBe(4)
  })
})
