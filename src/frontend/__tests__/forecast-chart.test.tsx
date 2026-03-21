/**
 * Tests for ForecastChart — time-series forecast visualization component.
 *
 * Covers:
 *  1. Renders the forecast-chart container
 *  2. Shows value_col name in the header
 *  3. Shows period count and label in the header
 *  4. Shows 'up' trend badge when trend === 'up'
 *  5. Shows 'down' trend badge when trend === 'down'
 *  6. Shows 'stable' trend badge when trend === 'stable'
 *  7. Shows the forecast summary text
 *  8. Renders without crashing when historical is empty
 *  9. Renders without crashing when forecast is empty
 * 10. Shows growth percentage in trend badge (up case)
 * 11. Store: attachForecastToLastMessage attaches forecast to last assistant message
 * 12. Store: attachForecastToLastMessage does not crash on empty messages
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ForecastChart } from "../components/data/forecast-chart"
import type { ForecastResult } from "../lib/types"
import { useAppStore } from "../lib/store"
import { act } from "react"

// ---------------------------------------------------------------------------
// Recharts mocks (ResponsiveContainer needs explicit dimensions in jsdom)
// ---------------------------------------------------------------------------
jest.mock("recharts", () => {
  const Original = jest.requireActual("recharts")
  return {
    ...Original,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container" style={{ width: 500, height: 300 }}>
        {children}
      </div>
    ),
  }
})

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

const MONTHLY_FORECAST: ForecastResult = {
  chart_type: "forecast",
  date_col: "month",
  value_col: "revenue",
  historical: [
    { date: "Jan 2024", value: 1000 },
    { date: "Feb 2024", value: 1100 },
    { date: "Mar 2024", value: 1200 },
    { date: "Apr 2024", value: 1300 },
    { date: "May 2024", value: 1400 },
    { date: "Jun 2024", value: 1500 },
  ],
  forecast: [
    { date: "Jul 2024", value: 1600, lower: 1400, upper: 1800 },
    { date: "Aug 2024", value: 1700, lower: 1500, upper: 1900 },
    { date: "Sep 2024", value: 1800, lower: 1600, upper: 2000 },
  ],
  period_label: "month",
  trend: "up",
  growth_pct: 20.0,
  summary: "Revenue is expected to increase by 20% over the next 3 months.",
  ci_level: 0.95,
}

const DOWN_FORECAST: ForecastResult = {
  ...MONTHLY_FORECAST,
  trend: "down",
  growth_pct: -15.0,
  summary: "Revenue is expected to decrease by 15% over the next 3 months.",
}

const STABLE_FORECAST: ForecastResult = {
  ...MONTHLY_FORECAST,
  trend: "stable",
  growth_pct: 1.2,
  summary: "Revenue is expected to remain relatively stable.",
}

// ---------------------------------------------------------------------------
// Component tests
// ---------------------------------------------------------------------------

describe("ForecastChart", () => {
  it("renders the forecast-chart container", () => {
    render(<ForecastChart result={MONTHLY_FORECAST} />)
    expect(screen.getByTestId("forecast-chart")).toBeInTheDocument()
  })

  it("shows value_col name in the header", () => {
    render(<ForecastChart result={MONTHLY_FORECAST} />)
    const items = screen.getAllByText(/revenue/i)
    expect(items.length).toBeGreaterThan(0)
  })

  it("shows period count and period label in the header", () => {
    render(<ForecastChart result={MONTHLY_FORECAST} />)
    // "3-month Forecast" should be present
    expect(screen.getByText(/3-month forecast/i)).toBeInTheDocument()
  })

  it("shows up trend badge with growth pct when trend is up", () => {
    render(<ForecastChart result={MONTHLY_FORECAST} />)
    const badge = screen.getByTestId("trend-badge-up")
    expect(badge).toBeInTheDocument()
    expect(badge.textContent).toContain("20")
  })

  it("shows down trend badge when trend is down", () => {
    render(<ForecastChart result={DOWN_FORECAST} />)
    expect(screen.getByTestId("trend-badge-down")).toBeInTheDocument()
  })

  it("shows stable trend badge when trend is stable", () => {
    render(<ForecastChart result={STABLE_FORECAST} />)
    expect(screen.getByTestId("trend-badge-stable")).toBeInTheDocument()
  })

  it("renders the forecast summary text", () => {
    render(<ForecastChart result={MONTHLY_FORECAST} />)
    const summary = screen.getByTestId("forecast-summary")
    expect(summary.textContent).toContain("Revenue is expected to increase")
  })

  it("renders without crashing when historical is empty", () => {
    const result: ForecastResult = {
      ...MONTHLY_FORECAST,
      historical: [],
    }
    expect(() => render(<ForecastChart result={result} />)).not.toThrow()
  })

  it("renders without crashing when forecast is empty", () => {
    const result: ForecastResult = {
      ...MONTHLY_FORECAST,
      forecast: [],
    }
    expect(() => render(<ForecastChart result={result} />)).not.toThrow()
  })

  it("shows 95% confidence band description", () => {
    render(<ForecastChart result={MONTHLY_FORECAST} />)
    expect(screen.getByText(/95%/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store tests
// ---------------------------------------------------------------------------

describe("store: attachForecastToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [],
      isStreaming: false,
    })
  })

  it("attaches forecast to the last assistant message", () => {
    const ts = new Date().toISOString()
    act(() => {
      useAppStore.getState().addMessage({ role: "assistant", content: "Here is the forecast", timestamp: ts })
    })
    act(() => {
      useAppStore.getState().attachForecastToLastMessage(MONTHLY_FORECAST)
    })
    const msgs = useAppStore.getState().messages
    expect(msgs[0].forecast).toEqual(MONTHLY_FORECAST)
  })

  it("does not attach forecast when messages list is empty", () => {
    expect(() => {
      act(() => {
        useAppStore.getState().attachForecastToLastMessage(MONTHLY_FORECAST)
      })
    }).not.toThrow()
    expect(useAppStore.getState().messages).toHaveLength(0)
  })

  it("does not attach to user messages", () => {
    const ts = new Date().toISOString()
    act(() => {
      useAppStore.getState().addMessage({ role: "user", content: "forecast sales", timestamp: ts })
    })
    act(() => {
      useAppStore.getState().attachForecastToLastMessage(MONTHLY_FORECAST)
    })
    const msgs = useAppStore.getState().messages
    expect(msgs[0].forecast).toBeUndefined()
  })
})
