import React from "react"
import { act } from "react"
import { render, screen, renderHook } from "@testing-library/react"
import { useAppStore } from "@/lib/store"
import { PredictionValueTrendCard } from "@/components/deploy/prediction-value-trend-card"
import type { PredictionValueTrendResult } from "@/lib/types"

function makeResult(overrides: Partial<PredictionValueTrendResult> = {}): PredictionValueTrendResult {
  return {
    deployment_id: "dep-1",
    algorithm: "linear_regression",
    target_col: "revenue",
    period: "day",
    periods: [
      { period_label: "Jan 01", period_start: "2025-01-01", mean: 100, count: 5, min: 90, max: 110 },
      { period_label: "Jan 02", period_start: "2025-01-02", mean: 110, count: 4, min: 100, max: 120 },
      { period_label: "Jan 03", period_start: "2025-01-03", mean: 120, count: 6, min: 110, max: 130 },
    ],
    n_total: 15,
    n_periods_with_data: 3,
    direction: "trending_up",
    direction_label: "Trending Up",
    slope: 10.0,
    slope_pct_per_period: 10.0,
    first_period_mean: 100,
    last_period_mean: 120,
    overall_change_pct: 20.0,
    summary: "Prediction values have increased +20.0% over the last 3 days (15 predictions).",
    ...overrides,
  }
}

// Mock Recharts to avoid canvas/SVG rendering issues in jsdom
jest.mock("recharts", () => {
  const actual = jest.requireActual("recharts")
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
    LineChart: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="line-chart">{children}</div>
    ),
    Line: () => <div data-testid="line" />,
    XAxis: () => null,
    YAxis: () => null,
    CartesianGrid: () => null,
    Tooltip: () => null,
  }
})

describe("PredictionValueTrendCard", () => {
  it("renders the card root", () => {
    render(<PredictionValueTrendCard result={makeResult()} />)
    expect(screen.getByTestId("prediction-value-trend-card")).toBeInTheDocument()
  })

  it("shows direction badge", () => {
    render(<PredictionValueTrendCard result={makeResult()} />)
    expect(screen.getByTestId("direction-badge")).toHaveTextContent("Trending Up")
  })

  it("shows first period mean stat", () => {
    render(<PredictionValueTrendCard result={makeResult()} />)
    expect(screen.getByTestId("first-period-mean")).toHaveTextContent("100")
  })

  it("shows last period mean stat", () => {
    render(<PredictionValueTrendCard result={makeResult()} />)
    expect(screen.getByTestId("last-period-mean")).toHaveTextContent("120")
  })

  it("shows net change percentage", () => {
    render(<PredictionValueTrendCard result={makeResult()} />)
    expect(screen.getByTestId("overall-change-pct")).toHaveTextContent("+20.0%")
  })

  it("renders the trend chart container", () => {
    render(<PredictionValueTrendCard result={makeResult()} />)
    expect(screen.getByTestId("trend-chart")).toBeInTheDocument()
  })

  it("shows period count", () => {
    render(<PredictionValueTrendCard result={makeResult()} />)
    expect(screen.getByTestId("period-count")).toHaveTextContent("3 days")
  })

  it("shows n_total predictions", () => {
    render(<PredictionValueTrendCard result={makeResult()} />)
    expect(screen.getByTestId("n-total")).toHaveTextContent("15 predictions")
  })

  it("renders summary paragraph", () => {
    render(<PredictionValueTrendCard result={makeResult()} />)
    expect(screen.getByTestId("trend-summary")).toBeInTheDocument()
  })

  it("includes sr-only figcaption for accessibility", () => {
    const { container } = render(<PredictionValueTrendCard result={makeResult()} />)
    const caption = container.querySelector("figcaption.sr-only")
    expect(caption).toBeInTheDocument()
    expect(caption?.textContent).toContain("Trending Up")
  })

  it("shows 'week' label for week period", () => {
    render(<PredictionValueTrendCard result={makeResult({ period: "week" })} />)
    expect(screen.getByTestId("period-count")).toHaveTextContent("weeks")
  })

  it("renders stable variant with amber badge", () => {
    render(
      <PredictionValueTrendCard result={makeResult({ direction: "stable", direction_label: "Stable", overall_change_pct: 1.0 })} />
    )
    expect(screen.getByTestId("prediction-value-trend-card")).toBeInTheDocument()
    expect(screen.getByTestId("direction-badge")).toHaveTextContent("Stable")
  })

  it("renders trending_down variant with rose badge", () => {
    render(
      <PredictionValueTrendCard
        result={makeResult({ direction: "trending_down", direction_label: "Trending Down", overall_change_pct: -25.0 })}
      />
    )
    expect(screen.getByTestId("direction-badge")).toHaveTextContent("Trending Down")
    expect(screen.getByTestId("overall-change-pct")).toHaveTextContent("-25.0%")
  })

  it("shows algorithm name when present", () => {
    render(<PredictionValueTrendCard result={makeResult({ algorithm: "linear_regression" })} />)
    expect(screen.getByText("linear_regression")).toBeInTheDocument()
  })

  it("renders no_data empty state without chart", () => {
    render(
      <PredictionValueTrendCard
        result={makeResult({ direction: "no_data", direction_label: "Not enough data", periods: [], n_total: 0, n_periods_with_data: 0 })}
      />
    )
    expect(screen.getByTestId("prediction-value-trend-card")).toBeInTheDocument()
    expect(screen.queryByTestId("trend-chart")).not.toBeInTheDocument()
  })

  it("shows 'no net change' label when change < 0.1%", () => {
    render(
      <PredictionValueTrendCard
        result={makeResult({ overall_change_pct: 0.05, direction: "stable", direction_label: "Stable" })}
      />
    )
    expect(screen.getByTestId("overall-change-pct")).toHaveTextContent("no net change")
  })
})

describe("attachPredictionValueTrendToLastMessage store action", () => {
  it("attaches prediction_value_trend to last assistant message", () => {
    const { result } = renderHook(() => useAppStore())

    act(() => {
      result.current.setMessages([
        { role: "user", content: "prediction trend?", timestamp: new Date().toISOString() },
        { role: "assistant", content: "Let me check.", timestamp: new Date().toISOString() },
      ])
    })

    const trendData = makeResult({ direction: "trending_up" })

    act(() => {
      result.current.attachPredictionValueTrendToLastMessage(trendData)
    })

    const msgs = result.current.messages
    expect(msgs[msgs.length - 1].prediction_value_trend).toBeDefined()
    expect(msgs[msgs.length - 1].prediction_value_trend?.direction).toBe("trending_up")
  })

  it("does not attach to user messages", () => {
    const { result } = renderHook(() => useAppStore())

    act(() => {
      result.current.setMessages([
        { role: "user", content: "prediction trend?", timestamp: new Date().toISOString() },
      ])
    })

    act(() => {
      result.current.attachPredictionValueTrendToLastMessage(makeResult())
    })

    expect(result.current.messages[0].prediction_value_trend).toBeUndefined()
  })
})
