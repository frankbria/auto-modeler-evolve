import React from "react"
import { render, screen } from "@testing-library/react"
import { WeeklyUsageReportCard } from "@/components/deploy/weekly-usage-report-card"
import type { WeeklyUsageReportResult } from "@/lib/types"

function makeResult(
  overrides: Partial<WeeklyUsageReportResult> = {}
): WeeklyUsageReportResult {
  const by_day = [
    { date: "2026-05-14", count: 5 },
    { date: "2026-05-15", count: 3 },
    { date: "2026-05-16", count: 8 },
    { date: "2026-05-17", count: 2 },
    { date: "2026-05-18", count: 6 },
    { date: "2026-05-19", count: 4 },
    { date: "2026-05-20", count: 7 },
  ]
  return {
    deployment_id: "dep-001",
    this_week_count: 35,
    last_week_count: 28,
    change_pct: 25.0,
    trend: "up",
    by_day,
    top_input_patterns: [
      {
        feature: "region",
        top_values: [
          { value: "East", count: 15, pct: 43 },
          { value: "North", count: 10, pct: 29 },
        ],
      },
      {
        feature: "product",
        top_values: [{ value: "Widget A", count: 20, pct: 57 }],
      },
    ],
    sample_size: 35,
    summary: "This week: 35 predictions (up 25% vs last week). Last week: 28 predictions.",
    ...overrides,
  }
}

describe("WeeklyUsageReportCard", () => {
  it("renders the card heading", () => {
    render(<WeeklyUsageReportCard result={makeResult()} />)
    expect(screen.getByText("Weekly Usage Report")).toBeInTheDocument()
  })

  it("shows this-week and last-week counts", () => {
    render(<WeeklyUsageReportCard result={makeResult()} />)
    expect(screen.getByTestId("this-week-count")).toHaveTextContent("35")
    expect(screen.getByTestId("last-week-count")).toHaveTextContent("28")
  })

  it("renders the summary paragraph", () => {
    render(<WeeklyUsageReportCard result={makeResult()} />)
    expect(screen.getByTestId("weekly-summary")).toHaveTextContent("35 predictions")
    expect(screen.getByTestId("weekly-summary")).toHaveTextContent("28 predictions")
  })

  it("shows an upward trend badge", () => {
    render(<WeeklyUsageReportCard result={makeResult({ trend: "up", change_pct: 25 })} />)
    expect(screen.getByText(/↑.*25%.*vs last week/i)).toBeInTheDocument()
  })

  it("shows a downward trend badge", () => {
    render(
      <WeeklyUsageReportCard
        result={makeResult({ trend: "down", change_pct: -15.5, summary: "down 15.5%" })}
      />
    )
    expect(screen.getByText(/↓.*15\.5%.*vs last week/i)).toBeInTheDocument()
  })

  it("shows stable badge when trend is flat", () => {
    render(
      <WeeklyUsageReportCard
        result={makeResult({ trend: "flat", change_pct: 0 })}
      />
    )
    expect(screen.getByText(/Stable/i)).toBeInTheDocument()
  })

  it("shows 'No prior data' badge when change_pct is null", () => {
    render(
      <WeeklyUsageReportCard result={makeResult({ change_pct: null, trend: "flat" })} />
    )
    expect(screen.getByText(/No prior data/i)).toBeInTheDocument()
  })

  it("renders 7 day-bar entries in the chart", () => {
    render(<WeeklyUsageReportCard result={makeResult()} />)
    // The bar chart container is aria-labeled
    const chart = screen.getByRole("img", {
      name: /bar chart.*predictions per day/i,
    })
    expect(chart).toBeInTheDocument()
    // 7 bars with aria-labels
    const bars = screen.getAllByLabelText(/\d{2}\/\d{2}: \d+ predictions/)
    expect(bars).toHaveLength(7)
  })

  it("renders top input patterns section", () => {
    render(<WeeklyUsageReportCard result={makeResult()} />)
    const patternsSection = screen.getByTestId("top-input-patterns")
    expect(patternsSection).toBeInTheDocument()
    expect(screen.getByText("region")).toBeInTheDocument()
    expect(screen.getByText("East")).toBeInTheDocument()
    expect(screen.getByText("Widget A")).toBeInTheDocument()
  })

  it("shows empty state when no input patterns", () => {
    render(
      <WeeklyUsageReportCard result={makeResult({ top_input_patterns: [] })} />
    )
    expect(
      screen.getByText(/No categorical input patterns found/i)
    ).toBeInTheDocument()
  })

  it("has accessible figure with aria-label", () => {
    render(<WeeklyUsageReportCard result={makeResult()} />)
    const figure = screen.getByRole("figure", {
      name: /weekly usage report/i,
    })
    expect(figure).toBeInTheDocument()
  })
})
