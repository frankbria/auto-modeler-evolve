import React from "react"
import { render, screen } from "@testing-library/react"
import { AggregateExplanationCard } from "@/components/chat/aggregate-explanation-card"
import type { AggregateExplanationResult } from "@/lib/types"

const MOCK_RESULT: AggregateExplanationResult = {
  deployment_id: "dep-1",
  sample_count: 10,
  summary: "units is the top driver, mostly pushing predictions up.",
  features: [
    {
      feature: "units",
      avg_abs_contribution: 2.5,
      positive_pct: 80,
      direction_label: "mostly positive",
      top_driver_pct: 60,
      sample_count: 10,
    },
    {
      feature: "price",
      avg_abs_contribution: 1.0,
      positive_pct: 20,
      direction_label: "mostly negative",
      top_driver_pct: 25,
      sample_count: 10,
    },
    {
      feature: "discount",
      avg_abs_contribution: 0.5,
      positive_pct: 50,
      direction_label: "mixed",
      top_driver_pct: 15,
      sample_count: 10,
    },
  ],
}

describe("AggregateExplanationCard", () => {
  it("renders the card heading", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText("Production Feature Influence")).toBeInTheDocument()
  })

  it("shows the sample count badge", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText("10 predictions")).toBeInTheDocument()
  })

  it("shows the feature count badge", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText("3 features")).toBeInTheDocument()
  })

  it("renders all feature names", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText("units")).toBeInTheDocument()
    expect(screen.getByText("price")).toBeInTheDocument()
    expect(screen.getByText("discount")).toBeInTheDocument()
  })

  it("renders positive direction badge for mostly positive feature", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText("↑ mostly positive")).toBeInTheDocument()
  })

  it("renders negative direction badge for mostly negative feature", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText("↓ mostly negative")).toBeInTheDocument()
  })

  it("renders mixed direction badge for mixed feature", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText("↔ mixed")).toBeInTheDocument()
  })

  it("shows top driver badge when top_driver_pct >= 30", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText("top driver 60%")).toBeInTheDocument()
  })

  it("does not show top driver badge when top_driver_pct < 30", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.queryByText("top driver 15%")).not.toBeInTheDocument()
    expect(screen.queryByText("top driver 25%")).not.toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText(MOCK_RESULT.summary)).toBeInTheDocument()
  })

  it("renders legend footnote", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText(/Bar width = average absolute influence/)).toBeInTheDocument()
  })

  it("renders accessible region with aria-label", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(
      screen.getByRole("region", { name: "Aggregate production explanation" })
    ).toBeInTheDocument()
  })

  it("renders feature list with aria-label", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(
      screen.getByRole("list", { name: "Feature influence in production predictions" })
    ).toBeInTheDocument()
  })

  it("renders sr-only figcaption", () => {
    const { container } = render(<AggregateExplanationCard result={MOCK_RESULT} />)
    const figcaption = container.querySelector("figcaption.sr-only")
    expect(figcaption).toBeInTheDocument()
    expect(figcaption?.textContent).toContain("10 production predictions")
  })

  it("shows empty state when features list is empty", () => {
    render(
      <AggregateExplanationCard
        result={{ ...MOCK_RESULT, features: [], sample_count: 0 }}
      />
    )
    expect(screen.getByText("No feature contributions found.")).toBeInTheDocument()
  })

  it("shows positive_pct in feature row", () => {
    render(<AggregateExplanationCard result={MOCK_RESULT} />)
    expect(screen.getByText("80% pos")).toBeInTheDocument()
  })

  it("limits display to 10 features", () => {
    const manyFeatures = Array.from({ length: 15 }, (_, i) => ({
      feature: `feat_${i}`,
      avg_abs_contribution: 15 - i,
      positive_pct: 50,
      direction_label: "mixed" as const,
      top_driver_pct: 10,
      sample_count: 5,
    }))
    render(
      <AggregateExplanationCard result={{ ...MOCK_RESULT, features: manyFeatures }} />
    )
    const items = screen.getAllByRole("listitem")
    expect(items.length).toBe(10)
  })
})
