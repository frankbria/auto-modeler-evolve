import { render, screen } from "@testing-library/react"
import { GroupTrendCard } from "@/components/data/group-trend-card"
import type { GroupTrendResult } from "@/lib/types"

const baseResult: GroupTrendResult = {
  dataset_id: "ds-1",
  date_col: "date",
  group_col: "region",
  value_col: "revenue",
  groups: [
    {
      group: "East",
      slope: 0.32,
      pct_change: 110.0,
      direction: "up",
      first_value: 1100,
      last_value: 2300,
      n_periods: 12,
      rank: 1,
    },
    {
      group: "North",
      slope: 0.01,
      pct_change: 0.3,
      direction: "flat",
      first_value: 3000,
      last_value: 3010,
      n_periods: 12,
      rank: 2,
    },
    {
      group: "West",
      slope: -0.45,
      pct_change: -48.0,
      direction: "down",
      first_value: 4800,
      last_value: 2500,
      n_periods: 12,
      rank: 3,
    },
  ],
  rising: 1,
  falling: 1,
  flat: 1,
  summary:
    "'East' is growing fastest in 'revenue' (+110.0% over the period), while 'West' is declining most (-48.0%).",
}

describe("GroupTrendCard", () => {
  it("renders the group column and value column names", () => {
    render(<GroupTrendCard result={baseResult} />)
    expect(screen.getAllByText(/region/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/revenue/i).length).toBeGreaterThan(0)
  })

  it("renders all three group names", () => {
    render(<GroupTrendCard result={baseResult} />)
    expect(screen.getByText("East")).toBeInTheDocument()
    expect(screen.getByText("North")).toBeInTheDocument()
    expect(screen.getByText("West")).toBeInTheDocument()
  })

  it("shows rising/falling/flat count badges", () => {
    render(<GroupTrendCard result={baseResult} />)
    expect(screen.getByText(/1 rising/i)).toBeInTheDocument()
    expect(screen.getByText(/1 falling/i)).toBeInTheDocument()
    expect(screen.getByText(/1 flat/i)).toBeInTheDocument()
  })

  it("renders direction arrows for each row", () => {
    render(<GroupTrendCard result={baseResult} />)
    // ▲ for up, ▼ for down, → for flat
    expect(document.body.textContent).toContain("▲")
    expect(document.body.textContent).toContain("▼")
    expect(document.body.textContent).toContain("→")
  })

  it("shows pct_change values with sign", () => {
    render(<GroupTrendCard result={baseResult} />)
    expect(screen.getByText("+110.0%")).toBeInTheDocument()
    expect(screen.getByText("-48.0%")).toBeInTheDocument()
  })

  it("shows rank numbers", () => {
    render(<GroupTrendCard result={baseResult} />)
    expect(screen.getByText("1")).toBeInTheDocument()
    expect(screen.getByText("2")).toBeInTheDocument()
    expect(screen.getByText("3")).toBeInTheDocument()
  })

  it("renders summary footer", () => {
    render(<GroupTrendCard result={baseResult} />)
    expect(screen.getByText(/East.*growing fastest/i)).toBeInTheDocument()
  })

  it("has accessible region label", () => {
    render(<GroupTrendCard result={baseResult} />)
    expect(
      screen.getByRole("region", { name: /Group trends: revenue by region/i })
    ).toBeInTheDocument()
  })

  it("formats first and last values", () => {
    render(<GroupTrendCard result={baseResult} />)
    // first_value=1100 → "1.1k", last_value=2300 → "2.3k"
    expect(screen.getAllByText(/1\.1k/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/2\.3k/i).length).toBeGreaterThan(0)
  })

  it("renders correctly when only rising groups exist", () => {
    const onlyRising: GroupTrendResult = {
      ...baseResult,
      groups: [baseResult.groups[0]],
      rising: 1,
      falling: 0,
      flat: 0,
      summary: "'East' is growing fastest.",
    }
    render(<GroupTrendCard result={onlyRising} />)
    expect(screen.getByText(/1 rising/i)).toBeInTheDocument()
    expect(screen.queryByText(/falling/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/flat/i)).not.toBeInTheDocument()
  })

  it("renders correctly when only falling groups exist", () => {
    const onlyFalling: GroupTrendResult = {
      ...baseResult,
      groups: [baseResult.groups[2]],
      rising: 0,
      falling: 1,
      flat: 0,
      summary: "All groups are declining.",
    }
    render(<GroupTrendCard result={onlyFalling} />)
    expect(screen.getByText(/1 falling/i)).toBeInTheDocument()
    expect(screen.queryByText(/rising/i)).not.toBeInTheDocument()
  })

  it("replaces underscores in column names with spaces", () => {
    const underscore: GroupTrendResult = {
      ...baseResult,
      group_col: "product_category",
      value_col: "total_revenue",
    }
    render(<GroupTrendCard result={underscore} />)
    expect(screen.getAllByText(/product category/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/total revenue/i).length).toBeGreaterThan(0)
  })

  it("formats large values with M suffix", () => {
    const bigResult: GroupTrendResult = {
      ...baseResult,
      groups: [
        {
          ...baseResult.groups[0],
          first_value: 2_500_000,
          last_value: 3_100_000,
        },
      ],
      rising: 1,
      falling: 0,
      flat: 0,
    }
    render(<GroupTrendCard result={bigResult} />)
    expect(screen.getAllByText(/2\.50M/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/3\.10M/i).length).toBeGreaterThan(0)
  })
})
