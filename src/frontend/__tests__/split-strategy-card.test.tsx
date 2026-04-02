import { render, screen } from "@testing-library/react"
import { SplitStrategyCard } from "@/components/models/split-strategy-card"
import type { SplitStrategyResult } from "@/lib/types"

const chronologicalResult: SplitStrategyResult = {
  split_strategy: "chronological",
  date_col: "sale_date",
  explanation:
    "Switched to time-based splitting on 'sale_date' — training on older data and testing on more recent data.",
}

const randomResult: SplitStrategyResult = {
  split_strategy: "random",
  date_col: null,
  explanation: "Switched to random split — rows will be shuffled and 20% held out at random for testing.",
}

describe("SplitStrategyCard — chronological", () => {
  it("renders the header with 'Split Strategy Updated'", () => {
    render(<SplitStrategyCard result={chronologicalResult} />)
    expect(screen.getByText("Split Strategy Updated")).toBeInTheDocument()
  })

  it("shows Time-based badge", () => {
    render(<SplitStrategyCard result={chronologicalResult} />)
    expect(screen.getByText("Time-based")).toBeInTheDocument()
  })

  it("shows the date column name", () => {
    render(<SplitStrategyCard result={chronologicalResult} />)
    expect(screen.getByText("sale_date")).toBeInTheDocument()
  })

  it("renders the explanation text", () => {
    render(<SplitStrategyCard result={chronologicalResult} />)
    expect(screen.getByText(/training on older data/i)).toBeInTheDocument()
  })

  it("shows train/test legend for chronological", () => {
    render(<SplitStrategyCard result={chronologicalResult} />)
    expect(screen.getByText(/80% train/i)).toBeInTheDocument()
    expect(screen.getByText(/20% test/i)).toBeInTheDocument()
  })

  it("uses sky-blue border styling for chronological", () => {
    const { container } = render(<SplitStrategyCard result={chronologicalResult} />)
    const card = container.firstChild as HTMLElement
    expect(card.className).toMatch(/border-sky/)
  })
})

describe("SplitStrategyCard — random", () => {
  it("shows Random badge", () => {
    render(<SplitStrategyCard result={randomResult} />)
    expect(screen.getByText("Random")).toBeInTheDocument()
  })

  it("renders the explanation text for random", () => {
    render(<SplitStrategyCard result={randomResult} />)
    expect(screen.getByText(/rows will be shuffled/i)).toBeInTheDocument()
  })

  it("does NOT show train/test legend for random", () => {
    render(<SplitStrategyCard result={randomResult} />)
    expect(screen.queryByText(/80% train/i)).not.toBeInTheDocument()
  })

  it("does not show date column label when date_col is null", () => {
    render(<SplitStrategyCard result={randomResult} />)
    expect(screen.queryByText(/Sorting by/i)).not.toBeInTheDocument()
  })

  it("uses slate border styling for random", () => {
    const { container } = render(<SplitStrategyCard result={randomResult} />)
    const card = container.firstChild as HTMLElement
    expect(card.className).toMatch(/border-slate/)
  })
})
