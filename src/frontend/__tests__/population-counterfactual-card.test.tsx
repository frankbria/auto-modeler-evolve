import React from "react"
import { render, screen } from "@testing-library/react"
import { PopulationCounterfactualCard } from "../components/deploy/population-counterfactual-card"
import { PopulationCounterfactualResult } from "../lib/types"
import { useAppStore } from "../lib/store"

// ---------------------------------------------------------------------------
// Test data factories
// ---------------------------------------------------------------------------

function makeFeatureSummary() {
  return [
    { feature: "tenure", flip_count: 7, flip_pct: 70.0, avg_change_pct: 45.2, direction: "increase" as const },
    { feature: "monthly_charges", flip_count: 4, flip_pct: 40.0, avg_change_pct: 22.5, direction: "decrease" as const },
    { feature: "contract_length", flip_count: 2, flip_pct: 20.0, avg_change_pct: 33.0, direction: "increase" as const },
  ]
}

function makeData(overrides: Partial<PopulationCounterfactualResult> = {}): PopulationCounterfactualResult {
  return {
    problem_type: "classification",
    target_column: "churn",
    total_rows: 10,
    flipped_count: 7,
    flip_rate: 0.7,
    dominant_feature: "tenure",
    dominant_direction: "increase",
    dominant_flip_count: 7,
    dominant_avg_change_pct: 45.2,
    feature_summary: makeFeatureSummary(),
    summary: "Increasing 'tenure' by ~45.2% is the single most impactful lever across the cohort.",
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Render / structure tests
// ---------------------------------------------------------------------------

describe("PopulationCounterfactualCard", () => {
  it("renders the card container with correct test id", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("population-counterfactual-card")).toBeInTheDocument()
  })

  it("renders as a <figure> element", () => {
    const { container } = render(<PopulationCounterfactualCard data={makeData()} />)
    expect(container.querySelector("figure")).not.toBeNull()
  })

  it("shows flippable badge when flips exist", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("flip-badge")).toBeInTheDocument()
    expect(screen.getByTestId("flip-badge")).toHaveTextContent("7 of 10 flippable")
  })

  it("shows no-flip badge when flipped_count is zero", () => {
    render(<PopulationCounterfactualCard data={makeData({ flipped_count: 0 })} />)
    expect(screen.getByTestId("no-flip-badge")).toBeInTheDocument()
    expect(screen.queryByTestId("flip-badge")).toBeNull()
  })

  // ---------------------------------------------------------------------------
  // Flip rate bar
  // ---------------------------------------------------------------------------

  it("shows flip rate numbers", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("flip-rate")).toHaveTextContent("7 / 10")
    expect(screen.getByTestId("flip-rate")).toHaveTextContent("70%")
  })

  it("renders progressbar with correct aria-valuenow", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toHaveAttribute("aria-valuenow", "70")
    expect(bar).toHaveAttribute("aria-valuemin", "0")
    expect(bar).toHaveAttribute("aria-valuemax", "100")
  })

  it("shows 0% flip rate when total_rows is zero", () => {
    render(<PopulationCounterfactualCard data={makeData({ total_rows: 0, flipped_count: 0 })} />)
    expect(screen.getByTestId("flip-rate")).toHaveTextContent("0 / 0")
  })

  // ---------------------------------------------------------------------------
  // Dominant intervention
  // ---------------------------------------------------------------------------

  it("shows dominant intervention section when flips exist", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("dominant-intervention")).toBeInTheDocument()
  })

  it("does not show dominant intervention when no flips", () => {
    render(<PopulationCounterfactualCard data={makeData({ flipped_count: 0 })} />)
    expect(screen.queryByTestId("dominant-intervention")).toBeNull()
  })

  it("does not show dominant intervention when dominant_feature is null", () => {
    render(<PopulationCounterfactualCard data={makeData({ dominant_feature: null })} />)
    expect(screen.queryByTestId("dominant-intervention")).toBeNull()
  })

  it("shows dominant feature name in intervention box", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    const box = screen.getByTestId("dominant-intervention")
    expect(box).toHaveTextContent("tenure")
  })

  it("shows dominant flip count in intervention box", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("dominant-flip-count")).toHaveTextContent("7")
  })

  it("shows Increase label for increase direction", () => {
    render(<PopulationCounterfactualCard data={makeData({ dominant_direction: "increase" })} />)
    const box = screen.getByTestId("dominant-intervention")
    expect(box).toHaveTextContent("Increase")
  })

  it("shows Decrease label for decrease direction", () => {
    render(
      <PopulationCounterfactualCard
        data={makeData({
          dominant_direction: "decrease",
          dominant_feature: "monthly_charges",
        })}
      />
    )
    const box = screen.getByTestId("dominant-intervention")
    expect(box).toHaveTextContent("Decrease")
  })

  // ---------------------------------------------------------------------------
  // Feature breakdown table
  // ---------------------------------------------------------------------------

  it("renders one feature row per feature_summary entry", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    const rows = screen.getAllByTestId("population-cf-feature-row")
    expect(rows).toHaveLength(3)
  })

  it("renders no feature table when feature_summary is empty", () => {
    render(<PopulationCounterfactualCard data={makeData({ feature_summary: [] })} />)
    expect(screen.queryByTestId("population-cf-feature-row")).toBeNull()
  })

  it("shows feature name with underscores replaced by spaces", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    // "contract_length" → "contract length"
    expect(screen.getByText("contract length")).toBeInTheDocument()
  })

  it("shows avg_change_pct in feature rows", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    // ~45.2% appears in both the dominant box and the feature row
    expect(screen.getAllByText(/~45\.2%/).length).toBeGreaterThanOrEqual(1)
  })

  // ---------------------------------------------------------------------------
  // No-flips empty state
  // ---------------------------------------------------------------------------

  it("shows no-flip descriptive text when flipped_count is 0", () => {
    render(<PopulationCounterfactualCard data={makeData({ flipped_count: 0, total_rows: 8 })} />)
    expect(screen.getByText(/None of the 8 analysed records could be flipped/i)).toBeInTheDocument()
  })

  it("does not show no-flip text when flips were found", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    expect(screen.queryByText(/could not be flipped/i)).toBeNull()
  })

  // ---------------------------------------------------------------------------
  // Summary paragraph
  // ---------------------------------------------------------------------------

  it("renders summary text", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("population-cf-summary")).toHaveTextContent(
      "Increasing 'tenure' by ~45.2%"
    )
  })

  // ---------------------------------------------------------------------------
  // Accessibility
  // ---------------------------------------------------------------------------

  it("renders sr-only figcaption for screen readers", () => {
    const { container } = render(<PopulationCounterfactualCard data={makeData()} />)
    const caption = container.querySelector("figcaption")
    expect(caption).not.toBeNull()
    expect(caption).toHaveClass("sr-only")
  })

  it("figcaption mentions flipped count", () => {
    const { container } = render(<PopulationCounterfactualCard data={makeData()} />)
    const caption = container.querySelector("figcaption")!
    expect(caption.textContent).toMatch(/7.*records/)
  })

  it("figure has descriptive aria-label", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    const figure = screen.getByTestId("population-counterfactual-card")
    expect(figure).toHaveAttribute("aria-label", expect.stringContaining("7 of 10"))
  })

  // ---------------------------------------------------------------------------
  // target_column badge
  // ---------------------------------------------------------------------------

  it("shows target_column with underscores replaced by spaces", () => {
    render(
      <PopulationCounterfactualCard
        data={makeData({ target_column: "customer_churn" })}
      />
    )
    expect(screen.getByText("customer churn")).toBeInTheDocument()
  })

  it("shows total_rows badge", () => {
    render(<PopulationCounterfactualCard data={makeData()} />)
    // "10 records" appears in the badge and in the sr-only figcaption
    expect(screen.getAllByText(/10 records/).length).toBeGreaterThanOrEqual(1)
  })
})

// ---------------------------------------------------------------------------
// Zustand store action
// ---------------------------------------------------------------------------

describe("attachPopulationCounterfactualToLastMessage (store)", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "what change helps most?", id: "u1", timestamp: "" },
        { role: "assistant", content: "Analysing...", id: "a1", timestamp: "" },
      ],
    })
  })

  it("attaches population_counterfactual to the last assistant message", () => {
    const { attachPopulationCounterfactualToLastMessage } = useAppStore.getState()
    const payload = makeData()
    attachPopulationCounterfactualToLastMessage(payload)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1] as any
    expect(last.population_counterfactual).toEqual(payload)
  })

  it("does not modify earlier messages", () => {
    const { attachPopulationCounterfactualToLastMessage } = useAppStore.getState()
    attachPopulationCounterfactualToLastMessage(makeData())
    const msgs = useAppStore.getState().messages
    expect((msgs[0] as any).population_counterfactual).toBeUndefined()
  })

  it("does not attach when last message is from user", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "who?", id: "u2", timestamp: "" },
      ],
    })
    const { attachPopulationCounterfactualToLastMessage } = useAppStore.getState()
    attachPopulationCounterfactualToLastMessage(makeData())
    const msgs = useAppStore.getState().messages
    expect((msgs[msgs.length - 1] as any).population_counterfactual).toBeUndefined()
  })
})
