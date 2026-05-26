import React from "react"
import { render, screen } from "@testing-library/react"
import { CounterfactualCard } from "../components/deploy/counterfactual-card"
import { CounterfactualResult } from "../lib/types"

function makeData(overrides: Partial<CounterfactualResult> = {}): CounterfactualResult {
  return {
    row_index: 3,
    problem_type: "classification",
    target_column: "churn",
    original_prediction: "yes",
    original_class: "yes",
    original_confidence: 0.82,
    counterfactual_prediction: "no",
    target_class: "no",
    counterfactual_confidence: 0.67,
    flipped: true,
    n_steps: 4,
    changed_features: [
      {
        name: "tenure",
        original_value: 2,
        counterfactual_value: 4.2,
        change_pct: 110.0,
        direction: "increase",
      },
      {
        name: "monthly_charges",
        original_value: 80.5,
        counterfactual_value: 60.3,
        change_pct: -25.1,
        direction: "decrease",
      },
    ],
    summary: "By increasing tenure from 2 to 4.2, the churn prediction flips from 'yes' to 'no'.",
    ...overrides,
  }
}

describe("CounterfactualCard", () => {
  it("renders the card container with correct test id", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("counterfactual-card")).toBeInTheDocument()
  })

  it("shows row index in heading", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByText(/Row 3/)).toBeInTheDocument()
  })

  it("shows target_column name", () => {
    render(<CounterfactualCard data={makeData()} />)
    // Exact match — the badge span contains exactly "churn"
    expect(screen.getByText("churn")).toBeInTheDocument()
  })

  it("shows original prediction", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("original-prediction")).toHaveTextContent("yes")
  })

  it("shows original confidence percentage", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("original-confidence")).toHaveTextContent("82% confidence")
  })

  it("shows counterfactual prediction when flipped", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("counterfactual-prediction")).toHaveTextContent("no")
  })

  it("shows 'Flip found' badge when flipped", () => {
    render(<CounterfactualCard data={makeData({ flipped: true })} />)
    expect(screen.getByText(/Flip found/)).toBeInTheDocument()
  })

  it("shows 'No flip found' badge when not flipped", () => {
    render(<CounterfactualCard data={makeData({ flipped: false })} />)
    expect(screen.getByText(/No flip found/)).toBeInTheDocument()
  })

  it("shows changed features table", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByText("Changes needed")).toBeInTheDocument()
  })

  it("shows feature names in table", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByText("tenure")).toBeInTheDocument()
    expect(screen.getByText("monthly charges")).toBeInTheDocument()
  })

  it("shows increase percentage badge for tenure", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByText("+110.0%")).toBeInTheDocument()
  })

  it("shows decrease percentage badge for monthly_charges", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByText("-25.1%")).toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByTestId("counterfactual-summary")).toHaveTextContent(/increasing tenure/)
  })

  it("renders accessibility figcaption", () => {
    render(<CounterfactualCard data={makeData()} />)
    const figcaption = screen.getByRole("figure").querySelector("figcaption")
    expect(figcaption).toBeInTheDocument()
  })

  it("shows 'Not reachable' when not flipped", () => {
    render(
      <CounterfactualCard
        data={makeData({
          flipped: false,
          counterfactual_confidence: null,
          n_steps: 15,
        })}
      />
    )
    expect(screen.getByText(/Not reachable in 15 steps/)).toBeInTheDocument()
  })

  it("shows empty state message when no changed features and flipped", () => {
    render(
      <CounterfactualCard
        data={makeData({
          flipped: true,
          changed_features: [],
          n_steps: 0,
          summary: "Prediction already at target class.",
        })}
      />
    )
    // The inline paragraph (not the summary) mentions "no feature changes"
    expect(screen.getByText(/No feature changes were needed/)).toBeInTheDocument()
  })

  it("shows empty-no-features message when not flipped and no changes", () => {
    render(
      <CounterfactualCard
        data={makeData({
          flipped: false,
          changed_features: [],
          summary: "No numeric features could be adjusted.",
          original_class: "yes",
          target_class: "no",
        })}
      />
    )
    // The inline paragraph says "firmly in class"
    expect(
      screen.getByText((content) => content.includes("firmly in class"))
    ).toBeInTheDocument()
  })

  it("shows Classification badge", () => {
    render(<CounterfactualCard data={makeData()} />)
    expect(screen.getByText("Classification")).toBeInTheDocument()
  })

  it("formats large values with k suffix", () => {
    render(
      <CounterfactualCard
        data={makeData({
          changed_features: [
            {
              name: "revenue",
              original_value: 1500000,
              counterfactual_value: 2000000,
              change_pct: 33.3,
              direction: "increase",
            },
          ],
        })}
      />
    )
    // 1500000 → 1.5M, 2000000 → 2.0M
    expect(screen.getByText("1.5M")).toBeInTheDocument()
    expect(screen.getByText("2.0M")).toBeInTheDocument()
  })
})

// ── Zustand store action test ─────────────────────────────────────────────
describe("attachCounterfactualToLastMessage", () => {
  it("attaches counterfactual to last assistant message", () => {
    const { useAppStore } = require("../lib/store")
    const store = useAppStore.getState()

    store.messages = [
      { id: "1", role: "user", content: "What would change prediction?" },
      { id: "2", role: "assistant", content: "Let me check..." },
    ]

    const data = makeData()
    store.attachCounterfactualToLastMessage(data)

    const updated = useAppStore.getState().messages
    expect(updated[1].counterfactual).toEqual(data)
  })

  it("does not modify non-assistant messages", () => {
    const { useAppStore } = require("../lib/store")
    const store = useAppStore.getState()

    store.messages = [
      { id: "1", role: "user", content: "flip prediction for row 5" },
    ]

    store.attachCounterfactualToLastMessage(makeData())

    const updated = useAppStore.getState().messages
    expect(updated[0].counterfactual).toBeUndefined()
  })
})
