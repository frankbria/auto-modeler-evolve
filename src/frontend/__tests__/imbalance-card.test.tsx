/**
 * Tests for ImbalanceCard component.
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { ImbalanceCard } from "../components/models/imbalance-card"
import type { ClassImbalanceResult } from "../lib/types"

const balancedData: ClassImbalanceResult = {
  project_id: "p1",
  problem_type: "classification",
  is_imbalanced: false,
  class_distribution: [
    { class: "A", count: 50, ratio: 0.5 },
    { class: "B", count: 50, ratio: 0.5 },
  ],
  minority_class: null,
  minority_ratio: 0.5,
  recommended_strategy: "none",
  explanation: "Your target classes are roughly balanced — no special handling needed.",
}

const imbalancedData: ClassImbalanceResult = {
  project_id: "p1",
  problem_type: "classification",
  is_imbalanced: true,
  class_distribution: [
    { class: "A", count: 90, ratio: 0.9 },
    { class: "B", count: 10, ratio: 0.1 },
  ],
  minority_class: "B",
  minority_ratio: 0.1,
  recommended_strategy: "class_weight",
  explanation:
    "Your data has a class imbalance: only 10.0% of rows belong to 'B'. Without correction, the model will be biased.",
}

describe("ImbalanceCard — balanced", () => {
  it("shows balanced badge when not imbalanced", () => {
    render(
      <ImbalanceCard
        data={balancedData}
        selectedStrategy={null}
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getByText("Balanced Classes")).toBeInTheDocument()
  })

  it("renders class distribution", () => {
    render(
      <ImbalanceCard
        data={balancedData}
        selectedStrategy={null}
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getAllByText(/^A$|^B$/).length).toBeGreaterThan(0)
  })

  it("shows explanation text", () => {
    render(
      <ImbalanceCard
        data={balancedData}
        selectedStrategy={null}
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getByText(/roughly balanced/i)).toBeInTheDocument()
  })
})

describe("ImbalanceCard — imbalanced", () => {
  it("shows imbalance detected heading", () => {
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy={null}
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getByText("Class Imbalance Detected")).toBeInTheDocument()
  })

  it("shows minority percentage badge", () => {
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy={null}
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getByText(/10%\s*minority/i)).toBeInTheDocument()
  })

  it("renders all three strategy buttons", () => {
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy={null}
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getByText("Class Weighting")).toBeInTheDocument()
    expect(screen.getByText("SMOTE Oversampling")).toBeInTheDocument()
    expect(screen.getByText("Threshold Tuning")).toBeInTheDocument()
  })

  it("shows recommended badge on recommended strategy", () => {
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy={null}
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getByText("recommended")).toBeInTheDocument()
  })

  it("calls onStrategyChange when a strategy is clicked", () => {
    const onChange = jest.fn()
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy={null}
        onStrategyChange={onChange}
      />
    )
    fireEvent.click(screen.getByText("Class Weighting"))
    expect(onChange).toHaveBeenCalledWith("class_weight")
  })

  it("calls onStrategyChange with null when selected strategy is clicked again", () => {
    const onChange = jest.fn()
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy="class_weight"
        onStrategyChange={onChange}
      />
    )
    fireEvent.click(screen.getByText("Class Weighting"))
    expect(onChange).toHaveBeenCalledWith(null)
  })

  it("shows selected badge on selected strategy", () => {
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy="smote"
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getByText("selected")).toBeInTheDocument()
  })

  it("shows strategy note when a strategy is selected", () => {
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy="class_weight"
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getByText(/Strategy will apply/i)).toBeInTheDocument()
  })

  it("does not show strategy note when no strategy selected", () => {
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy={null}
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.queryByText(/Strategy will apply/i)).not.toBeInTheDocument()
  })

  it("strategy buttons have aria-pressed", () => {
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy="smote"
        onStrategyChange={jest.fn()}
      />
    )
    const smoteBtn = screen.getByRole("button", { name: /smote oversampling/i })
    expect(smoteBtn).toHaveAttribute("aria-pressed", "true")
    const cwBtn = screen.getByRole("button", { name: /class weighting/i })
    expect(cwBtn).toHaveAttribute("aria-pressed", "false")
  })

  it("minority class badge appears in distribution", () => {
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy={null}
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getByText("minority")).toBeInTheDocument()
  })

  it("explanation text is rendered", () => {
    render(
      <ImbalanceCard
        data={imbalancedData}
        selectedStrategy={null}
        onStrategyChange={jest.fn()}
      />
    )
    expect(screen.getAllByText(/class imbalance/i).length).toBeGreaterThan(0)
  })
})
