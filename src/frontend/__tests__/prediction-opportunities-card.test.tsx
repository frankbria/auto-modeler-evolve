import { render, screen, fireEvent } from "@testing-library/react"
import { PredictionOpportunitiesCard } from "@/components/models/prediction-opportunities-card"
import type { PredictionOpportunitiesResult } from "@/lib/types"

const singleOpp: PredictionOpportunitiesResult = {
  dataset_id: "ds-1",
  total: 1,
  opportunities: [
    {
      target_col: "revenue",
      problem_type: "regression",
      feasibility_score: 90,
      reason: "'revenue' is a numeric column with 100% complete data and real variation.",
      business_value: "high",
      example_question: "Can you predict the Revenue for each record in my next dataset?",
      predictor_count: 4,
    },
  ],
}

const multipleOpps: PredictionOpportunitiesResult = {
  dataset_id: "ds-2",
  total: 3,
  opportunities: [
    {
      target_col: "revenue",
      problem_type: "regression",
      feasibility_score: 90,
      reason: "Revenue is numeric with good variation.",
      business_value: "high",
      example_question: "Can you predict Revenue?",
      predictor_count: 5,
    },
    {
      target_col: "churn",
      problem_type: "classification",
      feasibility_score: 80,
      reason: "Churn has 2 categories and 99% complete data.",
      business_value: "high",
      example_question: "Which records are likely to churn?",
      predictor_count: 5,
    },
    {
      target_col: "units",
      problem_type: "regression",
      feasibility_score: 65,
      reason: "Units is numeric with variation.",
      business_value: "medium",
      example_question: "What will the Units be for new records?",
      predictor_count: 4,
    },
  ],
}

const emptyResult: PredictionOpportunitiesResult = {
  dataset_id: "ds-3",
  total: 0,
  opportunities: [],
}

describe("PredictionOpportunitiesCard — heading and badges", () => {
  it("shows card heading", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    expect(screen.getByText("Prediction Opportunities")).toBeInTheDocument()
  })

  it("shows count badge with singular form", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    expect(screen.getByText("1 target found")).toBeInTheDocument()
  })

  it("shows count badge with plural form", () => {
    render(<PredictionOpportunitiesCard result={multipleOpps} />)
    expect(screen.getByText("3 targets found")).toBeInTheDocument()
  })

  it("shows high value badge when high-value opportunities exist", () => {
    render(<PredictionOpportunitiesCard result={multipleOpps} />)
    expect(screen.getByText("2 high value")).toBeInTheDocument()
  })
})

describe("PredictionOpportunitiesCard — opportunity rows", () => {
  it("shows target column name", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    expect(screen.getByText(/#1 revenue/)).toBeInTheDocument()
  })

  it("shows problem type badge — regression", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    expect(screen.getByText("regression")).toBeInTheDocument()
  })

  it("shows problem type badge — classification", () => {
    render(<PredictionOpportunitiesCard result={multipleOpps} />)
    expect(screen.getByText("classification")).toBeInTheDocument()
  })

  it("shows business value badge — high", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    expect(screen.getByText("High value")).toBeInTheDocument()
  })

  it("shows business value badge — medium", () => {
    render(<PredictionOpportunitiesCard result={multipleOpps} />)
    expect(screen.getByText("Medium value")).toBeInTheDocument()
  })

  it("shows reason text", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    expect(screen.getByText(/100% complete data/)).toBeInTheDocument()
  })

  it("shows example question", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    expect(screen.getByText(/Can you predict the Revenue/)).toBeInTheDocument()
  })

  it("shows feasibility score number", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    expect(screen.getByText("90")).toBeInTheDocument()
  })

  it("renders all three opportunities", () => {
    render(<PredictionOpportunitiesCard result={multipleOpps} />)
    expect(screen.getByText(/#1 revenue/)).toBeInTheDocument()
    expect(screen.getByText(/#2 churn/)).toBeInTheDocument()
    expect(screen.getByText(/#3 units/)).toBeInTheDocument()
  })
})

describe("PredictionOpportunitiesCard — accessibility", () => {
  it("has accessible figure with aria-label", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    expect(
      screen.getByRole("figure", {
        name: /Prediction opportunities: 1 targets? found/i,
      })
    ).toBeInTheDocument()
  })

  it("icon is aria-hidden", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    const icon = screen.getByText("🎯")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })
})

describe("PredictionOpportunitiesCard — empty state", () => {
  it("returns null for empty opportunities", () => {
    const { container } = render(<PredictionOpportunitiesCard result={emptyResult} />)
    expect(container.firstChild).toBeNull()
  })
})

describe("PredictionOpportunitiesCard — onSelectTarget callback", () => {
  it("shows Set target button when callback provided", () => {
    const onSelect = jest.fn()
    render(
      <PredictionOpportunitiesCard
        result={singleOpp}
        onSelectTarget={onSelect}
      />
    )
    expect(screen.getByRole("button", { name: /Set target/i })).toBeInTheDocument()
  })

  it("calls callback with column name when button clicked", () => {
    const onSelect = jest.fn()
    render(
      <PredictionOpportunitiesCard
        result={singleOpp}
        onSelectTarget={onSelect}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: /Set target/i }))
    expect(onSelect).toHaveBeenCalledWith("revenue")
  })

  it("does not show Set target button without callback", () => {
    render(<PredictionOpportunitiesCard result={singleOpp} />)
    expect(screen.queryByRole("button", { name: /Set target/i })).not.toBeInTheDocument()
  })
})
