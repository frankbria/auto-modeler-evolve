import { render, screen } from "@testing-library/react"
import { DataQualityImpactCard } from "@/components/models/data-quality-impact-card"
import type { DataQualityImpactResult } from "@/lib/types"

const BASE_RESULT: DataQualityImpactResult = {
  run_id: "run-001",
  algorithm: "random_forest_regressor",
  target_column: "revenue",
  problem_type: "regression",
  n_total: 200,
  n_outliers: 10,
  outlier_pct: 5.0,
  baseline_score: 0.78,
  clean_score: 0.84,
  delta: 0.06,
  metric_key: "r2",
  metric_label: "R²",
  verdict: "worthwhile",
  improvement: true,
  recommendation: "Removing the 10 outlier rows improved R² by +0.060.",
  summary: "Full dataset (200 rows): R² = 0.780. Without 10 outliers (190 rows): R² = 0.840 (+0.060).",
}

describe("DataQualityImpactCard", () => {
  it("renders without crashing", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByTestId("data-quality-impact-card")).toBeInTheDocument()
  })

  it("shows the heading", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByText("Data Quality Impact")).toBeInTheDocument()
  })

  it("shows the verdict badge — worthwhile", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByText("Worthwhile")).toBeInTheDocument()
  })

  it("shows the algorithm name", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByText("random_forest_regressor")).toBeInTheDocument()
  })

  it("shows metric label badge", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByText("R²")).toBeInTheDocument()
  })

  it("shows baseline score", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByTestId("baseline-score")).toHaveTextContent("0.780")
  })

  it("shows clean score", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByTestId("clean-score")).toHaveTextContent("0.840")
  })

  it("shows positive delta", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByTestId("delta-score")).toHaveTextContent("+0.060")
  })

  it("shows outlier counts", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByText(/10 of 200 rows flagged/)).toBeInTheDocument()
  })

  it("shows recommendation text", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByText(/Removing the 10 outlier rows improved/)).toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    expect(screen.getByTestId("quality-impact-summary")).toBeInTheDocument()
  })

  it("renders with marginal verdict", () => {
    const result: DataQualityImpactResult = {
      ...BASE_RESULT,
      verdict: "marginal",
      delta: 0.02,
      improvement: true,
    }
    render(<DataQualityImpactCard result={result} />)
    expect(screen.getByText("Marginal gain")).toBeInTheDocument()
  })

  it("renders with no_benefit verdict", () => {
    const result: DataQualityImpactResult = {
      ...BASE_RESULT,
      verdict: "no_benefit",
      delta: 0.001,
      improvement: false,
    }
    render(<DataQualityImpactCard result={result} />)
    expect(screen.getByText("No benefit")).toBeInTheDocument()
  })

  it("renders with harmful verdict", () => {
    const result: DataQualityImpactResult = {
      ...BASE_RESULT,
      verdict: "harmful",
      delta: -0.08,
      improvement: false,
    }
    render(<DataQualityImpactCard result={result} />)
    expect(screen.getByText("Keep outliers")).toBeInTheDocument()
  })

  it("shows accuracy as percentage for classification", () => {
    const result: DataQualityImpactResult = {
      ...BASE_RESULT,
      problem_type: "classification",
      metric_key: "accuracy",
      metric_label: "Accuracy",
      baseline_score: 0.82,
      clean_score: 0.87,
      delta: 0.05,
    }
    render(<DataQualityImpactCard result={result} />)
    expect(screen.getByTestId("baseline-score")).toHaveTextContent("82.0%")
    expect(screen.getByTestId("clean-score")).toHaveTextContent("87.0%")
  })

  it("has accessible aria-label on card", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    const card = screen.getByTestId("data-quality-impact-card")
    expect(card).toHaveAttribute("aria-label")
    expect(card.getAttribute("aria-label")).toContain("worthwhile")
  })

  it("has progressbar aria role on outlier bar", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    const bar = screen.getByRole("progressbar")
    expect(bar).toBeInTheDocument()
    expect(bar).toHaveAttribute("aria-valuenow", "5")
  })

  it("has sr-only figcaption", () => {
    render(<DataQualityImpactCard result={BASE_RESULT} />)
    const caption = document.querySelector("figcaption")
    expect(caption).not.toBeNull()
    expect(caption?.className).toContain("sr-only")
  })

  it("returns null for falsy result", () => {
    const { container } = render(
      <DataQualityImpactCard result={null as unknown as DataQualityImpactResult} />
    )
    expect(container.firstChild).toBeNull()
  })
})
