import { render, screen } from "@testing-library/react"
import { InlinePredictionCard } from "@/components/models/inline-prediction-card"
import type { InlinePredictionResult } from "@/lib/types"

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

function makeRegressionResult(
  overrides?: Partial<InlinePredictionResult>
): InlinePredictionResult {
  return {
    deployment_id: "dep-1",
    target_column: "revenue",
    prediction: 1500.5,
    probabilities: undefined,
    confidence_interval: { lower: 1200, upper: 1800 },
    confidence: null,
    provided_features: { Region: "East", Units: 100 },
    defaults_used_count: 1,
    total_features: 3,
    summary: "Predicted revenue: 1,501 (95% interval: 1,200 – 1,800). 1 feature used training-data averages.",
    problem_type: "regression",
    ...overrides,
  }
}

function makeClassificationResult(
  overrides?: Partial<InlinePredictionResult>
): InlinePredictionResult {
  return {
    deployment_id: "dep-2",
    target_column: "churn",
    prediction: "Yes",
    probabilities: { Yes: 0.72, No: 0.28 },
    confidence_interval: null,
    confidence: 0.72,
    provided_features: { Region: "West" },
    defaults_used_count: 2,
    total_features: 3,
    summary: "Predicted churn: Yes (72% probability). 2 features used training-data averages.",
    problem_type: "classification",
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Rendering tests — regression
// ---------------------------------------------------------------------------

test("renders card heading for regression", () => {
  render(<InlinePredictionCard result={makeRegressionResult()} />)
  expect(screen.getByText(/Prediction Result/i)).toBeInTheDocument()
})

test("renders target column name", () => {
  render(<InlinePredictionCard result={makeRegressionResult()} />)
  expect(screen.getByText(/revenue/i)).toBeInTheDocument()
})

test("renders prediction value", () => {
  render(<InlinePredictionCard result={makeRegressionResult()} />)
  // 1500.5 → formatted as 1.5k
  expect(screen.getByText(/1\.5k/)).toBeInTheDocument()
})

test("renders 95% confidence interval", () => {
  render(<InlinePredictionCard result={makeRegressionResult()} />)
  expect(screen.getByText(/95% interval/i)).toBeInTheDocument()
})

test("renders provided feature badges", () => {
  render(<InlinePredictionCard result={makeRegressionResult()} />)
  expect(screen.getByText(/Region=East/i)).toBeInTheDocument()
  expect(screen.getByText(/Units=100/i)).toBeInTheDocument()
})

test("renders defaults-used note when defaults_used_count > 0", () => {
  render(<InlinePredictionCard result={makeRegressionResult()} />)
  expect(screen.getByText(/1 feature.*training-data averages/i)).toBeInTheDocument()
})

test("does not render defaults note when defaults_used_count is 0", () => {
  render(
    <InlinePredictionCard
      result={makeRegressionResult({ defaults_used_count: 0 })}
    />
  )
  expect(screen.queryByText(/training-data averages/i)).not.toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Rendering tests — classification
// ---------------------------------------------------------------------------

test("renders classification probability bars", () => {
  render(<InlinePredictionCard result={makeClassificationResult()} />)
  expect(screen.getByText("Yes")).toBeInTheDocument()
  expect(screen.getByText("No")).toBeInTheDocument()
})

test("shows percentage for classification class probabilities", () => {
  render(<InlinePredictionCard result={makeClassificationResult()} />)
  expect(screen.getByText("72%")).toBeInTheDocument()
  expect(screen.getByText("28%")).toBeInTheDocument()
})

test("renders target column for classification", () => {
  render(<InlinePredictionCard result={makeClassificationResult()} />)
  expect(screen.getByText(/churn/i)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Accessibility
// ---------------------------------------------------------------------------

test("renders figure with aria-label", () => {
  render(<InlinePredictionCard result={makeRegressionResult()} />)
  expect(
    screen.getByRole("figure", {
      name: /Inline prediction result for revenue/i,
    })
  ).toBeInTheDocument()
})

test("prediction icon is aria-hidden", () => {
  render(<InlinePredictionCard result={makeRegressionResult()} />)
  const icon = screen.getByText("🔮")
  expect(icon).toHaveAttribute("aria-hidden", "true")
})

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

test("renders no confidence interval when null", () => {
  render(
    <InlinePredictionCard
      result={makeRegressionResult({ confidence_interval: null })}
    />
  )
  expect(screen.queryByText(/95% interval/i)).not.toBeInTheDocument()
})

test("shows confidence badge when no interval but confidence present", () => {
  render(
    <InlinePredictionCard
      result={makeRegressionResult({
        confidence_interval: null,
        confidence: 0.88,
      })}
    />
  )
  expect(screen.getByText(/Confidence: 88%/i)).toBeInTheDocument()
})

test("renders plural defaults note for multiple defaults", () => {
  render(
    <InlinePredictionCard
      result={makeRegressionResult({ defaults_used_count: 3 })}
    />
  )
  expect(screen.getByText(/3 features.*training-data averages/i)).toBeInTheDocument()
})
