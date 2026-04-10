import { render, screen } from "@testing-library/react"
import { MultiPredictionCard } from "@/components/deploy/multi-prediction-card"
import { useAppStore } from "@/lib/store"
import type { MultiPredictionResult } from "@/lib/types"

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

function makeRegressionResult(
  overrides?: Partial<MultiPredictionResult>
): MultiPredictionResult {
  return {
    deployment_id: "dep-1",
    target_column: "revenue",
    problem_type: "regression",
    rows: [
      {
        row_index: 1,
        provided_features: { Region: "East", Units: 100 },
        defaults_used_count: 1,
        prediction: 5000,
        probabilities: undefined,
        confidence: null,
        confidence_interval: null,
      },
      {
        row_index: 2,
        provided_features: { Region: "West", Units: 150 },
        defaults_used_count: 1,
        prediction: 7500,
        probabilities: undefined,
        confidence: null,
        confidence_interval: null,
      },
    ],
    summary: "2 predictions for revenue: range 5,000 – 7,500",
    ...overrides,
  }
}

function makeClassificationResult(
  overrides?: Partial<MultiPredictionResult>
): MultiPredictionResult {
  return {
    deployment_id: "dep-2",
    target_column: "churn",
    problem_type: "classification",
    rows: [
      {
        row_index: 1,
        provided_features: { Region: "East" },
        defaults_used_count: 2,
        prediction: "Yes",
        probabilities: { Yes: 0.82, No: 0.18 },
        confidence: 0.82,
        confidence_interval: null,
      },
      {
        row_index: 2,
        provided_features: { Region: "West" },
        defaults_used_count: 2,
        prediction: "No",
        probabilities: { Yes: 0.35, No: 0.65 },
        confidence: 0.65,
        confidence_interval: null,
      },
    ],
    summary: "2 predictions for churn: most common = No",
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Rendering tests — regression
// ---------------------------------------------------------------------------

test("renders card aria-label for regression", () => {
  render(<MultiPredictionCard result={makeRegressionResult()} />)
  expect(
    screen.getByRole("figure", { name: /multi-row prediction results for revenue/i })
  ).toBeInTheDocument()
})

test("renders scenario comparison heading", () => {
  render(<MultiPredictionCard result={makeRegressionResult()} />)
  expect(screen.getByText(/Scenario Comparison/i)).toBeInTheDocument()
})

test("renders target column name", () => {
  render(<MultiPredictionCard result={makeRegressionResult()} />)
  // Appears in header subtitle + table column header
  const elements = screen.getAllByText(/revenue/i)
  expect(elements.length).toBeGreaterThan(0)
})

test("renders scenario count badge", () => {
  render(<MultiPredictionCard result={makeRegressionResult()} />)
  expect(screen.getByText(/2 scenarios/i)).toBeInTheDocument()
})

test("renders regression badge", () => {
  render(<MultiPredictionCard result={makeRegressionResult()} />)
  expect(screen.getByText(/Regression/i)).toBeInTheDocument()
})

test("renders table with correct row count", () => {
  render(<MultiPredictionCard result={makeRegressionResult()} />)
  // Row indices 1 and 2
  expect(screen.getByText("1")).toBeInTheDocument()
  expect(screen.getByText("2")).toBeInTheDocument()
})

test("renders prediction values for regression", () => {
  render(<MultiPredictionCard result={makeRegressionResult()} />)
  // 5000 → "5.0k", 7500 → "7.5k"
  expect(screen.getByText("5.0k")).toBeInTheDocument()
  expect(screen.getByText("7.5k")).toBeInTheDocument()
})

test("renders feature columns as table headers", () => {
  render(<MultiPredictionCard result={makeRegressionResult()} />)
  expect(screen.getByText(/Region/i)).toBeInTheDocument()
  expect(screen.getByText(/Units/i)).toBeInTheDocument()
})

test("renders summary footer", () => {
  render(<MultiPredictionCard result={makeRegressionResult()} />)
  expect(screen.getByText(/2 predictions for revenue/i)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Rendering tests — classification
// ---------------------------------------------------------------------------

test("renders aria-label for classification", () => {
  render(<MultiPredictionCard result={makeClassificationResult()} />)
  expect(
    screen.getByRole("figure", { name: /multi-row prediction results for churn/i })
  ).toBeInTheDocument()
})

test("renders classification badge", () => {
  render(<MultiPredictionCard result={makeClassificationResult()} />)
  expect(screen.getByText(/Classification/i)).toBeInTheDocument()
})

test("renders top class and confidence for classification", () => {
  render(<MultiPredictionCard result={makeClassificationResult()} />)
  // Row 1: Yes (82%)
  expect(screen.getByText("Yes")).toBeInTheDocument()
  expect(screen.getByText("(82%)")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Store integration tests
// ---------------------------------------------------------------------------

test("store attachMultiPredictionToLastMessage attaches result", () => {
  const store = useAppStore.getState()

  store.addMessage({ role: "assistant", content: "Here are predictions." })
  store.attachMultiPredictionToLastMessage(makeRegressionResult())

  const messages = useAppStore.getState().messages
  const last = messages[messages.length - 1]
  expect(last.multi_prediction).toBeDefined()
  expect(last.multi_prediction?.target_column).toBe("revenue")
  expect(last.multi_prediction?.rows).toHaveLength(2)
})

test("store does not attach to user message", () => {
  const store = useAppStore.getState()

  store.addMessage({ role: "user", content: "batch predict" })
  store.attachMultiPredictionToLastMessage(makeRegressionResult())

  const messages = useAppStore.getState().messages
  const last = messages[messages.length - 1]
  expect(last.role).toBe("user")
  expect(last.multi_prediction).toBeUndefined()
})

test("store empty messages does not throw", () => {
  useAppStore.setState({ messages: [] })

  expect(() => {
    useAppStore.getState().attachMultiPredictionToLastMessage(makeRegressionResult())
  }).not.toThrow()
})
