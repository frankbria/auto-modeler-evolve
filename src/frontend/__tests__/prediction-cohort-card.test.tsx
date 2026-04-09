/**
 * Tests for PredictionCohortCard component and CSV download on RankedPredictionsCard.
 *
 * Covers:
 *  1.  Renders figure with correct aria-label
 *  2.  Renders 🔍 icon (aria-hidden)
 *  3.  Shows target column in heading
 *  4.  Shows "Highest" direction badge
 *  5.  Shows "Lowest" direction badge for lowest direction
 *  6.  Shows n of total_scored badge
 *  7.  Shows characterization text
 *  8.  Shows categorical breakdown section when categorical_profile present
 *  9.  Shows categorical column name
 * 10.  Shows numeric averages section when numeric_profile present
 * 11.  Shows numeric column name
 * 12.  Shows "No additional" message when both profiles empty
 * 13.  Store: attachPredictionCohortToLastMessage attaches to last assistant message
 * 14.  Store: does not attach to user message
 * 15.  Store: does not crash when messages list is empty
 * 16.  RankedPredictionsCard: renders "Download CSV" button
 * 17.  RankedPredictionsCard: download CSV button has aria-label
 * 18.  RankedPredictionsCard: download click creates blob and triggers download
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { PredictionCohortCard } from "@/components/deploy/prediction-cohort-card"
import { RankedPredictionsCard } from "@/components/deploy/ranked-predictions-card"
import type { PredictionCohortResult, RankedPredictionsResult, ChatMessage } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const categoricalProfile = [
  {
    column: "region",
    categories: [
      { value: "East", top_pct: 70.0, overall_pct: 40.0, ratio: 1.75 },
      { value: "West", top_pct: 20.0, overall_pct: 35.0, ratio: 0.57 },
    ],
    dominant: "East",
    dominant_top_pct: 70.0,
  },
]

const numericProfile = [
  {
    column: "units",
    top_mean: 45.0,
    overall_mean: 25.0,
    ratio: 1.8,
    direction: "higher" as const,
  },
]

const cohortResult: PredictionCohortResult = {
  target_column: "revenue",
  problem_type: "regression",
  n: 20,
  direction: "highest",
  total_scored: 200,
  categorical_profile: categoricalProfile,
  numeric_profile: numericProfile,
  characterization: "The 20 highest-scoring revenue predictions: 70% have region = 'East'.",
}

const regressionRanked: RankedPredictionsResult = {
  problem_type: "regression",
  target_column: "revenue",
  direction: "highest",
  n: 2,
  total_scored: 50,
  rows: [
    {
      rank: 1,
      row_index: 0,
      score: 1500.0,
      feature_values: { region: "East", units: 50 },
      prediction: 1500.0,
    },
    {
      rank: 2,
      row_index: 1,
      score: 1200.0,
      feature_values: { region: "West", units: 40 },
      prediction: 1200.0,
    },
  ],
  summary: "Scored 50 rows. Top predicted revenue: 1500.",
  class_names: null,
}

// ---------------------------------------------------------------------------
// PredictionCohortCard rendering tests
// ---------------------------------------------------------------------------

describe("PredictionCohortCard", () => {
  test("renders figure with correct aria-label", () => {
    render(<PredictionCohortCard result={cohortResult} />)
    expect(
      screen.getByRole("figure", {
        name: /cohort profile.*top 20.*highest.*revenue/i,
      })
    ).toBeInTheDocument()
  })

  test("renders 🔍 icon aria-hidden", () => {
    render(<PredictionCohortCard result={cohortResult} />)
    const icon = screen.getByText("🔍")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  test("shows target column in heading", () => {
    render(<PredictionCohortCard result={cohortResult} />)
    expect(screen.getByRole("heading", { name: /revenue/i })).toBeInTheDocument()
  })

  test("shows Highest direction badge", () => {
    render(<PredictionCohortCard result={cohortResult} />)
    expect(screen.getByText("Highest")).toBeInTheDocument()
  })

  test("shows Lowest direction badge for lowest direction", () => {
    render(
      <PredictionCohortCard result={{ ...cohortResult, direction: "lowest" }} />
    )
    expect(screen.getByText("Lowest")).toBeInTheDocument()
  })

  test("shows n of total_scored in badge", () => {
    render(<PredictionCohortCard result={cohortResult} />)
    expect(screen.getByText(/20 of 200/)).toBeInTheDocument()
  })

  test("shows characterization text", () => {
    render(<PredictionCohortCard result={cohortResult} />)
    expect(
      screen.getByText(/highest-scoring revenue predictions/i)
    ).toBeInTheDocument()
  })

  test("shows categorical breakdown section when profile present", () => {
    render(<PredictionCohortCard result={cohortResult} />)
    expect(screen.getByText(/categorical breakdown/i)).toBeInTheDocument()
  })

  test("shows categorical column name", () => {
    render(<PredictionCohortCard result={cohortResult} />)
    // "region" appears as column header <p> element
    const regionElements = screen.getAllByText(/region/i)
    expect(regionElements.length).toBeGreaterThan(0)
  })

  test("shows numeric averages section when profile present", () => {
    render(<PredictionCohortCard result={cohortResult} />)
    expect(screen.getByText(/numeric averages/i)).toBeInTheDocument()
  })

  test("shows numeric column name", () => {
    render(<PredictionCohortCard result={cohortResult} />)
    expect(screen.getByText(/units/i)).toBeInTheDocument()
  })

  test("shows no-additional message when both profiles empty", () => {
    render(
      <PredictionCohortCard
        result={{ ...cohortResult, categorical_profile: [], numeric_profile: [] }}
      />
    )
    expect(screen.getByText(/no additional/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachPredictionCohortToLastMessage", () => {
  test("attaches cohort to last assistant message", () => {
    const { attachPredictionCohortToLastMessage } = useAppStore.getState()
    useAppStore.setState({
      messages: [
        { id: "1", role: "user", content: "hi" },
        { id: "2", role: "assistant", content: "hello" },
      ] as ChatMessage[],
    })
    attachPredictionCohortToLastMessage(cohortResult)
    const msgs = useAppStore.getState().messages
    expect((msgs[1] as ChatMessage & { prediction_cohort?: unknown }).prediction_cohort).toEqual(
      cohortResult
    )
  })

  test("does not attach to user message", () => {
    const { attachPredictionCohortToLastMessage } = useAppStore.getState()
    useAppStore.setState({
      messages: [{ id: "1", role: "user", content: "hi" }] as ChatMessage[],
    })
    attachPredictionCohortToLastMessage(cohortResult)
    const msgs = useAppStore.getState().messages
    expect((msgs[0] as ChatMessage & { prediction_cohort?: unknown }).prediction_cohort).toBeUndefined()
  })

  test("does not crash when messages list is empty", () => {
    const { attachPredictionCohortToLastMessage } = useAppStore.getState()
    useAppStore.setState({ messages: [] })
    expect(() => attachPredictionCohortToLastMessage(cohortResult)).not.toThrow()
  })
})

// ---------------------------------------------------------------------------
// CSV download button on RankedPredictionsCard
// ---------------------------------------------------------------------------

describe("RankedPredictionsCard CSV download", () => {
  let mockUrl: string
  let originalCreateObjectURL: typeof URL.createObjectURL
  let originalRevokeObjectURL: typeof URL.revokeObjectURL
  let clickedDownload: string

  beforeEach(() => {
    mockUrl = "blob:mock-url"
    originalCreateObjectURL = URL.createObjectURL
    originalRevokeObjectURL = URL.revokeObjectURL
    URL.createObjectURL = jest.fn(() => mockUrl)
    URL.revokeObjectURL = jest.fn()

    // Spy on anchor element click
    const originalCreateElement = document.createElement.bind(document)
    jest.spyOn(document, "createElement").mockImplementation((tag: string) => {
      if (tag === "a") {
        const a = originalCreateElement("a") as HTMLAnchorElement
        jest.spyOn(a, "click").mockImplementation(() => {
          clickedDownload = a.download
        })
        return a
      }
      return originalCreateElement(tag)
    })
  })

  afterEach(() => {
    URL.createObjectURL = originalCreateObjectURL
    URL.revokeObjectURL = originalRevokeObjectURL
    jest.restoreAllMocks()
  })

  test("renders Download CSV button", () => {
    render(<RankedPredictionsCard result={regressionRanked} />)
    expect(screen.getByRole("button", { name: /download.*csv/i })).toBeInTheDocument()
  })

  test("Download CSV button has correct aria-label", () => {
    render(<RankedPredictionsCard result={regressionRanked} />)
    const btn = screen.getByRole("button", { name: /download ranked predictions as csv/i })
    expect(btn).toBeInTheDocument()
  })

  test("clicking download creates blob and triggers anchor download", () => {
    render(<RankedPredictionsCard result={regressionRanked} />)
    const btn = screen.getByRole("button", { name: /download.*csv/i })
    fireEvent.click(btn)
    expect(URL.createObjectURL).toHaveBeenCalledWith(expect.any(Blob))
    expect(URL.revokeObjectURL).toHaveBeenCalledWith(mockUrl)
    expect(clickedDownload).toBe("revenue_ranked_predictions.csv")
  })
})
