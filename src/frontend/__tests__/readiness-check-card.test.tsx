/**
 * Tests for ReadinessCheckCard component.
 */
import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { ReadinessCheckCard } from "@/components/data/readiness-check-card"
import type { DataReadinessResult } from "@/lib/types"

// --- Mock the API client -------------------------------------------------
jest.mock("@/lib/api", () => ({
  api: {
    data: {
      getReadinessCheck: jest.fn(),
    },
  },
}))

import { api } from "@/lib/api"
const mockGetReadinessCheck = api.data.getReadinessCheck as jest.MockedFunction<
  typeof api.data.getReadinessCheck
>

// --- Test fixtures -------------------------------------------------------

const goodResult: DataReadinessResult = {
  dataset_id: "ds-1",
  score: 92,
  grade: "A",
  status: "ready",
  summary: "Your data looks great — ready for modeling!",
  components: [
    {
      name: "Row Count",
      score: 25,
      max_score: 25,
      status: "good",
      detail: "1,250 rows — large dataset.",
    },
    {
      name: "Missing Values",
      score: 25,
      max_score: 25,
      status: "good",
      detail: "No missing values.",
    },
    {
      name: "Duplicate Rows",
      score: 20,
      max_score: 20,
      status: "good",
      detail: "No duplicate rows.",
    },
    {
      name: "Feature Diversity",
      score: 15,
      max_score: 15,
      status: "good",
      detail: "Good mix: 3 numeric + 2 categorical columns.",
    },
    {
      name: "Data Type Quality",
      score: 15,
      max_score: 15,
      status: "good",
      detail: "Column types look clean.",
    },
  ],
  recommendations: [],
}

const warningResult: DataReadinessResult = {
  dataset_id: "ds-2",
  score: 72,
  grade: "C",
  status: "needs_attention",
  summary: "Your data has 2 minor issues worth reviewing.",
  components: [
    {
      name: "Row Count",
      score: 12,
      max_score: 25,
      status: "warning",
      detail: "150 rows — usable but models may overfit.",
    },
    {
      name: "Missing Values",
      score: 15,
      max_score: 25,
      status: "warning",
      detail: "3 columns have missing values.",
      recommendation: "Fill missing values with median.",
    },
    {
      name: "Duplicate Rows",
      score: 20,
      max_score: 20,
      status: "good",
      detail: "No duplicate rows.",
    },
    {
      name: "Feature Diversity",
      score: 15,
      max_score: 15,
      status: "good",
      detail: "Good mix.",
    },
    {
      name: "Data Type Quality",
      score: 10,
      max_score: 15,
      status: "warning",
      detail: "High-cardinality text column detected.",
    },
  ],
  recommendations: ["Fill missing values with median.", "Consider dropping high-cardinality column."],
}

const criticalResult: DataReadinessResult = {
  dataset_id: "ds-3",
  score: 30,
  grade: "F",
  status: "not_ready",
  summary: "Your data has 1 critical issue that should be fixed before training.",
  components: [
    {
      name: "Row Count",
      score: 0,
      max_score: 25,
      status: "critical",
      detail: "30 rows — needs at least 50.",
      recommendation: "Collect more data.",
    },
    {
      name: "Missing Values",
      score: 5,
      max_score: 25,
      status: "critical",
      detail: "50% missing in revenue column.",
    },
    {
      name: "Duplicate Rows",
      score: 20,
      max_score: 20,
      status: "good",
      detail: "No duplicates.",
    },
    {
      name: "Feature Diversity",
      score: 0,
      max_score: 15,
      status: "critical",
      detail: "Only 1 column.",
    },
    {
      name: "Data Type Quality",
      score: 5,
      max_score: 15,
      status: "critical",
      detail: "All-null column detected.",
    },
  ],
  recommendations: ["Collect more data.", "Drop all-null columns."],
}

// -------------------------------------------------------------------------

describe("ReadinessCheckCard (no result — shows Check button)", () => {
  it("renders check button when no result provided", () => {
    render(<ReadinessCheckCard datasetId="ds-1" />)
    expect(screen.getByTestId("readiness-check-card")).toBeInTheDocument()
    expect(screen.getByTestId("check-readiness-btn")).toBeInTheDocument()
    expect(screen.getByText("Check Readiness")).toBeInTheDocument()
  })

  it("disables button when no datasetId", () => {
    render(<ReadinessCheckCard />)
    const btn = screen.getByTestId("check-readiness-btn")
    expect(btn).toBeDisabled()
  })

  it("calls API and shows result on button click", async () => {
    mockGetReadinessCheck.mockResolvedValueOnce(goodResult)
    render(<ReadinessCheckCard datasetId="ds-1" />)
    fireEvent.click(screen.getByTestId("check-readiness-btn"))
    await waitFor(() => {
      expect(screen.getByTestId("readiness-grade")).toBeInTheDocument()
    })
    expect(screen.getByTestId("readiness-grade")).toHaveTextContent("A")
    expect(screen.getByTestId("readiness-score")).toHaveTextContent("92/100")
  })

  it("shows error message when API fails", async () => {
    mockGetReadinessCheck.mockRejectedValueOnce(new Error("Network error"))
    render(<ReadinessCheckCard datasetId="ds-1" />)
    fireEvent.click(screen.getByTestId("check-readiness-btn"))
    await waitFor(() => {
      expect(screen.getByText("Could not compute readiness check.")).toBeInTheDocument()
    })
  })
})

describe("ReadinessCheckCard (result provided — pre-loaded)", () => {
  it("renders grade and score for a good dataset", () => {
    render(<ReadinessCheckCard result={goodResult} />)
    expect(screen.getByTestId("readiness-grade")).toHaveTextContent("A")
    expect(screen.getByTestId("readiness-score")).toHaveTextContent("92/100")
    expect(screen.getByTestId("readiness-status-badge")).toHaveTextContent("Ready to Train")
  })

  it("renders all 5 components", () => {
    render(<ReadinessCheckCard result={goodResult} />)
    const components = screen.getAllByTestId("readiness-component")
    expect(components).toHaveLength(5)
  })

  it("renders progress bars for core components", () => {
    render(<ReadinessCheckCard result={goodResult} />)
    const bars = screen.getAllByTestId("component-progress-bar")
    expect(bars.length).toBeGreaterThanOrEqual(5)
  })

  it("does not show recommendations when empty", () => {
    render(<ReadinessCheckCard result={goodResult} />)
    expect(screen.queryByTestId("readiness-recommendations")).not.toBeInTheDocument()
  })

  it("shows warning status badge for needs_attention", () => {
    render(<ReadinessCheckCard result={warningResult} />)
    expect(screen.getByTestId("readiness-status-badge")).toHaveTextContent("Needs Attention")
  })

  it("shows critical status badge for not_ready", () => {
    render(<ReadinessCheckCard result={criticalResult} />)
    expect(screen.getByTestId("readiness-status-badge")).toHaveTextContent("Not Ready")
  })

  it("renders recommendations list when present", () => {
    render(<ReadinessCheckCard result={warningResult} />)
    const recs = screen.getByTestId("readiness-recommendations")
    expect(recs).toBeInTheDocument()
    expect(recs.querySelectorAll("li")).toHaveLength(2)
  })

  it("renders summary text", () => {
    render(<ReadinessCheckCard result={goodResult} />)
    expect(
      screen.getByText("Your data looks great — ready for modeling!")
    ).toBeInTheDocument()
  })

  it("renders component detail text", () => {
    render(<ReadinessCheckCard result={goodResult} />)
    expect(screen.getByText("No missing values.")).toBeInTheDocument()
  })

  it("shows advisory label for advisory components", () => {
    const resultWithAdvisory: DataReadinessResult = {
      ...goodResult,
      components: [
        ...goodResult.components,
        {
          name: "Class Balance (Advisory)",
          score: 0,
          max_score: 0,
          status: "good",
          detail: "Class balance looks reasonable (60% / 40%).",
          advisory: true,
        },
      ],
    }
    render(<ReadinessCheckCard result={resultWithAdvisory} />)
    expect(screen.getByText("(advisory)")).toBeInTheDocument()
  })
})
