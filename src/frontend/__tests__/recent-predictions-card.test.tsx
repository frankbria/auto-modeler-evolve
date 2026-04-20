import React from "react"
import { render, screen } from "@testing-library/react"
import { RecentPredictionsCard } from "@/components/deploy/recent-predictions-card"
import type { RecentPredictionsResult } from "@/lib/types"

const EMPTY_RESULT: RecentPredictionsResult = {
  deployment_id: "dep-001",
  n_shown: 0,
  total_all_time: 0,
  predictions: [],
  export_url: "/api/deploy/dep-001/export-prediction-logs",
  summary: "No predictions yet.",
}

const ROW_BASE = {
  id: "abc12345",
  created_at: new Date(Date.now() - 5 * 60_000).toISOString(),
  prediction: "540.00",
  confidence: 82,
  response_ms: 45,
  input_summary: [
    { key: "units", value: "10" },
    { key: "region", value: "East" },
  ],
  ab_variant: null,
}

const RESULT_WITH_ROWS: RecentPredictionsResult = {
  deployment_id: "dep-001",
  n_shown: 2,
  total_all_time: 2,
  predictions: [
    ROW_BASE,
    {
      ...ROW_BASE,
      id: "def67890",
      created_at: new Date(Date.now() - 120 * 60_000).toISOString(),
      prediction: "1500000",
      confidence: 55,
      response_ms: 620,
      input_summary: [{ key: "x", value: "longvaluethattruncates" }],
      ab_variant: "challenger",
    },
  ],
  export_url: "/api/deploy/dep-001/export-prediction-logs",
  summary: "Showing 2 of 2 predictions.",
}

describe("RecentPredictionsCard", () => {
  describe("structure", () => {
    it("renders with aria-label", () => {
      render(<RecentPredictionsCard result={EMPTY_RESULT} />)
      expect(screen.getByLabelText(/recent predictions table/i)).toBeInTheDocument()
    })

    it("renders heading text", () => {
      render(<RecentPredictionsCard result={EMPTY_RESULT} />)
      expect(screen.getByText("Recent Predictions")).toBeInTheDocument()
    })
  })

  describe("empty state", () => {
    it("shows empty message when total_all_time is 0", () => {
      render(<RecentPredictionsCard result={EMPTY_RESULT} />)
      expect(screen.getByText(/no predictions recorded yet/i)).toBeInTheDocument()
    })

    it("does not render table in empty state", () => {
      render(<RecentPredictionsCard result={EMPTY_RESULT} />)
      expect(screen.queryByRole("table")).not.toBeInTheDocument()
    })
  })

  describe("badges", () => {
    it("shows n_shown badge", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByText("2 shown")).toBeInTheDocument()
    })

    it("shows total badge when total > n_shown", () => {
      const result: RecentPredictionsResult = {
        ...RESULT_WITH_ROWS,
        n_shown: 2,
        total_all_time: 50,
      }
      render(<RecentPredictionsCard result={result} />)
      expect(screen.getByText("50 total")).toBeInTheDocument()
    })

    it("hides total badge when total equals n_shown", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.queryByText("2 total")).not.toBeInTheDocument()
    })
  })

  describe("table rendering", () => {
    it("renders table with aria-label", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByRole("table", { name: /recent prediction log/i })).toBeInTheDocument()
    })

    it("renders all column headers", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByText("Time")).toBeInTheDocument()
      expect(screen.getByText("Prediction")).toBeInTheDocument()
      expect(screen.getByText("Confidence")).toBeInTheDocument()
      expect(screen.getByText("Latency")).toBeInTheDocument()
      expect(screen.getByText("Key Inputs")).toBeInTheDocument()
    })

    it("renders correct number of rows", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      const rows = screen.getAllByRole("row")
      expect(rows).toHaveLength(3) // 1 header + 2 data
    })
  })

  describe("prediction formatting", () => {
    it("renders numeric prediction", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByText("540")).toBeInTheDocument()
    })

    it("formats large numbers with M suffix", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByText("1.5M")).toBeInTheDocument()
    })
  })

  describe("confidence display", () => {
    it("renders confidence percentage", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByLabelText("82% confidence")).toBeInTheDocument()
      expect(screen.getByLabelText("55% confidence")).toBeInTheDocument()
    })

    it("applies emerald class for high confidence", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      const el = screen.getByLabelText("82% confidence")
      expect(el.className).toContain("emerald")
    })

    it("applies rose class for low confidence", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      const el = screen.getByLabelText("55% confidence")
      expect(el.className).toContain("rose")
    })

    it("renders dash when confidence is null", () => {
      const result: RecentPredictionsResult = {
        ...RESULT_WITH_ROWS,
        predictions: [{ ...ROW_BASE, confidence: null }],
        n_shown: 1,
        total_all_time: 1,
      }
      render(<RecentPredictionsCard result={result} />)
      const dashes = screen.getAllByText("—")
      expect(dashes.length).toBeGreaterThanOrEqual(1)
    })
  })

  describe("latency badge", () => {
    it("renders ms for fast response", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByLabelText("45ms response time")).toBeInTheDocument()
    })

    it("applies emerald class for fast latency (<100ms)", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      const badge = screen.getByLabelText("45ms response time")
      expect(badge.className).toContain("emerald")
    })

    it("applies rose class for slow latency (>=500ms)", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      const badge = screen.getByLabelText("620ms response time")
      expect(badge.className).toContain("rose")
    })
  })

  describe("input summary badges", () => {
    it("renders key=value badges", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByTitle("units=10")).toBeInTheDocument()
      expect(screen.getByTitle("region=East")).toBeInTheDocument()
    })

    it("truncates long values to 8 chars with ellipsis", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByText(/x=longvalu…/)).toBeInTheDocument()
    })
  })

  describe("relative time", () => {
    it("shows 'just now' for very recent predictions", () => {
      const result: RecentPredictionsResult = {
        ...RESULT_WITH_ROWS,
        predictions: [{ ...ROW_BASE, created_at: new Date(Date.now() - 30_000).toISOString() }],
        n_shown: 1,
        total_all_time: 1,
      }
      render(<RecentPredictionsCard result={result} />)
      expect(screen.getByText("just now")).toBeInTheDocument()
    })

    it("shows 'Xm ago' for predictions a few minutes old", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByText("5m ago")).toBeInTheDocument()
    })

    it("shows 'Xh ago' for predictions hours old", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByText("2h ago")).toBeInTheDocument()
    })
  })

  describe("A/B variant badge", () => {
    it("shows B badge for challenger variant", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      const badge = screen.getByLabelText("A/B variant: challenger")
      expect(badge).toBeInTheDocument()
      expect(badge.textContent).toBe("B")
    })

    it("does not show variant badge when ab_variant is null", () => {
      const result: RecentPredictionsResult = {
        ...RESULT_WITH_ROWS,
        predictions: [ROW_BASE],
        n_shown: 1,
        total_all_time: 1,
      }
      render(<RecentPredictionsCard result={result} />)
      expect(screen.queryByLabelText(/A\/B variant/)).not.toBeInTheDocument()
    })
  })

  describe("export link", () => {
    it("renders download CSV link", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      const link = screen.getByRole("link", { name: /download all prediction logs as csv/i })
      expect(link).toBeInTheDocument()
      expect(link).toHaveAttribute("href", "/api/deploy/dep-001/export-prediction-logs")
      expect(link).toHaveAttribute("download")
    })
  })

  describe("summary line", () => {
    it("renders summary text", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByText("Showing 2 of 2 predictions.")).toBeInTheDocument()
    })
  })

  describe("accessibility", () => {
    it("has sr-only figcaption describing the table", () => {
      render(<RecentPredictionsCard result={RESULT_WITH_ROWS} />)
      expect(screen.getByText(/recent predictions table: showing 2 of 2/i)).toBeInTheDocument()
    })

    it("has sr-only figcaption for empty state", () => {
      render(<RecentPredictionsCard result={EMPTY_RESULT} />)
      expect(screen.getByText(/recent predictions: no records available yet/i)).toBeInTheDocument()
    })
  })
})
