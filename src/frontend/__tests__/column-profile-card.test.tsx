/**
 * Tests for ColumnProfileCard component and related store/API plumbing.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"
import { ColumnProfileCard } from "@/components/data/column-profile-card"
import type { ColumnProfile } from "@/lib/types"
import { useAppStore } from "@/lib/store"

fetchMock.enableMocks()

// --- Fixtures ---------------------------------------------------------------

const numericProfile: ColumnProfile = {
  col_name: "revenue",
  col_type: "numeric",
  stats: {
    total_rows: 10,
    null_count: 0,
    null_pct: 0,
    unique_count: 10,
    min: 100.5,
    max: 300.1,
    mean: 199.95,
    median: 195.5,
    std: 62.3,
    p25: 150.7,
    p75: 250.9,
    skewness: 0.2,
  },
  distribution: {
    type: "histogram",
    bins: [100, 120, 140, 160, 180, 200, 220, 240, 260, 280],
    counts: [1, 0, 2, 1, 1, 2, 1, 0, 1, 1],
  },
  issues: [],
  summary: "Numeric column ranging from 100.5 to 300.1 with a mean of 199.95; no missing values.",
}

const categoricalProfile: ColumnProfile = {
  col_name: "region",
  col_type: "categorical",
  stats: {
    total_rows: 10,
    null_count: 0,
    null_pct: 0,
    unique_count: 3,
    most_common: "East",
    most_common_pct: 40,
    top_categories: [
      { label: "East", count: 4 },
      { label: "West", count: 4 },
      { label: "North", count: 2 },
    ],
  },
  distribution: {
    type: "bar",
    labels: ["East", "West", "North"],
    counts: [4, 4, 2],
  },
  issues: [],
  summary: "Categorical column with 3 unique values; most common is 'East' (40%); no missing values.",
}

const profileWithIssues: ColumnProfile = {
  col_name: "price",
  col_type: "numeric",
  stats: {
    total_rows: 10,
    null_count: 7,
    null_pct: 70,
    unique_count: 3,
    mean: 50,
    median: 50,
    std: 10,
    min: 40,
    max: 60,
    p25: 45,
    p75: 55,
  },
  distribution: {
    type: "histogram",
    bins: [40, 50, 60],
    counts: [1, 1, 1],
  },
  issues: [
    {
      type: "high_null_rate",
      severity: "critical",
      message: "70% of values are missing — consider filling or dropping",
    },
  ],
  summary: "Numeric column; 70% missing. ⚠️ 70% of values are missing — consider filling or dropping.",
}

// --- Tests ------------------------------------------------------------------

describe("ColumnProfileCard — numeric column", () => {
  it("renders column name", () => {
    render(<ColumnProfileCard profile={numericProfile} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("renders Numeric type badge", () => {
    render(<ColumnProfileCard profile={numericProfile} />)
    expect(screen.getAllByText(/numeric/i).length).toBeGreaterThan(0)
  })

  it("renders stat chips: Rows, Unique, Missing", () => {
    render(<ColumnProfileCard profile={numericProfile} />)
    expect(screen.getAllByText(/rows/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/unique/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/missing/i).length).toBeGreaterThan(0)
  })

  it("renders mean stat", () => {
    render(<ColumnProfileCard profile={numericProfile} />)
    expect(screen.getAllByText(/mean/i).length).toBeGreaterThan(0)
  })

  it("renders Distribution heading", () => {
    render(<ColumnProfileCard profile={numericProfile} />)
    expect(screen.getByText(/distribution/i)).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<ColumnProfileCard profile={numericProfile} />)
    expect(screen.getByText(/numeric column ranging from/i)).toBeInTheDocument()
  })

  it("does not render issues section when clean", () => {
    render(<ColumnProfileCard profile={numericProfile} />)
    expect(screen.queryByText(/missing.*consider/i)).not.toBeInTheDocument()
  })
})

describe("ColumnProfileCard — categorical column", () => {
  it("renders column name", () => {
    render(<ColumnProfileCard profile={categoricalProfile} />)
    expect(screen.getByText("region")).toBeInTheDocument()
  })

  it("renders Categorical type badge", () => {
    render(<ColumnProfileCard profile={categoricalProfile} />)
    expect(screen.getAllByText(/categorical/i).length).toBeGreaterThan(0)
  })

  it("renders Top Categories heading", () => {
    render(<ColumnProfileCard profile={categoricalProfile} />)
    expect(screen.getByText(/top categories/i)).toBeInTheDocument()
  })

  it("renders category labels from distribution", () => {
    render(<ColumnProfileCard profile={categoricalProfile} />)
    expect(screen.getAllByText("East").length).toBeGreaterThan(0)
    expect(screen.getAllByText("West").length).toBeGreaterThan(0)
  })
})

describe("ColumnProfileCard — issues", () => {
  it("renders critical issue message", () => {
    render(<ColumnProfileCard profile={profileWithIssues} />)
    expect(screen.getAllByText(/70% of values are missing/i).length).toBeGreaterThan(0)
  })

  it("renders issue icon for critical severity", () => {
    render(<ColumnProfileCard profile={profileWithIssues} />)
    expect(screen.getAllByText("✗").length).toBeGreaterThan(0)
  })
})

// --- Store plumbing ---------------------------------------------------------

describe("store.attachColumnProfileToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "tell me about revenue", timestamp: new Date().toISOString() },
        { role: "assistant", content: "Here is the profile...", timestamp: new Date().toISOString() },
      ],
    })
  })

  it("attaches column_profile to last assistant message", () => {
    const { attachColumnProfileToLastMessage } = useAppStore.getState()
    attachColumnProfileToLastMessage(numericProfile)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.column_profile).toEqual(numericProfile)
  })
})

// --- API plumbing -----------------------------------------------------------

import { api } from "@/lib/api"

describe("api.data.getColumnProfile", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
  })

  it("fetches the column profile", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(numericProfile))
    const result = await api.data.getColumnProfile("dataset-123", "revenue")
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/data/dataset-123/column-profile?col=revenue")
    )
    expect(result.col_name).toBe("revenue")
  })

  it("throws on non-ok response", async () => {
    fetchMock.mockResponseOnce("Not Found", { status: 404 })
    await expect(api.data.getColumnProfile("bad-id", "revenue")).rejects.toThrow()
  })
})
