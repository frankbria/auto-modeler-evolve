/**
 * Tests for DataVersionHistoryCard component and store action.
 *
 * Covers:
 *  1.  Renders figure with correct aria-label
 *  2.  Renders 📂 icon (aria-hidden)
 *  3.  Shows version count in header
 *  4.  Shows "stable" badge when overall_stability is stable
 *  5.  Shows "moderate" badge when overall_stability is moderate
 *  6.  Shows "high" badge when overall_stability is high
 *  7.  Renders correct number of version rows
 *  8.  "Latest" badge shown on last version
 *  9.  Filenames rendered in version rows
 * 10.  Row count shown in version rows
 * 11.  Column count shown in version rows
 * 12.  Drift score shown in connector between versions
 * 13.  Single version — no drift connector
 * 14.  Summary text is shown
 * 15.  Returns null for empty versions list
 * 16.  Store: attachVersionHistoryToLastMessage attaches to last assistant message
 * 17.  Store: does not attach to user message
 * 18.  Store: empty messages handled without crash
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { DataVersionHistoryCard } from "@/components/chat/data-version-history-card"
import type { DataVersionHistoryResult, DataVersionEntry } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

function makeEntry(
  version: number,
  overrides: Partial<DataVersionEntry> = {}
): DataVersionEntry {
  return {
    version,
    dataset_id: `ds-${version}`,
    filename: `data_v${version}.csv`,
    row_count: version * 100,
    column_count: 5,
    uploaded_at: `2024-0${version}-01T00:00:00`,
    size_bytes: version * 1024,
    drift_from_previous: null,
    ...overrides,
  }
}

function makeHistory(
  overrides: Partial<DataVersionHistoryResult> = {}
): DataVersionHistoryResult {
  const v1 = makeEntry(1)
  const v2 = makeEntry(2, {
    drift_from_previous: {
      drift_score: 35,
      summary: "Moderate shift in revenue column.",
      changed_columns: 2,
      new_columns: [],
      dropped_columns: [],
      row_count_change_pct: 10.0,
    },
  })
  return {
    version_count: 2,
    versions: [v1, v2],
    overall_stability: "moderate",
    summary: "2 versions uploaded. Distribution changes are moderate.",
    ...overrides,
  }
}

// ---------------------------------------------------------------------------
// Component rendering tests
// ---------------------------------------------------------------------------

describe("DataVersionHistoryCard", () => {
  it("renders figure with correct aria-label", () => {
    render(<DataVersionHistoryCard history={makeHistory()} />)
    expect(screen.getByRole("figure", { hidden: true })).toBeInTheDocument()
    const fig = document.querySelector("figure")
    expect(fig).toHaveAttribute("aria-label", "Data version history")
  })

  it("renders 📂 icon as aria-hidden", () => {
    render(<DataVersionHistoryCard history={makeHistory()} />)
    const icon = screen.getByText("📂")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("shows version count in header", () => {
    render(<DataVersionHistoryCard history={makeHistory()} />)
    const elements = screen.getAllByText(/2 versions/i)
    expect(elements.length).toBeGreaterThanOrEqual(1)
  })

  it("shows 'stable' badge for stable stability", () => {
    render(
      <DataVersionHistoryCard
        history={makeHistory({ overall_stability: "stable" })}
      />
    )
    expect(screen.getByText("Stable")).toBeInTheDocument()
  })

  it("shows 'moderate drift' badge for moderate stability", () => {
    render(<DataVersionHistoryCard history={makeHistory()} />)
    expect(screen.getByText("Moderate Drift")).toBeInTheDocument()
  })

  it("shows 'high drift' badge for high stability", () => {
    render(
      <DataVersionHistoryCard
        history={makeHistory({ overall_stability: "high" })}
      />
    )
    expect(screen.getByText("High Drift")).toBeInTheDocument()
  })

  it("renders correct number of version rows", () => {
    const history = makeHistory()
    render(<DataVersionHistoryCard history={history} />)
    // Both filenames should appear
    expect(screen.getByText("data_v1.csv")).toBeInTheDocument()
    expect(screen.getByText("data_v2.csv")).toBeInTheDocument()
  })

  it("shows 'Latest' badge on the most recent version", () => {
    render(<DataVersionHistoryCard history={makeHistory()} />)
    expect(screen.getByText("Latest")).toBeInTheDocument()
  })

  it("renders filenames in version rows", () => {
    render(<DataVersionHistoryCard history={makeHistory()} />)
    expect(screen.getByText("data_v2.csv")).toBeInTheDocument()
  })

  it("shows row count for each version", () => {
    render(<DataVersionHistoryCard history={makeHistory()} />)
    // v1 has 100 rows, v2 has 200 rows
    expect(screen.getByText(/100 rows/i)).toBeInTheDocument()
    expect(screen.getByText(/200 rows/i)).toBeInTheDocument()
  })

  it("shows column count for each version", () => {
    render(<DataVersionHistoryCard history={makeHistory()} />)
    const colLabels = screen.getAllByText(/5 cols/i)
    expect(colLabels.length).toBeGreaterThanOrEqual(1)
  })

  it("shows drift score in connector between versions", () => {
    render(<DataVersionHistoryCard history={makeHistory()} />)
    expect(screen.getByText(/35\/100/i)).toBeInTheDocument()
  })

  it("shows no drift connector for single version", () => {
    const single = makeHistory({
      version_count: 1,
      versions: [makeEntry(1)],
    })
    render(<DataVersionHistoryCard history={single} />)
    // No drift connector text
    expect(screen.queryByText(/Drift:/i)).not.toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<DataVersionHistoryCard history={makeHistory()} />)
    expect(
      screen.getByText(/2 versions uploaded/i)
    ).toBeInTheDocument()
  })

  it("returns null for empty versions list", () => {
    const empty = makeHistory({ version_count: 0, versions: [] })
    const { container } = render(<DataVersionHistoryCard history={empty} />)
    expect(container.firstChild).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Store tests
// ---------------------------------------------------------------------------

describe("attachVersionHistoryToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches version_history to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "show history" },
        { role: "assistant", content: "Here is your history." },
      ],
    })
    const history = makeHistory()
    useAppStore.getState().attachVersionHistoryToLastMessage(history)
    const messages = useAppStore.getState().messages
    expect(messages[messages.length - 1].version_history).toEqual(history)
  })

  it("does not attach to a user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "show history" }],
    })
    useAppStore
      .getState()
      .attachVersionHistoryToLastMessage(makeHistory())
    const messages = useAppStore.getState().messages
    expect(messages[messages.length - 1].version_history).toBeUndefined()
  })

  it("handles empty messages list without crash", () => {
    useAppStore.setState({ messages: [] })
    expect(() =>
      useAppStore.getState().attachVersionHistoryToLastMessage(makeHistory())
    ).not.toThrow()
  })
})
