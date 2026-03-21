/**
 * Tests for CorrelationBarCard component and related store/type changes.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { CorrelationBarCard } from "@/components/data/correlation-bar-card"
import type { TargetCorrelationResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Test fixtures -------------------------------------------------------

const fullResult: TargetCorrelationResult = {
  dataset_id: "ds-1",
  target_col: "revenue",
  correlations: [
    {
      column: "units",
      correlation: 0.95,
      strength: "very strong",
      direction: "positive",
    },
    {
      column: "cost",
      correlation: 0.72,
      strength: "strong",
      direction: "positive",
    },
    {
      column: "discount",
      correlation: -0.61,
      strength: "strong",
      direction: "negative",
    },
    {
      column: "returns",
      correlation: -0.28,
      strength: "weak",
      direction: "negative",
    },
  ],
  summary:
    "The strongest relationship with revenue is units (r = +0.95, very strong positively correlated).",
}

const emptyResult: TargetCorrelationResult = {
  dataset_id: "ds-2",
  target_col: "revenue",
  correlations: [],
  summary: "No meaningful correlations found with revenue.",
}

// --- Tests ---------------------------------------------------------------

describe("CorrelationBarCard", () => {
  it("renders the card with testid", () => {
    render(<CorrelationBarCard result={fullResult} />)
    expect(screen.getByTestId("correlation-bar-card")).toBeInTheDocument()
  })

  it("shows the target column name in the header", () => {
    render(<CorrelationBarCard result={fullResult} />)
    expect(screen.getAllByText(/revenue/i).length).toBeGreaterThan(0)
  })

  it("renders all correlation entries", () => {
    render(<CorrelationBarCard result={fullResult} />)
    expect(screen.getByText("units")).toBeInTheDocument()
    expect(screen.getByText("cost")).toBeInTheDocument()
    expect(screen.getByText("discount")).toBeInTheDocument()
    expect(screen.getByText("returns")).toBeInTheDocument()
  })

  it("shows strength badges", () => {
    render(<CorrelationBarCard result={fullResult} />)
    // Should show strength labels
    expect(screen.getAllByText("very strong").length).toBeGreaterThan(0)
    expect(screen.getAllByText("strong").length).toBeGreaterThan(0)
  })

  it("shows the summary text", () => {
    render(<CorrelationBarCard result={fullResult} />)
    expect(screen.getByText(/strongest relationship/i)).toBeInTheDocument()
  })

  it("shows positive/negative indicator in header", () => {
    render(<CorrelationBarCard result={fullResult} />)
    expect(screen.getByText(/Blue = positive/i)).toBeInTheDocument()
  })

  it("renders empty state when no correlations", () => {
    render(<CorrelationBarCard result={emptyResult} />)
    expect(screen.getByTestId("correlation-bar-card")).toBeInTheDocument()
    expect(screen.getByText(/No meaningful correlations/i)).toBeInTheDocument()
  })

  it("underscores in column name are replaced with spaces in display", () => {
    const resultWithUnderscore: TargetCorrelationResult = {
      dataset_id: "ds-3",
      target_col: "net_revenue",
      correlations: [
        {
          column: "gross_sales",
          correlation: 0.8,
          strength: "strong",
          direction: "positive",
        },
      ],
      summary: "Strong correlation with gross sales.",
    }
    render(<CorrelationBarCard result={resultWithUnderscore} />)
    expect(screen.getByText("gross sales")).toBeInTheDocument()
  })
})

// --- Store action tests --------------------------------------------------

describe("attachCorrelationToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches target_correlation to the last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "what drives revenue?", timestamp: "t1" },
        { role: "assistant", content: "Here are the correlations...", timestamp: "t2" },
      ],
    })
    const { attachCorrelationToLastMessage } = useAppStore.getState()
    attachCorrelationToLastMessage(fullResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].target_correlation).toEqual(fullResult)
  })

  it("does not modify user messages", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "what drives revenue?", timestamp: "t1" },
      ],
    })
    const { attachCorrelationToLastMessage } = useAppStore.getState()
    attachCorrelationToLastMessage(fullResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].target_correlation).toBeUndefined()
  })

  it("is a no-op on empty message list", () => {
    useAppStore.setState({ messages: [] })
    const { attachCorrelationToLastMessage } = useAppStore.getState()
    expect(() => attachCorrelationToLastMessage(fullResult)).not.toThrow()
  })
})
