/**
 * Tests for AutoInsightCard component.
 *
 * 1.  Renders the card heading "Interesting findings in your data"
 * 2.  Renders the 🔍 icon
 * 3.  Renders the dataset name badge
 * 4.  Renders the row count badge
 * 5.  Renders the column count badge
 * 6.  Renders the summary text
 * 7.  Renders each finding row
 * 8.  Renders the finding text for each finding
 * 9.  Renders the action button for each finding
 * 10. Clicking an action button calls onActionClick with the correct prompt
 * 11. sr-only figcaption is present for accessibility
 * 12. Store: attachAutoInsightToLastMessage attaches to last assistant message
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { AutoInsightCard } from "@/components/chat/auto-insight-card"
import type { AutoInsightResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const mockResult: AutoInsightResult = {
  dataset_name: "sales_q2.csv",
  row_count: 1250,
  column_count: 8,
  summary: "Found 2 interesting findings in **sales_q2.csv**.",
  findings: [
    {
      insight_type: "date_column",
      icon: "📅",
      finding:
        "I found a **order_date** column — this data has a time dimension.",
      suggested_action: "Show me revenue trends over time",
      priority: 1,
    },
    {
      insight_type: "class_imbalance",
      icon: "⚖️",
      finding:
        "Column **is_churn** is imbalanced: 87% No vs 13% Yes.",
      suggested_action: "How do I handle class imbalance in is_churn?",
      priority: 1,
    },
  ],
}

describe("AutoInsightCard", () => {
  it("renders the card heading", () => {
    render(<AutoInsightCard result={mockResult} />)
    expect(
      screen.getByTestId("auto-insight-heading")
    ).toHaveTextContent("Interesting findings in your data")
  })

  it("renders the 🔍 icon", () => {
    render(<AutoInsightCard result={mockResult} />)
    expect(screen.getByTestId("auto-insight-icon")).toHaveTextContent("🔍")
  })

  it("renders the dataset name badge", () => {
    render(<AutoInsightCard result={mockResult} />)
    expect(screen.getByTestId("auto-insight-card")).toHaveTextContent(
      "sales_q2.csv"
    )
  })

  it("renders the row count", () => {
    render(<AutoInsightCard result={mockResult} />)
    expect(screen.getByTestId("auto-insight-card")).toHaveTextContent("1,250")
  })

  it("renders the column count", () => {
    render(<AutoInsightCard result={mockResult} />)
    expect(screen.getByTestId("auto-insight-card")).toHaveTextContent(
      "8 columns"
    )
  })

  it("renders the summary text", () => {
    render(<AutoInsightCard result={mockResult} />)
    expect(screen.getByTestId("auto-insight-summary")).toBeTruthy()
  })

  it("renders a finding row for each finding", () => {
    render(<AutoInsightCard result={mockResult} />)
    expect(screen.getByTestId("insight-finding-0")).toBeTruthy()
    expect(screen.getByTestId("insight-finding-1")).toBeTruthy()
  })

  it("renders the finding text", () => {
    render(<AutoInsightCard result={mockResult} />)
    expect(screen.getByTestId("insight-text-0")).toHaveTextContent(
      "order_date"
    )
    expect(screen.getByTestId("insight-text-1")).toHaveTextContent("is_churn")
  })

  it("renders an action button for each finding", () => {
    render(<AutoInsightCard result={mockResult} />)
    expect(screen.getByTestId("insight-action-0")).toBeTruthy()
    expect(screen.getByTestId("insight-action-1")).toBeTruthy()
  })

  it("clicking an action button calls onActionClick with the correct prompt", () => {
    const onActionClick = jest.fn()
    render(<AutoInsightCard result={mockResult} onActionClick={onActionClick} />)
    fireEvent.click(screen.getByTestId("insight-action-0"))
    expect(onActionClick).toHaveBeenCalledWith("Show me revenue trends over time")
  })

  it("second action button calls onActionClick with correct prompt", () => {
    const onActionClick = jest.fn()
    render(<AutoInsightCard result={mockResult} onActionClick={onActionClick} />)
    fireEvent.click(screen.getByTestId("insight-action-1"))
    expect(onActionClick).toHaveBeenCalledWith(
      "How do I handle class imbalance in is_churn?"
    )
  })

  it("renders an sr-only figcaption for accessibility", () => {
    render(<AutoInsightCard result={mockResult} />)
    const fig = screen.getByTestId("auto-insight-card")
    const caption = fig.querySelector("figcaption")
    expect(caption).toBeTruthy()
    expect(caption?.classList.contains("sr-only")).toBe(true)
  })
})

describe("Store: attachAutoInsightToLastMessage", () => {
  it("attaches auto_insight to the last assistant message", () => {
    const store = useAppStore.getState()
    store.setMessages([
      { id: "1", role: "user", content: "Hi", timestamp: "" },
      { id: "2", role: "assistant", content: "Hello", timestamp: "" },
    ])
    store.attachAutoInsightToLastMessage(mockResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[1].auto_insight).toEqual(mockResult)
  })

  it("does not attach to a user message", () => {
    const store = useAppStore.getState()
    store.setMessages([
      { id: "1", role: "user", content: "Hi", timestamp: "" },
    ])
    store.attachAutoInsightToLastMessage(mockResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[0].auto_insight).toBeUndefined()
  })
})
