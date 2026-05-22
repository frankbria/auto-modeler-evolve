/**
 * Tests for ColumnTypeSuggestionCard component.
 *
 * 1.  Renders the "Column Type Check" heading
 * 2.  Renders the ⚠️ icon when there are suggestions
 * 3.  Renders the ✅ icon when no suggestions
 * 4.  Renders the dataset name badge
 * 5.  Renders "N issues found" badge when has_suggestions=true
 * 6.  Renders "All types correct" badge when has_suggestions=false
 * 7.  Renders the summary text
 * 8.  Renders each suggestion row
 * 9.  Renders column name in suggestion row
 * 10. Renders current_dtype and suggested_dtype badges
 * 11. Renders the reason text with bold markdown
 * 12. Renders sample values
 * 13. "Fix this →" button calls onActionClick with suggested_action
 * 14. All-good state renders when no suggestions
 * 15. sr-only figcaption for accessibility
 * 16. Store: attachColumnTypeSuggestionsToLastMessage attaches to last assistant message
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { ColumnTypeSuggestionCard } from "@/components/chat/column-type-suggestion-card"
import type { ColumnTypeSuggestionResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const mockResultWithSuggestions: ColumnTypeSuggestionResult = {
  dataset_name: "sales_q2.csv",
  suggestions: [
    {
      column: "price",
      current_dtype: "text",
      suggested_dtype: "numeric",
      reason:
        "The values in **price** look like numbers (e.g. 12.99, 8.50, 3.00), but the column is stored as text.",
      confidence: "high",
      sample_values: ["12.99", "8.50", "3.00", "15.00"],
      suggested_action: "Convert price to numeric",
    },
    {
      column: "is_active",
      current_dtype: "text",
      suggested_dtype: "boolean",
      reason: "**is_active** only contains values like 'true', 'false' — True/False flags.",
      confidence: "high",
      sample_values: ["true", "false", "true"],
      suggested_action: "Convert is_active to boolean",
    },
  ],
  has_suggestions: true,
  dataset_rows: 1250,
  dataset_cols: 8,
  summary: "Found 2 column type issues in **sales_q2.csv**.",
}

const mockResultNoSuggestions: ColumnTypeSuggestionResult = {
  dataset_name: "clean_data.csv",
  suggestions: [],
  has_suggestions: false,
  dataset_rows: 500,
  dataset_cols: 5,
  summary: "All column types in **clean_data.csv** look correct.",
}

describe("ColumnTypeSuggestionCard — with suggestions", () => {
  it("renders the Column Type Check heading", () => {
    render(<ColumnTypeSuggestionCard result={mockResultWithSuggestions} />)
    expect(screen.getByTestId("column-type-heading")).toHaveTextContent("Column Type Check")
  })

  it("renders the ⚠️ icon when there are suggestions", () => {
    render(<ColumnTypeSuggestionCard result={mockResultWithSuggestions} />)
    expect(screen.getByTestId("column-type-icon")).toHaveTextContent("⚠️")
  })

  it("renders the dataset name badge", () => {
    render(<ColumnTypeSuggestionCard result={mockResultWithSuggestions} />)
    expect(screen.getByTestId("column-type-suggestion-card")).toHaveTextContent("sales_q2.csv")
  })

  it("renders the issues-found badge", () => {
    render(<ColumnTypeSuggestionCard result={mockResultWithSuggestions} />)
    expect(screen.getByTestId("column-type-suggestion-card")).toHaveTextContent("2 issues found")
  })

  it("renders the summary text", () => {
    render(<ColumnTypeSuggestionCard result={mockResultWithSuggestions} />)
    expect(screen.getByTestId("column-type-summary")).toHaveTextContent(
      "Found 2 column type issues in"
    )
  })

  it("renders each suggestion row", () => {
    render(<ColumnTypeSuggestionCard result={mockResultWithSuggestions} />)
    expect(screen.getByTestId("type-suggestion-row-0")).toBeInTheDocument()
    expect(screen.getByTestId("type-suggestion-row-1")).toBeInTheDocument()
  })

  it("renders column name in suggestion row", () => {
    render(<ColumnTypeSuggestionCard result={mockResultWithSuggestions} />)
    expect(screen.getByTestId("type-suggestion-column-0")).toHaveTextContent("price")
    expect(screen.getByTestId("type-suggestion-column-1")).toHaveTextContent("is_active")
  })

  it("renders the reason text in suggestion row", () => {
    render(<ColumnTypeSuggestionCard result={mockResultWithSuggestions} />)
    expect(screen.getByTestId("type-suggestion-reason-0")).toBeInTheDocument()
  })

  it("renders sample values in suggestion row", () => {
    render(<ColumnTypeSuggestionCard result={mockResultWithSuggestions} />)
    expect(screen.getByTestId("type-suggestion-samples-0")).toBeInTheDocument()
  })

  it("clicking Fix button calls onActionClick with suggested_action", () => {
    const onActionClick = jest.fn()
    render(
      <ColumnTypeSuggestionCard
        result={mockResultWithSuggestions}
        onActionClick={onActionClick}
      />
    )
    fireEvent.click(screen.getByTestId("type-suggestion-fix-0"))
    expect(onActionClick).toHaveBeenCalledWith("Convert price to numeric")
  })

  it("has an sr-only figcaption for accessibility", () => {
    const { container } = render(
      <ColumnTypeSuggestionCard result={mockResultWithSuggestions} />
    )
    const caption = container.querySelector("figcaption.sr-only")
    expect(caption).toBeInTheDocument()
    expect(caption?.textContent).toMatch(/sales_q2.csv/)
  })
})

describe("ColumnTypeSuggestionCard — no suggestions", () => {
  it("renders ✅ icon when no suggestions", () => {
    render(<ColumnTypeSuggestionCard result={mockResultNoSuggestions} />)
    expect(screen.getByTestId("column-type-icon")).toHaveTextContent("✅")
  })

  it("renders all-types-correct badge", () => {
    render(<ColumnTypeSuggestionCard result={mockResultNoSuggestions} />)
    expect(screen.getByTestId("column-type-suggestion-card")).toHaveTextContent("All types correct")
  })

  it("renders the all-good message", () => {
    render(<ColumnTypeSuggestionCard result={mockResultNoSuggestions} />)
    expect(screen.getByTestId("column-type-all-good")).toBeInTheDocument()
  })

  it("does not render suggestion rows when no suggestions", () => {
    render(<ColumnTypeSuggestionCard result={mockResultNoSuggestions} />)
    expect(screen.queryByTestId("type-suggestion-row-0")).not.toBeInTheDocument()
  })
})

describe("Zustand store — attachColumnTypeSuggestionsToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches column_type_suggestions to the last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "check my column types" },
        { role: "assistant", content: "Let me check your column types." },
      ],
    })
    useAppStore
      .getState()
      .attachColumnTypeSuggestionsToLastMessage(mockResultWithSuggestions)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.column_type_suggestions).toEqual(mockResultWithSuggestions)
  })

  it("does not attach to a user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "check my column types" }],
    })
    useAppStore
      .getState()
      .attachColumnTypeSuggestionsToLastMessage(mockResultWithSuggestions)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect((last as { column_type_suggestions?: unknown }).column_type_suggestions).toBeUndefined()
  })
})
