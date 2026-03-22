/**
 * Tests for FilterSetCard, FilterBadge components and filter-related store actions.
 */
import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { FilterSetCard } from "@/components/chat/filter-set-card"
import { FilterBadge } from "@/components/data/filter-badge"
import type { FilterSetResult, ActiveFilter } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Fixtures -----------------------------------------------------------

const filterResult: FilterSetResult = {
  dataset_id: "ds-1",
  filter_summary: "region = East",
  conditions: [{ column: "region", operator: "eq", value: "East" }],
  original_rows: 1000,
  filtered_rows: 250,
  row_reduction_pct: 75.0,
}

const multiConditionFilter: FilterSetResult = {
  dataset_id: "ds-1",
  filter_summary: "revenue > 500 and region = North",
  conditions: [
    { column: "revenue", operator: "gt", value: 500 },
    { column: "region", operator: "eq", value: "North" },
  ],
  original_rows: 500,
  filtered_rows: 100,
  row_reduction_pct: 80.0,
}

const activeFilter: ActiveFilter = {
  dataset_id: "ds-1",
  active: true,
  filter_summary: "region = East",
  conditions: [{ column: "region", operator: "eq", value: "East" }],
  original_rows: 1000,
  filtered_rows: 250,
  row_reduction_pct: 75.0,
}

// --- FilterSetCard component tests -------------------------------------

describe("FilterSetCard", () => {
  it("renders filter-set-card testid", () => {
    render(<FilterSetCard result={filterResult} />)
    expect(screen.getByTestId("filter-set-card")).toBeInTheDocument()
  })

  it("shows column name in condition", () => {
    render(<FilterSetCard result={filterResult} />)
    expect(screen.getByText("region")).toBeInTheDocument()
  })

  it("shows operator as symbol (eq → =)", () => {
    render(<FilterSetCard result={filterResult} />)
    expect(screen.getByText("=")).toBeInTheDocument()
  })

  it("shows condition value", () => {
    render(<FilterSetCard result={filterResult} />)
    expect(screen.getByText("East")).toBeInTheDocument()
  })

  it("shows filtered row count", () => {
    render(<FilterSetCard result={filterResult} />)
    expect(screen.getAllByText(/250/).length).toBeGreaterThan(0)
  })

  it("shows original row count", () => {
    render(<FilterSetCard result={filterResult} />)
    expect(screen.getByText(/1,000/)).toBeInTheDocument()
  })

  it("shows reduction percentage when > 0", () => {
    render(<FilterSetCard result={filterResult} />)
    expect(screen.getByText(/75%/)).toBeInTheDocument()
  })

  it("renders multiple conditions", () => {
    render(<FilterSetCard result={multiConditionFilter} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
    expect(screen.getByText(">")).toBeInTheDocument()
    expect(screen.getByText("500")).toBeInTheDocument()
    expect(screen.getByText("North")).toBeInTheDocument()
  })

  it("shows gt operator as >", () => {
    render(<FilterSetCard result={multiConditionFilter} />)
    expect(screen.getByText(">")).toBeInTheDocument()
  })

  it("shows 'Filter Active' header", () => {
    render(<FilterSetCard result={filterResult} />)
    expect(screen.getByText(/Filter Active/i)).toBeInTheDocument()
  })

  it("shows instructions to clear filter", () => {
    render(<FilterSetCard result={filterResult} />)
    expect(screen.getByText(/clear filter/i)).toBeInTheDocument()
  })
})

// --- FilterBadge component tests ---------------------------------------

describe("FilterBadge", () => {
  it("renders filter-badge testid when active", () => {
    render(<FilterBadge filter={activeFilter} onClear={() => {}} />)
    expect(screen.getByTestId("filter-badge")).toBeInTheDocument()
  })

  it("shows filter summary", () => {
    render(<FilterBadge filter={activeFilter} onClear={() => {}} />)
    expect(screen.getByText("region = East")).toBeInTheDocument()
  })

  it("shows row counts", () => {
    render(<FilterBadge filter={activeFilter} onClear={() => {}} />)
    expect(screen.getByText(/250/)).toBeInTheDocument()
    expect(screen.getByText(/1,000/)).toBeInTheDocument()
  })

  it("renders clear button", () => {
    render(<FilterBadge filter={activeFilter} onClear={() => {}} />)
    expect(screen.getByTestId("filter-clear-btn")).toBeInTheDocument()
  })

  it("calls onClear when clear button clicked", () => {
    const onClear = jest.fn()
    render(<FilterBadge filter={activeFilter} onClear={onClear} />)
    fireEvent.click(screen.getByTestId("filter-clear-btn"))
    expect(onClear).toHaveBeenCalledTimes(1)
  })

  it("returns null when filter is inactive", () => {
    const inactiveFilter: ActiveFilter = { dataset_id: "ds-1", active: false }
    const { container } = render(<FilterBadge filter={inactiveFilter} onClear={() => {}} />)
    expect(container.firstChild).toBeNull()
  })

  it("returns null when filter_summary is missing", () => {
    const noSummary: ActiveFilter = { dataset_id: "ds-1", active: true }
    const { container } = render(<FilterBadge filter={noSummary} onClear={() => {}} />)
    expect(container.firstChild).toBeNull()
  })
})

// --- Store action tests ------------------------------------------------

describe("attachFilterToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches filter_set to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "show only East region", timestamp: "t1" },
        { role: "assistant", content: "Filter applied.", timestamp: "t2" },
      ],
    })

    useAppStore.getState().attachFilterToLastMessage(filterResult)

    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.filter_set).toBeDefined()
    expect(last.filter_set?.filter_summary).toBe("region = East")
    expect(last.filter_set?.filtered_rows).toBe(250)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "filter", timestamp: "t1" }],
    })

    useAppStore.getState().attachFilterToLastMessage(filterResult)

    expect(useAppStore.getState().messages[0].filter_set).toBeUndefined()
  })

  it("does nothing when no messages exist", () => {
    useAppStore.getState().attachFilterToLastMessage(filterResult)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })
})

describe("setActiveFilter", () => {
  beforeEach(() => {
    useAppStore.setState({ activeFilter: null })
  })

  it("sets active filter", () => {
    useAppStore.getState().setActiveFilter(activeFilter)
    expect(useAppStore.getState().activeFilter).toEqual(activeFilter)
  })

  it("clears active filter when null passed", () => {
    useAppStore.setState({ activeFilter })
    useAppStore.getState().setActiveFilter(null)
    expect(useAppStore.getState().activeFilter).toBeNull()
  })

  it("initializes as null", () => {
    useAppStore.setState({ activeFilter: null })
    expect(useAppStore.getState().activeFilter).toBeNull()
  })
})
