/**
 * Tests for DataStoryCard component and attachDataStoryToLastMessage store action.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { DataStoryCard } from "@/components/data/data-story-card"
import type { DataStory } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Fixtures -----------------------------------------------------------

const fullStory: DataStory = {
  dataset_id: "ds-1",
  filename: "sales.csv",
  row_count: 200,
  col_count: 4,
  readiness_score: 82,
  readiness_grade: "B",
  sections: [
    {
      type: "readiness",
      title: "Data Quality",
      insight: "Grade B (82/100). Consider addressing missing values.",
      data: { score: 82, grade: "B" },
    },
    {
      type: "group_by",
      title: "Breakdown by region",
      insight: "Highest: East (500.00). Top group is 46.3% of total.",
      data: {},
    },
    {
      type: "correlations",
      title: "What Drives revenue",
      insight: "Top correlates: quantity (r=+0.85, very strong).",
      data: {},
    },
    {
      type: "anomalies",
      title: "Anomaly Scan",
      insight: "Found 10 anomalous records out of 200 rows.",
      data: {},
    },
  ],
  summary: "Your dataset has 200 rows and 4 columns with a data quality grade of B (82/100). Highest: East. Top correlates: quantity.",
  recommended_next_step: "Your data looks ready to model! Say 'train a model to predict revenue'.",
}

const minimalStory: DataStory = {
  dataset_id: "ds-2",
  filename: "data.csv",
  row_count: 50,
  col_count: 2,
  readiness_score: 45,
  readiness_grade: "D",
  sections: [
    {
      type: "readiness",
      title: "Data Quality",
      insight: "Grade D (45/100). You need more rows.",
      data: {},
    },
  ],
  summary: "Your dataset has 50 rows and 2 columns with a data quality grade of D (45/100).",
  recommended_next_step: "Fix data quality first: add more data rows.",
}

// --- Component tests ----------------------------------------------------

describe("DataStoryCard", () => {
  it("renders the filename in header", () => {
    render(<DataStoryCard result={fullStory} />)
    expect(screen.getByText(/sales\.csv/)).toBeInTheDocument()
  })

  it("shows row and column count", () => {
    render(<DataStoryCard result={fullStory} />)
    expect(screen.getAllByText(/200/).length).toBeGreaterThan(0)
    expect(screen.getByText(/4 cols/)).toBeInTheDocument()
  })

  it("shows grade badge", () => {
    render(<DataStoryCard result={fullStory} />)
    expect(screen.getByText("Grade B")).toBeInTheDocument()
  })

  it("shows readiness score", () => {
    render(<DataStoryCard result={fullStory} />)
    expect(screen.getByText("82/100")).toBeInTheDocument()
  })

  it("renders all section titles", () => {
    render(<DataStoryCard result={fullStory} />)
    expect(screen.getByText("Data Quality")).toBeInTheDocument()
    expect(screen.getByText("Breakdown by region")).toBeInTheDocument()
    expect(screen.getByText("What Drives revenue")).toBeInTheDocument()
    expect(screen.getByText("Anomaly Scan")).toBeInTheDocument()
  })

  it("renders section insights", () => {
    render(<DataStoryCard result={fullStory} />)
    expect(screen.getAllByText(/Grade B/).length).toBeGreaterThan(0)
    expect(screen.getByText(/Highest: East/)).toBeInTheDocument()
  })

  it("renders recommended next step", () => {
    render(<DataStoryCard result={fullStory} />)
    expect(screen.getByText(/ready to model/i)).toBeInTheDocument()
  })

  it("renders with poor grade", () => {
    render(<DataStoryCard result={minimalStory} />)
    expect(screen.getByText("Grade D")).toBeInTheDocument()
    expect(screen.getByText("45/100")).toBeInTheDocument()
  })

  it("renders recommended step for poor data", () => {
    render(<DataStoryCard result={minimalStory} />)
    expect(screen.getByText(/Fix data quality/i)).toBeInTheDocument()
  })

  it("shows 'Data Story' in header", () => {
    render(<DataStoryCard result={fullStory} />)
    expect(screen.getByText(/Data Story/i)).toBeInTheDocument()
  })
})

// --- Store action tests -------------------------------------------------

describe("attachDataStoryToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches data_story to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "analyze my data", timestamp: "t1" },
        { role: "assistant", content: "Here is your analysis.", timestamp: "t2" },
      ],
    })

    useAppStore.getState().attachDataStoryToLastMessage(fullStory)

    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.data_story).toBeDefined()
    expect(last.data_story?.filename).toBe("sales.csv")
    expect(last.data_story?.readiness_grade).toBe("B")
    expect(last.data_story?.sections).toHaveLength(4)
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "analyze", timestamp: "t1" }],
    })

    useAppStore.getState().attachDataStoryToLastMessage(fullStory)

    const messages = useAppStore.getState().messages
    expect(messages[0].data_story).toBeUndefined()
  })

  it("does nothing when no messages exist", () => {
    useAppStore.getState().attachDataStoryToLastMessage(fullStory)
    expect(useAppStore.getState().messages).toHaveLength(0)
  })
})
