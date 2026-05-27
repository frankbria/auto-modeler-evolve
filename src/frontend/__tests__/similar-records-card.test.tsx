import React from "react"
import { render, screen } from "@testing-library/react"
import { SimilarRecordsCard } from "../components/deploy/similar-records-card"
import { SimilarRecordsResult } from "../lib/types"

function makeNeighbor(overrides: Partial<import("../lib/types").SimilarRecordNeighbor> = {}) {
  return {
    row_index: 12,
    distance: 0.45,
    similarity_score: 0.69,
    features: { age: 35, income: 52000, tenure: 3 },
    actual_label: "yes",
    predicted_label: "yes",
    predicted_confidence: 0.81,
    ...overrides,
  }
}

function makeData(overrides: Partial<SimilarRecordsResult> = {}): SimilarRecordsResult {
  return {
    query_row_index: 5,
    query_prediction: "yes",
    query_confidence: 0.88,
    feature_columns: ["age", "income", "tenure"],
    neighbors: [
      makeNeighbor({ row_index: 12, similarity_score: 0.9, distance: 0.2 }),
      makeNeighbor({ row_index: 7, similarity_score: 0.75, distance: 0.4, predicted_label: "no" }),
      makeNeighbor({ row_index: 23, similarity_score: 0.6, distance: 0.7 }),
    ],
    summary: "Found 3 similar records. The most similar record has a similarity score of 90%.",
    ...overrides,
  }
}

describe("SimilarRecordsCard", () => {
  it("renders the card container with correct test id", () => {
    render(<SimilarRecordsCard data={makeData()} />)
    expect(screen.getByTestId("similar-records-card")).toBeInTheDocument()
  })

  it("shows query row index in heading", () => {
    render(<SimilarRecordsCard data={makeData()} />)
    expect(screen.getByText(/Row 5/)).toBeInTheDocument()
  })

  it("shows query prediction", () => {
    render(<SimilarRecordsCard data={makeData()} />)
    expect(screen.getByTestId("query-prediction")).toHaveTextContent("yes")
  })

  it("renders the correct number of neighbor rows", () => {
    render(<SimilarRecordsCard data={makeData()} />)
    const rows = screen.getAllByTestId("neighbor-row")
    expect(rows).toHaveLength(3)
  })

  it("shows summary text", () => {
    render(<SimilarRecordsCard data={makeData()} />)
    expect(screen.getByText(/Found 3 similar records/)).toBeInTheDocument()
  })

  it("shows same outcome badge count", () => {
    // 2 out of 3 neighbors share 'yes' outcome
    render(<SimilarRecordsCard data={makeData()} />)
    expect(screen.getByText(/2\/3 same outcome/)).toBeInTheDocument()
  })

  it("shows empty state when no neighbors", () => {
    render(<SimilarRecordsCard data={makeData({ neighbors: [] })} />)
    expect(screen.getByText(/No similar records found/)).toBeInTheDocument()
  })

  it("renders confidence for query row", () => {
    render(<SimilarRecordsCard data={makeData()} />)
    // 0.88 → 88%
    expect(screen.getByText(/88% confidence/)).toBeInTheDocument()
  })
})
