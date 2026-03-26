/**
 * Tests for ClusteringCard component and related store/API plumbing.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import fetchMock from "jest-fetch-mock"
import { ClusteringCard } from "@/components/data/clustering-card"
import type { ClusteringResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"
import { api } from "@/lib/api"

fetchMock.enableMocks()

// --- Fixtures ---------------------------------------------------------------

const twoClusterResult: ClusteringResult = {
  n_clusters: 2,
  features_used: ["revenue", "units"],
  auto_k: true,
  rows_clustered: 10,
  summary:
    "Found 2 natural groups in 10 rows using 2 features. Largest group has 6 records (60%), smallest has 4 (40%).",
  clusters: [
    {
      cluster_id: 0,
      size: 6,
      size_pct: 60.0,
      centroid: { revenue: 800.0, units: 80 },
      distinguishing: [
        {
          feature: "revenue",
          cluster_mean: 800.0,
          global_mean: 500.0,
          direction: "above",
          magnitude: 1.5,
        },
        {
          feature: "units",
          cluster_mean: 80,
          global_mean: 50,
          direction: "above",
          magnitude: 1.2,
        },
      ],
      description: "Group 1 (60% of data): tends toward high revenue, high units.",
    },
    {
      cluster_id: 1,
      size: 4,
      size_pct: 40.0,
      centroid: { revenue: 100.0, units: 10 },
      distinguishing: [
        {
          feature: "revenue",
          cluster_mean: 100.0,
          global_mean: 500.0,
          direction: "below",
          magnitude: 2.0,
        },
      ],
      description: "Group 2 (40% of data): tends toward low revenue.",
    },
  ],
}

const threeClusterResult: ClusteringResult = {
  n_clusters: 3,
  features_used: ["x", "y", "z"],
  auto_k: false,
  rows_clustered: 15,
  summary: "Found 3 natural groups in 15 rows using 3 features. Largest group has 7 records (47%), smallest has 3 (20%).",
  clusters: [
    {
      cluster_id: 0,
      size: 7,
      size_pct: 46.7,
      centroid: { x: 1.0, y: 2.0, z: 3.0 },
      distinguishing: [],
      description: "Group 1 (47% of data): no strongly distinguishing features.",
    },
    {
      cluster_id: 1,
      size: 5,
      size_pct: 33.3,
      centroid: { x: 10.0, y: 20.0, z: 30.0 },
      distinguishing: [
        {
          feature: "x",
          cluster_mean: 10.0,
          global_mean: 5.0,
          direction: "above",
          magnitude: 0.8,
        },
      ],
      description: "Group 2 (33% of data): tends toward high x.",
    },
    {
      cluster_id: 2,
      size: 3,
      size_pct: 20.0,
      centroid: { x: 5.0, y: 5.0, z: 5.0 },
      distinguishing: [],
      description: "Group 3 (20% of data): no strongly distinguishing features.",
    },
  ],
}

// --- Component rendering tests -----------------------------------------------

describe("ClusteringCard", () => {
  it("renders the header with cluster count", () => {
    render(<ClusteringCard result={twoClusterResult} />)
    expect(screen.getByText(/Customer Segmentation/i)).toBeInTheDocument()
    expect(screen.getByText(/2 groups/i)).toBeInTheDocument()
  })

  it("shows 'auto' badge when auto_k is true", () => {
    render(<ClusteringCard result={twoClusterResult} />)
    // Badge text is "2 groups · auto" — check it exists somewhere on the page
    const badge = screen.getAllByText(/auto/i)
    expect(badge.length).toBeGreaterThan(0)
  })

  it("shows 'manual' badge when auto_k is false", () => {
    render(<ClusteringCard result={threeClusterResult} />)
    const badge = screen.getAllByText(/manual/i)
    expect(badge.length).toBeGreaterThan(0)
  })

  it("renders the summary text", () => {
    render(<ClusteringCard result={twoClusterResult} />)
    expect(screen.getByText(/Found 2 natural groups/i)).toBeInTheDocument()
  })

  it("renders feature chips", () => {
    render(<ClusteringCard result={twoClusterResult} />)
    // "revenue" and "units" appear in feature chip list
    const revenueEls = screen.getAllByText(/revenue/i)
    expect(revenueEls.length).toBeGreaterThan(0)
    const unitsEls = screen.getAllByText(/units/i)
    expect(unitsEls.length).toBeGreaterThan(0)
  })

  it("renders cluster descriptions", () => {
    render(<ClusteringCard result={twoClusterResult} />)
    const group1Els = screen.getAllByText(/Group 1/i)
    expect(group1Els.length).toBeGreaterThan(0)
    const group2Els = screen.getAllByText(/Group 2/i)
    expect(group2Els.length).toBeGreaterThan(0)
  })

  it("renders size percentages", () => {
    render(<ClusteringCard result={twoClusterResult} />)
    const sixtyPct = screen.getAllByText(/60%/)
    expect(sixtyPct.length).toBeGreaterThan(0)
    const fortyPct = screen.getAllByText(/40%/)
    expect(fortyPct.length).toBeGreaterThan(0)
  })

  it("renders distinguishing feature badges with direction arrows", () => {
    render(<ClusteringCard result={twoClusterResult} />)
    // The ↑ arrow for "above" direction
    const upArrows = screen.getAllByText(/↑/)
    expect(upArrows.length).toBeGreaterThan(0)
    // The ↓ arrow for "below" direction
    const downArrows = screen.getAllByText(/↓/)
    expect(downArrows.length).toBeGreaterThan(0)
  })

  it("renders 3 clusters correctly", () => {
    render(<ClusteringCard result={threeClusterResult} />)
    const threeGroupBadge = screen.getAllByText(/3 groups/i)
    expect(threeGroupBadge.length).toBeGreaterThan(0)
    const group3Els = screen.getAllByText(/Group 3/i)
    expect(group3Els.length).toBeGreaterThan(0)
  })

  it("renders footer with rows clustered and k", () => {
    render(<ClusteringCard result={twoClusterResult} />)
    expect(screen.getByText(/10 rows clustered/i)).toBeInTheDocument()
    expect(screen.getByText(/k=2/i)).toBeInTheDocument()
  })

  it("renders footer indicating auto-selected k", () => {
    render(<ClusteringCard result={twoClusterResult} />)
    expect(screen.getByText(/auto-selected/i)).toBeInTheDocument()
  })

  it("renders footer indicating specified k", () => {
    render(<ClusteringCard result={threeClusterResult} />)
    expect(screen.getByText(/specified/i)).toBeInTheDocument()
  })
})

// --- Zustand store tests -----------------------------------------------------

describe("ClusteringCard store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attachClustersToLastMessage attaches clusters to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "cluster my data", timestamp: "" },
        { role: "assistant", content: "Found 2 groups.", timestamp: "" },
      ],
    })

    useAppStore.getState().attachClustersToLastMessage(twoClusterResult)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.clusters).toBeDefined()
    expect(last.clusters?.n_clusters).toBe(2)
  })

  it("does not modify user message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "cluster my data", timestamp: "" },
      ],
    })

    useAppStore.getState().attachClustersToLastMessage(twoClusterResult)
    const msgs = useAppStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.clusters).toBeUndefined()
  })
})

// --- API client tests ---------------------------------------------------------

describe("api.data.getClusters", () => {
  beforeEach(() => {
    fetchMock.resetMocks()
  })

  it("calls the correct URL without params", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(twoClusterResult))
    await api.data.getClusters("ds-1")
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/data/ds-1/clusters")
    )
  })

  it("appends n_clusters param when provided", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(twoClusterResult))
    await api.data.getClusters("ds-1", undefined, 3)
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("n_clusters=3")
    )
  })

  it("appends features param when provided", async () => {
    fetchMock.mockResponseOnce(JSON.stringify(twoClusterResult))
    await api.data.getClusters("ds-1", ["revenue", "units"])
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("features=revenue%2Cunits")
    )
  })

  it("throws on HTTP error", async () => {
    fetchMock.mockResponseOnce("Bad Request", { status: 400 })
    await expect(api.data.getClusters("ds-1")).rejects.toThrow("HTTP 400")
  })
})
