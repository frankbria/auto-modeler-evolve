/**
 * Tests for FeaturePsiCard component and store action.
 */

import React from "react"
import { render, screen, act } from "@testing-library/react"

import { FeaturePsiCard } from "@/components/deploy/feature-psi-card"
import type { FeaturePsiResult, FeaturePsiEntry } from "@/lib/types"
import { useAppStore } from "@/lib/store"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const stableEntry: FeaturePsiEntry = {
  feature: "revenue",
  feature_type: "numeric",
  psi: 0.04,
  psi_label: "Stable",
  severity: "stable",
  n_new_categories: null,
  n_dropped_categories: null,
}

const criticalEntry: FeaturePsiEntry = {
  feature: "price",
  feature_type: "numeric",
  psi: 0.35,
  psi_label: "Critical",
  severity: "critical",
  n_new_categories: null,
  n_dropped_categories: null,
}

const watchEntry: FeaturePsiEntry = {
  feature: "region",
  feature_type: "categorical",
  psi: 0.14,
  psi_label: "Watch",
  severity: "watch",
  n_new_categories: 1,
  n_dropped_categories: 0,
}

const stableResult: FeaturePsiResult = {
  deployment_id: "dep-1",
  features: [stableEntry],
  features_analyzed: 1,
  stable_count: 1,
  watch_count: 0,
  critical_count: 0,
  overall_psi: 0.04,
  verdict: "stable",
  verdict_label: "All features stable",
  sample_count: 80,
  training_count: 500,
  summary: "All 1 analyzed feature(s) are stable (PSI < 0.10).",
}

const criticalResult: FeaturePsiResult = {
  deployment_id: "dep-2",
  features: [criticalEntry, watchEntry, stableEntry],
  features_analyzed: 3,
  stable_count: 1,
  watch_count: 1,
  critical_count: 1,
  overall_psi: 0.177,
  verdict: "critical",
  verdict_label: "1 feature with major shift",
  sample_count: 150,
  training_count: 600,
  summary: "1 of 3 input feature(s) show major distribution shift (PSI ≥ 0.20).",
}

const noDataResult: FeaturePsiResult = {
  deployment_id: "dep-3",
  features: [],
  features_analyzed: 0,
  stable_count: 0,
  watch_count: 0,
  critical_count: 0,
  overall_psi: 0.0,
  verdict: "no_data",
  verdict_label: "Not enough production predictions",
  sample_count: 3,
  training_count: 0,
  summary: "Only 3 production predictions recorded — need at least 10 to compute PSI.",
}

// ---------------------------------------------------------------------------
// Component tests
// ---------------------------------------------------------------------------

describe("FeaturePsiCard", () => {
  it("renders stable verdict badge", () => {
    render(<FeaturePsiCard result={stableResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("All features stable")
  })

  it("renders critical verdict badge", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("1 feature with major shift")
  })

  it("renders critical count", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByTestId("critical-count")).toHaveTextContent("1")
  })

  it("renders watch count", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByTestId("watch-count")).toHaveTextContent("1")
  })

  it("renders stable count", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByTestId("stable-count")).toHaveTextContent("1")
  })

  it("renders feature list", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByTestId("feature-list")).toBeInTheDocument()
  })

  it("renders each feature name", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByText("price")).toBeInTheDocument()
    expect(screen.getByText("region")).toBeInTheDocument()
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("renders overall PSI value", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByTestId("overall-psi")).toHaveTextContent("0.177")
  })

  it("renders summary text", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByTestId("summary")).toBeInTheDocument()
  })

  it("renders no_data fallback without feature list", () => {
    render(<FeaturePsiCard result={noDataResult} />)
    expect(screen.queryByTestId("feature-list")).not.toBeInTheDocument()
  })

  it("renders no_data summary message", () => {
    render(<FeaturePsiCard result={noDataResult} />)
    expect(screen.getAllByText(/Only 3 production predictions/).length).toBeGreaterThanOrEqual(1)
  })

  it("shows psi label for each feature (stable result)", () => {
    render(<FeaturePsiCard result={stableResult} />)
    expect(screen.getByLabelText(/PSI severity: Stable/)).toBeInTheDocument()
  })

  it("shows psi label for critical entry", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByLabelText(/PSI severity: Critical/)).toBeInTheDocument()
  })

  it("shows psi label for watch entry", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByLabelText(/PSI severity: Watch/)).toBeInTheDocument()
  })

  it("shows new categories annotation for categorical with n_new > 0", () => {
    render(<FeaturePsiCard result={criticalResult} />)
    expect(screen.getByText("+1 new")).toBeInTheDocument()
  })

  it("has sr-only figcaption for accessibility", () => {
    const { container } = render(<FeaturePsiCard result={stableResult} />)
    const caption = container.querySelector("figcaption")
    expect(caption).not.toBeNull()
    expect(caption?.className).toContain("sr-only")
  })

  it("renders PSI legend items", () => {
    render(<FeaturePsiCard result={stableResult} />)
    expect(screen.getByText(/Stable: PSI/)).toBeInTheDocument()
    expect(screen.getByText(/Watch: PSI/)).toBeInTheDocument()
    expect(screen.getByText(/Critical: PSI/)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Store action tests
// ---------------------------------------------------------------------------

describe("attachFeaturePsiToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches feature_psi to the last assistant message", () => {
    act(() => {
      useAppStore.setState({
        messages: [{ role: "assistant", content: "Hello" }],
      })
      useAppStore.getState().attachFeaturePsiToLastMessage(stableResult)
    })
    const msgs = useAppStore.getState().messages
    expect(msgs[0].feature_psi).toEqual(stableResult)
  })

  it("does not attach when last message is from user", () => {
    act(() => {
      useAppStore.setState({
        messages: [{ role: "user", content: "Hello" }],
      })
      useAppStore.getState().attachFeaturePsiToLastMessage(stableResult)
    })
    const msgs = useAppStore.getState().messages
    expect(msgs[0].feature_psi).toBeUndefined()
  })
})
