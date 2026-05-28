import React from "react"
import { render, screen } from "@testing-library/react"
import { FeatureRedundancyCard } from "@/components/models/feature-redundancy-card"
import type { FeatureRedundancyResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

function makeResult(overrides: Partial<FeatureRedundancyResult> = {}): FeatureRedundancyResult {
  return {
    dataset_id: "ds-1",
    target_column: "revenue",
    redundant_pairs: [],
    redundant_groups: [],
    n_redundant: 0,
    n_features_checked: 5,
    threshold: 0.85,
    verdict: "none",
    verdict_label: "No Redundancy Detected",
    summary: "No redundant features found above the 85% threshold.",
    ...overrides,
  }
}

const pairA = {
  feature_a: "price",
  feature_b: "discounted_price",
  correlation: 0.95,
  correlation_abs: 0.95,
  direction: "positive" as const,
  keep: "price",
  drop: "discounted_price",
  reason:
    "price and discounted_price are 95% correlated — they carry nearly identical information.",
}

const pairB = {
  feature_a: "units",
  feature_b: "neg_units",
  correlation: -0.91,
  correlation_abs: 0.91,
  direction: "negative" as const,
  keep: "units",
  drop: "neg_units",
  reason: "units and neg_units are 91% correlated.",
}

describe("FeatureRedundancyCard", () => {
  it("renders the card container", () => {
    render(<FeatureRedundancyCard result={makeResult()} />)
    expect(screen.getByTestId("feature-redundancy-card")).toBeInTheDocument()
  })

  it("shows the verdict badge", () => {
    render(<FeatureRedundancyCard result={makeResult()} />)
    expect(screen.getByTestId("verdict-badge")).toHaveTextContent("No Redundancy Detected")
  })

  it("shows the features-checked badge", () => {
    render(<FeatureRedundancyCard result={makeResult({ n_features_checked: 7 })} />)
    expect(screen.getByTestId("features-checked-badge")).toHaveTextContent("7 features checked")
  })

  it("does not show redundant-count badge when no redundant features", () => {
    render(<FeatureRedundancyCard result={makeResult()} />)
    expect(screen.queryByTestId("redundant-count-badge")).not.toBeInTheDocument()
  })

  it("shows redundant-count badge when pairs found", () => {
    render(
      <FeatureRedundancyCard
        result={makeResult({ redundant_pairs: [pairA], n_redundant: 2 })}
      />
    )
    expect(screen.getByTestId("redundant-count-badge")).toHaveTextContent("2 redundant")
  })

  it("shows no-pairs message when verdict is none", () => {
    render(<FeatureRedundancyCard result={makeResult()} />)
    expect(screen.getByTestId("no-pairs-message")).toBeInTheDocument()
  })

  it("does not show no-pairs message when pairs exist", () => {
    render(
      <FeatureRedundancyCard
        result={makeResult({ redundant_pairs: [pairA], n_redundant: 2 })}
      />
    )
    expect(screen.queryByTestId("no-pairs-message")).not.toBeInTheDocument()
  })

  it("renders pair rows for each redundant pair", () => {
    render(
      <FeatureRedundancyCard
        result={makeResult({ redundant_pairs: [pairA, pairB], n_redundant: 3 })}
      />
    )
    expect(screen.getAllByTestId("redundant-pair-row")).toHaveLength(2)
  })

  it("shows feature_a and feature_b names in a pair row", () => {
    render(
      <FeatureRedundancyCard
        result={makeResult({ redundant_pairs: [pairA], n_redundant: 2 })}
      />
    )
    expect(screen.getByTestId("feature-a-name")).toHaveTextContent("price")
    expect(screen.getByTestId("feature-b-name")).toHaveTextContent("discounted_price")
  })

  it("shows direction badge", () => {
    render(
      <FeatureRedundancyCard
        result={makeResult({ redundant_pairs: [pairA], n_redundant: 2 })}
      />
    )
    expect(screen.getByTestId("direction-badge")).toHaveTextContent("positive")
  })

  it("shows negative direction badge for negative correlation", () => {
    render(
      <FeatureRedundancyCard
        result={makeResult({ redundant_pairs: [pairB], n_redundant: 2 })}
      />
    )
    expect(screen.getByTestId("direction-badge")).toHaveTextContent("negative")
  })

  it("shows keep and drop badges", () => {
    render(
      <FeatureRedundancyCard
        result={makeResult({ redundant_pairs: [pairA], n_redundant: 2 })}
      />
    )
    expect(screen.getByTestId("keep-badge")).toHaveTextContent("Keep: price")
    expect(screen.getByTestId("drop-badge")).toHaveTextContent("Drop: discounted_price")
  })

  it("shows drop recommendation when pairs found", () => {
    render(
      <FeatureRedundancyCard
        result={makeResult({ redundant_pairs: [pairA], n_redundant: 2 })}
      />
    )
    expect(screen.getByTestId("drop-recommendation")).toBeInTheDocument()
    expect(screen.getByTestId("drop-recommendation")).toHaveTextContent("discounted_price")
  })

  it("does not show drop recommendation when no pairs", () => {
    render(<FeatureRedundancyCard result={makeResult()} />)
    expect(screen.queryByTestId("drop-recommendation")).not.toBeInTheDocument()
  })

  it("shows the summary text", () => {
    render(
      <FeatureRedundancyCard
        result={makeResult({ summary: "Found 1 redundant pair." })}
      />
    )
    expect(screen.getByTestId("redundancy-summary")).toHaveTextContent("Found 1 redundant pair.")
  })

  it("renders sr-only figcaption for accessibility", () => {
    render(<FeatureRedundancyCard result={makeResult()} />)
    const fig = screen.getByRole("figure")
    const figcaption = fig.querySelector("figcaption")
    expect(figcaption).toBeInTheDocument()
    expect(figcaption).toHaveClass("sr-only")
  })

  it("figcaption mentions pair count", () => {
    render(
      <FeatureRedundancyCard
        result={makeResult({ redundant_pairs: [pairA], n_redundant: 2 })}
      />
    )
    const figcaption = screen.getByRole("figure").querySelector("figcaption")
    expect(figcaption?.textContent).toMatch(/1.*pair/)
  })

  it("renders null when result is not provided", () => {
    const { container } = render(<FeatureRedundancyCard result={null as unknown as FeatureRedundancyResult} />)
    expect(container.firstChild).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Zustand store action tests
// ---------------------------------------------------------------------------

describe("attachFeatureRedundancyToLastMessage store action", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "Are any features redundant?" },
        { role: "assistant", content: "Let me check." },
      ],
    })
  })

  it("attaches feature_redundancy to the last assistant message", () => {
    const result = makeResult({ verdict: "low", verdict_label: "Low Redundancy" })
    useAppStore.getState().attachFeatureRedundancyToLastMessage(result)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.feature_redundancy).toEqual(result)
  })

  it("does not attach to a user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "test" }],
    })
    const result = makeResult()
    useAppStore.getState().attachFeatureRedundancyToLastMessage(result)
    const messages = useAppStore.getState().messages
    expect(messages[0].feature_redundancy).toBeUndefined()
  })

  it("does not modify earlier messages", () => {
    const result = makeResult()
    useAppStore.getState().attachFeatureRedundancyToLastMessage(result)
    const messages = useAppStore.getState().messages
    expect(messages[0].feature_redundancy).toBeUndefined()
  })
})
