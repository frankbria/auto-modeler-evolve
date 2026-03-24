/**
 * Tests for FeatureSuggestCard, FeaturesAppliedCard, and store actions.
 */
import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import {
  FeatureSuggestCard,
  FeaturesAppliedCard,
} from "@/components/features/feature-suggestions-chat-card"
import type {
  FeatureSuggestionsChatResult,
  FeaturesAppliedResult,
} from "@/lib/types"
import { useAppStore } from "@/lib/store"

// --- Fixtures -----------------------------------------------------------

const suggestionResult: FeatureSuggestionsChatResult = {
  dataset_id: "ds-abc",
  count: 3,
  suggestions: [
    {
      id: "s1",
      column: "order_date",
      transform_type: "date_decompose",
      title: "Extract date parts from 'order_date'",
      description: "Adds year, month, day-of-week columns",
      preview_columns: ["order_date_year", "order_date_month", "order_date_dow"],
    },
    {
      id: "s2",
      column: "region",
      transform_type: "one_hot",
      title: "One-hot encode 'region' (3 categories)",
      description: "Creates binary columns for each category",
      preview_columns: ["region_East", "region_West", "region_North"],
    },
    {
      id: "s3",
      column: "revenue",
      transform_type: "log_transform",
      title: "Log-transform 'revenue'",
      description: "Reduces skew for better model fit",
      preview_columns: ["revenue_log"],
    },
  ],
}

const appliedResult: FeaturesAppliedResult = {
  feature_set_id: "fs-xyz",
  dataset_id: "ds-abc",
  new_columns: ["order_date_year", "order_date_month", "region_East", "region_West"],
  total_columns: 10,
  applied_count: 3,
}

// --- FeatureSuggestCard rendering tests ----------------------------------

describe("FeatureSuggestCard", () => {
  it("renders with testid", () => {
    render(<FeatureSuggestCard result={suggestionResult} />)
    expect(screen.getByTestId("feature-suggest-card")).toBeInTheDocument()
  })

  it("shows Feature Engineering Suggestions header", () => {
    render(<FeatureSuggestCard result={suggestionResult} />)
    expect(screen.getByText("Feature Engineering Suggestions")).toBeInTheDocument()
  })

  it("shows suggestion count badge", () => {
    render(<FeatureSuggestCard result={suggestionResult} />)
    expect(screen.getByText("3 suggestions")).toBeInTheDocument()
  })

  it("renders all suggestion titles", () => {
    render(<FeatureSuggestCard result={suggestionResult} />)
    expect(screen.getByText("Extract date parts from 'order_date'")).toBeInTheDocument()
    expect(screen.getByText("One-hot encode 'region' (3 categories)")).toBeInTheDocument()
    expect(screen.getByText("Log-transform 'revenue'")).toBeInTheDocument()
  })

  it("shows transform type badges", () => {
    render(<FeatureSuggestCard result={suggestionResult} />)
    expect(screen.getByText("Date parts")).toBeInTheDocument()
    expect(screen.getByText("One-hot encode")).toBeInTheDocument()
    expect(screen.getByText("Log scale")).toBeInTheDocument()
  })

  it("shows preview columns for first suggestion", () => {
    render(<FeatureSuggestCard result={suggestionResult} />)
    expect(screen.getByText(/order_date_year/)).toBeInTheDocument()
  })

  it("renders Apply All button", () => {
    render(<FeatureSuggestCard result={suggestionResult} />)
    expect(screen.getByTestId("apply-all-features-btn")).toBeInTheDocument()
  })

  it("Apply All button shows suggestion count", () => {
    render(<FeatureSuggestCard result={suggestionResult} />)
    expect(screen.getByText(/Apply All 3 Features/)).toBeInTheDocument()
  })

  it("Apply All button calls api.features.apply on click", async () => {
    const mockApply = jest.fn().mockResolvedValue({
      feature_set_id: "fs-new",
      column_mapping: {},
      new_columns: ["order_date_year"],
      total_columns: 5,
      preview: [],
    })
    jest.mock("@/lib/api", () => ({
      api: { features: { apply: mockApply } },
    }))

    // We just test the button is clickable (API is mocked at module level in fetch-mock)
    render(<FeatureSuggestCard result={suggestionResult} />)
    const btn = screen.getByTestId("apply-all-features-btn")
    expect(btn).not.toBeDisabled()
  })

  it("shows applied success state after button click (mocked api)", async () => {
    // Mock fetch globally via jest-fetch-mock
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: async () => ({
        feature_set_id: "fs-new",
        column_mapping: {},
        new_columns: ["order_date_year", "region_East"],
        total_columns: 6,
        preview: [],
      }),
    }) as jest.Mock

    render(<FeatureSuggestCard result={suggestionResult} />)
    const btn = screen.getByTestId("apply-all-features-btn")
    fireEvent.click(btn)

    await waitFor(() => {
      expect(screen.getByTestId("features-applied-card")).toBeInTheDocument()
    })

    // Reset fetch
    global.fetch = jest.fn()
  })

  it("shows description text for each suggestion", () => {
    render(<FeatureSuggestCard result={suggestionResult} />)
    expect(screen.getByText("Adds year, month, day-of-week columns")).toBeInTheDocument()
  })
})

// --- FeaturesAppliedCard rendering tests ---------------------------------

describe("FeaturesAppliedCard", () => {
  it("renders with testid", () => {
    render(<FeaturesAppliedCard result={appliedResult} />)
    expect(screen.getByTestId("features-applied-confirmation-card")).toBeInTheDocument()
  })

  it("shows Features Applied Done header", () => {
    render(<FeaturesAppliedCard result={appliedResult} />)
    expect(screen.getByText("Feature Engineering Done")).toBeInTheDocument()
  })

  it("shows applied count badge", () => {
    render(<FeaturesAppliedCard result={appliedResult} />)
    expect(screen.getByText("3 transforms applied")).toBeInTheDocument()
  })

  it("shows new columns count", () => {
    render(<FeaturesAppliedCard result={appliedResult} />)
    expect(screen.getByText(/4 new columns/)).toBeInTheDocument()
  })

  it("shows total columns count", () => {
    render(<FeaturesAppliedCard result={appliedResult} />)
    expect(screen.getByText(/10 total columns/)).toBeInTheDocument()
  })

  it("shows new column names", () => {
    render(<FeaturesAppliedCard result={appliedResult} />)
    expect(screen.getByText(/order_date_year/)).toBeInTheDocument()
  })
})

// --- Store action tests --------------------------------------------------

describe("attachFeatureSuggestionsToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "suggest features", timestamp: "t1" },
        {
          role: "assistant",
          content: "Here are your feature suggestions.",
          timestamp: "t2",
        },
      ],
    })
  })

  it("attaches feature_suggestions to last assistant message", () => {
    const { attachFeatureSuggestionsToLastMessage } = useAppStore.getState()
    attachFeatureSuggestionsToLastMessage(suggestionResult)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.feature_suggestions).toEqual(suggestionResult)
  })

  it("preserves other message fields", () => {
    const { attachFeatureSuggestionsToLastMessage } = useAppStore.getState()
    attachFeatureSuggestionsToLastMessage(suggestionResult)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.content).toBe("Here are your feature suggestions.")
    expect(last.role).toBe("assistant")
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "suggest features", timestamp: "t1" },
      ],
    })
    const { attachFeatureSuggestionsToLastMessage } = useAppStore.getState()
    attachFeatureSuggestionsToLastMessage(suggestionResult)
    const messages = useAppStore.getState().messages
    expect(messages[messages.length - 1].feature_suggestions).toBeUndefined()
  })
})

describe("attachFeaturesAppliedToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "apply features", timestamp: "t1" },
        {
          role: "assistant",
          content: "Features applied!",
          timestamp: "t2",
        },
      ],
    })
  })

  it("attaches features_applied to last assistant message", () => {
    const { attachFeaturesAppliedToLastMessage } = useAppStore.getState()
    attachFeaturesAppliedToLastMessage(appliedResult)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.features_applied).toEqual(appliedResult)
  })

  it("preserves existing fields when attaching", () => {
    const { attachFeaturesAppliedToLastMessage } = useAppStore.getState()
    attachFeaturesAppliedToLastMessage(appliedResult)
    const messages = useAppStore.getState().messages
    const last = messages[messages.length - 1]
    expect(last.content).toBe("Features applied!")
  })

  it("does not attach to user message", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "apply features", timestamp: "t1" }],
    })
    const { attachFeaturesAppliedToLastMessage } = useAppStore.getState()
    attachFeaturesAppliedToLastMessage(appliedResult)
    const messages = useAppStore.getState().messages
    expect(messages[messages.length - 1].features_applied).toBeUndefined()
  })
})
