/**
 * Tests for ExecutiveBriefingCard and Zustand store action.
 *
 * Covers:
 *  1.  Renders region with aria-label "Executive briefing"
 *  2.  Shows 📋 icon (aria-hidden)
 *  3.  Shows "Executive Briefing" heading
 *  4.  Shows algorithm badge
 *  5.  Shows metric label badge
 *  6.  Renders one-line summary text
 *  7.  Renders each section heading
 *  8.  Renders each section body text
 *  9.  Renders "Recommended Actions" heading when action_items present
 * 10.  Renders action item text
 * 11.  Shows "Open prediction dashboard" link when prediction_url present
 * 12.  Shows deploy-prompt when no prediction_url
 * 13.  Shows "Copy to clipboard" button
 * 14.  Store: attachExecutiveBriefingToLastMessage attaches to last assistant message
 * 15.  Store: does not attach to user message
 * 16.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ExecutiveBriefingCard } from "@/components/deploy/executive-briefing-card"
import type { ExecutiveBriefingResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const fullBriefing: ExecutiveBriefingResult = {
  project_id: "proj-1",
  project_name: "Sales Forecast",
  target_column: "revenue",
  problem_type: "regression",
  algorithm: "Random Forest",
  metric_label: "R² = 0.870",
  metric_value: 0.87,
  sections: [
    {
      heading: "What We Analyzed",
      body: "We analyzed sales_data.csv (200 rows across 8 columns) to predict revenue.",
    },
    {
      heading: "How Accurate Is It?",
      body: "The model achieves an R² of 0.87 — it explains 87% of the variation.",
    },
  ],
  summary: 'AutoModeler built a Random Forest model for "Sales Forecast" to predict revenue.',
  action_items: [
    "Use the prediction dashboard to run scenarios.",
    "Schedule a quarterly model review.",
  ],
  prediction_url: "http://localhost:3000/predict/abc123",
}

const minimalBriefing: ExecutiveBriefingResult = {
  project_id: "proj-2",
  project_name: "Empty Project",
  target_column: null,
  problem_type: null,
  algorithm: null,
  metric_label: null,
  metric_value: null,
  sections: [],
  summary: 'AutoModeler built a Machine Learning Model for "Empty Project".',
  action_items: ["Schedule a quarterly model review."],
  prediction_url: null,
}

describe("ExecutiveBriefingCard", () => {
  it("renders region with aria-label", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    expect(screen.getByRole("region", { name: /executive briefing/i })).toBeInTheDocument()
  })

  it("shows 📋 icon as aria-hidden", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    const icon = screen.getByText("📋")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("shows Executive Briefing heading", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    expect(screen.getByText("Executive Briefing")).toBeInTheDocument()
  })

  it("shows algorithm badge", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    expect(screen.getByText("Random Forest")).toBeInTheDocument()
  })

  it("shows metric label badge", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    expect(screen.getByText("R² = 0.870")).toBeInTheDocument()
  })

  it("renders one-line summary text", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    expect(
      screen.getByText(/AutoModeler built a Random Forest model/)
    ).toBeInTheDocument()
  })

  it("renders section headings", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    expect(screen.getByText("What We Analyzed")).toBeInTheDocument()
    expect(screen.getByText("How Accurate Is It?")).toBeInTheDocument()
  })

  it("renders section body text", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    expect(screen.getByText(/We analyzed sales_data.csv/)).toBeInTheDocument()
  })

  it("renders Recommended Actions heading when action_items present", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    expect(screen.getByText("Recommended Actions")).toBeInTheDocument()
  })

  it("renders action item text", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    expect(screen.getByText("Schedule a quarterly model review.")).toBeInTheDocument()
  })

  it("shows prediction dashboard link when prediction_url present", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    const link = screen.getByRole("link", { name: /open prediction dashboard/i })
    expect(link).toHaveAttribute("href", "http://localhost:3000/predict/abc123")
  })

  it("shows deploy-prompt when no prediction_url", () => {
    render(<ExecutiveBriefingCard briefing={minimalBriefing} />)
    expect(screen.getByText(/deploy the model/i)).toBeInTheDocument()
  })

  it("shows Copy to clipboard button", () => {
    render(<ExecutiveBriefingCard briefing={fullBriefing} />)
    expect(screen.getByRole("button", { name: /copy briefing to clipboard/i })).toBeInTheDocument()
  })
})

describe("Store: attachExecutiveBriefingToLastMessage", () => {
  beforeEach(() => {
    useAppStore.setState({ messages: [] })
  })

  it("attaches executive_briefing to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "write a briefing", id: "1" },
        { role: "assistant", content: "Here is your briefing.", id: "2" },
      ],
    })
    useAppStore.getState().attachExecutiveBriefingToLastMessage(fullBriefing)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].executive_briefing).toEqual(fullBriefing)
  })

  it("does not attach when last message is from user", () => {
    useAppStore.setState({
      messages: [
        { role: "assistant", content: "Hello", id: "1" },
        { role: "user", content: "write a briefing", id: "2" },
      ],
    })
    useAppStore.getState().attachExecutiveBriefingToLastMessage(fullBriefing)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].executive_briefing).toBeUndefined()
  })

  it("does not crash when messages list is empty", () => {
    useAppStore.setState({ messages: [] })
    expect(() => {
      useAppStore.getState().attachExecutiveBriefingToLastMessage(fullBriefing)
    }).not.toThrow()
  })
})
