/**
 * Tests for the Analysis Template cards.
 *
 * Covers:
 *   TemplateSavedCard:
 *     1. aria-label includes template name
 *     2. 💾 icon aria-hidden
 *     3. Shows template name
 *     4. Shows query count badge (singular/plural)
 *     5. Lists saved queries
 *     6. Shows replay hint text
 *
 *   TemplateListCard:
 *     7. Empty state shows "No templates" message
 *     8. Shows template count badge
 *     9. Shows each template name
 *     10. Shows query count per template
 *     11. Replay button calls onReplay callback
 *
 *   TemplateReplayCard:
 *     12. aria-label includes template name
 *     13. Shows template name in heading
 *     14. Shows query count badge
 *     15. Lists queries as clickable buttons
 *     16. Clicking a query calls onQueryClick
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import {
  TemplateSavedCard,
  TemplateListCard,
  TemplateReplayCard,
} from "../components/data/analysis-template-card"
import type { TemplateSavedInfo, TemplateListInfo, TemplateReplayInfo } from "../lib/types"

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const SAVED_INFO: TemplateSavedInfo = {
  id: "tpl-1",
  name: "Monthly Sales Review",
  queries: [
    "show me revenue by region",
    "what's the average revenue?",
    "which products are trending up?",
  ],
  query_count: 3,
}

const SAVED_INFO_SINGLE: TemplateSavedInfo = {
  id: "tpl-2",
  name: "Quick Check",
  queries: ["show me revenue by region"],
  query_count: 1,
}

const LIST_INFO: TemplateListInfo = {
  count: 2,
  templates: [
    { id: "tpl-1", name: "Monthly Sales Review", queries: ["q1", "q2"], query_count: 2, created_at: "2026-04-07T20:00:00" },
    { id: "tpl-2", name: "Quick Check", queries: ["q3"], query_count: 1, created_at: "2026-04-06T12:00:00" },
  ],
}

const LIST_INFO_EMPTY: TemplateListInfo = {
  count: 0,
  templates: [],
}

const REPLAY_INFO: TemplateReplayInfo = {
  id: "tpl-1",
  name: "Monthly Sales Review",
  queries: ["show me revenue by region", "what's the average revenue?"],
  query_count: 2,
}

// ---------------------------------------------------------------------------
// TemplateSavedCard tests
// ---------------------------------------------------------------------------

describe("TemplateSavedCard", () => {
  it("has aria-label including template name", () => {
    render(<TemplateSavedCard info={SAVED_INFO} />)
    expect(screen.getByRole("figure", { name: /Monthly Sales Review/i })).toBeInTheDocument()
  })

  it("icon is aria-hidden", () => {
    render(<TemplateSavedCard info={SAVED_INFO} />)
    const icon = screen.getByText("💾")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("shows template name in content", () => {
    render(<TemplateSavedCard info={SAVED_INFO} />)
    const nameElements = screen.getAllByText("Monthly Sales Review")
    expect(nameElements.length).toBeGreaterThanOrEqual(1)
  })

  it("shows plural query count badge", () => {
    render(<TemplateSavedCard info={SAVED_INFO} />)
    expect(screen.getByText("3 queries")).toBeInTheDocument()
  })

  it("shows singular query count badge", () => {
    render(<TemplateSavedCard info={SAVED_INFO_SINGLE} />)
    expect(screen.getByText("1 query")).toBeInTheDocument()
  })

  it("lists all saved queries", () => {
    render(<TemplateSavedCard info={SAVED_INFO} />)
    expect(screen.getByText(/show me revenue by region/)).toBeInTheDocument()
    expect(screen.getByText(/what's the average revenue/)).toBeInTheDocument()
  })

  it("shows replay hint text", () => {
    render(<TemplateSavedCard info={SAVED_INFO} />)
    expect(screen.getByText(/Replay anytime by saying/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// TemplateListCard tests
// ---------------------------------------------------------------------------

describe("TemplateListCard", () => {
  it("shows empty state when no templates", () => {
    render(<TemplateListCard info={LIST_INFO_EMPTY} />)
    expect(screen.getByText(/No templates saved yet/i)).toBeInTheDocument()
  })

  it("shows template count badge", () => {
    render(<TemplateListCard info={LIST_INFO} />)
    expect(screen.getByText("2 saved")).toBeInTheDocument()
  })

  it("shows each template name", () => {
    render(<TemplateListCard info={LIST_INFO} />)
    expect(screen.getByText("Monthly Sales Review")).toBeInTheDocument()
    expect(screen.getByText("Quick Check")).toBeInTheDocument()
  })

  it("shows query count per template", () => {
    render(<TemplateListCard info={LIST_INFO} />)
    expect(screen.getByText("2 queries")).toBeInTheDocument()
    expect(screen.getByText("1 query")).toBeInTheDocument()
  })

  it("replay button calls onReplay with template name", () => {
    const onReplay = jest.fn()
    render(<TemplateListCard info={LIST_INFO} onReplay={onReplay} />)
    const replayButtons = screen.getAllByRole("button", { name: /Replay/i })
    fireEvent.click(replayButtons[0])
    expect(onReplay).toHaveBeenCalledWith("Monthly Sales Review")
  })
})

// ---------------------------------------------------------------------------
// TemplateReplayCard tests
// ---------------------------------------------------------------------------

describe("TemplateReplayCard", () => {
  it("has aria-label including template name", () => {
    render(<TemplateReplayCard info={REPLAY_INFO} />)
    expect(screen.getByRole("figure", { name: /Monthly Sales Review/i })).toBeInTheDocument()
  })

  it("shows template name in heading", () => {
    render(<TemplateReplayCard info={REPLAY_INFO} />)
    expect(screen.getByText(/Replay: Monthly Sales Review/)).toBeInTheDocument()
  })

  it("shows query count badge", () => {
    render(<TemplateReplayCard info={REPLAY_INFO} />)
    expect(screen.getByText("2 queries")).toBeInTheDocument()
  })

  it("lists queries as clickable buttons", () => {
    render(<TemplateReplayCard info={REPLAY_INFO} />)
    const buttons = screen.getAllByRole("button")
    expect(buttons.length).toBe(2)
  })

  it("clicking a query fires onQueryClick with the query text", () => {
    const onQueryClick = jest.fn()
    render(<TemplateReplayCard info={REPLAY_INFO} onQueryClick={onQueryClick} />)
    const buttons = screen.getAllByRole("button")
    fireEvent.click(buttons[0])
    expect(onQueryClick).toHaveBeenCalledWith("show me revenue by region")
  })
})
