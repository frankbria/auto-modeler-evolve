/**
 * Tests for AlertRuleCard component (Day 61 — Custom Prediction Alert Rules).
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { AlertRuleCard } from "@/components/chat/alert-rule-card"
import type { AlertRuleEventResult, AlertRuleEntry } from "@/lib/types"

const CREATED_RESULT: AlertRuleEventResult = {
  action: "created",
  deployment_id: "dep-1",
  rule_id: "rule-1",
  name: "Low revenue alert",
  condition_type: "prediction_value",
  condition_op: "lt",
  condition_value: 100000,
  condition_class: null,
  description: "Fires when Prediction value < 100000",
  summary: "Alert rule 'Low revenue alert' created.",
}

const RULE_ENTRY: AlertRuleEntry = {
  id: "rule-1",
  name: "Low revenue alert",
  condition_type: "prediction_value",
  condition_op: "lt",
  condition_value: 100000,
  condition_class: null,
  trigger_count: 3,
  last_triggered_at: null,
  description: "Fires when Prediction value < 100000",
}

const LIST_RESULT: AlertRuleEventResult = {
  action: "list",
  deployment_id: "dep-1",
  count: 2,
  rules: [
    RULE_ENTRY,
    {
      id: "rule-2",
      name: "High confidence alert",
      condition_type: "confidence",
      condition_op: "gt",
      condition_value: 95,
      condition_class: null,
      trigger_count: 0,
      last_triggered_at: null,
      description: "Fires when Confidence > 95%",
    },
  ],
  summary: "2 active alert rules.",
}

const EMPTY_LIST_RESULT: AlertRuleEventResult = {
  action: "list",
  deployment_id: "dep-1",
  count: 0,
  rules: [],
  summary: "No alert rules configured.",
}

const DELETED_RESULT: AlertRuleEventResult = {
  action: "deleted",
  deployment_id: "dep-1",
  deleted_count: 2,
  deleted_names: ["Low revenue alert", "High confidence alert"],
  summary: "2 alert rules removed.",
}

// ---------------------------------------------------------------------------
// Created action
// ---------------------------------------------------------------------------

describe("AlertRuleCard — created", () => {
  it("renders 'Alert Rule Created' heading", () => {
    render(<AlertRuleCard result={CREATED_RESULT} />)
    expect(screen.getByText("Alert Rule Created")).toBeInTheDocument()
  })

  it("shows Active badge", () => {
    render(<AlertRuleCard result={CREATED_RESULT} />)
    expect(screen.getByText("Active")).toBeInTheDocument()
  })

  it("shows rule name", () => {
    render(<AlertRuleCard result={CREATED_RESULT} />)
    expect(screen.getByText("Low revenue alert")).toBeInTheDocument()
  })

  it("shows description", () => {
    render(<AlertRuleCard result={CREATED_RESULT} />)
    expect(
      screen.getByText("Fires when Prediction value < 100000")
    ).toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<AlertRuleCard result={CREATED_RESULT} />)
    expect(
      screen.getByText("Alert rule 'Low revenue alert' created.")
    ).toBeInTheDocument()
  })

  it("has accessible region label", () => {
    render(<AlertRuleCard result={CREATED_RESULT} />)
    expect(screen.getByRole("region", { name: /alert rule created/i })).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// List action
// ---------------------------------------------------------------------------

describe("AlertRuleCard — list", () => {
  it("renders 'Alert Rules' heading", () => {
    render(<AlertRuleCard result={LIST_RESULT} />)
    expect(screen.getByText("Alert Rules")).toBeInTheDocument()
  })

  it("shows count badge", () => {
    render(<AlertRuleCard result={LIST_RESULT} />)
    expect(screen.getByText("2")).toBeInTheDocument()
  })

  it("renders each rule name", () => {
    render(<AlertRuleCard result={LIST_RESULT} />)
    expect(screen.getByText("Low revenue alert")).toBeInTheDocument()
    expect(screen.getByText("High confidence alert")).toBeInTheDocument()
  })

  it("shows trigger count badges", () => {
    render(<AlertRuleCard result={LIST_RESULT} />)
    expect(screen.getByText("3 fired")).toBeInTheDocument()
    expect(screen.getByText("0 fired")).toBeInTheDocument()
  })

  it("shows empty state when no rules", () => {
    render(<AlertRuleCard result={EMPTY_LIST_RESULT} />)
    expect(screen.getByText(/no alert rules active/i)).toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<AlertRuleCard result={LIST_RESULT} />)
    expect(screen.getAllByText("2 active alert rules.").length).toBeGreaterThan(0)
  })
})

// ---------------------------------------------------------------------------
// Deleted action
// ---------------------------------------------------------------------------

describe("AlertRuleCard — deleted", () => {
  it("renders 'Alert Rules Removed' heading", () => {
    render(<AlertRuleCard result={DELETED_RESULT} />)
    expect(screen.getByText("Alert Rules Removed")).toBeInTheDocument()
  })

  it("shows deleted count badge", () => {
    render(<AlertRuleCard result={DELETED_RESULT} />)
    expect(screen.getByText("2 removed")).toBeInTheDocument()
  })

  it("lists deleted rule names", () => {
    render(<AlertRuleCard result={DELETED_RESULT} />)
    expect(screen.getByText("Low revenue alert")).toBeInTheDocument()
    expect(screen.getByText("High confidence alert")).toBeInTheDocument()
  })

  it("shows summary", () => {
    render(<AlertRuleCard result={DELETED_RESULT} />)
    expect(screen.getAllByText("2 alert rules removed.").length).toBeGreaterThan(0)
  })
})
