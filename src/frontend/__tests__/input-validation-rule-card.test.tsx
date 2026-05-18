/**
 * Tests for InputValidationRuleCard (chat-inline input validation rule card).
 *
 * Covers:
 *  1.  Renders region with aria-label "Input validation rule card"
 *  2.  Shows 🛡️ icon for "created" action
 *  3.  Shows "Validation Rule Added" heading for created action
 *  4.  Shows "Input Validation Rules" heading for list action
 *  5.  Shows "Validation Rules Removed" heading for deleted action
 *  6.  Shows "total_rules" badge when action is "created"
 *  7.  Shows rule detail (feature_name + rule_type) when created
 *  8.  Shows rule list rows when action is "list"
 *  9.  Shows empty state when list is empty
 * 10.  Shows "deleted_count" badge when action is "deleted"
 * 11.  Shows summary text
 * 12.  Shows footer hint text
 * 13.  RuleTypeBadge shows "Range" for "range" type
 * 14.  RuleTypeBadge shows "One of" for "one_of" type
 * 15.  RuleTypeBadge shows "Required" for "not_null" type
 * 16.  Store: attachInputValidationRuleToLastMessage attaches to last assistant message
 * 17.  Store: does not attach to user message
 * 18.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { InputValidationRuleCard } from "@/components/deploy/input-validation-rule-card"
import type { InputValidationRuleResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const createdResult: InputValidationRuleResult = {
  action: "created",
  deployment_id: "dep-1",
  rule_id: "rule-123",
  feature_name: "units",
  rule_type: "range",
  min_val: 1,
  max_val: 10000,
  description: "'units' must be between 1 and 10000.",
  total_rules: 2,
  summary: "Validation rule created: 'units' must be between 1 and 10000. The prediction API will now reject inputs that violate this rule.",
}

const listResult: InputValidationRuleResult = {
  action: "list",
  deployment_id: "dep-2",
  count: 2,
  rules: [
    {
      id: "r1",
      deployment_id: "dep-2",
      feature_name: "age",
      rule_type: "range",
      min_val: 0,
      max_val: 120,
      allowed_values: null,
      description: "'age' must be between 0 and 120.",
      created_at: "2026-05-17T10:00:00",
    },
    {
      id: "r2",
      deployment_id: "dep-2",
      feature_name: "region",
      rule_type: "one_of",
      min_val: null,
      max_val: null,
      allowed_values: ["East", "West"],
      description: "'region' must be one of: 'East', 'West'.",
      created_at: "2026-05-17T11:00:00",
    },
  ],
  summary: "2 validation rule(s) active on this deployment.",
}

const emptyListResult: InputValidationRuleResult = {
  action: "list",
  deployment_id: "dep-3",
  count: 0,
  rules: [],
  summary: "No input validation rules configured yet.",
}

const deletedResult: InputValidationRuleResult = {
  action: "deleted",
  deployment_id: "dep-4",
  deleted_count: 3,
  summary: "Removed 3 validation rule(s). The prediction API will now accept any input values.",
}

const guidanceResult: InputValidationRuleResult = {
  action: "guidance",
  deployment_id: "dep-5",
  summary: "To add a validation rule, try: 'validate that age is between 0 and 120'.",
}

// ---------------------------------------------------------------------------
// Aria / structure
// ---------------------------------------------------------------------------

test("renders region with aria-label", () => {
  render(<InputValidationRuleCard result={createdResult} />)
  expect(screen.getByRole("region", { name: /input validation rule card/i })).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Created action
// ---------------------------------------------------------------------------

test("shows shield icon for created action", () => {
  render(<InputValidationRuleCard result={createdResult} />)
  expect(screen.getByText("🛡️")).toBeInTheDocument()
})

test("shows 'Validation Rule Added' heading for created", () => {
  render(<InputValidationRuleCard result={createdResult} />)
  expect(screen.getByText("Validation Rule Added")).toBeInTheDocument()
})

test("shows total_rules badge when created", () => {
  render(<InputValidationRuleCard result={createdResult} />)
  expect(screen.getByText(/2 rules active/i)).toBeInTheDocument()
})

test("shows feature_name and rule_type when created", () => {
  render(<InputValidationRuleCard result={createdResult} />)
  expect(screen.getByText("units")).toBeInTheDocument()
  expect(screen.getAllByText("Range").length).toBeGreaterThan(0)
})

test("shows description when created", () => {
  render(<InputValidationRuleCard result={createdResult} />)
  expect(screen.getAllByText(/'units' must be between 1 and 10000\./i).length).toBeGreaterThan(0)
})

// ---------------------------------------------------------------------------
// List action
// ---------------------------------------------------------------------------

test("shows 'Input Validation Rules' heading for list", () => {
  render(<InputValidationRuleCard result={listResult} />)
  expect(screen.getByText("Input Validation Rules")).toBeInTheDocument()
})

test("shows count badge for list action", () => {
  render(<InputValidationRuleCard result={listResult} />)
  expect(screen.getByText("2 rules")).toBeInTheDocument()
})

test("renders rule rows for each rule in list", () => {
  render(<InputValidationRuleCard result={listResult} />)
  expect(screen.getByTestId("validation-rule-row-age")).toBeInTheDocument()
  expect(screen.getByTestId("validation-rule-row-region")).toBeInTheDocument()
})

test("shows empty state when list is empty", () => {
  render(<InputValidationRuleCard result={emptyListResult} />)
  expect(screen.getByText(/No validation rules configured/i)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Deleted action
// ---------------------------------------------------------------------------

test("shows 'Validation Rules Removed' heading for deleted", () => {
  render(<InputValidationRuleCard result={deletedResult} />)
  expect(screen.getByText("Validation Rules Removed")).toBeInTheDocument()
})

test("shows deleted_count badge when deleted", () => {
  render(<InputValidationRuleCard result={deletedResult} />)
  expect(screen.getByText(/3 removed/i)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Summary and footer
// ---------------------------------------------------------------------------

test("shows summary text", () => {
  render(<InputValidationRuleCard result={createdResult} />)
  expect(screen.getByText(/Validation rule created:/i)).toBeInTheDocument()
})

test("shows footer hint text for list action", () => {
  render(<InputValidationRuleCard result={listResult} />)
  expect(screen.getByText(/validate that age is between/i)).toBeInTheDocument()
})

test("shows API rejection note in footer for created action", () => {
  render(<InputValidationRuleCard result={createdResult} />)
  expect(screen.getByText(/422 error/i)).toBeInTheDocument()
})

test("shows guidance heading and hint for guidance action", () => {
  render(<InputValidationRuleCard result={guidanceResult} />)
  expect(screen.getByText("Validation Rule Guidance")).toBeInTheDocument()
  expect(screen.getAllByText(/validate that age is between/i).length).toBeGreaterThan(0)
})

// ---------------------------------------------------------------------------
// Rule type badge labels
// ---------------------------------------------------------------------------

test("RuleTypeBadge shows Range for range type", () => {
  render(<InputValidationRuleCard result={listResult} />)
  expect(screen.getAllByText("Range").length).toBeGreaterThan(0)
})

test("RuleTypeBadge shows One of for one_of type", () => {
  render(<InputValidationRuleCard result={listResult} />)
  expect(screen.getByText("One of")).toBeInTheDocument()
})

test("RuleTypeBadge shows Required for not_null type", () => {
  const notNullResult: InputValidationRuleResult = {
    action: "created",
    deployment_id: "dep-6",
    rule_id: "rule-xyz",
    feature_name: "customer_id",
    rule_type: "not_null",
    description: "'customer_id' must be provided.",
    total_rules: 1,
    summary: "Validation rule created.",
  }
  render(<InputValidationRuleCard result={notNullResult} />)
  expect(screen.getByText("Required")).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Zustand store
// ---------------------------------------------------------------------------

function resetStore() {
  useAppStore.setState({ messages: [] })
}

test("attachInputValidationRuleToLastMessage attaches to last assistant message", () => {
  resetStore()
  useAppStore.getState().addMessage({ role: "user", content: "show my rules" })
  useAppStore.getState().addMessage({ role: "assistant", content: "Here are your rules." })
  useAppStore.getState().attachInputValidationRuleToLastMessage(listResult)
  const msgs = useAppStore.getState().messages
  expect(msgs[msgs.length - 1].input_validation_rule).toEqual(listResult)
})

test("does not attach when last message is from user", () => {
  resetStore()
  useAppStore.getState().addMessage({ role: "user", content: "validate that units is between 1 and 100" })
  useAppStore.getState().attachInputValidationRuleToLastMessage(createdResult)
  const msgs = useAppStore.getState().messages
  expect(msgs[msgs.length - 1].input_validation_rule).toBeUndefined()
})

test("does not crash when messages list is empty", () => {
  resetStore()
  expect(() =>
    useAppStore.getState().attachInputValidationRuleToLastMessage(createdResult)
  ).not.toThrow()
})
