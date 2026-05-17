/**
 * Tests for RollbackChatCard and attachRollbackChatToLastMessage store action.
 *
 * Covers:
 *  1.  Renders region with aria-label "Deployment version card"
 *  2.  Shows "Deployment Versions" heading in list mode
 *  3.  Shows "Rollback Complete" heading after rollback
 *  4.  Shows version count badge
 *  5.  Shows "Rolled back to vN" badge after rollback
 *  6.  Shows error alert with role="alert" when error_message is set
 *  7.  Shows "Endpoint URL unchanged" note after rollback
 *  8.  Renders version rows with version number
 *  9.  Shows "Current" badge on current version
 * 10.  Shows "Restored" badge on rolled-back version
 * 11.  Shows algorithm name (formatted)
 * 12.  Shows metric display when present
 * 13.  Shows "To roll back" footer hint in list mode with multiple versions
 * 14.  Shows "No previous versions" text in list mode with 1 version
 * 15.  Shows "Current version is now vN" footer after rollback
 * 16.  Store: attachRollbackChatToLastMessage attaches to last assistant message
 * 17.  Store: does not attach to user message
 * 18.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { RollbackChatCard } from "@/components/deploy/rollback-chat-card"
import type { RollbackChatResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const listResultSingle: RollbackChatResult = {
  action: "list",
  deployment_id: "dep-1",
  current_version_number: 1,
  rolled_back: false,
  rolled_back_to_version: null,
  new_version_number: null,
  error_message: null,
  total_versions: 1,
  versions: [
    {
      version_number: 1,
      algorithm: "linear_regression",
      is_current: true,
      deployed_at: "2024-01-15T10:00:00Z",
      metric_display: "R²=0.91",
    },
  ],
}

const listResultMultiple: RollbackChatResult = {
  action: "list",
  deployment_id: "dep-1",
  current_version_number: 2,
  rolled_back: false,
  rolled_back_to_version: null,
  new_version_number: null,
  error_message: null,
  total_versions: 2,
  versions: [
    {
      version_number: 1,
      algorithm: "linear_regression",
      is_current: false,
      deployed_at: "2024-01-10T10:00:00Z",
      metric_display: "R²=0.88",
    },
    {
      version_number: 2,
      algorithm: "random_forest",
      is_current: true,
      deployed_at: "2024-01-15T10:00:00Z",
      metric_display: "R²=0.94",
    },
  ],
}

const rollbackResult: RollbackChatResult = {
  action: "rollback",
  deployment_id: "dep-1",
  current_version_number: 3,
  rolled_back: true,
  rolled_back_to_version: 1,
  new_version_number: 3,
  error_message: null,
  total_versions: 3,
  versions: [
    {
      version_number: 1,
      algorithm: "linear_regression",
      is_current: true,
      deployed_at: "2024-01-10T10:00:00Z",
      metric_display: "R²=0.88",
    },
    {
      version_number: 2,
      algorithm: "random_forest",
      is_current: false,
      deployed_at: "2024-01-15T10:00:00Z",
      metric_display: "R²=0.94",
    },
    {
      version_number: 3,
      algorithm: "linear_regression",
      is_current: true,
      deployed_at: "2024-01-16T10:00:00Z",
      metric_display: "R²=0.88",
    },
  ],
}

const errorResult: RollbackChatResult = {
  action: "rollback",
  deployment_id: "dep-1",
  current_version_number: 1,
  rolled_back: false,
  rolled_back_to_version: null,
  new_version_number: null,
  error_message: "No previous version to roll back to.",
  total_versions: 1,
  versions: [
    {
      version_number: 1,
      algorithm: "linear_regression",
      is_current: true,
      deployed_at: "2024-01-10T10:00:00Z",
      metric_display: null,
    },
  ],
}

// ---------------------------------------------------------------------------
// Tests — RollbackChatCard rendering
// ---------------------------------------------------------------------------

test("1. renders region with aria-label", () => {
  render(<RollbackChatCard result={listResultSingle} />)
  expect(screen.getByRole("region", { name: "Deployment version card" })).toBeInTheDocument()
})

test("2. shows Deployment Versions heading in list mode", () => {
  render(<RollbackChatCard result={listResultSingle} />)
  expect(screen.getByText("Deployment Versions")).toBeInTheDocument()
})

test("3. shows Rollback Complete heading after rollback", () => {
  render(<RollbackChatCard result={rollbackResult} />)
  expect(screen.getByText("Rollback Complete")).toBeInTheDocument()
})

test("4. shows version count badge", () => {
  render(<RollbackChatCard result={listResultMultiple} />)
  expect(screen.getByText("2 versions")).toBeInTheDocument()
})

test("5. shows rolled back badge after rollback", () => {
  render(<RollbackChatCard result={rollbackResult} />)
  expect(screen.getByText(/Rolled back to v1/)).toBeInTheDocument()
})

test("6. shows error alert with role=alert when error_message is set", () => {
  render(<RollbackChatCard result={errorResult} />)
  const alert = screen.getByRole("alert")
  expect(alert).toHaveTextContent("No previous version to roll back to.")
})

test("7. shows Endpoint URL unchanged note after rollback", () => {
  render(<RollbackChatCard result={rollbackResult} />)
  expect(screen.getByText(/Endpoint URL unchanged/)).toBeInTheDocument()
})

test("8. renders version rows with version number", () => {
  render(<RollbackChatCard result={listResultMultiple} />)
  expect(screen.getByTestId("version-row-1")).toBeInTheDocument()
  expect(screen.getByTestId("version-row-2")).toBeInTheDocument()
})

test("9. shows Current badge on current version", () => {
  render(<RollbackChatCard result={listResultMultiple} />)
  expect(screen.getByText("Current")).toBeInTheDocument()
})

test("10. shows Restored badge on rolled-back version", () => {
  render(<RollbackChatCard result={rollbackResult} />)
  expect(screen.getByText(/Restored/)).toBeInTheDocument()
})

test("11. shows algorithm name formatted", () => {
  render(<RollbackChatCard result={listResultSingle} />)
  expect(screen.getByText("Linear Regression")).toBeInTheDocument()
})

test("12. shows metric display when present", () => {
  render(<RollbackChatCard result={listResultSingle} />)
  expect(screen.getByText("R²=0.91")).toBeInTheDocument()
})

test("13. shows To roll back footer hint in list mode with multiple versions", () => {
  render(<RollbackChatCard result={listResultMultiple} />)
  expect(screen.getByText(/To roll back/)).toBeInTheDocument()
})

test("14. shows No previous versions text in list mode with 1 version", () => {
  render(<RollbackChatCard result={listResultSingle} />)
  expect(screen.getByText(/No previous versions to roll back to/)).toBeInTheDocument()
})

test("15. shows Current version is now footer after rollback", () => {
  render(<RollbackChatCard result={rollbackResult} />)
  expect(screen.getByText(/Current version is now v3/)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Tests — Zustand store actions
// ---------------------------------------------------------------------------

beforeEach(() => {
  useAppStore.setState({ messages: [] })
})

test("16. store: attachRollbackChatToLastMessage attaches to last assistant message", () => {
  useAppStore.setState({
    messages: [
      { id: "1", role: "user", content: "show my versions" },
      { id: "2", role: "assistant", content: "Here are your versions." },
    ],
  })
  useAppStore.getState().attachRollbackChatToLastMessage(listResultSingle)
  const msgs = useAppStore.getState().messages
  expect(msgs[1].rollback_chat).toEqual(listResultSingle)
})

test("17. store: does not attach to user message", () => {
  useAppStore.setState({
    messages: [{ id: "1", role: "user", content: "roll back to version 1" }],
  })
  useAppStore.getState().attachRollbackChatToLastMessage(listResultSingle)
  const msgs = useAppStore.getState().messages
  expect(msgs[0].rollback_chat).toBeUndefined()
})

test("18. store: does not crash when messages list is empty", () => {
  useAppStore.setState({ messages: [] })
  expect(() =>
    useAppStore.getState().attachRollbackChatToLastMessage(listResultSingle)
  ).not.toThrow()
})
