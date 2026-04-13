/**
 * Tests for ScheduleSetChatCard and Zustand store action.
 *
 * Covers:
 *  1.  Renders region with aria-label "Batch schedule created" (action=created)
 *  2.  Shows 🗓️ icon (aria-hidden)
 *  3.  Shows "Batch Schedule Created" heading
 *  4.  Shows frequency badge (Daily/Weekly/Monthly)
 *  5.  Shows description text
 *  6.  Shows next run timestamp
 *  7.  Shows help text footer
 *  8.  Renders region with aria-label "Batch schedules" (action=list)
 *  9.  Shows schedule count badge for list
 * 10.  Shows schedule rows for non-empty list
 * 11.  Shows empty-state message when no schedules
 * 12.  Store: attachScheduleSetToLastMessage attaches to last assistant message
 * 13.  Store: does not attach to user message
 * 14.  Store: does not crash when messages list is empty
 */

import React from "react"
import { render, screen } from "@testing-library/react"
import { ScheduleSetChatCard } from "@/components/deploy/schedule-set-chat-card"
import type { ScheduleSetResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const createdResult: ScheduleSetResult = {
  action: "created",
  deployment_id: "dep-1",
  schedule_id: "sched-1",
  frequency: "daily",
  run_hour: 9,
  run_minute: 0,
  day_of_week: null,
  day_of_month: null,
  next_run: "2026-04-14T09:00:00",
  description: "Every day at 09:00 UTC",
  summary: "Batch predictions scheduled: Every day at 09:00 UTC.",
}

const weeklyResult: ScheduleSetResult = {
  action: "created",
  deployment_id: "dep-2",
  schedule_id: "sched-2",
  frequency: "weekly",
  run_hour: 8,
  run_minute: 0,
  day_of_week: 0,
  day_of_month: null,
  next_run: "2026-04-14T08:00:00",
  description: "Every Monday at 08:00 UTC",
  summary: "Batch predictions scheduled: Every Monday at 08:00 UTC.",
}

const listWithSchedules: ScheduleSetResult = {
  action: "list",
  deployment_id: "dep-3",
  count: 2,
  schedules: [
    {
      id: "s1",
      frequency: "daily",
      run_hour: 9,
      run_minute: 0,
      day_of_week: null,
      day_of_month: null,
      next_run: "2026-04-14T09:00:00",
      last_run: null,
      last_row_count: null,
      description: "Every day at 09:00 UTC",
    },
    {
      id: "s2",
      frequency: "weekly",
      run_hour: 14,
      run_minute: 30,
      day_of_week: 2,
      day_of_month: null,
      next_run: "2026-04-15T14:30:00",
      last_run: "2026-04-08T14:30:00",
      last_row_count: 150,
      description: "Every Wednesday at 14:30 UTC",
    },
  ],
  summary: "You have 2 batch schedule(s).",
}

const emptyList: ScheduleSetResult = {
  action: "list",
  deployment_id: "dep-4",
  count: 0,
  schedules: [],
  summary: "No batch schedules configured yet.",
}

// ---------------------------------------------------------------------------
// Tests — ScheduleSetChatCard rendering (action=created)
// ---------------------------------------------------------------------------

test("1. renders region with aria-label 'Batch schedule created'", () => {
  render(<ScheduleSetChatCard result={createdResult} />)
  expect(screen.getByRole("region", { name: "Batch schedule created" })).toBeInTheDocument()
})

test("2. shows calendar icon with aria-hidden", () => {
  render(<ScheduleSetChatCard result={createdResult} />)
  const icon = screen.getByText("🗓️")
  expect(icon).toHaveAttribute("aria-hidden", "true")
})

test("3. shows 'Batch Schedule Created' heading", () => {
  render(<ScheduleSetChatCard result={createdResult} />)
  expect(screen.getByText("Batch Schedule Created")).toBeInTheDocument()
})

test("4. shows frequency badge 'Daily'", () => {
  render(<ScheduleSetChatCard result={createdResult} />)
  expect(screen.getByText("Daily")).toBeInTheDocument()
})

test("4b. shows frequency badge 'Weekly'", () => {
  render(<ScheduleSetChatCard result={weeklyResult} />)
  expect(screen.getByText("Weekly")).toBeInTheDocument()
})

test("5. shows schedule description text", () => {
  render(<ScheduleSetChatCard result={createdResult} />)
  expect(screen.getByText("Every day at 09:00 UTC")).toBeInTheDocument()
})

test("6. shows next run label", () => {
  render(<ScheduleSetChatCard result={createdResult} />)
  expect(screen.getByText(/Next run:/)).toBeInTheDocument()
})

test("7. shows help text footer about Deployment panel", () => {
  render(<ScheduleSetChatCard result={createdResult} />)
  expect(screen.getByText(/Deployment panel/)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Tests — action=list
// ---------------------------------------------------------------------------

test("8. renders region with aria-label 'Batch schedules' for list", () => {
  render(<ScheduleSetChatCard result={listWithSchedules} />)
  expect(screen.getByRole("region", { name: "Batch schedules" })).toBeInTheDocument()
})

test("9. shows count badge for list", () => {
  render(<ScheduleSetChatCard result={listWithSchedules} />)
  expect(screen.getByText("2 schedules")).toBeInTheDocument()
})

test("10. shows schedule rows for non-empty list", () => {
  render(<ScheduleSetChatCard result={listWithSchedules} />)
  expect(screen.getByText("Every day at 09:00 UTC")).toBeInTheDocument()
  expect(screen.getByText("Every Wednesday at 14:30 UTC")).toBeInTheDocument()
})

test("11. shows empty-state message when no schedules", () => {
  render(<ScheduleSetChatCard result={emptyList} />)
  expect(screen.getByText(/No batch schedules configured yet/)).toBeInTheDocument()
})

// ---------------------------------------------------------------------------
// Tests — Zustand store action
// ---------------------------------------------------------------------------

beforeEach(() => {
  useAppStore.setState({ messages: [] })
})

test("12. store attaches to last assistant message", () => {
  useAppStore.setState({
    messages: [
      { id: "m1", role: "assistant", content: "I'll set that up.", timestamp: "" },
    ],
  })
  useAppStore.getState().attachScheduleSetToLastMessage(createdResult)
  const msgs = useAppStore.getState().messages
  expect(msgs[0].schedule_set).toEqual(createdResult)
})

test("13. store does not attach to user message", () => {
  useAppStore.setState({
    messages: [
      { id: "m1", role: "user", content: "schedule daily predictions", timestamp: "" },
    ],
  })
  useAppStore.getState().attachScheduleSetToLastMessage(createdResult)
  const msgs = useAppStore.getState().messages
  expect(msgs[0].schedule_set).toBeUndefined()
})

test("14. store does not crash when messages list is empty", () => {
  useAppStore.setState({ messages: [] })
  expect(() =>
    useAppStore.getState().attachScheduleSetToLastMessage(createdResult)
  ).not.toThrow()
})
