/**
 * Tests for ScheduleCard component.
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { ScheduleCard } from "@/components/deploy/schedule-card"
import { api } from "@/lib/api"
import type { BatchSchedule, BatchJobRun } from "@/lib/types"

jest.mock("@/lib/api")

const mockSchedule: BatchSchedule = {
  id: "sched-1",
  deployment_id: "dep-1",
  frequency: "daily",
  run_hour: 9,
  run_minute: 0,
  day_of_week: null,
  day_of_month: null,
  is_active: true,
  last_run: null,
  next_run: "2024-03-20T09:00:00",
  last_output_path: null,
  last_row_count: null,
  last_error: null,
  created_at: "2024-03-15T12:00:00",
}

const mockJobRun: BatchJobRun = {
  id: "run-1",
  schedule_id: "sched-1",
  deployment_id: "dep-1",
  started_at: "2024-03-19T09:00:00",
  completed_at: "2024-03-19T09:00:05",
  status: "success",
  row_count: 42,
  error: null,
  download_url: "/api/deploy/batch-outputs/sched-1_20240319_090000.csv",
}

describe("ScheduleCard", () => {
  beforeEach(() => {
    jest.clearAllMocks()
    ;(api.deploy.getSchedules as jest.Mock).mockResolvedValue([])
    ;(api.deploy.createSchedule as jest.Mock).mockResolvedValue(mockSchedule)
    ;(api.deploy.deleteSchedule as jest.Mock).mockResolvedValue(undefined)
    ;(api.deploy.triggerSchedule as jest.Mock).mockResolvedValue({
      status: "running",
      schedule_id: "sched-1",
    })
    ;(api.deploy.getScheduleRuns as jest.Mock).mockResolvedValue([mockJobRun])
  })

  it("renders header and empty state", async () => {
    render(<ScheduleCard deploymentId="dep-1" />)
    await waitFor(() => {
      expect(screen.getByText("Scheduled Batch Predictions")).toBeInTheDocument()
    })
    expect(
      screen.getByText(/No schedules yet/)
    ).toBeInTheDocument()
  })

  it("shows create form when Add Schedule is clicked", async () => {
    render(<ScheduleCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByText("Scheduled Batch Predictions"))

    fireEvent.click(screen.getByTestId("add-schedule-btn"))
    expect(screen.getByTestId("schedule-form")).toBeInTheDocument()
    expect(screen.getByTestId("frequency-select")).toBeInTheDocument()
    expect(screen.getByTestId("hour-select")).toBeInTheDocument()
    expect(screen.getByTestId("minute-select")).toBeInTheDocument()
  })

  it("hides form when Cancel is clicked", async () => {
    render(<ScheduleCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByText("Scheduled Batch Predictions"))

    fireEvent.click(screen.getByTestId("add-schedule-btn"))
    expect(screen.getByTestId("schedule-form")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("add-schedule-btn")) // now shows "Cancel"
    await waitFor(() => {
      expect(screen.queryByTestId("schedule-form")).not.toBeInTheDocument()
    })
  })

  it("creates a schedule and adds it to the list", async () => {
    render(<ScheduleCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByText("Scheduled Batch Predictions"))

    fireEvent.click(screen.getByTestId("add-schedule-btn"))
    fireEvent.click(screen.getByTestId("create-schedule-btn"))

    await waitFor(() => {
      expect(api.deploy.createSchedule).toHaveBeenCalledWith("dep-1", expect.objectContaining({
        frequency: "daily",
        run_hour: 9,
        run_minute: 0,
      }))
    })
    // The new schedule should appear
    await waitFor(() => {
      expect(screen.getByText(/Every day at/)).toBeInTheDocument()
    })
  })

  it("renders existing schedules", async () => {
    ;(api.deploy.getSchedules as jest.Mock).mockResolvedValue([mockSchedule])
    render(<ScheduleCard deploymentId="dep-1" />)

    await waitFor(() => {
      expect(screen.getByText(/Every day at/)).toBeInTheDocument()
    })
    expect(screen.getAllByText("Active").length).toBeGreaterThan(0)
  })

  it("shows weekly day selector when frequency is weekly", async () => {
    render(<ScheduleCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByText("Scheduled Batch Predictions"))

    fireEvent.click(screen.getByTestId("add-schedule-btn"))
    const select = screen.getByTestId("frequency-select")
    fireEvent.change(select, { target: { value: "weekly" } })

    expect(screen.getByTestId("day-of-week-select")).toBeInTheDocument()
  })

  it("shows day-of-month input when frequency is monthly", async () => {
    render(<ScheduleCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByText("Scheduled Batch Predictions"))

    fireEvent.click(screen.getByTestId("add-schedule-btn"))
    const select = screen.getByTestId("frequency-select")
    fireEvent.change(select, { target: { value: "monthly" } })

    expect(screen.getByTestId("day-of-month-input")).toBeInTheDocument()
  })

  it("shows Run Now and History buttons for existing schedule", async () => {
    ;(api.deploy.getSchedules as jest.Mock).mockResolvedValue([mockSchedule])
    render(<ScheduleCard deploymentId="dep-1" />)

    await waitFor(() => {
      expect(screen.getByTestId("run-now-btn-sched-1")).toBeInTheDocument()
    })
    expect(screen.getByTestId("history-btn-sched-1")).toBeInTheDocument()
    expect(screen.getByTestId("delete-schedule-btn-sched-1")).toBeInTheDocument()
  })

  it("triggers a run when Run Now is clicked", async () => {
    ;(api.deploy.getSchedules as jest.Mock).mockResolvedValue([mockSchedule])
    render(<ScheduleCard deploymentId="dep-1" />)

    await waitFor(() => screen.getByTestId("run-now-btn-sched-1"))
    fireEvent.click(screen.getByTestId("run-now-btn-sched-1"))

    await waitFor(() => {
      expect(api.deploy.triggerSchedule).toHaveBeenCalledWith("dep-1", "sched-1")
    })
  })

  it("shows run history when History is clicked", async () => {
    ;(api.deploy.getSchedules as jest.Mock).mockResolvedValue([mockSchedule])
    render(<ScheduleCard deploymentId="dep-1" />)

    await waitFor(() => screen.getByTestId("history-btn-sched-1"))
    fireEvent.click(screen.getByTestId("history-btn-sched-1"))

    await waitFor(() => {
      expect(api.deploy.getScheduleRuns).toHaveBeenCalledWith("dep-1", "sched-1")
    })
    await waitFor(() => {
      expect(screen.getByText("success")).toBeInTheDocument()
    })
    expect(screen.getByText("42 rows")).toBeInTheDocument()
  })

  it("removes schedule when Remove is clicked", async () => {
    ;(api.deploy.getSchedules as jest.Mock).mockResolvedValue([mockSchedule])
    render(<ScheduleCard deploymentId="dep-1" />)

    await waitFor(() => screen.getByTestId("delete-schedule-btn-sched-1"))
    fireEvent.click(screen.getByTestId("delete-schedule-btn-sched-1"))

    await waitFor(() => {
      expect(api.deploy.deleteSchedule).toHaveBeenCalledWith("dep-1", "sched-1")
    })
    // Schedule should be removed from UI
    await waitFor(() => {
      expect(screen.queryByTestId("run-now-btn-sched-1")).not.toBeInTheDocument()
    })
  })

  it("shows download link for successful runs", async () => {
    ;(api.deploy.getSchedules as jest.Mock).mockResolvedValue([mockSchedule])
    render(<ScheduleCard deploymentId="dep-1" />)

    await waitFor(() => screen.getByTestId("history-btn-sched-1"))
    fireEvent.click(screen.getByTestId("history-btn-sched-1"))

    await waitFor(() => {
      expect(screen.getByTestId("download-run-run-1")).toBeInTheDocument()
    })
  })

  it("shows last_error when schedule has an error", async () => {
    const schedWithError = { ...mockSchedule, last_error: "Dataset file not found" }
    ;(api.deploy.getSchedules as jest.Mock).mockResolvedValue([schedWithError])
    render(<ScheduleCard deploymentId="dep-1" />)

    await waitFor(() => {
      expect(screen.getByText(/Dataset file not found/)).toBeInTheDocument()
    })
  })
})
