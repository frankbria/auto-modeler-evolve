import React from "react"
import { render, screen } from "@testing-library/react"
import { BatchJobHistoryCard } from "@/components/deploy/batch-job-history-card"
import { BatchJobHistoryResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const healthyResult: BatchJobHistoryResult = {
  deployment_id: "dep-1",
  runs: [
    {
      id: "run-1",
      status: "success",
      started_at: "2024-01-15T09:00:00",
      completed_at: "2024-01-15T09:01:30",
      row_count: 500,
      duration_sec: 90,
      error: null,
    },
    {
      id: "run-2",
      status: "success",
      started_at: "2024-01-14T09:00:00",
      completed_at: "2024-01-14T09:01:00",
      row_count: 480,
      duration_sec: 60,
      error: null,
    },
  ],
  total_runs: 10,
  success_count: 10,
  failed_count: 0,
  running_count: 0,
  success_rate: 1.0,
  avg_row_count: 490,
  avg_duration_sec: 75,
  last_run_at: "2024-01-15T09:01:30",
  verdict: "healthy",
  summary: "Your batch schedule is running reliably: 10 of 10 jobs succeeded (100% success rate).",
}

const someFailuresResult: BatchJobHistoryResult = {
  deployment_id: "dep-2",
  runs: [
    {
      id: "run-a",
      status: "success",
      started_at: "2024-01-15T09:00:00",
      completed_at: "2024-01-15T09:01:30",
      row_count: 300,
      duration_sec: 90,
      error: null,
    },
    {
      id: "run-b",
      status: "failed",
      started_at: "2024-01-14T09:00:00",
      completed_at: "2024-01-14T09:00:05",
      row_count: null,
      duration_sec: 5,
      error: "connection refused",
    },
  ],
  total_runs: 10,
  success_count: 6,
  failed_count: 4,
  running_count: 0,
  success_rate: 0.6,
  avg_row_count: 300,
  avg_duration_sec: 47.5,
  last_run_at: "2024-01-15T09:01:30",
  verdict: "some_failures",
  summary: "Your batch schedule has had some issues: 6 succeeded and 4 failed out of 10 total runs.",
}

const noDataResult: BatchJobHistoryResult = {
  deployment_id: "dep-3",
  runs: [],
  total_runs: 0,
  success_count: 0,
  failed_count: 0,
  running_count: 0,
  success_rate: 0,
  avg_row_count: null,
  avg_duration_sec: null,
  last_run_at: null,
  verdict: "no_data",
  summary: "No batch jobs have run yet for this deployment.",
}

const allFailedResult: BatchJobHistoryResult = {
  deployment_id: "dep-4",
  runs: [
    {
      id: "run-x",
      status: "failed",
      started_at: "2024-01-15T09:00:00",
      completed_at: "2024-01-15T09:00:02",
      row_count: null,
      duration_sec: 2,
      error: "deployment offline",
    },
  ],
  total_runs: 5,
  success_count: 0,
  failed_count: 5,
  running_count: 0,
  success_rate: 0.0,
  avg_row_count: null,
  avg_duration_sec: 2,
  last_run_at: "2024-01-15T09:00:02",
  verdict: "all_failed",
  summary: "Most batch runs have failed: only 0 of 5 succeeded.",
}

describe("BatchJobHistoryCard", () => {
  it("renders card for healthy result", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    expect(screen.getByText("Batch Job History")).toBeInTheDocument()
  })

  it("shows Healthy verdict badge", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    expect(screen.getByText("Healthy")).toBeInTheDocument()
  })

  it("shows total runs badge", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    expect(screen.getByText("10 runs")).toBeInTheDocument()
  })

  it("shows 100% success badge", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    expect(screen.getByText("100% success")).toBeInTheDocument()
  })

  it("renders summary text", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    expect(screen.getByText(healthyResult.summary)).toBeInTheDocument()
  })

  it("shows succeeded count in stats grid", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    expect(screen.getByText("Succeeded")).toBeInTheDocument()
  })

  it("shows failed count in stats grid", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    expect(screen.getByText("Failed")).toBeInTheDocument()
  })

  it("shows avg rows per run stat", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    expect(screen.getByText("Avg rows/run")).toBeInTheDocument()
  })

  it("shows avg duration stat", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    expect(screen.getByText("Avg duration")).toBeInTheDocument()
  })

  it("renders run timeline table", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    expect(screen.getByText("Recent Runs")).toBeInTheDocument()
  })

  it("renders success status badges in table", () => {
    render(<BatchJobHistoryCard result={healthyResult} />)
    const badges = screen.getAllByText("success")
    expect(badges.length).toBeGreaterThan(0)
  })

  it("shows Some failures verdict for mixed result", () => {
    render(<BatchJobHistoryCard result={someFailuresResult} />)
    expect(screen.getByText("Some failures")).toBeInTheDocument()
  })

  it("shows failed status in table for failed run", () => {
    render(<BatchJobHistoryCard result={someFailuresResult} />)
    expect(screen.getByText("failed")).toBeInTheDocument()
  })

  it("shows No data verdict for empty result", () => {
    render(<BatchJobHistoryCard result={noDataResult} />)
    expect(screen.getByText("No data")).toBeInTheDocument()
  })

  it("does not render stats grid when no runs", () => {
    render(<BatchJobHistoryCard result={noDataResult} />)
    expect(screen.queryByText("Succeeded")).not.toBeInTheDocument()
  })

  it("does not render timeline table when no runs", () => {
    render(<BatchJobHistoryCard result={noDataResult} />)
    expect(screen.queryByText("Recent Runs")).not.toBeInTheDocument()
  })

  it("shows Failing verdict for all_failed", () => {
    render(<BatchJobHistoryCard result={allFailedResult} />)
    expect(screen.getByText("Failing")).toBeInTheDocument()
  })
})

describe("BatchJobHistoryCard store action", () => {
  it("attachBatchJobHistoryToLastMessage exists in store", () => {
    const state = useAppStore.getState()
    expect(typeof state.attachBatchJobHistoryToLastMessage).toBe("function")
  })

  it("attaches batch_job_history to last assistant message", () => {
    useAppStore.setState({
      messages: [
        { role: "user", content: "show batch history" },
        { role: "assistant", content: "Here is your history" },
      ],
    })
    useAppStore.getState().attachBatchJobHistoryToLastMessage(healthyResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].batch_job_history).toEqual(healthyResult)
  })

  it("does not overwrite if last message is user", () => {
    useAppStore.setState({
      messages: [{ role: "user", content: "show batch history" }],
    })
    useAppStore.getState().attachBatchJobHistoryToLastMessage(healthyResult)
    const msgs = useAppStore.getState().messages
    expect(msgs[msgs.length - 1].batch_job_history).toBeUndefined()
  })
})
