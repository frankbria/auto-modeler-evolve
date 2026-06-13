"use client"

import { BatchJobHistoryResult, BatchJobHistoryRun, BatchJobHistoryVerdict } from "@/lib/types"

interface BatchJobHistoryCardProps {
  result: BatchJobHistoryResult
}

function verdictColors(v: BatchJobHistoryVerdict): { border: string; badge: string } {
  switch (v) {
    case "healthy":
      return { border: "border-emerald-500", badge: "bg-emerald-100 text-emerald-800" }
    case "some_failures":
      return { border: "border-amber-500", badge: "bg-amber-100 text-amber-800" }
    case "all_failed":
      return { border: "border-rose-500", badge: "bg-rose-100 text-rose-800" }
    default:
      return { border: "border-slate-300", badge: "bg-slate-100 text-slate-600" }
  }
}

function verdictLabel(v: BatchJobHistoryVerdict): string {
  switch (v) {
    case "healthy":
      return "Healthy"
    case "some_failures":
      return "Some failures"
    case "all_failed":
      return "Failing"
    default:
      return "No data"
  }
}

function statusColors(status: BatchJobHistoryRun["status"]): string {
  switch (status) {
    case "success":
      return "bg-emerald-100 text-emerald-800"
    case "failed":
      return "bg-rose-100 text-rose-800"
    case "running":
      return "bg-blue-100 text-blue-800"
    default:
      return "bg-slate-100 text-slate-600"
  }
}

function formatDateTime(iso: string | null): string {
  if (!iso) return "—"
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
  } catch {
    return iso
  }
}

function formatDuration(sec: number | null): string {
  if (sec === null || sec === undefined) return "—"
  if (sec < 60) return `${sec.toFixed(0)}s`
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  return `${m}m ${s}s`
}

export function BatchJobHistoryCard({ result }: BatchJobHistoryCardProps) {
  const { border, badge } = verdictColors(result.verdict)

  return (
    <div className={`rounded-lg border-2 ${border} bg-white p-4 space-y-4`} data-slot="batch-job-history-card">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xl" aria-hidden="true">📋</span>
          <h3 className="font-semibold text-slate-800 text-sm">Batch Job History</h3>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${badge}`}>
            {verdictLabel(result.verdict)}
          </span>
          <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-700">
            {result.total_runs} {result.total_runs === 1 ? "run" : "runs"}
          </span>
          {result.total_runs > 0 && (
            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-700">
              {(result.success_rate * 100).toFixed(0)}% success
            </span>
          )}
        </div>
      </div>

      {/* Summary */}
      <p className="text-sm text-slate-600">{result.summary}</p>

      {/* Aggregate stats grid */}
      {result.total_runs > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="rounded-md bg-slate-50 p-2.5 text-center">
            <div className="text-lg font-semibold text-slate-800">{result.success_count}</div>
            <div className="text-xs text-slate-500">Succeeded</div>
          </div>
          <div className="rounded-md bg-slate-50 p-2.5 text-center">
            <div className="text-lg font-semibold text-slate-800">{result.failed_count}</div>
            <div className="text-xs text-slate-500">Failed</div>
          </div>
          <div className="rounded-md bg-slate-50 p-2.5 text-center">
            <div className="text-lg font-semibold text-slate-800">
              {result.avg_row_count !== null ? result.avg_row_count.toLocaleString() : "—"}
            </div>
            <div className="text-xs text-slate-500">Avg rows/run</div>
          </div>
          <div className="rounded-md bg-slate-50 p-2.5 text-center">
            <div className="text-lg font-semibold text-slate-800">
              {formatDuration(result.avg_duration_sec)}
            </div>
            <div className="text-xs text-slate-500">Avg duration</div>
          </div>
        </div>
      )}

      {/* Run timeline table */}
      {result.runs.length > 0 && (
        <div className="space-y-1.5">
          <h4 className="text-xs font-medium text-slate-500 uppercase tracking-wide">Recent Runs</h4>
          <div className="overflow-x-auto rounded-md border border-slate-200">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-50 text-slate-500">
                <tr>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Started</th>
                  <th className="px-3 py-2 font-medium">Completed</th>
                  <th className="px-3 py-2 font-medium">Rows</th>
                  <th className="px-3 py-2 font-medium">Duration</th>
                  <th className="px-3 py-2 font-medium">Error</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {result.runs.map((run) => (
                  <tr key={run.id} className="hover:bg-slate-50 transition-colors">
                    <td className="px-3 py-2">
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${statusColors(run.status)}`}>
                        {run.status}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-slate-600">{formatDateTime(run.started_at)}</td>
                    <td className="px-3 py-2 text-slate-600">{formatDateTime(run.completed_at)}</td>
                    <td className="px-3 py-2 text-slate-600">
                      {run.row_count !== null ? run.row_count.toLocaleString() : "—"}
                    </td>
                    <td className="px-3 py-2 text-slate-600">{formatDuration(run.duration_sec)}</td>
                    <td className="px-3 py-2 text-slate-500 max-w-[200px] truncate" title={run.error ?? undefined}>
                      {run.error ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <figcaption className="sr-only">
            Table of batch job run history showing status, timing, row count, duration, and any errors.
          </figcaption>
        </div>
      )}

      {/* Last run footer */}
      {result.last_run_at && (
        <p className="text-xs text-slate-400">
          Last activity: {formatDateTime(result.last_run_at)}
        </p>
      )}
    </div>
  )
}
