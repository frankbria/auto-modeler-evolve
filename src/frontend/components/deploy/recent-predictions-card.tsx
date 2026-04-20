"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { RecentPredictionsResult } from "@/lib/types"

// ---------------------------------------------------------------------------
// RecentPredictionsCard — inline chat card showing the last N prediction log rows
// ---------------------------------------------------------------------------

interface RecentPredictionsCardProps {
  result: RecentPredictionsResult
}

function formatRelativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60_000)
  if (mins < 1) return "just now"
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return `${days}d ago`
}

function formatPrediction(value: string): string {
  const num = parseFloat(value)
  if (!isNaN(num) && isFinite(num)) {
    return num >= 1_000_000
      ? `${(num / 1_000_000).toFixed(1)}M`
      : num >= 1_000
      ? `${(num / 1_000).toFixed(1)}k`
      : num % 1 === 0
      ? String(num)
      : num.toFixed(2)
  }
  return value.length > 20 ? `${value.slice(0, 20)}…` : value
}

function LatencyBadge({ ms }: { ms: number | null }) {
  if (ms === null) return <span className="text-slate-400">—</span>
  const color =
    ms < 100
      ? "text-emerald-700 bg-emerald-50 border-emerald-200"
      : ms < 500
      ? "text-amber-700 bg-amber-50 border-amber-200"
      : "text-rose-700 bg-rose-50 border-rose-200"
  return (
    <span
      className={`inline-flex items-center rounded border px-1 py-0.5 text-xs font-mono ${color}`}
      aria-label={`${ms}ms response time`}
    >
      {ms}ms
    </span>
  )
}

export function RecentPredictionsCard({ result }: RecentPredictionsCardProps) {
  const { n_shown, total_all_time, predictions, export_url, summary } = result
  const isEmpty = total_all_time === 0

  return (
    <Card
      className="border-slate-300 bg-slate-50 w-full max-w-2xl"
      aria-label="Recent predictions table"
    >
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold text-slate-800">
          <span aria-hidden="true">📋</span>
          Recent Predictions
        </CardTitle>
        <div className="flex flex-wrap gap-1 mt-1">
          <Badge className="bg-slate-200 text-slate-700 border border-slate-300 text-xs">
            {n_shown} shown
          </Badge>
          {total_all_time > n_shown && (
            <Badge className="bg-blue-100 text-blue-700 border border-blue-200 text-xs">
              {total_all_time.toLocaleString()} total
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-3 text-sm">
        {isEmpty ? (
          <p className="text-slate-500 italic">
            No predictions recorded yet. Predictions will appear here once the deployed model
            receives API requests.
          </p>
        ) : (
          <>
            <div className="overflow-x-auto rounded border border-slate-200">
              <table className="w-full text-xs" role="table" aria-label="Recent prediction log">
                <thead>
                  <tr className="bg-slate-100 text-slate-600 border-b border-slate-200">
                    <th className="text-left px-2 py-1.5 font-medium">Time</th>
                    <th className="text-left px-2 py-1.5 font-medium">Prediction</th>
                    <th className="text-left px-2 py-1.5 font-medium">Confidence</th>
                    <th className="text-left px-2 py-1.5 font-medium">Latency</th>
                    <th className="text-left px-2 py-1.5 font-medium">Key Inputs</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((row, i) => (
                    <tr
                      key={row.id}
                      className={`border-b border-slate-100 ${i % 2 === 0 ? "bg-white" : "bg-slate-50"} hover:bg-slate-100 transition-colors`}
                    >
                      <td className="px-2 py-1.5 text-slate-500 whitespace-nowrap">
                        {formatRelativeTime(row.created_at)}
                      </td>
                      <td className="px-2 py-1.5 font-medium text-slate-800 whitespace-nowrap">
                        {formatPrediction(row.prediction)}
                        {row.ab_variant && (
                          <span
                            className="ml-1 text-[10px] text-purple-600 bg-purple-50 border border-purple-200 rounded px-0.5"
                            aria-label={`A/B variant: ${row.ab_variant}`}
                          >
                            {row.ab_variant === "challenger" ? "B" : "A"}
                          </span>
                        )}
                      </td>
                      <td className="px-2 py-1.5">
                        {row.confidence !== null ? (
                          <span
                            className={`font-medium ${row.confidence >= 80 ? "text-emerald-700" : row.confidence >= 60 ? "text-amber-700" : "text-rose-700"}`}
                            aria-label={`${row.confidence}% confidence`}
                          >
                            {row.confidence}%
                          </span>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                      </td>
                      <td className="px-2 py-1.5">
                        <LatencyBadge ms={row.response_ms} />
                      </td>
                      <td className="px-2 py-1.5">
                        <div className="flex flex-wrap gap-0.5">
                          {row.input_summary.map((kv) => (
                            <span
                              key={kv.key}
                              className="text-[10px] bg-blue-50 text-blue-700 border border-blue-200 rounded px-1 py-0.5 font-mono"
                              title={`${kv.key}=${kv.value}`}
                            >
                              {kv.key}={kv.value.length > 8 ? `${kv.value.slice(0, 8)}…` : kv.value}
                            </span>
                          ))}
                          {row.input_summary.length === 0 && (
                            <span className="text-slate-400">—</span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-between text-xs text-slate-500">
              <span>{summary}</span>
              <a
                href={export_url}
                download
                className="inline-flex items-center gap-1 rounded border border-slate-300 bg-white px-2 py-1 hover:bg-slate-50 transition-colors text-slate-600"
                aria-label="Download all prediction logs as CSV"
              >
                <span aria-hidden="true">⬇</span>
                Download all as CSV
              </a>
            </div>
          </>
        )}

        <figcaption className="sr-only">
          {isEmpty
            ? "Recent predictions: no records available yet."
            : `Recent predictions table: showing ${n_shown} of ${total_all_time.toLocaleString()} total predictions.`}
        </figcaption>
      </CardContent>
    </Card>
  )
}
