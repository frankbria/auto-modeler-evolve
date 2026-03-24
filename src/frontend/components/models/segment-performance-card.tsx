"use client"

import type {
  SegmentPerformanceResult,
  SegmentPerformanceSegment,
} from "@/lib/types"

interface SegmentPerformanceCardProps {
  result: SegmentPerformanceResult
}

function statusColor(status: SegmentPerformanceSegment["status"]): string {
  switch (status) {
    case "strong":
      return "bg-green-100 text-green-800"
    case "moderate":
      return "bg-blue-50 text-blue-700"
    case "weak":
      return "bg-yellow-100 text-yellow-800"
    case "poor":
      return "bg-red-100 text-red-800"
    default:
      return "bg-muted text-muted-foreground"
  }
}

function formatMetric(
  value: number | null,
  problemType: string,
): string {
  if (value === null || value === undefined) return "—"
  if (problemType === "classification") return `${(value * 100).toFixed(1)}%`
  return value.toFixed(3)
}

function MetricBar({ value, max }: { value: number | null; max: number }) {
  if (value === null || max === 0) return null
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  const barColor =
    value >= 0.85
      ? "bg-green-500"
      : value >= 0.65
        ? "bg-blue-400"
        : value >= 0.4
          ? "bg-yellow-400"
          : "bg-red-400"
  return (
    <div className="ml-2 h-2 w-20 overflow-hidden rounded-full bg-muted">
      <div
        className={`h-full rounded-full ${barColor}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

export function SegmentPerformanceCard({
  result,
}: SegmentPerformanceCardProps) {
  const {
    group_col,
    metric_name,
    problem_type,
    segments,
    best_segment,
    worst_segment,
    summary,
  } = result

  const maxMetric = Math.max(
    ...segments.map((s) => s.metric ?? 0).filter(Boolean),
    0.01,
  )

  return (
    <div
      className="mt-2 rounded-lg border bg-card p-3 text-sm"
      data-testid="segment-performance-card"
    >
      <p className="mb-1 text-xs font-semibold text-muted-foreground">
        Model Performance by{" "}
        <span className="text-foreground">{group_col}</span>
      </p>
      <p className="mb-2 text-xs text-muted-foreground">{summary}</p>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              <th className="border border-border bg-muted px-2 py-1 text-left font-semibold text-muted-foreground">
                Segment
              </th>
              <th className="border border-border bg-muted px-2 py-1 text-center font-semibold text-muted-foreground">
                Rows
              </th>
              <th className="border border-border bg-muted px-2 py-1 text-right font-semibold text-muted-foreground">
                {metric_name}
              </th>
              <th className="border border-border bg-muted px-2 py-1 text-center font-semibold text-muted-foreground">
                Performance
              </th>
            </tr>
          </thead>
          <tbody>
            {segments.map((seg, idx) => {
              const isWorst = seg.name === worst_segment
              const isBest = seg.name === best_segment && seg.name !== worst_segment
              const rowClass = isWorst
                ? "bg-amber-50"
                : isBest
                  ? "bg-green-50"
                  : idx % 2 === 0
                    ? "bg-background"
                    : "bg-muted/30"

              return (
                <tr key={seg.name} className={rowClass}>
                  <td className="border border-border px-2 py-1 font-medium">
                    {seg.name.replace(/_/g, " ")}
                    {isWorst && (
                      <span className="ml-1 text-[10px] text-amber-600">
                        ▼ lowest
                      </span>
                    )}
                    {isBest && (
                      <span className="ml-1 text-[10px] text-green-600">
                        ▲ best
                      </span>
                    )}
                  </td>
                  <td className="border border-border px-2 py-1 text-center tabular-nums text-muted-foreground">
                    {seg.n}
                    {seg.low_sample && (
                      <span className="ml-1 text-[9px] text-yellow-600">
                        !
                      </span>
                    )}
                  </td>
                  <td className="border border-border px-2 py-1 text-right tabular-nums">
                    {formatMetric(seg.metric, problem_type)}
                  </td>
                  <td className="border border-border px-2 py-1">
                    <div className="flex items-center justify-between">
                      <span
                        className={`rounded px-1 py-0.5 text-[10px] font-medium ${statusColor(seg.status)}`}
                      >
                        {seg.status === "insufficient_data"
                          ? "n/a"
                          : seg.status}
                      </span>
                      <MetricBar value={seg.metric} max={maxMetric} />
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {segments.some((s) => s.low_sample) && (
        <p className="mt-1 text-[10px] text-muted-foreground">
          ! = fewer than 10 rows — metric may not be reliable.
        </p>
      )}
    </div>
  )
}
