"use client"

import type { SegmentComparisonResult, SegmentColumnStats } from "@/lib/types"

interface SegmentComparisonCardProps {
  result: SegmentComparisonResult
}

function formatStat(value: number | null): string {
  if (value === null || value === undefined) return "—"
  if (Math.abs(value) >= 1_000_000)
    return value.toLocaleString(undefined, { maximumFractionDigits: 1 })
  if (Number.isInteger(value)) return value.toLocaleString()
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

function effectLabel(effect: number | null): string {
  if (effect === null) return ""
  const abs = Math.abs(effect)
  if (abs < 0.2) return "similar"
  if (abs < 0.5) return "moderate"
  if (abs < 0.8) return "large"
  return "very large"
}

function EffectBadge({ col }: { col: SegmentColumnStats }) {
  if (col.effect_size === null || Math.abs(col.effect_size) < 0.2) return null
  const abs = Math.abs(col.effect_size)
  const colorClass =
    abs >= 0.8
      ? "bg-orange-100 text-orange-800"
      : abs >= 0.5
        ? "bg-yellow-100 text-yellow-800"
        : "bg-blue-50 text-blue-700"
  return (
    <span
      className={`ml-1 rounded px-1 py-0.5 text-[10px] font-medium ${colorClass}`}
      title={`Effect size: ${col.effect_size?.toFixed(2)}`}
    >
      {effectLabel(col.effect_size)}
    </span>
  )
}

export function SegmentComparisonCard({ result }: SegmentComparisonCardProps) {
  const { group_col, val1, val2, count1, count2, columns, summary } = result

  // Show at most 8 numeric columns; sort notable diffs first
  const notableNames = new Set(result.notable_diffs.map((n) => n.name))
  const sorted = [
    ...columns.filter((c) => notableNames.has(c.name)),
    ...columns.filter((c) => !notableNames.has(c.name)),
  ].slice(0, 8)

  return (
    <div
      className="mt-2 rounded-lg border bg-card p-3 text-sm"
      data-testid="segment-comparison-card"
    >
      <p className="mb-1 text-xs font-semibold text-muted-foreground">
        Segment Comparison: <span className="text-foreground">{group_col}</span>
      </p>
      <p className="mb-2 text-xs text-muted-foreground">{summary}</p>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              <th className="border border-border bg-muted px-2 py-1 text-left font-semibold text-muted-foreground">
                Metric
              </th>
              <th className="border border-border bg-blue-50 px-2 py-1 text-right font-semibold text-blue-800">
                {val1}
                <span className="ml-1 font-normal text-blue-600">({count1})</span>
              </th>
              <th className="border border-border bg-purple-50 px-2 py-1 text-right font-semibold text-purple-800">
                {val2}
                <span className="ml-1 font-normal text-purple-600">({count2})</span>
              </th>
              <th className="border border-border bg-muted px-2 py-1 text-center font-semibold text-muted-foreground">
                Difference
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((col, idx) => {
              const isNotable = notableNames.has(col.name)
              const rowClass = isNotable
                ? idx % 2 === 0
                  ? "bg-amber-50"
                  : "bg-amber-50/60"
                : idx % 2 === 0
                  ? "bg-background"
                  : "bg-muted/30"

              return (
                <tr key={col.name} className={rowClass}>
                  <td className="border border-border px-2 py-1 font-medium">
                    {col.name.replace(/_/g, " ")}
                    <EffectBadge col={col} />
                  </td>
                  <td className="border border-border px-2 py-1 text-right tabular-nums text-blue-800">
                    {formatStat(col.mean1)}
                  </td>
                  <td className="border border-border px-2 py-1 text-right tabular-nums text-purple-800">
                    {formatStat(col.mean2)}
                  </td>
                  <td className="border border-border px-2 py-1 text-center text-xs text-muted-foreground">
                    {col.direction === "higher_in_val1" ? (
                      <span className="text-blue-700">↑ {val1}</span>
                    ) : col.direction === "higher_in_val2" ? (
                      <span className="text-purple-700">↑ {val2}</span>
                    ) : (
                      <span>≈ similar</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {columns.length > 8 && (
        <p className="mt-1 text-xs text-muted-foreground">
          Showing top 8 of {columns.length} numeric columns.
        </p>
      )}
    </div>
  )
}
