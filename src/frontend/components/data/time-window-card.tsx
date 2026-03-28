"use client"

import type { TimeWindowComparison, TimeWindowColumn } from "@/lib/types"

interface TimeWindowCardProps {
  result: TimeWindowComparison
}

function formatValue(val: number): string {
  if (Math.abs(val) >= 1_000_000) return `${(val / 1_000_000).toFixed(2)}M`
  if (Math.abs(val) >= 1_000) return `${(val / 1_000).toFixed(1)}k`
  if (Number.isInteger(val)) return val.toLocaleString()
  return val.toFixed(2)
}

function ChangeArrow({ direction, pctChange }: { direction: string; pctChange: number }) {
  if (direction === "flat") {
    return <span className="text-muted-foreground text-xs">→ 0%</span>
  }
  const isUp = direction === "up"
  const color = isUp ? "text-emerald-600" : "text-rose-600"
  const arrow = isUp ? "↑" : "↓"
  return (
    <span className={`font-semibold text-xs ${color}`}>
      {arrow} {Math.abs(pctChange).toFixed(1)}%
    </span>
  )
}

function ColumnRow({ col }: { col: TimeWindowColumn; p1Name: string; p2Name: string }) {
  const rowClass = col.notable ? "bg-amber-50 border-l-2 border-amber-400" : ""
  return (
    <tr className={`border-b border-border/40 last:border-0 ${rowClass}`}>
      <td className="py-2 px-3 text-sm font-medium text-foreground">
        {col.column.replace(/_/g, " ")}
        {col.notable && (
          <span className="ml-1 text-xs font-normal text-amber-600">notable</span>
        )}
      </td>
      <td className="py-2 px-3 text-sm text-muted-foreground text-right">
        {formatValue(col.p1_mean)}
      </td>
      <td className="py-2 px-3 text-sm text-foreground text-right font-medium">
        {formatValue(col.p2_mean)}
      </td>
      <td className="py-2 px-3 text-right">
        <ChangeArrow direction={col.direction} pctChange={col.pct_change} />
      </td>
    </tr>
  )
}

export function TimeWindowCard({ result }: TimeWindowCardProps) {
  const { period1, period2, columns, notable_changes, summary } = result
  const up = columns.filter((c) => c.direction === "up").length
  const down = columns.filter((c) => c.direction === "down").length

  return (
    <div className="rounded-lg border-2 border-orange-300 bg-orange-50/30 p-4 mt-3 space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-orange-700">
            Period Comparison
          </p>
          <p className="text-xs text-muted-foreground mt-0.5">
            {period1.name} vs {period2.name}
          </p>
        </div>
        <div className="flex gap-2 text-xs">
          {up > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 font-medium">
              ↑ {up} up
            </span>
          )}
          {down > 0 && (
            <span className="px-2 py-0.5 rounded-full bg-rose-100 text-rose-700 font-medium">
              ↓ {down} down
            </span>
          )}
        </div>
      </div>

      {/* Period chips */}
      <div className="flex gap-3 text-xs">
        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-muted/60 border border-border/60">
          <span className="font-medium text-foreground">{period1.name}</span>
          <span className="text-muted-foreground">{period1.row_count} rows</span>
        </div>
        <span className="text-muted-foreground self-center">vs</span>
        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-orange-100 border border-orange-200">
          <span className="font-medium text-orange-800">{period2.name}</span>
          <span className="text-orange-600">{period2.row_count} rows</span>
        </div>
      </div>

      {/* Comparison table */}
      <div className="overflow-x-auto rounded-md border border-border/60 bg-card">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-muted/40 border-b border-border/60">
              <th className="py-2 px-3 text-left text-xs font-semibold text-muted-foreground">
                Metric
              </th>
              <th className="py-2 px-3 text-right text-xs font-semibold text-muted-foreground">
                {period1.name}
              </th>
              <th className="py-2 px-3 text-right text-xs font-semibold text-foreground">
                {period2.name}
              </th>
              <th className="py-2 px-3 text-right text-xs font-semibold text-muted-foreground">
                Change
              </th>
            </tr>
          </thead>
          <tbody>
            {columns.map((col) => (
              <ColumnRow
                key={col.column}
                col={col}
                p1Name={period1.name}
                p2Name={period2.name}
              />
            ))}
          </tbody>
        </table>
      </div>

      {/* Notable changes callout */}
      {notable_changes.length > 0 && (
        <div className="text-xs bg-amber-50 border border-amber-200 rounded px-3 py-1.5 text-amber-700">
          <span className="font-semibold">Notable changes (&gt;20%):</span>{" "}
          {notable_changes.map((c) => c.replace(/_/g, " ")).join(", ")}
        </div>
      )}

      {/* Summary footer */}
      <p className="text-xs text-muted-foreground border-t border-border/40 pt-2">{summary}</p>
    </div>
  )
}
