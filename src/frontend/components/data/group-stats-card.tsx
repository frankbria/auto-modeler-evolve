"use client"

import type { GroupStatsResult, GroupStatsRow } from "@/lib/types"

interface GroupStatsCardProps {
  result: GroupStatsResult
}

function formatValue(val: number | string | null, agg: string): string {
  if (val === null || val === undefined) return "—"
  if (typeof val === "string") return val
  if (agg === "count") return val.toLocaleString()
  if (Number.isInteger(val)) return val.toLocaleString()
  return val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function BarRow({
  row,
  valueCol,
  maxVal,
  agg,
  rank,
}: {
  row: GroupStatsRow
  valueCol: string
  maxVal: number
  agg: string
  rank: number
}) {
  const rawVal = row[valueCol] ?? row["value"] ?? null
  const numVal = typeof rawVal === "number" ? rawVal : null
  const widthPct = maxVal > 0 && numVal !== null ? Math.round((numVal / maxVal) * 100) : 0

  const barColor =
    rank === 0
      ? "bg-blue-500"
      : rank === 1
        ? "bg-blue-400"
        : rank <= 4
          ? "bg-blue-300"
          : "bg-blue-200"

  return (
    <div className="flex items-center gap-2 py-0.5">
      {/* Group label */}
      <span
        className="w-28 shrink-0 truncate text-right text-xs text-gray-700"
        title={row.group}
      >
        {row.group}
      </span>

      {/* Bar + value */}
      <div className="flex flex-1 items-center gap-1">
        <div className="h-5 flex-1 overflow-hidden rounded bg-gray-100">
          <div
            className={`h-full rounded ${barColor} transition-all`}
            style={{ width: `${widthPct}%` }}
          />
        </div>
        <span className="w-20 text-right text-xs font-medium tabular-nums text-gray-700">
          {formatValue(rawVal, agg)}
        </span>
      </div>
    </div>
  )
}

export function GroupStatsCard({ result }: GroupStatsCardProps) {
  const { group_col, value_col, agg, rows, total, summary } = result

  if (!rows || rows.length === 0) {
    return (
      <div
        data-testid="group-stats-card"
        className="mt-2 rounded-lg border border-gray-200 bg-gray-50 p-3"
      >
        <p className="text-sm text-gray-500">{summary}</p>
      </div>
    )
  }

  const aggLabel =
    agg === "sum"
      ? "Total"
      : agg === "mean"
        ? "Average"
        : agg === "count"
          ? "Count"
          : agg === "min"
            ? "Minimum"
            : agg === "max"
              ? "Maximum"
              : agg === "median"
                ? "Median"
                : agg

  // Compute max for bar scaling from the displayed value column
  const maxVal = Math.max(
    ...rows.map((r) => {
      const v = r[value_col] ?? r["value"]
      return typeof v === "number" ? v : 0
    })
  )

  const groupLabel = group_col.replace(/_/g, " ")
  const valueLabel = value_col.replace(/_/g, " ")

  // Show at most 15 rows; collapse the rest
  const visibleRows = rows.slice(0, 15)

  return (
    <div
      data-testid="group-stats-card"
      className="mt-2 rounded-lg border border-blue-100 bg-white shadow-sm"
    >
      {/* Header */}
      <div className="border-b border-gray-100 px-3 py-2">
        <h3 className="text-sm font-semibold text-gray-800">
          {aggLabel}{" "}
          <span className="text-blue-600">{valueLabel}</span>{" "}
          by{" "}
          <span className="text-purple-600">{groupLabel}</span>
        </h3>
        <div className="mt-0.5 flex flex-wrap gap-2 text-xs text-gray-500">
          <span>{visibleRows.length} groups</span>
          {total !== null && agg === "sum" && (
            <span>· Total: {formatValue(total, agg)}</span>
          )}
        </div>
      </div>

      {/* Bar chart rows */}
      <div className="px-3 py-2">
        {visibleRows.map((row, i) => (
          <BarRow
            key={row.group}
            row={row}
            valueCol={value_col}
            maxVal={maxVal}
            agg={agg}
            rank={i}
          />
        ))}
      </div>

      {/* Summary */}
      <div className="border-t border-gray-100 px-3 py-2">
        <p className="text-xs text-gray-600">{summary}</p>
      </div>
    </div>
  )
}
