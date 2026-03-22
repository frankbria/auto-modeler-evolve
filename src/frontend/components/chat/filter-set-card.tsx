"use client"

import type { FilterSetResult } from "@/lib/types"

interface FilterSetCardProps {
  result: FilterSetResult
}

const OP_LABELS: Record<string, string> = {
  eq: "=",
  ne: "≠",
  gt: ">",
  lt: "<",
  gte: "≥",
  lte: "≤",
  contains: "contains",
  not_contains: "doesn't contain",
}

export function FilterSetCard({ result }: FilterSetCardProps) {
  const { conditions, original_rows, filtered_rows, row_reduction_pct } = result

  return (
    <div
      data-testid="filter-set-card"
      className="mt-2 rounded-lg border border-blue-200 bg-blue-50 p-3"
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm font-semibold text-blue-800">🔍 Filter Active</span>
        <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
          {filtered_rows.toLocaleString()} rows
        </span>
      </div>

      <div className="mb-2 space-y-1">
        {conditions.map((cond, i) => (
          <div key={i} className="flex items-center gap-1.5 text-sm text-blue-900">
            <code className="rounded bg-blue-100 px-1.5 py-0.5 font-mono text-xs">
              {cond.column}
            </code>
            <span className="text-blue-600">{OP_LABELS[cond.operator] ?? cond.operator}</span>
            <code className="rounded bg-blue-100 px-1.5 py-0.5 font-mono text-xs">
              {String(cond.value)}
            </code>
          </div>
        ))}
      </div>

      <div className="text-xs text-blue-600">
        Showing {filtered_rows.toLocaleString()} of {original_rows.toLocaleString()} rows
        {row_reduction_pct > 0 && (
          <span className="ml-1">({row_reduction_pct}% reduction)</span>
        )}
        . All analyses use this subset. Say &quot;clear filter&quot; to return to the full dataset.
      </div>
    </div>
  )
}
