"use client"

import type { ActiveFilter } from "@/lib/types"

interface FilterBadgeProps {
  filter: ActiveFilter
  onClear: () => void
}

export function FilterBadge({ filter, onClear }: FilterBadgeProps) {
  if (!filter.active || !filter.filter_summary) return null

  return (
    <div
      data-testid="filter-badge"
      className="flex items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-sm text-blue-800"
    >
      <span className="text-xs font-semibold uppercase tracking-wide text-blue-500">
        Filter
      </span>
      <span className="font-medium">{filter.filter_summary}</span>
      {filter.filtered_rows !== undefined && filter.original_rows !== undefined && (
        <span className="text-xs text-blue-500">
          ({filter.filtered_rows.toLocaleString()} of {filter.original_rows.toLocaleString()} rows)
        </span>
      )}
      <button
        onClick={onClear}
        data-testid="filter-clear-btn"
        className="ml-1 rounded-full p-0.5 text-blue-400 transition-colors hover:bg-blue-200 hover:text-blue-700"
        aria-label="Clear filter"
        title="Clear filter — return to full dataset"
      >
        ✕
      </button>
    </div>
  )
}
