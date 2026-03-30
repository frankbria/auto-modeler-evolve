"use client"

import { Badge } from "@/components/ui/badge"
import type { NullMapResult, NullMapColumn } from "@/lib/types"

interface NullMapCardProps {
  result: NullMapResult
}

function CompletionBar({ complete_pct }: { complete_pct: number }) {
  const color =
    complete_pct === 100
      ? "bg-emerald-500"
      : complete_pct >= 90
      ? "bg-amber-400"
      : "bg-rose-500"
  return (
    <div className="flex items-center gap-2 min-w-0">
      <div className="h-2 w-24 rounded-full bg-muted overflow-hidden flex-shrink-0">
        <div
          className={`h-full rounded-full ${color} transition-all`}
          style={{ width: `${complete_pct}%` }}
        />
      </div>
      <span className="text-xs text-muted-foreground whitespace-nowrap">
        {complete_pct}%
      </span>
    </div>
  )
}

function NullRow({ col }: { col: NullMapColumn }) {
  const hasMissing = col.null_count > 0
  return (
    <tr className="border-t border-border/50">
      <td className="py-1.5 pr-3 text-xs font-mono truncate max-w-[140px]" title={col.column}>
        {col.column.replace(/_/g, " ")}
      </td>
      <td className="py-1.5 pr-3">
        <CompletionBar complete_pct={col.complete_pct} />
      </td>
      <td className="py-1.5 text-right text-xs">
        {hasMissing ? (
          <span className="text-rose-600 font-medium">
            {col.null_count.toLocaleString()} missing
          </span>
        ) : (
          <span className="text-emerald-600">complete</span>
        )}
      </td>
    </tr>
  )
}

export function NullMapCard({ result }: NullMapCardProps) {
  const completenessColor =
    result.overall_completeness === 100
      ? "bg-emerald-100 text-emerald-800 border-emerald-300"
      : result.overall_completeness >= 90
      ? "bg-amber-100 text-amber-800 border-amber-300"
      : "bg-rose-100 text-rose-800 border-rose-300"

  return (
    <div
      className="rounded-lg border-2 border-teal-200 bg-card p-4 mt-2"
      role="region"
      aria-label="Missing values overview"
    >
      <div className="flex items-center gap-2 mb-1">
        <span aria-hidden="true" className="text-teal-500">◉</span>
        <span className="font-semibold text-sm">Data Completeness</span>
        <Badge className={`${completenessColor} text-xs`}>
          {result.overall_completeness}% complete
        </Badge>
      </div>
      <p className="text-xs text-muted-foreground mb-3">
        {result.columns_with_nulls > 0
          ? `${result.columns_with_nulls} of ${result.total_columns} columns have missing values`
          : `All ${result.total_columns} columns are fully complete`}{" "}
        &mdash; {result.total_rows.toLocaleString()} rows
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-sm" aria-label="Column completeness table">
          <thead>
            <tr className="text-left">
              <th className="pb-1 text-xs text-muted-foreground font-medium pr-3">Column</th>
              <th className="pb-1 text-xs text-muted-foreground font-medium pr-3">Completeness</th>
              <th className="pb-1 text-xs text-muted-foreground font-medium text-right">Status</th>
            </tr>
          </thead>
          <tbody>
            {result.columns.map((col) => (
              <NullRow key={col.column} col={col} />
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-3 text-xs text-muted-foreground border-t border-border/50 pt-2">
        {result.summary}
      </p>
    </div>
  )
}
