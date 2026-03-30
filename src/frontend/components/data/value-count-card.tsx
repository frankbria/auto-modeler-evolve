"use client"

import { Badge } from "@/components/ui/badge"
import type { ValueCountResult, ValueCountRow } from "@/lib/types"

interface ValueCountCardProps {
  result: ValueCountResult
}

function ValueRow({ row, maxCount }: { row: ValueCountRow; maxCount: number }) {
  const barWidth = maxCount > 0 ? Math.round((row.count / maxCount) * 100) : 0
  return (
    <tr className="border-t border-border/50 hover:bg-muted/30">
      <td
        className="py-1.5 pr-3 text-xs truncate max-w-[120px] font-medium"
        title={row.value}
      >
        &ldquo;{row.value}&rdquo;
      </td>
      <td className="py-1.5 pr-3">
        <div className="flex items-center gap-2">
          <div className="h-2 w-24 rounded-full bg-muted overflow-hidden flex-shrink-0">
            <div
              className="h-full rounded-full bg-lime-500 transition-all"
              style={{ width: `${barWidth}%` }}
              aria-hidden="true"
            />
          </div>
          <span className="text-xs tabular-nums text-muted-foreground whitespace-nowrap">
            {row.count.toLocaleString()}
          </span>
        </div>
      </td>
      <td className="py-1.5 text-right text-xs tabular-nums text-muted-foreground">
        {row.pct}%
      </td>
    </tr>
  )
}

export function ValueCountCard({ result }: ValueCountCardProps) {
  const maxCount = result.rows.length > 0 ? result.rows[0].count : 1

  return (
    <div
      className="rounded-lg border-2 border-lime-300 bg-card p-4 mt-2"
      role="region"
      aria-label={`Value frequency table for ${result.column}`}
    >
      <div className="flex items-center gap-2 mb-1">
        <span aria-hidden="true" className="text-lime-600">#</span>
        <span className="font-semibold text-sm">
          Value Counts:{" "}
          <span className="font-mono">{result.column.replace(/_/g, " ")}</span>
        </span>
        <Badge variant="secondary" className="text-xs">
          {result.unique_count} unique
        </Badge>
        {result.null_count > 0 && (
          <Badge className="text-xs bg-rose-100 text-rose-800 border-rose-300">
            {result.null_count} null
          </Badge>
        )}
      </div>
      <p className="text-xs text-muted-foreground mb-3">
        {result.total_rows.toLocaleString()} total rows
        {result.null_count > 0
          ? `, ${result.non_null.toLocaleString()} non-null`
          : ""}
      </p>

      <div className="overflow-x-auto">
        <table
          className="w-full text-sm"
          aria-label={`Frequency table for column ${result.column}`}
        >
          <thead>
            <tr className="text-left">
              <th className="pb-1 text-xs text-muted-foreground font-medium pr-3">Value</th>
              <th className="pb-1 text-xs text-muted-foreground font-medium pr-3">Count</th>
              <th className="pb-1 text-xs text-muted-foreground font-medium text-right">%</th>
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row) => (
              <ValueRow key={row.value} row={row} maxCount={maxCount} />
            ))}
          </tbody>
        </table>
      </div>

      {result.has_more && (
        <p className="mt-2 text-xs text-muted-foreground italic">
          Showing top {result.rows.length} of {result.unique_count} values.
        </p>
      )}

      <p className="mt-3 text-xs text-muted-foreground border-t border-border/50 pt-2">
        {result.summary}
      </p>
    </div>
  )
}
