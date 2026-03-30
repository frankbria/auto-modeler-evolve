"use client"

import { Badge } from "@/components/ui/badge"
import type { StatQueryResult } from "@/lib/types"

interface StatQueryCardProps {
  result: StatQueryResult
}

const AGG_ICONS: Record<string, string> = {
  sum: "Σ",
  mean: "x̄",
  median: "m",
  max: "↑",
  min: "↓",
  std: "σ",
  count: "#",
}

const AGG_COLORS: Record<string, string> = {
  sum: "border-blue-300",
  mean: "border-cyan-300",
  median: "border-teal-300",
  max: "border-emerald-300",
  min: "border-orange-300",
  std: "border-purple-300",
  count: "border-amber-300",
}

export function StatQueryCard({ result }: StatQueryCardProps) {
  const icon = AGG_ICONS[result.agg] ?? "?"
  const borderColor = AGG_COLORS[result.agg] ?? "border-muted"
  const label = result.label ?? result.agg

  return (
    <div
      className={`rounded-lg border-2 ${borderColor} bg-card p-4 mt-2`}
      role="region"
      aria-label={`${label}${result.col ? ` of ${result.col}` : ""}: ${result.formatted_value}`}
    >
      <div className="flex items-center gap-2 mb-1 flex-wrap">
        <span
          aria-hidden="true"
          className="text-lg font-bold text-muted-foreground font-mono"
        >
          {icon}
        </span>
        <span className="font-semibold text-sm capitalize">
          {label}
          {result.col && (
            <>
              {" of "}
              <span className="font-mono">{result.col.replace(/_/g, " ")}</span>
            </>
          )}
        </span>
        <Badge variant="secondary" className="text-xs capitalize">
          {result.agg}
        </Badge>
      </div>

      {result.n_valid !== undefined && result.n_valid < result.n_rows && (
        <p className="text-xs text-muted-foreground mb-2">
          {result.n_valid.toLocaleString()} non-null values out of {result.n_rows.toLocaleString()} rows
        </p>
      )}

      {/* Big number display */}
      <p
        className="text-3xl font-bold tabular-nums text-foreground my-3"
        aria-label={`Result: ${result.formatted_value}`}
      >
        {result.formatted_value}
      </p>

      <p className="text-xs text-muted-foreground border-t border-border/50 pt-2">
        {result.summary}
      </p>
    </div>
  )
}
