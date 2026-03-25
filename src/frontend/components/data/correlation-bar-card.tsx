"use client"

import type { TargetCorrelationResult, CorrelationEntry } from "@/lib/types"

interface CorrelationBarCardProps {
  result: TargetCorrelationResult
}

function strengthColor(entry: CorrelationEntry): string {
  const abs = Math.abs(entry.correlation)
  if (entry.direction === "positive") {
    if (abs >= 0.6) return "bg-blue-500"
    if (abs >= 0.4) return "bg-blue-400"
    if (abs >= 0.2) return "bg-blue-300"
    return "bg-blue-200"
  } else {
    if (abs >= 0.6) return "bg-red-500"
    if (abs >= 0.4) return "bg-red-400"
    if (abs >= 0.2) return "bg-red-300"
    return "bg-red-200"
  }
}

function StrengthBadge({ entry }: { entry: CorrelationEntry }) {
  const colorClass =
    entry.strength === "very strong" || entry.strength === "strong"
      ? entry.direction === "positive"
        ? "bg-blue-100 text-blue-800"
        : "bg-red-100 text-red-800"
      : entry.strength === "moderate"
        ? "bg-yellow-100 text-yellow-800"
        : "bg-muted text-muted-foreground"

  return (
    <span
      className={`ml-1 rounded px-1 py-0.5 text-[10px] font-medium ${colorClass}`}
    >
      {entry.strength}
    </span>
  )
}

export function CorrelationBarCard({ result }: CorrelationBarCardProps) {
  const { target_col, correlations, summary } = result

  if (!correlations || correlations.length === 0) {
    return (
      <div
        data-testid="correlation-bar-card"
        className="mt-2 rounded-lg border border bg-muted/30 p-3"
      >
        <p className="text-sm text-muted-foreground">{summary}</p>
      </div>
    )
  }

  const maxAbs = Math.max(...correlations.map((e) => Math.abs(e.correlation)))

  return (
    <div
      data-testid="correlation-bar-card"
      className="mt-2 rounded-lg border border bg-card shadow-sm"
    >
      {/* Header */}
      <div className="border-b border px-3 py-2">
        <h3 className="text-sm font-semibold text-foreground">
          Correlations with{" "}
          <span className="text-blue-600">{target_col.replace(/_/g, " ")}</span>
        </h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          ↑ positive (blue) · ↓ negative (red) · bar width = strength
        </p>
      </div>

      {/* Correlation bars */}
      <div className="space-y-1 px-3 py-2">
        {correlations.map((entry) => {
          const widthPct =
            maxAbs > 0
              ? Math.round((Math.abs(entry.correlation) / maxAbs) * 100)
              : 0
          const dirSymbol = entry.direction === "positive" ? "+" : "−"

          return (
            <div key={entry.column} className="flex items-center gap-2">
              {/* Column name */}
              <span
                className="w-28 shrink-0 truncate text-right text-xs text-foreground"
                title={entry.column}
              >
                {entry.column.replace(/_/g, " ")}
              </span>

              {/* Bar */}
              <div className="flex flex-1 items-center gap-1">
                <div className="h-4 flex-1 overflow-hidden rounded bg-muted">
                  <div
                    className={`h-full rounded ${strengthColor(entry)} transition-all`}
                    style={{ width: `${widthPct}%` }}
                  />
                </div>
                {/* Direction arrow + numeric value */}
                <span
                  className={`w-16 text-right text-xs font-medium tabular-nums ${
                    entry.direction === "positive"
                      ? "text-blue-700"
                      : "text-red-700"
                  }`}
                >
                  <span aria-label={entry.direction === "positive" ? "positive" : "negative"}>
                    {entry.direction === "positive" ? "↑" : "↓"}
                  </span>{" "}
                  {Math.abs(entry.correlation).toFixed(2)}
                </span>
              </div>

              <StrengthBadge entry={entry} />
            </div>
          )
        })}
      </div>

      {/* Summary */}
      <div className="border-t border px-3 py-2">
        <p className="text-xs text-muted-foreground">{summary}</p>
      </div>
    </div>
  )
}
