"use client"

import { InteractionResult } from "@/lib/types"

interface InteractionCardProps {
  result: InteractionResult
}

function fmtVal(v: number | string | null | undefined): string {
  if (v === null || v === undefined) return "—"
  if (typeof v === "string") return v
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}k`
  return parseFloat(v.toFixed(3)).toString()
}

function cellColor(
  value: number | string,
  minVal: number | null,
  maxVal: number | null,
): string {
  if (typeof value !== "number" || minVal === null || maxVal === null) {
    return "bg-violet-100 text-violet-800"
  }
  const range = maxVal - minVal
  if (range === 0) return "bg-sky-100 text-sky-800"
  const ratio = (value - minVal) / range // 0 = min, 1 = max
  if (ratio >= 0.8) return "bg-emerald-200 text-emerald-900 font-semibold"
  if (ratio >= 0.6) return "bg-emerald-100 text-emerald-800"
  if (ratio >= 0.4) return "bg-sky-100 text-sky-800"
  if (ratio >= 0.2) return "bg-amber-100 text-amber-800"
  return "bg-rose-100 text-rose-800"
}

export function InteractionCard({ result }: InteractionCardProps) {
  const {
    feature1,
    feature2,
    target_column,
    problem_type,
    row_labels,
    col_labels,
    values,
    min_val,
    max_val,
    summary,
  } = result

  const feat1Label = feature1.replace(/_/g, " ")
  const feat2Label = feature2.replace(/_/g, " ")
  const targetLabel = (target_column ?? "prediction").replace(/_/g, " ")
  const isRegression = problem_type === "regression"

  return (
    <figure
      className="rounded-lg border-2 border-violet-400 bg-violet-50 dark:bg-violet-950/20 dark:border-violet-600 p-4 text-sm my-2"
      aria-label={`Feature interaction: ${feat1Label} × ${feat2Label} vs ${targetLabel}`}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-base" aria-hidden="true">🔬</span>
        <p className="font-semibold text-foreground">
          Interaction: <span className="capitalize">{feat1Label}</span>
          <span className="text-muted-foreground mx-1">×</span>
          <span className="capitalize">{feat2Label}</span>
          <span className="text-muted-foreground mx-1">→</span>
          <span className="capitalize">{targetLabel}</span>
        </p>
        <span className="rounded-full border border-violet-300 bg-violet-100 px-2 py-0.5 text-xs font-medium text-violet-800 dark:bg-violet-900/40 dark:text-violet-300">
          {isRegression ? "Regression" : "Classification"}
        </span>
      </div>

      {/* Min / Max range for regression */}
      {isRegression && min_val !== null && max_val !== null && (
        <div className="flex gap-3 mb-3">
          <div className="rounded-md border border-border bg-white dark:bg-violet-950/30 px-3 py-1.5 text-center">
            <p className="text-xs text-muted-foreground">Min {targetLabel}</p>
            <p className="font-bold text-sm text-rose-700">{fmtVal(min_val)}</p>
          </div>
          <div className="rounded-md border border-violet-300 bg-violet-100/60 dark:bg-violet-900/20 px-3 py-1.5 text-center">
            <p className="text-xs text-muted-foreground">Max {targetLabel}</p>
            <p className="font-bold text-sm text-emerald-700">{fmtVal(max_val)}</p>
          </div>
        </div>
      )}

      {/* Heatmap grid */}
      <div className="overflow-x-auto mb-3">
        <table className="text-xs border-collapse" data-testid="interaction-grid">
          <thead>
            <tr>
              {/* Corner cell: feat1 (rows) \ feat2 (cols) */}
              <th className="px-2 py-1 text-left font-medium text-muted-foreground whitespace-nowrap border border-violet-200 bg-violet-50">
                <span className="capitalize">{feat1Label}</span>
                <span className="text-muted-foreground mx-0.5">\</span>
                <span className="capitalize">{feat2Label}</span>
              </th>
              {col_labels.map((col) => (
                <th
                  key={col}
                  className="px-2 py-1 text-center font-medium text-muted-foreground whitespace-nowrap border border-violet-200 bg-violet-50 max-w-[80px] truncate"
                  title={col}
                >
                  {col.length > 8 ? col.slice(0, 8) + "…" : col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {row_labels.map((rowLabel, ri) => (
              <tr key={rowLabel}>
                <td className="px-2 py-1 font-medium text-muted-foreground whitespace-nowrap border border-violet-200 bg-violet-50 max-w-[80px] truncate"
                    title={rowLabel}>
                  {rowLabel.length > 8 ? rowLabel.slice(0, 8) + "…" : rowLabel}
                </td>
                {col_labels.map((_col, ci) => {
                  const val = values[ri]?.[ci]
                  const numVal = typeof val === "number" ? val : null
                  const colorClass = cellColor(val ?? "", min_val, max_val)
                  return (
                    <td
                      key={ci}
                      className={`px-2 py-1 text-center tabular-nums border border-violet-200 ${colorClass}`}
                      title={`${feat1Label}=${rowLabel}, ${feat2Label}=${_col}: ${fmtVal(val)}`}
                    >
                      {fmtVal(numVal ?? val)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Color legend for regression */}
      {isRegression && (
        <div className="flex items-center gap-1 text-xs text-muted-foreground mb-2">
          <span>Low</span>
          <div className="flex gap-0.5">
            <span className="w-4 h-3 rounded-sm bg-rose-100" />
            <span className="w-4 h-3 rounded-sm bg-amber-100" />
            <span className="w-4 h-3 rounded-sm bg-sky-100" />
            <span className="w-4 h-3 rounded-sm bg-emerald-100" />
            <span className="w-4 h-3 rounded-sm bg-emerald-200" />
          </div>
          <span>High</span>
        </div>
      )}

      {/* Summary footer */}
      <figcaption className="mt-1 text-xs text-muted-foreground border-t border-violet-200 pt-2">
        {summary}
      </figcaption>
    </figure>
  )
}
