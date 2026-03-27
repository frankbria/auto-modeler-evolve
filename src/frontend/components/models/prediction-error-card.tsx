"use client"

import type { PredictionErrorResult, PredictionErrorRow } from "@/lib/types"

interface PredictionErrorCardProps {
  result: PredictionErrorResult
}

function formatValue(v: string | number | null | undefined): string {
  if (v === null || v === undefined) return "—"
  if (typeof v === "number") {
    if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
    if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}k`
    return Number.isInteger(v) ? String(v) : v.toFixed(3)
  }
  return String(v)
}

function ErrorBadge({ error, absError, problemType }: {
  error: string | number
  absError: number | null
  problemType: string
}) {
  if (problemType === "regression" && typeof error === "number") {
    const color = error > 0 ? "bg-blue-50 text-blue-700" : "bg-red-50 text-red-700"
    const sign = error > 0 ? "+" : ""
    return (
      <span className={`inline-block rounded px-1.5 py-0.5 text-xs font-medium ${color}`}>
        {sign}{formatValue(error)}
        {absError !== null && (
          <span className="ml-1 opacity-60">(|{formatValue(absError)}|)</span>
        )}
      </span>
    )
  }
  return (
    <span className="inline-block rounded bg-red-50 px-1.5 py-0.5 text-xs text-red-700">
      {String(error)}
    </span>
  )
}

function FeatureChips({ features }: { features: Record<string, string | number> }) {
  const entries = Object.entries(features).slice(0, 4)
  return (
    <div className="mt-1 flex flex-wrap gap-1">
      {entries.map(([k, v]) => (
        <span
          key={k}
          className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground"
        >
          {k.replace(/_/g, " ")}: {formatValue(v as string | number)}
        </span>
      ))}
      {Object.keys(features).length > 4 && (
        <span className="text-xs text-muted-foreground">
          +{Object.keys(features).length - 4} more
        </span>
      )}
    </div>
  )
}

function ErrorRow({ row, problemType, index }: {
  row: PredictionErrorRow
  problemType: string
  index: number
}) {
  const isEven = index % 2 === 0
  return (
    <div className={`rounded-md px-3 py-2 ${isEven ? "bg-muted/30" : ""}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="w-5 text-right text-xs font-semibold text-muted-foreground">
            {row.rank}
          </span>
          <div>
            <div className="flex items-center gap-1.5 text-sm">
              <span className="font-medium">{formatValue(row.actual)}</span>
              <span className="text-muted-foreground">→</span>
              <span className="text-muted-foreground">{formatValue(row.predicted)}</span>
            </div>
            {row.features && Object.keys(row.features).length > 0 && (
              <FeatureChips features={row.features} />
            )}
          </div>
        </div>
        <ErrorBadge
          error={row.error}
          absError={row.abs_error}
          problemType={problemType}
        />
      </div>
    </div>
  )
}

export function PredictionErrorCard({ result }: PredictionErrorCardProps) {
  const isRegression = result.problem_type === "regression"
  const errorCount = result.errors.length

  return (
    <div
      className="my-2 rounded-lg border border-rose-200 bg-rose-50/30 p-4"
      data-testid="prediction-error-card"
    >
      {/* Header */}
      <div className="mb-3 flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-base font-semibold text-foreground">
              Prediction Errors
            </span>
            <span className="rounded-full bg-rose-100 px-2 py-0.5 text-xs font-medium text-rose-700">
              {result.algorithm}
            </span>
            <span className="rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground capitalize">
              {result.problem_type}
            </span>
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Target: <span className="font-medium">{result.target_col}</span>
            {!isRegression && result.total_errors > 0 && (
              <span className="ml-2">
                · {result.total_errors} wrong
                {" "}({(result.error_rate * 100).toFixed(0)}% error rate)
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Column headers */}
      {errorCount > 0 && (
        <div className="mb-1 flex items-center gap-2 px-3 text-xs font-medium text-muted-foreground">
          <span className="w-5 text-right">#</span>
          <div className="flex-1">
            <span>Actual → Predicted</span>
          </div>
          <span>{isRegression ? "Error" : "Misclassified"}</span>
        </div>
      )}

      {/* Error rows */}
      {errorCount === 0 ? (
        <p className="py-4 text-center text-sm text-muted-foreground">
          No prediction errors found — the model fits the training data perfectly.
        </p>
      ) : (
        <div className="space-y-1">
          {result.errors.map((row, i) => (
            <ErrorRow
              key={row.rank}
              row={row}
              problemType={result.problem_type}
              index={i}
            />
          ))}
        </div>
      )}

      {/* Summary footer */}
      <p className="mt-3 border-t border-rose-100 pt-2 text-xs text-muted-foreground">
        {result.summary}
      </p>
    </div>
  )
}
