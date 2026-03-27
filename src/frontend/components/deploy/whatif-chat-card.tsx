"use client"

import { WhatIfChatResult } from "@/lib/types"

interface WhatIfChatCardProps {
  result: WhatIfChatResult
}

function formatValue(val: number | string | null | undefined): string {
  if (val === null || val === undefined) return "N/A"
  if (typeof val === "number") {
    if (Math.abs(val) >= 1_000_000) return `${(val / 1_000_000).toFixed(2)}M`
    if (Math.abs(val) >= 1_000) return `${(val / 1_000).toFixed(1)}k`
    return Number.isInteger(val) ? String(val) : val.toFixed(4).replace(/\.?0+$/, "")
  }
  return String(val)
}

function DeltaBadge({
  direction,
  delta,
  pct,
}: {
  direction: string | null
  delta: number | null
  pct: number | null
}) {
  if (direction === null || delta === null) return null
  const color =
    direction === "increase"
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : direction === "decrease"
        ? "bg-rose-50 text-rose-700 border-rose-200"
        : "bg-muted text-muted-foreground border-border"
  const arrow =
    direction === "increase" ? "↑" : direction === "decrease" ? "↓" : "→"
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${color}`}
    >
      {arrow} {delta > 0 ? "+" : ""}
      {formatValue(delta)}
      {pct !== null && ` (${pct > 0 ? "+" : ""}${pct.toFixed(1)}%)`}
    </span>
  )
}

function ProbabilityRow({
  label,
  probs,
}: {
  label: string
  probs: Record<string, number>
}) {
  return (
    <div className="mt-1">
      <p className="text-xs font-medium text-muted-foreground mb-1">{label}</p>
      <div className="flex flex-wrap gap-1">
        {Object.entries(probs).map(([cls, prob]) => (
          <span
            key={cls}
            className="inline-block rounded border border-border bg-muted px-1.5 py-0.5 text-xs"
          >
            {cls}: {(prob * 100).toFixed(1)}%
          </span>
        ))}
      </div>
    </div>
  )
}

export function WhatIfChatCard({ result }: WhatIfChatCardProps) {
  const {
    changed_feature,
    original_feature_value,
    new_feature_value,
    original_prediction,
    modified_prediction,
    delta,
    percent_change,
    direction,
    summary,
    problem_type,
    target_column,
    original_probabilities,
    modified_probabilities,
  } = result

  const featureLabel = changed_feature.replace(/_/g, " ")
  const targetLabel = target_column?.replace(/_/g, " ") ?? "prediction"
  const isClassification = problem_type === "classification"

  return (
    <div
      className="rounded-lg border-2 border-amber-400 bg-amber-50 dark:bg-amber-950/20 dark:border-amber-600 p-4 text-sm my-2"
      data-testid="whatif-chat-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-base" aria-hidden="true">
          🔀
        </span>
        <p className="font-semibold text-foreground">What-If Analysis</p>
        {problem_type && (
          <span className="rounded-full border border-amber-300 bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-900/40 dark:text-amber-300">
            {isClassification ? "Classification" : "Regression"}
          </span>
        )}
      </div>

      {/* Feature change row */}
      <div className="mb-3 rounded-md border border-amber-200 bg-white dark:bg-amber-950/30 p-3">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
          Hypothetical Change
        </p>
        <div className="flex items-center gap-2 text-sm">
          <span className="font-semibold capitalize">{featureLabel}</span>
          <span className="text-muted-foreground">
            {formatValue(original_feature_value)}
          </span>
          <span className="text-muted-foreground">→</span>
          <span className="font-semibold text-amber-700 dark:text-amber-400">
            {formatValue(new_feature_value)}
          </span>
        </div>
      </div>

      {/* Prediction comparison */}
      <div className="mb-3 grid grid-cols-2 gap-2">
        <div className="rounded-md border border-border bg-muted/40 p-3 text-center">
          <p className="text-xs text-muted-foreground mb-1">
            Original {targetLabel}
          </p>
          <p className="text-lg font-bold text-foreground">
            {formatValue(original_prediction)}
          </p>
        </div>
        <div className="rounded-md border border-amber-300 bg-amber-100/60 dark:bg-amber-900/20 p-3 text-center">
          <p className="text-xs text-muted-foreground mb-1">
            Modified {targetLabel}
          </p>
          <p className="text-lg font-bold text-amber-700 dark:text-amber-300">
            {formatValue(modified_prediction)}
          </p>
        </div>
      </div>

      {/* Delta badge */}
      {direction !== null && (
        <div className="mb-3 flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Change:</span>
          <DeltaBadge direction={direction} delta={delta} pct={percent_change} />
        </div>
      )}

      {/* Classification probabilities */}
      {isClassification && original_probabilities && (
        <ProbabilityRow label="Original probabilities" probs={original_probabilities} />
      )}
      {isClassification && modified_probabilities && (
        <ProbabilityRow label="Modified probabilities" probs={modified_probabilities} />
      )}

      {/* Summary footer */}
      <p className="mt-3 text-xs text-muted-foreground border-t border-amber-200 pt-2">
        {summary}
      </p>
    </div>
  )
}
