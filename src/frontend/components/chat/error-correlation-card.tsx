"use client"

import { Badge } from "@/components/ui/badge"
import type { ErrorCorrelationFeature, ErrorCorrelationResult } from "@/lib/types"

interface Props {
  result: ErrorCorrelationResult
}

function VerdictBadge({ verdict }: { verdict: ErrorCorrelationResult["verdict"] }) {
  if (verdict === "clear_drivers") {
    return (
      <Badge className="bg-rose-100 text-rose-800 border border-rose-200">
        Clear Error Drivers Found
      </Badge>
    )
  }
  if (verdict === "weak_drivers") {
    return (
      <Badge className="bg-amber-100 text-amber-800 border border-amber-200">
        Weak Error Drivers
      </Badge>
    )
  }
  return (
    <Badge className="bg-emerald-100 text-emerald-800 border border-emerald-200">
      No Clear Error Drivers
    </Badge>
  )
}

function DirectionBadge({ direction }: { direction: ErrorCorrelationFeature["direction"] }) {
  if (direction === "positive") {
    return (
      <span
        className="text-xs text-rose-600 font-medium"
        title="Higher values → more errors"
      >
        ↑ more errors
      </span>
    )
  }
  if (direction === "negative") {
    return (
      <span
        className="text-xs text-sky-600 font-medium"
        title="Lower values → more errors"
      >
        ↓ more errors
      </span>
    )
  }
  return <span className="text-xs text-muted-foreground">≈ neutral</span>
}

function CorrelationBar({ value, maxAbs }: { value: number; maxAbs: number }) {
  const pct = maxAbs > 0 ? (Math.abs(value) / maxAbs) * 100 : 0
  const color =
    Math.abs(value) >= 0.3
      ? value >= 0
        ? "bg-rose-500"
        : "bg-sky-500"
      : Math.abs(value) >= 0.1
        ? "bg-amber-400"
        : "bg-muted-foreground/30"

  return (
    <div
      className="h-2 rounded-full bg-muted overflow-hidden"
      aria-label={`Correlation: ${value.toFixed(3)}`}
      role="img"
    >
      <div
        className={`h-full rounded-full transition-all ${color}`}
        style={{ width: `${Math.max(2, pct)}%` }}
      />
    </div>
  )
}

export function ErrorCorrelationCard({ result }: Props) {
  const {
    features,
    error_type,
    n_total,
    n_errors,
    error_rate,
    verdict,
    algorithm,
    target_col,
    top_driver,
    summary,
  } = result

  const maxAbs =
    features.length > 0 ? Math.max(...features.map((f) => f.correlation_abs)) : 1

  const borderColor =
    verdict === "clear_drivers"
      ? "border-rose-300 bg-rose-50"
      : verdict === "weak_drivers"
        ? "border-amber-300 bg-amber-50"
        : "border-emerald-300 bg-emerald-50"

  const errorLabel =
    error_type === "absolute_residual"
      ? "absolute residual"
      : "misclassification"

  return (
    <figure
      className={`rounded-lg border-2 p-4 space-y-3 ${borderColor}`}
      aria-label="Prediction error correlation analysis"
    >
      {/* Header */}
      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-base font-semibold text-foreground">
          🔎 Error Correlation
        </span>
        <VerdictBadge verdict={verdict} />
        <Badge
          variant="outline"
          className="font-mono text-xs"
          data-testid="algorithm-badge"
        >
          {algorithm}
        </Badge>
        <Badge variant="outline" className="text-xs" data-testid="target-badge">
          target: {target_col}
        </Badge>
      </div>

      {/* Stats row */}
      <div className="flex flex-wrap gap-4 text-sm">
        <span className="text-muted-foreground">
          <span className="font-medium text-foreground">{n_total}</span> rows
        </span>
        <span className="text-muted-foreground">
          <span className="font-medium text-foreground">{n_errors}</span>{" "}
          {error_type === "absolute_residual" ? "with residual > 0" : "misclassified"}
        </span>
        {error_rate !== null && (
          <span className="text-muted-foreground">
            error rate:{" "}
            <span className="font-medium text-foreground">
              {(error_rate * 100).toFixed(1)}%
            </span>
          </span>
        )}
        <span className="text-muted-foreground text-xs">
          error signal: {errorLabel}
        </span>
      </div>

      {/* Feature correlation list */}
      {features.length > 0 ? (
        <div className="space-y-2" role="list" aria-label="Feature correlation with errors">
          <div className="grid grid-cols-[1.5rem_1fr_6rem_5rem] gap-x-2 text-xs font-medium text-muted-foreground pb-1 border-b">
            <span>#</span>
            <span>Feature</span>
            <span>Correlation</span>
            <span>Direction</span>
          </div>
          {features.map((feat) => (
            <div
              key={feat.feature}
              className="grid grid-cols-[1.5rem_1fr_6rem_5rem] gap-x-2 items-center"
              role="listitem"
              data-testid={`error-correlation-row-${feat.feature}`}
            >
              <span className="text-xs text-muted-foreground">{feat.rank}</span>
              <div className="space-y-1 min-w-0">
                <span className="text-sm font-medium truncate block" title={feat.feature}>
                  {feat.feature}
                </span>
                <CorrelationBar value={feat.correlation} maxAbs={maxAbs} />
              </div>
              <span
                className="text-xs tabular-nums font-mono"
                data-testid={`correlation-value-${feat.feature}`}
              >
                r = {feat.correlation.toFixed(3)}
              </span>
              <DirectionBadge direction={feat.direction} />
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground italic">
          No numeric features available for correlation analysis.
        </p>
      )}

      {/* Top driver callout */}
      {top_driver && verdict === "clear_drivers" && (
        <div
          className="rounded-md bg-rose-100 border border-rose-200 p-3 text-sm text-rose-800"
          data-testid="top-driver-callout"
        >
          <span className="font-semibold">Top error driver:</span>{" "}
          <code className="bg-rose-50 px-1 rounded text-xs font-mono">{top_driver}</code>
          {" — focus data collection or feature engineering here."}
        </div>
      )}

      {/* Summary */}
      <p className="text-sm text-muted-foreground" data-testid="error-correlation-summary">
        {summary}
      </p>

      <figcaption className="sr-only">
        Prediction error correlation analysis for {target_col} using {algorithm}.
        {top_driver
          ? ` Top error driver: ${top_driver}.`
          : " No strong error drivers identified."}
      </figcaption>
    </figure>
  )
}
