/**
 * PredictionDeltaCard — explains why a prediction changed between model versions.
 *
 * Shown when the analyst asks: "why did my prediction change?" or
 * "explain the prediction delta after retraining".
 *
 * Layout:
 *   Header: version badge (v{prev} → v{curr}), target, direction chip
 *   Prediction row: old value → new value with delta / % change
 *   Feature-delta bar chart: contribution_delta per feature (horizontal bars)
 *   Top drivers chips
 *   Summary sentence
 */
"use client"

import { CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { PredictionDeltaResult, PredictionDeltaFeature } from "@/lib/types"

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmtPred(v: number | string): string {
  if (typeof v === "number") {
    return v.toLocaleString(undefined, { maximumFractionDigits: 4 })
  }
  return String(v)
}

function fmtDelta(delta: number, pct_change: number): string {
  const sign = delta >= 0 ? "+" : ""
  return `${sign}${delta.toLocaleString(undefined, { maximumFractionDigits: 4 })} (${sign}${pct_change.toFixed(1)}%)`
}

function directionColors(direction: PredictionDeltaResult["direction"]) {
  switch (direction) {
    case "up":
      return {
        bg: "bg-emerald-50",
        border: "border-emerald-300",
        header: "bg-emerald-50 border-emerald-200",
        title: "text-emerald-900",
        arrow: "↑",
        badge: "bg-emerald-100 text-emerald-800 border-emerald-300",
        label: "Increased",
      }
    case "down":
      return {
        bg: "bg-rose-50",
        border: "border-rose-300",
        header: "bg-rose-50 border-rose-200",
        title: "text-rose-900",
        arrow: "↓",
        badge: "bg-rose-100 text-rose-800 border-rose-300",
        label: "Decreased",
      }
    default:
      return {
        bg: "bg-slate-50",
        border: "border-slate-300",
        header: "bg-slate-50 border-slate-200",
        title: "text-slate-900",
        arrow: "→",
        badge: "bg-slate-100 text-slate-700 border-slate-300",
        label: direction === "changed" ? "Class changed" : "Unchanged",
      }
  }
}

// ── Feature bar ───────────────────────────────────────────────────────────────

function FeatureDeltaBar({ feat, maxAbs }: { feat: PredictionDeltaFeature; maxAbs: number }) {
  const pct = maxAbs > 0 ? Math.min(100, (Math.abs(feat.contribution_delta) / maxAbs) * 100) : 0
  const isIncrease = feat.direction === "increased"
  const isStable = feat.direction === "stable"
  const barColor = isStable
    ? "bg-muted"
    : isIncrease
    ? "bg-emerald-400"
    : "bg-rose-400"

  return (
    <div
      className="flex items-center gap-2 text-sm"
      data-testid="feature-delta-row"
      role="row"
    >
      <div className="w-32 shrink-0 truncate text-xs text-muted-foreground text-right pr-1" title={feat.feature}>
        {feat.feature.replace(/_/g, " ")}
      </div>
      <div className="flex-1 h-3 bg-muted/40 rounded overflow-hidden">
        <div
          className={`h-full rounded transition-all ${barColor}`}
          style={{ width: `${pct}%` }}
          aria-label={`${feat.feature} contribution delta ${feat.contribution_delta > 0 ? "+" : ""}${feat.contribution_delta.toFixed(4)}`}
        />
      </div>
      <div
        className={`w-20 text-right text-xs tabular-nums shrink-0 ${
          isIncrease ? "text-emerald-700" : isStable ? "text-muted-foreground" : "text-rose-700"
        }`}
      >
        {feat.contribution_delta > 0 ? "+" : ""}
        {feat.contribution_delta.toFixed(4)}
      </div>
    </div>
  )
}

// ── Main card ─────────────────────────────────────────────────────────────────

interface Props {
  result: PredictionDeltaResult
}

export function PredictionDeltaCard({ result }: Props) {
  const {
    old_prediction,
    new_prediction,
    delta,
    pct_change,
    direction,
    problem_type,
    target_column,
    provided_features,
    feature_delta,
    top_drivers,
    summary,
    current_version,
    previous_version,
    inputs_from_message,
  } = result

  const colors = directionColors(direction)
  const isRegression = problem_type === "regression"

  // Feature delta bars — limit to top 8
  const maxAbs = feature_delta.length > 0 ? Math.abs(feature_delta[0].contribution_delta) : 1
  const topFeatures = feature_delta.slice(0, 8)

  const n_provided = Object.keys(provided_features).length

  return (
    <div
      className={`border ${colors.border} rounded-lg overflow-hidden`}
      role="region"
      aria-label="Prediction delta analysis"
      data-testid="prediction-delta-card"
    >
      {/* Header */}
      <CardHeader className={`${colors.header} border-b py-3 px-4`}>
        <div className="flex items-center gap-2 flex-wrap">
          <span aria-hidden="true" className="text-lg">🔍</span>
          <CardTitle className={`text-base font-semibold ${colors.title}`}>
            Why Did the Prediction Change?
          </CardTitle>
          <Badge
            className="bg-slate-100 text-slate-700 border-slate-300 text-xs font-mono"
            variant="outline"
            data-testid="version-badge"
          >
            v{previous_version} → v{current_version}
          </Badge>
          <Badge
            className={`text-xs font-medium ${colors.badge}`}
            variant="outline"
            data-testid="direction-badge"
          >
            {colors.arrow} {colors.label}
          </Badge>
          <Badge
            className="bg-sky-100 text-sky-700 border-sky-300 text-xs"
            variant="outline"
          >
            Target: {target_column}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="p-4 space-y-4">
        {/* Prediction before / after */}
        <div
          className="flex items-center gap-3 flex-wrap"
          data-testid="prediction-comparison"
        >
          <div className="flex flex-col items-center">
            <span className="text-xs text-muted-foreground mb-0.5">v{previous_version}</span>
            <span className="text-lg font-semibold text-muted-foreground" data-testid="old-prediction">
              {fmtPred(old_prediction)}
            </span>
          </div>

          <span
            className={`text-2xl font-bold ${
              direction === "up" ? "text-emerald-500" : direction === "down" ? "text-rose-500" : "text-muted-foreground"
            }`}
            aria-hidden="true"
          >
            {colors.arrow}
          </span>

          <div className="flex flex-col items-center">
            <span className="text-xs text-muted-foreground mb-0.5">v{current_version}</span>
            <span
              className={`text-lg font-semibold ${
                direction === "up" ? "text-emerald-700" : direction === "down" ? "text-rose-700" : "text-foreground"
              }`}
              data-testid="new-prediction"
            >
              {fmtPred(new_prediction)}
            </span>
          </div>

          {isRegression && direction !== "unchanged" && (
            <span
              className={`ml-2 text-sm font-medium ${
                direction === "up" ? "text-emerald-700" : "text-rose-700"
              }`}
              data-testid="delta-display"
            >
              {fmtDelta(delta, pct_change)}
            </span>
          )}
        </div>

        {/* Feature inputs summary */}
        {n_provided > 0 && (
          <div className="flex flex-wrap gap-1 items-center" data-testid="inputs-summary">
            <span className="text-xs text-muted-foreground font-medium">
              {inputs_from_message ? "Inputs from your message:" : "Using training-data averages:"}
            </span>
            {Object.entries(provided_features)
              .slice(0, 6)
              .map(([k, v]) => (
                <Badge
                  key={k}
                  variant="outline"
                  className="text-xs bg-blue-50 text-blue-700 border-blue-200"
                >
                  {k.replace(/_/g, " ")}={typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : String(v)}
                </Badge>
              ))}
            {n_provided > 6 && (
              <span className="text-xs text-muted-foreground">+{n_provided - 6} more</span>
            )}
          </div>
        )}

        {/* Feature delta bars */}
        {topFeatures.length > 0 && (
          <div className="space-y-1.5" role="table" aria-label="Feature contribution deltas">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Feature Contribution Shift
            </p>
            {topFeatures.map((feat) => (
              <FeatureDeltaBar key={feat.feature} feat={feat} maxAbs={maxAbs} />
            ))}
          </div>
        )}

        {/* Top drivers chips */}
        {top_drivers.length > 0 && (
          <div className="flex flex-wrap gap-1.5 items-center" data-testid="top-drivers">
            <span className="text-xs text-muted-foreground font-medium">Top drivers:</span>
            {top_drivers.map((d, i) => (
              <Badge
                key={i}
                variant="outline"
                className="text-xs bg-amber-50 text-amber-800 border-amber-300"
              >
                {d}
              </Badge>
            ))}
          </div>
        )}

        {/* Summary */}
        <p
          className="text-xs text-muted-foreground italic border-t border-muted pt-2"
          data-testid="prediction-delta-summary"
        >
          {summary}
        </p>

        {/* Accessibility */}
        <figcaption className="sr-only">
          Prediction change analysis from version {previous_version} to {current_version}. {summary}
        </figcaption>
      </CardContent>
    </div>
  )
}
