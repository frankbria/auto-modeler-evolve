/**
 * GoalSeekCard — shows reverse-prediction results for "what inputs produce target output?"
 *
 * Given a desired prediction target, displays the suggested input values the model
 * optimizer found, how close the achieved prediction is, and a plain-English summary.
 */
"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { GoalSeekResult, GoalSeekSuggestion } from "@/lib/types"

interface GoalSeekCardProps {
  result: GoalSeekResult
}

function DirectionBadge({ direction, changePct }: { direction: string; changePct: number }) {
  if (direction === "increase") {
    return (
      <Badge
        data-testid="direction-badge-increase"
        className="bg-emerald-100 text-emerald-800 border-emerald-200 text-[10px]"
      >
        ↑ +{Math.abs(changePct)}%
      </Badge>
    )
  }
  if (direction === "decrease") {
    return (
      <Badge
        data-testid="direction-badge-decrease"
        className="bg-rose-100 text-rose-800 border-rose-200 text-[10px]"
      >
        ↓ -{Math.abs(changePct)}%
      </Badge>
    )
  }
  return (
    <Badge className="bg-gray-100 text-gray-600 border-gray-200 text-[10px]">→ no change</Badge>
  )
}

function SuggestionRow({ suggestion }: { suggestion: GoalSeekSuggestion }) {
  return (
    <div
      data-testid={`suggestion-row-${suggestion.feature}`}
      className="flex items-center justify-between gap-2 py-1.5 border-b border-border/40 last:border-0"
    >
      <span className="text-sm font-medium text-foreground truncate max-w-[140px]" title={suggestion.feature}>
        {suggestion.feature.replace(/_/g, " ")}
      </span>
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-xs text-muted-foreground">
          avg: <span className="font-mono">{suggestion.current_mean.toLocaleString()}</span>
        </span>
        <span className="text-xs text-muted-foreground">→</span>
        <span className="text-sm font-semibold font-mono text-foreground">
          {suggestion.suggested_value.toLocaleString()}
        </span>
        <DirectionBadge direction={suggestion.direction} changePct={suggestion.change_pct} />
      </div>
    </div>
  )
}

function fmt(v: number | string): string {
  if (typeof v === "number") {
    return v.toLocaleString(undefined, { maximumFractionDigits: 2 })
  }
  return String(v)
}

export function GoalSeekCard({ result }: GoalSeekCardProps) {
  const borderClass = result.achieved
    ? "border-emerald-500/40"
    : "border-amber-500/40"

  const achievedBadge = result.achieved ? (
    <Badge
      data-testid="goal-achieved-badge"
      className="bg-emerald-100 text-emerald-800 border-emerald-200"
    >
      ✓ Goal Achieved
    </Badge>
  ) : (
    <Badge
      data-testid="goal-best-effort-badge"
      className="bg-amber-100 text-amber-800 border-amber-200"
    >
      Best Effort
    </Badge>
  )

  const feasibilityNote = !result.feasible && (
    <p className="text-xs text-muted-foreground italic mt-1">
      Note: Optimizer did not fully converge — results are approximate.
    </p>
  )

  return (
    <Card data-testid="goal-seek-card" className={`${borderClass} w-full`}>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2 flex-wrap">
          <span aria-hidden="true">🎯</span>
          <span>Goal Seek</span>
          <Badge className="bg-sky-100 text-sky-800 border-sky-200 text-[10px]">
            {result.target_column}
          </Badge>
          <Badge className="bg-slate-100 text-slate-700 border-slate-200 text-[10px]">
            {result.algorithm_plain}
          </Badge>
          {achievedBadge}
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* Target vs Achieved */}
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-md bg-muted/40 p-2 text-center">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wide mb-0.5">
              Target
            </div>
            <div
              data-testid="target-value"
              className="text-lg font-bold text-foreground font-mono"
            >
              {fmt(result.target_value)}
            </div>
          </div>
          <div className="rounded-md bg-muted/40 p-2 text-center">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wide mb-0.5">
              Model Achieves
            </div>
            <div
              data-testid="achieved-value"
              className={`text-lg font-bold font-mono ${
                result.achieved ? "text-emerald-700" : "text-amber-700"
              }`}
            >
              {fmt(result.achieved_value)}
            </div>
          </div>
        </div>

        {/* Gap indicator (regression only) */}
        {result.gap_pct !== null && result.problem_type === "regression" && !result.achieved && (
          <div
            data-testid="gap-indicator"
            className="text-xs text-center text-amber-700 bg-amber-50 rounded px-2 py-1"
          >
            {result.gap_pct}% gap from target
          </div>
        )}

        {/* Suggestions */}
        {result.suggestions.length > 0 && (
          <div>
            <div className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5">
              Suggested Input Changes
            </div>
            <div data-testid="suggestions-list" className="space-y-0">
              {result.suggestions.map((s) => (
                <SuggestionRow key={s.feature} suggestion={s} />
              ))}
            </div>
            <p className="text-[10px] text-muted-foreground mt-1.5 italic">
              All other features use their training-data averages.
            </p>
          </div>
        )}

        {/* No free features */}
        {result.n_optimized === 0 && (
          <p
            data-testid="no-features-note"
            className="text-sm text-muted-foreground italic"
          >
            No free numeric features available to optimize. Try specifying target values
            for individual features via what-if analysis.
          </p>
        )}

        {/* Fixed features */}
        {Object.keys(result.fixed_features).length > 0 && (
          <div className="text-xs text-muted-foreground">
            <span className="font-medium">Fixed features: </span>
            {Object.entries(result.fixed_features)
              .map(([k, v]) => `${k.replace(/_/g, " ")}=${v}`)
              .join(", ")}
          </div>
        )}

        {feasibilityNote}

        {/* Summary */}
        <p
          data-testid="goal-seek-summary"
          className="text-sm text-muted-foreground italic border-t border-border/40 pt-2"
        >
          {result.summary}
        </p>

        {/* Accessibility */}
        <figcaption className="sr-only">
          Goal seek result: target {result.target_column} = {fmt(result.target_value)}.
          Model achieves {fmt(result.achieved_value)}.
          {result.achieved ? " Goal achieved." : ` Gap: ${result.gap_pct}%.`}
        </figcaption>
      </CardContent>
    </Card>
  )
}
