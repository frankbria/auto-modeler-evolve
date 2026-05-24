/**
 * GoalSeekCard — shows reverse-prediction results for "what inputs produce target output?"
 *
 * Given a desired prediction target, displays the suggested input values the model
 * optimizer found, how close the achieved prediction is, and a plain-English summary.
 *
 * Analysts can lock individual suggestions (pin feature values) and re-run the goal seek
 * with those features held constant, letting the optimizer search the remaining degrees
 * of freedom. Locked features are sent as fixed_features in the chat message.
 */
"use client"

import { useState, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { GoalSeekResult, GoalSeekSuggestion } from "@/lib/types"

interface GoalSeekCardProps {
  result: GoalSeekResult
  /** Called when the analyst clicks "Re-run with locked features".
   *  Receives the message to send directly to chat. */
  onActionClick?: (message: string) => void
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

function SuggestionRow({
  suggestion,
  isLocked,
  onToggleLock,
}: {
  suggestion: GoalSeekSuggestion
  isLocked: boolean
  onToggleLock: () => void
}) {
  return (
    <div
      data-testid={`suggestion-row-${suggestion.feature}`}
      className={`flex items-center justify-between gap-2 py-1.5 border-b border-border/40 last:border-0 ${
        isLocked ? "bg-amber-50/60 rounded" : ""
      }`}
    >
      <span
        className="text-sm font-medium text-foreground truncate max-w-[120px]"
        title={suggestion.feature}
      >
        {suggestion.feature.replace(/_/g, " ")}
      </span>
      <div className="flex items-center gap-1.5 shrink-0">
        <span className="text-xs text-muted-foreground">
          avg: <span className="font-mono">{suggestion.current_mean.toLocaleString()}</span>
        </span>
        <span className="text-xs text-muted-foreground">→</span>
        <span className="text-sm font-semibold font-mono text-foreground">
          {suggestion.suggested_value.toLocaleString()}
        </span>
        <DirectionBadge direction={suggestion.direction} changePct={suggestion.change_pct} />
        {/* Lock toggle */}
        <button
          onClick={onToggleLock}
          aria-pressed={isLocked}
          aria-label={isLocked ? `Unlock ${suggestion.feature}` : `Lock ${suggestion.feature} at ${suggestion.suggested_value}`}
          data-testid={`lock-toggle-${suggestion.feature}`}
          className={`ml-1 rounded p-0.5 text-base leading-none transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
            isLocked
              ? "text-amber-600 bg-amber-100 hover:bg-amber-200"
              : "text-muted-foreground hover:text-foreground hover:bg-muted"
          }`}
          title={isLocked ? "Locked — click to unlock" : "Click to lock this feature value"}
        >
          {isLocked ? "🔒" : "🔓"}
        </button>
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

export function GoalSeekCard({ result, onActionClick }: GoalSeekCardProps) {
  // Track which suggestion features are locked (pinned)
  const [lockedFeatures, setLockedFeatures] = useState<Set<string>>(new Set())

  const toggleLock = useCallback((featureName: string) => {
    setLockedFeatures((prev) => {
      const next = new Set(prev)
      if (next.has(featureName)) {
        next.delete(featureName)
      } else {
        next.add(featureName)
      }
      return next
    })
  }, [])

  const borderClass = result.achieved ? "border-emerald-500/40" : "border-amber-500/40"

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

  // Build re-run message with locked features as key=value pairs
  const handleRerun = useCallback(() => {
    if (lockedFeatures.size === 0 || !onActionClick) return

    const lockedParts = result.suggestions
      .filter((s) => lockedFeatures.has(s.feature))
      .map((s) => `${s.feature}=${s.suggested_value}`)
      .join(" ")

    const message = `goal seek for ${result.target_column} = ${fmt(result.target_value)} with ${lockedParts} locked`
    onActionClick(message)
  }, [lockedFeatures, result, onActionClick])

  const hasLockedFeatures = lockedFeatures.size > 0

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

        {/* Suggestions with lock toggles */}
        {result.suggestions.length > 0 && (
          <div>
            <div className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wide mb-1.5 flex items-center gap-1.5">
              Suggested Input Changes
              <span className="font-normal normal-case text-muted-foreground/70">
                (🔓 = lock to pin a value)
              </span>
            </div>
            <div data-testid="suggestions-list" className="space-y-0">
              {result.suggestions.map((s) => (
                <SuggestionRow
                  key={s.feature}
                  suggestion={s}
                  isLocked={lockedFeatures.has(s.feature)}
                  onToggleLock={() => toggleLock(s.feature)}
                />
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

        {/* Fixed features from previous run */}
        {Object.keys(result.fixed_features).length > 0 && (
          <div className="text-xs text-muted-foreground">
            <span className="font-medium">Previously locked: </span>
            {Object.entries(result.fixed_features)
              .map(([k, v]) => `${k.replace(/_/g, " ")}=${v}`)
              .join(", ")}
          </div>
        )}

        {/* Re-run button — shown when any feature is locked */}
        {hasLockedFeatures && onActionClick && (
          <Button
            size="sm"
            variant="outline"
            onClick={handleRerun}
            data-testid="rerun-with-locked-button"
            className="w-full border-amber-300 text-amber-800 hover:bg-amber-50"
          >
            🔒 Re-run keeping {lockedFeatures.size} feature{lockedFeatures.size !== 1 ? "s" : ""} locked
          </Button>
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
