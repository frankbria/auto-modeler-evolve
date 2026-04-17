"use client"

import { Badge } from "@/components/ui/badge"
import type { CvScoreDistributionResult } from "@/lib/types"

interface CvScoreDistributionCardProps {
  result: CvScoreDistributionResult
}

function FoldBar({ score, max }: { score: number; max: number }) {
  const pct = max > 0 ? Math.max(0, (score / max) * 100) : 0
  const color =
    score >= 0.8
      ? "bg-emerald-500"
      : score >= 0.6
        ? "bg-sky-500"
        : score >= 0.4
          ? "bg-amber-500"
          : "bg-rose-500"
  return (
    <div className="flex items-center gap-2 w-full" data-testid="fold-bar">
      <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color} transition-all`}
          style={{ width: `${pct}%` }}
          aria-hidden="true"
        />
      </div>
      <span className="text-xs tabular-nums text-foreground w-12 text-right">
        {score.toFixed(3)}
      </span>
    </div>
  )
}

export function CvScoreDistributionCard({ result }: CvScoreDistributionCardProps) {
  const {
    algorithm_plain,
    problem_type,
    metric_plain,
    scores,
    mean,
    std,
    ci_low,
    ci_high,
    n_splits,
    consistency,
    consistency_pct,
    summary,
  } = result

  const maxScore = scores.length > 0 ? Math.max(...scores) : 1

  const consistencyConfig = {
    stable: {
      border: "border-emerald-200",
      badge: "bg-emerald-100 text-emerald-800",
      label: "Stable",
    },
    moderate: {
      border: "border-amber-200",
      badge: "bg-amber-100 text-amber-800",
      label: "Moderate Variance",
    },
    variable: {
      border: "border-rose-200",
      badge: "bg-rose-100 text-rose-800",
      label: "High Variance",
    },
  }[consistency] ?? {
    border: "border-slate-200",
    badge: "bg-slate-100 text-slate-800",
    label: consistency,
  }

  return (
    <figure
      className={`border ${consistencyConfig.border} rounded-lg p-4 bg-white my-2 space-y-3`}
      aria-label="Cross-validation score distribution"
      data-testid="cv-score-distribution-card"
    >
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-lg">📊</span>
          <h3 className="text-sm font-semibold text-foreground">
            Cross-Validation Scores
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {problem_type && (
            <Badge className="text-xs bg-muted text-muted-foreground border-0 capitalize">
              {problem_type}
            </Badge>
          )}
          <Badge className="text-xs bg-muted text-muted-foreground border-0">
            {algorithm_plain}
          </Badge>
          <Badge
            className={`text-xs border-0 ${consistencyConfig.badge}`}
            data-testid="consistency-badge"
          >
            {consistencyConfig.label}
          </Badge>
        </div>
      </div>

      {/* Summary stats row */}
      <div
        className="grid grid-cols-3 gap-2 bg-muted/30 rounded-md px-3 py-2"
        data-testid="cv-stats-row"
      >
        <div className="text-center">
          <p className="text-xs text-muted-foreground">Mean {metric_plain}</p>
          <p className="text-sm font-semibold text-foreground tabular-nums">
            {mean.toFixed(3)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-muted-foreground">Std Dev</p>
          <p className="text-sm font-semibold text-foreground tabular-nums">
            ±{std.toFixed(3)}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-muted-foreground">Coeff of Var</p>
          <p className="text-sm font-semibold text-foreground tabular-nums">
            {consistency_pct.toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Per-fold bars */}
      {scores.length > 0 && (
        <div className="space-y-1.5" data-testid="fold-bars-container">
          <p className="text-xs font-medium text-foreground">
            Fold scores ({n_splits}-fold):
          </p>
          {scores.map((s, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-10">Fold {i + 1}</span>
              <FoldBar score={s} max={maxScore} />
            </div>
          ))}
        </div>
      )}

      {/* 95% CI */}
      <p className="text-xs text-muted-foreground" data-testid="cv-ci">
        95% CI: {ci_low.toFixed(3)} – {ci_high.toFixed(3)}
      </p>

      {/* Plain-English summary */}
      <p
        className="text-xs text-muted-foreground leading-relaxed"
        data-testid="cv-summary"
      >
        {summary}
      </p>

      <figcaption className="text-xs text-muted-foreground border-t pt-2 mt-1">
        {consistency === "stable"
          ? "Low variance across folds — predictions are consistent regardless of which data the model trains on."
          : consistency === "moderate"
            ? "Moderate variance — the model is reasonably consistent, but performance may shift slightly with different data."
            : "High variance across folds — the model is sensitive to which data it trains on. Consider more training data or a simpler model."}
      </figcaption>
    </figure>
  )
}
