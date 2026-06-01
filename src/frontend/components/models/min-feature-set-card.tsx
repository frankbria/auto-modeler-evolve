"use client"

import { Badge } from "@/components/ui/badge"
import type { MinFeatureSetResult } from "@/lib/types"

interface MinFeatureSetCardProps {
  result: MinFeatureSetResult
}

export function MinFeatureSetCard({ result }: MinFeatureSetCardProps) {
  const {
    n_original,
    n_minimal,
    features_dropped,
    features_retained,
    features_dropped_list,
    baseline_score,
    minimal_score,
    score_loss,
    can_simplify,
    reduction_pct,
    metric,
    summary,
    algorithm,
    target_col,
    features,
  } = result

  const metricLabel = metric === "r2" ? "R²" : "Accuracy"
  const fmtScore = (s: number) =>
    metric === "r2" ? s.toFixed(3) : `${(s * 100).toFixed(1)}%`

  const borderColor = can_simplify
    ? "border-sky-500 bg-sky-50 dark:bg-sky-950/20"
    : "border-slate-400 bg-slate-50 dark:bg-slate-900/20"

  const maxImp = Math.max(...features.map((f) => f.importance), 0.001)

  return (
    <div
      className={`rounded-lg border-2 p-4 mt-2 ${borderColor}`}
      role="region"
      aria-label="Minimum viable feature set analysis"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span aria-hidden="true" className="text-lg">
          ✂️
        </span>
        <span className="font-semibold text-sm">
          Minimum Viable Feature Set
        </span>
        <Badge variant="secondary" className="text-xs">
          {algorithm}
        </Badge>
        <Badge variant="outline" className="text-xs">
          target: {target_col}
        </Badge>
        {can_simplify ? (
          <Badge className="text-xs bg-sky-600 text-white">
            {features_dropped} droppable
          </Badge>
        ) : (
          <Badge className="text-xs bg-slate-500 text-white">
            all features needed
          </Badge>
        )}
      </div>

      {/* Score comparison */}
      <div className="grid grid-cols-3 gap-2 mb-3 text-center">
        <div className="rounded bg-white dark:bg-slate-800 p-2 border">
          <div className="text-xs text-muted-foreground">{metricLabel} (full)</div>
          <div className="font-mono font-semibold text-sm">
            {fmtScore(baseline_score)}
          </div>
        </div>
        <div
          className={`rounded p-2 border ${
            can_simplify
              ? "bg-sky-100 dark:bg-sky-900/40 border-sky-400"
              : "bg-white dark:bg-slate-800"
          }`}
        >
          <div className="text-xs text-muted-foreground">{metricLabel} (minimal)</div>
          <div className="font-mono font-semibold text-sm">
            {fmtScore(minimal_score)}
          </div>
        </div>
        <div className="rounded bg-white dark:bg-slate-800 p-2 border">
          <div className="text-xs text-muted-foreground">Score loss</div>
          <div
            className={`font-mono font-semibold text-sm ${
              score_loss <= 0.01
                ? "text-emerald-600"
                : score_loss <= 0.05
                  ? "text-amber-600"
                  : "text-rose-600"
            }`}
          >
            {score_loss >= 0 ? "−" : "+"}
            {Math.abs(score_loss).toFixed(3)}
          </div>
        </div>
      </div>

      {/* Feature list */}
      <div className="space-y-1 mb-3">
        {features.map((f) => {
          const barWidth = Math.round((f.importance / maxImp) * 100)
          return (
            <div
              key={f.feature}
              className={`flex items-center gap-2 rounded px-2 py-1 ${
                f.kept
                  ? "bg-white dark:bg-slate-800"
                  : "bg-slate-100 dark:bg-slate-800/50 opacity-60"
              }`}
            >
              <span
                className={`text-xs font-mono w-4 text-center ${
                  f.kept ? "text-sky-600" : "text-slate-400"
                }`}
                aria-label={f.kept ? "kept" : "dropped"}
              >
                {f.kept ? "✓" : "✗"}
              </span>
              <span
                className={`text-xs flex-1 truncate ${
                  f.kept ? "font-medium" : "line-through text-muted-foreground"
                }`}
                title={f.feature}
              >
                {f.feature}
              </span>
              <div
                className="h-2 rounded-full"
                style={{
                  width: `${barWidth}%`,
                  maxWidth: "80px",
                  minWidth: "4px",
                  backgroundColor: f.kept ? "#0284c7" : "#94a3b8",
                }}
                role="img"
                aria-label={`importance ${f.importance.toFixed(4)}`}
              />
              <span className="text-xs font-mono text-muted-foreground w-16 text-right">
                {f.importance.toFixed(4)}
              </span>
            </div>
          )
        })}
      </div>

      {/* Retained / dropped lists */}
      {can_simplify && (
        <div className="grid grid-cols-2 gap-2 mb-3 text-xs">
          <div className="rounded bg-sky-50 dark:bg-sky-950/30 border border-sky-200 dark:border-sky-800 p-2">
            <div className="font-semibold text-sky-700 dark:text-sky-400 mb-1">
              Retained ({n_minimal})
            </div>
            <ul className="space-y-0.5">
              {features_retained.map((f) => (
                <li key={f} className="truncate text-foreground">
                  {f}
                </li>
              ))}
            </ul>
          </div>
          <div className="rounded bg-slate-50 dark:bg-slate-900/50 border border-slate-200 dark:border-slate-700 p-2">
            <div className="font-semibold text-slate-500 mb-1">
              Can drop ({features_dropped})
            </div>
            <ul className="space-y-0.5">
              {features_dropped_list.map((f) => (
                <li key={f} className="truncate text-muted-foreground line-through">
                  {f}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Reduction stat */}
      {can_simplify && (
        <div className="text-xs text-center text-sky-700 dark:text-sky-400 mb-2 font-medium">
          {reduction_pct.toFixed(0)}% feature reduction · {n_original} → {n_minimal} features
        </div>
      )}

      {/* Summary */}
      <p className="text-xs text-muted-foreground leading-relaxed">{summary}</p>

      <figcaption className="sr-only">
        Minimum viable feature set: {n_minimal} of {n_original} features retained with{" "}
        {fmtScore(minimal_score)} {metricLabel}.
      </figcaption>
    </div>
  )
}
