"use client"

import { Badge } from "@/components/ui/badge"
import type { OverfittingAnalysisResult } from "@/lib/types"

interface OverfittingAnalysisCardProps {
  result: OverfittingAnalysisResult
}

const VERDICT_STYLES: Record<
  string,
  { border: string; bg: string; badge: string; badgeText: string; icon: string }
> = {
  well_fit: {
    border: "border-emerald-300",
    bg: "bg-emerald-50",
    badge: "bg-emerald-100 text-emerald-800 border-emerald-200",
    badgeText: "Well Fitted",
    icon: "✅",
  },
  mild_overfit: {
    border: "border-amber-300",
    bg: "bg-amber-50",
    badge: "bg-amber-100 text-amber-800 border-amber-200",
    badgeText: "Mild Overfitting",
    icon: "⚠️",
  },
  overfit: {
    border: "border-rose-300",
    bg: "bg-rose-50",
    badge: "bg-rose-100 text-rose-800 border-rose-200",
    badgeText: "Overfitting",
    icon: "🔥",
  },
  underfit: {
    border: "border-slate-300",
    bg: "bg-slate-50",
    badge: "bg-slate-100 text-slate-700 border-slate-200",
    badgeText: "Underfitting",
    icon: "📉",
  },
}

function fmtScore(score: number, metricKey: string): string {
  return metricKey === "accuracy"
    ? `${(score * 100).toFixed(1)}%`
    : score.toFixed(3)
}

export function OverfittingAnalysisCard({ result }: OverfittingAnalysisCardProps) {
  if (!result) return null

  const styles = VERDICT_STYLES[result.verdict] ?? VERDICT_STYLES.well_fit
  const gapSign = result.gap > 0 ? "+" : ""
  const gapStr = `${gapSign}${result.gap.toFixed(3)}`

  const trainDisplay = fmtScore(result.train_score, result.metric_key)
  const cvDisplay = fmtScore(result.cv_mean, result.metric_key)

  const gapBarPct = Math.min(Math.abs(result.gap_pct), 100)
  const gapBarColor =
    result.verdict === "well_fit"
      ? "bg-emerald-400"
      : result.verdict === "mild_overfit"
        ? "bg-amber-400"
        : result.verdict === "overfit"
          ? "bg-rose-500"
          : "bg-slate-400"

  return (
    <figure
      className={`rounded-lg border ${styles.border} ${styles.bg} p-4 mt-2`}
      data-testid="overfitting-analysis-card"
      aria-label={`Overfitting analysis: ${result.verdict_label}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span aria-hidden="true" className="text-lg">{styles.icon}</span>
        <span className="font-semibold text-sm">Model Fit Analysis</span>
        <Badge
          variant="outline"
          className={`text-xs ${styles.badge}`}
          data-testid="verdict-badge"
        >
          {styles.badgeText}
        </Badge>
        <Badge variant="outline" className="bg-gray-100 text-gray-700 border-gray-200 text-xs">
          {result.algorithm_plain}
        </Badge>
        <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-200 text-xs">
          {result.metric_label}
        </Badge>
      </div>

      {/* Train vs CV comparison */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="rounded-md bg-white/70 border border-gray-200 p-2 text-center">
          <p className="text-xs text-gray-500 mb-1">Train {result.metric_label}</p>
          <p
            className="font-mono text-sm font-semibold text-gray-800"
            data-testid="train-score"
          >
            {trainDisplay}
          </p>
          <p className="text-xs text-gray-400">on training data</p>
        </div>
        <div className="flex items-center justify-center">
          <div className="text-center">
            <p
              className={`font-mono text-sm font-bold ${
                result.verdict === "well_fit"
                  ? "text-emerald-700"
                  : result.verdict === "underfit"
                    ? "text-slate-600"
                    : "text-rose-700"
              }`}
              data-testid="gap-value"
            >
              {gapStr}
            </p>
            <p className="text-xs text-gray-400">gap</p>
          </div>
        </div>
        <div className="rounded-md bg-white/70 border border-gray-200 p-2 text-center">
          <p className="text-xs text-gray-500 mb-1">CV {result.metric_label}</p>
          <p
            className="font-mono text-sm font-semibold text-gray-800"
            data-testid="cv-score"
          >
            {cvDisplay}
          </p>
          <p className="text-xs text-gray-400">
            ±{result.cv_std.toFixed(3)} ({result.cv_scores.length}-fold)
          </p>
        </div>
      </div>

      {/* Gap bar */}
      <div className="flex items-center gap-2 mb-3">
        <div className="flex-1 bg-gray-200 rounded-full h-2">
          <div
            className={`${gapBarColor} h-2 rounded-full transition-all`}
            style={{ width: `${gapBarPct}%` }}
            role="progressbar"
            aria-valuenow={gapBarPct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Generalization gap: ${result.gap_pct.toFixed(1)}%`}
          />
        </div>
        <span className="text-xs text-gray-600 whitespace-nowrap">
          {result.gap_pct.toFixed(1)}% gap
        </span>
      </div>

      {/* Recommendations */}
      {result.recommendations.length > 0 && (
        <ul className="mb-3 space-y-1" data-testid="recommendations-list">
          {result.recommendations.map((rec, i) => (
            <li key={i} className="text-xs text-gray-700 flex gap-1">
              <span aria-hidden="true">→</span>
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      )}

      {/* Summary */}
      <p
        className="text-xs text-gray-500 italic"
        data-testid="overfitting-summary"
      >
        {result.summary}
      </p>

      <figcaption className="sr-only">
        Model fit analysis for {result.target_column} using {result.algorithm_plain}:{" "}
        {result.verdict_label}. Train {result.metric_label} {trainDisplay}, CV{" "}
        {result.metric_label} {cvDisplay} (gap {gapStr}).
      </figcaption>
    </figure>
  )
}
