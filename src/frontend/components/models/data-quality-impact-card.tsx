"use client"

import { Badge } from "@/components/ui/badge"
import type { DataQualityImpactResult } from "@/lib/types"

interface DataQualityImpactCardProps {
  result: DataQualityImpactResult
}

const VERDICT_STYLES: Record<
  string,
  { border: string; bg: string; badge: string; badgeText: string; icon: string }
> = {
  worthwhile: {
    border: "border-emerald-300",
    bg: "bg-emerald-50",
    badge: "bg-emerald-100 text-emerald-800 border-emerald-200",
    badgeText: "Worthwhile",
    icon: "✨",
  },
  marginal: {
    border: "border-amber-300",
    bg: "bg-amber-50",
    badge: "bg-amber-100 text-amber-800 border-amber-200",
    badgeText: "Marginal gain",
    icon: "📉",
  },
  no_benefit: {
    border: "border-gray-300",
    bg: "bg-gray-50",
    badge: "bg-gray-100 text-gray-700 border-gray-200",
    badgeText: "No benefit",
    icon: "➡️",
  },
  harmful: {
    border: "border-rose-300",
    bg: "bg-rose-50",
    badge: "bg-rose-100 text-rose-800 border-rose-200",
    badgeText: "Keep outliers",
    icon: "⚠️",
  },
}

export function DataQualityImpactCard({ result }: DataQualityImpactCardProps) {
  if (!result) return null

  const styles = VERDICT_STYLES[result.verdict] ?? VERDICT_STYLES.no_benefit
  const deltaStr =
    result.delta > 0
      ? `+${result.delta.toFixed(3)}`
      : result.delta < 0
        ? result.delta.toFixed(3)
        : "0.000"

  const baselinePct =
    result.metric_key === "accuracy"
      ? `${(result.baseline_score * 100).toFixed(1)}%`
      : result.baseline_score.toFixed(3)
  const cleanPct =
    result.metric_key === "accuracy"
      ? `${(result.clean_score * 100).toFixed(1)}%`
      : result.clean_score.toFixed(3)

  return (
    <figure
      className={`rounded-lg border ${styles.border} ${styles.bg} p-4 mt-2`}
      data-testid="data-quality-impact-card"
      aria-label={`Data quality impact: ${result.verdict}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span aria-hidden="true" className="text-lg">{styles.icon}</span>
        <span className="font-semibold text-sm">Data Quality Impact</span>
        <Badge variant="outline" className={`text-xs ${styles.badge}`}>
          {styles.badgeText}
        </Badge>
        <Badge variant="outline" className="bg-gray-100 text-gray-700 border-gray-200 text-xs">
          {result.algorithm}
        </Badge>
        <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-200 text-xs">
          {result.metric_label}
        </Badge>
      </div>

      {/* Score comparison */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="rounded-md bg-white/70 border border-gray-200 p-2 text-center">
          <p className="text-xs text-gray-500 mb-1">With all rows</p>
          <p className="font-mono text-sm font-semibold text-gray-800" data-testid="baseline-score">
            {baselinePct}
          </p>
          <p className="text-xs text-gray-400">{result.n_total} rows</p>
        </div>
        <div className="flex items-center justify-center">
          <div className="text-center">
            <p
              className={`font-mono text-sm font-bold ${result.improvement ? "text-emerald-700" : result.delta < -0.01 ? "text-rose-700" : "text-gray-600"}`}
              data-testid="delta-score"
            >
              {deltaStr}
            </p>
            <p className="text-xs text-gray-400">{result.metric_label} Δ</p>
          </div>
        </div>
        <div className="rounded-md bg-white/70 border border-gray-200 p-2 text-center">
          <p className="text-xs text-gray-500 mb-1">Without outliers</p>
          <p className="font-mono text-sm font-semibold text-gray-800" data-testid="clean-score">
            {cleanPct}
          </p>
          <p className="text-xs text-gray-400">{result.n_total - result.n_outliers} rows</p>
        </div>
      </div>

      {/* Outlier count */}
      <div className="flex items-center gap-2 mb-3">
        <div className="flex-1 bg-gray-200 rounded-full h-2">
          <div
            className="bg-amber-400 h-2 rounded-full transition-all"
            style={{ width: `${result.outlier_pct}%` }}
            role="progressbar"
            aria-valuenow={result.outlier_pct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`${result.outlier_pct}% of rows flagged as outliers`}
          />
        </div>
        <span className="text-xs text-gray-600 whitespace-nowrap">
          {result.n_outliers} of {result.n_total} rows flagged ({result.outlier_pct}%)
        </span>
      </div>

      {/* Recommendation */}
      <p className="text-xs text-gray-700 leading-relaxed mb-1">{result.recommendation}</p>

      {/* Summary */}
      <p className="text-xs text-gray-500 italic" data-testid="quality-impact-summary">
        {result.summary}
      </p>

      <figcaption className="sr-only">
        Data quality impact for {result.target_column}: removing {result.n_outliers} outlier
        rows changes {result.metric_label} by {deltaStr}. Verdict: {result.verdict}.
      </figcaption>
    </figure>
  )
}
