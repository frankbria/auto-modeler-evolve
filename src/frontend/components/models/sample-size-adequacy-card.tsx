"use client"

import { Badge } from "@/components/ui/badge"
import type { SampleSizeAdequacyResult } from "@/lib/types"

interface SampleSizeAdequacyCardProps {
  result: SampleSizeAdequacyResult
}

const VERDICT_STYLES: Record<
  string,
  { border: string; bg: string; badge: string; icon: string }
> = {
  adequate: {
    border: "border-emerald-300",
    bg: "bg-emerald-50",
    badge: "bg-emerald-100 text-emerald-800 border-emerald-200",
    icon: "✅",
  },
  borderline: {
    border: "border-amber-300",
    bg: "bg-amber-50",
    badge: "bg-amber-100 text-amber-800 border-amber-200",
    icon: "⚠️",
  },
  insufficient: {
    border: "border-rose-300",
    bg: "bg-rose-50",
    badge: "bg-rose-100 text-rose-800 border-rose-200",
    icon: "🚨",
  },
}

function fmtNum(n: number): string {
  return n.toLocaleString()
}

export function SampleSizeAdequacyCard({ result }: SampleSizeAdequacyCardProps) {
  if (!result) return null

  const styles = VERDICT_STYLES[result.verdict] ?? VERDICT_STYLES.adequate
  const barColor =
    result.verdict === "adequate"
      ? "bg-emerald-400"
      : result.verdict === "borderline"
        ? "bg-amber-400"
        : "bg-rose-500"

  const coveragePct = Math.min(result.coverage_pct, 100)

  return (
    <figure
      className={`rounded-lg border ${styles.border} ${styles.bg} p-4 mt-2`}
      data-testid="sample-size-adequacy-card"
      aria-label={`Sample size adequacy: ${result.verdict_label}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span aria-hidden="true" className="text-lg">{styles.icon}</span>
        <span className="font-semibold text-sm">Sample Size Adequacy</span>
        <Badge
          variant="outline"
          className={`text-xs ${styles.badge}`}
          data-testid="verdict-badge"
        >
          {result.verdict_label}
        </Badge>
        <Badge variant="outline" className="bg-gray-100 text-gray-700 border-gray-200 text-xs">
          {result.algorithm}
        </Badge>
        <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-200 text-xs">
          {result.problem_type}
        </Badge>
      </div>

      {/* Key metrics: current / recommended / gap */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="rounded-md bg-white/70 border border-gray-200 p-2 text-center">
          <p className="text-xs text-gray-500 mb-1">Current Rows</p>
          <p
            className="font-mono text-sm font-semibold text-gray-800"
            data-testid="current-n"
          >
            {fmtNum(result.n_rows)}
          </p>
          <p className="text-xs text-gray-400">{result.n_features} features</p>
        </div>
        <div className="rounded-md bg-white/70 border border-gray-200 p-2 text-center">
          <p className="text-xs text-gray-500 mb-1">Recommended</p>
          <p
            className="font-mono text-sm font-semibold text-gray-800"
            data-testid="recommended-n"
          >
            {fmtNum(result.recommended_n)}
          </p>
          <p className="text-xs text-gray-400">
            {result.n_classes != null ? `${result.n_classes} classes` : "rows"}
          </p>
        </div>
        <div className="rounded-md bg-white/70 border border-gray-200 p-2 text-center">
          <p className="text-xs text-gray-500 mb-1">Shortfall</p>
          <p
            className={`font-mono text-sm font-semibold ${
              result.shortfall === 0 ? "text-emerald-700" : "text-rose-700"
            }`}
            data-testid="shortfall"
          >
            {result.shortfall === 0 ? "None" : `-${fmtNum(result.shortfall)}`}
          </p>
          <p className="text-xs text-gray-400">rows needed</p>
        </div>
      </div>

      {/* Coverage progress bar */}
      <div className="flex items-center gap-2 mb-3">
        <div className="flex-1 bg-gray-200 rounded-full h-2">
          <div
            className={`${barColor} h-2 rounded-full transition-all`}
            style={{ width: `${coveragePct}%` }}
            role="progressbar"
            aria-valuenow={coveragePct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Coverage: ${coveragePct.toFixed(0)}% of recommended sample size`}
          />
        </div>
        <span className="text-xs text-gray-600 whitespace-nowrap">
          {coveragePct.toFixed(0)}% of target
        </span>
      </div>

      {/* Feature/row ratio */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-gray-500">Feature/row ratio:</span>
        <Badge
          variant="outline"
          className={`text-xs ${
            result.ratio > 0.1
              ? "bg-amber-100 text-amber-800 border-amber-200"
              : "bg-gray-100 text-gray-700 border-gray-200"
          }`}
          data-testid="ratio-badge"
        >
          {result.ratio.toFixed(3)} ({result.n_features} / {fmtNum(result.n_rows)})
        </Badge>
        {result.ratio > 0.1 && (
          <span className="text-xs text-amber-700">High — consider dropping features</span>
        )}
      </div>

      {/* CV stability badge */}
      {result.cv_stable !== null && (
        <div className="flex items-center gap-2 mb-3" data-testid="cv-stability-row">
          <span className="text-xs text-gray-500">CV stability:</span>
          <Badge
            variant="outline"
            className={`text-xs ${
              result.cv_stable
                ? "bg-emerald-100 text-emerald-800 border-emerald-200"
                : "bg-rose-100 text-rose-800 border-rose-200"
            }`}
            data-testid="cv-stability-badge"
          >
            {result.cv_stable ? "Stable" : "Unstable"}{" "}
            {result.cv_std != null ? `(std=${result.cv_std.toFixed(3)})` : ""}
          </Badge>
        </div>
      )}

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
        data-testid="sample-size-summary"
      >
        {result.summary}
      </p>

      <figcaption className="sr-only">
        Sample size adequacy for {result.target_col} using {result.algorithm}:{" "}
        {result.verdict_label}. Current: {fmtNum(result.n_rows)} rows,
        recommended: {fmtNum(result.recommended_n)} rows,
        shortfall: {result.shortfall === 0 ? "none" : fmtNum(result.shortfall)}.
      </figcaption>
    </figure>
  )
}
