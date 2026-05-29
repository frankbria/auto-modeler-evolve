"use client"

import { Badge } from "@/components/ui/badge"
import type { LeakyFeature, TargetLeakageResult } from "@/lib/types"

interface TargetLeakageCardProps {
  result: TargetLeakageResult
}

const VERDICT_STYLES: Record<
  string,
  { border: string; bg: string; badge: string; icon: string }
> = {
  none: {
    border: "border-emerald-300",
    bg: "bg-emerald-50",
    badge: "bg-emerald-100 text-emerald-800 border-emerald-200",
    icon: "✅",
  },
  warning: {
    border: "border-amber-300",
    bg: "bg-amber-50",
    badge: "bg-amber-100 text-amber-800 border-amber-200",
    icon: "⚠️",
  },
  severe: {
    border: "border-rose-300",
    bg: "bg-rose-50",
    badge: "bg-rose-100 text-rose-800 border-rose-200",
    icon: "🚨",
  },
}

function LeakBar({ value }: { value: number }) {
  const pct = Math.round(Math.abs(value) * 100)
  const color = pct >= 90 ? "bg-rose-500" : pct >= 75 ? "bg-amber-500" : "bg-sky-400"
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-200 rounded-full h-1.5">
        <div
          className={`${color} h-1.5 rounded-full`}
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`Correlation with target: ${pct}%`}
        />
      </div>
      <span className="text-xs font-mono text-gray-700 w-10 text-right">
        {Math.abs(value).toFixed(2)}
      </span>
    </div>
  )
}

function FeatureRow({ feature }: { feature: LeakyFeature }) {
  const riskClass =
    feature.risk_level === "high"
      ? "bg-rose-50 text-rose-700 border-rose-200"
      : "bg-amber-50 text-amber-700 border-amber-200"

  return (
    <li
      className="rounded-md border border-gray-200 bg-white/70 p-3 space-y-1.5"
      data-testid="leaky-feature-row"
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className="font-mono text-sm font-semibold text-gray-800"
          data-testid="leaky-feature-name"
        >
          {feature.feature}
        </span>
        <Badge
          variant="outline"
          className={`text-xs ${riskClass}`}
          data-testid="risk-badge"
        >
          {feature.risk_label}
        </Badge>
      </div>
      <LeakBar value={feature.correlation_abs} />
      <p className="text-xs text-gray-500">{feature.reason}</p>
    </li>
  )
}

export function TargetLeakageCard({ result }: TargetLeakageCardProps) {
  if (!result) return null

  const styles = VERDICT_STYLES[result.verdict] ?? VERDICT_STYLES.none
  const highRisk = result.leaky_features.filter((f) => f.risk_level === "high")
  const moderateRisk = result.leaky_features.filter((f) => f.risk_level === "moderate")

  return (
    <figure
      className={`rounded-lg border ${styles.border} ${styles.bg} p-4 mt-2`}
      data-testid="target-leakage-card"
      aria-label={`Target leakage check: ${result.verdict_label}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span aria-hidden="true" className="text-lg">{styles.icon}</span>
        <span className="font-semibold text-sm">Target Leakage Check</span>
        <Badge
          variant="outline"
          className={`text-xs ${styles.badge}`}
          data-testid="verdict-badge"
        >
          {result.verdict_label}
        </Badge>
        <Badge
          variant="outline"
          className="bg-gray-100 text-gray-700 border-gray-200 text-xs"
          data-testid="features-checked-badge"
        >
          {result.n_checked} features checked
        </Badge>
        {result.leaky_features.length > 0 && (
          <Badge
            variant="outline"
            className="bg-rose-100 text-rose-700 border-rose-200 text-xs"
            data-testid="leaky-count-badge"
          >
            {result.leaky_features.length} suspicious
          </Badge>
        )}
      </div>

      {/* Target column */}
      <div className="text-xs text-gray-500 mb-3">
        Target:{" "}
        <code className="font-mono bg-gray-100 rounded px-1">{result.target_col}</code>
      </div>

      {/* Leaky features list */}
      {result.leaky_features.length > 0 ? (
        <>
          {result.verdict === "severe" && (
            <div
              className="rounded-md bg-rose-50 border border-rose-300 p-2 mb-3 text-xs text-rose-800"
              role="alert"
              data-testid="severe-alert"
            >
              <span className="font-semibold">⚠ Likely Data Leakage: </span>
              {highRisk.length} feature(s) are suspiciously correlated with the target.
              If these are only known after the target is determined, your model accuracy
              is artificially inflated and will fail in production.
            </div>
          )}
          <ul className="space-y-2 mb-3" data-testid="leaky-feature-list">
            {result.leaky_features.map((f, i) => (
              <FeatureRow key={i} feature={f} />
            ))}
          </ul>
          <div
            className="rounded-md bg-amber-50 border border-amber-200 p-2 text-xs text-amber-800"
            data-testid="leakage-recommendation"
          >
            <span className="font-semibold">What to do: </span>
            Ask yourself — would this feature be available at prediction time, before the
            outcome is known? If not, remove it from training data and retrain.
          </div>
        </>
      ) : (
        <p
          className="text-sm text-emerald-700 mb-3"
          data-testid="no-leakage-message"
        >
          No suspicious features found — all {result.n_checked} feature(s) appear safe.
          The model&apos;s accuracy reflects genuine patterns, not data shortcuts.
        </p>
      )}

      {/* Summary */}
      <p
        className="text-xs text-gray-500 italic mt-3"
        data-testid="leakage-summary"
      >
        {result.summary}
      </p>

      <figcaption className="sr-only">
        Target leakage check for &apos;{result.target_col}&apos;: {result.verdict_label}.
        Checked {result.n_checked} features.
        {result.leaky_features.length > 0
          ? ` Found ${highRisk.length} high-risk and ${moderateRisk.length} moderate-risk feature(s).`
          : " No leakage detected."}
      </figcaption>
    </figure>
  )
}
