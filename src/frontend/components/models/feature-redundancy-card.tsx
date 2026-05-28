"use client"

import { Badge } from "@/components/ui/badge"
import type { FeatureRedundancyResult, RedundantPair } from "@/lib/types"

interface FeatureRedundancyCardProps {
  result: FeatureRedundancyResult
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
  low: {
    border: "border-amber-300",
    bg: "bg-amber-50",
    badge: "bg-amber-100 text-amber-800 border-amber-200",
    icon: "⚠️",
  },
  high: {
    border: "border-rose-300",
    bg: "bg-rose-50",
    badge: "bg-rose-100 text-rose-800 border-rose-200",
    icon: "🔴",
  },
}

function CorrBar({ value }: { value: number }) {
  const pct = Math.round(Math.abs(value) * 100)
  const color = pct >= 95 ? "bg-rose-500" : pct >= 90 ? "bg-orange-400" : "bg-amber-400"
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
          aria-label={`Correlation strength: ${pct}%`}
        />
      </div>
      <span className="text-xs font-mono text-gray-700 w-10 text-right">
        {value > 0 ? "+" : ""}{value.toFixed(2)}
      </span>
    </div>
  )
}

function PairRow({ pair }: { pair: RedundantPair }) {
  return (
    <li
      className="rounded-md border border-gray-200 bg-white/70 p-3 space-y-1"
      data-testid="redundant-pair-row"
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className="font-mono text-sm font-semibold text-gray-800"
          data-testid="feature-a-name"
        >
          {pair.feature_a}
        </span>
        <span className="text-xs text-gray-400">↔</span>
        <span
          className="font-mono text-sm font-semibold text-gray-800"
          data-testid="feature-b-name"
        >
          {pair.feature_b}
        </span>
        <Badge
          variant="outline"
          className={
            pair.direction === "positive"
              ? "bg-blue-50 text-blue-700 border-blue-200 text-xs"
              : "bg-purple-50 text-purple-700 border-purple-200 text-xs"
          }
          data-testid="direction-badge"
        >
          {pair.direction === "positive" ? "↑↑ positive" : "↑↓ negative"}
        </Badge>
      </div>
      <CorrBar value={pair.correlation} />
      <div className="flex items-center gap-2 flex-wrap pt-1">
        <Badge
          variant="outline"
          className="bg-emerald-50 text-emerald-700 border-emerald-200 text-xs"
          data-testid="keep-badge"
        >
          Keep: {pair.keep}
        </Badge>
        <Badge
          variant="outline"
          className="bg-rose-50 text-rose-700 border-rose-200 text-xs"
          data-testid="drop-badge"
        >
          Drop: {pair.drop}
        </Badge>
      </div>
    </li>
  )
}

export function FeatureRedundancyCard({ result }: FeatureRedundancyCardProps) {
  if (!result) return null

  const styles = VERDICT_STYLES[result.verdict] ?? VERDICT_STYLES.none
  const dropSet = new Set(result.redundant_pairs.map((p) => p.drop))

  return (
    <figure
      className={`rounded-lg border ${styles.border} ${styles.bg} p-4 mt-2`}
      data-testid="feature-redundancy-card"
      aria-label={`Feature redundancy check: ${result.verdict_label}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span aria-hidden="true" className="text-lg">{styles.icon}</span>
        <span className="font-semibold text-sm">Feature Redundancy Check</span>
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
          {result.n_features_checked} features checked
        </Badge>
        {result.n_redundant > 0 && (
          <Badge
            variant="outline"
            className="bg-rose-100 text-rose-700 border-rose-200 text-xs"
            data-testid="redundant-count-badge"
          >
            {result.n_redundant} redundant
          </Badge>
        )}
      </div>

      {/* Pair list */}
      {result.redundant_pairs.length > 0 ? (
        <ul className="space-y-2 mb-3" data-testid="pair-list">
          {result.redundant_pairs.map((pair, i) => (
            <PairRow key={i} pair={pair} />
          ))}
        </ul>
      ) : (
        <p
          className="text-sm text-emerald-700 mb-3"
          data-testid="no-pairs-message"
        >
          No redundant feature pairs found above the {Math.round(result.threshold * 100)}% correlation threshold.
        </p>
      )}

      {/* Drop recommendation */}
      {dropSet.size > 0 && (
        <div
          className="rounded-md bg-rose-50 border border-rose-200 p-2 mb-3 text-xs text-rose-800"
          data-testid="drop-recommendation"
        >
          <span className="font-semibold">Recommended to drop: </span>
          {[...dropSet].sort().join(", ")}
          <span className="ml-1 text-rose-600">
            — removing these won&apos;t lose information already captured by the kept features.
          </span>
        </div>
      )}

      {/* Summary */}
      <p
        className="text-xs text-gray-500 italic"
        data-testid="redundancy-summary"
      >
        {result.summary}
      </p>

      <figcaption className="sr-only">
        Feature redundancy check: {result.verdict_label}. Checked{" "}
        {result.n_features_checked} numeric features, found {result.redundant_pairs.length}{" "}
        redundant pair(s).{" "}
        {dropSet.size > 0
          ? `Suggested to drop: ${[...dropSet].sort().join(", ")}.`
          : "No redundant features detected."}
      </figcaption>
    </figure>
  )
}
