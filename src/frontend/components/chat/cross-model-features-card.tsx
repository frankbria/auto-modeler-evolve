"use client"

import type { CrossModelFeatureEntry, CrossModelFeatureResult } from "@/lib/types"

interface CrossModelFeaturesCardProps {
  result: CrossModelFeatureResult
}

function ConsistencyBadge({ consistency }: { consistency: CrossModelFeatureEntry["consistency"] }) {
  const styles: Record<string, string> = {
    high: "bg-emerald-100 text-emerald-800",
    medium: "bg-amber-100 text-amber-800",
    variable: "bg-rose-100 text-rose-800",
  }
  const labels: Record<string, string> = {
    high: "Consistent",
    medium: "Moderate",
    variable: "Variable",
  }
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${styles[consistency]}`}>
      {labels[consistency]}
    </span>
  )
}

function ImportanceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-slate-200 rounded-full h-1.5 overflow-hidden">
        <div
          className="h-full bg-violet-500 rounded-full transition-all"
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <span className="text-xs font-mono text-slate-600 w-8 text-right">{pct}%</span>
    </div>
  )
}

function FeatureRow({
  entry,
  rank,
  isConsensus,
}: {
  entry: CrossModelFeatureEntry
  rank: number
  isConsensus: boolean
}) {
  return (
    <tr className={isConsensus ? "bg-violet-50" : "even:bg-slate-50"}>
      <td className="py-2 px-3 text-sm text-slate-500 font-mono w-6 text-center">
        {rank}
      </td>
      <td className="py-2 px-3 text-sm font-medium text-slate-800 whitespace-nowrap">
        {entry.feature}
        {isConsensus && (
          <span className="ml-1.5 text-xs text-violet-600 font-semibold" aria-label="All models agree">
            ★
          </span>
        )}
      </td>
      <td className="py-2 px-3 min-w-[120px]">
        <ImportanceBar value={entry.mean_importance} />
      </td>
      <td className="py-2 px-3 text-xs text-slate-600 text-center">
        {entry.agreement_count}/{entry.n_models_with_data}
      </td>
      <td className="py-2 px-3">
        <ConsistencyBadge consistency={entry.consistency} />
      </td>
    </tr>
  )
}

export function CrossModelFeaturesCard({ result }: CrossModelFeaturesCardProps) {
  if (!result || result.n_models === 0) return null

  const consensusSet = new Set(result.consensus_features)

  return (
    <figure
      className="mt-3 rounded-xl border border-violet-200 bg-violet-50 p-4 shadow-sm max-w-xl"
      aria-label={`Cross-model feature importance — ${result.n_models} model${result.n_models !== 1 ? "s" : ""}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">🔍</span>
        <div>
          <p className="font-semibold text-violet-900 text-sm leading-tight">
            Feature Importance Across Models
          </p>
          <p className="text-xs text-violet-600">
            {result.n_models} trained model{result.n_models !== 1 ? "s" : ""}
            {result.consensus_features.length > 0 &&
              ` · ${result.consensus_features.length} consensus feature${result.consensus_features.length !== 1 ? "s" : ""}`}
          </p>
        </div>
      </div>

      {/* Summary */}
      <p className="text-sm text-violet-900 mb-3 leading-relaxed">
        {result.summary}
      </p>

      {/* Consensus callout */}
      {result.consensus_features.length > 0 && (
        <div className="rounded-lg bg-violet-100 border border-violet-200 px-3 py-2 mb-3">
          <p className="text-xs font-semibold text-violet-800 mb-1">
            All models agree on these top predictors:
          </p>
          <div className="flex flex-wrap gap-1.5">
            {result.consensus_features.map((f) => (
              <span
                key={f}
                className="px-2 py-0.5 rounded-full bg-violet-200 text-violet-900 text-xs font-medium"
              >
                {f}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Feature table */}
      <div className="overflow-x-auto rounded-lg border border-violet-100 mb-2">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-violet-100 text-violet-800">
              <th className="py-1.5 px-3 text-center font-semibold">#</th>
              <th className="py-1.5 px-3 text-left font-semibold">Feature</th>
              <th className="py-1.5 px-3 text-left font-semibold">Mean importance</th>
              <th className="py-1.5 px-3 text-center font-semibold">Top-5 in</th>
              <th className="py-1.5 px-3 text-left font-semibold">Consistency</th>
            </tr>
          </thead>
          <tbody>
            {result.features.map((entry, i) => (
              <FeatureRow
                key={entry.feature}
                entry={entry}
                rank={i + 1}
                isConsensus={consensusSet.has(entry.feature)}
              />
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-violet-600 italic">
        ★ = appears in top 5 across all models · "Top-5 in" = models where this feature ranks ≤ 5
      </p>

      <figcaption className="sr-only">
        {result.summary}
      </figcaption>
    </figure>
  )
}
