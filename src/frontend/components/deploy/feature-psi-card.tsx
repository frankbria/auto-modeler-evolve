"use client"

import { FeaturePsiResult, FeaturePsiEntry } from "@/lib/types"

interface Props {
  result: FeaturePsiResult
}

function severityColors(severity: string) {
  if (severity === "critical")
    return { border: "border-rose-400/40 bg-rose-50/30", badge: "bg-rose-100 text-rose-700", bar: "bg-rose-400" }
  if (severity === "watch")
    return { border: "border-amber-400/40 bg-amber-50/30", badge: "bg-amber-100 text-amber-700", bar: "bg-amber-400" }
  return { border: "border-emerald-400/40 bg-emerald-50/30", badge: "bg-emerald-100 text-emerald-700", bar: "bg-emerald-400" }
}

function verdictBorderClass(verdict: string): string {
  if (verdict === "critical") return "border-rose-400/40 bg-rose-50/20"
  if (verdict === "watch") return "border-amber-400/40 bg-amber-50/20"
  if (verdict === "stable") return "border-emerald-400/40 bg-emerald-50/20"
  return "border-gray-300/40 bg-gray-50/20"
}

function PsiBar({ psi, maxPsi }: { psi: number; maxPsi: number }) {
  const width = maxPsi > 0 ? Math.round((psi / Math.max(maxPsi, 0.20)) * 100) : 0
  const clampedWidth = Math.min(width, 100)
  const barColor =
    psi >= 0.20 ? "bg-rose-400" : psi >= 0.10 ? "bg-amber-400" : "bg-emerald-400"
  return (
    <div className="flex items-center gap-2 flex-1 min-w-0">
      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${barColor}`}
          style={{ width: `${clampedWidth}%` }}
        />
      </div>
      <span className="text-xs font-mono text-gray-600 w-12 text-right">
        {psi.toFixed(3)}
      </span>
    </div>
  )
}

function FeatureRow({ entry, maxPsi }: { entry: FeaturePsiEntry; maxPsi: number }) {
  const { badge } = severityColors(entry.severity)
  return (
    <div className="flex items-center gap-3 py-2 border-b border-gray-100 last:border-0">
      <div className="w-32 min-w-0 flex-shrink-0">
        <span
          className="text-xs font-medium text-gray-800 truncate block"
          title={entry.feature}
        >
          {entry.feature}
        </span>
        <span className="text-[10px] text-gray-400 capitalize">{entry.feature_type}</span>
      </div>
      <PsiBar psi={entry.psi} maxPsi={maxPsi} />
      <span
        className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full whitespace-nowrap ${badge}`}
        aria-label={`PSI severity: ${entry.psi_label}`}
      >
        {entry.psi_label}
      </span>
      {entry.feature_type === "categorical" && entry.n_new_categories !== null && entry.n_new_categories > 0 && (
        <span className="text-[10px] text-rose-600 whitespace-nowrap" title="New categories in production not seen during training">
          +{entry.n_new_categories} new
        </span>
      )}
    </div>
  )
}

export function FeaturePsiCard({ result }: Props) {
  const borderClass = verdictBorderClass(result.verdict)
  const maxPsi = result.features.length > 0 ? result.features[0].psi : 0.20

  if (result.verdict === "no_data") {
    return (
      <div className={`rounded-xl border p-4 text-sm ${borderClass}`}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-base" aria-hidden="true">📊</span>
          <span className="font-semibold text-gray-700">Feature PSI Analysis</span>
        </div>
        <p className="text-gray-500 text-xs">{result.summary}</p>
        <figcaption className="sr-only">Feature PSI analysis: {result.summary}</figcaption>
      </div>
    )
  }

  return (
    <figure className={`rounded-xl border p-4 text-sm ${borderClass}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-base" aria-hidden="true">📊</span>
        <span className="font-semibold text-gray-700">Feature PSI Analysis</span>
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded-full ${
            result.verdict === "critical"
              ? "bg-rose-100 text-rose-700"
              : result.verdict === "watch"
              ? "bg-amber-100 text-amber-700"
              : "bg-emerald-100 text-emerald-700"
          }`}
          data-testid="verdict-badge"
        >
          {result.verdict_label}
        </span>
        <span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">
          {result.features_analyzed} features
        </span>
        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
          n={result.sample_count} production
        </span>
      </div>

      {/* Summary stats row */}
      <div className="grid grid-cols-3 gap-2 mb-3 text-center">
        <div className="rounded-lg bg-rose-50 p-2">
          <div className="text-lg font-bold text-rose-700" data-testid="critical-count">
            {result.critical_count}
          </div>
          <div className="text-[10px] text-rose-600">Critical (≥0.20)</div>
        </div>
        <div className="rounded-lg bg-amber-50 p-2">
          <div className="text-lg font-bold text-amber-700" data-testid="watch-count">
            {result.watch_count}
          </div>
          <div className="text-[10px] text-amber-600">Watch (0.10–0.20)</div>
        </div>
        <div className="rounded-lg bg-emerald-50 p-2">
          <div className="text-lg font-bold text-emerald-700" data-testid="stable-count">
            {result.stable_count}
          </div>
          <div className="text-[10px] text-emerald-600">Stable (&lt;0.10)</div>
        </div>
      </div>

      {/* Feature list */}
      <div className="mb-3" data-testid="feature-list">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Features ranked by PSI
          </span>
          <span className="text-[10px] text-gray-400">(higher = more drift)</span>
        </div>
        {result.features.map((entry) => (
          <FeatureRow key={entry.feature} entry={entry} maxPsi={maxPsi} />
        ))}
      </div>

      {/* Overall PSI */}
      <div className="flex items-center gap-2 mb-3 p-2 rounded-lg bg-gray-50">
        <span className="text-xs text-gray-500">Overall mean PSI:</span>
        <span className="text-sm font-mono font-semibold text-gray-700" data-testid="overall-psi">
          {result.overall_psi.toFixed(3)}
        </span>
        <span className="text-[10px] text-gray-400 ml-auto">
          {result.training_count.toLocaleString()} training rows
        </span>
      </div>

      {/* PSI legend */}
      <div className="flex flex-wrap gap-3 text-[10px] text-gray-500 mb-3">
        <span><span className="inline-block w-2 h-2 rounded-full bg-emerald-400 mr-1" />Stable: PSI &lt; 0.10</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-amber-400 mr-1" />Watch: PSI 0.10–0.20</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-rose-400 mr-1" />Critical: PSI ≥ 0.20</span>
      </div>

      {/* Summary */}
      <p className="text-xs text-gray-600 italic" data-testid="summary">
        {result.summary}
      </p>

      <figcaption className="sr-only">
        Feature PSI drift analysis: {result.verdict_label}. {result.summary}
      </figcaption>
    </figure>
  )
}
