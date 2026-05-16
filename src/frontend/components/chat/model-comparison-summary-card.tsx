"use client"

import type { ModelComparisonRunSummary, ModelComparisonSummaryResult } from "@/lib/types"

interface ModelComparisonSummaryCardProps {
  result: ModelComparisonSummaryResult
}

function MetricBadge({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 text-xs font-medium">
      <span className="text-blue-500">{label}:</span> {value}
    </span>
  )
}

function ExplainabilityBadge({ label }: { label: string }) {
  const color =
    label.includes("Very high") || label.includes("High")
      ? "bg-emerald-100 text-emerald-800"
      : label.includes("Medium")
        ? "bg-amber-100 text-amber-800"
        : "bg-rose-100 text-rose-800"
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${color}`}>
      {label} interpretability
    </span>
  )
}

function SpeedBadge({ label }: { label: string }) {
  const color =
    label.includes("Very fast") || label.includes("Fast")
      ? "bg-sky-100 text-sky-800"
      : label.includes("Medium")
        ? "bg-slate-100 text-slate-600"
        : "bg-orange-100 text-orange-800"
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${color}`}>
      {label}
    </span>
  )
}

function RunRow({
  run,
  isWinner,
}: {
  run: ModelComparisonRunSummary
  isWinner: boolean
}) {
  const metricPct = run.primary_metric_pct.toFixed(1)
  const cvLabel =
    run.cv_mean !== null && run.cv_std !== null
      ? `${(run.cv_mean * 100).toFixed(1)}% ±${(run.cv_std * 100).toFixed(1)}`
      : "—"

  return (
    <tr className={isWinner ? "bg-blue-50" : "even:bg-slate-50"}>
      <td className="py-2 px-3 text-sm font-medium text-slate-800 whitespace-nowrap">
        {isWinner && (
          <span className="mr-1.5 text-xs text-blue-600 font-semibold" aria-label="Winner">
            ✓
          </span>
        )}
        {run.algorithm_plain}
        {run.is_selected && (
          <span className="ml-1.5 text-xs text-emerald-600">(selected)</span>
        )}
        {run.is_deployed && (
          <span className="ml-1.5 text-xs text-indigo-600">(live)</span>
        )}
      </td>
      <td className="py-2 px-3 text-sm text-slate-700 text-right font-mono">
        {metricPct}%
      </td>
      <td className="py-2 px-3 text-sm text-slate-600 text-right font-mono">
        {cvLabel}
      </td>
      <td className="py-2 px-3">
        <ExplainabilityBadge label={run.explainability_label} />
      </td>
      <td className="py-2 px-3">
        <SpeedBadge label={run.speed_label} />
      </td>
    </tr>
  )
}

export function ModelComparisonSummaryCard({ result }: ModelComparisonSummaryCardProps) {
  if (!result || result.n_runs === 0) return null

  const metricName = result.winner?.primary_metric_name ?? (result.problem_type === "regression" ? "R²" : "accuracy")

  return (
    <figure
      className="mt-3 rounded-xl border border-blue-200 bg-blue-50 p-4 shadow-sm max-w-xl"
      aria-label={`Model comparison summary — ${result.n_runs} model${result.n_runs !== 1 ? "s" : ""}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">📊</span>
        <div>
          <p className="font-semibold text-blue-900 text-sm leading-tight">
            Model Comparison
          </p>
          <p className="text-xs text-blue-600">
            {result.n_runs} trained model{result.n_runs !== 1 ? "s" : ""} · {result.problem_type}
          </p>
        </div>
        <div className="ml-auto">
          <MetricBadge label="metric" value={metricName} />
        </div>
      </div>

      {/* Narrative */}
      <p className="text-sm text-blue-900 mb-3 leading-relaxed">
        {result.narrative}
      </p>

      {/* Comparison table */}
      <div className="overflow-x-auto rounded-lg border border-blue-100 mb-3">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-blue-100 text-blue-800">
              <th className="py-1.5 px-3 text-left font-semibold">Algorithm</th>
              <th className="py-1.5 px-3 text-right font-semibold">{metricName}</th>
              <th className="py-1.5 px-3 text-right font-semibold">CV (5-fold)</th>
              <th className="py-1.5 px-3 text-left font-semibold">Interpretability</th>
              <th className="py-1.5 px-3 text-left font-semibold">Speed</th>
            </tr>
          </thead>
          <tbody>
            {result.runs_compared.map((run, i) => (
              <RunRow
                key={run.run_id || i}
                run={run}
                isWinner={i === 0}
              />
            ))}
          </tbody>
        </table>
      </div>

      {/* Trade-offs */}
      {result.trade_offs.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-semibold text-blue-800 mb-1.5">Key trade-offs</p>
          <ul className="space-y-1">
            {result.trade_offs.map((t, i) => (
              <li key={i} className="flex gap-1.5 text-xs text-slate-700">
                <span className="text-blue-400 shrink-0 mt-0.5" aria-hidden="true">•</span>
                {t}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Summary footer */}
      <p className="text-xs text-blue-700 italic border-t border-blue-100 pt-2 mt-2">
        {result.summary}
      </p>

      <figcaption className="sr-only">
        {result.narrative} {result.trade_offs.join(" ")}
      </figcaption>
    </figure>
  )
}
