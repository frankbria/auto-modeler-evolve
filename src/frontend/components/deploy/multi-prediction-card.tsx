"use client"

import type { MultiPredictionResult } from "@/lib/types"

interface MultiPredictionCardProps {
  result: MultiPredictionResult
}

function formatValue(v: number | string): string {
  if (typeof v === "number") {
    if (Math.abs(v) >= 1_000_000) return (v / 1_000_000).toFixed(2) + "M"
    if (Math.abs(v) >= 1_000) return (v / 1_000).toFixed(1) + "k"
    if (Number.isInteger(v)) return v.toString()
    return v.toFixed(4).replace(/\.?0+$/, "")
  }
  return String(v)
}

function PredictionCell({
  row,
  isClassification,
}: {
  row: MultiPredictionResult["rows"][number]
  isClassification: boolean
}) {
  if (isClassification && row.probabilities) {
    const topClass = Object.entries(row.probabilities).sort(
      ([, a], [, b]) => b - a
    )[0]
    const confidence = Math.round(topClass[1] * 100)
    const color =
      confidence >= 70
        ? "text-emerald-700"
        : confidence >= 50
        ? "text-amber-700"
        : "text-rose-700"
    return (
      <span className={`font-medium ${color}`}>
        {topClass[0]}{" "}
        <span className="text-xs text-gray-500">({confidence}%)</span>
      </span>
    )
  }
  return (
    <span className="font-medium text-sky-800">
      {formatValue(row.prediction as number | string)}
    </span>
  )
}

export function MultiPredictionCard({ result }: MultiPredictionCardProps) {
  const isClassification = result.problem_type === "classification"
  const target = result.target_column ?? "output"

  // Collect all feature keys across all rows to display as columns
  const allFeatureKeys = Array.from(
    new Set(result.rows.flatMap((r) => Object.keys(r.provided_features)))
  ).slice(0, 4)

  return (
    <figure
      className="mt-3 rounded-xl border border-violet-200 bg-violet-50 p-4 shadow-sm max-w-2xl"
      aria-label={`Multi-row prediction results for ${target}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">
          📊
        </span>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-violet-900 text-sm leading-tight">
            Scenario Comparison
          </p>
          <p className="text-xs text-violet-600 capitalize">{target}</p>
        </div>
        <div className="flex gap-1 flex-shrink-0">
          <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-violet-100 text-violet-700 text-xs font-medium">
            {result.rows.length} scenarios
          </span>
          <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-violet-100 text-violet-700 text-xs font-medium capitalize">
            {isClassification ? "Classification" : "Regression"}
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-violet-100">
              <th className="text-left p-2 font-semibold text-violet-700 rounded-tl-md w-8">
                #
              </th>
              <th className="text-left p-2 font-semibold text-violet-700 capitalize">
                {target}
              </th>
              {allFeatureKeys.map((k) => (
                <th
                  key={k}
                  className="text-left p-2 font-semibold text-violet-700 truncate max-w-24"
                  title={k}
                >
                  {k.replace(/_/g, " ")}
                </th>
              ))}
              {result.rows[0]?.defaults_used_count > 0 && (
                <th className="text-left p-2 font-semibold text-violet-700">
                  Defaults
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {result.rows.map((row, i) => (
              <tr
                key={row.row_index}
                className={i % 2 === 0 ? "bg-white" : "bg-violet-50/50"}
              >
                <td className="p-2 text-violet-400 font-mono">{row.row_index}</td>
                <td className="p-2">
                  <PredictionCell row={row} isClassification={isClassification} />
                </td>
                {allFeatureKeys.map((k) => (
                  <td key={k} className="p-2 text-gray-700 truncate max-w-24" title={String(row.provided_features[k] ?? "—")}>
                    {row.provided_features[k] !== undefined
                      ? formatValue(row.provided_features[k] as number | string)
                      : <span className="text-gray-300">—</span>}
                  </td>
                ))}
                {result.rows[0]?.defaults_used_count > 0 && (
                  <td className="p-2 text-gray-400 text-xs">
                    {row.defaults_used_count > 0 ? `+${row.defaults_used_count}` : "—"}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary footer */}
      {result.summary && (
        <p className="mt-2 text-xs text-violet-600 bg-violet-100 rounded-md px-3 py-1.5">
          {result.summary}
        </p>
      )}

      {/* Defaults note */}
      {result.rows.some((r) => r.defaults_used_count > 0) && (
        <p className="mt-1.5 text-xs text-violet-400">
          <span aria-hidden="true">ℹ️ </span>
          Features not specified per scenario used training-data averages.
        </p>
      )}
    </figure>
  )
}
