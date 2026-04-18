"use client"

import { Badge } from "@/components/ui/badge"
import type { LocalExplanationContribution, LocalExplanationResult } from "@/lib/types"

interface Props {
  result: LocalExplanationResult
}

function ContributionBar({ contrib, maxAbs }: { contrib: LocalExplanationContribution; maxAbs: number }) {
  const pct = maxAbs > 0 ? Math.abs(contrib.contribution) / maxAbs : 0
  const widthPct = Math.round(pct * 100)
  const isPositive = contrib.direction === "positive"

  return (
    <div className="flex items-center gap-2 py-1 text-xs">
      <span className="w-32 truncate text-right text-gray-700 shrink-0" title={contrib.feature}>
        {contrib.feature}
      </span>
      <span className="w-16 text-right text-gray-500 shrink-0 tabular-nums">
        {typeof contrib.value === "number" && !Number.isInteger(contrib.value)
          ? contrib.value.toFixed(2)
          : contrib.value}
      </span>
      <div className="flex-1 flex items-center">
        <div
          className={`h-4 rounded ${isPositive ? "bg-sky-400" : "bg-rose-400"}`}
          style={{ width: `${widthPct}%`, minWidth: widthPct > 0 ? "2px" : "0" }}
          aria-label={`${contrib.feature}: ${isPositive ? "+" : ""}${contrib.contribution.toFixed(4)}`}
        />
      </div>
      <span className={`w-16 text-right shrink-0 font-mono tabular-nums ${isPositive ? "text-sky-700" : "text-rose-700"}`}>
        {isPositive ? "+" : ""}{contrib.contribution.toFixed(3)}
      </span>
    </div>
  )
}

function formatPrediction(value: string | number, problemType: string): string {
  if (problemType === "classification") return String(value)
  if (typeof value === "number") return value.toFixed(4)
  return String(value)
}

export function LocalExplanationCard({ result }: Props) {
  const {
    row_index,
    algorithm,
    target_col,
    problem_type,
    actual_value,
    predicted_value,
    contributions,
    summary,
  } = result

  const maxAbs = contributions.length > 0
    ? Math.max(...contributions.map((c) => Math.abs(c.contribution)))
    : 1

  const positiveCount = contributions.filter((c) => c.direction === "positive").length
  const negativeCount = contributions.filter((c) => c.direction === "negative").length

  const isCorrect =
    problem_type === "classification"
      ? String(actual_value) === String(predicted_value)
      : null

  return (
    <figure
      className="rounded-lg border border-violet-300 bg-violet-50 p-4 my-2"
      aria-label={`Local prediction explanation for row ${row_index}`}
    >
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span role="img" aria-label="explanation" className="text-xl">🔍</span>
        <span className="font-semibold text-gray-800">Prediction Explanation</span>
        <Badge className="bg-gray-100 text-gray-700 text-xs">Row {row_index}</Badge>
        <Badge className="bg-blue-100 text-blue-800 text-xs">{algorithm}</Badge>
        <Badge className="bg-purple-100 text-purple-800 text-xs">Target: {target_col}</Badge>
        {problem_type === "classification" && isCorrect !== null && (
          <Badge className={`text-xs ${isCorrect ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"}`}>
            {isCorrect ? "✓ Correct" : "✗ Wrong"}
          </Badge>
        )}
      </div>

      {/* Prediction vs actual */}
      <div className="flex gap-4 mb-4 text-sm">
        <div className="bg-white rounded border border-gray-200 px-3 py-2 flex-1">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Actual</div>
          <div className="font-semibold text-gray-800">{formatPrediction(actual_value, problem_type)}</div>
        </div>
        <div className="bg-white rounded border border-violet-200 px-3 py-2 flex-1">
          <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Predicted</div>
          <div className="font-semibold text-violet-700">{formatPrediction(predicted_value, problem_type)}</div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mb-2 text-xs text-gray-500">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-sky-400" aria-hidden="true" />
          <span>Pushed prediction up ({positiveCount} features)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-rose-400" aria-hidden="true" />
          <span>Pushed prediction down ({negativeCount} features)</span>
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center gap-2 text-xs text-gray-400 font-medium uppercase tracking-wide mb-1">
        <span className="w-32 text-right shrink-0">Feature</span>
        <span className="w-16 text-right shrink-0">Value</span>
        <div className="flex-1" />
        <span className="w-16 text-right shrink-0">Impact</span>
      </div>

      {/* Waterfall bars */}
      <div aria-label="Feature contribution waterfall chart">
        {contributions.map((c) => (
          <ContributionBar key={c.feature} contrib={c} maxAbs={maxAbs} />
        ))}
      </div>

      <figcaption className="text-xs text-gray-500 mt-3 border-t border-violet-200 pt-2">
        {summary}
      </figcaption>
    </figure>
  )
}
