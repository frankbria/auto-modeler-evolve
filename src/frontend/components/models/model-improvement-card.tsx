"use client"

import { Badge } from "@/components/ui/badge"
import type {
  ModelImprovementResult,
  ImprovementSuggestion,
  ImprovementDifficulty,
  ImprovementImpact,
} from "@/lib/types"

interface ModelImprovementCardProps {
  result: ModelImprovementResult
}

const DIFFICULTY_LABEL: Record<ImprovementDifficulty, string> = {
  easy: "Easy",
  medium: "Medium",
  hard: "Hard",
}

const DIFFICULTY_COLOR: Record<ImprovementDifficulty, string> = {
  easy: "bg-emerald-100 text-emerald-800",
  medium: "bg-amber-100 text-amber-800",
  hard: "bg-rose-100 text-rose-800",
}

const IMPACT_LABEL: Record<ImprovementImpact, string> = {
  low: "Low impact",
  moderate: "Moderate impact",
  high: "High impact",
}

const IMPACT_COLOR: Record<ImprovementImpact, string> = {
  low: "bg-slate-100 text-slate-700",
  moderate: "bg-blue-100 text-blue-800",
  high: "bg-violet-100 text-violet-800",
}

const CATEGORY_ICON: Record<string, string> = {
  features: "🎯",
  algorithm: "🤖",
  data: "📊",
  reliability: "⚖️",
}

function SuggestionRow({ suggestion }: { suggestion: ImprovementSuggestion }) {
  const icon = CATEGORY_ICON[suggestion.category] ?? "💡"
  return (
    <div className="flex gap-3 py-2 border-b border-violet-100 last:border-0">
      <span
        className="text-lg flex-shrink-0 mt-0.5"
        aria-hidden="true"
      >
        {icon}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap mb-1">
          <span className="font-medium text-sm text-violet-900">
            #{suggestion.rank} {suggestion.title}
          </span>
          <Badge
            className={`text-xs font-normal ${IMPACT_COLOR[suggestion.expected_impact]}`}
          >
            {IMPACT_LABEL[suggestion.expected_impact]}
          </Badge>
          <Badge
            className={`text-xs font-normal ${DIFFICULTY_COLOR[suggestion.difficulty]}`}
          >
            {DIFFICULTY_LABEL[suggestion.difficulty]}
          </Badge>
        </div>
        <p className="text-xs text-violet-700 leading-snug">{suggestion.explanation}</p>
      </div>
    </div>
  )
}

/**
 * ModelImprovementCard — displayed in chat when the user asks
 * "how do I improve my model?" or similar phrases.
 *
 * Shows ranked improvement suggestions ordered by expected impact,
 * with difficulty indicators so the analyst can pick the fastest win.
 */
export function ModelImprovementCard({ result }: ModelImprovementCardProps) {
  const metricPct =
    result.problem_type === "regression"
      ? `${Math.round(result.primary_metric * 100)}% ${result.primary_metric_name}`
      : `${Math.round(result.primary_metric * 100)}% ${result.primary_metric_name}`

  return (
    <figure
      className="mt-2 rounded-lg border border-violet-300 bg-violet-50 p-3"
      aria-label="Model improvement suggestions"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-1">
        <span className="text-base font-semibold text-violet-900">
          💡 Improvement Suggestions
        </span>
        <Badge className="bg-violet-100 text-violet-800 text-xs font-normal">
          {result.n_suggestions} suggestion{result.n_suggestions !== 1 ? "s" : ""}
        </Badge>
        <Badge className="bg-slate-100 text-slate-700 text-xs font-normal">
          {metricPct}
        </Badge>
      </div>

      {/* Summary */}
      <figcaption className="text-xs text-violet-700 mb-2 leading-snug">
        {result.summary}
      </figcaption>

      {/* Suggestions list */}
      {result.suggestions.length === 0 ? (
        <p className="text-xs text-violet-600 italic">
          No obvious improvements detected — your model looks well-optimised.
        </p>
      ) : (
        <div>
          {result.suggestions.map((s) => (
            <SuggestionRow key={s.rank} suggestion={s} />
          ))}
        </div>
      )}

      {/* Legend */}
      <div className="mt-2 flex gap-3 text-xs text-violet-500">
        <span>
          Difficulty:{" "}
          <span className="text-emerald-700 font-medium">easy</span> ·{" "}
          <span className="text-amber-700 font-medium">medium</span> ·{" "}
          <span className="text-rose-700 font-medium">hard</span>
        </span>
        <span>
          Impact:{" "}
          <span className="text-violet-700 font-medium">high</span> ranked first
        </span>
      </div>
    </figure>
  )
}
