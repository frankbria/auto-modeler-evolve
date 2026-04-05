"use client"

import { Badge } from "@/components/ui/badge"
import type { PredictionOpportunitiesResult, PredictionOpportunity } from "@/lib/types"

interface PredictionOpportunitiesCardProps {
  result: PredictionOpportunitiesResult
  onSelectTarget?: (col: string) => void
}

const BV_LABEL: Record<string, string> = {
  high: "High value",
  medium: "Medium value",
  low: "Lower value",
}

const BV_COLOR: Record<string, string> = {
  high: "bg-emerald-100 text-emerald-800",
  medium: "bg-blue-100 text-blue-800",
  low: "bg-slate-100 text-slate-700",
}

const PROBLEM_COLOR: Record<string, string> = {
  regression: "bg-violet-100 text-violet-800",
  classification: "bg-amber-100 text-amber-800",
}

function FeasibilityBar({ score }: { score: number }) {
  const width = `${score}%`
  const color =
    score >= 80
      ? "bg-emerald-500"
      : score >= 60
        ? "bg-blue-500"
        : "bg-amber-500"
  return (
    <div className="flex items-center gap-2 mt-1">
      <div className="flex-1 bg-purple-100 rounded-full h-1.5">
        <div
          className={`h-1.5 rounded-full ${color}`}
          style={{ width }}
        />
      </div>
      <span className="text-xs text-purple-700 font-medium w-8 text-right">
        {score}
      </span>
    </div>
  )
}

function OpportunityRow({
  opp,
  rank,
  onSelect,
}: {
  opp: PredictionOpportunity
  rank: number
  onSelect?: (col: string) => void
}) {
  return (
    <div className="py-3 border-b border-purple-100 last:border-0">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="font-semibold text-sm text-purple-900">
              #{rank} {opp.target_col}
            </span>
            <Badge
              className={`text-xs font-normal ${PROBLEM_COLOR[opp.problem_type] ?? "bg-slate-100 text-slate-700"}`}
            >
              {opp.problem_type}
            </Badge>
            <Badge
              className={`text-xs font-normal ${BV_COLOR[opp.business_value] ?? "bg-slate-100 text-slate-700"}`}
            >
              {BV_LABEL[opp.business_value]}
            </Badge>
          </div>
          <p className="text-xs text-purple-700 leading-snug mb-1">{opp.reason}</p>
          <p className="text-xs text-purple-500 italic leading-snug">
            &ldquo;{opp.example_question}&rdquo;
          </p>
          <FeasibilityBar score={opp.feasibility_score} />
        </div>
        {onSelect && (
          <button
            onClick={() => onSelect(opp.target_col)}
            className="flex-shrink-0 text-xs bg-purple-600 hover:bg-purple-700 text-white px-2 py-1 rounded transition-colors"
          >
            Set target
          </button>
        )}
      </div>
    </div>
  )
}

/**
 * PredictionOpportunitiesCard — displayed in chat when the user asks
 * "what can I predict?" or similar phrases. Shows ranked prediction
 * target suggestions with feasibility scores and business value ratings.
 */
export function PredictionOpportunitiesCard({
  result,
  onSelectTarget,
}: PredictionOpportunitiesCardProps) {
  if (!result.opportunities || result.opportunities.length === 0) {
    return null
  }

  const highCount = result.opportunities.filter(
    (o) => o.business_value === "high"
  ).length

  return (
    <figure
      className="rounded-lg border-2 border-purple-300 bg-purple-50 p-4 mt-2"
      aria-label={`Prediction opportunities: ${result.total} targets found`}
    >
      <div className="flex items-center gap-2 mb-3">
        <span
          className="text-xl"
          aria-hidden="true"
        >
          🎯
        </span>
        <h3 className="font-semibold text-purple-900 text-sm">
          Prediction Opportunities
        </h3>
        <Badge className="bg-purple-200 text-purple-800 text-xs font-normal">
          {result.total} target{result.total !== 1 ? "s" : ""} found
        </Badge>
        {highCount > 0 && (
          <Badge className="bg-emerald-100 text-emerald-800 text-xs font-normal">
            {highCount} high value
          </Badge>
        )}
      </div>

      <p className="text-xs text-purple-700 mb-3">
        Based on your dataset columns, here are the best prediction targets — ranked by
        how feasible they are to model accurately.
      </p>

      <div>
        {result.opportunities.map((opp, i) => (
          <OpportunityRow
            key={opp.target_col}
            opp={opp}
            rank={i + 1}
            onSelect={onSelectTarget}
          />
        ))}
      </div>

      <p className="text-xs text-purple-500 mt-3">
        Feasibility score = data completeness + column variety + available predictors.
        Higher is better.
      </p>
    </figure>
  )
}
