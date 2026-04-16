"use client"

import { Badge } from "@/components/ui/badge"
import type { EnsembleOption, EnsembleRecommendationResult } from "@/lib/types"

interface EnsembleRecommendationCardProps {
  result: EnsembleRecommendationResult
}

function ComplexityBadge({ complexity }: { complexity: EnsembleOption["complexity"] }) {
  if (complexity === "easy") {
    return (
      <Badge className="text-xs bg-emerald-100 text-emerald-800 border-0">
        Easy
      </Badge>
    )
  }
  if (complexity === "medium") {
    return (
      <Badge className="text-xs bg-amber-100 text-amber-800 border-0">
        Medium
      </Badge>
    )
  }
  return (
    <Badge className="text-xs bg-rose-100 text-rose-800 border-0">
      Advanced
    </Badge>
  )
}

function OptionRow({ option }: { option: EnsembleOption }) {
  return (
    <div
      className={`rounded-md border p-3 space-y-1.5 ${
        option.is_recommended
          ? "border-violet-200 bg-violet-50"
          : "border-border bg-muted/30"
      }`}
      data-testid={`ensemble-option-${option.ensemble_type}`}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span
          className={`text-sm font-semibold ${
            option.is_recommended ? "text-violet-900" : "text-foreground"
          }`}
        >
          {option.name}
        </span>
        {option.is_recommended && (
          <Badge className="text-xs bg-violet-200 text-violet-800 border-0">
            Recommended
          </Badge>
        )}
        <ComplexityBadge complexity={option.complexity} />
        <Badge className="text-xs bg-muted text-muted-foreground border-0 capitalize">
          {option.ensemble_type}
        </Badge>
      </div>

      <p className="text-xs text-muted-foreground leading-relaxed">
        {option.plain_english}
      </p>

      <p className="text-xs text-muted-foreground">
        <strong className="text-foreground">Best for:</strong>{" "}
        {option.best_for}
      </p>

      <p className="text-xs text-muted-foreground italic">
        💡 Say &ldquo;train a {option.name.toLowerCase()}&rdquo; to start training
      </p>
    </div>
  )
}

export function EnsembleRecommendationCard({
  result,
}: EnsembleRecommendationCardProps) {
  const {
    problem_type,
    best_current_algorithm,
    best_current_score,
    metric_name,
    options,
    recommended_name,
    summary,
  } = result

  const metricDisplay = metric_name.toUpperCase()
  const scoreDisplay =
    best_current_score !== null && best_current_score !== undefined
      ? `${metricDisplay} ${best_current_score.toFixed(3)}`
      : null

  const plainAlgo = (algo: string | null) => {
    if (!algo) return null
    return algo
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase())
  }

  return (
    <figure
      className="border border-violet-200 rounded-lg p-4 bg-white my-2 space-y-3"
      aria-label="Ensemble model recommendation"
    >
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-lg">🧩</span>
          <h3 className="text-sm font-semibold text-violet-900">
            Ensemble Models
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge className="text-xs bg-violet-100 text-violet-800 border-0 capitalize">
            {problem_type}
          </Badge>
          {scoreDisplay && (
            <Badge className="text-xs bg-muted text-muted-foreground border-0">
              Current best: {scoreDisplay}
            </Badge>
          )}
          {best_current_algorithm && (
            <Badge className="text-xs bg-muted text-muted-foreground border-0">
              {plainAlgo(best_current_algorithm)}
            </Badge>
          )}
        </div>
      </div>

      {/* Explanation */}
      <div className="bg-violet-50 border border-violet-100 rounded-md px-3 py-2">
        <p className="text-xs text-violet-800 leading-relaxed">
          <strong>What is an ensemble model?</strong> It combines multiple models to
          reduce errors that any single model makes — like getting a second opinion
          from several experts and averaging their answers. Ensembles typically
          improve accuracy by 1–5% over the best single model.
        </p>
      </div>

      {/* Recommendation summary */}
      <p
        className="text-xs text-muted-foreground leading-relaxed"
        data-testid="ensemble-summary"
      >
        <strong className="text-foreground">Recommendation:</strong>{" "}
        {recommended_name} — {summary}
      </p>

      {/* Options */}
      <div className="space-y-2" role="list" aria-label="Ensemble options">
        {options.map((opt) => (
          <div key={opt.algorithm} role="listitem">
            <OptionRow option={opt} />
          </div>
        ))}
      </div>

      <figcaption className="text-xs text-muted-foreground border-t pt-2 mt-1">
        Ensembles combine base models (Linear, Random Forest, Gradient Boosting)
        without requiring new data — they use what you&apos;ve already uploaded.
      </figcaption>
    </figure>
  )
}
