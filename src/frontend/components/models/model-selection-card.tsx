"use client"

import { Badge } from "@/components/ui/badge"
import type { ModelSelectionResult, ModelSelectionRun, SelectionCriteria } from "@/lib/types"

interface ModelSelectionCardProps {
  result: ModelSelectionResult
}

const CRITERIA_LABEL: Record<SelectionCriteria, string> = {
  accuracy: "Highest Accuracy",
  explainability: "Most Explainable",
  stability: "Most Stable",
  speed: "Fastest",
  balanced: "Best Overall",
}

const CRITERIA_COLOR: Record<SelectionCriteria, string> = {
  accuracy: "bg-blue-100 text-blue-800",
  explainability: "bg-purple-100 text-purple-800",
  stability: "bg-teal-100 text-teal-800",
  speed: "bg-amber-100 text-amber-800",
  balanced: "bg-indigo-100 text-indigo-800",
}

const COMPONENT_LABEL: Record<string, string> = {
  accuracy: "Accuracy",
  explainability: "Explainability",
  stability: "Stability",
  speed: "Speed",
}

function ScoreBar({ score, label }: { score: number; label: string }) {
  const pct = Math.round(score * 100)
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-24 text-muted-foreground shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-400 rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-8 text-right text-muted-foreground">{pct}%</span>
    </div>
  )
}

function RunRow({ run, isWinner }: { run: ModelSelectionRun; isWinner: boolean }) {
  const scorePct = Math.round(run.score * 100)
  const metricPct = Math.round(run.primary_metric * 100)

  return (
    <li
      className={`flex items-start gap-3 py-3 ${isWinner ? "bg-indigo-50 -mx-4 px-4 rounded-md" : ""}`}
    >
      <span
        className={`mt-0.5 text-sm font-semibold w-5 shrink-0 ${isWinner ? "text-indigo-600" : "text-muted-foreground"}`}
        aria-label={`Rank ${run.rank}`}
      >
        {run.rank === 1 ? "🏆" : run.rank}
      </span>

      <div className="flex-1 min-w-0">
        <div className="flex flex-wrap items-center gap-1.5 mb-1">
          <span className={`text-sm font-medium ${isWinner ? "text-indigo-800" : "text-foreground"}`}>
            {run.algorithm_plain}
          </span>
          {run.is_selected && (
            <Badge className="text-xs bg-emerald-100 text-emerald-800 border-0">
              Currently selected
            </Badge>
          )}
          {run.is_deployed && (
            <Badge className="text-xs bg-blue-100 text-blue-800 border-0">
              Deployed
            </Badge>
          )}
        </div>

        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground mb-1.5">
          <span>
            {run.primary_metric_name}: <strong>{metricPct}%</strong>
          </span>
          <span>Score: <strong>{scorePct}%</strong></span>
        </div>

        {isWinner && (
          <div className="space-y-1 mt-1.5">
            {Object.entries(run.component_scores).map(([key, val]) => (
              <ScoreBar key={key} score={val} label={COMPONENT_LABEL[key] ?? key} />
            ))}
          </div>
        )}
      </div>
    </li>
  )
}

export function ModelSelectionCard({ result }: ModelSelectionCardProps) {
  const { criteria, winner, ranked_runs, summary, n_runs } = result

  if (!winner) {
    return (
      <figure
        className="border border-indigo-200 rounded-lg p-4 bg-white my-2"
        aria-label="Model selection advisor"
      >
        <p className="text-sm text-muted-foreground">No completed model runs to compare.</p>
      </figure>
    )
  }

  return (
    <figure
      className="border border-indigo-200 rounded-lg p-4 bg-white my-2 space-y-3"
      aria-label="Model selection advisor"
    >
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-lg">🏆</span>
          <h3 className="text-sm font-semibold text-indigo-800">
            Model Selection Recommendation
          </h3>
        </div>
        <div className="flex items-center gap-1.5">
          <Badge className={`text-xs border-0 ${CRITERIA_COLOR[criteria] ?? "bg-gray-100 text-gray-800"}`}>
            {CRITERIA_LABEL[criteria] ?? criteria}
          </Badge>
          <Badge className="text-xs bg-muted text-muted-foreground border-0">
            {n_runs} model{n_runs !== 1 ? "s" : ""} compared
          </Badge>
        </div>
      </div>

      {/* Winner highlight */}
      <div className="bg-indigo-50 border border-indigo-100 rounded-md p-3">
        <p className="text-xs font-medium text-indigo-600 uppercase tracking-wide mb-0.5">
          Recommended
        </p>
        <p className="text-sm font-semibold text-indigo-900">{winner.algorithm_plain}</p>
        <p className="text-xs text-indigo-700 mt-1 leading-relaxed">{winner.why}</p>
      </div>

      {/* Summary sentence */}
      <figcaption className="text-xs text-muted-foreground leading-relaxed">
        {summary}
      </figcaption>

      {/* Ranked runs */}
      {ranked_runs.length > 1 && (
        <div>
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">
            All models ranked
          </p>
          <ul className="divide-y divide-border" aria-label="Ranked model runs">
            {ranked_runs.map((run) => (
              <RunRow key={run.run_id} run={run} isWinner={run.rank === 1} />
            ))}
          </ul>
        </div>
      )}
    </figure>
  )
}
