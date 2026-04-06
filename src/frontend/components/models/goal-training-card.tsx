"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { GoalTrainingResult, GoalTrainingTrial } from "@/lib/types"

interface GoalTrainingCardProps {
  result: GoalTrainingResult
}

const METRIC_LABELS: Record<string, string> = {
  r2: "R²",
  accuracy: "Accuracy",
  f1: "F1 Score",
  precision: "Precision",
  recall: "Recall",
}

function formatScore(score: number, metric: string): string {
  if (metric === "r2") return score.toFixed(3)
  return `${(score * 100).toFixed(1)}%`
}

function formatTarget(target: number, metric: string): string {
  if (metric === "r2") return target.toFixed(2)
  return `${Math.round(target * 100)}%`
}

function TrialRow({ trial, goalMetric }: { trial: GoalTrainingTrial; goalMetric: string }) {
  return (
    <tr className="border-b last:border-0">
      <td className="py-1.5 pr-3 text-sm font-medium">{trial.algorithm_name}</td>
      <td className="py-1.5 pr-3 text-sm tabular-nums">
        {formatScore(trial.score, goalMetric)}
      </td>
      <td className="py-1.5">
        {trial.achieved_goal ? (
          <span aria-label="Goal achieved" className="text-emerald-600 font-bold text-base">
            ✓
          </span>
        ) : (
          <span aria-label="Goal not achieved" className="text-muted-foreground text-base">
            ✗
          </span>
        )}
      </td>
    </tr>
  )
}

export function GoalTrainingCard({ result }: GoalTrainingCardProps) {
  const metricLabel = METRIC_LABELS[result.goal_metric] ?? result.goal_metric.toUpperCase()
  const targetStr = formatTarget(result.goal_target, result.goal_metric)
  const winnerScoreStr = formatScore(result.winner_score, result.goal_metric)

  return (
    <figure aria-label="Goal-driven training result">
      <Card
        className={`border-2 ${result.achieved ? "border-emerald-500" : "border-amber-400"} mt-2`}
      >
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2">
            <span aria-hidden="true">🎯</span>
            Goal-Driven Training
            <Badge
              className={
                result.achieved
                  ? "bg-emerald-100 text-emerald-800"
                  : "bg-amber-100 text-amber-800"
              }
            >
              {result.achieved ? "Goal Achieved ✓" : "Best Effort"}
            </Badge>
            <Badge variant="outline" className="text-xs">
              {metricLabel} ≥ {targetStr}
            </Badge>
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-3">
          {/* Winner highlight */}
          <div
            className={`rounded-md p-3 ${result.achieved ? "bg-emerald-50 border border-emerald-200" : "bg-amber-50 border border-amber-200"}`}
          >
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-0.5">
              Best Result
            </p>
            <p className="font-semibold">
              {result.winner_algorithm_name}
              <span className="ml-2 text-muted-foreground font-normal text-sm">
                {metricLabel} = {winnerScoreStr}
              </span>
            </p>
          </div>

          {/* Trials table */}
          {result.trials.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                Algorithms Tried
              </p>
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b">
                    <th className="py-1 pr-3 text-xs text-muted-foreground font-medium">
                      Algorithm
                    </th>
                    <th className="py-1 pr-3 text-xs text-muted-foreground font-medium">
                      {metricLabel}
                    </th>
                    <th className="py-1 text-xs text-muted-foreground font-medium">
                      Goal?
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.trials.map((trial, i) => (
                    <TrialRow key={i} trial={trial} goalMetric={result.goal_metric} />
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Tuning note */}
          {result.tried_tuning && (
            <p className="text-xs text-muted-foreground">
              <span aria-hidden="true">⚙️</span> Hyperparameter tuning was also attempted on
              the best algorithm.
            </p>
          )}

          {/* Summary */}
          <p className="text-sm text-foreground border-t pt-2">{result.summary}</p>
        </CardContent>
      </Card>
    </figure>
  )
}
