/**
 * CrossDeployPredictionCard — side-by-side prediction comparison across all active
 * deployed model versions for a project.
 *
 * Shown when the analyst asks: "compare what my two models predict for units=150"
 * "which deployment gives the highest prediction?" "run my models side by side"
 *
 * Each row = one active deployment. Columns: algorithm, environment badge,
 * prediction value, confidence (classification) or CI (regression), deployed date.
 * Winner row is highlighted when predictions are numeric.
 */
"use client"

import { CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { CrossDeployPredictionResult, CrossDeployPredictionRow } from "@/lib/types"

// Format a numeric prediction for display
function fmtPred(v: number | string): string {
  if (typeof v === "number") {
    return v.toLocaleString(undefined, { maximumFractionDigits: 3 })
  }
  return String(v)
}

// Format ISO timestamp to short date
function fmtDate(isoStr: string | null): string {
  if (!isoStr) return "—"
  try {
    return new Date(isoStr + "Z").toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "2-digit",
    })
  } catch {
    return isoStr
  }
}

function EnvironmentBadge({ env }: { env: string }) {
  const isProd = env === "production"
  return (
    <Badge
      className={`text-xs font-medium ${
        isProd
          ? "bg-emerald-100 text-emerald-800 border-emerald-300"
          : "bg-sky-100 text-sky-700 border-sky-300"
      }`}
      variant="outline"
    >
      {isProd ? "Production" : "Staging"}
    </Badge>
  )
}

function PredictionCell({
  row,
  isWinner,
}: {
  row: CrossDeployPredictionRow
  isWinner: boolean
}) {
  if (row.error) {
    return (
      <span className="text-rose-500 text-xs italic" data-testid="cross-deploy-error">
        Error
      </span>
    )
  }
  if (row.problem_type === "classification" && row.probabilities) {
    const topClass = Object.entries(row.probabilities).sort(
      ([, a], [, b]) => b - a
    )[0]
    const confPct = topClass ? Math.round(topClass[1] * 100) : null
    return (
      <span
        className={`font-semibold ${isWinner ? "text-emerald-700" : "text-foreground"}`}
        data-testid="cross-deploy-prediction"
      >
        {String(row.prediction)}
        {confPct !== null && (
          <span className="ml-1 text-xs text-muted-foreground">({confPct}%)</span>
        )}
      </span>
    )
  }
  if (row.confidence_interval) {
    const ci = row.confidence_interval
    return (
      <span data-testid="cross-deploy-prediction">
        <span
          className={`font-semibold ${isWinner ? "text-emerald-700" : "text-foreground"}`}
        >
          {fmtPred(row.prediction)}
        </span>
        <span className="text-xs text-muted-foreground ml-1">
          [{fmtPred(ci.lower)}–{fmtPred(ci.upper)}]
        </span>
      </span>
    )
  }
  return (
    <span
      className={`font-semibold ${isWinner ? "text-emerald-700" : "text-foreground"}`}
      data-testid="cross-deploy-prediction"
    >
      {fmtPred(row.prediction)}
    </span>
  )
}

interface Props {
  result: CrossDeployPredictionResult
}

export function CrossDeployPredictionCard({ result }: Props) {
  const {
    target_column,
    problem_type,
    n_deployments,
    provided_features,
    defaults_used,
    results,
    summary,
  } = result

  // Find winner (highest numeric prediction for regression, highest confidence for classification)
  let winnerId: string | null = null
  if (problem_type === "regression") {
    const best = results
      .filter((r) => !r.error && typeof r.prediction === "number")
      .sort((a, b) => (b.prediction as number) - (a.prediction as number))[0]
    if (best) winnerId = best.deployment_id
  } else {
    const best = results
      .filter((r) => !r.error && r.confidence !== undefined)
      .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))[0]
    if (best) winnerId = best.deployment_id
  }

  return (
    <div
      className="border border-orange-300 rounded-lg overflow-hidden"
      role="region"
      aria-label="Cross-deployment prediction comparison"
      data-testid="cross-deploy-prediction-card"
    >
      <CardHeader className="bg-orange-50 border-b border-orange-200 py-3 px-4">
        <div className="flex items-center gap-2 flex-wrap">
          <span aria-hidden="true" className="text-lg">
            🔀
          </span>
          <CardTitle className="text-base font-semibold text-orange-900">
            Model Comparison
          </CardTitle>
          <Badge
            className="bg-orange-100 text-orange-800 border-orange-300 text-xs"
            variant="outline"
            data-testid="cross-deploy-count-badge"
          >
            {n_deployments} model{n_deployments !== 1 ? "s" : ""}
          </Badge>
          <Badge
            className="bg-slate-100 text-slate-600 border-slate-300 text-xs"
            variant="outline"
          >
            {problem_type === "regression" ? "Regression" : "Classification"}
          </Badge>
          <Badge
            className="bg-sky-100 text-sky-700 border-sky-300 text-xs font-medium"
            variant="outline"
          >
            Target: {target_column}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="p-4 space-y-3">
        {/* Feature inputs summary */}
        {(provided_features.length > 0 || defaults_used > 0) && (
          <div className="flex flex-wrap gap-1.5 items-center" data-testid="cross-deploy-features">
            <span className="text-xs text-muted-foreground font-medium">Inputs:</span>
            {provided_features.map((f) => (
              <Badge
                key={f}
                variant="outline"
                className="text-xs bg-blue-50 text-blue-700 border-blue-200"
              >
                {f}
              </Badge>
            ))}
            {defaults_used > 0 && (
              <span className="text-xs text-muted-foreground italic">
                +{defaults_used} using training-data averages
              </span>
            )}
          </div>
        )}

        {/* Comparison table */}
        <div className="overflow-x-auto" role="table" aria-label="Deployment predictions">
          <table className="w-full text-sm min-w-[400px]">
            <thead>
              <tr className="border-b border-muted">
                <th className="text-left py-1.5 px-2 text-xs font-semibold text-muted-foreground">
                  Model
                </th>
                <th className="text-left py-1.5 px-2 text-xs font-semibold text-muted-foreground">
                  Env
                </th>
                <th className="text-left py-1.5 px-2 text-xs font-semibold text-muted-foreground">
                  Prediction
                </th>
                <th className="text-right py-1.5 px-2 text-xs font-semibold text-muted-foreground">
                  Deployed
                </th>
              </tr>
            </thead>
            <tbody>
              {results.map((row) => {
                const isWinner = row.deployment_id === winnerId
                return (
                  <tr
                    key={row.deployment_id}
                    className={`border-b border-muted/50 ${
                      isWinner ? "bg-emerald-50" : ""
                    }`}
                    data-testid="cross-deploy-row"
                    aria-label={`${row.algorithm_plain}${isWinner ? " (best)" : ""}`}
                  >
                    <td className="py-2 px-2">
                      <div className="flex items-center gap-1">
                        {isWinner && (
                          <span aria-label="highest prediction" className="text-sm">
                            🏆
                          </span>
                        )}
                        <span
                          className="font-medium text-sm"
                          data-testid="cross-deploy-algorithm"
                        >
                          {row.algorithm_plain}
                        </span>
                      </div>
                    </td>
                    <td className="py-2 px-2">
                      <EnvironmentBadge env={row.environment} />
                    </td>
                    <td className="py-2 px-2">
                      <PredictionCell row={row} isWinner={isWinner} />
                    </td>
                    <td className="py-2 px-2 text-right text-xs text-muted-foreground">
                      {fmtDate(row.deployed_at)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Summary */}
        <p
          className="text-xs text-muted-foreground italic border-t border-muted pt-2"
          data-testid="cross-deploy-summary"
        >
          {summary}
        </p>

        {/* Accessibility */}
        <figcaption className="sr-only">
          Cross-deployment prediction comparison across {n_deployments} model
          {n_deployments !== 1 ? "s" : ""} for target {target_column}.{" "}
          {summary}
        </figcaption>
      </CardContent>
    </div>
  )
}
