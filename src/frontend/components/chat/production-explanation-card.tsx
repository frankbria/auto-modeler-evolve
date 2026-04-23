"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ProdPredictionExplanationResult } from "@/lib/types"

interface ProductionExplanationCardProps {
  result: ProdPredictionExplanationResult
}

function fmtDate(iso: string | null): string {
  if (!iso) return "unknown time"
  try {
    return new Date(iso).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
  } catch {
    return iso
  }
}

function algoName(raw: string | null): string {
  if (!raw) return "Unknown"
  const map: Record<string, string> = {
    linear_regression: "Linear Regression",
    logistic_regression: "Logistic Regression",
    random_forest_regressor: "Random Forest",
    random_forest_classifier: "Random Forest",
    gradient_boosting_regressor: "Gradient Boosting",
    gradient_boosting_classifier: "Gradient Boosting",
    xgboost_regressor: "XGBoost",
    xgboost_classifier: "XGBoost",
    lightgbm_regressor: "LightGBM",
    lightgbm_classifier: "LightGBM",
    mlp_regressor: "Neural Network",
    mlp_classifier: "Neural Network",
    decision_tree_regressor: "Decision Tree",
    decision_tree_classifier: "Decision Tree",
  }
  return map[raw] ?? raw.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

export function ProductionExplanationCard({ result }: ProductionExplanationCardProps) {
  const maxAbs = Math.max(
    ...result.contributions.map((c) => Math.abs(c.contribution)),
    0.001
  )

  return (
    <figure
      className="not-prose"
      aria-label="Production prediction explanation"
      role="region"
    >
      <Card className="border-violet-400/40 bg-violet-50/30">
        <CardHeader className="pb-2">
          <CardTitle className="flex flex-wrap items-center gap-2 text-base">
            <span aria-hidden="true">🔍</span>
            <span>Production Prediction Explained</span>
            {result.algorithm && (
              <Badge variant="outline" className="text-xs font-normal">
                {algoName(result.algorithm)}
              </Badge>
            )}
            {result.problem_type && (
              <Badge variant="outline" className="text-xs font-normal">
                {result.problem_type === "classification" ? "Classification" : "Regression"}
              </Badge>
            )}
            <span className="ml-auto text-xs font-normal text-muted-foreground">
              {fmtDate(result.created_at)}
            </span>
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-3">
          {/* Prediction value */}
          <div className="flex items-center gap-3 rounded-md border border-violet-200/60 bg-white/60 px-3 py-2">
            <span className="text-xs text-muted-foreground">
              {result.target_column ?? "Prediction"}:
            </span>
            <span className="text-lg font-semibold text-violet-700">
              {result.prediction}
            </span>
            {result.confidence != null && (
              <Badge className="ml-auto bg-violet-100 text-violet-700 text-xs">
                {(result.confidence * 100).toFixed(0)}% confidence
              </Badge>
            )}
          </div>

          {/* Feature contributions */}
          <div>
            <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Feature Contributions
            </p>
            <div className="space-y-1.5" role="list" aria-label="Feature contributions">
              {result.contributions.slice(0, 10).map((c) => {
                const width = Math.round((Math.abs(c.contribution) / maxAbs) * 100)
                const isPos = c.direction === "positive"
                return (
                  <div
                    key={c.feature}
                    className="flex items-center gap-2"
                    role="listitem"
                    aria-label={`${c.feature}: ${isPos ? "+" : ""}${c.contribution.toFixed(4)} contribution`}
                  >
                    <span className="w-32 shrink-0 truncate text-xs text-right text-foreground">
                      {c.feature}
                    </span>
                    <div className="flex flex-1 items-center gap-1.5">
                      <div className="h-3 rounded-sm overflow-hidden bg-muted w-full relative">
                        <div
                          className={`h-full rounded-sm ${isPos ? "bg-sky-400" : "bg-rose-400"}`}
                          style={{ width: `${width}%` }}
                        />
                      </div>
                      <span className={`w-14 shrink-0 text-xs text-right ${isPos ? "text-sky-600" : "text-rose-600"}`}>
                        {isPos ? "+" : ""}{c.contribution.toFixed(3)}
                      </span>
                    </div>
                    <span className="w-16 shrink-0 text-xs text-muted-foreground text-right">
                      val: {c.value}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Summary */}
          {result.summary && (
            <p className="text-xs text-muted-foreground italic">{result.summary}</p>
          )}

          <figcaption className="sr-only">
            {`Production prediction explanation for ${result.target_column ?? "target"}:
             predicted ${result.prediction} at ${fmtDate(result.created_at)}.
             Top drivers: ${result.top_drivers.slice(0, 3).join(", ")}.`}
          </figcaption>
        </CardContent>
      </Card>
    </figure>
  )
}
