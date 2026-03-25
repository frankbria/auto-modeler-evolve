"use client"

import { Badge } from "@/components/ui/badge"
import type { TrainingStartedResult } from "@/lib/types"

interface TrainingStartedCardProps {
  result: TrainingStartedResult
  onNavigateToModels?: () => void
}

const ALGO_LABELS: Record<string, string> = {
  linear_regression: "Linear Regression",
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
  logistic_regression: "Logistic Regression",
  random_forest_classifier: "Random Forest",
  gradient_boosting_classifier: "Gradient Boosting",
  xgboost: "XGBoost",
  xgboost_classifier: "XGBoost",
  lightgbm: "LightGBM",
  lightgbm_classifier: "LightGBM",
  mlp_regressor: "Neural Network",
  mlp_classifier: "Neural Network",
}

function algoLabel(algo: string): string {
  return ALGO_LABELS[algo] ?? algo.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

export function TrainingStartedCard({ result, onNavigateToModels }: TrainingStartedCardProps) {
  const problemLabel =
    result.problem_type === "classification" ? "Classification" : "Regression"

  return (
    <div
      className="mt-2 rounded-lg border border-primary/20 bg-primary/5 p-3"
      data-testid="training-started-card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="text-xs font-semibold text-primary">Training Started</span>
        <Badge variant="secondary">{problemLabel}</Badge>
      </div>
      <p className="mb-2 text-sm text-foreground">
        Training{" "}
        <span className="font-semibold">{result.run_count}</span>{" "}
        model{result.run_count !== 1 ? "s" : ""} to predict{" "}
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs font-semibold">
          {result.target_column}
        </span>
      </p>
      <div className="flex flex-wrap gap-1.5">
        {result.algorithms.map((algo) => (
          <Badge key={algo} variant="outline">{algoLabel(algo)}</Badge>
        ))}
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        Check the{" "}
        {onNavigateToModels ? (
          <button
            onClick={onNavigateToModels}
            className="font-medium text-foreground underline-offset-2 hover:underline"
          >
            Models tab
          </button>
        ) : (
          <span className="font-medium text-foreground">Models tab</span>
        )}{" "}
        for real-time progress →
      </p>
    </div>
  )
}
