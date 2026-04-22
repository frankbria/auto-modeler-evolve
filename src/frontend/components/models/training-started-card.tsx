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

const STRATEGY_LABELS: Record<string, { label: string; color: string }> = {
  class_weight: { label: "Class Weighting", color: "bg-blue-100 text-blue-800 border-blue-300" },
  smote: { label: "SMOTE Oversampling", color: "bg-violet-100 text-violet-800 border-violet-300" },
  threshold: { label: "Threshold Tuning", color: "bg-amber-100 text-amber-800 border-amber-300" },
}

export function TrainingStartedCard({ result, onNavigateToModels }: TrainingStartedCardProps) {
  const problemLabel =
    result.problem_type === "classification" ? "Classification" : "Regression"
  const strategy = result.imbalance_strategy ? STRATEGY_LABELS[result.imbalance_strategy] : null

  return (
    <div
      className="mt-2 rounded-lg border border-primary/20 bg-primary/5 p-3"
      data-testid="training-started-card"
    >
      <div className="mb-2 flex items-center gap-2 flex-wrap">
        <span className="text-xs font-semibold text-primary">Training Started</span>
        <Badge variant="secondary">{problemLabel}</Badge>
        {strategy && (
          <span
            className={`inline-flex items-center rounded border px-2 py-0.5 text-xs font-medium ${strategy.color}`}
            data-testid="imbalance-strategy-badge"
          >
            {strategy.label}
          </span>
        )}
      </div>
      <p className="mb-2 text-sm text-foreground">
        Training{" "}
        <span className="font-semibold">{result.run_count}</span>{" "}
        model{result.run_count !== 1 ? "s" : ""} to predict{" "}
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs font-semibold">
          {result.target_column}
        </span>
        {strategy && (
          <span className="text-muted-foreground"> with {strategy.label.toLowerCase()}</span>
        )}
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
