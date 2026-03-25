"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import type { DeployedResult } from "@/lib/types"

interface DeployedCardProps {
  result: DeployedResult
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

function formatMetric(key: string, value: number): string {
  if (key === "r2") return `R² ${value.toFixed(3)}`
  if (key === "accuracy") return `Accuracy ${(value * 100).toFixed(1)}%`
  if (key === "mae") return `MAE ${value.toFixed(3)}`
  return `${key}: ${value.toFixed(3)}`
}

function primaryMetric(metrics: Record<string, number>): string | null {
  if (metrics.r2 !== undefined) return formatMetric("r2", metrics.r2)
  if (metrics.accuracy !== undefined) return formatMetric("accuracy", metrics.accuracy)
  const first = Object.entries(metrics)[0]
  return first ? formatMetric(first[0], first[1]) : null
}

export function DeployedCard({ result }: DeployedCardProps) {
  const [copied, setCopied] = useState(false)

  const apiEndpoint = `http://localhost:8000${result.endpoint_path}`
  const metricLabel = primaryMetric(result.metrics)

  function copyEndpoint() {
    navigator.clipboard.writeText(apiEndpoint).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div
      className="mt-2 rounded-lg border border-green-500/30 bg-green-500/5 p-3"
      data-testid="deployed-card"
    >
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-green-500" />
        <span className="text-xs font-semibold text-green-700 dark:text-green-400">
          Model Deployed
        </span>
        <Badge className="bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-800">
          {result.problem_type === "classification" ? "Classification" : "Regression"}
        </Badge>
        {metricLabel && (
          <span className="ml-auto text-xs text-muted-foreground">{metricLabel}</span>
        )}
      </div>

      {/* Algorithm + target */}
      <p className="mb-3 text-sm text-foreground">
        <span className="font-semibold">{algoLabel(result.algorithm)}</span> predicting{" "}
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs font-semibold">
          {result.target_column}
        </span>{" "}
        is now live.
      </p>

      {/* Dashboard link */}
      <div className="mb-2 flex items-center gap-2 rounded-md border border-border bg-background px-3 py-2">
        <span className="text-xs text-muted-foreground">Dashboard</span>
        <a
          href={result.dashboard_url}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto text-xs font-medium text-primary hover:underline"
          data-testid="dashboard-link"
        >
          Open →
        </a>
      </div>

      {/* API endpoint + copy */}
      <div className="flex items-center gap-2 rounded-md border border-border bg-background px-3 py-2">
        <span className="flex-1 truncate font-mono text-xs text-muted-foreground">
          {apiEndpoint}
        </span>
        <button
          onClick={copyEndpoint}
          className="shrink-0 text-xs font-medium text-primary hover:underline"
          data-testid="copy-endpoint-btn"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
    </div>
  )
}
