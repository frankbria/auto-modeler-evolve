"use client"

import { Badge } from "@/components/ui/badge"
import type { ReportReady } from "@/lib/types"

interface ReportReadyCardProps {
  result: ReportReady
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
  return (
    ALGO_LABELS[algo] ??
    algo.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  )
}

function metricDisplay(name: string, value: number | null): string {
  if (value === null) return ""
  if (name === "r2") return `R² ${value.toFixed(3)}`
  if (name === "accuracy") return `Accuracy ${(value * 100).toFixed(1)}%`
  return `${name.toUpperCase()}: ${value.toFixed(3)}`
}

const API_URL =
  typeof window !== "undefined"
    ? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
    : "http://localhost:8000"

export function ReportReadyCard({ result }: ReportReadyCardProps) {
  const downloadUrl = `${API_URL}${result.download_url}`
  const metricLabel = metricDisplay(result.metric_name, result.metric_value)

  return (
    <div
      className="mt-2 rounded-lg border border-teal-500/30 bg-teal-500/5 p-3"
      data-testid="report-ready-card"
    >
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm">📄</span>
        <span className="text-xs font-semibold text-teal-700 dark:text-teal-400">
          PDF Report Ready
        </span>
        <Badge className="bg-teal-100 text-teal-800 border-teal-200 dark:bg-teal-900/30 dark:text-teal-400 dark:border-teal-800">
          {result.problem_type === "classification" ? "Classification" : "Regression"}
        </Badge>
        {metricLabel && (
          <span className="ml-auto text-xs text-muted-foreground">
            {metricLabel}
          </span>
        )}
      </div>

      {/* Algorithm + description */}
      <p className="mb-3 text-sm text-foreground">
        Your{" "}
        <span className="font-semibold">{algoLabel(result.algorithm)}</span>{" "}
        model report includes metrics, feature importance, and confidence
        assessment — ready to share with stakeholders.
      </p>

      {/* Download button */}
      <a
        href={downloadUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="flex w-full items-center justify-center gap-2 rounded-md border border-teal-500/30 bg-teal-500/10 px-3 py-2 text-sm font-medium text-teal-700 transition-colors hover:bg-teal-500/20 dark:text-teal-400"
        data-testid="download-report-btn"
      >
        <span>⬇</span>
        Download PDF Report
      </a>
    </div>
  )
}
