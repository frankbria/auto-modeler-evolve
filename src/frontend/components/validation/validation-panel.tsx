"use client"

import { useState, useCallback } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ScatterChart,
  Scatter,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { api } from "@/lib/api"
import type {
  ValidationResult,
  GlobalImportanceResult,
  PredictionExplanation,
  MetricWithCI,
  ConfusionMatrixAnalysis,
  ResidualsAnalysis,
} from "@/lib/types"

// ---------------------------------------------------------------------------
// MetricCI — a single metric card with confidence interval
// ---------------------------------------------------------------------------

function MetricCICard({
  name,
  info,
  problemType,
}: {
  name: string
  info: MetricWithCI
  problemType: string
}) {
  const isPercent = ["accuracy", "f1", "precision", "recall", "roc_auc"].includes(name)
  const fmt = (v: number) =>
    isPercent ? `${(v * 100).toFixed(1)}%` : name === "r2" ? v.toFixed(3) : v.toFixed(2)

  const label = name === "roc_auc" ? "ROC AUC" : name.toUpperCase().replace("_", " ")

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-xs font-semibold text-muted-foreground">{label}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-xl font-bold tabular-nums">{fmt(info.mean)}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">
          ±{fmt(info.std)} (95% CI: {fmt(info.ci_low)} – {fmt(info.ci_high)})
        </p>
        <div className="mt-2 flex gap-1">
          {info.fold_scores.map((s, i) => (
            <div
              key={i}
              title={`Fold ${i + 1}: ${fmt(s)}`}
              className="h-2 flex-1 rounded-sm bg-primary/40"
              style={{ opacity: 0.4 + s * 0.6 }}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// ConfusionMatrixView
// ---------------------------------------------------------------------------

function ConfusionMatrixView({ analysis }: { analysis: ConfusionMatrixAnalysis }) {
  const { labels, matrix, test_accuracy, per_class_accuracy } = analysis
  const maxVal = Math.max(...matrix.flat(), 1)

  return (
    <div>
      <p className="mb-2 text-xs text-muted-foreground">
        Test accuracy: <strong>{(test_accuracy * 100).toFixed(1)}%</strong>
        {" — "}diagonal cells are correct predictions; off-diagonal are mistakes.
      </p>
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="px-2 py-1 text-muted-foreground">Actual ↓ / Predicted →</th>
              {labels.map((l) => (
                <th key={l} className="px-2 py-1 font-medium">{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="px-2 py-1 font-medium">{labels[i]}</td>
                {row.map((val, j) => {
                  const intensity = val / maxVal
                  const isCorrect = i === j
                  return (
                    <td
                      key={j}
                      className="px-3 py-2 text-center tabular-nums"
                      style={{
                        backgroundColor: isCorrect
                          ? `rgba(34,197,94,${0.15 + intensity * 0.6})`
                          : val > 0
                          ? `rgba(239,68,68,${0.1 + intensity * 0.5})`
                          : "transparent",
                      }}
                    >
                      {val}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {Object.keys(per_class_accuracy).length > 0 && (
        <div className="mt-3">
          <p className="mb-1 text-xs font-medium text-muted-foreground">Per-class accuracy</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(per_class_accuracy).map(([cls, acc]) => (
              <span key={cls} className="rounded-md border px-2 py-0.5 text-xs">
                {cls}: {(acc * 100).toFixed(1)}%
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// ResidualsView
// ---------------------------------------------------------------------------

function ResidualsView({ analysis }: { analysis: ResidualsAnalysis }) {
  const { actual, predicted, mae, rmse } = analysis
  const scatterData = actual.map((a, i) => ({ actual: a, predicted: predicted[i] }))
  // Reference line: perfect prediction line range
  const allVals = [...actual, ...predicted]
  const minVal = Math.min(...allVals)
  const maxVal = Math.max(...allVals)

  return (
    <div>
      <p className="mb-2 text-xs text-muted-foreground">
        MAE: <strong>{mae.toFixed(2)}</strong> · RMSE: <strong>{rmse.toFixed(2)}</strong>
        {" — "}points close to the diagonal line indicate accurate predictions.
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <ScatterChart margin={{ top: 8, right: 8, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="actual"
            name="Actual"
            type="number"
            domain={[minVal, maxVal]}
            tick={{ fontSize: 10 }}
            label={{ value: "Actual", position: "insideBottom", offset: -10, fontSize: 10 }}
          />
          <YAxis
            dataKey="predicted"
            name="Predicted"
            type="number"
            domain={[minVal, maxVal]}
            tick={{ fontSize: 10 }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{ fontSize: 11 }}
            formatter={(value, name) => [
              value != null ? Number(value).toFixed(2) : "",
              String(name),
            ]}
          />
          <ReferenceLine
            segment={[
              { x: minVal, y: minVal },
              { x: maxVal, y: maxVal },
            ]}
            stroke="hsl(var(--muted-foreground))"
            strokeDasharray="4 4"
          />
          <Scatter data={scatterData} fill="hsl(var(--primary))" fillOpacity={0.6} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// FeatureImportanceView
// ---------------------------------------------------------------------------

function FeatureImportanceView({ result }: { result: GlobalImportanceResult }) {
  const top = result.features.slice(0, 10)
  return (
    <div>
      <p className="mb-3 text-xs text-muted-foreground">{result.summary}</p>
      <ResponsiveContainer width="100%" height={top.length * 28 + 40}>
        <BarChart
          data={top}
          layout="vertical"
          margin={{ top: 4, right: 32, bottom: 4, left: 8 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(var(--border))" />
          <XAxis type="number" tick={{ fontSize: 10 }} />
          <YAxis
            type="category"
            dataKey="column"
            width={100}
            tick={{ fontSize: 10 }}
          />
          <Tooltip
            contentStyle={{ fontSize: 11 }}
            formatter={(v, _, entry) => {
              const val = v != null ? Number(v) : 0
              const std = (entry as { payload?: ImportanceFeature })?.payload?.std ?? 0
              return [`${(val * 100).toFixed(1)}% (±${(std * 100).toFixed(1)}%)`, "Importance"]
            }}
          />
          <Bar dataKey="importance" fill="hsl(var(--primary))" radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// PredictionExplanationView
// ---------------------------------------------------------------------------

function PredictionExplanationView({ explanation }: { explanation: PredictionExplanation }) {
  const { prediction, probability, top_factors, target_column, problem_type } = explanation

  const predStr =
    problem_type === "regression"
      ? Number(prediction).toFixed(2)
      : String(prediction)

  const maxImpact = Math.max(...top_factors.map((f) => Math.abs(f.impact)), 0.001)

  return (
    <div>
      <div className="mb-4 rounded-lg border bg-muted/40 px-4 py-3">
        <p className="text-xs text-muted-foreground">Prediction for {target_column}</p>
        <p className="text-lg font-bold">{predStr}</p>
        {probability != null && (
          <p className="text-xs text-muted-foreground">
            Confidence: {(probability * 100).toFixed(1)}%
          </p>
        )}
      </div>

      <p className="mb-2 text-xs font-medium text-muted-foreground">
        What drove this prediction (top factors):
      </p>
      <div className="flex flex-col gap-1.5">
        {top_factors.map((f) => {
          const barWidth = Math.abs(f.impact) / maxImpact * 100
          const isPositive = f.direction === "positive"
          const isNegative = f.direction === "negative"
          return (
            <div key={f.column} className="flex items-center gap-2">
              <span className="w-28 shrink-0 truncate text-xs">{f.column}</span>
              <div className="flex-1 overflow-hidden rounded-sm bg-muted/40">
                <div
                  className={`h-4 rounded-sm ${isPositive ? "bg-emerald-500/70" : isNegative ? "bg-red-400/70" : "bg-muted-foreground/30"}`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>
              <span className="w-12 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
                {f.impact > 0 ? "+" : ""}{f.impact.toFixed(3)}
              </span>
            </div>
          )
        })}
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        Green bars push the prediction up; red bars push it down.
        Values shown are the change in prediction when each feature is removed.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// ValidationPanel — main export
// ---------------------------------------------------------------------------

interface ValidationPanelProps {
  modelRunId: string
  displayName: string
  totalRows: number
}

export function ValidationPanel({
  modelRunId,
  displayName,
  totalRows,
}: ValidationPanelProps) {
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null)
  const [importanceResult, setImportanceResult] = useState<GlobalImportanceResult | null>(null)
  const [explanation, setExplanation] = useState<PredictionExplanation | null>(null)

  const [loadingMetrics, setLoadingMetrics] = useState(false)
  const [loadingImportance, setLoadingImportance] = useState(false)
  const [loadingExplain, setLoadingExplain] = useState(false)
  const [rowInput, setRowInput] = useState("0")
  const [explainError, setExplainError] = useState("")

  const [activeSection, setActiveSection] = useState<"metrics" | "importance" | "explain">("metrics")

  const handleLoadMetrics = useCallback(async () => {
    setLoadingMetrics(true)
    try {
      const result = await api.validate.metrics(modelRunId)
      setValidationResult(result)
    } finally {
      setLoadingMetrics(false)
    }
  }, [modelRunId])

  const handleLoadImportance = useCallback(async () => {
    setLoadingImportance(true)
    try {
      const result = await api.validate.explain(modelRunId)
      setImportanceResult(result)
    } finally {
      setLoadingImportance(false)
    }
  }, [modelRunId])

  const handleExplainRow = useCallback(async () => {
    const idx = parseInt(rowInput, 10)
    if (isNaN(idx) || idx < 0) {
      setExplainError("Enter a valid row number (0 or greater)")
      return
    }
    setExplainError("")
    setLoadingExplain(true)
    try {
      const result = await api.validate.explainRow(modelRunId, idx)
      setExplanation(result)
    } catch {
      setExplainError("Could not explain this row — check the row number and try again.")
    } finally {
      setLoadingExplain(false)
    }
  }, [modelRunId, rowInput])

  const sectionTabs = (
    ["metrics", "importance", "explain"] as const
  ).map((s) => ({
    id: s,
    label: s === "metrics" ? "CV Results" : s === "importance" ? "Feature Impact" : "Explain Row",
  }))

  return (
    <div className="flex flex-col gap-4 p-4">
      <div>
        <h3 className="text-sm font-semibold">{displayName} — Validation</h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Cross-validation results, feature importance, and per-prediction explanations.
        </p>
      </div>

      {/* Sub-tabs */}
      <div className="flex gap-1 rounded-lg bg-muted/50 p-1">
        {sectionTabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveSection(t.id)}
            className={`flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
              activeSection === t.id
                ? "bg-background shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ---- CV Metrics ---- */}
      {activeSection === "metrics" && (
        <div className="flex flex-col gap-4">
          {!validationResult && (
            <Button size="sm" onClick={handleLoadMetrics} disabled={loadingMetrics}>
              {loadingMetrics ? "Running validation..." : "Run cross-validation"}
            </Button>
          )}

          {validationResult && (
            <>
              {/* Metric cards */}
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(validationResult.metrics).map(([name, info]) => (
                  <MetricCICard
                    key={name}
                    name={name}
                    info={info}
                    problemType={validationResult.problem_type}
                  />
                ))}
              </div>

              {/* Consistency note */}
              <div className="rounded-lg border bg-blue-50 px-3 py-2 text-xs text-blue-800 dark:bg-blue-950 dark:text-blue-200">
                Results are from {validationResult.n_folds}-fold cross-validation on{" "}
                {validationResult.n_rows.toLocaleString()} rows. Confidence intervals show how
                consistent the model is across different data splits.
              </div>

              {/* Error analysis */}
              <div>
                <h4 className="mb-2 text-xs font-semibold">Error Analysis</h4>
                {validationResult.error_analysis.type === "confusion_matrix" ? (
                  <ConfusionMatrixView
                    analysis={validationResult.error_analysis as ConfusionMatrixAnalysis}
                  />
                ) : (
                  <ResidualsView
                    analysis={validationResult.error_analysis as ResidualsAnalysis}
                  />
                )}
              </div>

              {/* Limitations */}
              <div>
                <h4 className="mb-2 text-xs font-semibold">Limitations & Caveats</h4>
                <div className="flex flex-col gap-1.5">
                  {validationResult.limitations.map((l, i) => (
                    <div key={i} className="rounded-md border bg-amber-50/60 px-3 py-2 text-xs text-amber-900 dark:bg-amber-950/60 dark:text-amber-200">
                      {l}
                    </div>
                  ))}
                </div>
              </div>

              <Button
                size="sm"
                variant="outline"
                onClick={handleLoadMetrics}
                disabled={loadingMetrics}
              >
                {loadingMetrics ? "Refreshing..." : "Refresh"}
              </Button>
            </>
          )}
        </div>
      )}

      {/* ---- Feature Importance ---- */}
      {activeSection === "importance" && (
        <div className="flex flex-col gap-4">
          {!importanceResult && (
            <Button size="sm" onClick={handleLoadImportance} disabled={loadingImportance}>
              {loadingImportance ? "Computing importance..." : "Compute feature importance"}
            </Button>
          )}
          {importanceResult && (
            <>
              <FeatureImportanceView result={importanceResult} />
              <Button
                size="sm"
                variant="outline"
                onClick={handleLoadImportance}
                disabled={loadingImportance}
              >
                {loadingImportance ? "..." : "Refresh"}
              </Button>
            </>
          )}
        </div>
      )}

      {/* ---- Explain Row ---- */}
      {activeSection === "explain" && (
        <div className="flex flex-col gap-4">
          <p className="text-xs text-muted-foreground">
            Enter a row number (0–{totalRows - 1}) to see why the model made that specific prediction.
          </p>
          <div className="flex gap-2">
            <Input
              type="number"
              min={0}
              max={totalRows - 1}
              value={rowInput}
              onChange={(e) => setRowInput(e.target.value)}
              className="w-24 text-xs"
              onKeyDown={(e) => { if (e.key === "Enter") handleExplainRow() }}
            />
            <Button size="sm" onClick={handleExplainRow} disabled={loadingExplain}>
              {loadingExplain ? "Explaining..." : "Explain"}
            </Button>
          </div>
          {explainError && (
            <p className="text-xs text-destructive">{explainError}</p>
          )}
          {explanation && <PredictionExplanationView explanation={explanation} />}
        </div>
      )}
    </div>
  )
}

// Re-export individual types for convenience
type ImportanceFeature = import("@/lib/types").ImportanceFeature
