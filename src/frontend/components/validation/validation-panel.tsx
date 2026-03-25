"use client"

import { useState } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine,
} from "recharts"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { api } from "@/lib/api"
import type {
  ValidationMetricsResponse,
  GlobalExplanationResponse,
  RowExplanationResponse,
  ResidualsResult,
  ConfusionMatrixResult,
} from "@/lib/types"

interface Props {
  projectId: string
  selectedRunId: string | null
  algorithmName: string | null
  onNavigateToModels?: () => void
}

type SubTab = "cv" | "error" | "importance" | "explain"

export function ValidationPanel({ selectedRunId, algorithmName, onNavigateToModels }: Props) {
  const [subTab, setSubTab] = useState<SubTab>("cv")
  const [loading, setLoading] = useState(false)
  const [metrics, setMetrics] = useState<ValidationMetricsResponse | null>(null)
  const [globalExplain, setGlobalExplain] = useState<GlobalExplanationResponse | null>(null)
  const [rowExplain, setRowExplain] = useState<RowExplanationResponse | null>(null)
  const [rowIndex, setRowIndex] = useState("0")
  const [error, setError] = useState<string | null>(null)

  async function loadMetrics() {
    if (!selectedRunId) return
    setLoading(true)
    setError(null)
    try {
      const result = await api.validation.metrics(selectedRunId)
      setMetrics(result)
    } catch {
      setError("Failed to load validation metrics.")
    } finally {
      setLoading(false)
    }
  }

  async function loadExplain() {
    if (!selectedRunId) return
    setLoading(true)
    setError(null)
    try {
      const result = await api.validation.explain(selectedRunId)
      setGlobalExplain(result)
    } catch {
      setError("Failed to load feature importance.")
    } finally {
      setLoading(false)
    }
  }

  async function loadRowExplain() {
    if (!selectedRunId) return
    const idx = parseInt(rowIndex, 10)
    if (isNaN(idx) || idx < 0) return
    setLoading(true)
    setError(null)
    try {
      const result = await api.validation.explainRow(selectedRunId, idx)
      setRowExplain(result)
    } catch {
      setError("Failed to load row explanation. Check that the row index is valid.")
    } finally {
      setLoading(false)
    }
  }

  const handleTabChange = (tab: SubTab) => {
    setSubTab(tab)
    if (tab === "cv" && !metrics) loadMetrics()
    if (tab === "error" && !metrics) loadMetrics()
    if (tab === "importance" && !globalExplain) loadExplain()
  }

  if (!selectedRunId) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-sm text-muted-foreground">
        <p>Select a model in the Models tab first to validate it.</p>
        {onNavigateToModels && (
          <Button variant="outline" size="sm" onClick={onNavigateToModels}>
            Go to Models tab
          </Button>
        )}
      </div>
    )
  }

  const confidenceColor = (level: string) =>
    level === "high"
      ? "bg-green-100 text-green-800 border-green-200"
      : level === "medium"
      ? "bg-amber-100 text-amber-800 border-amber-200"
      : "bg-red-100 text-red-800 border-red-200"

  return (
    <ScrollArea className="flex-1">
      <div className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold">Model Validation</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {algorithmName ?? "Selected model"} — cross-validation, error analysis, explainability
            </p>
          </div>
          {metrics && (
            <Badge
              variant="outline"
              className={`text-xs ${confidenceColor(metrics.confidence.overall_confidence)}`}
            >
              {metrics.confidence.overall_confidence.toUpperCase()} confidence
            </Badge>
          )}
        </div>

        {/* Sub-tabs */}
        <div role="tablist" aria-label="Validation sections" className="mb-4 flex gap-1 border-b">
          {(["cv", "error", "importance", "explain"] as SubTab[]).map((tab) => {
            const labels: Record<SubTab, string> = {
              cv: "Cross-Validation",
              error: "Error Analysis",
              importance: "Feature Importance",
              explain: "Explain Row",
            }
            return (
              <button
                key={tab}
                role="tab"
                aria-selected={subTab === tab}
                aria-controls={`validation-tabpanel-${tab}`}
                id={`validation-tab-${tab}`}
                onClick={() => handleTabChange(tab)}
                className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                  subTab === tab
                    ? "border-b-2 border-primary text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {labels[tab]}
              </button>
            )
          })}
        </div>

        {loading && (
          <p className="text-xs text-muted-foreground">Loading...</p>
        )}

        {error && (
          <p className="rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {error}
          </p>
        )}

        {/* Cross-Validation tab */}
        {subTab === "cv" && metrics && !loading && (
          <CrossValidationView cv={metrics.cross_validation} />
        )}

        {subTab === "cv" && !metrics && !loading && (
          <Button size="sm" onClick={loadMetrics}>
            Run Cross-Validation
          </Button>
        )}

        {/* Error Analysis tab */}
        {subTab === "error" && metrics && !loading && (
          <ErrorAnalysisView analysis={metrics.error_analysis} />
        )}

        {subTab === "error" && !metrics && !loading && (
          <Button size="sm" onClick={loadMetrics}>
            Load Error Analysis
          </Button>
        )}

        {/* Confidence card (shown below CV and Error) */}
        {(subTab === "cv" || subTab === "error") && metrics && !loading && (
          <ConfidenceCard confidence={metrics.confidence} />
        )}

        {/* Feature Importance tab */}
        {subTab === "importance" && globalExplain && !loading && (
          <GlobalImportanceView data={globalExplain} />
        )}

        {subTab === "importance" && !globalExplain && !loading && (
          <Button size="sm" onClick={loadExplain}>
            Compute Feature Importance
          </Button>
        )}

        {/* Explain Row tab */}
        {subTab === "explain" && (
          <div className="space-y-4">
            <div className="flex gap-2">
              <Input
                type="number"
                min={0}
                value={rowIndex}
                onChange={(e) => setRowIndex(e.target.value)}
                placeholder="Row index (0-based)"
                className="w-40 text-xs"
              />
              <Button size="sm" onClick={loadRowExplain} disabled={loading}>
                {loading ? "..." : "Explain"}
              </Button>
            </div>
            {rowExplain && !loading && (
              <RowExplainView data={rowExplain} />
            )}
          </div>
        )}
      </div>
    </ScrollArea>
  )
}


// ---------------------------------------------------------------------------
// Sub-views
// ---------------------------------------------------------------------------

function CrossValidationView({ cv }: { cv: ValidationMetricsResponse["cross_validation"] }) {
  const scoreData = cv.scores.map((s, i) => ({ fold: `Fold ${i + 1}`, score: s }))

  return (
    <div className="space-y-4">
      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-xs">
            {cv.n_splits}-Fold Cross-Validation ({cv.metric})
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-1 text-xs text-muted-foreground">{cv.summary}</p>
          <div className="mt-2 grid grid-cols-3 gap-2">
            <div className="rounded border p-2 text-center">
              <p className="text-xs text-muted-foreground">Mean</p>
              <p className="text-sm font-semibold">{cv.mean?.toFixed(3) ?? "—"}</p>
            </div>
            <div className="rounded border p-2 text-center">
              <p className="text-xs text-muted-foreground">Std Dev</p>
              <p className="text-sm font-semibold">{cv.std?.toFixed(3) ?? "—"}</p>
            </div>
            <div className="rounded border p-2 text-center">
              <p className="text-xs text-muted-foreground">95% CI</p>
              <p className="text-sm font-semibold">
                [{cv.ci_low?.toFixed(2) ?? "—"}, {cv.ci_high?.toFixed(2) ?? "—"}]
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {scoreData.length > 0 && (
        <div>
          <p className="mb-2 text-xs font-medium">Score per fold</p>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={scoreData} margin={{ top: 0, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="fold" tick={{ fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ fontSize: 11 }}
                formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
              />
              <Bar dataKey="score" fill="hsl(var(--primary))" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}


function ErrorAnalysisView({ analysis }: { analysis: ValidationMetricsResponse["error_analysis"] }) {
  if (analysis.type === "residuals") {
    return <ResidualsView data={analysis as ResidualsResult} />
  }
  return <ConfusionMatrixView data={analysis as ConfusionMatrixResult} />
}


function ResidualsView({ data }: { data: ResidualsResult }) {
  return (
    <div className="space-y-4">
      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-xs">Residual Analysis (Regression)</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-2 text-xs text-muted-foreground">{data.summary}</p>
          <div className="grid grid-cols-3 gap-2">
            <div className="rounded border p-2 text-center">
              <p className="text-xs text-muted-foreground">MAE</p>
              <p className="text-sm font-semibold">{data.mae.toFixed(3)}</p>
            </div>
            <div className="rounded border p-2 text-center">
              <p className="text-xs text-muted-foreground">Bias</p>
              <p className="text-sm font-semibold">{data.bias.toFixed(3)}</p>
            </div>
            <div className="rounded border p-2 text-center">
              <p className="text-xs text-muted-foreground">P90 Error</p>
              <p className="text-sm font-semibold">{data.percentile_90.toFixed(3)}</p>
            </div>
          </div>
        </CardContent>
      </Card>
      <div>
        <p className="mb-1 text-xs font-medium">Predicted vs Residual</p>
        <p className="mb-2 text-xs text-muted-foreground">
          Points scattered around zero = good. Patterns or drift = systematic errors.
        </p>
        <ResponsiveContainer width="100%" height={180}>
          <ScatterChart margin={{ top: 0, right: 8, left: 8, bottom: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis
              dataKey="predicted"
              name="Predicted"
              tick={{ fontSize: 9 }}
              label={{ value: "Predicted", position: "insideBottom", offset: -4, fontSize: 10 }}
            />
            <YAxis
              dataKey="residual"
              name="Residual"
              tick={{ fontSize: 9 }}
              label={{ value: "Residual (actual − predicted)", angle: -90, position: "insideLeft", offset: 10, fontSize: 9 }}
            />
            <ReferenceLine y={0} stroke="hsl(var(--destructive))" strokeDasharray="4 4" />
            <Tooltip
              contentStyle={{ fontSize: 11 }}
              formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
            />
            <Scatter data={data.scatter} fill="hsl(var(--primary))" opacity={0.6} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}


function ConfusionMatrixView({ data }: { data: ConfusionMatrixResult }) {
  return (
    <div className="space-y-4">
      <Card size="sm">
        <CardHeader>
          <CardTitle className="text-xs">Confusion Matrix (Classification)</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-2 text-xs text-muted-foreground">{data.summary}</p>
          <p className="mb-3 text-xs">
            Accuracy: <span className="font-semibold">{(data.accuracy * 100).toFixed(1)}%</span>
            &nbsp;({data.correct}/{data.total} correct)
          </p>
          <div className="overflow-x-auto">
            <table className="text-xs border-collapse">
              <thead>
                <tr>
                  <th className="px-2 py-1 text-left text-muted-foreground">Actual ↓ / Predicted →</th>
                  {data.labels.map((l) => (
                    <th key={l} className="px-2 py-1 text-center font-medium">{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.matrix.map((row, i) => (
                  <tr key={i}>
                    <td className="px-2 py-1 font-medium text-muted-foreground">
                      {data.labels[i]}
                    </td>
                    {row.map((cell, j) => (
                      <td
                        key={j}
                        className={`px-3 py-1 text-center rounded text-xs ${
                          i === j
                            ? "bg-green-100 font-semibold text-green-800 dark:bg-green-900 dark:text-green-200"
                            : cell > 0
                            ? "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300"
                            : ""
                        }`}
                      >
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}


function ConfidenceCard({ confidence }: { confidence: ValidationMetricsResponse["confidence"] }) {
  const borderColor =
    confidence.overall_confidence === "high"
      ? "border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950"
      : confidence.overall_confidence === "medium"
      ? "border-amber-200 bg-amber-50 dark:border-amber-900 dark:bg-amber-950"
      : "border-red-200 bg-red-50 dark:border-red-950 dark:bg-red-950"

  return (
    <div className={`mt-4 rounded-lg border p-3 ${borderColor}`}>
      <p className="mb-1 text-xs font-semibold">
        Confidence & Limitations
      </p>
      <ul className="space-y-1">
        {confidence.limitations.map((l, i) => (
          <li key={i} className="text-xs text-muted-foreground">
            • {l}
          </li>
        ))}
      </ul>
    </div>
  )
}


function GlobalImportanceView({ data }: { data: GlobalExplanationResponse }) {
  const chartData = data.feature_importance.slice(0, 15).map((item) => ({
    feature: item.feature.length > 18 ? item.feature.slice(0, 16) + "…" : item.feature,
    importance: +(item.importance * 100).toFixed(2),
    fullName: item.feature,
  }))

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">{data.summary}</p>
      <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 22)}>
        <BarChart
          layout="vertical"
          data={chartData}
          margin={{ top: 0, right: 16, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            type="number"
            tick={{ fontSize: 9 }}
            label={{ value: "Importance (%)", position: "insideBottom", offset: -2, fontSize: 10 }}
          />
          <YAxis type="category" dataKey="feature" tick={{ fontSize: 9 }} width={110} />
          <Tooltip
            contentStyle={{ fontSize: 11 }}
            formatter={(v, _name, props) => [
              typeof v === "number" ? `${v.toFixed(2)}%` : String(v),
              props.payload?.fullName ?? "Importance",
            ]}
          />
          <Bar dataKey="importance" fill="hsl(var(--primary))" radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}


function RowExplainView({ data }: { data: RowExplanationResponse }) {
  const top = data.contributions.slice(0, 10)
  const maxAbs = Math.max(...top.map((c) => Math.abs(c.contribution)), 1e-10)

  return (
    <div className="space-y-3">
      <Card size="sm">
        <CardContent className="pt-3">
          <p className="text-xs text-muted-foreground">{data.summary}</p>
          <div className="mt-2 grid grid-cols-2 gap-2">
            <div className="rounded border p-2 text-center">
              <p className="text-xs text-muted-foreground">Prediction</p>
              <p className="text-sm font-semibold">{data.prediction_value.toFixed(4)}</p>
            </div>
            {data.actual_value !== null && (
              <div className="rounded border p-2 text-center">
                <p className="text-xs text-muted-foreground">Actual</p>
                <p className="text-sm font-semibold">{data.actual_value.toFixed(4)}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <div>
        <p className="mb-2 text-xs font-medium">Feature Contributions (top 10)</p>
        <p className="mb-3 text-xs text-muted-foreground">
          Green bars pushed the prediction higher; red bars pushed it lower.
        </p>
        <div className="space-y-2">
          {top.map((c) => {
            const pct = (Math.abs(c.contribution) / maxAbs) * 100
            const isPos = c.direction === "positive"
            return (
              <div key={c.feature} className="flex items-center gap-2">
                <span
                  className="min-w-[100px] truncate text-xs text-right text-muted-foreground"
                  title={c.feature}
                >
                  {c.feature}
                </span>
                <div className="flex flex-1 items-center gap-1">
                  <div
                    className={`h-4 rounded text-xs text-white flex items-center px-1 min-w-[4px] ${
                      isPos ? "bg-green-500" : "bg-red-400"
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-xs tabular-nums text-muted-foreground w-16 text-right">
                  {c.value.toFixed(2)}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
