"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { api } from "@/lib/api"
import type { Deployment, FeatureSchemaEntry, PredictionResult, PredictionExplanation, FeatureContribution, ConfidenceInterval, ModelComparisonResult, ComparisonResponse, GuardRailWarning, DashboardFieldEntry } from "@/lib/types"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert snake_case / underscored algorithm ID to plain English. */
function algoName(raw: string | null): string {
  if (!raw) return "Unknown"
  const map: Record<string, string> = {
    linear_regression: "Linear Regression",
    ridge_regression: "Ridge Regression",
    lasso_regression: "Lasso Regression",
    random_forest_regressor: "Random Forest",
    gradient_boosting_regressor: "Gradient Boosting",
    xgboost_regressor: "XGBoost",
    lightgbm_regressor: "LightGBM",
    mlp_regressor: "Neural Network",
    voting_regressor: "Ensemble (Voting)",
    stacking_regressor: "Ensemble (Stacking)",
    logistic_regression: "Logistic Regression",
    random_forest_classifier: "Random Forest",
    gradient_boosting_classifier: "Gradient Boosting",
    xgboost_classifier: "XGBoost",
    lightgbm_classifier: "LightGBM",
    mlp_classifier: "Neural Network",
    voting_classifier: "Ensemble (Voting)",
    stacking_classifier: "Ensemble (Stacking)",
  }
  return map[raw] ?? raw.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Format a column name for display. */
function colLabel(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Pick the primary accuracy metric and render it in plain English. */
function primaryMetricSummary(metrics: Record<string, number>, problemType: string | null): string {
  if (problemType === "regression") {
    const r2 = metrics.r2
    if (r2 !== undefined) {
      const pct = Math.round(r2 * 100)
      if (pct >= 90) return `Explains ${pct}% of variation (excellent)`
      if (pct >= 75) return `Explains ${pct}% of variation (good)`
      if (pct >= 50) return `Explains ${pct}% of variation (moderate)`
      return `Explains ${pct}% of variation`
    }
  }
  if (problemType === "classification") {
    const acc = metrics.accuracy
    if (acc !== undefined) {
      const pct = Math.round(acc * 100)
      return `${pct}% accuracy on training data`
    }
  }
  return ""
}

/** Format a number with k/M suffix. */
function fmtNum(n: number, decimals = 2): string {
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return n.toLocaleString(undefined, { maximumFractionDigits: decimals })
}

// ---------------------------------------------------------------------------
// Feature contribution waterfall (horizontal bar chart)
// ---------------------------------------------------------------------------

function ContributionBar({
  item,
  maxAbs,
}: {
  item: FeatureContribution
  maxAbs: number
}) {
  const pct = maxAbs > 0 ? Math.abs(item.contribution) / maxAbs : 0
  const barWidth = Math.max(2, Math.round(pct * 100))
  const isPositive = item.direction === "positive"

  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="w-32 shrink-0 truncate text-right text-muted-foreground" title={item.feature}>
        {colLabel(item.feature)}
      </div>
      <div className="flex flex-1 items-center gap-1">
        {/* Left (negative) side */}
        <div className="flex flex-1 justify-end">
          {!isPositive && (
            <div
              className="h-4 rounded-l bg-red-400/70"
              style={{ width: `${barWidth}%` }}
              title={`contribution: ${item.contribution.toFixed(4)}`}
            />
          )}
        </div>
        {/* Centre divider */}
        <div className="h-5 w-px bg-border shrink-0" />
        {/* Right (positive) side */}
        <div className="flex flex-1 justify-start">
          {isPositive && (
            <div
              className="h-4 rounded-r bg-primary/60"
              style={{ width: `${barWidth}%` }}
              title={`contribution: ${item.contribution.toFixed(4)}`}
            />
          )}
        </div>
      </div>
      <div className="w-20 shrink-0 text-muted-foreground tabular-nums">
        {item.value} <span className="text-[10px]">(avg {item.mean_value})</span>
      </div>
    </div>
  )
}

function ExplanationCard({ explanation }: { explanation: PredictionExplanation }) {
  const top = explanation.contributions.slice(0, 8)
  const maxAbs = Math.max(...top.map((c) => Math.abs(c.contribution)), 1e-8)

  return (
    <Card className="border-primary/20 bg-primary/5">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Why this prediction?</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">{explanation.summary}</p>

        {/* Legend */}
        <div className="flex items-center gap-4 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-4 rounded bg-red-400/70" />
            Pushed prediction down
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-4 rounded bg-primary/60" />
            Pushed prediction up
          </span>
        </div>

        {/* Contribution bars */}
        <div className="space-y-1.5">
          {top.map((item) => (
            <ContributionBar key={item.feature} item={item} maxAbs={maxAbs} />
          ))}
        </div>

        <p className="text-[10px] text-muted-foreground">
          Each bar shows how much that feature pushed the prediction relative to the training average.
        </p>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Cross-deployment model comparison card
// ---------------------------------------------------------------------------

function CompareResultRow({ result }: { result: ModelComparisonResult }) {
  const algo = result.algorithm ? algoName(result.algorithm) : "Unknown"
  const date = result.trained_at
    ? new Date(result.trained_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })
    : "—"

  if (result.error) {
    return (
      <tr className="border-b last:border-0">
        <td className="px-3 py-2 text-sm font-medium">{algo}</td>
        <td className="px-3 py-2 text-xs text-muted-foreground">{date}</td>
        <td className="px-3 py-2 text-sm text-destructive" colSpan={2}>{result.error}</td>
      </tr>
    )
  }

  const pred = typeof result.prediction === "number"
    ? result.prediction.toLocaleString(undefined, { maximumFractionDigits: 4 })
    : String(result.prediction ?? "—")

  const ci = result.confidence_interval
  const conf = result.confidence

  return (
    <tr className="border-b last:border-0 hover:bg-muted/20">
      <td className="px-3 py-2 text-sm font-medium">{algo}</td>
      <td className="px-3 py-2 text-xs text-muted-foreground tabular-nums">{date}</td>
      <td className="px-3 py-2 text-sm font-bold tabular-nums" data-testid="compare-prediction">{pred}</td>
      <td className="px-3 py-2 text-xs text-muted-foreground tabular-nums">
        {ci ? `${ci.lower.toLocaleString()} – ${ci.upper.toLocaleString()}` : conf != null ? `${Math.round(conf * 100)}% confidence` : "—"}
      </td>
    </tr>
  )
}

interface CompareModelsCardProps {
  currentDeploymentId: string
  projectId: string
  features: Record<string, unknown>
}

function CompareModelsCard({ currentDeploymentId, projectId, features }: CompareModelsCardProps) {
  const [otherDeployments, setOtherDeployments] = useState<Deployment[]>([])
  const [selectedId, setSelectedId] = useState<string>("")
  const [comparing, setComparing] = useState(false)
  const [comparisonResult, setComparisonResult] = useState<ComparisonResponse | null>(null)
  const [compareError, setCompareError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    api.deploy.listByProject(projectId).then((deps) => {
      const others = deps.filter((d) => d.id !== currentDeploymentId)
      setOtherDeployments(others)
      if (others.length > 0) setSelectedId(others[0].id)
    }).catch(() => {})
  }, [projectId, currentDeploymentId])

  if (otherDeployments.length === 0) return null

  const handleCompare = async () => {
    if (!selectedId) return
    setComparing(true)
    setCompareError(null)
    setComparisonResult(null)
    try {
      const result = await api.deploy.compareModels([currentDeploymentId, selectedId], features)
      setComparisonResult(result)
    } catch {
      setCompareError("Comparison failed. Please try again.")
    } finally {
      setComparing(false)
    }
  }

  return (
    <Card data-testid="compare-models-card">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Compare Model Versions</CardTitle>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            onClick={() => setOpen((v) => !v)}
            data-testid="compare-toggle"
          >
            {open ? "Hide" : `Compare with another version (${otherDeployments.length} available)`}
          </Button>
        </div>
      </CardHeader>
      {open && (
        <CardContent className="space-y-3">
          <div className="flex items-center gap-2">
            <select
              className="flex-1 rounded-md border bg-background px-3 py-1.5 text-sm"
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              data-testid="compare-select"
            >
              {otherDeployments.map((d) => (
                <option key={d.id} value={d.id}>
                  {algoName(d.algorithm ?? null)} — {d.created_at ? new Date(d.created_at).toLocaleDateString() : "Unknown date"}
                </option>
              ))}
            </select>
            <Button
              size="sm"
              onClick={handleCompare}
              disabled={comparing || !selectedId}
              data-testid="compare-button"
            >
              {comparing ? "Comparing..." : "Compare"}
            </Button>
          </div>

          {compareError && (
            <p className="text-xs text-destructive">{compareError}</p>
          )}

          {comparisonResult && (
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b bg-muted/40">
                    <th className="px-3 py-1.5 text-left font-medium text-muted-foreground">Model</th>
                    <th className="px-3 py-1.5 text-left font-medium text-muted-foreground">Trained</th>
                    <th className="px-3 py-1.5 text-left font-medium text-muted-foreground">Prediction</th>
                    <th className="px-3 py-1.5 text-left font-medium text-muted-foreground">Uncertainty</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonResult.results.map((r) => (
                    <CompareResultRow key={r.deployment_id} result={r} />
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <p className="text-[10px] text-muted-foreground">
            Runs the same inputs through each model version to help you verify whether a retrained model improved.
          </p>
        </CardContent>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Confidence interval display
// ---------------------------------------------------------------------------

function ConfidenceIntervalBadge({ interval }: { interval: ConfidenceInterval }) {
  const pct = Math.round(interval.level * 100)
  const fmt = (v: number) =>
    v.toLocaleString(undefined, { maximumFractionDigits: 4 })

  return (
    <div
      className="rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-center dark:border-blue-800 dark:bg-blue-950/30"
      data-testid="confidence-interval"
    >
      <p className="text-xs text-muted-foreground">
        {pct}% prediction interval
      </p>
      <p className="mt-0.5 text-sm font-medium tabular-nums">
        {fmt(interval.lower)} – {fmt(interval.upper)}
      </p>
      <p className="mt-1 text-[10px] text-muted-foreground">
        The actual value is likely to fall within this range based on past prediction accuracy.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Model context card (for VP trust)
// ---------------------------------------------------------------------------

interface ModelContextCardProps {
  deployment: Deployment
}

function ModelContextCard({ deployment }: ModelContextCardProps) {
  const accuracy = primaryMetricSummary(deployment.metrics ?? {}, deployment.problem_type)
  const deployedDate = deployment.created_at
    ? new Date(deployment.created_at).toLocaleDateString(undefined, {
        month: "long",
        day: "numeric",
        year: "numeric",
      })
    : null

  return (
    <Card className="border-muted bg-muted/30" data-testid="model-context-card">
      <CardContent className="pt-4 pb-3">
        <div className="flex flex-wrap items-start gap-x-4 gap-y-1 text-sm">
          <span className="text-muted-foreground">
            <span className="font-medium text-foreground">Algorithm:</span>{" "}
            {algoName(deployment.algorithm ?? null)}
          </span>
          {accuracy && (
            <span className="text-muted-foreground">
              <span className="font-medium text-foreground">Accuracy:</span>{" "}
              {accuracy}
            </span>
          )}
          {deployedDate && (
            <span className="text-muted-foreground">
              <span className="font-medium text-foreground">Deployed:</span>{" "}
              {deployedDate}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Main prediction dashboard
// ---------------------------------------------------------------------------

interface PredictionHistoryRecord {
  id: number
  inputs: Record<string, string>
  prediction: string | number
  probabilities?: Record<string, number>
  timestamp: string
}

export default function PredictionDashboard() {
  const params = useParams<{ id: string }>()
  const deploymentId = params.id

  const [deployment, setDeployment] = useState<Deployment | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [inputs, setInputs] = useState<Record<string, string>>({})
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [explanation, setExplanation] = useState<PredictionExplanation | null>(null)
  const [predError, setPredError] = useState<string | null>(null)
  const [showExplanation, setShowExplanation] = useState(false)
  const [fetchingExplanation, setFetchingExplanation] = useState(false)
  const [explanationError, setExplanationError] = useState(false)
  const [history, setHistory] = useState<PredictionHistoryRecord[]>([])
  const [, setHistoryCounter] = useState(0)
  const [presets, setPresets] = useState<import("@/lib/types").DeploymentPreset[]>([])
  const [dashboardConfig, setDashboardConfig] = useState<DashboardFieldEntry[]>([])

  useEffect(() => {
    api.deploy
      .get(deploymentId)
      .then((d) => {
        setDeployment(d)
        // Pre-fill inputs with training-average defaults
        const defaults: Record<string, string> = {}
        for (const entry of d.feature_schema ?? []) {
          if (entry.type === "numeric") {
            defaults[entry.name] = String(entry.median ?? "")
          } else if (entry.options && entry.options.length > 0) {
            defaults[entry.name] = entry.options[0]
          } else {
            defaults[entry.name] = ""
          }
        }
        setInputs(defaults)
      })
      .catch(() => setError("Prediction service not found or inactive."))
      .finally(() => setLoading(false))
  }, [deploymentId])

  useEffect(() => {
    api.deploy.getPresets(deploymentId).then(setPresets).catch(() => {})
  }, [deploymentId])

  useEffect(() => {
    api.deploy
      .getDashboardConfig(deploymentId)
      .then((cfg) => setDashboardConfig(cfg.fields ?? []))
      .catch(() => {})
  }, [deploymentId])

  const loadPreset = (featureValues: Record<string, string | number>) => {
    const next = { ...inputs }
    for (const [key, val] of Object.entries(featureValues)) {
      next[key] = String(val)
    }
    setInputs(next)
    setResult(null)
    setExplanation(null)
    setShowExplanation(false)
  }

  const cfgMap = Object.fromEntries(
    dashboardConfig.map((f) => [f.feature_name, f])
  )
  const hiddenCount = dashboardConfig.filter((f) => !f.is_visible).length
  const lockedCount = dashboardConfig.filter((f) => f.is_locked).length

  const buildPayload = () => {
    if (!deployment) return {}
    const payload: Record<string, unknown> = {}
    for (const entry of deployment.feature_schema ?? []) {
      const cfg = cfgMap[entry.name]
      // Inject locked value from config — bypasses user input for locked fields
      if (cfg?.is_locked && cfg.locked_value != null) {
        payload[entry.name] =
          entry.type === "numeric" ? parseFloat(cfg.locked_value) : cfg.locked_value
        continue
      }
      const raw = inputs[entry.name] ?? ""
      if (entry.type === "numeric") {
        payload[entry.name] = raw === "" ? null : parseFloat(raw)
      } else {
        payload[entry.name] = raw
      }
    }
    return payload
  }

  const handlePredict = async () => {
    if (!deployment) return
    setPredicting(true)
    setPredError(null)
    setResult(null)
    setExplanation(null)
    setShowExplanation(false)

    const payload = buildPayload()
    try {
      const r = await api.deploy.predict(deploymentId, payload)
      setResult(r)
      setHistoryCounter((n) => {
        const next = n + 1
        const record: PredictionHistoryRecord = {
          id: next,
          inputs: { ...inputs },
          prediction: r.prediction,
          probabilities: r.probabilities,
          timestamp: new Date().toLocaleTimeString(),
        }
        setHistory((prev) => [record, ...prev].slice(0, 20))
        return next
      })
    } catch {
      setPredError("Prediction failed. Please check your inputs and try again.")
    } finally {
      setPredicting(false)
    }
  }

  const handleExplain = async () => {
    if (!deployment || !result) return
    if (explanation) {
      setShowExplanation(true)
      return
    }
    setFetchingExplanation(true)
    setExplanationError(false)
    const payload = buildPayload()
    try {
      const exp = await api.deploy.explain(deploymentId, payload)
      setExplanation(exp)
      setShowExplanation(true)
    } catch {
      setExplanationError(true)
    } finally {
      setFetchingExplanation(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background px-4 py-8">
        <div className="mx-auto max-w-xl space-y-4 animate-pulse">
          <div className="h-8 w-3/4 rounded-md bg-muted" />
          <div className="h-4 w-1/2 rounded-md bg-muted" />
          <div className="h-32 rounded-lg bg-muted" />
          <div className="h-64 rounded-lg bg-muted" />
        </div>
      </div>
    )
  }

  if (error || !deployment) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Card className="max-w-md">
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">
              {error ?? "This prediction service is unavailable."}
            </p>
          </CardContent>
        </Card>
      </div>
    )
  }

  const rawSchema = deployment.feature_schema ?? []
  // Filter out fields hidden by the analyst's dashboard config
  const schema = rawSchema.filter((e) => cfgMap[e.name]?.is_visible !== false)
  const targetLabel = colLabel(deployment.target_column ?? "Output")
  const pageTitle = `${targetLabel} Predictor`
  const isSimplifiedView = hiddenCount > 0 || lockedCount > 0

  return (
    <div className="min-h-screen bg-background px-4 py-8">
      <div className="mx-auto max-w-xl space-y-6">
        {/* Header */}
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="text-2xl font-bold" data-testid="page-title">{pageTitle}</h1>
            {deployment.environment && (
              <Badge
                className={
                  deployment.environment === "production"
                    ? "bg-green-100 text-green-800 border-green-200"
                    : "bg-amber-100 text-amber-800 border-amber-200"
                }
                data-testid="environment-badge"
              >
                {deployment.environment === "production" ? "Production" : "Staging"}
              </Badge>
            )}
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Enter your data below to get a predicted{" "}
            <strong>{deployment.target_column}</strong> value.
          </p>
        </div>

        {/* Model context (trust panel) */}
        <ModelContextCard deployment={deployment} />

        {/* Quick scenarios (presets) */}
        {presets.length > 0 && (
          <div data-testid="preset-section">
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Quick Scenarios
            </p>
            <div className="flex flex-wrap gap-2">
              {presets.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => loadPreset(preset.feature_values)}
                  className="rounded-full border px-3 py-1 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  data-testid="preset-button"
                  aria-label={`Load preset: ${preset.name}`}
                >
                  {preset.name}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input form */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between flex-wrap gap-2">
              <CardTitle>Your Scenario</CardTitle>
              {isSimplifiedView && (
                <Badge
                  variant="outline"
                  className="border-sky-300 text-sky-700 text-xs"
                  data-testid="simplified-view-badge"
                >
                  Simplified view
                </Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground mt-0.5">
              Fields are pre-filled with training averages — adjust them for your specific situation.
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            {schema.length === 0 && (
              <p className="text-xs text-muted-foreground">
                No feature schema available for this deployment.
              </p>
            )}
            {schema.map((entry: FeatureSchemaEntry) => {
              const fieldCfg = cfgMap[entry.name]
              const isLocked = fieldCfg?.is_locked === true
              const lockedVal = fieldCfg?.locked_value ?? null
              const displayLabel = fieldCfg?.display_label
                ? fieldCfg.display_label
                : colLabel(entry.name)
              return (
                <div key={entry.name}>
                  <label className="mb-1 block text-sm font-medium">
                    {displayLabel}
                    {isLocked && (
                      <span className="ml-1.5 text-xs font-normal text-amber-600">(locked)</span>
                    )}
                    {!isLocked && entry.type === "numeric" && entry.mean !== undefined && entry.mean !== null && (
                      <span className="ml-1.5 text-xs font-normal text-muted-foreground">
                        (avg: {fmtNum(entry.mean)})
                      </span>
                    )}
                  </label>
                  {isLocked ? (
                    <Input
                      type="text"
                      value={lockedVal ?? ""}
                      disabled
                      className="text-sm bg-amber-50 border-amber-200"
                      aria-label={`${displayLabel} (locked)`}
                      data-testid={`locked-field-${entry.name}`}
                    />
                  ) : entry.type === "categorical" && entry.options ? (
                    <select
                      className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                      value={inputs[entry.name] ?? ""}
                      onChange={(e) =>
                        setInputs((prev) => ({ ...prev, [entry.name]: e.target.value }))
                      }
                      aria-label={displayLabel}
                    >
                      {entry.options.map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <Input
                      type="number"
                      step="any"
                      placeholder={entry.median != null ? `Default: ${fmtNum(entry.median)}` : "Enter a value"}
                      value={inputs[entry.name] ?? ""}
                      onChange={(e) =>
                        setInputs((prev) => ({ ...prev, [entry.name]: e.target.value }))
                      }
                      className="text-sm"
                      aria-label={displayLabel}
                    />
                  )}
                </div>
              )
            })}
          </CardContent>
        </Card>

        <Button
          onClick={handlePredict}
          disabled={predicting}
          className="w-full"
          size="lg"
          data-testid="predict-button"
        >
          {predicting ? "Calculating..." : `Get ${targetLabel} Prediction`}
        </Button>

        {predError && (
          <p className="text-sm text-destructive">{predError}</p>
        )}

        {/* Result */}
        {result && (
          <Card className="border-primary/30 bg-primary/5">
            <CardHeader>
              <CardTitle className="text-base" data-testid="result-title">
                Predicted {result.target_column}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="rounded-lg bg-background p-4 text-center" data-testid="prediction-value">
                <p className="text-4xl font-bold tabular-nums">
                  {typeof result.prediction === "number"
                    ? result.prediction.toLocaleString(undefined, {
                        maximumFractionDigits: 4,
                      })
                    : result.prediction}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Predicted {result.target_column}
                </p>
              </div>

              {result.confidence_interval && (
                <ConfidenceIntervalBadge interval={result.confidence_interval} />
              )}

              {result.confidence !== undefined && result.problem_type === "classification" && (
                <div
                  className="flex items-center justify-between rounded-md border border-green-200 bg-green-50 px-3 py-2 dark:border-green-800 dark:bg-green-950/30"
                  data-testid="classification-confidence"
                >
                  <span className="text-xs text-muted-foreground">Model confidence</span>
                  <span className="text-sm font-semibold tabular-nums text-green-700 dark:text-green-400">
                    {Math.round(result.confidence * 100)}%
                  </span>
                </div>
              )}

              {result.below_threshold && (
                <div
                  className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 dark:border-amber-700 dark:bg-amber-950/30"
                  data-testid="below-threshold-warning"
                >
                  <p className="text-xs font-semibold text-amber-800 dark:text-amber-300">
                    ⚠ Low-confidence prediction
                  </p>
                  <p className="mt-0.5 text-xs text-amber-700 dark:text-amber-400">
                    {result.threshold_message ?? "This prediction is below the configured confidence threshold and may be unreliable."}
                  </p>
                </div>
              )}

              {result.probabilities && (
                <div>
                  <p className="mb-2 text-xs font-medium">Probability per outcome</p>
                  <div className="space-y-1">
                    {Object.entries(result.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([cls, prob]) => (
                        <div key={cls} className="flex items-center gap-2 text-xs">
                          <span className="w-24 truncate font-medium">{cls}</span>
                          <div className="flex-1 overflow-hidden rounded-full bg-muted">
                            <div
                              className="h-2 rounded-full bg-primary transition-all"
                              style={{ width: `${Math.round(prob * 100)}%` }}
                            />
                          </div>
                          <span className="w-10 text-right tabular-nums text-muted-foreground">
                            {Math.round(prob * 100)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Guard-rail input warnings */}
              {result.guard_rail_warnings && result.guard_rail_warnings.length > 0 && (
                <div
                  className="rounded-lg border border-amber-300 bg-amber-50 p-3 space-y-1"
                  aria-label="Input validation warnings"
                  data-testid="guard-rail-warnings"
                >
                  <p className="text-xs font-semibold text-amber-800 flex items-center gap-1">
                    <span aria-hidden="true">⚠️</span>
                    Input warnings ({result.guard_rail_warnings.length})
                  </p>
                  {result.guard_rail_warnings.map((w: GuardRailWarning, i: number) => (
                    <p key={i} className="text-xs text-amber-700">
                      {w.message}
                    </p>
                  ))}
                </div>
              )}

              {/* Explain button */}
              <Button
                size="sm"
                variant="outline"
                className="w-full"
                onClick={handleExplain}
                disabled={fetchingExplanation}
                data-testid="explain-button"
              >
                {fetchingExplanation
                  ? "Analyzing..."
                  : showExplanation
                  ? "Hide explanation"
                  : "Why this prediction?"}
              </Button>
              {explanationError && (
                <p className="text-xs text-muted-foreground text-center" data-testid="explanation-error">
                  Explanation unavailable for this prediction.
                </p>
              )}
            </CardContent>
          </Card>
        )}

        {/* Explanation waterfall */}
        {showExplanation && explanation && (
          <ExplanationCard explanation={explanation} />
        )}

        {/* Session history */}
        {history.length > 0 && (
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">
                  Session History ({history.length})
                </CardTitle>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 text-xs"
                  onClick={() => {
                    const schema = deployment.feature_schema ?? []
                    const headers = [
                      "#",
                      "Time",
                      ...schema.map((e) => colLabel(e.name)),
                      `Predicted ${deployment.target_column}`,
                    ]
                    const rows = [...history].reverse().map((rec) => [
                      rec.id,
                      rec.timestamp,
                      ...schema.map((e) => rec.inputs[e.name] ?? ""),
                      rec.prediction,
                    ])
                    const csv = [headers, ...rows]
                      .map((r) => r.map(String).join(","))
                      .join("\n")
                    const blob = new Blob([csv], { type: "text/csv" })
                    const url = URL.createObjectURL(blob)
                    const a = document.createElement("a")
                    a.href = url
                    a.download = `${deployment.target_column ?? "predictions"}-session.csv`
                    a.click()
                    URL.revokeObjectURL(url)
                  }}
                >
                  Download CSV
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b bg-muted/40">
                      <th className="px-3 py-1.5 text-left font-medium text-muted-foreground">#</th>
                      <th className="px-3 py-1.5 text-left font-medium text-muted-foreground">Time</th>
                      <th className="px-3 py-1.5 text-left font-medium text-muted-foreground">Key Inputs</th>
                      <th className="px-3 py-1.5 text-right font-medium text-muted-foreground">Prediction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((rec) => {
                      // Show up to 3 key inputs (prefer those with non-default values, or just first 3)
                      const schemaItems = deployment.feature_schema ?? []
                      const keyInputs = schemaItems
                        .slice(0, 3)
                        .map((e) => `${colLabel(e.name)}: ${rec.inputs[e.name] ?? "—"}`)
                        .join(" · ")

                      return (
                        <tr key={rec.id} className="border-b last:border-0 hover:bg-muted/20">
                          <td className="px-3 py-1.5 tabular-nums text-muted-foreground">{rec.id}</td>
                          <td className="px-3 py-1.5 text-muted-foreground">{rec.timestamp}</td>
                          <td className="px-3 py-1.5 text-muted-foreground truncate max-w-[140px]" title={keyInputs}>{keyInputs}</td>
                          <td className="px-3 py-1.5 text-right font-medium tabular-nums">
                            {typeof rec.prediction === "number"
                              ? rec.prediction.toLocaleString(undefined, { maximumFractionDigits: 4 })
                              : String(rec.prediction)}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {deployment.project_id && (
          <CompareModelsCard
            currentDeploymentId={deploymentId}
            projectId={deployment.project_id}
            features={buildPayload()}
          />
        )}

        <p className="text-center text-xs text-muted-foreground">
          Powered by AutoModeler &middot; {deployment.request_count.toLocaleString()} predictions served
        </p>
      </div>
    </div>
  )
}
