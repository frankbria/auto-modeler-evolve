"use client"

import { useState, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import type { Deployment, DeploymentAnalytics, ModelReadiness, DriftReport, WhatIfResult, FeedbackAccuracy, ModelHealth } from "@/lib/types"

interface DeploymentPanelProps {
  projectId: string
  selectedRunId: string | null
  algorithmName: string | null
  onDeployed?: (deployment: Deployment) => void
}

function ReadinessBadge({ verdict }: { verdict: ModelReadiness["verdict"] }) {
  if (verdict === "ready") return <Badge className="bg-green-100 text-green-800 border-green-200">Ready to deploy</Badge>
  if (verdict === "needs_attention") return <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200">Needs attention</Badge>
  return <Badge className="bg-red-100 text-red-800 border-red-200">Not ready</Badge>
}

function ReadinessCard({ readiness }: { readiness: ModelReadiness }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Model Readiness</CardTitle>
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold">{readiness.score}</span>
            <span className="text-xs text-muted-foreground">/100</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <ReadinessBadge verdict={readiness.verdict} />
        <p className="text-xs text-muted-foreground">{readiness.summary}</p>
        <div className="space-y-1">
          {readiness.checks.map((check) => (
            <div key={check.id} className="flex items-start gap-1.5 text-xs">
              <span className={check.passed ? "text-green-600" : "text-red-500"}>
                {check.passed ? "✓" : "✗"}
              </span>
              <span className={check.passed ? "text-foreground" : "text-muted-foreground"}>
                {check.label}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function AnalyticsMiniChart({ data }: { data: { date: string; count: number }[] }) {
  if (!data.length) return null
  const max = Math.max(...data.map((d) => d.count), 1)
  const recent = data.slice(-7)
  return (
    <div className="flex items-end gap-0.5 h-10">
      {recent.map((d) => (
        <div key={d.date} className="flex-1 flex flex-col items-center gap-0.5">
          <div
            className="w-full rounded-sm bg-primary/60"
            style={{ height: `${Math.max(2, (d.count / max) * 36)}px` }}
            title={`${d.date}: ${d.count} predictions`}
          />
        </div>
      ))}
    </div>
  )
}

function AnalyticsCard({ analytics }: { analytics: DeploymentAnalytics }) {
  const hasData = analytics.predictions_by_day.length > 0
  const avgText = analytics.recent_avg !== null
    ? `Avg prediction: ${analytics.recent_avg}`
    : null
  const classText = analytics.class_counts
    ? Object.entries(analytics.class_counts)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 3)
        .map(([label, count]) => `${label}: ${count}`)
        .join("  ·  ")
    : null

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Usage Analytics</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-xs">
        <div className="flex items-center justify-between text-muted-foreground">
          <span>Total predictions: <strong className="text-foreground">{analytics.total_predictions}</strong></span>
          {hasData && <span className="text-[10px]">last 7 days</span>}
        </div>
        {hasData && <AnalyticsMiniChart data={analytics.predictions_by_day} />}
        {avgText && <p className="text-muted-foreground">{avgText}</p>}
        {classText && <p className="text-muted-foreground">{classText}</p>}
        {!hasData && (
          <p className="text-muted-foreground italic">No predictions yet — share the dashboard link to get started.</p>
        )}
      </CardContent>
    </Card>
  )
}

function DriftStatusBadge({ status }: { status: DriftReport["status"] }) {
  if (status === "stable") return <Badge className="bg-green-100 text-green-800 border-green-200">Stable</Badge>
  if (status === "mild_drift") return <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200">Mild drift</Badge>
  if (status === "significant_drift") return <Badge className="bg-red-100 text-red-800 border-red-200">Significant drift</Badge>
  return <Badge variant="outline" className="text-muted-foreground">Insufficient data</Badge>
}

function DriftCard({ drift }: { drift: DriftReport }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Prediction Drift</CardTitle>
          {drift.drift_score !== null && (
            <div className="flex items-center gap-1">
              <span className="text-2xl font-bold">{drift.drift_score}</span>
              <span className="text-xs text-muted-foreground">/100</span>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <DriftStatusBadge status={drift.status} />
        <p className="text-xs text-muted-foreground">{drift.explanation}</p>
        {drift.baseline_stats && drift.recent_stats && (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded bg-muted/50 p-2">
              <p className="font-medium">Baseline</p>
              <p className="text-muted-foreground">Mean: {drift.baseline_stats.mean}</p>
              <p className="text-muted-foreground">Std: {drift.baseline_stats.std}</p>
            </div>
            <div className="rounded bg-muted/50 p-2">
              <p className="font-medium">Recent</p>
              <p className="text-muted-foreground">Mean: {drift.recent_stats.mean}</p>
              <p className="text-muted-foreground">Std: {drift.recent_stats.std}</p>
            </div>
          </div>
        )}
        {drift.baseline_dist && drift.recent_dist && (
          <div className="space-y-1">
            {Object.keys({ ...drift.baseline_dist, ...drift.recent_dist }).map((cls) => {
              const base = (drift.baseline_dist?.[cls] ?? 0) * 100
              const recent = (drift.recent_dist?.[cls] ?? 0) * 100
              return (
                <div key={cls} className="text-xs">
                  <span className="font-medium">{cls}: </span>
                  <span className="text-muted-foreground">{base.toFixed(0)}% → {recent.toFixed(0)}%</span>
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function WhatIfCard({ deployment }: { deployment: Deployment }) {
  const featureNames = deployment.feature_names ?? []
  const [baseValues, setBaseValues] = useState<Record<string, string>>({})
  const [overrideKey, setOverrideKey] = useState(featureNames[0] ?? "")
  const [overrideValue, setOverrideValue] = useState("")
  const [result, setResult] = useState<WhatIfResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (featureNames.length === 0) return null

  const handleCompare = async () => {
    if (!overrideKey || !overrideValue) return
    setLoading(true)
    setError(null)
    try {
      const base: Record<string, unknown> = {}
      featureNames.forEach((f) => { base[f] = baseValues[f] ?? "" })
      const overrides: Record<string, unknown> = { [overrideKey]: overrideValue }
      const r = await api.deploy.whatif(deployment.id, base, overrides)
      setResult(r)
    } catch {
      setError("What-if analysis failed. Check that base values are filled in.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">What-if Analysis</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Fill in your base values, then change one feature to see how the prediction shifts.
        </p>
        <div className="space-y-1.5">
          {featureNames.slice(0, 4).map((f) => (
            <div key={f} className="flex items-center gap-2 text-xs">
              <label className="w-24 shrink-0 truncate text-muted-foreground">{f}</label>
              <input
                className="flex-1 rounded border bg-background px-2 py-0.5 text-xs"
                placeholder="value"
                value={baseValues[f] ?? ""}
                onChange={(e) => setBaseValues((prev) => ({ ...prev, [f]: e.target.value }))}
              />
            </div>
          ))}
          {featureNames.length > 4 && (
            <p className="text-[10px] text-muted-foreground">+{featureNames.length - 4} more features use defaults</p>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground">Change</span>
          <select
            className="flex-1 rounded border bg-background px-2 py-0.5 text-xs"
            value={overrideKey}
            onChange={(e) => setOverrideKey(e.target.value)}
          >
            {featureNames.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
          <span className="text-muted-foreground">to</span>
          <input
            className="w-24 rounded border bg-background px-2 py-0.5 text-xs"
            placeholder="new value"
            value={overrideValue}
            onChange={(e) => setOverrideValue(e.target.value)}
          />
        </div>
        <Button size="sm" className="w-full" onClick={handleCompare} disabled={loading || !overrideKey || !overrideValue}>
          {loading ? "Comparing..." : "Compare Predictions"}
        </Button>
        {error && <p className="text-xs text-destructive">{error}</p>}
        {result && (
          <div className="rounded border bg-muted/30 p-3 space-y-1 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Original</span>
              <span className="font-mono font-medium">{String(result.original_prediction)}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Modified</span>
              <span className="font-mono font-medium">{String(result.modified_prediction)}</span>
            </div>
            {result.delta !== null && result.delta !== 0 && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Change</span>
                <span className={`font-mono font-medium ${result.delta > 0 ? "text-green-600" : "text-red-600"}`}>
                  {result.delta > 0 ? "+" : ""}{result.delta}
                  {result.percent_change !== null && ` (${result.percent_change > 0 ? "+" : ""}${result.percent_change}%)`}
                </span>
              </div>
            )}
            <p className="text-muted-foreground pt-1 border-t">{result.summary}</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function FeedbackCard({ deploymentId, problemType }: { deploymentId: string; problemType: string | null }) {
  const [accuracy, setAccuracy] = useState<FeedbackAccuracy | null>(null)
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [actualValue, setActualValue] = useState("")
  const [actualLabel, setActualLabel] = useState("")
  const [comment, setComment] = useState("")
  const [error, setError] = useState<string | null>(null)

  const loadAccuracy = useCallback(() => {
    setLoading(true)
    api.deploy.feedbackAccuracy(deploymentId)
      .then(setAccuracy)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [deploymentId])

  useEffect(() => { loadAccuracy() }, [loadAccuracy])

  const verdictColor = (v?: string) => {
    if (v === "excellent") return "text-green-600"
    if (v === "good") return "text-blue-600"
    if (v === "moderate") return "text-yellow-600"
    return "text-red-600"
  }

  const handleSubmit = async () => {
    setSubmitting(true)
    setError(null)
    try {
      const body: Record<string, unknown> = {}
      if (problemType === "regression" && actualValue) {
        const n = parseFloat(actualValue)
        if (isNaN(n)) { setError("Enter a valid number."); return }
        body.actual_value = n
      } else if (actualLabel) {
        body.actual_label = actualLabel
        body.is_correct = undefined  // let backend compute
      } else {
        setError("Enter an actual value before submitting.")
        return
      }
      if (comment) body.comment = comment
      await api.deploy.submitFeedback(deploymentId, body)
      setSubmitted(true)
      setActualValue("")
      setActualLabel("")
      setComment("")
      await loadAccuracy()
    } catch {
      setError("Failed to submit feedback.")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Real-world Accuracy</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {accuracy && accuracy.status === "computed" && (
          <div className="space-y-1">
            <p className={`text-sm font-semibold ${verdictColor(accuracy.verdict)}`}>
              {accuracy.problem_type === "regression"
                ? `${accuracy.pct_error?.toFixed(1)}% avg error (MAE ${accuracy.mae?.toFixed(3)})`
                : `${((accuracy.accuracy_from_feedback ?? 0) * 100).toFixed(0)}% real-world accuracy`}
            </p>
            <p className="text-xs text-muted-foreground">{accuracy.message}</p>
          </div>
        )}
        {accuracy && accuracy.status !== "computed" && (
          <p className="text-xs text-muted-foreground">{accuracy.message}</p>
        )}
        {loading && <p className="text-xs text-muted-foreground">Loading…</p>}

        <div className="border-t pt-3 space-y-2">
          <p className="text-xs font-medium">Record an actual outcome</p>
          {problemType === "regression" ? (
            <input
              className="w-full rounded border bg-background px-2 py-1 text-xs"
              placeholder="Actual value (e.g. 1234.56)"
              value={actualValue}
              onChange={(e) => setActualValue(e.target.value)}
            />
          ) : (
            <input
              className="w-full rounded border bg-background px-2 py-1 text-xs"
              placeholder="Actual class label (e.g. 'churned')"
              value={actualLabel}
              onChange={(e) => setActualLabel(e.target.value)}
            />
          )}
          <input
            className="w-full rounded border bg-background px-2 py-1 text-xs"
            placeholder="Optional note (e.g. 'Customer actually churned in Q3')"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
          {error && <p className="text-xs text-destructive">{error}</p>}
          {submitted && <p className="text-xs text-green-600">Feedback recorded!</p>}
          <Button size="sm" className="w-full" onClick={handleSubmit} disabled={submitting}>
            {submitting ? "Saving…" : "Submit Feedback"}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function HealthBadge({ status }: { status: ModelHealth["status"] }) {
  if (status === "healthy") return <Badge className="bg-green-100 text-green-800 border-green-200">Healthy</Badge>
  if (status === "warning") return <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200">Warning</Badge>
  return <Badge className="bg-red-100 text-red-800 border-red-200">Critical</Badge>
}

function ModelHealthCard({
  deploymentId,
  projectId,
}: {
  deploymentId: string
  projectId: string
}) {
  const [health, setHealth] = useState<ModelHealth | null>(null)
  const [retraining, setRetraining] = useState(false)
  const [retrainMessage, setRetrainMessage] = useState<string | null>(null)

  useEffect(() => {
    api.deploy.health(deploymentId)
      .then(setHealth)
      .catch(() => {})
  }, [deploymentId])

  const handleRetrain = async () => {
    setRetraining(true)
    setRetrainMessage(null)
    try {
      const result = await api.models.retrain(projectId)
      setRetrainMessage(result.message)
    } catch {
      setRetrainMessage("Retraining failed. Please try again.")
    } finally {
      setRetraining(false)
    }
  }

  if (!health) return null

  const scoreColor =
    health.health_score >= 75 ? "text-green-600" :
    health.health_score >= 50 ? "text-yellow-600" : "text-red-600"

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Model Health</CardTitle>
          <HealthBadge status={health.status} />
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-3">
          <span className={`text-3xl font-bold tabular-nums ${scoreColor}`}>
            {health.health_score}
          </span>
          <div className="text-xs text-muted-foreground">
            <div>out of 100</div>
            <div>{health.model_age_days} day(s) old</div>
          </div>
        </div>

        <div className="space-y-1">
          {[
            { label: "Freshness", score: health.component_scores.age, note: health.component_notes.age },
            health.has_feedback_data
              ? { label: "Real-world accuracy", score: health.component_scores.feedback, note: health.component_notes.feedback }
              : null,
            health.has_drift_data
              ? { label: "Distribution stability", score: health.component_scores.drift, note: health.component_notes.drift }
              : null,
          ]
            .filter(Boolean)
            .map((item) => (
              <div key={item!.label} className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">{item!.label}</span>
                <span className={
                  (item!.score ?? 100) >= 75 ? "font-medium text-green-600" :
                  (item!.score ?? 100) >= 50 ? "font-medium text-yellow-600" : "font-medium text-red-600"
                }>
                  {item!.score ?? "—"}/100
                </span>
              </div>
            ))}
        </div>

        {health.recommendations.length > 0 && (
          <div className="border-t pt-2 space-y-1">
            {health.recommendations.map((rec, i) => (
              <p key={i} className="text-xs text-muted-foreground">· {rec}</p>
            ))}
          </div>
        )}

        {retrainMessage && (
          <p className="text-xs text-blue-600">{retrainMessage}</p>
        )}

        <Button
          size="sm"
          variant="outline"
          className="w-full"
          onClick={handleRetrain}
          disabled={retraining}
        >
          {retraining ? "Starting retrain…" : "Retrain Model"}
        </Button>
      </CardContent>
    </Card>
  )
}

export function DeploymentPanel({
  projectId,
  selectedRunId,
  algorithmName,
  onDeployed,
}: DeploymentPanelProps) {
  const [deploying, setDeploying] = useState(false)
  const [deployment, setDeployment] = useState<Deployment | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [undeploying, setUndeploying] = useState(false)
  const [copied, setCopied] = useState(false)
  const [readiness, setReadiness] = useState<ModelReadiness | null>(null)
  const [analytics, setAnalytics] = useState<DeploymentAnalytics | null>(null)
  const [drift, setDrift] = useState<DriftReport | null>(null)

  // Load readiness check when a run is selected
  useEffect(() => {
    if (!selectedRunId) {
      setReadiness(null)
      return
    }
    api.models.readiness(selectedRunId)
      .then(setReadiness)
      .catch(() => {/* not ready yet — ignore */})
  }, [selectedRunId])

  // Load analytics + drift when deployed
  useEffect(() => {
    if (!deployment) {
      setAnalytics(null)
      setDrift(null)
      return
    }
    api.deploy.analytics(deployment.id)
      .then(setAnalytics)
      .catch(() => {})
    api.deploy.drift(deployment.id)
      .then(setDrift)
      .catch(() => {})
  }, [deployment])

  const handleCopyLink = useCallback((url: string) => {
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }).catch(() => {})
  }, [])

  const handleDeploy = async () => {
    if (!selectedRunId) return
    setDeploying(true)
    setError(null)
    try {
      const result = await api.deploy.deploy(selectedRunId)
      setDeployment(result)
      onDeployed?.(result)
    } catch (e) {
      setError("Deployment failed. Please try again.")
      console.error(e)
    } finally {
      setDeploying(false)
    }
  }

  const handleUndeploy = async () => {
    if (!deployment) return
    setUndeploying(true)
    try {
      await api.deploy.undeploy(deployment.id)
      setDeployment(null)
    } catch (e) {
      console.error(e)
    } finally {
      setUndeploying(false)
    }
  }

  if (!selectedRunId) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-center">
        <p className="text-sm text-muted-foreground">
          Select a model in the <strong>Models</strong> tab before deploying.
        </p>
      </div>
    )
  }

  if (deployment) {
    const dashboardUrl = `${typeof window !== "undefined" ? window.location.origin : ""}${deployment.dashboard_url}`
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-green-500" />
          <span className="text-sm font-medium text-green-700 dark:text-green-400">
            Model deployed
          </span>
          <Badge variant="outline" className="ml-auto">
            {deployment.algorithm ?? "Unknown"}
          </Badge>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Live Prediction Dashboard</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div>
              <p className="text-xs font-medium text-muted-foreground">Dashboard URL</p>
              <div className="mt-0.5 flex items-center gap-2">
                <a
                  href={deployment.dashboard_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-1 truncate text-primary underline-offset-2 hover:underline text-xs"
                >
                  {dashboardUrl}
                </a>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-6 shrink-0 px-2 text-[10px]"
                  onClick={() => handleCopyLink(dashboardUrl)}
                >
                  {copied ? "Copied!" : "Copy link"}
                </Button>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                Share this link — anyone can paste in values and see predictions.
              </p>
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground">API Endpoint</p>
              <code className="mt-0.5 block rounded bg-muted px-2 py-1 text-xs">
                POST {typeof window !== "undefined" ? `${window.location.origin}/api` : "http://localhost:8000"}
                {deployment.endpoint_path}
              </code>
              <p className="mt-1 text-xs text-muted-foreground">
                Send JSON with feature values, get back a prediction.
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>Requests: {deployment.request_count}</span>
              {deployment.last_predicted_at && (
                <span>
                  · Last used:{" "}
                  {new Date(deployment.last_predicted_at).toLocaleDateString()}
                </span>
              )}
            </div>
          </CardContent>
        </Card>

        {analytics && <AnalyticsCard analytics={analytics} />}
        {drift && <DriftCard drift={drift} />}
        {deployment && <WhatIfCard deployment={deployment} />}
        {deployment && <FeedbackCard deploymentId={deployment.id} problemType={deployment.problem_type} />}
        {deployment && <ModelHealthCard deploymentId={deployment.id} projectId={projectId} />}

        <div className="flex justify-end">
          <Button
            variant="outline"
            size="sm"
            onClick={handleUndeploy}
            disabled={undeploying}
          >
            {undeploying ? "Undeploying..." : "Undeploy"}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {readiness && <ReadinessCard readiness={readiness} />}

      <div className="rounded-lg border p-4">
        <h4 className="text-sm font-semibold">Ready to deploy</h4>
        <p className="mt-1 text-xs text-muted-foreground">
          Deploying <strong>{algorithmName ?? "selected model"}</strong> will create a live
          prediction API endpoint and an interactive dashboard you can share with anyone.
          No code required.
        </p>
        <ul className="mt-3 space-y-1 text-xs text-muted-foreground">
          <li>✓ Auto-generated prediction form with your feature columns</li>
          <li>✓ JSON API endpoint for developers</li>
          <li>✓ Batch prediction via CSV upload</li>
          <li>✓ Usage analytics dashboard</li>
        </ul>
      </div>

      {error && (
        <p className="text-xs text-destructive">{error}</p>
      )}

      <Button
        onClick={handleDeploy}
        disabled={deploying}
        className="w-full"
      >
        {deploying ? "Deploying..." : "Deploy Model"}
      </Button>
    </div>
  )
}
