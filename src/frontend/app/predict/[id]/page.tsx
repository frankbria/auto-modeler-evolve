"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { api } from "@/lib/api"
import type { Deployment, FeatureSchemaEntry, PredictionResult, PredictionExplanation, FeatureContribution } from "@/lib/types"

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
        {item.feature.replace(/_/g, " ")}
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
  const [history, setHistory] = useState<PredictionHistoryRecord[]>([])
  const [historyCounter, setHistoryCounter] = useState(0)

  useEffect(() => {
    api.deploy
      .get(deploymentId)
      .then((d) => {
        setDeployment(d)
        // Pre-fill inputs with default values
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

  const buildPayload = () => {
    if (!deployment) return {}
    const payload: Record<string, unknown> = {}
    for (const entry of deployment.feature_schema ?? []) {
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
    const payload = buildPayload()
    try {
      const exp = await api.deploy.explain(deploymentId, payload)
      setExplanation(exp)
      setShowExplanation(true)
    } catch {
      // Silently fall back — explanation is optional enhancement
    } finally {
      setFetchingExplanation(false)
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-sm text-muted-foreground">Loading prediction service...</p>
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

  const schema = deployment.feature_schema ?? []

  return (
    <div className="min-h-screen bg-background px-4 py-8">
      <div className="mx-auto max-w-xl space-y-6">
        {/* Header */}
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold">Prediction Dashboard</h1>
            <Badge variant="outline">{deployment.algorithm}</Badge>
            <Badge
              variant="secondary"
              className="ml-auto capitalize"
            >
              {deployment.problem_type}
            </Badge>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Fill in the values below to get a prediction for{" "}
            <strong>{deployment.target_column}</strong>.
          </p>
        </div>

        {/* Input form */}
        <Card>
          <CardHeader>
            <CardTitle>Input Features</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {schema.length === 0 && (
              <p className="text-xs text-muted-foreground">
                No feature schema available for this deployment.
              </p>
            )}
            {schema.map((entry: FeatureSchemaEntry) => (
              <div key={entry.name}>
                <label className="mb-1 block text-xs font-medium capitalize">
                  {entry.name.replace(/_/g, " ")}
                  <span className="ml-1 font-normal text-muted-foreground">
                    ({entry.type})
                  </span>
                </label>
                {entry.type === "categorical" && entry.options ? (
                  <select
                    className="w-full rounded-md border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                    value={inputs[entry.name] ?? ""}
                    onChange={(e) =>
                      setInputs((prev) => ({ ...prev, [entry.name]: e.target.value }))
                    }
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
                    placeholder={entry.median != null ? `default: ${entry.median}` : ""}
                    value={inputs[entry.name] ?? ""}
                    onChange={(e) =>
                      setInputs((prev) => ({ ...prev, [entry.name]: e.target.value }))
                    }
                    className="text-sm"
                  />
                )}
              </div>
            ))}
          </CardContent>
        </Card>

        <Button
          onClick={handlePredict}
          disabled={predicting}
          className="w-full"
          size="lg"
        >
          {predicting ? "Predicting..." : "Get Prediction"}
        </Button>

        {predError && (
          <p className="text-sm text-destructive">{predError}</p>
        )}

        {/* Result */}
        {result && (
          <Card className="border-primary/30 bg-primary/5">
            <CardHeader>
              <CardTitle className="text-base">
                Prediction for <em>{result.target_column}</em>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="rounded-lg bg-background p-4 text-center">
                <p className="text-3xl font-bold tabular-nums">
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

              {result.probabilities && (
                <div>
                  <p className="mb-2 text-xs font-medium">Class Probabilities</p>
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

              {/* Explain button */}
              <Button
                size="sm"
                variant="outline"
                className="w-full"
                onClick={handleExplain}
                disabled={fetchingExplanation}
              >
                {fetchingExplanation
                  ? "Analyzing..."
                  : showExplanation
                  ? "Hide explanation"
                  : "Why this prediction?"}
              </Button>
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
                      ...schema.map((e) => e.name),
                      "Prediction",
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
                    a.download = `predictions-session.csv`
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
                      <th className="px-3 py-1.5 text-right font-medium text-muted-foreground">Prediction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((rec) => (
                      <tr key={rec.id} className="border-b last:border-0 hover:bg-muted/20">
                        <td className="px-3 py-1.5 tabular-nums text-muted-foreground">{rec.id}</td>
                        <td className="px-3 py-1.5 text-muted-foreground">{rec.timestamp}</td>
                        <td className="px-3 py-1.5 text-right font-medium tabular-nums">
                          {typeof rec.prediction === "number"
                            ? rec.prediction.toLocaleString(undefined, { maximumFractionDigits: 4 })
                            : String(rec.prediction)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        <p className="text-center text-xs text-muted-foreground">
          Powered by AutoModeler · {deployment.request_count} predictions served
        </p>
      </div>
    </div>
  )
}
