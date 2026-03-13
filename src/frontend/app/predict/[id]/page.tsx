"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { api } from "@/lib/api"
import type { Deployment, FeatureSchemaEntry, PredictionResult } from "@/lib/types"

export default function PredictionDashboard() {
  const params = useParams<{ id: string }>()
  const deploymentId = params.id

  const [deployment, setDeployment] = useState<Deployment | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [inputs, setInputs] = useState<Record<string, string>>({})
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [predError, setPredError] = useState<string | null>(null)

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

  const handlePredict = async () => {
    if (!deployment) return
    setPredicting(true)
    setPredError(null)
    setResult(null)

    // Coerce numeric inputs
    const payload: Record<string, unknown> = {}
    for (const entry of deployment.feature_schema ?? []) {
      const raw = inputs[entry.name] ?? ""
      if (entry.type === "numeric") {
        payload[entry.name] = raw === "" ? null : parseFloat(raw)
      } else {
        payload[entry.name] = raw
      }
    }

    try {
      const r = await api.deploy.predict(deploymentId, payload)
      setResult(r)
    } catch {
      setPredError("Prediction failed. Please check your inputs and try again.")
    } finally {
      setPredicting(false)
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
