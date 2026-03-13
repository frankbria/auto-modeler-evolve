"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { api } from "@/lib/api"
import type { Deployment, FeatureSchemaEntry, PredictResult } from "@/lib/types"

export default function PredictionDashboard() {
  const params = useParams<{ id: string }>()
  const deploymentId = params.id

  const [deployment, setDeployment] = useState<Deployment | null>(null)
  const [loading, setLoading] = useState(true)
  const [inputs, setInputs] = useState<Record<string, string>>({})
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.deploy
      .get(deploymentId)
      .then((dep) => {
        setDeployment(dep)
        // Initialize inputs from feature schema
        const initial: Record<string, string> = {}
        dep.feature_schema.forEach((f) => {
          initial[f.name] = ""
        })
        setInputs(initial)
      })
      .catch(() => setError("Could not load this prediction dashboard."))
      .finally(() => setLoading(false))
  }, [deploymentId])

  async function handlePredict() {
    if (!deployment) return
    setError(null)
    setPredicting(true)
    try {
      // Coerce numeric inputs to numbers
      const coerced: Record<string, string | number | null> = {}
      deployment.feature_schema.forEach((f) => {
        const raw = inputs[f.name]
        if (raw === "" || raw == null) {
          coerced[f.name] = null
        } else if (f.dtype === "numeric") {
          const num = parseFloat(raw)
          coerced[f.name] = isNaN(num) ? null : num
        } else {
          coerced[f.name] = raw
        }
      })
      const res = await api.deploy.predict(deploymentId, coerced)
      setResult(res)
    } catch {
      setError("Prediction failed. Please check your inputs and try again.")
    } finally {
      setPredicting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-sm text-muted-foreground">Loading prediction dashboard...</p>
      </div>
    )
  }

  if (error && !deployment) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Card className="max-w-sm">
          <CardContent className="pt-6">
            <p className="text-center text-sm text-muted-foreground">{error}</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!deployment) return null

  const numericFeatures = deployment.feature_schema.filter((f) => f.dtype === "numeric")
  const categoricalFeatures = deployment.feature_schema.filter((f) => f.dtype === "categorical")

  return (
    <div className="mx-auto max-w-2xl px-4 py-10">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold">
            {deployment.display_name ?? "Prediction Dashboard"}
          </h1>
          <Badge variant="outline" className="border-green-500 text-green-700 dark:text-green-300">
            Live
          </Badge>
        </div>
        {deployment.target_column && (
          <p className="mt-1 text-sm text-muted-foreground">
            Predicting{" "}
            <strong className="text-foreground">{deployment.target_column}</strong>
            {deployment.problem_type && (
              <> · {deployment.problem_type}</>
            )}
          </p>
        )}
      </div>

      {/* Input Form */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-sm">Enter values to predict</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {numericFeatures.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Numeric Features
              </p>
              <div className="grid grid-cols-2 gap-3">
                {numericFeatures.map((f) => (
                  <FeatureInput
                    key={f.name}
                    feature={f}
                    value={inputs[f.name] ?? ""}
                    onChange={(v) => setInputs((prev) => ({ ...prev, [f.name]: v }))}
                  />
                ))}
              </div>
            </div>
          )}
          {categoricalFeatures.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Categorical Features
              </p>
              <div className="grid grid-cols-2 gap-3">
                {categoricalFeatures.map((f) => (
                  <FeatureInput
                    key={f.name}
                    feature={f}
                    value={inputs[f.name] ?? ""}
                    onChange={(v) => setInputs((prev) => ({ ...prev, [f.name]: v }))}
                  />
                ))}
              </div>
            </div>
          )}

          {error && (
            <p className="text-xs text-red-600 dark:text-red-400">{error}</p>
          )}

          <Button
            className="w-full"
            onClick={handlePredict}
            disabled={predicting}
          >
            {predicting ? "Predicting..." : "Get Prediction"}
          </Button>
        </CardContent>
      </Card>

      {/* Result */}
      {result && (
        <Card className="border-primary/20 bg-primary/5">
          <CardHeader>
            <CardTitle className="text-sm">Result</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-baseline gap-3">
              <span className="text-3xl font-bold">
                {typeof result.prediction === "number"
                  ? result.prediction.toLocaleString(undefined, { maximumFractionDigits: 4 })
                  : result.prediction}
              </span>
              {result.target_column && (
                <span className="text-sm text-muted-foreground">{result.target_column}</span>
              )}
            </div>

            {result.probability != null && (
              <div className="text-sm text-muted-foreground">
                Confidence:{" "}
                <span className="font-medium text-foreground">
                  {(result.probability * 100).toFixed(1)}%
                </span>
              </div>
            )}

            <p className="text-xs text-muted-foreground">
              Model: {result.algorithm?.replace(/_/g, " ") ?? "Unknown"}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Footer */}
      <p className="mt-8 text-center text-xs text-muted-foreground">
        Powered by AutoModeler · {deployment.request_count} predictions served
      </p>
    </div>
  )
}

function FeatureInput({
  feature,
  value,
  onChange,
}: {
  feature: FeatureSchemaEntry
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div>
      <label className="mb-1 block text-xs font-medium text-foreground">
        {feature.name.replace(/_/g, " ")}
      </label>
      <Input
        type={feature.dtype === "numeric" ? "number" : "text"}
        placeholder={feature.dtype === "numeric" ? "0" : "value"}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="text-xs"
      />
    </div>
  )
}
