"use client"

import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import type { ModelRecommendation, ModelRun, ModelComparison, ModelMetrics } from "@/lib/types"

interface ModelTrainingPanelProps {
  projectId: string
  onModelSelected?: (runId: string) => void
}

const ALGORITHM_DISPLAY: Record<string, string> = {
  linear_regression: "Linear Regression",
  random_forest_regressor: "Random Forest",
  gradient_boosting_regressor: "Gradient Boosting",
  logistic_regression: "Logistic Regression",
  random_forest_classifier: "Random Forest",
  gradient_boosting_classifier: "Gradient Boosting",
}

export function ModelTrainingPanel({ projectId, onModelSelected }: ModelTrainingPanelProps) {
  const [recommendations, setRecommendations] = useState<ModelRecommendation[]>([])
  const [problemType, setProblemType] = useState("")
  const [targetColumn, setTargetColumn] = useState("")
  const [selectedAlgos, setSelectedAlgos] = useState<Set<string>>(new Set())
  const [runs, setRuns] = useState<ModelRun[]>([])
  const [comparison, setComparison] = useState<ModelComparison | null>(null)
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Load recommendations on mount
  useEffect(() => {
    api.models
      .recommendations(projectId)
      .then((data) => {
        setRecommendations(data.recommendations)
        setProblemType(data.problem_type)
        setTargetColumn(data.target_column)
        // Pre-select the first 2 algorithms
        const defaults = data.recommendations.slice(0, 2).map((r) => r.algorithm)
        setSelectedAlgos(new Set(defaults))
      })
      .catch((e) => setError(e?.message ?? "Could not load recommendations"))
      .finally(() => setLoading(false))
  }, [projectId])

  // Poll for run status while any are in progress
  useEffect(() => {
    const inProgress = runs.some((r) => r.status === "pending" || r.status === "training")
    if (!inProgress) return

    const interval = setInterval(async () => {
      try {
        const data = await api.models.runs(projectId)
        setRuns(data.runs)

        // When all done, load comparison
        const allDone = data.runs.every((r) => r.status === "done" || r.status === "failed")
        if (allDone) {
          const cmp = await api.models.compare(projectId)
          setComparison(cmp)
        }
      } catch {
        // silent
      }
    }, 1500)

    return () => clearInterval(interval)
  }, [runs, projectId])

  const toggleAlgo = useCallback((algo: string) => {
    setSelectedAlgos((prev) => {
      const next = new Set(prev)
      if (next.has(algo)) {
        next.delete(algo)
      } else {
        next.add(algo)
      }
      return next
    })
  }, [])

  const handleTrain = useCallback(async () => {
    if (selectedAlgos.size === 0) return
    setTraining(true)
    setError(null)
    try {
      await api.models.train(projectId, Array.from(selectedAlgos))
      const data = await api.models.runs(projectId)
      setRuns(data.runs)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Training failed to start")
    } finally {
      setTraining(false)
    }
  }, [projectId, selectedAlgos])

  const handleSelect = useCallback(
    async (runId: string) => {
      try {
        await api.models.select(runId)
        const data = await api.models.runs(projectId)
        setRuns(data.runs)
        onModelSelected?.(runId)
      } catch {
        // silent
      }
    },
    [projectId, onModelSelected]
  )

  if (loading) {
    return <p className="text-xs text-muted-foreground p-4">Loading model recommendations...</p>
  }

  if (error && recommendations.length === 0) {
    return (
      <div className="p-4 text-xs text-destructive">
        <p className="font-medium">Cannot load recommendations</p>
        <p className="mt-1 text-muted-foreground">{error}</p>
        <p className="mt-2 text-muted-foreground">
          Make sure you have uploaded a dataset and set a target column in the Features tab.
        </p>
      </div>
    )
  }

  const anyTraining = runs.some((r) => r.status === "pending" || r.status === "training")

  return (
    <div className="flex flex-col gap-5">
      {/* Context summary */}
      {targetColumn && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-muted-foreground">Predicting:</span>
          <Badge variant="secondary">{targetColumn}</Badge>
          <Badge variant="outline">{problemType}</Badge>
        </div>
      )}

      {/* Algorithm selection */}
      {runs.length === 0 && (
        <div>
          <h4 className="text-xs font-semibold mb-2">Select algorithms to train</h4>
          <div className="flex flex-col gap-2">
            {recommendations.map((rec) => (
              <AlgorithmCard
                key={rec.algorithm}
                rec={rec}
                selected={selectedAlgos.has(rec.algorithm)}
                onToggle={() => toggleAlgo(rec.algorithm)}
              />
            ))}
          </div>

          <Button
            className="mt-4 w-full"
            onClick={handleTrain}
            disabled={selectedAlgos.size === 0 || training}
          >
            {training ? "Starting..." : `Train ${selectedAlgos.size} model${selectedAlgos.size !== 1 ? "s" : ""}`}
          </Button>
          {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
        </div>
      )}

      {/* Training progress */}
      {runs.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-semibold">Training runs</h4>
            {!anyTraining && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setRuns([])
                  setComparison(null)
                }}
                className="text-xs h-6"
              >
                Train more
              </Button>
            )}
          </div>
          <div className="flex flex-col gap-2">
            {runs.map((run) => (
              <RunCard
                key={run.id}
                run={run}
                problemType={problemType}
                isRecommended={comparison?.recommendation?.model_run_id === run.id}
                onSelect={() => handleSelect(run.id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Comparison summary */}
      {comparison && comparison.models.length > 0 && comparison.recommendation && (
        <div className="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-xs dark:border-green-900 dark:bg-green-950">
          <p className="font-semibold text-green-800 dark:text-green-200">
            Recommendation: {ALGORITHM_DISPLAY[comparison.recommendation.algorithm] ?? comparison.recommendation.algorithm}
          </p>
          <p className="mt-0.5 text-green-700 dark:text-green-300">
            {comparison.recommendation.reason}
          </p>
        </div>
      )}
    </div>
  )
}


function AlgorithmCard({
  rec,
  selected,
  onToggle,
}: {
  rec: ModelRecommendation
  selected: boolean
  onToggle: () => void
}) {
  return (
    <button
      onClick={onToggle}
      className={`rounded-lg border px-3 py-2 text-left text-xs transition-colors ${
        selected
          ? "border-primary bg-primary/5 text-foreground"
          : "border-border bg-card text-card-foreground hover:border-muted-foreground/40"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="font-medium">{rec.name}</span>
        {selected && (
          <span className="shrink-0 rounded bg-primary px-1 py-0.5 text-[10px] text-primary-foreground">
            Selected
          </span>
        )}
      </div>
      <p className="mt-0.5 text-muted-foreground">{rec.plain_english}</p>
      <p className="mt-1 text-muted-foreground/70">{rec.recommended_because}</p>
    </button>
  )
}


function RunCard({
  run,
  problemType,
  isRecommended,
  onSelect,
}: {
  run: ModelRun
  problemType: string
  isRecommended: boolean
  onSelect: () => void
}) {
  const displayName = ALGORITHM_DISPLAY[run.algorithm] ?? run.algorithm

  const statusBadge = {
    pending: <Badge variant="secondary">Queued</Badge>,
    training: <Badge variant="secondary" className="animate-pulse">Training...</Badge>,
    done: <Badge variant="outline" className="border-green-500 text-green-700 dark:text-green-400">Done</Badge>,
    failed: <Badge variant="destructive">Failed</Badge>,
  }[run.status]

  return (
    <Card size="sm" className={isRecommended ? "border-green-300 dark:border-green-700" : undefined}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span>{displayName}</span>
          {statusBadge}
          {run.is_selected && (
            <Badge variant="default" className="ml-auto">Selected</Badge>
          )}
          {isRecommended && !run.is_selected && (
            <span className="ml-auto text-[10px] text-green-600 dark:text-green-400">Recommended</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {run.status === "done" && run.metrics && (
          <div className="space-y-1">
            {run.summary && (
              <p className="text-xs text-muted-foreground">{run.summary}</p>
            )}
            <MetricsRow metrics={run.metrics} problemType={problemType} />
            {run.training_duration_ms != null && (
              <p className="text-[10px] text-muted-foreground/60">
                Trained in {(run.training_duration_ms / 1000).toFixed(1)}s
              </p>
            )}
            {!run.is_selected && (
              <Button size="sm" variant="outline" className="mt-2 h-7 text-xs" onClick={onSelect}>
                Select this model
              </Button>
            )}
          </div>
        )}
        {run.status === "failed" && run.error_message && (
          <p className="text-xs text-destructive">{run.error_message}</p>
        )}
        {(run.status === "pending" || run.status === "training") && (
          <p className="text-xs text-muted-foreground animate-pulse">
            {run.status === "pending" ? "Waiting to start..." : "Training model..."}
          </p>
        )}
      </CardContent>
    </Card>
  )
}


function MetricsRow({ metrics, problemType }: { metrics: ModelMetrics; problemType: string }) {
  const m = metrics as unknown as Record<string, number>
  if (problemType === "regression") {
    return (
      <div className="flex gap-3 text-xs">
        <span>R² <strong>{m.r2?.toFixed(3) ?? "—"}</strong></span>
        <span>MAE <strong>{m.mae?.toFixed(2) ?? "—"}</strong></span>
        <span>RMSE <strong>{m.rmse?.toFixed(2) ?? "—"}</strong></span>
      </div>
    )
  }
  return (
    <div className="flex gap-3 text-xs">
      <span>Accuracy <strong>{m.accuracy != null ? `${(m.accuracy * 100).toFixed(1)}%` : "—"}</strong></span>
      <span>F1 <strong>{m.f1?.toFixed(3) ?? "—"}</strong></span>
      <span>Precision <strong>{m.precision?.toFixed(3) ?? "—"}</strong></span>
    </div>
  )
}
