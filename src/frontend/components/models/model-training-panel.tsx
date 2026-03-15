"use client"

import { useState, useEffect, useCallback } from "react"
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import type { ChartSpec, ModelRecommendation, ModelRun, ModelComparison, ModelMetrics, TuningResult } from "@/lib/types"

interface ModelTrainingPanelProps {
  projectId: string
  onModelSelected?: (runId: string, algorithm: string) => void
  onModelDownload?: (runId: string) => void
  onModelReport?: (runId: string) => void
}

const ALGORITHM_DISPLAY: Record<string, string> = {
  linear_regression: "Linear Regression",
  random_forest_regressor: "Random Forest",
  gradient_boosting_regressor: "Gradient Boosting",
  logistic_regression: "Logistic Regression",
  random_forest_classifier: "Random Forest",
  gradient_boosting_classifier: "Gradient Boosting",
}

export function ModelTrainingPanel({ projectId, onModelSelected, onModelDownload, onModelReport }: ModelTrainingPanelProps) {
  const [recommendations, setRecommendations] = useState<ModelRecommendation[]>([])
  const [problemType, setProblemType] = useState("")
  const [targetColumn, setTargetColumn] = useState("")
  const [selectedAlgos, setSelectedAlgos] = useState<Set<string>>(new Set())
  const [runs, setRuns] = useState<ModelRun[]>([])
  const [comparison, setComparison] = useState<ModelComparison | null>(null)
  const [radarChart, setRadarChart] = useState<ChartSpec | null>(null)
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tuningRunId, setTuningRunId] = useState<string | null>(null)
  const [tuningResults, setTuningResults] = useState<Record<string, TuningResult>>({})

  // Load recommendations and any existing runs on mount
  useEffect(() => {
    Promise.all([
      api.models.recommendations(projectId),
      api.models.runs(projectId).catch(() => ({ runs: [] })),
    ])
      .then(([recData, runsData]) => {
        setRecommendations(recData.recommendations)
        setProblemType(recData.problem_type)
        setTargetColumn(recData.target_column)
        // Pre-select the first 2 algorithms
        const defaults = recData.recommendations.slice(0, 2).map((r) => r.algorithm)
        setSelectedAlgos(new Set(defaults))
        // Restore any runs from a previous training session
        if (runsData.runs && runsData.runs.length > 0) {
          setRuns(runsData.runs)
          // Load comparison summary and radar if there are completed runs
          const hasDone = runsData.runs.some((r: ModelRun) => r.status === "done")
          if (hasDone) {
            Promise.all([
              api.models.compare(projectId),
              api.models.comparisonRadar(projectId),
            ])
              .then(([cmp, radar]) => {
                setComparison(cmp)
                setRadarChart(radar?.chart ?? null)
              })
              .catch(() => {})
          }
        }
      })
      .catch((e) => setError(e?.message ?? "Could not load recommendations"))
      .finally(() => setLoading(false))
  }, [projectId])

  // Subscribe to SSE training stream while any runs are in progress
  useEffect(() => {
    const inProgress = runs.some((r) => r.status === "pending" || r.status === "training")
    if (!inProgress) return

    const es = new EventSource(api.models.trainingStreamUrl(projectId))

    es.onmessage = async (e) => {
      try {
        const event = JSON.parse(e.data)
        if (event.type === "all_done") {
          es.close()
          const [data, cmp, radar] = await Promise.all([
            api.models.runs(projectId),
            api.models.compare(projectId),
            api.models.comparisonRadar(projectId),
          ])
          setRuns(data.runs)
          setComparison(cmp)
          setRadarChart(radar?.chart ?? null)
        } else if (event.type === "status" || event.type === "done" || event.type === "failed") {
          setRuns((prev) =>
            prev.map((r) =>
              r.id === event.run_id
                ? {
                    ...r,
                    status: event.status,
                    metrics: event.metrics ?? r.metrics,
                    summary: event.summary ?? r.summary,
                    training_duration_ms: event.training_duration_ms ?? r.training_duration_ms,
                    error_message: event.error ?? r.error_message,
                  }
                : r
            )
          )
        }
      } catch {
        // malformed event — ignore
      }
    }

    es.onerror = () => {
      es.close()
      api.models.runs(projectId).then((d) => setRuns(d.runs)).catch(() => {})
    }

    return () => es.close()
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
    async (runId: string, algorithm: string) => {
      try {
        await api.models.select(runId)
        const data = await api.models.runs(projectId)
        setRuns(data.runs)
        onModelSelected?.(runId, algorithm)
      } catch {
        // silent
      }
    },
    [projectId, onModelSelected]
  )

  const handleTune = useCallback(
    async (runId: string) => {
      setTuningRunId(runId)
      try {
        const result = await api.models.tune(runId)
        setTuningResults((prev) => ({ ...prev, [runId]: result }))
        // Refresh runs to include the newly created tuned run
        const data = await api.models.runs(projectId)
        setRuns(data.runs)
      } catch {
        // silent — TuningCard will show error state
      } finally {
        setTuningRunId(null)
      }
    },
    [projectId]
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
              <div key={run.id}>
                <RunCard
                  run={run}
                  problemType={problemType}
                  isRecommended={comparison?.recommendation?.model_run_id === run.id}
                  onSelect={() => handleSelect(run.id, run.algorithm)}
                  onDownload={onModelDownload ? () => onModelDownload(run.id) : undefined}
                  onReport={onModelReport ? () => onModelReport(run.id) : undefined}
                  onTune={() => handleTune(run.id)}
                  isTuning={tuningRunId === run.id}
                />
                {tuningResults[run.id] && (
                  <TuningCard result={tuningResults[run.id]} problemType={problemType} />
                )}
              </div>
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

      {/* Radar chart — only when 2+ models are compared */}
      {radarChart && radarChart.data.length > 0 && (
        <ModelRadarChart chart={radarChart} />
      )}
    </div>
  )
}


// Palette for radar polygons — one color per model
const RADAR_COLORS = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]

function ModelRadarChart({ chart }: { chart: ChartSpec }) {
  return (
    <div>
      <h4 className="mb-2 text-xs font-semibold">{chart.title}</h4>
      <p className="mb-2 text-[11px] text-muted-foreground">
        Normalized 0–1 (higher = better on every axis)
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <RadarChart data={chart.data}>
          <PolarGrid />
          <PolarAngleAxis dataKey={chart.x_key} tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ fontSize: 11 }}
            formatter={(v) => (typeof v === "number" ? `${(v * 100).toFixed(0)}%` : String(v))}
          />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {chart.y_keys.map((key, i) => (
            <Radar
              key={key}
              name={key}
              dataKey={key}
              stroke={RADAR_COLORS[i % RADAR_COLORS.length]}
              fill={RADAR_COLORS[i % RADAR_COLORS.length]}
              fillOpacity={0.15}
            />
          ))}
        </RadarChart>
      </ResponsiveContainer>
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
  onDownload,
  onReport,
  onTune,
  isTuning,
}: {
  run: ModelRun
  problemType: string
  isRecommended: boolean
  onSelect: () => void
  onDownload?: () => void
  onReport?: () => void
  onTune?: () => void
  isTuning?: boolean
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
            <div className="mt-2 flex flex-wrap gap-2">
              {!run.is_selected && (
                <Button size="sm" variant="outline" className="h-7 text-xs" onClick={onSelect}>
                  Select this model
                </Button>
              )}
              {onTune && (
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs"
                  onClick={onTune}
                  disabled={isTuning}
                >
                  {isTuning ? "Tuning..." : "Auto-Tune"}
                </Button>
              )}
              {onDownload && (
                <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={onDownload}>
                  Download .joblib
                </Button>
              )}
              {onReport && (
                <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={onReport}>
                  Download Report
                </Button>
              )}
            </div>
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


function TuningCard({ result, problemType }: { result: TuningResult; problemType: string }) {
  if (!result.tunable) {
    return (
      <div className="mt-1 rounded border border-muted px-3 py-2 text-xs text-muted-foreground">
        {result.summary}
      </div>
    )
  }

  const primary = problemType === "regression" ? "r2" : "accuracy"
  const origVal = result.original_metrics?.[primary]
  const tunedVal = result.tuned_metrics?.[primary]
  const improved = result.improved

  return (
    <Card className={`mt-1 ${improved ? "border-blue-200 bg-blue-50 dark:border-blue-900 dark:bg-blue-950" : "border-muted"}`}>
      <CardHeader className="pb-1">
        <CardTitle className="text-xs">
          Auto-Tune Result
          {improved && (
            <Badge className="ml-2 bg-blue-100 text-blue-800 border-blue-200 text-[10px]">
              Improved +{result.improvement_pct?.toFixed(1)}%
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1">
        <p className="text-xs text-muted-foreground">{result.summary}</p>
        {origVal != null && tunedVal != null && (
          <div className="flex gap-4 text-xs mt-1">
            <span>
              Before:{" "}
              <strong>
                {primary === "accuracy"
                  ? `${(origVal * 100).toFixed(1)}%`
                  : origVal.toFixed(3)}
              </strong>
            </span>
            <span>
              After:{" "}
              <strong className={improved ? "text-blue-700 dark:text-blue-300" : ""}>
                {primary === "accuracy"
                  ? `${(tunedVal * 100).toFixed(1)}%`
                  : tunedVal.toFixed(3)}
              </strong>
            </span>
          </div>
        )}
        {result.best_params && Object.keys(result.best_params).length > 0 && (
          <div className="mt-1 text-[10px] text-muted-foreground/70">
            Best params:{" "}
            {Object.entries(result.best_params)
              .map(([k, v]) => `${k}=${v}`)
              .join(", ")}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
