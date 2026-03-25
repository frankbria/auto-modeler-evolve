"use client"

import { useState, useEffect } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ImportanceBar } from "@/components/ui/importance-bar"
import type {
  FeatureSuggestion,
  FeatureSetResult,
  FeatureImportanceEntry,
  DatasetListItem,
  JoinKeySuggestion,
  MergeResponse,
} from "@/lib/types"
import { api } from "@/lib/api"

const TRANSFORM_LABELS: Record<FeatureSuggestion["transform_type"], string> = {
  date_decompose: "Date Parts",
  log_transform: "Log Transform",
  one_hot: "One-Hot Encode",
  label_encode: "Label Encode",
  bin_quartile: "Bin into Quartiles",
  interaction: "Interaction Term",
}

const TRANSFORM_COLORS: Record<FeatureSuggestion["transform_type"], string> = {
  date_decompose: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  log_transform: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  one_hot: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  label_encode: "bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-200",
  bin_quartile: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200",
  interaction: "bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-200",
}

interface Props {
  datasetId: string
  suggestions: FeatureSuggestion[]
  onApplied?: (result: FeatureSetResult) => void
}

export function FeatureSuggestionsPanel({ datasetId, suggestions, onApplied }: Props) {
  const [approved, setApproved] = useState<Set<string>>(new Set())
  const [applying, setApplying] = useState(false)
  const [result, setResult] = useState<FeatureSetResult | null>(null)

  function toggleApprove(id: string) {
    setApproved((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  async function handleApply() {
    const selected = suggestions.filter((s) => approved.has(s.id))
    if (selected.length === 0) return

    setApplying(true)
    try {
      const transforms = selected.map((s) => ({
        column: s.column,
        transform_type: s.transform_type,
      }))
      const res = await api.features.apply(datasetId, transforms)
      setResult(res)
      onApplied?.(res)
    } finally {
      setApplying(false)
    }
  }

  if (suggestions.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        No feature suggestions available for this dataset.
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          {approved.size} of {suggestions.length} selected
        </p>
        <Button
          size="sm"
          onClick={handleApply}
          disabled={approved.size === 0 || applying}
        >
          {applying ? "Applying…" : `Apply ${approved.size > 0 ? `(${approved.size})` : ""}`}
        </Button>
      </div>

      {result && (
        <div className="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-xs dark:border-green-900 dark:bg-green-950">
          <p className="font-semibold text-green-800 dark:text-green-200">
            {result.new_columns.length} new column{result.new_columns.length !== 1 ? "s" : ""} created
          </p>
          <p className="mt-0.5 text-green-700 dark:text-green-300">
            {result.new_columns.slice(0, 5).join(", ")}
            {result.new_columns.length > 5 ? ` +${result.new_columns.length - 5} more` : ""}
          </p>
        </div>
      )}

      {suggestions.map((s) => {
        const isApproved = approved.has(s.id)
        return (
          <button
            key={s.id}
            role="checkbox"
            aria-checked={isApproved}
            className={`w-full rounded-lg border px-3 py-2 text-left transition-colors ${
              isApproved
                ? "border-primary bg-primary/5"
                : "border-border hover:border-muted-foreground/40"
            }`}
            onClick={() => toggleApprove(s.id)}
          >
            <div className="flex items-start gap-2">
              <div
                aria-hidden="true"
                className={`mt-0.5 h-4 w-4 flex-shrink-0 rounded border-2 ${
                  isApproved ? "border-primary bg-primary" : "border-muted-foreground/50"
                }`}
              />
              <div className="flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-1.5 mb-1">
                  <span className="text-xs font-semibold">{s.title}</span>
                  <Badge className={TRANSFORM_COLORS[s.transform_type]}>
                    {TRANSFORM_LABELS[s.transform_type]}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {s.description}
                </p>
                {s.preview_columns.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {s.preview_columns.slice(0, 4).map((col) => (
                      <Badge key={col} variant="secondary" className="text-[10px] px-1.5 py-0">
                        {col}
                      </Badge>
                    ))}
                    {s.preview_columns.length > 4 && (
                      <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                        +{s.preview_columns.length - 4} more
                      </Badge>
                    )}
                  </div>
                )}
              </div>
            </div>
          </button>
        )
      })}
    </div>
  )
}


// ---------------------------------------------------------------------------
// PipelinePanel — shows active transformation steps with undo support
// ---------------------------------------------------------------------------

interface PipelineStep {
  index: number
  column: string
  transform_type: string
  params?: Record<string, unknown>
}

interface PipelinePanelProps {
  featureSetId: string
  onStepRemoved?: (newColumns: string[]) => void
}

export function PipelinePanel({ featureSetId, onStepRemoved }: PipelinePanelProps) {
  const [steps, setSteps] = useState<PipelineStep[]>([])
  const [loading, setLoading] = useState(true)
  const [removing, setRemoving] = useState<number | null>(null)

  useEffect(() => {
    api.features.getSteps(featureSetId).then((res) => {
      setSteps(res.steps)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [featureSetId])

  async function handleUndo(stepIndex: number) {
    setRemoving(stepIndex)
    try {
      const res = await api.features.removeStep(featureSetId, stepIndex)
      setSteps(res.steps)
      onStepRemoved?.(res.new_columns)
    } finally {
      setRemoving(null)
    }
  }

  if (loading) {
    return <p className="text-xs text-muted-foreground">Loading pipeline…</p>
  }

  if (steps.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        No transformations applied yet. Select suggestions above and click Apply.
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-xs text-muted-foreground">
        {steps.length} transformation{steps.length !== 1 ? "s" : ""} in pipeline
      </p>
      {steps.map((step) => (
        <div
          key={step.index}
          className="flex items-center gap-2 rounded-md border border-border bg-muted/30 px-2.5 py-1.5"
        >
          <span className="flex-shrink-0 rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-mono text-primary">
            {step.index + 1}
          </span>
          <div className="flex-1 min-w-0">
            <span className="text-xs font-medium truncate">{step.column}</span>
            <span className="mx-1.5 text-muted-foreground">→</span>
            <span
              className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium ${
                TRANSFORM_COLORS[step.transform_type as FeatureSuggestion["transform_type"]] ??
                "bg-gray-100 text-gray-800"
              }`}
            >
              {TRANSFORM_LABELS[step.transform_type as FeatureSuggestion["transform_type"]] ??
                step.transform_type}
            </span>
          </div>
          <button
            onClick={() => handleUndo(step.index)}
            disabled={removing !== null}
            className="flex-shrink-0 rounded px-1.5 py-0.5 text-[10px] text-muted-foreground hover:bg-destructive/10 hover:text-destructive disabled:opacity-40 transition-colors"
            title="Undo this step"
          >
            {removing === step.index ? "…" : "Undo"}
          </button>
        </div>
      ))}
    </div>
  )
}


// ---------------------------------------------------------------------------
// DatasetListPanel — lists project datasets + guided merge UI
// ---------------------------------------------------------------------------

interface DatasetListPanelProps {
  projectId: string
  onMerged?: (result: MergeResponse) => void
}

export function DatasetListPanel({ projectId, onMerged }: DatasetListPanelProps) {
  const [datasets, setDatasets] = useState<DatasetListItem[]>([])
  const [loading, setLoading] = useState(true)

  // Merge UI state
  const [mergeOpen, setMergeOpen] = useState(false)
  const [leftId, setLeftId] = useState("")
  const [rightId, setRightId] = useState("")
  const [joinKeys, setJoinKeys] = useState<JoinKeySuggestion[]>([])
  const [loadingKeys, setLoadingKeys] = useState(false)
  const [joinKey, setJoinKey] = useState("")
  const [joinHow, setJoinHow] = useState("inner")
  const [merging, setMerging] = useState(false)
  const [mergeResult, setMergeResult] = useState<MergeResponse | null>(null)
  const [mergeError, setMergeError] = useState("")

  useEffect(() => {
    api.data
      .listByProject(projectId)
      .then(setDatasets)
      .catch(() => setDatasets([]))
      .finally(() => setLoading(false))
  }, [projectId])

  // When both datasets are selected, fetch join key suggestions
  useEffect(() => {
    if (!leftId || !rightId || leftId === rightId) {
      setJoinKeys([])
      setJoinKey("")
      return
    }
    setLoadingKeys(true)
    api.data
      .joinKeys(leftId, rightId)
      .then((r) => {
        setJoinKeys(r.join_key_suggestions)
        const best = r.join_key_suggestions.find((k) => k.recommended)
        setJoinKey(best?.name ?? r.join_key_suggestions[0]?.name ?? "")
      })
      .catch(() => setJoinKeys([]))
      .finally(() => setLoadingKeys(false))
  }, [leftId, rightId])

  async function handleMerge() {
    if (!leftId || !rightId || !joinKey) return
    setMerging(true)
    setMergeError("")
    setMergeResult(null)
    try {
      const result = await api.data.merge(projectId, {
        dataset_id_1: leftId,
        dataset_id_2: rightId,
        join_key: joinKey,
        how: joinHow,
      })
      setMergeResult(result)
      // Refresh the dataset list
      const updated = await api.data.listByProject(projectId)
      setDatasets(updated)
      onMerged?.(result)
    } catch {
      setMergeError("Merge failed. Check that the join key exists in both datasets.")
    } finally {
      setMerging(false)
    }
  }

  if (loading) {
    return <p className="text-xs text-muted-foreground">Loading datasets…</p>
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          {datasets.length} dataset{datasets.length !== 1 ? "s" : ""} in this project
        </p>
        {datasets.length >= 2 && (
          <button
            onClick={() => setMergeOpen((v) => !v)}
            className="text-xs text-primary hover:underline underline-offset-2"
          >
            {mergeOpen ? "Cancel merge" : "Merge two datasets"}
          </button>
        )}
      </div>

      {/* Dataset list */}
      <div className="flex flex-col gap-1.5">
        {datasets.map((ds) => (
          <div
            key={ds.dataset_id}
            className="flex items-center gap-2 rounded-md border border-border bg-muted/20 px-3 py-2"
          >
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium truncate">{ds.filename}</p>
              <p className="text-[10px] text-muted-foreground">
                {ds.row_count.toLocaleString()} rows · {ds.column_count} columns
              </p>
            </div>
            <Badge variant="secondary" className="text-[10px] shrink-0">
              {new Date(ds.uploaded_at).toLocaleDateString()}
            </Badge>
          </div>
        ))}
      </div>

      {/* Merge UI */}
      {mergeOpen && datasets.length >= 2 && (
        <div className="rounded-lg border border-dashed border-primary/40 bg-primary/5 p-3 flex flex-col gap-3">
          <p className="text-xs font-semibold">Merge datasets</p>
          <p className="text-[11px] text-muted-foreground">
            Combine two datasets on a shared column. Conflicting column names get a suffix to avoid clashes.
          </p>

          <div className="grid grid-cols-2 gap-2">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-muted-foreground">Left dataset</label>
              <select
                value={leftId}
                onChange={(e) => setLeftId(e.target.value)}
                className="rounded-md border border-border bg-background px-2 py-1 text-xs"
              >
                <option value="">Select…</option>
                {datasets.map((ds) => (
                  <option key={ds.dataset_id} value={ds.dataset_id}>
                    {ds.filename}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-muted-foreground">Right dataset</label>
              <select
                value={rightId}
                onChange={(e) => setRightId(e.target.value)}
                className="rounded-md border border-border bg-background px-2 py-1 text-xs"
              >
                <option value="">Select…</option>
                {datasets
                  .filter((ds) => ds.dataset_id !== leftId)
                  .map((ds) => (
                    <option key={ds.dataset_id} value={ds.dataset_id}>
                      {ds.filename}
                    </option>
                  ))}
              </select>
            </div>
          </div>

          {loadingKeys && (
            <p className="text-[11px] text-muted-foreground">Finding common columns…</p>
          )}

          {!loadingKeys && joinKeys.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-muted-foreground">Join on</label>
                <select
                  value={joinKey}
                  onChange={(e) => setJoinKey(e.target.value)}
                  className="rounded-md border border-border bg-background px-2 py-1 text-xs"
                >
                  {joinKeys.map((k) => (
                    <option key={k.name} value={k.name}>
                      {k.name} {k.recommended ? "★" : ""}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-muted-foreground">Join type</label>
                <select
                  value={joinHow}
                  onChange={(e) => setJoinHow(e.target.value)}
                  className="rounded-md border border-border bg-background px-2 py-1 text-xs"
                >
                  <option value="inner">Inner — keep matching rows only</option>
                  <option value="left">Left — keep all rows from left</option>
                  <option value="right">Right — keep all rows from right</option>
                  <option value="outer">Outer — keep all rows from both</option>
                </select>
              </div>
            </div>
          )}

          {!loadingKeys && leftId && rightId && joinKeys.length === 0 && (
            <p className="text-[11px] text-amber-600">
              No common columns found — these datasets cannot be merged directly.
            </p>
          )}

          {mergeError && (
            <p className="text-[11px] text-destructive">{mergeError}</p>
          )}

          {mergeResult && (
            <div className="rounded-md border border-green-200 bg-green-50 px-3 py-2 text-[11px] dark:border-green-900 dark:bg-green-950">
              <p className="font-semibold text-green-800 dark:text-green-200">
                Merged! {mergeResult.row_count.toLocaleString()} rows · {mergeResult.column_count} columns
              </p>
              <p className="mt-0.5 text-green-700 dark:text-green-300">
                Saved as <strong>{mergeResult.filename}</strong>
                {mergeResult.conflict_columns.length > 0
                  ? ` (${mergeResult.conflict_columns.length} column${mergeResult.conflict_columns.length !== 1 ? "s" : ""} renamed with suffixes)`
                  : ""}
              </p>
            </div>
          )}

          <Button
            size="sm"
            onClick={handleMerge}
            disabled={!leftId || !rightId || !joinKey || merging || joinKeys.length === 0}
          >
            {merging ? "Merging…" : "Merge datasets"}
          </Button>
        </div>
      )}
    </div>
  )
}


interface ImportancePanelProps {
  features: FeatureImportanceEntry[]
  targetColumn: string
  problemType: string
}

export function FeatureImportancePanel({
  features,
  targetColumn,
  problemType,
}: ImportancePanelProps) {
  const maxImportance = features[0]?.importance_pct ?? 1
  const label = problemType === "classification" ? "Classification" : "Regression"

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-muted-foreground">
        Predicting <strong>{targetColumn}</strong> ({label}). Bars show relative predictive signal.
      </p>
      {features.map((f, idx) => (
        <div key={f.column} className="group">
          <ImportanceBar
            feature={f.column}
            importance={maxImportance > 0 ? f.importance_pct / maxImportance : 0}
            rank={idx + 1}
            label={`${f.importance_pct.toFixed(1)}%`}
          />
          {f.description && (
            <p className="text-[10px] text-muted-foreground mt-0.5 hidden group-hover:block pl-7">
              {f.description}
            </p>
          )}
        </div>
      ))}
    </div>
  )
}
