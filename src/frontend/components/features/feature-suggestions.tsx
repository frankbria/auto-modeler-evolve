"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { FeatureSuggestion, FeatureSetResult, FeatureImportanceEntry } from "@/lib/types"
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
          <div
            key={s.id}
            className={`rounded-lg border px-3 py-2 transition-colors cursor-pointer ${
              isApproved
                ? "border-primary bg-primary/5"
                : "border-border hover:border-muted-foreground/40"
            }`}
            onClick={() => toggleApprove(s.id)}
          >
            <div className="flex items-start gap-2">
              <div
                className={`mt-0.5 h-4 w-4 flex-shrink-0 rounded border-2 ${
                  isApproved ? "border-primary bg-primary" : "border-muted-foreground/50"
                }`}
              />
              <div className="flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-1.5 mb-1">
                  <span className="text-xs font-semibold">{s.title}</span>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
                      TRANSFORM_COLORS[s.transform_type]
                    }`}
                  >
                    {TRANSFORM_LABELS[s.transform_type]}
                  </span>
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
          </div>
        )
      })}
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
      {features.map((f) => (
        <div key={f.column} className="group">
          <div className="flex items-center justify-between mb-0.5">
            <span className="text-xs font-medium truncate max-w-[60%]">{f.column}</span>
            <span className="text-xs text-muted-foreground">{f.importance_pct.toFixed(1)}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
            <div
              className="h-full rounded-full bg-primary transition-all"
              style={{ width: `${(f.importance_pct / maxImportance) * 100}%` }}
            />
          </div>
          <p className="text-[10px] text-muted-foreground mt-0.5 hidden group-hover:block">
            {f.description}
          </p>
        </div>
      ))}
    </div>
  )
}
