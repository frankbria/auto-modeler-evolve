"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type {
  ModelComparison,
  ModelMetrics,
  ModelRanking,
  ModelRecommendation,
} from "@/lib/types"

// ---------------------------------------------------------------------------
// RecommendationsPanel — shown before training
// ---------------------------------------------------------------------------

interface RecommendationsPanelProps {
  recommendations: ModelRecommendation[]
  selected: Set<string>
  onToggle: (algorithm: string) => void
}

export function RecommendationsPanel({
  recommendations,
  selected,
  onToggle,
}: RecommendationsPanelProps) {
  return (
    <div className="flex flex-col gap-2">
      {recommendations.map((rec) => {
        const isSelected = selected.has(rec.algorithm)
        return (
          <button
            key={rec.algorithm}
            onClick={() => onToggle(rec.algorithm)}
            className={`w-full rounded-lg border p-3 text-left transition-colors ${
              isSelected
                ? "border-primary bg-primary/5"
                : "border-border hover:border-muted-foreground/50"
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1">
                <p className="text-xs font-semibold">{rec.display_name}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">{rec.description}</p>
                <p className="mt-1 text-xs text-muted-foreground/70 italic">{rec.reason}</p>
              </div>
              <div
                className={`mt-0.5 h-4 w-4 shrink-0 rounded-sm border transition-colors ${
                  isSelected ? "border-primary bg-primary" : "border-muted-foreground/40"
                }`}
              />
            </div>
          </button>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// MetricCell — a single metric value with formatting
// ---------------------------------------------------------------------------

function MetricCell({ value, name }: { value: number | undefined; name: string }) {
  if (value === undefined || value === null) {
    return <span className="text-muted-foreground/40">—</span>
  }
  const isPercent = ["accuracy", "f1", "precision", "recall", "roc_auc"].includes(name)
  const formatted = isPercent
    ? `${(value * 100).toFixed(1)}%`
    : name === "r2"
    ? value.toFixed(3)
    : value.toFixed(2)
  return <span>{formatted}</span>
}

// ---------------------------------------------------------------------------
// ComparisonTable — shown after training
// ---------------------------------------------------------------------------

interface ComparisonTableProps {
  comparison: ModelComparison
  onSelect: (modelRunId: string) => void
  selecting: string | null
}

export function ComparisonTable({
  comparison,
  onSelect,
  selecting,
}: ComparisonTableProps) {
  const { summary, rankings, primary_metric } = comparison

  // Collect all metric keys that appear in any ranking
  const metricKeys = Array.from(
    new Set(rankings.flatMap((r) => Object.keys(r.metrics ?? {})))
  )

  return (
    <div className="flex flex-col gap-4">
      {/* Summary banner */}
      <div className="rounded-lg border bg-muted/40 px-4 py-3">
        <p className="text-xs leading-relaxed text-foreground">{summary}</p>
      </div>

      {/* Comparison table */}
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="px-3 py-2 font-medium">#</th>
              <th className="px-3 py-2 font-medium">Model</th>
              {metricKeys.map((k) => (
                <th key={k} className="px-3 py-2 font-medium capitalize">
                  {k === "roc_auc" ? "ROC AUC" : k.toUpperCase().replace("_", " ")}
                  {k === primary_metric && (
                    <span className="ml-1 text-primary">★</span>
                  )}
                </th>
              ))}
              <th className="px-3 py-2 font-medium">Time</th>
              <th className="px-3 py-2 font-medium"></th>
            </tr>
          </thead>
          <tbody>
            {rankings.map((ranking) => (
              <ModelRow
                key={ranking.id}
                ranking={ranking}
                metricKeys={metricKeys}
                onSelect={onSelect}
                selecting={selecting}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ModelRow({
  ranking,
  metricKeys,
  onSelect,
  selecting,
}: {
  ranking: ModelRanking
  metricKeys: string[]
  onSelect: (id: string) => void
  selecting: string | null
}) {
  const isSelecting = selecting === ranking.id

  return (
    <tr
      className={`border-b last:border-b-0 ${
        ranking.is_selected ? "bg-primary/5" : ""
      }`}
    >
      <td className="px-3 py-2 text-muted-foreground">{ranking.rank}</td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="font-medium">{ranking.display_name}</span>
          {ranking.is_selected && (
            <Badge variant="outline" className="text-xs">Selected</Badge>
          )}
        </div>
      </td>
      {metricKeys.map((k) => (
        <td key={k} className="px-3 py-2 tabular-nums">
          <MetricCell value={(ranking.metrics as ModelMetrics)?.[k as keyof ModelMetrics]} name={k} />
        </td>
      ))}
      <td className="px-3 py-2 text-muted-foreground tabular-nums">
        {ranking.training_duration_ms != null
          ? ranking.training_duration_ms < 1000
            ? `${ranking.training_duration_ms}ms`
            : `${(ranking.training_duration_ms / 1000).toFixed(1)}s`
          : "—"}
      </td>
      <td className="px-3 py-2">
        {!ranking.is_selected && (
          <Button
            size="sm"
            variant="outline"
            className="h-6 px-2 text-xs"
            onClick={() => onSelect(ranking.id)}
            disabled={isSelecting}
          >
            {isSelecting ? "..." : "Select"}
          </Button>
        )}
      </td>
    </tr>
  )
}
