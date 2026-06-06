"use client"

import { Badge } from "@/components/ui/badge"
import type { SegmentDriftEntry, SegmentDriftResult, SegmentDriftStatus, SegmentDriftVerdict } from "@/lib/types"

function statusColor(status: SegmentDriftStatus): string {
  switch (status) {
    case "high":
      return "bg-rose-500"
    case "moderate":
      return "bg-amber-400"
    case "low":
      return "bg-sky-400"
    default:
      return "bg-emerald-400"
  }
}

function statusBadgeClass(status: SegmentDriftStatus): string {
  switch (status) {
    case "high":
      return "bg-rose-100 text-rose-800 border-rose-200"
    case "moderate":
      return "bg-amber-100 text-amber-800 border-amber-200"
    case "low":
      return "bg-sky-100 text-sky-800 border-sky-200"
    default:
      return "bg-emerald-100 text-emerald-800 border-emerald-200"
  }
}

function verdictBadgeClass(verdict: SegmentDriftVerdict): string {
  switch (verdict) {
    case "concentrated":
      return "bg-amber-100 text-amber-800 border-amber-200"
    case "widespread":
      return "bg-rose-100 text-rose-800 border-rose-200"
    case "minimal":
      return "bg-emerald-100 text-emerald-800 border-emerald-200"
    default:
      return "bg-muted text-muted-foreground border-border"
  }
}

function verdictLabel(verdict: SegmentDriftVerdict): string {
  switch (verdict) {
    case "concentrated":
      return "Drift Concentrated"
    case "widespread":
      return "Drift Widespread"
    case "minimal":
      return "Minimal Drift"
    default:
      return "No Data"
  }
}

interface SegmentRowProps {
  entry: SegmentDriftEntry
}

function SegmentRow({ entry }: SegmentRowProps) {
  return (
    <div className="flex items-center gap-3 py-2 border-b last:border-0">
      <div className="w-28 truncate font-medium text-sm" title={entry.name} data-testid="segment-name">
        {entry.name}
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <div
            className="h-2 rounded-full transition-all"
            style={{ width: `${Math.min(entry.drift_score, 100)}%`, minWidth: "4px" }}
          >
            <div className={`h-full w-full rounded-full ${statusColor(entry.status)}`} />
          </div>
          <span className="text-xs text-muted-foreground w-12 shrink-0">
            {entry.drift_score.toFixed(1)}%
          </span>
        </div>
        {entry.top_drifting_features.length > 0 && (
          <div className="text-xs text-muted-foreground truncate" data-testid="top-features">
            {entry.top_drifting_features.map((f) => f.name).join(", ")}
          </div>
        )}
      </div>
      <div className="flex flex-col items-end gap-1 shrink-0">
        <Badge
          variant="outline"
          className={`text-xs ${statusBadgeClass(entry.status)}`}
          data-testid="status-badge"
        >
          {entry.status}
        </Badge>
        <span className="text-xs text-muted-foreground">{entry.sample_count} preds</span>
      </div>
    </div>
  )
}

interface SegmentDriftCardProps {
  result: SegmentDriftResult
}

export function SegmentDriftCard({ result }: SegmentDriftCardProps) {
  const borderClass =
    result.verdict === "widespread"
      ? "border-rose-300"
      : result.verdict === "concentrated"
        ? "border-amber-300"
        : result.verdict === "minimal"
          ? "border-emerald-300"
          : "border-border"

  return (
    <figure
      className={`rounded-lg border-2 ${borderClass} bg-card p-4 space-y-3 my-2`}
      data-testid="segment-drift-card"
    >
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true">🗺️</span>
          <h3 className="font-semibold text-sm">Segment Drift Analysis</h3>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {result.segment_column && (
            <Badge variant="outline" className="text-xs" data-testid="segment-col-badge">
              By: {result.segment_column}
            </Badge>
          )}
          {result.n_segments > 0 && (
            <Badge variant="outline" className="text-xs" data-testid="n-segments-badge">
              {result.n_segments} segment{result.n_segments !== 1 ? "s" : ""}
            </Badge>
          )}
          <Badge
            variant="outline"
            className={`text-xs ${verdictBadgeClass(result.verdict)}`}
            data-testid="verdict-badge"
          >
            {verdictLabel(result.verdict)}
          </Badge>
        </div>
      </div>

      <p className="text-sm text-muted-foreground" data-testid="summary-text">
        {result.summary}
      </p>

      {result.segments.length > 0 && (
        <>
          {(result.high_count > 0 || result.moderate_count > 0) && (
            <div
              className="flex gap-3 flex-wrap text-xs"
              role="status"
              aria-live="polite"
              data-testid="status-counts"
            >
              {result.high_count > 0 && (
                <span className="text-rose-700 font-medium">
                  {result.high_count} high
                </span>
              )}
              {result.moderate_count > 0 && (
                <span className="text-amber-700 font-medium">
                  {result.moderate_count} moderate
                </span>
              )}
              {result.low_count > 0 && (
                <span className="text-sky-700">
                  {result.low_count} low
                </span>
              )}
              {result.minimal_count > 0 && (
                <span className="text-emerald-700">
                  {result.minimal_count} minimal
                </span>
              )}
            </div>
          )}

          <div data-testid="segments-list">
            {result.segments.map((entry) => (
              <SegmentRow key={entry.name} entry={entry} />
            ))}
          </div>

          <div className="text-xs text-muted-foreground pt-1">
            Avg drift: <strong>{result.avg_drift_score.toFixed(1)}%</strong> across{" "}
            {result.n_samples} predictions · Drift score = % of inputs outside training distribution
          </div>
        </>
      )}

      <figcaption className="sr-only">
        Segment drift analysis by {result.segment_column}: {result.summary}
      </figcaption>
    </figure>
  )
}
