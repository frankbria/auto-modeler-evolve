"use client"

import type {
  DataVersionEntry,
  DataVersionHistoryResult,
} from "@/lib/types"

interface DataVersionHistoryCardProps {
  history: DataVersionHistoryResult
}

const STABILITY_COLORS: Record<
  "stable" | "moderate" | "high",
  { border: string; bg: string; badge: string; text: string; label: string }
> = {
  stable: {
    border: "border-emerald-200",
    bg: "bg-emerald-50",
    badge: "bg-emerald-100 text-emerald-800",
    text: "text-emerald-900",
    label: "Stable",
  },
  moderate: {
    border: "border-amber-200",
    bg: "bg-amber-50",
    badge: "bg-amber-100 text-amber-800",
    text: "text-amber-900",
    label: "Moderate Drift",
  },
  high: {
    border: "border-rose-200",
    bg: "bg-rose-50",
    badge: "bg-rose-100 text-rose-800",
    text: "text-rose-900",
    label: "High Drift",
  },
}

function driftBadgeColor(score: number) {
  if (score < 20) return "bg-emerald-100 text-emerald-800"
  if (score < 50) return "bg-amber-100 text-amber-800"
  return "bg-rose-100 text-rose-800"
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(iso: string): string {
  if (!iso) return "Unknown"
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    })
  } catch {
    return iso
  }
}

function VersionDriftConnector({ drift }: { drift: DataVersionEntry["drift_from_previous"] }) {
  if (!drift) return null
  const { drift_score, changed_columns, row_count_change_pct, new_columns, dropped_columns } = drift
  const sign = row_count_change_pct >= 0 ? "+" : ""
  return (
    <div className="flex flex-col items-center my-1.5">
      {/* Vertical connector line */}
      <div className="w-px h-3 bg-gray-300" aria-hidden="true" />
      {/* Drift badge */}
      <div
        className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-medium border ${driftBadgeColor(drift_score)} border-transparent`}
        title={drift.summary}
      >
        <span aria-hidden="true">⇕</span>
        <span>Drift: {drift_score}/100</span>
        {changed_columns > 0 && (
          <span className="text-xs opacity-75">
            · {changed_columns} col{changed_columns !== 1 ? "s" : ""} changed
          </span>
        )}
        {row_count_change_pct !== 0 && (
          <span className="text-xs opacity-75">
            · rows {sign}{row_count_change_pct.toFixed(1)}%
          </span>
        )}
        {new_columns.length > 0 && (
          <span className="text-xs text-emerald-700">
            · +{new_columns.length} new col{new_columns.length !== 1 ? "s" : ""}
          </span>
        )}
        {dropped_columns.length > 0 && (
          <span className="text-xs text-rose-700">
            · -{dropped_columns.length} dropped
          </span>
        )}
      </div>
      <div className="w-px h-3 bg-gray-300" aria-hidden="true" />
    </div>
  )
}

function VersionRow({ entry, isLatest }: { entry: DataVersionEntry; isLatest: boolean }) {
  return (
    <div
      className={`rounded-lg border px-3 py-2.5 ${
        isLatest
          ? "border-blue-300 bg-blue-50"
          : "border-gray-200 bg-white"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className={`shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
              isLatest ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-600"
            }`}
            aria-label={`Version ${entry.version}`}
          >
            {entry.version}
          </span>
          <div className="min-w-0">
            <p className="text-xs font-semibold text-gray-800 truncate">{entry.filename}</p>
            <p className="text-xs text-gray-500">{formatDate(entry.uploaded_at)}</p>
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0 text-xs text-gray-500">
          {isLatest && (
            <span className="px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 font-medium text-xs">
              Latest
            </span>
          )}
          <span>{entry.row_count.toLocaleString()} rows</span>
          <span aria-hidden="true">·</span>
          <span>{entry.column_count} cols</span>
          {entry.size_bytes > 0 && (
            <>
              <span aria-hidden="true">·</span>
              <span>{formatBytes(entry.size_bytes)}</span>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export function DataVersionHistoryCard({ history }: DataVersionHistoryCardProps) {
  const { versions, overall_stability, summary, version_count } = history

  if (!versions || versions.length === 0) return null

  const colors = STABILITY_COLORS[overall_stability] ?? STABILITY_COLORS.stable

  // Render versions in reverse (latest first) for the timeline
  const orderedVersions = [...versions].reverse()

  return (
    <figure
      className={`mt-3 rounded-xl border ${colors.border} ${colors.bg} p-4 shadow-sm max-w-md`}
      aria-label="Data version history"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">
          📂
        </span>
        <div className="flex-1 min-w-0">
          <p className={`font-semibold ${colors.text} text-sm leading-tight`}>
            Data Version History
          </p>
          <p className="text-xs text-gray-500 line-clamp-2">{summary}</p>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${colors.badge}`}>
            {colors.label}
          </span>
          <span className="text-xs text-gray-500">
            {version_count} version{version_count !== 1 ? "s" : ""}
          </span>
        </div>
      </div>

      {/* Timeline */}
      <div className="flex flex-col">
        {orderedVersions.map((entry, idx) => {
          const isLatest = idx === 0
          // The drift connector sits between the current entry and the one below it.
          // Since we reversed the array, entry.drift_from_previous is shown BELOW the entry.
          const nextEntry = orderedVersions[idx + 1]
          const showDrift = nextEntry !== undefined && entry.drift_from_previous !== null

          return (
            <div key={entry.dataset_id}>
              <VersionRow entry={entry} isLatest={isLatest} />
              {showDrift && (
                <VersionDriftConnector drift={entry.drift_from_previous} />
              )}
            </div>
          )
        })}
      </div>
    </figure>
  )
}
