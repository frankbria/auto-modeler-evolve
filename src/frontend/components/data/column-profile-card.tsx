"use client"

import type { ColumnProfile, ColumnProfileIssue } from "@/lib/types"

interface ColumnProfileCardProps {
  profile: ColumnProfile
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-100 text-red-800 border-red-200",
  warning: "bg-amber-100 text-amber-800 border-amber-200",
  info: "bg-blue-50 text-blue-700 border-blue-100",
}

const TYPE_COLORS: Record<string, string> = {
  numeric: "bg-blue-100 text-blue-800",
  categorical: "bg-purple-100 text-purple-800",
  date: "bg-green-100 text-green-800",
}

const TYPE_LABELS: Record<string, string> = {
  numeric: "Numeric",
  categorical: "Categorical",
  date: "Date / Time",
}

function StatChip({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded border bg-muted/50 px-2 py-1 text-center">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm font-semibold text-foreground">{value}</div>
    </div>
  )
}

function DistributionBars({
  bins,
  counts,
  labels,
  type,
}: {
  bins?: number[]
  counts?: number[]
  labels?: string[]
  type: string
}) {
  if (type === "histogram" && bins && counts && counts.length > 0) {
    const maxCount = Math.max(...counts, 1)
    return (
      <div className="mt-2">
        <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground mb-1">
          Distribution
        </div>
        <div className="flex items-end gap-0.5 h-12">
          {counts.map((c, i) => (
            <div
              key={i}
              className="flex-1 bg-primary/60 rounded-t transition-all"
              style={{ height: `${Math.max(4, Math.round((c / maxCount) * 48))}px` }}
              title={`${typeof bins?.[i] === 'number' ? (bins[i] as number).toFixed(2) : bins?.[i]}: ${c} rows`}
            />
          ))}
        </div>
        <div className="flex justify-between text-[9px] text-muted-foreground mt-0.5">
          <span>{bins?.[0] !== undefined ? bins[0].toFixed(2) : ""}</span>
          <span>{bins?.[bins.length - 1] !== undefined ? bins[bins.length - 1].toFixed(2) : ""}</span>
        </div>
      </div>
    )
  }
  if (type === "bar" && labels && counts && labels.length > 0) {
    const maxCount = Math.max(...counts, 1)
    const displayLabels = labels.slice(0, 8)
    const displayCounts = counts.slice(0, 8)
    return (
      <div className="mt-2">
        <div className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground mb-1">
          Top Categories
        </div>
        <div className="space-y-0.5">
          {displayLabels.map((label, i) => (
            <div key={i} className="flex items-center gap-1.5">
              <div className="w-16 truncate text-[10px] text-muted-foreground" title={label}>
                {label}
              </div>
              <div className="flex-1 bg-muted rounded h-3">
                <div
                  className="h-3 bg-primary/70 rounded transition-all"
                  style={{ width: `${Math.max(4, Math.round((displayCounts[i] / maxCount) * 100))}%` }}
                />
              </div>
              <div className="w-8 text-right text-[10px] text-muted-foreground">
                {displayCounts[i].toLocaleString()}
              </div>
            </div>
          ))}
          {labels.length > 8 && (
            <div className="text-[10px] text-muted-foreground italic">
              +{labels.length - 8} more categories
            </div>
          )}
        </div>
      </div>
    )
  }
  return null
}

function IssueRow({ issue }: { issue: ColumnProfileIssue }) {
  const colorClass = SEVERITY_COLORS[issue.severity] ?? SEVERITY_COLORS.info
  const icon = issue.severity === "critical" ? "✗" : issue.severity === "warning" ? "⚠" : "ℹ"
  return (
    <div className={`rounded border px-2 py-1 text-[11px] flex gap-1.5 items-start ${colorClass}`}>
      <span aria-hidden="true" className="mt-0.5 shrink-0">{icon}</span>
      <span>{issue.message}</span>
    </div>
  )
}

export function ColumnProfileCard({ profile }: ColumnProfileCardProps) {
  const { col_name, col_type, stats, distribution, issues, summary } = profile
  const typeColor = TYPE_COLORS[col_type] ?? "bg-gray-100 text-gray-800"
  const typeLabel = TYPE_LABELS[col_type] ?? col_type

  const statChips: { label: string; value: string | number }[] = [
    { label: "Rows", value: stats.total_rows.toLocaleString() },
    { label: "Unique", value: stats.unique_count.toLocaleString() },
    { label: "Missing", value: `${stats.null_pct}%` },
  ]

  if (col_type === "numeric") {
    if (stats.mean !== undefined) statChips.push({ label: "Mean", value: stats.mean })
    if (stats.median !== undefined) statChips.push({ label: "Median", value: stats.median })
    if (stats.std !== undefined) statChips.push({ label: "Std Dev", value: stats.std })
  } else if (col_type === "categorical" && stats.most_common) {
    statChips.push({ label: "Most Common", value: stats.most_common })
    if (stats.most_common_pct !== undefined)
      statChips.push({ label: "Top %", value: `${stats.most_common_pct}%` })
  } else if (col_type === "date") {
    if (stats.min_date) statChips.push({ label: "From", value: stats.min_date })
    if (stats.max_date) statChips.push({ label: "To", value: stats.max_date })
    if (stats.estimated_frequency)
      statChips.push({ label: "Frequency", value: stats.estimated_frequency })
  }

  return (
    <div className="mt-2 rounded-lg border border-cyan-200 bg-cyan-50/50 p-3 text-sm">
      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <span className="font-mono text-sm font-semibold text-foreground">{col_name}</span>
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium uppercase ${typeColor}`}>
          {typeLabel}
        </span>
      </div>

      {/* Summary */}
      <p className="text-[12px] text-muted-foreground mb-2">{summary}</p>

      {/* Stats chips */}
      <div className="grid grid-cols-3 gap-1 mb-2">
        {statChips.slice(0, 6).map((chip) => (
          <StatChip key={chip.label} label={chip.label} value={chip.value} />
        ))}
      </div>

      {/* Distribution chart */}
      <DistributionBars
        type={distribution.type}
        bins={distribution.bins}
        counts={distribution.counts}
        labels={distribution.labels}
      />

      {/* Issues */}
      {issues.length > 0 && (
        <div className="mt-2 space-y-1">
          {issues.map((issue, i) => (
            <IssueRow key={i} issue={issue} />
          ))}
        </div>
      )}
    </div>
  )
}
