"use client"

import type { AutoInsightResult, AutoInsightFinding } from "@/lib/types"

interface AutoInsightCardProps {
  result: AutoInsightResult
  /** Pre-fills the chat input with the suggested follow-up action */
  onActionClick?: (prompt: string) => void
}

// Priority → badge color
function PriorityBadge({ priority }: { priority: number }) {
  if (priority === 1)
    return (
      <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-medium text-sky-700">
        High interest
      </span>
    )
  if (priority === 2)
    return (
      <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
        Worth noting
      </span>
    )
  return (
    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
      FYI
    </span>
  )
}

function FindingRow({
  finding,
  index,
  onActionClick,
}: {
  finding: AutoInsightFinding
  index: number
  onActionClick?: (prompt: string) => void
}) {
  // Render **bold** markdown in the finding text
  const parts = finding.finding.split(/(\*\*[^*]+\*\*)/)
  const renderedFinding = parts.map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={i}>{part}</span>
    )
  )

  return (
    <div
      className="flex items-start gap-3 rounded-lg border border-slate-100 bg-slate-50/60 p-3"
      data-testid={`insight-finding-${index}`}
    >
      {/* Icon */}
      <span
        aria-hidden="true"
        className="mt-0.5 text-xl leading-none"
        data-testid={`insight-icon-${index}`}
      >
        {finding.icon}
      </span>

      {/* Content */}
      <div className="flex-1 space-y-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <PriorityBadge priority={finding.priority} />
        </div>
        <p
          className="text-sm text-foreground leading-relaxed"
          data-testid={`insight-text-${index}`}
        >
          {renderedFinding}
        </p>
        <button
          onClick={() => onActionClick?.(finding.suggested_action)}
          className="mt-1 inline-flex items-center gap-1 rounded-full border border-sky-200 bg-white px-3 py-1 text-xs font-medium text-sky-700 transition-colors hover:bg-sky-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={`Try: ${finding.suggested_action}`}
          data-testid={`insight-action-${index}`}
        >
          {finding.suggested_action} →
        </button>
      </div>
    </div>
  )
}

export function AutoInsightCard({ result, onActionClick }: AutoInsightCardProps) {
  return (
    <figure
      className="rounded-xl border-2 border-sky-300 bg-card p-4 shadow-sm"
      aria-label={`Auto-insights for ${result.dataset_name}`}
      data-testid="auto-insight-card"
    >
      {/* Header */}
      <div className="mb-3 flex items-center gap-2">
        <span
          aria-hidden="true"
          className="text-2xl leading-none"
          data-testid="auto-insight-icon"
        >
          🔍
        </span>
        <div className="flex-1">
          <h3
            className="text-sm font-bold text-foreground"
            data-testid="auto-insight-heading"
          >
            Interesting findings in your data
          </h3>
          <div className="mt-0.5 flex flex-wrap gap-1.5">
            <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs text-sky-700">
              {result.dataset_name}
            </span>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
              {result.row_count.toLocaleString()} rows
            </span>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
              {result.column_count} columns
            </span>
          </div>
        </div>
      </div>

      {/* Summary */}
      <p
        className="mb-3 text-sm text-muted-foreground"
        data-testid="auto-insight-summary"
      >
        {result.summary}
      </p>

      {/* Findings list */}
      <div className="space-y-2" data-testid="auto-insight-findings">
        {result.findings.map((finding, i) => (
          <FindingRow
            key={i}
            finding={finding}
            index={i}
            onActionClick={onActionClick}
          />
        ))}
      </div>

      <figcaption className="sr-only">
        Auto-insights for {result.dataset_name}: {result.summary}{" "}
        {result.findings.map((f) => f.finding).join(" ")}
      </figcaption>
    </figure>
  )
}
