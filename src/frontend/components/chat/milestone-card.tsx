"use client"

import type { MilestoneResult } from "@/lib/types"

// Milestone type → color tokens
const MILESTONE_COLORS: Record<
  string,
  { border: string; badge: string; bar: string; bg: string }
> = {
  upload: {
    border: "border-emerald-300",
    badge: "bg-emerald-100 text-emerald-700",
    bar: "bg-emerald-500",
    bg: "bg-emerald-50",
  },
  train: {
    border: "border-amber-300",
    badge: "bg-amber-100 text-amber-700",
    bar: "bg-amber-500",
    bg: "bg-amber-50",
  },
  deploy: {
    border: "border-violet-300",
    badge: "bg-violet-100 text-violet-700",
    bar: "bg-violet-500",
    bg: "bg-violet-50",
  },
}

interface MilestoneCardProps {
  result: MilestoneResult
  /** Pre-fills the chat input with the action prompt */
  onActionClick?: (prompt: string) => void
}

export function MilestoneCard({ result, onActionClick }: MilestoneCardProps) {
  const colors =
    MILESTONE_COLORS[result.milestone_type] ?? MILESTONE_COLORS.upload

  return (
    <figure
      className={`rounded-xl border-2 bg-card p-4 shadow-sm ${colors.border}`}
      aria-label={`Milestone: ${result.title}`}
      data-testid="milestone-card"
    >
      {/* Header */}
      <div className={`mb-3 flex items-center gap-3 rounded-lg p-3 ${colors.bg}`}>
        <span
          aria-hidden="true"
          className="text-3xl leading-none"
          data-testid="milestone-icon"
        >
          {result.icon}
        </span>
        <div className="flex-1">
          <h3
            className="text-base font-bold text-foreground"
            data-testid="milestone-title"
          >
            {result.title}
          </h3>
          <span
            className={`mt-0.5 inline-block rounded-full px-2 py-0.5 text-xs font-medium ${colors.badge}`}
            data-testid="milestone-subtitle"
          >
            {result.subtitle}
          </span>
        </div>
      </div>

      {/* Progress bar */}
      <div
        className="mb-3 h-2 w-full rounded-full bg-muted"
        role="progressbar"
        aria-valuenow={result.progress}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Workflow progress: ${result.progress}%`}
        data-testid="milestone-progress"
      >
        <div
          className={`h-full rounded-full transition-all ${colors.bar}`}
          style={{ width: `${result.progress}%` }}
        />
      </div>

      {/* Summary */}
      <p
        className="mb-3 text-sm text-muted-foreground"
        data-testid="milestone-summary"
      >
        {result.summary}
      </p>

      {/* Action chips */}
      <div
        className="flex flex-wrap gap-2"
        aria-label="Suggested next actions"
        data-testid="milestone-actions"
      >
        {result.actions.map((action, i) => (
          <button
            key={i}
            onClick={() => onActionClick?.(action.prompt)}
            className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${colors.badge} hover:opacity-80`}
            aria-label={`Try: ${action.label}`}
            data-testid={`milestone-action-${i}`}
          >
            {action.label} →
          </button>
        ))}
      </div>

      <figcaption className="sr-only">
        {result.title}. {result.subtitle}. Progress: {result.progress}%.{" "}
        {result.summary}
      </figcaption>
    </figure>
  )
}
