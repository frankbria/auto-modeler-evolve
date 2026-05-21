"use client"

import type { WhatNextResult, WhatNextStep } from "@/lib/types"

// Stage → color tokens
const STAGE_COLORS: Record<
  string,
  { border: string; badge: string; bar: string }
> = {
  upload: {
    border: "border-blue-200",
    badge: "bg-blue-100 text-blue-700",
    bar: "bg-blue-500",
  },
  explore: {
    border: "border-emerald-200",
    badge: "bg-emerald-100 text-emerald-700",
    bar: "bg-emerald-500",
  },
  validate: {
    border: "border-amber-200",
    badge: "bg-amber-100 text-amber-700",
    bar: "bg-amber-500",
  },
  monitor: {
    border: "border-violet-200",
    badge: "bg-violet-100 text-violet-700",
    bar: "bg-violet-500",
  },
}

interface StepRowProps {
  step: WhatNextStep
  index: number
  onAction: (action: string) => void
}

function StepRow({ step, index, onAction }: StepRowProps) {
  return (
    <li
      className="flex items-start gap-3 rounded-lg border bg-card p-3"
      data-testid={`what-next-step-${index}`}
    >
      <span aria-hidden="true" className="mt-0.5 text-xl leading-none">
        {step.icon}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-foreground">{step.title}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {step.description}
        </p>
      </div>
      <button
        onClick={() => onAction(step.action)}
        className="shrink-0 rounded-md border border-primary/30 bg-primary/5 px-2 py-1 text-xs font-medium text-primary hover:bg-primary/10 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-label={`Try: ${step.title}`}
        data-testid={`what-next-try-${index}`}
      >
        Try this →
      </button>
    </li>
  )
}

interface WhatNextCardProps {
  result: WhatNextResult
  /** Pre-fills the chat input with the action string */
  onActionClick?: (action: string) => void
}

export function WhatNextCard({ result, onActionClick }: WhatNextCardProps) {
  const colors = STAGE_COLORS[result.stage] ?? STAGE_COLORS.explore

  const handleAction = (action: string) => {
    onActionClick?.(action)
  }

  return (
    <figure
      className={`rounded-xl border-2 bg-card p-4 shadow-sm ${colors.border}`}
      aria-label="What to do next"
      data-testid="what-next-card"
    >
      {/* Header */}
      <div className="mb-3 flex items-center gap-2">
        <span aria-hidden="true" className="text-xl">
          🎯
        </span>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-bold text-foreground">
              What To Do Next
            </h3>
            <span
              className={`rounded-full px-2 py-0.5 text-xs font-medium ${colors.badge}`}
              data-testid="what-next-stage-badge"
            >
              {result.stage_label}
            </span>
          </div>
          {/* Progress bar */}
          <div
            className="mt-1.5 h-1.5 w-full rounded-full bg-muted"
            role="progressbar"
            aria-valuenow={result.progress}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Workflow progress: ${result.progress}%`}
          >
            <div
              className={`h-full rounded-full transition-all ${colors.bar}`}
              style={{ width: `${result.progress}%` }}
              data-testid="what-next-progress-bar"
            />
          </div>
        </div>
      </div>

      {/* Summary */}
      <p
        className="mb-3 text-sm text-muted-foreground"
        data-testid="what-next-summary"
      >
        {result.summary}
      </p>

      {/* Steps */}
      <ul className="space-y-2" aria-label="Recommended next steps">
        {result.steps.map((step, i) => (
          <StepRow
            key={i}
            step={step}
            index={i}
            onAction={handleAction}
          />
        ))}
      </ul>

      <figcaption className="sr-only">
        Current stage: {result.stage_label}. Progress: {result.progress}%.{" "}
        {result.summary}
      </figcaption>
    </figure>
  )
}
