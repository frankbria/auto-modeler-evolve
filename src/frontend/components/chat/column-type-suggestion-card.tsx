"use client"

import type { ColumnTypeSuggestionResult, ColumnTypeSuggestion } from "@/lib/types"

interface ColumnTypeSuggestionCardProps {
  result: ColumnTypeSuggestionResult
  /** Pre-fills the chat input with the fix action */
  onActionClick?: (prompt: string) => void
}

function ConfidenceBadge({ confidence }: { confidence: string }) {
  if (confidence === "high")
    return (
      <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
        High confidence
      </span>
    )
  return (
    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
      Medium confidence
    </span>
  )
}

function DtypeBadge({ label, variant }: { label: string; variant: "current" | "suggested" }) {
  if (variant === "current")
    return (
      <span className="rounded border border-slate-200 bg-slate-100 px-2 py-0.5 font-mono text-xs text-slate-700">
        {label}
      </span>
    )
  return (
    <span className="rounded border border-amber-200 bg-amber-50 px-2 py-0.5 font-mono text-xs text-amber-700">
      {label}
    </span>
  )
}

function SuggestionRow({
  suggestion,
  index,
  onActionClick,
}: {
  suggestion: ColumnTypeSuggestion
  index: number
  onActionClick?: (prompt: string) => void
}) {
  // Render **bold** markdown in the reason text
  const parts = suggestion.reason.split(/(\*\*[^*]+\*\*)/)
  const renderedReason = parts.map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={i}>{part}</span>
    )
  )

  return (
    <div
      className="flex items-start gap-3 rounded-lg border border-slate-100 bg-slate-50/60 p-3"
      data-testid={`type-suggestion-row-${index}`}
    >
      {/* Column + type change */}
      <div className="flex-1 space-y-2">
        {/* Column name + dtype arrow */}
        <div className="flex flex-wrap items-center gap-2">
          <span
            className="text-sm font-semibold text-foreground"
            data-testid={`type-suggestion-column-${index}`}
          >
            {suggestion.column}
          </span>
          <DtypeBadge label={suggestion.current_dtype} variant="current" />
          <span aria-hidden="true" className="text-slate-400 text-xs">
            →
          </span>
          <DtypeBadge label={suggestion.suggested_dtype} variant="suggested" />
          <ConfidenceBadge confidence={suggestion.confidence} />
        </div>

        {/* Reason */}
        <p
          className="text-sm text-muted-foreground leading-relaxed"
          data-testid={`type-suggestion-reason-${index}`}
        >
          {renderedReason}
        </p>

        {/* Sample values */}
        {suggestion.sample_values.length > 0 && (
          <p className="text-xs text-slate-400" data-testid={`type-suggestion-samples-${index}`}>
            Samples: {suggestion.sample_values.slice(0, 4).map((v, i) => (
              <code key={i} className="mr-1 rounded bg-slate-100 px-1 py-0.5">
                {v}
              </code>
            ))}
          </p>
        )}

        {/* Fix button */}
        <button
          onClick={() => onActionClick?.(suggestion.suggested_action)}
          className="mt-1 inline-flex items-center gap-1 rounded-full border border-amber-200 bg-white px-3 py-1 text-xs font-medium text-amber-700 transition-colors hover:bg-amber-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={`Fix: ${suggestion.suggested_action}`}
          data-testid={`type-suggestion-fix-${index}`}
        >
          Fix this →
        </button>
      </div>
    </div>
  )
}

export function ColumnTypeSuggestionCard({
  result,
  onActionClick,
}: ColumnTypeSuggestionCardProps) {
  return (
    <figure
      className={`rounded-xl border-2 ${
        result.has_suggestions ? "border-amber-300" : "border-emerald-300"
      } bg-card p-4 shadow-sm`}
      aria-label={`Column type check for ${result.dataset_name}`}
      data-testid="column-type-suggestion-card"
    >
      {/* Header */}
      <div className="mb-3 flex items-center gap-2">
        <span
          aria-hidden="true"
          className="text-2xl leading-none"
          data-testid="column-type-icon"
        >
          {result.has_suggestions ? "⚠️" : "✅"}
        </span>
        <div className="flex-1">
          <h3
            className="text-sm font-bold text-foreground"
            data-testid="column-type-heading"
          >
            Column Type Check
          </h3>
          <div className="mt-0.5 flex flex-wrap gap-1.5">
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
              {result.dataset_name}
            </span>
            {result.has_suggestions ? (
              <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
                {result.suggestions.length}{" "}
                {result.suggestions.length === 1 ? "issue" : "issues"} found
              </span>
            ) : (
              <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
                All types correct
              </span>
            )}
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
              {result.dataset_cols} columns checked
            </span>
          </div>
        </div>
      </div>

      {/* Summary */}
      <p
        className="mb-3 text-sm text-muted-foreground"
        data-testid="column-type-summary"
      >
        {result.summary}
      </p>

      {/* Suggestion rows — only if there are issues */}
      {result.has_suggestions && result.suggestions.length > 0 && (
        <div className="space-y-2" data-testid="column-type-suggestions-list">
          {result.suggestions.map((suggestion, i) => (
            <SuggestionRow
              key={i}
              suggestion={suggestion}
              index={i}
              onActionClick={onActionClick}
            />
          ))}
        </div>
      )}

      {/* All-good state */}
      {!result.has_suggestions && (
        <p
          className="text-sm text-emerald-600 font-medium"
          data-testid="column-type-all-good"
        >
          ✓ No type mismatches detected. Your data is ready for analysis.
        </p>
      )}

      <figcaption className="sr-only">
        Column type check for {result.dataset_name}: {result.summary}
      </figcaption>
    </figure>
  )
}
