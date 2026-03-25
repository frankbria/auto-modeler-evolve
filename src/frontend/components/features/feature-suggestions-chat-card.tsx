"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import type {
  FeatureSuggestionsChatResult,
  FeaturesAppliedResult,
  FeatureSetResult,
} from "@/lib/types"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TRANSFORM_LABELS: Record<string, string> = {
  date_decompose: "Date parts",
  log_transform: "Log scale",
  one_hot: "One-hot encode",
  label_encode: "Label encode",
  bin_quartile: "Bin quartiles",
  interaction: "Interaction",
}

const TRANSFORM_COLORS: Record<string, string> = {
  date_decompose: "bg-blue-500/10 text-blue-700 dark:text-blue-400",
  log_transform: "bg-orange-500/10 text-orange-700 dark:text-orange-400",
  one_hot: "bg-green-500/10 text-green-700 dark:text-green-400",
  label_encode: "bg-teal-500/10 text-teal-700 dark:text-teal-400",
  bin_quartile: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
  interaction: "bg-purple-500/10 text-purple-700 dark:text-purple-400",
}

// ---------------------------------------------------------------------------
// FeatureSuggestCard — shows chat-triggered suggestions with Apply All button
// ---------------------------------------------------------------------------

interface FeatureSuggestCardProps {
  result: FeatureSuggestionsChatResult
}

export function FeatureSuggestCard({ result }: FeatureSuggestCardProps) {
  const [applied, setApplied] = useState<FeatureSetResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function applyAll() {
    setLoading(true)
    setError(null)
    try {
      const transforms = result.suggestions.map((s) => ({
        column: s.column,
        transform_type: s.transform_type,
      }))
      const data = await api.features.apply(result.dataset_id, transforms)
      setApplied(data)
    } catch {
      setError("Failed to apply features. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  if (applied) {
    return (
      <div
        className="mt-2 rounded-lg border border-purple-500/30 bg-purple-500/5 p-3"
        data-testid="features-applied-card"
      >
        <div className="mb-2 flex items-center gap-2">
          <span className="text-sm" aria-hidden="true">✅</span>
          <span className="text-xs font-semibold text-purple-700 dark:text-purple-400">
            Features Applied
          </span>
          <Badge className="bg-purple-100 text-purple-800 border-purple-200 dark:bg-purple-900/30 dark:text-purple-400 dark:border-purple-800">
            {applied.new_columns.length} new columns
          </Badge>
        </div>
        <p className="text-sm text-foreground">
          Your feature set is now active with{" "}
          <span className="font-semibold">{applied.total_columns} total columns</span>.
          {applied.new_columns.length > 0 && (
            <>
              {" "}New:{" "}
              <span className="font-mono text-xs">
                {applied.new_columns.slice(0, 4).join(", ")}
                {applied.new_columns.length > 4 && ` +${applied.new_columns.length - 4} more`}
              </span>
            </>
          )}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          Say &ldquo;train a model to predict [column]&rdquo; to start training with these features.
        </p>
      </div>
    )
  }

  return (
    <div
      className="mt-2 rounded-lg border border-purple-500/30 bg-purple-500/5 p-3"
      data-testid="feature-suggest-card"
    >
      {/* Header */}
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm" aria-hidden="true">⚙️</span>
        <span className="text-xs font-semibold text-purple-700 dark:text-purple-400">
          Feature Engineering Suggestions
        </span>
        <Badge className="bg-purple-100 text-purple-800 border-purple-200 dark:bg-purple-900/30 dark:text-purple-400 dark:border-purple-800">
          {result.count} suggestions
        </Badge>
      </div>

      {/* Suggestion list */}
      <ul className="mb-3 space-y-2">
        {result.suggestions.map((s) => (
          <li
            key={s.id}
            className="flex items-start gap-2 rounded-md border border-purple-500/10 bg-background/60 px-2 py-1.5"
          >
            <Badge
              className={`mt-0.5 shrink-0 ${TRANSFORM_COLORS[s.transform_type] ?? "bg-gray-100 text-gray-700"}`}
            >
              {TRANSFORM_LABELS[s.transform_type] ?? s.transform_type}
            </Badge>
            <div className="min-w-0">
              <p className="truncate text-xs font-medium text-foreground">{s.title}</p>
              <p className="text-xs text-muted-foreground">{s.description}</p>
              {s.preview_columns.length > 0 && (
                <p className="mt-0.5 text-[10px] text-muted-foreground">
                  Adds:{" "}
                  <span className="font-mono">
                    {s.preview_columns.slice(0, 3).join(", ")}
                    {s.preview_columns.length > 3 && " …"}
                  </span>
                </p>
              )}
            </div>
          </li>
        ))}
      </ul>

      {/* Apply All button */}
      <button
        onClick={applyAll}
        disabled={loading}
        className="flex w-full items-center justify-center gap-2 rounded-md border border-purple-500/30 bg-purple-500/10 px-3 py-2 text-sm font-medium text-purple-700 transition-colors hover:bg-purple-500/20 disabled:opacity-50 dark:text-purple-400"
        data-testid="apply-all-features-btn"
      >
        {loading ? "Applying…" : `Apply All ${result.count} Features`}
      </button>

      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// FeaturesAppliedCard — shown when apply happens via chat message
// ---------------------------------------------------------------------------

interface FeaturesAppliedCardProps {
  result: FeaturesAppliedResult
}

export function FeaturesAppliedCard({ result }: FeaturesAppliedCardProps) {
  return (
    <div
      className="mt-2 rounded-lg border border-purple-500/30 bg-purple-500/5 p-3"
      data-testid="features-applied-confirmation-card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="text-sm" aria-hidden="true">✅</span>
        <span className="text-xs font-semibold text-purple-700 dark:text-purple-400">
          Feature Engineering Done
        </span>
        <Badge className="bg-purple-100 text-purple-800 border-purple-200 dark:bg-purple-900/30 dark:text-purple-400 dark:border-purple-800">
          {result.applied_count} transforms applied
        </Badge>
      </div>
      <p className="text-sm text-foreground">
        Added{" "}
        <span className="font-semibold">{result.new_columns.length} new columns</span> —
        dataset now has{" "}
        <span className="font-semibold">{result.total_columns} total columns</span>.
      </p>
      {result.new_columns.length > 0 && (
        <p className="mt-1 text-xs text-muted-foreground font-mono">
          {result.new_columns.slice(0, 5).join(", ")}
          {result.new_columns.length > 5 && ` +${result.new_columns.length - 5} more`}
        </p>
      )}
    </div>
  )
}
