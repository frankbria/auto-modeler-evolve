"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import type { CleaningSuggestion, CleanOperation, CleanResult } from "@/lib/types"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const OP_LABELS: Record<string, string> = {
  remove_duplicates: "Remove duplicates",
  fill_missing: "Fill missing values",
  filter_rows: "Filter rows",
  cap_outliers: "Cap outliers",
  drop_column: "Drop column",
}

function operationDescription(op: CleanOperation): string {
  switch (op.operation) {
    case "remove_duplicates":
      return "Remove all exact duplicate rows from the dataset."
    case "fill_missing":
      return `Fill missing values in '${op.column}' using ${op.strategy}${op.strategy === "value" ? ` (${op.fill_value})` : ""}.`
    case "filter_rows":
      return `Remove rows where '${op.column}' ${op.operator} ${op.value}.`
    case "cap_outliers":
      return `Clip extreme values in '${op.column}' to the ${op.percentile ?? 99}th / ${100 - (op.percentile ?? 99)}th percentile.`
    case "drop_column":
      return `Permanently remove the '${op.column}' column.`
    default:
      return "Apply cleaning operation."
  }
}

// ---------------------------------------------------------------------------
// CleaningCard props
// ---------------------------------------------------------------------------

interface CleaningCardProps {
  /** Pre-computed suggestion pushed via chat SSE. */
  suggestion?: CleaningSuggestion
  /** Dataset ID — required to apply operations. */
  datasetId?: string
  /** Called after a successful clean so the parent can refresh preview. */
  onCleaned?: (result: CleanResult) => void
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function CleaningCard({ suggestion: initialSuggestion, datasetId, onCleaned }: CleaningCardProps) {
  const [suggestion] = useState<CleaningSuggestion | null>(initialSuggestion ?? null)
  const [result, setResult] = useState<CleanResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const canApply = !!datasetId && !!suggestion?.suggested_operation

  async function applyOperation() {
    if (!datasetId || !suggestion?.suggested_operation) return
    setLoading(true)
    setError(null)
    try {
      const data = await api.data.clean(datasetId, suggestion.suggested_operation)
      setResult(data)
      onCleaned?.(data)
    } catch {
      setError("Cleaning operation failed. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const qualSummary = suggestion?.quality_summary
  const hasIssues =
    (qualSummary?.duplicate_rows ?? 0) > 0 ||
    Object.keys(qualSummary?.missing_value_columns ?? {}).length > 0

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Data Cleaning</CardTitle>
          {hasIssues && (
            <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200 text-xs">
              Issues found
            </Badge>
          )}
          {!hasIssues && qualSummary && (
            <Badge className="bg-green-100 text-green-800 border-green-200 text-xs">
              Data looks clean
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* Applied result — show after successful operation */}
        {result && (
          <div className="rounded-md bg-green-50 border border-green-200 p-3 space-y-1">
            <p className="text-xs font-medium text-green-800">Done!</p>
            <p className="text-xs text-green-700">{result.operation_result.summary}</p>
            <p className="text-xs text-muted-foreground">
              {result.updated_stats.row_count} rows × {result.updated_stats.column_count} columns
            </p>
          </div>
        )}

        {/* Quality summary */}
        {!result && qualSummary && (
          <div className="space-y-1">
            {qualSummary.duplicate_rows > 0 && (
              <p className="text-xs text-muted-foreground">
                <span className="font-medium text-yellow-700">{qualSummary.duplicate_rows}</span>
                {" duplicate row(s) found"}
              </p>
            )}
            {Object.entries(qualSummary.missing_value_columns).slice(0, 4).map(([col, count]) => (
              <p key={col} className="text-xs text-muted-foreground">
                <span className="font-medium text-yellow-700">{count}</span>
                {" missing in "}
                <span className="font-mono">{col}</span>
              </p>
            ))}
            {!hasIssues && (
              <p className="text-xs text-green-700">
                No duplicate rows or missing values detected.
              </p>
            )}
          </div>
        )}

        {/* Suggested operation */}
        {!result && suggestion?.suggested_operation && (
          <div className="rounded-md bg-blue-50 border border-blue-200 p-3 space-y-2">
            <div className="flex items-center gap-2">
              <Badge className="bg-blue-100 text-blue-800 border-blue-200 text-xs">
                {OP_LABELS[suggestion.suggested_operation.operation] ?? suggestion.suggested_operation.operation}
              </Badge>
              <span className="text-xs text-muted-foreground">suggested</span>
            </div>
            <p className="text-xs text-blue-800">
              {operationDescription(suggestion.suggested_operation)}
            </p>
          </div>
        )}

        {/* Apply button */}
        {!result && canApply && (
          <Button
            size="sm"
            className="text-xs w-full"
            onClick={applyOperation}
            disabled={loading}
          >
            {loading ? "Applying…" : `Apply: ${OP_LABELS[suggestion!.suggested_operation!.operation]}`}
          </Button>
        )}

        {/* Error */}
        {error && <p className="text-xs text-red-600">{error}</p>}

        {/* Helpful hint when no specific op detected */}
        {!result && suggestion && !suggestion.suggested_operation && (
          <p className="text-xs text-muted-foreground">
            Ask me to: &ldquo;fill missing sales with median&rdquo;, &ldquo;remove duplicate rows&rdquo;,
            &ldquo;drop rows where quantity &lt; 0&rdquo;, or &ldquo;cap outliers in revenue at 99%&rdquo;.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
