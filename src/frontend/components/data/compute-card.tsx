"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import type { ComputedColumnSuggestion, ComputeResult } from "@/lib/types"

// ---------------------------------------------------------------------------
// ComputeCard — shows a suggested computed column before the user confirms it
// ---------------------------------------------------------------------------

interface ComputeCardProps {
  /** Pre-computed suggestion pushed via chat SSE. */
  suggestion: ComputedColumnSuggestion
  /** Called after the column is successfully added. */
  onComputed?: (result: ComputeResult) => void
}

export function ComputeCard({ suggestion, onComputed }: ComputeCardProps) {
  const [result, setResult] = useState<ComputeResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function applyColumn() {
    setLoading(true)
    setError(null)
    try {
      const data = await api.data.computeColumn(
        suggestion.dataset_id,
        suggestion.name,
        suggestion.expression
      )
      setResult(data)
      onComputed?.(data)
    } catch {
      setError("Failed to add the computed column. Please check the expression and try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card data-testid="compute-card">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Computed Column</CardTitle>
          <Badge className="bg-purple-100 text-purple-800 border-purple-200 text-xs">
            New column
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* Success state */}
        {result && (
          <div className="rounded-md bg-green-50 border border-green-200 p-3 space-y-1">
            <p className="text-xs font-medium text-green-800">Column added!</p>
            <p className="text-xs text-green-700">{result.compute_result.summary}</p>
            <p className="text-xs text-muted-foreground">
              {result.updated_stats.row_count} rows × {result.updated_stats.column_count} columns
            </p>
          </div>
        )}

        {/* Formula display */}
        {!result && (
          <div className="space-y-2">
            <div className="rounded-md bg-purple-50 border border-purple-200 p-3 space-y-2">
              <p className="text-xs text-muted-foreground">New column name</p>
              <p className="text-sm font-mono font-semibold text-purple-900">
                {suggestion.name}
              </p>
              <p className="text-xs text-muted-foreground">Formula</p>
              <p className="text-sm font-mono text-purple-800">
                = {suggestion.expression}
              </p>
            </div>

            {/* Sample preview */}
            {suggestion.sample_values.length > 0 && (
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">
                  Preview ({suggestion.dtype})
                </p>
                <div className="flex gap-2 flex-wrap">
                  {suggestion.sample_values.slice(0, 5).map((v, i) => (
                    <Badge
                      key={i}
                      variant="outline"
                      className="font-mono text-xs"
                    >
                      {v === null ? "null" : String(typeof v === "number" ? v.toFixed(4).replace(/\.?0+$/, "") : v)}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Apply button */}
        {!result && (
          <Button
            size="sm"
            className="text-xs w-full"
            onClick={applyColumn}
            disabled={loading}
          >
            {loading ? "Adding column…" : `Add column '${suggestion.name}'`}
          </Button>
        )}

        {/* Error */}
        {error && <p className="text-xs text-red-600">{error}</p>}
      </CardContent>
    </Card>
  )
}
