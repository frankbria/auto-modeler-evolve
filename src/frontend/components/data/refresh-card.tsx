"use client"

import { useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import type { DatasetRefreshResult, RefreshPrompt } from "@/lib/types"

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface RefreshCardProps {
  /** Dataset to replace. Required to call the refresh endpoint. */
  datasetId: string
  /** Optional pre-computed prompt from chat SSE (shows current dataset info). */
  prompt?: RefreshPrompt
  /** Called after a successful refresh so parent can reload preview / stats. */
  onRefreshed?: (result: DatasetRefreshResult) => void
}

// ---------------------------------------------------------------------------
// RefreshCard
// ---------------------------------------------------------------------------

export function RefreshCard({ datasetId, prompt, onRefreshed }: RefreshCardProps) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [result, setResult] = useState<DatasetRefreshResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.data.refresh(datasetId, file)
      const maybeError = res as unknown as { detail?: string }
      if (maybeError.detail) {
        setError(maybeError.detail)
      } else {
        setResult(res)
        onRefreshed?.(res)
      }
    } catch {
      setError("Failed to upload file. Please try again.")
    } finally {
      setLoading(false)
      if (fileRef.current) fileRef.current.value = ""
    }
  }

  return (
    <Card data-testid="refresh-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          Replace Dataset
          {result && (
            <Badge
              variant={result.compatible ? "default" : "destructive"}
              className="text-xs"
            >
              {result.compatible ? "Compatible" : "Incompatible"}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Current dataset info from chat SSE prompt or from result */}
        {!result && prompt && (
          <p className="text-xs text-muted-foreground">
            Current file: <span className="font-medium">{prompt.current_filename}</span>{" "}
            ({prompt.current_row_count.toLocaleString()} rows). Upload a new file to replace
            it — your model configuration will be preserved.
          </p>
        )}

        {!result && !prompt && (
          <p className="text-xs text-muted-foreground">
            Upload a new CSV to replace the current dataset in-place. Your feature
            engineering and model history will be kept.
          </p>
        )}

        {/* Success result */}
        {result && (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground">
              <span className="font-medium">{result.filename}</span> —{" "}
              {result.row_count.toLocaleString()} rows, {result.column_count} columns
            </p>

            {result.new_columns.length > 0 && (
              <div className="text-xs">
                <span className="text-green-600 font-medium">New columns: </span>
                {result.new_columns.join(", ")}
              </div>
            )}

            {result.removed_columns.length > 0 && (
              <div className="text-xs">
                <span className="text-amber-600 font-medium">Removed columns: </span>
                {result.removed_columns.join(", ")}
              </div>
            )}

            {result.feature_columns_missing.length > 0 && (
              <div className="text-xs text-destructive">
                <span className="font-medium">Model feature columns missing: </span>
                {result.feature_columns_missing.join(", ")}
                <br />
                <span className="text-muted-foreground">
                  These columns were used to train your model. Retraining will require you to
                  re-select features.
                </span>
              </div>
            )}

            {result.compatible && result.feature_columns_missing.length === 0 && (
              <p className="text-xs text-green-600">
                All model feature columns are present. You can retrain immediately.
              </p>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <p className="text-xs text-destructive" data-testid="refresh-error">
            {error}
          </p>
        )}

        {/* Hidden file input */}
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          data-testid="refresh-file-input"
          onChange={handleFileChange}
        />

        <Button
          size="sm"
          variant={result ? "outline" : "default"}
          disabled={loading}
          onClick={() => fileRef.current?.click()}
          data-testid="replace-data-button"
        >
          {loading ? "Uploading…" : result ? "Replace Again" : "Choose New File"}
        </Button>
      </CardContent>
    </Card>
  )
}
