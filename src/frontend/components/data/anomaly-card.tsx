"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import type { AnomalyResult, AnomalyRecord } from "@/lib/types"

// ---------------------------------------------------------------------------
// Score badge
// ---------------------------------------------------------------------------

function ScoreBadge({ score }: { score: number }) {
  if (score >= 80)
    return <Badge className="bg-red-100 text-red-800 border-red-200 text-xs">High {score.toFixed(0)}</Badge>
  if (score >= 50)
    return <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200 text-xs">Medium {score.toFixed(0)}</Badge>
  return <Badge className="bg-gray-100 text-gray-700 border-gray-200 text-xs">Low {score.toFixed(0)}</Badge>
}

// ---------------------------------------------------------------------------
// AnomalyCard props
// ---------------------------------------------------------------------------

interface AnomalyCardProps {
  /** Pre-computed result pushed via SSE (chat-triggered). */
  result?: AnomalyResult
  /** Dataset ID — when provided, shows a manual detection button. */
  datasetId?: string
  /** Numeric column names available for detection. */
  numericFeatures?: string[]
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function AnomalyCard({ result: initialResult, datasetId, numericFeatures }: AnomalyCardProps) {
  const [result, setResult] = useState<AnomalyResult | null>(initialResult ?? null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(false)

  const canDetect = !!datasetId && (numericFeatures?.length ?? 0) > 0

  async function runDetection() {
    if (!datasetId || !numericFeatures?.length) return
    setLoading(true)
    setError(null)
    try {
      const data = await api.data.detectAnomalies(datasetId, numericFeatures)
      setResult(data)
    } catch {
      setError("Anomaly detection failed. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const displayedAnomalies: AnomalyRecord[] = result
    ? (showAll ? result.top_anomalies : result.top_anomalies.slice(0, 5))
    : []

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Anomaly Detection</CardTitle>
          {result && (
            <Badge
              className={
                result.anomaly_count > 0
                  ? "bg-orange-100 text-orange-800 border-orange-200"
                  : "bg-green-100 text-green-800 border-green-200"
              }
            >
              {result.anomaly_count > 0
                ? `${result.anomaly_count} unusual row${result.anomaly_count !== 1 ? "s" : ""}`
                : "All rows normal"}
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* Summary */}
        {result && (
          <p className="text-xs text-muted-foreground">{result.summary}</p>
        )}

        {/* Features used */}
        {result && result.features_used.length > 0 && (
          <p className="text-xs text-muted-foreground">
            <span className="font-medium">Features analysed:</span>{" "}
            {result.features_used.join(", ")}
          </p>
        )}

        {/* Top anomalies table */}
        {result && result.top_anomalies.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium">Top unusual rows:</p>
            <div className="overflow-x-auto rounded border border-border">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border bg-muted/40">
                    <th className="px-2 py-1 text-left font-medium text-muted-foreground">Row</th>
                    <th className="px-2 py-1 text-left font-medium text-muted-foreground">Score</th>
                    {result.features_used.slice(0, 4).map((f) => (
                      <th key={f} className="px-2 py-1 text-left font-medium text-muted-foreground truncate max-w-[80px]">
                        {f.length > 8 ? f.slice(0, 8) + "…" : f}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {displayedAnomalies.map((rec) => (
                    <tr
                      key={rec.row_index}
                      className={`border-b border-border last:border-0 ${rec.is_anomaly ? "bg-orange-50/40" : ""}`}
                    >
                      <td className="px-2 py-1 tabular-nums text-muted-foreground">
                        {rec.row_index + 1}
                      </td>
                      <td className="px-2 py-1">
                        <ScoreBadge score={rec.anomaly_score} />
                      </td>
                      {result.features_used.slice(0, 4).map((f) => (
                        <td key={f} className="px-2 py-1 tabular-nums">
                          {rec.values[f] != null
                            ? Number(rec.values[f]).toLocaleString(undefined, { maximumFractionDigits: 2 })
                            : <span className="text-muted-foreground">—</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {result.top_anomalies.length > 5 && (
              <button
                className="text-xs text-primary underline-offset-2 hover:underline"
                onClick={() => setShowAll((v) => !v)}
              >
                {showAll
                  ? "Show less"
                  : `Show ${result.top_anomalies.length - 5} more`}
              </button>
            )}
          </div>
        )}

        {/* No anomalies found */}
        {result && result.anomaly_count === 0 && (
          <p className="text-xs text-green-700">
            No multi-dimensional anomalies detected in the top {result.total_rows} rows.
          </p>
        )}

        {/* Manual detection button */}
        {canDetect && (
          <Button
            size="sm"
            variant="outline"
            className="text-xs w-full"
            onClick={runDetection}
            disabled={loading}
          >
            {loading ? "Scanning…" : result ? "Re-scan for anomalies" : "Scan for anomalies"}
          </Button>
        )}

        {/* Error */}
        {error && <p className="text-xs text-red-600">{error}</p>}
      </CardContent>
    </Card>
  )
}
