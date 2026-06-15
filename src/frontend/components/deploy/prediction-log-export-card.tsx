"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { PredictionLogExportResult } from "@/lib/types"

// ---------------------------------------------------------------------------
// PredictionLogExportCard — inline chat card for downloading prediction history
// ---------------------------------------------------------------------------

interface PredictionLogExportCardProps {
  result: PredictionLogExportResult
}

function formatDate(iso: string | null): string {
  if (!iso) return "—"
  return new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

export function PredictionLogExportCard({ result }: PredictionLogExportCardProps) {
  const { total_predictions, download_url, first_prediction_at, last_prediction_at } = result
  const isEmpty = total_predictions === 0

  return (
    <Card
      className="border-emerald-300 bg-emerald-50 w-full max-w-md"
      aria-label="Prediction log export"
    >
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold text-emerald-800">
          <span aria-hidden="true">⬇</span>
          Prediction Log Export
        </CardTitle>
        <div className="flex flex-wrap gap-1 mt-1">
          <Badge className="bg-emerald-100 text-emerald-700 border border-emerald-200 text-xs">
            {total_predictions.toLocaleString()} prediction{total_predictions !== 1 ? "s" : ""}
          </Badge>
          <Badge className="bg-slate-100 text-slate-600 border border-slate-200 text-xs">
            CSV
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-3 text-sm">
        {isEmpty ? (
          <p className="text-slate-500 italic">
            No predictions recorded yet. Export will be available once the model receives API
            requests.
          </p>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-2 text-xs text-slate-600">
              <div>
                <span className="font-medium block">First prediction</span>
                <span>{formatDate(first_prediction_at)}</span>
              </div>
              <div>
                <span className="font-medium block">Last prediction</span>
                <span>{formatDate(last_prediction_at)}</span>
              </div>
            </div>

            <p className="text-slate-600 text-xs">
              Includes all input features, predictions, confidence scores, and response times in
              spreadsheet-ready format.
            </p>

            <a
              href={download_url}
              download
              className="inline-flex items-center gap-1.5 rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-700 transition-colors"
              aria-label={`Download ${total_predictions.toLocaleString()} prediction records as CSV`}
            >
              <span aria-hidden="true">⬇</span>
              Download CSV
            </a>
          </>
        )}

        <figcaption className="sr-only">
          {isEmpty
            ? "Prediction log export: no records available yet."
            : `Prediction log export: ${total_predictions.toLocaleString()} records from ${formatDate(first_prediction_at)} to ${formatDate(last_prediction_at)}.`}
        </figcaption>
      </CardContent>
    </Card>
  )
}
