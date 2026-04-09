"use client"

import { useCallback } from "react"
import { RankedPredictionsResult, RankedPredictionRow } from "@/lib/types"

function buildCsv(result: RankedPredictionsResult): string {
  const allFeatureCols =
    result.rows.length > 0 ? Object.keys(result.rows[0].feature_values) : []
  const predictionHeader =
    result.problem_type === "regression"
      ? `predicted_${result.target_column}`
      : `predicted_class,confidence_pct`

  const headers = ["rank", "row_index", predictionHeader, ...allFeatureCols].join(",")

  const escapeCell = (v: string | number | null | undefined): string => {
    const s = v === null || v === undefined ? "" : String(v)
    return s.includes(",") || s.includes('"') || s.includes("\n")
      ? `"${s.replace(/"/g, '""')}"`
      : s
  }

  const rows = result.rows.map((row) => {
    const predCells =
      result.problem_type === "regression"
        ? [escapeCell(row.prediction)]
        : [escapeCell(row.predicted_class), escapeCell(row.confidence !== undefined ? Math.round(row.confidence * 100) : "")]
    const featureCells = allFeatureCols.map((col) => escapeCell(row.feature_values[col]))
    return [escapeCell(row.rank), escapeCell(row.row_index), ...predCells, ...featureCells].join(",")
  })

  return [headers, ...rows].join("\n")
}

interface RankedPredictionsCardProps {
  result: RankedPredictionsResult
}

function fmtScore(v: number | undefined): string {
  if (v === undefined) return "—"
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(1)}k`
  return parseFloat(v.toFixed(3)).toString()
}

function fmtFeatureVal(v: string | number | null): string {
  if (v === null || v === undefined) return "—"
  if (typeof v === "number") return fmtScore(v)
  return String(v)
}

function confidenceBadge(confidence: number): string {
  if (confidence >= 0.8) return "bg-emerald-100 text-emerald-800"
  if (confidence >= 0.6) return "bg-sky-100 text-sky-800"
  if (confidence >= 0.4) return "bg-amber-100 text-amber-800"
  return "bg-rose-100 text-rose-800"
}

function RankBadge({ rank }: { rank: number }) {
  const base = "inline-flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold"
  if (rank === 1) return <span className={`${base} bg-amber-400 text-amber-900`}>1</span>
  if (rank === 2) return <span className={`${base} bg-slate-300 text-slate-700`}>2</span>
  if (rank === 3) return <span className={`${base} bg-orange-300 text-orange-800`}>3</span>
  return <span className={`${base} bg-slate-100 text-slate-600`}>{rank}</span>
}

function PredictionCell({ row, problemType }: { row: RankedPredictionRow; problemType: string }) {
  if (problemType === "regression") {
    return (
      <span className="font-semibold text-sky-700">
        {fmtScore(row.prediction)}
      </span>
    )
  }
  const conf = row.confidence ?? 0
  return (
    <span
      className={`rounded px-2 py-0.5 text-xs font-medium ${confidenceBadge(conf)}`}
    >
      {row.predicted_class} ({Math.round(conf * 100)}%)
    </span>
  )
}

export function RankedPredictionsCard({ result }: RankedPredictionsCardProps) {
  const dirLabel = result.direction === "highest" ? "Highest" : "Lowest"
  const displayCols = result.rows.length > 0 ? Object.keys(result.rows[0].feature_values) : []

  const handleDownloadCsv = useCallback(() => {
    const csv = buildCsv(result)
    const blob = new Blob([csv], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${result.target_column}_ranked_predictions.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [result])

  return (
    <figure aria-label={`Ranked predictions: top ${result.n} ${result.direction} ${result.target_column}`}>
      <div className="rounded-lg border-2 border-amber-400 bg-white p-4 shadow-sm">
        {/* Header */}
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span aria-hidden="true" className="text-lg">🏆</span>
          <h3 className="font-semibold text-slate-800">
            Top {result.n} Predictions — {result.target_column}
          </h3>
          <span className="rounded bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
            {dirLabel}
          </span>
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
            {result.n.toLocaleString()} of {result.total_scored.toLocaleString()} rows
          </span>
          <span className="rounded bg-sky-100 px-2 py-0.5 text-xs font-medium text-sky-700">
            {result.problem_type === "regression" ? "Regression" : "Classification"}
          </span>
          <button
            onClick={handleDownloadCsv}
            className="ml-auto rounded bg-amber-600 px-2 py-1 text-xs font-medium text-white hover:bg-amber-700"
            aria-label={`Download ranked predictions as CSV`}
          >
            ⬇ Download CSV
          </button>
        </div>

        {/* Ranked table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs font-medium text-slate-500">
                <th className="pb-2 pr-3">#</th>
                <th className="pb-2 pr-3">
                  {result.problem_type === "regression" ? "Predicted" : "Prediction"}
                </th>
                {displayCols.map((col) => (
                  <th key={col} className="pb-2 pr-3 max-w-[120px] truncate">
                    {col.replace(/_/g, " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row) => (
                <tr
                  key={row.row_index}
                  className="border-b border-slate-100 last:border-0 hover:bg-slate-50"
                >
                  <td className="py-2 pr-3">
                    <RankBadge rank={row.rank} />
                  </td>
                  <td className="py-2 pr-3">
                    <PredictionCell row={row} problemType={result.problem_type} />
                  </td>
                  {displayCols.map((col) => (
                    <td key={col} className="py-2 pr-3 text-slate-700 max-w-[120px] truncate">
                      {fmtFeatureVal(row.feature_values[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Summary footer */}
        <p className="mt-3 rounded bg-amber-50 px-3 py-2 text-xs text-amber-800 italic">
          {result.summary}
        </p>
      </div>
    </figure>
  )
}
