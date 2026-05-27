"use client"

import { SimilarRecordsResult, SimilarRecordNeighbor } from "@/lib/types"

interface SimilarRecordsCardProps {
  data: SimilarRecordsResult
}

function SimilarityBar({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const colorClass =
    pct >= 80
      ? "bg-emerald-500"
      : pct >= 60
        ? "bg-blue-500"
        : pct >= 40
          ? "bg-amber-500"
          : "bg-gray-400"
  return (
    <div className="flex items-center gap-2" aria-label={`${pct}% similarity`}>
      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${colorClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs tabular-nums text-gray-600 w-9 text-right">{pct}%</span>
    </div>
  )
}

function NeighborRow({
  neighbor,
  queryPrediction,
  featureColumns,
}: {
  neighbor: SimilarRecordNeighbor
  queryPrediction: string
  featureColumns: string[]
}) {
  const sameOutcome = neighbor.predicted_label === queryPrediction
  const outcomeBadge = sameOutcome ? (
    <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
      {neighbor.predicted_label}
    </span>
  ) : (
    <span className="text-xs px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
      {neighbor.predicted_label}
    </span>
  )

  // Show top 3 feature values inline
  const previewFeats = featureColumns.slice(0, 3)

  return (
    <tr
      className="border-b border-gray-100 last:border-0 hover:bg-gray-50 transition-colors"
      data-testid="neighbor-row"
    >
      <td className="py-2 pr-3 text-xs text-gray-500 tabular-nums">#{neighbor.row_index}</td>
      <td className="py-2 pr-3 min-w-[100px]">
        <SimilarityBar score={neighbor.similarity_score} />
      </td>
      <td className="py-2 pr-3">{outcomeBadge}</td>
      <td className="py-2 text-xs text-gray-500">
        {previewFeats.map((col) => {
          const v = neighbor.features[col]
          const display =
            typeof v === "number"
              ? Math.abs(v) >= 1000
                ? `${(v / 1000).toFixed(1)}k`
                : v.toFixed(2)
              : String(v ?? "—")
          return (
            <span key={col} className="mr-2 whitespace-nowrap">
              <span className="text-gray-400">{col.replace(/_/g, " ")}:</span> {display}
            </span>
          )
        })}
      </td>
    </tr>
  )
}

export function SimilarRecordsCard({ data }: SimilarRecordsCardProps) {
  const topSim = data.neighbors[0]?.similarity_score ?? 0
  const sameCount = data.neighbors.filter(
    (n) => n.predicted_label === data.query_prediction
  ).length

  const headerBadgeClass =
    sameCount === data.neighbors.length
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : sameCount === 0
        ? "bg-amber-50 text-amber-700 border-amber-200"
        : "bg-blue-50 text-blue-700 border-blue-200"

  return (
    <figure
      className="rounded-lg border-2 border-blue-200 bg-blue-50 p-4 my-2"
      aria-label={`Similar records for row ${data.query_row_index}`}
      data-testid="similar-records-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span className="text-base" aria-hidden="true">🔍</span>
        <span className="font-semibold text-gray-800 text-sm">
          Similar Records — Row {data.query_row_index}
        </span>
        <span
          className={`text-xs px-2 py-0.5 rounded-full font-medium border ${headerBadgeClass}`}
        >
          {sameCount}/{data.neighbors.length} same outcome
        </span>
        <span className="text-xs px-2 py-0.5 rounded-full bg-white text-gray-600 border border-gray-200">
          Top similarity {Math.round(topSim * 100)}%
        </span>
      </div>

      {/* Query prediction box */}
      <div className="flex gap-3 mb-4">
        <div className="rounded-md bg-white border border-blue-200 px-4 py-2 text-center">
          <p className="text-xs text-gray-500 mb-0.5">Query prediction</p>
          <p className="text-base font-bold text-gray-800" data-testid="query-prediction">
            {data.query_prediction}
          </p>
          <p className="text-xs text-gray-400">
            {(data.query_confidence * 100).toFixed(0)}% confidence
          </p>
        </div>
        <div className="flex-1 rounded-md bg-white border border-gray-200 px-3 py-2">
          <p className="text-xs text-gray-500 mb-1">Summary</p>
          <p className="text-xs text-gray-700 leading-relaxed">{data.summary}</p>
        </div>
      </div>

      {/* Neighbors table */}
      {data.neighbors.length > 0 ? (
        <div className="rounded-md bg-white border border-blue-100 overflow-x-auto">
          <table className="w-full text-sm" aria-label="Similar records table">
            <thead>
              <tr className="border-b border-gray-100">
                <th className="py-2 pr-3 text-left text-xs font-medium text-gray-500">Row</th>
                <th className="py-2 pr-3 text-left text-xs font-medium text-gray-500 min-w-[120px]">
                  Similarity
                </th>
                <th className="py-2 pr-3 text-left text-xs font-medium text-gray-500">Outcome</th>
                <th className="py-2 text-left text-xs font-medium text-gray-500">Key features</th>
              </tr>
            </thead>
            <tbody>
              {data.neighbors.map((nb) => (
                <NeighborRow
                  key={nb.row_index}
                  neighbor={nb}
                  queryPrediction={data.query_prediction}
                  featureColumns={data.feature_columns}
                />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-sm text-gray-500 italic">No similar records found.</p>
      )}
    </figure>
  )
}
