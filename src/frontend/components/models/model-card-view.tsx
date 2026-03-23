"use client"

import type { ModelCard } from "@/lib/types"

interface ModelCardViewProps {
  card: ModelCard
}

function ProblemTypeBadge({ type }: { type: string }) {
  const color =
    type === "classification"
      ? "bg-purple-100 text-purple-800"
      : "bg-blue-100 text-blue-800"
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${color}`}
    >
      {type === "classification" ? "Classification" : "Regression"}
    </span>
  )
}

function ImportanceBar({ feature, importance, rank }: { feature: string; importance: number; rank: number }) {
  const maxWidth = 100
  const width = Math.round(importance * maxWidth * 5) // scale for visual clarity
  const clampedWidth = Math.min(width, maxWidth)
  const blue = `hsl(${220 - rank * 20}, 70%, ${55 + rank * 5}%)`

  return (
    <div className="flex items-center gap-2 text-xs" data-testid="importance-bar">
      <span className="w-5 text-gray-400 font-mono text-right">{rank}</span>
      <span className="flex-1 min-w-0 truncate text-gray-700">{feature}</span>
      <div className="w-28 flex-shrink-0 flex items-center gap-1">
        <div className="flex-1 bg-gray-100 rounded-full h-2">
          <div
            className="h-2 rounded-full"
            style={{ width: `${clampedWidth}%`, backgroundColor: blue }}
          />
        </div>
        <span className="w-10 text-right text-gray-500">
          {(importance * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  )
}

export function ModelCardView({ card }: ModelCardViewProps) {
  return (
    <div
      className="mt-2 rounded-lg border border-indigo-200 bg-indigo-50 p-4 text-sm"
      data-testid="model-card-view"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-indigo-900">Model Explained</span>
          <ProblemTypeBadge type={card.problem_type} />
          {card.is_deployed && (
            <span className="inline-flex items-center gap-1 text-xs text-green-700 font-medium">
              <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
              Live
            </span>
          )}
        </div>
        <span className="text-xs text-indigo-600 font-mono">{card.algorithm_name}</span>
      </div>

      {/* Summary */}
      <p className="text-gray-700 mb-3 leading-relaxed">{card.summary}</p>

      {/* Metric */}
      <div className="mb-3 flex items-center gap-3 rounded-md bg-white border border-indigo-100 px-3 py-2">
        <div>
          <div className="text-lg font-bold text-indigo-700">{card.metric.display}</div>
          <div className="text-xs text-gray-500">{card.metric.name}</div>
        </div>
        <p className="text-xs text-gray-600 flex-1">{card.metric.plain_english}</p>
      </div>

      {/* Top features */}
      {card.top_features.length > 0 && (
        <div className="mb-3">
          <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1.5">
            What drives predictions
          </div>
          <div className="space-y-1.5">
            {card.top_features.map((f) => (
              <ImportanceBar
                key={f.feature}
                feature={f.feature}
                importance={f.importance}
                rank={f.rank}
              />
            ))}
          </div>
        </div>
      )}

      {/* Limitation */}
      {card.limitations.length > 0 && (
        <div className="rounded-md bg-amber-50 border border-amber-200 px-3 py-2 text-xs text-amber-800">
          <span className="font-medium">Keep in mind: </span>
          {card.limitations[0]}
        </div>
      )}

      {/* Footer stats */}
      <div className="mt-3 flex gap-4 text-xs text-gray-500 border-t border-indigo-100 pt-2">
        <span>Trained on <strong>{card.row_count.toLocaleString()}</strong> rows</span>
        <span><strong>{card.feature_count}</strong> input features</span>
        <span>Predicts <strong className="font-mono">{card.target_col}</strong></span>
      </div>
    </div>
  )
}
