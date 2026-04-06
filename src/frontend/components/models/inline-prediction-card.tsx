"use client"

import type { InlinePredictionResult } from "@/lib/types"

interface InlinePredictionCardProps {
  result: InlinePredictionResult
}

function formatValue(v: number | string): string {
  if (typeof v === "number") {
    if (Math.abs(v) >= 1_000_000) return (v / 1_000_000).toFixed(2) + "M"
    if (Math.abs(v) >= 1_000) return (v / 1_000).toFixed(1) + "k"
    if (Number.isInteger(v)) return v.toString()
    return v.toFixed(4).replace(/\.?0+$/, "")
  }
  return String(v)
}

export function InlinePredictionCard({ result }: InlinePredictionCardProps) {
  const isClassification = !!result.probabilities
  const target = result.target_column ?? "output"

  return (
    <figure
      className="mt-3 rounded-xl border border-blue-200 bg-blue-50 p-4 shadow-sm max-w-md"
      aria-label={`Inline prediction result for ${target}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">
          🔮
        </span>
        <div>
          <p className="font-semibold text-blue-900 text-sm leading-tight">
            Prediction Result
          </p>
          <p className="text-xs text-blue-600 capitalize">{target}</p>
        </div>
      </div>

      {/* Main prediction value */}
      {isClassification && result.probabilities ? (
        <div className="mb-3">
          {Object.entries(result.probabilities)
            .sort(([, a], [, b]) => b - a)
            .map(([cls, prob]) => (
              <div key={cls} className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-blue-800 w-24 truncate">
                  {cls}
                </span>
                <div className="flex-1 bg-blue-100 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{ width: `${Math.round(prob * 100)}%` }}
                  />
                </div>
                <span className="text-xs text-blue-700 w-10 text-right">
                  {Math.round(prob * 100)}%
                </span>
              </div>
            ))}
        </div>
      ) : (
        <div className="mb-3 text-center py-2 bg-white rounded-lg border border-blue-200">
          <p className="text-2xl font-bold text-blue-900">
            {formatValue(result.prediction as number)}
          </p>
          {result.confidence_interval && (
            <p className="text-xs text-blue-500 mt-0.5">
              95% interval:{" "}
              {formatValue(result.confidence_interval.lower)} –{" "}
              {formatValue(result.confidence_interval.upper)}
            </p>
          )}
          {result.confidence && !result.confidence_interval && (
            <p className="text-xs text-blue-500 mt-0.5">
              Confidence: {Math.round(result.confidence * 100)}%
            </p>
          )}
        </div>
      )}

      {/* Provided inputs */}
      {Object.keys(result.provided_features).length > 0 && (
        <div className="mb-2">
          <p className="text-xs font-semibold text-blue-700 mb-1">
            Inputs you provided:
          </p>
          <div className="flex flex-wrap gap-1">
            {Object.entries(result.provided_features).map(([k, v]) => (
              <span
                key={k}
                className="inline-flex items-center px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 text-xs font-medium"
              >
                {k.replace(/_/g, " ")}={formatValue(v as number | string)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Defaults note */}
      {result.defaults_used_count > 0 && (
        <p className="text-xs text-blue-500">
          <span aria-hidden="true">ℹ️ </span>
          {result.defaults_used_count} feature
          {result.defaults_used_count !== 1 ? "s" : ""} used training-data
          averages as defaults.
        </p>
      )}
    </figure>
  )
}
