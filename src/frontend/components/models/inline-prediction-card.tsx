"use client"

import type { GuardRailWarning, InlinePredictionResult } from "@/lib/types"

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

function severityLabel(severity: GuardRailWarning["severity"]): string {
  if (severity === "extreme_outlier") return "Extreme outlier"
  if (severity === "unknown_category") return "Unknown category"
  return "Out of range"
}

function severityColor(severity: GuardRailWarning["severity"]): string {
  if (severity === "extreme_outlier") return "border-red-300 bg-red-50 text-red-800"
  if (severity === "unknown_category") return "border-orange-300 bg-orange-50 text-orange-800"
  return "border-amber-300 bg-amber-50 text-amber-800"
}

export function InlinePredictionCard({ result }: InlinePredictionCardProps) {
  const isClassification = !!result.probabilities
  const target = result.target_column ?? "output"
  const warnings = result.guard_rail_warnings ?? []
  const hasWarnings = warnings.length > 0
  const borderClass = hasWarnings
    ? "border-amber-300 bg-amber-50"
    : "border-blue-200 bg-blue-50"

  return (
    <figure
      className={`mt-3 rounded-xl border p-4 shadow-sm max-w-md ${borderClass}`}
      aria-label={`Inline prediction result for ${target}`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">
          🔮
        </span>
        <div>
          <p className={`font-semibold text-sm leading-tight ${hasWarnings ? "text-amber-900" : "text-blue-900"}`}>
            Prediction Result
          </p>
          <p className={`text-xs capitalize ${hasWarnings ? "text-amber-600" : "text-blue-600"}`}>{target}</p>
        </div>
        {hasWarnings && (
          <span
            className="ml-auto inline-flex items-center gap-1 px-2 py-0.5 rounded-full border border-amber-400 bg-amber-100 text-amber-800 text-xs font-medium"
            aria-label={`${warnings.length} input warning${warnings.length !== 1 ? "s" : ""}`}
          >
            <span aria-hidden="true">⚠️</span>{" "}
            {warnings.length} warning{warnings.length !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Guard-rail warnings */}
      {hasWarnings && (
        <div className="mb-3 space-y-1" aria-label="Input validation warnings">
          {warnings.map((w, i) => (
            <div
              key={i}
              className={`rounded-lg border px-3 py-2 text-xs ${severityColor(w.severity)}`}
              role="alert"
            >
              <span className="font-semibold">{severityLabel(w.severity)}: </span>
              {w.message}
              {w.severity !== "unknown_category" && w.expected_min != null && w.expected_max != null && (
                <span className="block mt-0.5 opacity-75">
                  Typical training range: {formatValue(w.expected_min)} – {formatValue(w.expected_max)}
                </span>
              )}
              {w.severity === "unknown_category" && w.known_categories && w.known_categories.length > 0 && (
                <span className="block mt-0.5 opacity-75">
                  Known values: {w.known_categories.slice(0, 5).join(", ")}
                  {w.known_categories.length > 5 ? `, +${w.known_categories.length - 5} more` : ""}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

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
        <div className={`mb-3 text-center py-2 bg-white rounded-lg border ${hasWarnings ? "border-amber-200" : "border-blue-200"}`}>
          <p className={`text-2xl font-bold ${hasWarnings ? "text-amber-900" : "text-blue-900"}`}>
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
