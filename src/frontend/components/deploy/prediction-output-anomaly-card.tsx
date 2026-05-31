"use client"

import { PredictionOutputAnomalyResult, OutputAnomalyEntry } from "@/lib/types"

interface Props {
  result: PredictionOutputAnomalyResult
}

function relativeTime(isoStr: string): string {
  if (!isoStr) return ""
  try {
    const diff = Math.floor((Date.now() - new Date(isoStr).getTime()) / 1000)
    if (diff < 60) return "just now"
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
    return `${Math.floor(diff / 86400)}d ago`
  } catch {
    return ""
  }
}

function AnomalyRow({ entry, index }: { entry: OutputAnomalyEntry; index: number }) {
  const inputChips = Object.entries(entry.input_summary || {}).slice(0, 3)
  const timeLabel = relativeTime(entry.created_at)

  return (
    <div
      className="border border-gray-200 rounded-md p-3 text-sm"
      data-testid={`anomaly-row-${index}`}
      aria-label={`Anomalous prediction ${index + 1}: ${entry.reason}`}
    >
      <div className="flex items-start justify-between gap-2 mb-1">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono font-semibold text-rose-700" data-testid="prediction-value">
            {entry.prediction_value}
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 font-medium" data-testid="deviation-badge">
            {entry.deviation}
          </span>
          {entry.z_score !== null && (
            <span className="text-xs text-gray-500" aria-label={`z-score ${entry.z_score}`}>
              z = {entry.z_score}
            </span>
          )}
          {entry.confidence !== null && (
            <span className="text-xs text-gray-500" aria-label={`confidence ${entry.confidence}%`}>
              {entry.confidence}% confident
            </span>
          )}
        </div>
        {timeLabel && (
          <span className="text-xs text-gray-400 shrink-0">{timeLabel}</span>
        )}
      </div>
      <p className="text-gray-600 text-xs mb-2">{entry.reason}</p>
      {inputChips.length > 0 && (
        <div className="flex flex-wrap gap-1" aria-label="Input features">
          {inputChips.map(([k, v]) => (
            <span
              key={k}
              className="text-xs px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded font-mono"
            >
              {k}={String(v)}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export function PredictionOutputAnomalyCard({ result }: Props) {
  const { verdict, verdict_label, n_total, n_anomalous, anomaly_rate, problem_type, stats, anomalies, summary } = result

  const borderColor =
    verdict === "many_anomalies"
      ? "border-rose-300 bg-rose-50/30"
      : verdict === "few_anomalies"
      ? "border-amber-300 bg-amber-50/30"
      : verdict === "no_data"
      ? "border-gray-200 bg-gray-50/30"
      : "border-emerald-300 bg-emerald-50/30"

  const verdictColor =
    verdict === "many_anomalies"
      ? "bg-rose-100 text-rose-800"
      : verdict === "few_anomalies"
      ? "bg-amber-100 text-amber-800"
      : verdict === "no_data"
      ? "bg-gray-100 text-gray-600"
      : "bg-emerald-100 text-emerald-800"

  return (
    <figure
      className={`border rounded-lg p-4 mt-2 space-y-3 ${borderColor}`}
      aria-label="Prediction Output Anomaly Analysis"
      data-testid="prediction-output-anomaly-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-lg" aria-hidden="true">🔍</span>
        <span className="font-semibold text-gray-800">Prediction Output Anomalies</span>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${verdictColor}`} data-testid="verdict-badge">
          {verdict_label}
        </span>
        <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600" data-testid="problem-type-badge">
          {problem_type === "regression" ? "Regression" : "Classification"}
        </span>
        <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-600" data-testid="n-total-badge">
          {n_total} predictions analyzed
        </span>
      </div>

      {/* Stats row */}
      {verdict !== "no_data" && (
        <div className="grid grid-cols-3 gap-2 text-xs text-gray-600">
          {problem_type === "regression" && stats.mean !== undefined ? (
            <>
              <div className="bg-white rounded p-2 border border-gray-100">
                <div className="font-medium text-gray-800">{stats.mean?.toFixed(2)}</div>
                <div className="text-gray-400">Mean prediction</div>
              </div>
              <div className="bg-white rounded p-2 border border-gray-100">
                <div className="font-medium text-gray-800">{stats.std?.toFixed(2)}</div>
                <div className="text-gray-400">Std deviation</div>
              </div>
              <div className="bg-white rounded p-2 border border-gray-100">
                <div className={`font-medium ${n_anomalous > 0 ? "text-rose-700" : "text-emerald-700"}`} data-testid="anomaly-rate">
                  {n_anomalous} ({anomaly_rate}%)
                </div>
                <div className="text-gray-400">Anomalous</div>
              </div>
            </>
          ) : stats.mean_confidence !== undefined ? (
            <>
              <div className="bg-white rounded p-2 border border-gray-100">
                <div className="font-medium text-gray-800">{((stats.mean_confidence || 0) * 100).toFixed(0)}%</div>
                <div className="text-gray-400">Mean confidence</div>
              </div>
              <div className="bg-white rounded p-2 border border-gray-100">
                <div className="font-medium text-gray-800">{((stats.min_confidence || 0) * 100).toFixed(0)}%</div>
                <div className="text-gray-400">Min confidence</div>
              </div>
              <div className="bg-white rounded p-2 border border-gray-100">
                <div className={`font-medium ${n_anomalous > 0 ? "text-rose-700" : "text-emerald-700"}`} data-testid="anomaly-rate">
                  {n_anomalous} ({anomaly_rate}%)
                </div>
                <div className="text-gray-400">Low confidence</div>
              </div>
            </>
          ) : null}
        </div>
      )}

      {/* Anomaly list */}
      {anomalies.length > 0 ? (
        <div className="space-y-2" role="list" aria-label="Anomalous predictions">
          <div className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            {problem_type === "regression" ? "Most extreme predictions" : "Lowest-confidence predictions"}
          </div>
          {anomalies.map((entry, i) => (
            <div key={entry.id + i} role="listitem">
              <AnomalyRow entry={entry} index={i} />
            </div>
          ))}
        </div>
      ) : verdict !== "no_data" ? (
        <div className="text-sm text-emerald-700 py-2 text-center">
          ✓ All recent predictions are within the normal range
        </div>
      ) : null}

      {/* Summary */}
      <p className="text-sm text-gray-600">{summary}</p>

      <figcaption className="sr-only">
        Prediction output anomaly analysis: {verdict_label}. {summary}
      </figcaption>
    </figure>
  )
}
