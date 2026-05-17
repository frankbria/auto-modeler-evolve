import type { ConfidenceThresholdConfig } from "@/lib/types"

interface ConfidenceThresholdCardProps {
  config: ConfidenceThresholdConfig
}

export function ConfidenceThresholdCard({ config }: ConfidenceThresholdCardProps) {
  const {
    confidence_threshold,
    threshold_enabled,
    below_threshold_count_30d,
    total_predictions_30d,
    below_threshold_pct,
    summary,
  } = config

  const thresholdDisplay =
    confidence_threshold !== null
      ? `${(confidence_threshold * 100).toFixed(0)}%`
      : null

  const hasActivity = total_predictions_30d > 0

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-amber-200 bg-amber-50/50 p-3 text-sm"
      aria-label="Confidence threshold card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">🎯</span>
        <span className="font-semibold text-amber-900">Confidence Threshold</span>
        {threshold_enabled && thresholdDisplay ? (
          <span className="rounded bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
            Min {thresholdDisplay} confidence
          </span>
        ) : (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            Disabled
          </span>
        )}
      </div>

      <p className="mb-3 text-xs text-amber-800">{summary}</p>

      {threshold_enabled && thresholdDisplay && (
        <div className="mb-3 rounded bg-amber-100/60 px-3 py-2 text-xs text-amber-900">
          <span aria-hidden="true">⚠️</span> Predictions with model confidence below{" "}
          <strong>{thresholdDisplay}</strong> will be flagged as unreliable. The API
          response will include <code className="rounded bg-amber-200 px-1">below_threshold: true</code>{" "}
          and a plain-English explanation.
        </div>
      )}

      {hasActivity && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium text-slate-700">Last 30 days</span>
            <span className="text-slate-600">
              {total_predictions_30d.toLocaleString()} predictions
            </span>
          </div>
          {threshold_enabled && below_threshold_count_30d > 0 && (
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-600">Below threshold</span>
              <span className="rounded bg-amber-100 px-2 py-0.5 font-semibold text-amber-800">
                {below_threshold_count_30d.toLocaleString()}
                {below_threshold_pct !== null ? ` (${below_threshold_pct.toFixed(1)}%)` : ""}
              </span>
            </div>
          )}
          {threshold_enabled && below_threshold_count_30d === 0 && (
            <div className="text-xs text-emerald-700">
              All recent predictions met the confidence threshold.
            </div>
          )}
        </div>
      )}

      {!hasActivity && (
        <p className="text-xs text-slate-500">
          No predictions in the last 30 days yet.
        </p>
      )}

      <p className="mt-3 border-t border-amber-100 pt-2 text-xs text-slate-500">
        To configure: say &ldquo;set confidence threshold to 80%&rdquo; or &ldquo;only accept
        predictions above 70% confidence&rdquo;. Say &ldquo;disable confidence threshold&rdquo; to
        remove. Applies to classification models only.
      </p>
    </div>
  )
}
