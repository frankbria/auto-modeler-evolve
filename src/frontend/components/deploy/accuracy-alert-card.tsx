import type { AccuracyAlertConfig } from "@/lib/types"

interface AccuracyAlertCardProps {
  config: AccuracyAlertConfig
}

export function AccuracyAlertCard({ config }: AccuracyAlertCardProps) {
  const {
    accuracy_alert_enabled,
    accuracy_alert_threshold,
    accuracy_alert_fired,
    problem_type,
    metric_label,
    current_metric,
    n_feedback,
    summary,
  } = config

  const thresholdDisplay =
    accuracy_alert_threshold !== null
      ? problem_type === "regression"
        ? `${accuracy_alert_threshold}% error`
        : `${(accuracy_alert_threshold * 100).toFixed(0)}% accuracy`
      : null

  const currentDisplay =
    current_metric !== null
      ? problem_type === "regression"
        ? `${current_metric.toFixed(1)}%`
        : `${(current_metric * 100).toFixed(1)}%`
      : null

  const isBreach =
    current_metric !== null &&
    accuracy_alert_threshold !== null &&
    (problem_type === "regression"
      ? current_metric > accuracy_alert_threshold
      : current_metric < accuracy_alert_threshold)

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-amber-200 bg-amber-50/50 p-3 text-sm"
      aria-label="Accuracy alert card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">🎯</span>
        <span className="font-semibold text-amber-900">Accuracy Alert</span>
        {accuracy_alert_enabled ? (
          <span className="rounded bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
            Alert at {thresholdDisplay}
          </span>
        ) : (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            Disabled
          </span>
        )}
        {accuracy_alert_fired && (
          <span className="rounded bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700">
            ⚠ Alert fired
          </span>
        )}
      </div>

      <p className="mb-3 text-xs text-amber-800">{summary}</p>

      {accuracy_alert_enabled && thresholdDisplay && (
        <div className="mb-3 rounded bg-amber-100/60 px-3 py-2 text-xs text-amber-900">
          <span aria-hidden="true">⚠️</span> You will receive a webhook notification
          when {metric_label} {problem_type === "regression" ? "exceeds" : "drops below"}{" "}
          <strong>{thresholdDisplay}</strong>.
        </div>
      )}

      {current_metric !== null && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium text-slate-700">
              Current {metric_label}
            </span>
            <span
              className={`rounded px-2 py-0.5 font-semibold ${
                isBreach
                  ? "bg-red-100 text-red-800"
                  : "bg-emerald-100 text-emerald-800"
              }`}
            >
              {currentDisplay}
            </span>
          </div>
          <div className="flex justify-between text-xs text-slate-500">
            <span>Based on {n_feedback} feedback records</span>
            {isBreach && (
              <span className="font-medium text-red-600">Below threshold</span>
            )}
          </div>
        </div>
      )}

      {current_metric === null && (
        <p className="text-xs text-slate-500">
          No feedback data yet — submit predictions with feedback to track accuracy.
        </p>
      )}

      <p className="mt-3 border-t border-amber-100 pt-2 text-xs text-slate-500">
        To configure: say &ldquo;alert me when accuracy drops below 80%&rdquo; or &ldquo;set
        accuracy alert at 70%&rdquo;. Say &ldquo;disable accuracy alert&rdquo; to remove.
        Webhooks must be registered to receive notifications.
      </p>
    </div>
  )
}
