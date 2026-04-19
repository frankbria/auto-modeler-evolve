import type { CovariateDriftAlertResult, CovariateDriftFeatureAlert } from "@/lib/types"

interface CovariateDriftAlertCardProps {
  result: CovariateDriftAlertResult
}

function severityColors(severity: CovariateDriftAlertResult["severity"]) {
  if (severity === "high")
    return {
      border: "border-rose-300 bg-rose-50",
      badge: "bg-rose-100 text-rose-700",
      label: "Significant Drift",
    }
  if (severity === "medium")
    return {
      border: "border-amber-300 bg-amber-50",
      badge: "bg-amber-100 text-amber-700",
      label: "Some Drift",
    }
  return {
    border: "border-emerald-300 bg-emerald-50",
    badge: "bg-emerald-100 text-emerald-700",
    label: "No Significant Drift",
  }
}

function AlertRow({ alert }: { alert: CovariateDriftFeatureAlert }) {
  const isHigh = alert.severity === "high"
  const rowColor = isHigh
    ? "border-rose-200 bg-rose-50"
    : "border-amber-200 bg-amber-50"
  const pct =
    alert.feature_type === "numeric" ? alert.oor_pct : alert.unseen_pct

  return (
    <div
      className={`flex items-start gap-3 rounded border px-3 py-2 text-sm ${rowColor}`}
      aria-label={`Drift alert for ${alert.feature}: ${alert.description}`}
    >
      <span
        className={`mt-0.5 rounded px-1.5 py-0.5 text-xs font-medium ${
          isHigh
            ? "bg-rose-200 text-rose-800"
            : "bg-amber-200 text-amber-800"
        }`}
        aria-hidden="true"
      >
        {isHigh ? "HIGH" : "MED"}
      </span>
      <div className="flex-1 min-w-0">
        <span className="font-mono font-medium text-gray-800">
          {alert.feature}
        </span>
        <span className="ml-2 text-gray-600">
          {alert.feature_type === "numeric"
            ? `${pct}% out-of-range`
            : `${pct}% unseen categories`}
        </span>
        <p className="mt-0.5 text-xs text-gray-500">{alert.description}</p>
      </div>
    </div>
  )
}

export function CovariateDriftAlertCard({ result }: CovariateDriftAlertCardProps) {
  const colors = severityColors(result.severity)

  return (
    <figure
      className={`my-2 rounded-lg border-2 p-4 ${colors.border}`}
      aria-label="Covariate drift alert"
    >
      {/* Header */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span aria-hidden="true" className="text-lg">
          🌊
        </span>
        <span className="font-semibold text-gray-800">
          Production Input Drift Check
        </span>
        <span
          className={`rounded px-2 py-0.5 text-xs font-medium ${colors.badge}`}
        >
          {colors.label}
        </span>
        <span className="rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-600">
          {result.sample_count} prediction
          {result.sample_count !== 1 ? "s" : ""} analyzed
        </span>
        {result.alert_count > 0 && (
          <span className="rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-600">
            {result.alert_count} feature
            {result.alert_count !== 1 ? "s" : ""} flagged
          </span>
        )}
      </div>

      {/* Summary */}
      <p className="mb-3 text-sm text-gray-700">{result.summary}</p>

      {/* Per-feature alerts */}
      {result.alerts.length > 0 && (
        <div className="mb-3 space-y-2">
          {result.alerts.map((alert) => (
            <AlertRow key={alert.feature} alert={alert} />
          ))}
        </div>
      )}

      {/* Guidance footer */}
      {result.has_alerts && (
        <p className="mt-2 text-xs text-gray-500 italic">
          Ask &quot;show production input distribution&quot; for full per-feature stats, or
          &quot;retrain my model&quot; to update with recent data.
        </p>
      )}

      <figcaption className="sr-only">
        Covariate drift severity: {result.severity_label}. {result.summary}
      </figcaption>
    </figure>
  )
}
