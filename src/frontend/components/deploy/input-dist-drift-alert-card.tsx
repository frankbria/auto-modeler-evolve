import type { InputDistDriftAlertConfig } from "@/lib/types"

interface InputDistDriftAlertCardProps {
  config: InputDistDriftAlertConfig
}

export function InputDistDriftAlertCard({ config }: InputDistDriftAlertCardProps) {
  const {
    input_dist_drift_alert_enabled,
    input_dist_drift_severity_threshold,
    input_dist_drift_alert_last_fired_at,
    cooldown_hours,
    summary,
  } = config

  const lastFiredDisplay = input_dist_drift_alert_last_fired_at
    ? new Date(input_dist_drift_alert_last_fired_at).toLocaleString()
    : null

  const thresholdLabel =
    input_dist_drift_severity_threshold === "high" ? "High (≥30% OOR)" : "Medium (≥15% OOR)"

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-cyan-200 bg-cyan-50/50 p-3 text-sm"
      aria-label="Input distribution drift alert card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">🌊</span>
        <span className="font-semibold text-cyan-900">Input Distribution Drift Alert</span>
        {input_dist_drift_alert_enabled ? (
          <span className="rounded bg-cyan-100 px-2 py-0.5 text-xs font-medium text-cyan-700">
            Enabled
          </span>
        ) : (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            Disabled
          </span>
        )}
      </div>

      <p className="mb-3 text-xs text-cyan-800">{summary}</p>

      {input_dist_drift_alert_enabled && (
        <div className="mb-3 rounded bg-cyan-100/60 px-3 py-2 text-xs text-cyan-900">
          <span aria-hidden="true">📡</span> A webhook fires when live prediction inputs
          diverge from the training distribution at{" "}
          <strong>{thresholdLabel}</strong> severity.
          Cooldown: <strong>{cooldown_hours} hours</strong> between alerts.
        </div>
      )}

      {lastFiredDisplay && (
        <div className="mb-2 flex items-center justify-between text-xs">
          <span className="text-slate-500">Last alert fired</span>
          <span className="rounded bg-amber-100 px-2 py-0.5 font-medium text-amber-800">
            {lastFiredDisplay}
          </span>
        </div>
      )}

      {!lastFiredDisplay && input_dist_drift_alert_enabled && (
        <p className="mb-2 text-xs text-slate-500">
          No alerts fired yet — webhook will fire when input distribution drift is detected.
        </p>
      )}

      <p className="mt-3 border-t border-cyan-100 pt-2 text-xs text-slate-500">
        To configure: say &ldquo;enable input distribution drift alerts&rdquo; or
        &ldquo;alert me when live inputs diverge from training&rdquo;. Add &ldquo;high
        severity&rdquo; for stricter threshold. Webhooks must be registered to receive
        notifications.
      </p>
    </div>
  )
}
