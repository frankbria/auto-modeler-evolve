import type { FeatureDriftAlertConfig } from "@/lib/types"

interface FeatureDriftAlertCardProps {
  config: FeatureDriftAlertConfig
}

export function FeatureDriftAlertCard({ config }: FeatureDriftAlertCardProps) {
  const {
    feature_drift_alert_enabled,
    feature_drift_alert_last_fired_at,
    cooldown_hours,
    summary,
  } = config

  const lastFiredDisplay = feature_drift_alert_last_fired_at
    ? new Date(feature_drift_alert_last_fired_at).toLocaleString()
    : null

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-sky-200 bg-sky-50/50 p-3 text-sm"
      aria-label="Feature drift alert card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">🔔</span>
        <span className="font-semibold text-sky-900">Feature Drift Alert</span>
        {feature_drift_alert_enabled ? (
          <span className="rounded bg-sky-100 px-2 py-0.5 text-xs font-medium text-sky-700">
            Enabled
          </span>
        ) : (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            Disabled
          </span>
        )}
      </div>

      <p className="mb-3 text-xs text-sky-800">{summary}</p>

      {feature_drift_alert_enabled && (
        <div className="mb-3 rounded bg-sky-100/60 px-3 py-2 text-xs text-sky-900">
          <span aria-hidden="true">📡</span> A webhook fires when critical-priority features
          are detected — those with both high drift and high model importance.
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

      {!lastFiredDisplay && feature_drift_alert_enabled && (
        <p className="mb-2 text-xs text-slate-500">
          No alerts fired yet — webhook will fire when critical drift is detected.
        </p>
      )}

      <p className="mt-3 border-t border-sky-100 pt-2 text-xs text-slate-500">
        To configure: say &ldquo;enable feature drift alerts&rdquo; or &ldquo;disable drift
        webhook&rdquo;. Webhooks must be registered to receive notifications.
      </p>
    </div>
  )
}
