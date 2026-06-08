import type { LowActivityAlertConfig } from "@/lib/types"

interface LowActivityAlertCardProps {
  config: LowActivityAlertConfig
}

export function LowActivityAlertCard({ config }: LowActivityAlertCardProps) {
  const {
    low_activity_alert_enabled,
    threshold_per_day,
    low_activity_alert_last_fired_at,
    cooldown_hours,
    summary,
  } = config

  const lastFiredDisplay = low_activity_alert_last_fired_at
    ? new Date(low_activity_alert_last_fired_at).toLocaleString()
    : null

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-sky-200 bg-sky-50/50 p-3 text-sm"
      aria-label="Low-activity alert card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">🔕</span>
        <span className="font-semibold text-sky-900">Low-Activity Alert</span>
        {low_activity_alert_enabled ? (
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

      {low_activity_alert_enabled && threshold_per_day !== null && (
        <div className="mb-3 rounded bg-sky-100/60 px-3 py-2 text-xs text-sky-900">
          <span aria-hidden="true">📉</span> A webhook fires when your deployment
          receives fewer than{" "}
          <strong>
            {threshold_per_day} prediction{threshold_per_day !== 1 ? "s" : ""}
          </strong>{" "}
          in a day — detecting silent integration failures. Cooldown:{" "}
          <strong>{cooldown_hours} hours</strong> between alerts.
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

      {!lastFiredDisplay && low_activity_alert_enabled && (
        <p className="mb-2 text-xs text-slate-500">
          No alerts fired yet — webhook fires if daily predictions drop below threshold.
        </p>
      )}

      <p className="mt-3 border-t border-sky-100 pt-2 text-xs text-slate-500">
        To configure: say &ldquo;alert me if fewer than 10 predictions per day&rdquo;
        or &ldquo;disable low-activity alert&rdquo;. Webhooks must be registered to receive notifications.
      </p>
    </div>
  )
}
