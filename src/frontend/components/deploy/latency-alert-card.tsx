import type { LatencyAlertConfig } from "@/lib/types"

interface LatencyAlertCardProps {
  config: LatencyAlertConfig
}

export function LatencyAlertCard({ config }: LatencyAlertCardProps) {
  const {
    latency_alert_enabled,
    threshold_ms,
    latency_alert_last_fired_at,
    cooldown_hours,
    summary,
  } = config

  const lastFiredDisplay = latency_alert_last_fired_at
    ? new Date(latency_alert_last_fired_at).toLocaleString()
    : null

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-orange-200 bg-orange-50/50 p-3 text-sm"
      aria-label="Latency alert card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">⏱</span>
        <span className="font-semibold text-orange-900">Prediction Latency Alert</span>
        {latency_alert_enabled ? (
          <span className="rounded bg-orange-100 px-2 py-0.5 text-xs font-medium text-orange-700">
            Enabled
          </span>
        ) : (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            Disabled
          </span>
        )}
      </div>

      <p className="mb-3 text-xs text-orange-800">{summary}</p>

      {latency_alert_enabled && threshold_ms !== null && (
        <div className="mb-3 rounded bg-orange-100/60 px-3 py-2 text-xs text-orange-900">
          <span aria-hidden="true">🔔</span> A webhook fires when the p95 prediction
          latency over the last 100 requests exceeds{" "}
          <strong>{threshold_ms} ms</strong> — helping you catch model slowdowns
          before users notice. Cooldown:{" "}
          <strong>
            {cooldown_hours} hour{cooldown_hours !== 1 ? "s" : ""}
          </strong>{" "}
          between alerts.
        </div>
      )}

      {lastFiredDisplay && (
        <div className="mb-2 flex items-center justify-between text-xs">
          <span className="text-slate-500">Last alert fired</span>
          <span className="rounded bg-rose-100 px-2 py-0.5 font-medium text-rose-800">
            {lastFiredDisplay}
          </span>
        </div>
      )}

      {!lastFiredDisplay && latency_alert_enabled && (
        <p className="mb-2 text-xs text-slate-500">
          No alerts fired yet — webhook fires if p95 latency exceeds threshold.
        </p>
      )}

      <p className="mt-3 border-t border-orange-100 pt-2 text-xs text-slate-500">
        To configure: say &ldquo;alert me if predictions take more than 500ms&rdquo;
        or &ldquo;disable latency alert&rdquo;. Webhooks must be registered to receive
        notifications.
      </p>
    </div>
  )
}
