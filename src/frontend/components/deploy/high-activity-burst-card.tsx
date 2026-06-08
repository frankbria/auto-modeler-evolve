import type { HighActivityBurstConfig } from "@/lib/types"

interface HighActivityBurstCardProps {
  config: HighActivityBurstConfig
}

export function HighActivityBurstCard({ config }: HighActivityBurstCardProps) {
  const {
    high_activity_burst_enabled,
    threshold_per_hour,
    high_activity_burst_last_fired_at,
    cooldown_hours,
    summary,
  } = config

  const lastFiredDisplay = high_activity_burst_last_fired_at
    ? new Date(high_activity_burst_last_fired_at).toLocaleString()
    : null

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-amber-200 bg-amber-50/50 p-3 text-sm"
      aria-label="High-activity burst alert card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">📈</span>
        <span className="font-semibold text-amber-900">High-Activity Burst Alert</span>
        {high_activity_burst_enabled ? (
          <span className="rounded bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
            Enabled
          </span>
        ) : (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            Disabled
          </span>
        )}
      </div>

      <p className="mb-3 text-xs text-amber-800">{summary}</p>

      {high_activity_burst_enabled && threshold_per_hour !== null && (
        <div className="mb-3 rounded bg-amber-100/60 px-3 py-2 text-xs text-amber-900">
          <span aria-hidden="true">🔔</span> A webhook fires when your deployment
          receives more than{" "}
          <strong>
            {threshold_per_hour} prediction{threshold_per_hour !== 1 ? "s" : ""}
          </strong>{" "}
          in an hour — detecting runaway loops, API abuse, or unexpected demand spikes.
          Cooldown: <strong>{cooldown_hours} hour{cooldown_hours !== 1 ? "s" : ""}</strong>{" "}
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

      {!lastFiredDisplay && high_activity_burst_enabled && (
        <p className="mb-2 text-xs text-slate-500">
          No alerts fired yet — webhook fires if hourly predictions exceed threshold.
        </p>
      )}

      <p className="mt-3 border-t border-amber-100 pt-2 text-xs text-slate-500">
        To configure: say &ldquo;alert me if more than 100 predictions per hour&rdquo;
        or &ldquo;disable burst alert&rdquo;. Webhooks must be registered to receive notifications.
      </p>
    </div>
  )
}
