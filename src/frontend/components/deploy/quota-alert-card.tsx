import type { QuotaAlertConfig } from "@/lib/types"

interface QuotaAlertCardProps {
  config: QuotaAlertConfig
}

function UsageBar({ pctUsed }: { pctUsed: number }) {
  const color =
    pctUsed >= 90 ? "bg-red-500" : pctUsed >= 70 ? "bg-amber-500" : "bg-emerald-500"
  return (
    <div
      className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200"
      role="progressbar"
      aria-valuenow={pctUsed}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={`Quota used: ${pctUsed}%`}
    >
      <div
        className={`h-full rounded-full transition-all ${color}`}
        style={{ width: `${Math.min(pctUsed, 100)}%` }}
      />
    </div>
  )
}

export function QuotaAlertCard({ config }: QuotaAlertCardProps) {
  const {
    quota_alert_enabled,
    quota_alert_threshold_pct,
    monthly_quota,
    used_this_month,
    pct_used,
    summary,
  } = config

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-orange-200 bg-orange-50/50 p-3 text-sm"
      aria-label="Quota alert card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">🔔</span>
        <span className="font-semibold text-orange-900">Quota Alert</span>
        {quota_alert_enabled ? (
          <span className="rounded bg-orange-100 px-2 py-0.5 text-xs font-medium text-orange-700">
            Alert at {quota_alert_threshold_pct}%
          </span>
        ) : (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            Disabled
          </span>
        )}
      </div>

      <p className="mb-3 text-xs text-orange-800">{summary}</p>

      {quota_alert_enabled && quota_alert_threshold_pct && (
        <div className="mb-3 rounded bg-orange-100/60 px-3 py-2 text-xs text-orange-900">
          <span aria-hidden="true">⚠️</span>{" "}
          You will receive a webhook notification when monthly usage reaches{" "}
          <strong>{quota_alert_threshold_pct}%</strong>
          {monthly_quota ? ` of ${monthly_quota.toLocaleString()} predictions` : ""}.
        </div>
      )}

      {monthly_quota && pct_used !== null && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium text-slate-700">Current usage</span>
            <span className="rounded bg-orange-100 px-2 py-0.5 font-semibold text-orange-800">
              {used_this_month.toLocaleString()} / {monthly_quota.toLocaleString()}
            </span>
          </div>
          <UsageBar pctUsed={pct_used} />
          <div className="flex justify-between text-xs text-slate-500">
            <span>{pct_used}% used</span>
            <span>
              {monthly_quota - used_this_month > 0
                ? `${(monthly_quota - used_this_month).toLocaleString()} remaining`
                : "quota reached"}
            </span>
          </div>
        </div>
      )}

      <p className="mt-3 border-t border-orange-100 pt-2 text-xs text-slate-500">
        To configure: say &ldquo;alert me when I hit 80% of my quota&rdquo; or &ldquo;set quota
        alert at 90%&rdquo;. Say &ldquo;disable quota alert&rdquo; to remove. Webhooks must be
        registered to receive notifications.
      </p>
    </div>
  )
}
