import type { QuotaRunwayResult } from "@/lib/types"

interface QuotaRunwayCardProps {
  result: QuotaRunwayResult
}

function UsageBar({ used, total }: { used: number; total: number }) {
  const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0
  const color =
    pct >= 90 ? "bg-rose-500" : pct >= 70 ? "bg-amber-500" : "bg-emerald-500"
  return (
    <div className="w-full bg-gray-200 rounded-full h-2" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label={`${pct}% of quota used`}>
      <div className={`h-2 rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
    </div>
  )
}

export function QuotaRunwayCard({ result }: QuotaRunwayCardProps) {
  const borderColor = result.will_exhaust
    ? "border-rose-300 bg-rose-50"
    : result.has_quota
    ? "border-amber-300 bg-amber-50"
    : "border-emerald-300 bg-emerald-50"

  const statusBadge = result.will_exhaust
    ? { cls: "bg-rose-100 text-rose-700", label: "Quota At Risk" }
    : result.has_quota
    ? { cls: "bg-amber-100 text-amber-700", label: "Quota Set" }
    : { cls: "bg-emerald-100 text-emerald-700", label: "Unlimited" }

  const pct =
    result.monthly_quota && result.monthly_quota > 0
      ? Math.min(100, Math.round((result.used_this_month / result.monthly_quota) * 100))
      : 0

  return (
    <figure
      className={`rounded-lg border p-4 my-2 space-y-3 ${borderColor}`}
      aria-label="Quota runway analysis"
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-lg" aria-hidden="true">📊</span>
        <span className="font-semibold text-sm">Quota Runway</span>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusBadge.cls}`}>
          {statusBadge.label}
        </span>
        {result.avg_per_day > 0 && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-sky-100 text-sky-700">
            {result.avg_per_day}/day avg
          </span>
        )}
      </div>

      {result.has_quota && result.monthly_quota !== null ? (
        <>
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-gray-600">
              <span>{result.used_this_month.toLocaleString()} used</span>
              <span>{result.monthly_quota.toLocaleString()} total</span>
            </div>
            <UsageBar used={result.used_this_month} total={result.monthly_quota} />
            <div className="text-xs text-gray-500 text-right">{pct}% used this month</div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-white rounded p-2 border border-gray-200">
              <div className="text-xs text-gray-500">Remaining</div>
              <div className="font-bold">
                {result.remaining !== null ? result.remaining.toLocaleString() : "—"}
              </div>
            </div>
            <div className="bg-white rounded p-2 border border-gray-200">
              <div className="text-xs text-gray-500">Days left in month</div>
              <div className="font-bold">{result.days_remaining_in_month}</div>
            </div>
          </div>

          {result.days_left_at_rate !== null && (
            <div
              className={`text-sm rounded p-2 border ${result.will_exhaust ? "bg-rose-100 border-rose-200 text-rose-800" : "bg-white border-gray-200 text-gray-700"}`}
              role={result.will_exhaust ? "alert" : undefined}
            >
              {result.will_exhaust ? (
                <>
                  <strong>⚠ Quota at risk:</strong> at {result.avg_per_day}/day, quota runs out
                  in <strong>{result.days_left_at_rate} days</strong>. Projected month total:{" "}
                  <strong>{result.est_month_total.toLocaleString()}</strong> (limit:{" "}
                  {result.monthly_quota.toLocaleString()}).
                </>
              ) : (
                <>
                  ✓ Quota lasts <strong>{result.days_left_at_rate} more days</strong> at current
                  rate. Projected month total:{" "}
                  <strong>{result.est_month_total.toLocaleString()}</strong> of{" "}
                  {result.monthly_quota.toLocaleString()}.
                </>
              )}
            </div>
          )}

          {result.avg_per_day === 0 && (
            <p className="text-xs text-gray-500 italic">
              No recent activity — daily rate based on last 7 days.
            </p>
          )}
        </>
      ) : (
        <div className="space-y-2">
          <p className="text-sm text-emerald-800">
            No monthly quota configured — predictions are <strong>unlimited</strong>.
          </p>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-white rounded p-2 border border-gray-200">
              <div className="text-xs text-gray-500">Used this month</div>
              <div className="font-bold">{result.used_this_month.toLocaleString()}</div>
            </div>
            <div className="bg-white rounded p-2 border border-gray-200">
              <div className="text-xs text-gray-500">Avg predictions/day</div>
              <div className="font-bold">{result.avg_per_day}</div>
            </div>
          </div>
        </div>
      )}

      {result.rate_limit_rpm !== null && (
        <p className="text-xs text-gray-500">
          Rate limit: <strong>{result.rate_limit_rpm} req/min</strong> — max{" "}
          {(result.rate_limit_rpm * 60).toLocaleString()} predictions/hour.
        </p>
      )}

      <figcaption className="sr-only">
        Quota runway analysis: {result.has_quota ? `${result.used_this_month} of ${result.monthly_quota} predictions used this month` : "unlimited predictions"}.
        {result.days_left_at_rate !== null && ` At current rate, quota lasts ${result.days_left_at_rate} more days.`}
      </figcaption>
    </figure>
  )
}
