import type { CostEstimateResult } from "@/lib/types"

interface CostEstimateCardProps {
  result: CostEstimateResult
}

function CapacityBar({ value, max, label }: { value: number; max: number; label: string }) {
  const pct = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0
  const color = pct >= 90 ? "bg-rose-500" : pct >= 70 ? "bg-amber-500" : "bg-emerald-500"
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-600">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div
        className="w-full bg-gray-200 rounded-full h-2"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label}: ${pct}% of capacity`}
      >
        <div className={`h-2 rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function CostEstimateCard({ result }: CostEstimateCardProps) {
  const noQuota = result.monthly_quota === null
  const fits = result.within_quota === true
  const exceeds = result.within_quota === false

  const borderColor = exceeds
    ? "border-rose-300 bg-rose-50"
    : noQuota
    ? "border-emerald-300 bg-emerald-50"
    : "border-amber-300 bg-amber-50"

  const statusBadge = exceeds
    ? { cls: "bg-rose-100 text-rose-700", label: "Exceeds remaining quota" }
    : fits
    ? { cls: "bg-emerald-100 text-emerald-700", label: "Fits in quota" }
    : { cls: "bg-sky-100 text-sky-700", label: "Unlimited" }

  return (
    <figure
      className={`rounded-lg border p-4 my-2 space-y-3 ${borderColor}`}
      aria-label="Prediction capacity estimate"
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-lg" aria-hidden="true">💰</span>
        <span className="font-semibold text-sm">Prediction Capacity Estimate</span>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${statusBadge.cls}`}>
          {statusBadge.label}
        </span>
        <span className="text-xs px-2 py-0.5 rounded-full bg-sky-100 text-sky-700">
          {result.n_predictions.toLocaleString()} predictions
        </span>
      </div>

      {/* Quota impact section */}
      {result.monthly_quota !== null ? (
        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-700">
            Quota Impact
          </div>
          <CapacityBar
            value={result.n_predictions}
            max={result.monthly_quota - result.used_this_month}
            label={`${result.n_predictions.toLocaleString()} of ${(result.monthly_quota - result.used_this_month).toLocaleString()} remaining`}
          />
          {result.quota_pct !== null && (
            <p className="text-xs text-gray-600">
              {result.n_predictions.toLocaleString()} predictions ={" "}
              <strong>{result.quota_pct}%</strong> of your {result.monthly_quota.toLocaleString()}-prediction
              monthly quota ({result.used_this_month.toLocaleString()} used so far).
            </p>
          )}
          {exceeds && (
            <p className="text-xs text-rose-700 font-medium" role="alert">
              ⚠ This exceeds your remaining quota. Increase your monthly quota or wait until next month.
            </p>
          )}
        </div>
      ) : (
        <p className="text-sm text-emerald-800">
          No monthly quota configured — <strong>{result.n_predictions.toLocaleString()} predictions</strong> can
          be served without any quota limit.
        </p>
      )}

      {/* Daily capacity and rate */}
      <div className="grid grid-cols-2 gap-2 text-sm">
        {result.daily_capacity !== null && (
          <div className="bg-white rounded p-2 border border-gray-200">
            <div className="text-xs text-gray-500">Daily capacity</div>
            <div className="font-bold">{result.daily_capacity.toLocaleString()}</div>
            <div className="text-xs text-gray-400">at {result.current_rpm} RPM</div>
          </div>
        )}
        {result.days_needed !== null && (
          <div className="bg-white rounded p-2 border border-gray-200">
            <div className="text-xs text-gray-500">Days to serve</div>
            <div className="font-bold">{result.days_needed}</div>
            <div className="text-xs text-gray-400">at current rate</div>
          </div>
        )}
        {result.daily_capacity === null && (
          <div className="bg-white rounded p-2 border border-gray-200">
            <div className="text-xs text-gray-500">Avg predictions/day</div>
            <div className="font-bold">{result.avg_per_day}</div>
            <div className="text-xs text-gray-400">last 7 days</div>
          </div>
        )}
      </div>

      {/* Recommended rate limit */}
      <div className="bg-white rounded p-2 border border-gray-200 text-sm">
        <div className="text-xs text-gray-500 mb-1">Recommended rate limit</div>
        <div className="flex items-center gap-2">
          <span className="font-bold text-base">{result.recommended_rpm} RPM</span>
          <span className="text-xs text-gray-500">
            to spread {result.n_predictions.toLocaleString()} predictions evenly over 30 days
          </span>
        </div>
      </div>

      {result.daily_capacity === null && (
        <p className="text-xs text-gray-500 italic">
          No rate limit set — set one via &quot;set rate limit to {result.recommended_rpm} RPM&quot; to control throughput.
        </p>
      )}

      <figcaption className="sr-only">
        Prediction capacity estimate for {result.n_predictions.toLocaleString()} predictions.
        {result.monthly_quota !== null
          ? ` Uses ${result.quota_pct}% of monthly quota. Recommended rate limit: ${result.recommended_rpm} RPM.`
          : " No monthly quota configured."}
      </figcaption>
    </figure>
  )
}
