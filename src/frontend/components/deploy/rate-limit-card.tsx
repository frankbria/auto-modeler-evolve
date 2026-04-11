import type { RateLimitInfo } from "@/lib/types"

interface RateLimitCardProps {
  info: RateLimitInfo
}

function UsageBar({ pctUsed }: { pctUsed: number }) {
  const color =
    pctUsed >= 90 ? "bg-red-500" : pctUsed >= 70 ? "bg-amber-500" : "bg-emerald-500"
  return (
    <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-slate-200" role="progressbar" aria-valuenow={pctUsed} aria-valuemin={0} aria-valuemax={100} aria-label={`Quota used: ${pctUsed}%`}>
      <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${Math.min(pctUsed, 100)}%` }} />
    </div>
  )
}

export function RateLimitCard({ info }: RateLimitCardProps) {
  const hasRpm = info.rate_limit_enabled && info.rate_limit_rpm
  const hasQuota = info.quota_enabled && info.monthly_quota

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-amber-200 bg-amber-50/50 p-3 text-sm"
      aria-label="Rate limit card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">⚡</span>
        <span className="font-semibold text-amber-900">Rate Limits &amp; Quotas</span>
        {hasRpm || hasQuota ? (
          <span className="rounded bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
            Active
          </span>
        ) : (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            No limits
          </span>
        )}
      </div>

      <p className="mb-3 text-xs text-amber-800">{info.summary}</p>

      <div className="space-y-3">
        {/* Per-minute rate limit */}
        <div>
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-700">Per-minute limit</span>
            {hasRpm ? (
              <span className="rounded bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-800" aria-label={`Rate limit: ${info.rate_limit_rpm} requests per minute`}>
                {info.rate_limit_rpm} req/min
              </span>
            ) : (
              <span className="text-xs text-slate-400">Unlimited</span>
            )}
          </div>
          <p className="mt-0.5 text-xs text-slate-500">
            {hasRpm
              ? `Requests beyond ${info.rate_limit_rpm}/min receive a 429 response.`
              : "No per-minute rate limit is set — the endpoint accepts any request volume."}
          </p>
        </div>

        {/* Monthly quota */}
        <div>
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-700">Monthly quota</span>
            {hasQuota ? (
              <span className="rounded bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-800" aria-label={`Monthly quota: ${info.used_this_month} of ${info.monthly_quota} used`}>
                {info.used_this_month.toLocaleString()} / {info.monthly_quota!.toLocaleString()}
              </span>
            ) : (
              <span className="text-xs text-slate-400">Unlimited</span>
            )}
          </div>

          {hasQuota && info.pct_used !== null && (
            <>
              <UsageBar pctUsed={info.pct_used} />
              <div className="mt-1 flex justify-between text-xs text-slate-500">
                <span>{info.pct_used}% used</span>
                <span>
                  {info.remaining !== null
                    ? `${info.remaining.toLocaleString()} remaining`
                    : "quota exceeded"}
                </span>
              </div>
            </>
          )}

          {!hasQuota && (
            <p className="mt-0.5 text-xs text-slate-500">
              No monthly quota is set — the endpoint has no 30-day prediction cap.
            </p>
          )}
        </div>
      </div>

      <p className="mt-3 border-t border-amber-100 pt-2 text-xs text-slate-500">
        To update: say &ldquo;set rate limit to 100 requests per minute&rdquo; or &ldquo;set monthly
        quota to 1000 predictions&rdquo;. Say &ldquo;disable rate limit&rdquo; to remove a limit.
      </p>
    </div>
  )
}
