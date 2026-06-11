"use client"

import type { CanaryStatusResult } from "@/lib/types"

interface CanaryCardProps {
  result: CanaryStatusResult
}

function VersionBadge({ version, algorithm, isCurrent }: { version: number; algorithm: string | null; isCurrent: boolean }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-muted text-foreground">
      v{version}
      {algorithm && <span className="text-muted-foreground">· {algorithm}</span>}
      {isCurrent && <span className="text-emerald-600 font-semibold">current</span>}
    </span>
  )
}

function DeltaBadge({ deltaPct }: { deltaPct: number | null }) {
  if (deltaPct === null) return null
  const isUp = deltaPct > 0
  const color = isUp ? "text-emerald-600" : "text-rose-600"
  const sign = isUp ? "+" : ""
  return (
    <span className={`text-xs font-semibold ${color}`}>
      {sign}{deltaPct.toFixed(1)}%
    </span>
  )
}

export function CanaryCard({ result }: CanaryCardProps) {
  const {
    canary_is_active,
    canary_traffic_pct,
    canary_version_number,
    canary_algorithm,
    canary_started_at,
    current_version_number,
    current_algorithm,
    available_versions,
    comparison,
    summary,
  } = result

  const borderColor = canary_is_active
    ? "border-amber-400"
    : "border-slate-200 dark:border-slate-700"

  return (
    <div
      className={`rounded-xl border-2 ${borderColor} bg-card p-4 space-y-3`}
      role="region"
      aria-label="Canary deployment status"
    >
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span aria-hidden="true">🐤</span>
          <span className="font-semibold text-foreground">Canary Deployment</span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
              canary_is_active
                ? "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200"
                : "bg-muted text-muted-foreground"
            }`}
          >
            {canary_is_active ? "Active" : "Inactive"}
          </span>
          {canary_is_active && (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
              {canary_traffic_pct}% canary traffic
            </span>
          )}
        </div>
      </div>

      {/* Summary */}
      <p className="text-sm text-muted-foreground">{summary}</p>

      {/* Active canary: current vs canary info */}
      {canary_is_active && (
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg bg-muted/50 p-3">
            <div className="text-xs text-muted-foreground mb-1">Control (current)</div>
            <VersionBadge version={current_version_number} algorithm={current_algorithm} isCurrent={false} />
            <div className="mt-1 text-xs text-muted-foreground">
              {100 - (canary_traffic_pct ?? 0)}% of traffic
            </div>
          </div>
          <div className="rounded-lg bg-amber-50 dark:bg-amber-950/30 p-3">
            <div className="text-xs text-muted-foreground mb-1">Canary (testing)</div>
            {canary_version_number != null && (
              <VersionBadge version={canary_version_number} algorithm={canary_algorithm} isCurrent={false} />
            )}
            <div className="mt-1 text-xs text-muted-foreground">
              {canary_traffic_pct}% of traffic
            </div>
            {canary_started_at && (
              <div className="mt-0.5 text-xs text-muted-foreground">
                Started {new Date(canary_started_at).toLocaleDateString()}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Comparison stats */}
      {canary_is_active && comparison && (
        <div className="rounded-lg border border-border p-3 space-y-2">
          <div className="text-xs font-semibold text-foreground">Live Comparison</div>
          {comparison.has_enough_data ? (
            <>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                  <div className="text-xs text-muted-foreground">Control</div>
                  <div className="text-sm font-semibold">
                    {comparison.control_avg != null
                      ? Number(comparison.control_avg).toFixed(3)
                      : "—"}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {comparison.control_count} preds
                  </div>
                </div>
                <div className="flex flex-col items-center justify-center">
                  <DeltaBadge deltaPct={comparison.delta_pct} />
                  <div className="text-xs text-muted-foreground mt-0.5">
                    {comparison.metric_label}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Canary</div>
                  <div className="text-sm font-semibold">
                    {comparison.canary_avg != null
                      ? Number(comparison.canary_avg).toFixed(3)
                      : "—"}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {comparison.canary_count} preds
                  </div>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">{comparison.summary}</p>
            </>
          ) : (
            <p className="text-xs text-muted-foreground">{comparison.summary}</p>
          )}
        </div>
      )}

      {/* Available versions (when no canary active) */}
      {!canary_is_active && available_versions.length > 1 && (
        <div className="space-y-1">
          <div className="text-xs font-medium text-muted-foreground">Available versions:</div>
          <div className="flex flex-wrap gap-1.5">
            {available_versions.map((v) => (
              <VersionBadge
                key={v.version_number}
                version={v.version_number}
                algorithm={v.algorithm}
                isCurrent={v.is_current}
              />
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            To start a canary: <em>&ldquo;start a canary with v1 at 10% traffic&rdquo;</em>
          </p>
        </div>
      )}

      {/* Actions footer */}
      {canary_is_active && (
        <div className="pt-1 border-t border-border text-xs text-muted-foreground">
          <strong>Commands:</strong>{" "}
          <em>&ldquo;promote the canary&rdquo;</em> to make it production,{" "}
          <em>&ldquo;cancel canary&rdquo;</em> to revert to 100% control.
        </div>
      )}
    </div>
  )
}
