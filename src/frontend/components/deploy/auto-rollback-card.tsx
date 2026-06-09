"use client"

import { Badge } from "@/components/ui/badge"
import type { AutoRollbackConfig } from "@/lib/types"

function formatDate(iso: string | null): string {
  if (!iso) return "—"
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

interface AutoRollbackCardProps {
  data: AutoRollbackConfig
}

export function AutoRollbackCard({ data }: AutoRollbackCardProps) {
  const isEnabled = data.auto_rollback_enabled && data.accuracy_threshold_pct !== null

  const borderClass = isEnabled
    ? data.auto_rollback_triggered_at
      ? "border-amber-300"
      : "border-emerald-300"
    : "border-slate-300"

  const badgeClass = isEnabled
    ? "bg-emerald-100 text-emerald-800 border-emerald-200"
    : "bg-slate-100 text-slate-600 border-slate-200"

  return (
    <figure
      role="region"
      aria-label="Auto-rollback configuration"
      className={`rounded-lg border ${borderClass} bg-white p-4 my-2 shadow-sm`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span aria-hidden="true" className="text-lg">🔄</span>
        <h3 className="font-semibold text-slate-800 text-sm">Accuracy-Triggered Auto-Rollback</h3>
        <Badge className={`${badgeClass} text-xs ml-auto`}>
          {isEnabled ? "Enabled" : "Disabled"}
        </Badge>
      </div>

      {/* Summary */}
      <p className="text-sm text-slate-600 mb-3">{data.summary}</p>

      {isEnabled && data.accuracy_threshold_pct !== null && (
        <>
          {/* Threshold tile */}
          <div className="bg-slate-50 rounded p-3 mb-3 text-sm">
            <div className="flex items-center gap-2 mb-1">
              <span aria-hidden="true">🎯</span>
              <span className="font-medium text-slate-700">Rollback threshold</span>
              <Badge className="bg-rose-100 text-rose-800 border-rose-200 text-xs ml-auto">
                {data.accuracy_threshold_pct}% accuracy floor
              </Badge>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              When feedback accuracy from the last 50 predictions drops below{" "}
              <strong>{data.accuracy_threshold_pct}%</strong>, the deployment
              automatically rolls back to the previous version.
            </p>
          </div>

          {/* How it works info block */}
          <div className="bg-blue-50 border border-blue-100 rounded p-3 mb-3 text-xs text-blue-800">
            <p className="font-medium mb-1">How it works</p>
            <ul className="space-y-0.5 list-disc list-inside text-blue-700">
              <li>
                Checked every 60 seconds · requires{" "}
                <strong>{data.min_feedback_required}+</strong> labeled feedback
                points
              </li>
              <li>
                <strong>{data.cooldown_hours}-hour</strong> cooldown between
                rollbacks to prevent loops
              </li>
              <li>
                Fires a <code>rollback_triggered</code> webhook on execution
              </li>
              <li>Endpoint URL stays the same — zero downtime</li>
            </ul>
          </div>

          {/* Last triggered */}
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span aria-hidden="true">⏱</span>
            {data.auto_rollback_triggered_at ? (
              <span>
                Last triggered:{" "}
                <span className="text-amber-600 font-medium">
                  {formatDate(data.auto_rollback_triggered_at)}
                </span>
              </span>
            ) : (
              <span className="text-slate-400 italic">No auto-rollbacks triggered yet</span>
            )}
          </div>
        </>
      )}

      {!isEnabled && (
        <p className="text-xs text-slate-400 italic mt-1">
          Say <em>"roll back if accuracy drops below 80%"</em> to enable
          automatic rollback protection.
        </p>
      )}

      <figcaption className="sr-only">
        Auto-rollback is{" "}
        {isEnabled
          ? `enabled at ${data.accuracy_threshold_pct}% accuracy threshold`
          : "disabled"}
        .
        {data.auto_rollback_triggered_at
          ? ` Last triggered: ${formatDate(data.auto_rollback_triggered_at)}.`
          : ""}
      </figcaption>

      {/* Footer */}
      <p className="text-xs text-slate-400 mt-3 italic">
        Distinct from manual rollback (which you initiate) and accuracy alerts
        (which notify only). Requires webhooks registered to receive
        notifications.
      </p>
    </figure>
  )
}
