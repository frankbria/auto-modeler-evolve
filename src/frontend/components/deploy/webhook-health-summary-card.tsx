"use client"

import { Badge } from "@/components/ui/badge"
import type {
  WebhookHealthSummaryResult,
  WebhookDeploymentHealth,
  WebhookHealthRow,
} from "@/lib/types"

// ---- helpers ----------------------------------------------------------------

const STATUS_BADGE: Record<
  string,
  { label: string; className: string }
> = {
  healthy: {
    label: "Healthy",
    className: "bg-emerald-100 text-emerald-800 border-emerald-200",
  },
  warning: {
    label: "Warning",
    className: "bg-amber-100 text-amber-800 border-amber-200",
  },
  critical: {
    label: "Critical",
    className: "bg-red-100 text-red-800 border-red-200",
  },
  no_events: {
    label: "No events yet",
    className: "bg-slate-100 text-slate-600 border-slate-200",
  },
  no_webhooks: {
    label: "No webhooks",
    className: "bg-slate-100 text-slate-500 border-slate-200",
  },
}

function formatDate(iso: string | null): string {
  if (!iso) return "—"
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

// ---- sub-components ---------------------------------------------------------

function WebhookRowItem({ row }: { row: WebhookHealthRow }) {
  const { label, className } = STATUS_BADGE[row.status] ?? STATUS_BADGE.no_events
  return (
    <div className="flex items-start gap-2 py-2 border-b border-slate-100 last:border-0 text-xs">
      <Badge className={`${className} text-xs shrink-0 mt-0.5`}>{label}</Badge>
      <div className="flex-1 min-w-0">
        <p
          className="text-slate-700 font-mono truncate"
          title={row.url}
        >
          {row.url}
        </p>
        <p className="text-slate-400 mt-0.5">
          {row.total_events === 0
            ? "No events yet"
            : `${row.total_events} event${row.total_events !== 1 ? "s" : ""}, `
              + `${row.failed_events} failed`
              + (row.success_rate !== null
                ? ` (${row.success_rate}% success)`
                : "")
          }
          {row.last_event ? ` · Last: ${formatDate(row.last_event)}` : ""}
        </p>
      </div>
    </div>
  )
}

function DeploymentSection({ dep }: { dep: WebhookDeploymentHealth }) {
  const { label, className } = STATUS_BADGE[dep.status] ?? STATUS_BADGE.no_events
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs font-medium text-slate-600">{dep.deployment_name}</span>
        <Badge className={`${className} text-xs`}>{label}</Badge>
      </div>
      <div className="pl-2 border-l-2 border-slate-100">
        {dep.webhooks.map((row) => (
          <WebhookRowItem key={row.webhook_id} row={row} />
        ))}
      </div>
    </div>
  )
}

// ---- main card --------------------------------------------------------------

interface WebhookHealthSummaryCardProps {
  data: WebhookHealthSummaryResult
}

export function WebhookHealthSummaryCard({
  data,
}: WebhookHealthSummaryCardProps) {
  const { label: overallLabel, className: overallClass } =
    STATUS_BADGE[data.overall_status] ?? STATUS_BADGE.no_webhooks

  // Choose border color based on overall status
  const borderClass =
    data.overall_status === "healthy"
      ? "border-emerald-300"
      : data.overall_status === "critical"
      ? "border-red-300"
      : data.overall_status === "warning"
      ? "border-amber-300"
      : "border-slate-300"

  return (
    <div
      role="region"
      aria-label="Webhook health summary"
      className={`rounded-lg border ${borderClass} bg-white p-4 my-2 shadow-sm`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span aria-hidden="true" className="text-lg">🔗</span>
        <h3 className="font-semibold text-slate-800 text-sm">Webhook Health Summary</h3>
        <Badge className={`${overallClass} text-xs ml-auto`}>{overallLabel}</Badge>
        {data.total_webhooks > 0 && (
          <Badge className="bg-slate-100 text-slate-700 border-slate-200 text-xs">
            {data.total_webhooks} webhook{data.total_webhooks !== 1 ? "s" : ""}
          </Badge>
        )}
      </div>

      {/* Summary */}
      <p className="text-sm text-slate-600 mb-3">{data.summary}</p>

      {/* No webhooks state */}
      {data.overall_status === "no_webhooks" && (
        <p className="text-sm text-slate-400 italic">
          Register webhooks in the Deployment panel to receive notifications when
          batch jobs complete, drift is detected, or model health degrades.
        </p>
      )}

      {/* Per-deployment breakdown */}
      {data.deployments.length > 0 && (
        <div className="mt-2">
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">
            Per-Deployment Breakdown
          </p>
          {data.deployments.map((dep) => (
            <DeploymentSection key={dep.deployment_id} dep={dep} />
          ))}
        </div>
      )}

      {/* Stats footer */}
      {data.total_events > 0 && (
        <div className="flex gap-4 mt-3 pt-2 border-t border-slate-100 text-xs text-slate-500">
          <span>{data.total_events} total event{data.total_events !== 1 ? "s" : ""}</span>
          <span>{data.total_failed} failed</span>
          {data.total_failed > 0 && (
            <span className="text-red-500">
              {Math.round((data.total_failed / data.total_events) * 100)}% failure rate
            </span>
          )}
        </div>
      )}

      {/* Help footer */}
      <p className="text-xs text-slate-400 mt-3 italic">
        Configure webhooks in the Deployment panel → Webhooks section.
      </p>
    </div>
  )
}
