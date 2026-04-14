"use client"

import { Badge } from "@/components/ui/badge"
import type { WebhookHistoryResult, WebhookEventRecord } from "@/lib/types"

// Map event type strings to human-readable labels + badge colors
const EVENT_TYPE_META: Record<
  string,
  { label: string; className: string }
> = {
  batch_complete: {
    label: "Batch Complete",
    className: "bg-sky-100 text-sky-800 border-sky-200",
  },
  drift_detected: {
    label: "Drift Detected",
    className: "bg-amber-100 text-amber-800 border-amber-200",
  },
  health_degraded: {
    label: "Health Degraded",
    className: "bg-red-100 text-red-800 border-red-200",
  },
  quota_alert: {
    label: "Quota Alert",
    className: "bg-orange-100 text-orange-800 border-orange-200",
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

function StatusBadge({ success, code }: { success: boolean; code: number | null }) {
  if (code === null) {
    return (
      <Badge className="bg-slate-100 text-slate-600 border-slate-200 text-xs">
        —
      </Badge>
    )
  }
  if (success) {
    return (
      <Badge className="bg-emerald-100 text-emerald-800 border-emerald-200 text-xs">
        {code} OK
      </Badge>
    )
  }
  return (
    <Badge className="bg-red-100 text-red-800 border-red-200 text-xs">
      {code === 0 ? "Error" : code}
    </Badge>
  )
}

function EventRow({ event }: { event: WebhookEventRecord }) {
  const meta =
    EVENT_TYPE_META[event.event_type] ?? {
      label: event.event_type,
      className: "bg-slate-100 text-slate-700 border-slate-200",
    }
  return (
    <div className="flex items-center gap-2 py-2 border-b border-slate-100 last:border-0 text-sm">
      <Badge className={`${meta.className} text-xs shrink-0`}>{meta.label}</Badge>
      <span className="text-slate-500 text-xs truncate flex-1 min-w-0" title={event.webhook_url}>
        {event.webhook_url}
      </span>
      <span className="text-slate-400 text-xs shrink-0 hidden sm:block">
        {formatDate(event.fired_at)}
      </span>
      <StatusBadge success={event.success} code={event.status_code} />
    </div>
  )
}

interface WebhookHistoryCardProps {
  data: WebhookHistoryResult
}

export function WebhookHistoryCard({ data }: WebhookHistoryCardProps) {
  return (
    <div
      role="region"
      aria-label="Webhook event history"
      className="rounded-lg border border-slate-300 bg-white p-4 my-2 shadow-sm"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span aria-hidden="true" className="text-lg">🔔</span>
        <h3 className="font-semibold text-slate-800 text-sm">Webhook Event History</h3>
        <Badge className="bg-slate-100 text-slate-700 border-slate-200 text-xs ml-auto">
          {data.total} event{data.total !== 1 ? "s" : ""}
        </Badge>
      </div>

      {/* Summary */}
      <p className="text-sm text-slate-600 mb-3">{data.summary}</p>

      {/* Events */}
      {data.total === 0 ? (
        <p className="text-sm text-slate-400 italic">
          No webhook events recorded yet. Events are logged when batch prediction
          jobs complete, drift is detected, model health degrades, or quota
          thresholds are crossed.
        </p>
      ) : (
        <div>
          {/* Column header */}
          <div className="flex items-center gap-2 pb-1 mb-1 border-b border-slate-200 text-xs font-medium text-slate-500">
            <span className="shrink-0 w-[110px]">Event</span>
            <span className="flex-1 min-w-0">URL</span>
            <span className="shrink-0 hidden sm:block w-[130px]">Time</span>
            <span className="shrink-0 w-[60px] text-right">Status</span>
          </div>
          {data.events.map((evt) => (
            <EventRow key={evt.id} event={evt} />
          ))}
        </div>
      )}

      {/* Help footer */}
      <p className="text-xs text-slate-400 mt-3 italic">
        Showing up to 10 most recent events. Register webhooks in the Deployment panel.
      </p>
    </div>
  )
}
