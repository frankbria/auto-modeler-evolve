"use client"

import { WeeklyDigestConfigResult } from "@/lib/types"

interface Props {
  data: WeeklyDigestConfigResult
}

const DAY_LABELS = [
  "Monday", "Tuesday", "Wednesday", "Thursday",
  "Friday", "Saturday", "Sunday",
]

export default function WeeklyDigestConfigCard({ data }: Props) {
  const {
    action,
    enabled,
    day_name,
    send_hour,
    last_sent_at,
    summary,
  } = data

  const borderColor =
    action === "enabled"
      ? "border-teal-400/40 bg-teal-50/30"
      : action === "disabled"
      ? "border-slate-300/40 bg-slate-50/30"
      : enabled
      ? "border-teal-400/40 bg-teal-50/30"
      : "border-slate-300/40 bg-slate-50/30"

  const statusBadge = enabled ? (
    <span
      className="inline-flex items-center rounded-full bg-teal-100 px-2 py-0.5 text-xs font-medium text-teal-800"
      data-testid="digest-enabled-badge"
    >
      Enabled
    </span>
  ) : (
    <span
      className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700"
      data-testid="digest-disabled-badge"
    >
      Disabled
    </span>
  )

  const actionHeading =
    action === "enabled"
      ? "Weekly Digest Enabled"
      : action === "disabled"
      ? "Weekly Digest Disabled"
      : "Weekly Digest Status"

  return (
    <figure
      className={`rounded-xl border p-4 text-sm space-y-3 ${borderColor}`}
      data-testid="weekly-digest-config-card"
      aria-label="Weekly monitoring digest configuration"
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span aria-hidden="true" className="text-base">📅</span>
        <span className="font-semibold text-base">{actionHeading}</span>
        {statusBadge}
      </div>

      {enabled && (
        <div className="grid grid-cols-2 gap-3" data-testid="digest-schedule-grid">
          <div className="rounded-lg bg-white/60 border border-teal-200/60 p-3 text-center">
            <div className="text-xs text-muted-foreground mb-1">Send Day</div>
            <div className="font-semibold text-teal-700" data-testid="digest-day">
              {day_name}
            </div>
          </div>
          <div className="rounded-lg bg-white/60 border border-teal-200/60 p-3 text-center">
            <div className="text-xs text-muted-foreground mb-1">Send Time</div>
            <div className="font-semibold text-teal-700" data-testid="digest-time">
              {String(send_hour).padStart(2, "0")}:00 UTC
            </div>
          </div>
        </div>
      )}

      {last_sent_at && (
        <p className="text-xs text-muted-foreground" data-testid="digest-last-sent">
          Last sent:{" "}
          {new Date(last_sent_at).toLocaleDateString(undefined, {
            weekday: "long",
            year: "numeric",
            month: "short",
            day: "numeric",
          })}
        </p>
      )}

      <div
        className="rounded-lg bg-white/50 border border-current/10 p-3 text-xs text-muted-foreground"
        data-testid="digest-webhook-note"
      >
        <strong>How it works:</strong> AutoModeler computes all monitoring signals (anomalies,
        drift, retraining readiness) automatically and dispatches the summary to any webhook
        registered for <code className="bg-muted px-1 rounded">weekly_digest</code> events.
        {!enabled && (
          <> To receive reports, enable the digest and register a webhook.</>
        )}
      </div>

      <p className="text-xs text-muted-foreground" data-testid="digest-summary">
        {summary}
      </p>

      <figcaption className="sr-only">
        Weekly monitoring digest is {enabled ? "enabled" : "disabled"}.
        {enabled && ` Sends every ${day_name} at ${String(send_hour).padStart(2, "0")}:00 UTC.`}
      </figcaption>
    </figure>
  )
}
