"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { WebhookListChatResult, WebhookListEntry } from "@/lib/types"

const EVENT_LABELS: Record<string, string> = {
  batch_complete: "Batch Complete",
  drift_detected: "Drift Detected",
  health_degraded: "Health Degraded",
  quota_alert: "Quota Alert",
}

function formatRelative(iso: string | null): string {
  if (!iso) return "never"
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function WebhookRow({ hook }: { hook: WebhookListEntry }) {
  const lastStatus = hook.last_status_code
  const statusOk = lastStatus !== null && lastStatus >= 200 && lastStatus < 300

  return (
    <li
      role="listitem"
      aria-label={`Webhook at ${hook.url}`}
      className="py-2 border-b border-gray-100 last:border-0 space-y-1"
    >
      <div className="flex items-start justify-between gap-2">
        <code className="text-xs text-gray-700 break-all flex-1">{hook.url}</code>
        {lastStatus !== null && (
          <Badge
            className={
              statusOk
                ? "text-xs bg-emerald-100 text-emerald-800 border-emerald-300 shrink-0"
                : "text-xs bg-rose-100 text-rose-800 border-rose-300 shrink-0"
            }
          >
            {lastStatus}
          </Badge>
        )}
      </div>
      <div className="flex flex-wrap gap-1">
        {hook.event_types.map((et) => (
          <Badge
            key={et}
            className="text-xs bg-blue-100 text-blue-800 border-blue-300"
          >
            {EVENT_LABELS[et] ?? et}
          </Badge>
        ))}
      </div>
      <p className="text-xs text-gray-500">
        Last fired: {formatRelative(hook.last_fired_at)}
      </p>
    </li>
  )
}

interface WebhookListChatCardProps {
  result: WebhookListChatResult
}

export function WebhookListChatCard({ result }: WebhookListChatCardProps) {
  return (
    <figure
      role="region"
      aria-label="Active webhooks"
      className="rounded-lg border border-slate-300 bg-slate-50/40 overflow-hidden"
    >
      <Card className="border-0 bg-transparent shadow-none">
        <CardHeader className="pb-2 pt-3 px-4">
          <CardTitle className="text-sm font-semibold text-slate-800 flex items-center gap-2">
            <span aria-hidden="true">🔗</span>
            Active Webhooks
            <Badge className="ml-auto text-xs bg-slate-200 text-slate-700 border-slate-300">
              {result.total}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-3">
          {result.total === 0 ? (
            <p className="text-sm text-gray-500 italic">
              No webhooks registered. Say &ldquo;register a webhook at
              https://...&rdquo; to add one.
            </p>
          ) : (
            <ul role="list" aria-label="Registered webhooks" className="divide-y divide-gray-100">
              {result.webhooks.map((hook) => (
                <WebhookRow key={hook.id} hook={hook} />
              ))}
            </ul>
          )}
          <p className="text-xs text-gray-400 mt-2">{result.summary}</p>
        </CardContent>
      </Card>
      <figcaption className="sr-only">{result.summary}</figcaption>
    </figure>
  )
}
