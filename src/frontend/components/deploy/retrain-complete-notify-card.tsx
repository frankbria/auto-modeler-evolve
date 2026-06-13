"use client"

import { Badge } from "@/components/ui/badge"
import type { RetrainCompleteNotifyResult } from "@/lib/types"

function formatDate(iso: string | null): string {
  if (!iso) return "—"
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

function formatDuration(ms: number | null): string {
  if (ms === null || ms === undefined) return "—"
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}m`
}

interface RetrainCompleteNotifyCardProps {
  data: RetrainCompleteNotifyResult
}

export function RetrainCompleteNotifyCard({ data }: RetrainCompleteNotifyCardProps) {
  const hasLastRun = data.last_completed_retrain !== null
  const hasWebhooks = data.has_notification

  const borderClass = hasWebhooks
    ? hasLastRun
      ? "border-emerald-300"
      : "border-violet-300"
    : "border-amber-300"

  const badgeClass = hasWebhooks
    ? "bg-emerald-100 text-emerald-800 border-emerald-200"
    : "bg-amber-100 text-amber-800 border-amber-200"

  return (
    <figure
      role="region"
      aria-label="Retrain completion notification status"
      className={`rounded-lg border ${borderClass} bg-white p-4 my-2 shadow-sm`}
      data-testid="retrain-complete-notify-card"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span aria-hidden="true" className="text-lg">🔔</span>
        <h3 className="font-semibold text-slate-800 text-sm">
          Retrain Completion Notification
        </h3>
        <Badge className={`${badgeClass} text-xs ml-auto`}>
          {hasWebhooks ? "Webhook registered" : "No webhook"}
        </Badge>
      </div>

      {/* Summary */}
      <p className="text-sm text-slate-600 mb-3">{data.summary}</p>

      {/* Last completed retrain */}
      {hasLastRun && data.last_completed_retrain && (
        <div className="bg-slate-50 rounded p-3 mb-3 text-sm">
          <div className="flex items-center gap-2 mb-2">
            <span aria-hidden="true">✅</span>
            <span className="font-medium text-slate-700">Last completed retrain</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-slate-500">Algorithm</span>
              <p className="font-medium text-slate-800" data-testid="last-algorithm">
                {data.last_completed_retrain.algorithm}
              </p>
            </div>
            <div>
              <span className="text-slate-500">Training time</span>
              <p className="font-medium text-slate-800" data-testid="training-duration">
                {formatDuration(data.last_completed_retrain.training_duration_ms)}
              </p>
            </div>
            {data.last_completed_retrain.primary_metric && data.last_completed_retrain.primary_metric_value !== null && (
              <div>
                <span className="text-slate-500 capitalize">
                  {data.last_completed_retrain.primary_metric}
                </span>
                <p className="font-medium text-slate-800" data-testid="primary-metric-value">
                  {typeof data.last_completed_retrain.primary_metric_value === "number"
                    ? data.last_completed_retrain.primary_metric_value.toFixed(4)
                    : data.last_completed_retrain.primary_metric_value}
                </p>
              </div>
            )}
            <div>
              <span className="text-slate-500">Completed at</span>
              <p className="font-medium text-slate-800" data-testid="completed-at">
                {formatDate(data.last_completed_retrain.completed_at)}
              </p>
            </div>
          </div>
        </div>
      )}

      {!hasLastRun && (
        <p className="text-xs text-slate-400 italic mb-3">
          No completed model runs found yet for this deployment.
        </p>
      )}

      {/* Webhook registrations */}
      {hasWebhooks && data.retrain_complete_webhooks.length > 0 && (
        <div className="bg-emerald-50 border border-emerald-100 rounded p-3 mb-3 text-xs text-emerald-800">
          <p className="font-medium mb-1">Registered notification URLs</p>
          <ul className="space-y-0.5 list-disc list-inside text-emerald-700">
            {data.retrain_complete_webhooks.map((url, i) => (
              <li key={i} className="truncate" data-testid={`webhook-url-${i}`}>
                {url}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* How-to info block */}
      <div className="bg-violet-50 border border-violet-100 rounded p-3 mb-3 text-xs text-violet-800">
        <p className="font-medium mb-1">How retrain completion notifications work</p>
        <ul className="space-y-0.5 list-disc list-inside text-violet-700">
          <li>
            When degradation-triggered retraining completes, a{" "}
            <code>retrain_complete</code> webhook fires automatically
          </li>
          <li>
            Payload includes: algorithm, primary metric value, training duration, and a plain-English message
          </li>
          <li>
            Register a webhook with event type <code>retrain_complete</code> via
            the Webhooks section of the deployment panel
          </li>
          <li>
            Use with Slack, Teams, or Zapier to get notified the moment your
            auto-retrain finishes
          </li>
        </ul>
      </div>

      {!hasWebhooks && (
        <p className="text-xs text-slate-400 italic mt-1">
          Say <em>&quot;register a webhook&quot;</em> or use the Webhooks section
          of the deployment panel to set up notifications.
        </p>
      )}

      <figcaption className="sr-only">
        Retrain completion notification:{" "}
        {hasWebhooks
          ? `${data.retrain_complete_webhooks.length} webhook(s) registered for retrain_complete event.`
          : "No retrain_complete webhooks registered."}
        {hasLastRun && data.last_completed_retrain
          ? ` Last completed: ${data.last_completed_retrain.algorithm} at ${formatDate(data.last_completed_retrain.completed_at)}.`
          : ""}
      </figcaption>

      <p className="text-xs text-slate-400 mt-3 italic">
        Fires when auto-retrain completes — distinct from the{" "}
        <code>degradation_retrain</code> webhook (fires when retraining starts).
      </p>
    </figure>
  )
}
