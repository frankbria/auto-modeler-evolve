"use client"

import { Badge } from "@/components/ui/badge"
import type { PredValueAlertConfig } from "@/lib/types"

function formatDate(iso: string | null): string {
  if (!iso) return "—"
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

interface PredValueAlertCardProps {
  data: PredValueAlertConfig
}

export function PredValueAlertCard({ data }: PredValueAlertCardProps) {
  const isEnabled = data.pred_value_alert_enabled && data.alert_pct !== null

  const borderClass = isEnabled
    ? data.pred_value_alert_last_fired_at
      ? "border-amber-300"
      : "border-emerald-300"
    : "border-slate-300"

  const badgeClass = isEnabled
    ? "bg-emerald-100 text-emerald-800 border-emerald-200"
    : "bg-slate-100 text-slate-600 border-slate-200"

  return (
    <figure
      role="region"
      aria-label="Prediction value trend alert configuration"
      className={`rounded-lg border ${borderClass} bg-white p-4 my-2 shadow-sm`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span aria-hidden="true" className="text-lg">📈</span>
        <h3 className="font-semibold text-slate-800 text-sm">Prediction Value Trend Alert</h3>
        <Badge className={`${badgeClass} text-xs ml-auto`}>
          {isEnabled ? "Enabled" : "Disabled"}
        </Badge>
      </div>

      {/* Summary */}
      <p className="text-sm text-slate-600 mb-3">{data.summary}</p>

      {isEnabled && data.alert_pct !== null && (
        <>
          {/* Threshold tile */}
          <div className="bg-slate-50 rounded p-3 mb-3 text-sm">
            <div className="flex items-center gap-2 mb-1">
              <span aria-hidden="true">🎯</span>
              <span className="font-medium text-slate-700">Shift threshold</span>
              <Badge className="bg-orange-100 text-orange-800 border-orange-200 text-xs ml-auto">
                {data.alert_pct}% change triggers alert
              </Badge>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              When the mean prediction value shifts by more than{" "}
              <strong>{data.alert_pct}%</strong> comparing the early half vs.
              the recent half of the last 100 predictions, a webhook fires.
            </p>
          </div>

          {/* How it works info block */}
          <div className="bg-blue-50 border border-blue-100 rounded p-3 mb-3 text-xs text-blue-800">
            <p className="font-medium mb-1">How it works</p>
            <ul className="space-y-0.5 list-disc list-inside text-blue-700">
              <li>Checked every 60 seconds · requires at least 20 predictions</li>
              <li>
                Compares mean of older 50 predictions vs. newest 50 predictions
              </li>
              <li>
                Works for regression (uses prediction value) and classification
                (uses confidence score)
              </li>
              <li>
                <strong>{data.cooldown_hours}-hour</strong> cooldown between
                alerts to prevent noise
              </li>
              <li>
                Fires a <code>pred_value_trend_alert</code> webhook with early
                mean, recent mean, and change %
              </li>
            </ul>
          </div>

          {/* Last fired */}
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span aria-hidden="true">⏱</span>
            {data.pred_value_alert_last_fired_at ? (
              <span>
                Last fired:{" "}
                <span className="text-amber-600 font-medium">
                  {formatDate(data.pred_value_alert_last_fired_at)}
                </span>
              </span>
            ) : (
              <span className="text-slate-400 italic">No alerts fired yet</span>
            )}
          </div>
        </>
      )}

      {!isEnabled && (
        <p className="text-xs text-slate-400 italic mt-1">
          Say{" "}
          <em>
            &quot;alert me if average prediction drops more than 15%&quot;
          </em>{" "}
          to enable proactive monitoring.
        </p>
      )}

      <figcaption className="sr-only">
        Prediction value trend alert is{" "}
        {isEnabled
          ? `enabled with a ${data.alert_pct}% shift threshold`
          : "disabled"}
        .
        {data.pred_value_alert_last_fired_at
          ? ` Last fired: ${formatDate(data.pred_value_alert_last_fired_at)}.`
          : ""}
      </figcaption>

      {/* Footer */}
      <p className="text-xs text-slate-400 mt-3 italic">
        Distinct from accuracy alerts (which require labels) and output drift
        cards (historical snapshot only). Requires webhooks registered.
      </p>
    </figure>
  )
}
