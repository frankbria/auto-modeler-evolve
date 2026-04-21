"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { PredictionAuditResult } from "@/lib/types"

// ---------------------------------------------------------------------------
// PredictionAuditCard — comprehensive deployment monitoring digest
// ---------------------------------------------------------------------------

interface PredictionAuditCardProps {
  result: PredictionAuditResult
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    healthy: "bg-emerald-100 text-emerald-700 border-emerald-300",
    warning: "bg-amber-100 text-amber-700 border-amber-300",
    critical: "bg-rose-100 text-rose-700 border-rose-300",
  }
  const icons: Record<string, string> = {
    healthy: "✓",
    warning: "⚠",
    critical: "✗",
  }
  const cls = styles[status] ?? styles.healthy
  return (
    <Badge className={`border text-xs font-semibold ${cls}`}>
      <span aria-hidden="true">{icons[status] ?? "?"}</span>
      <span className="ml-1">{status === "healthy" ? "Healthy" : status === "warning" ? "Needs Attention" : "Critical"}</span>
    </Badge>
  )
}

function ConfBar({
  pct,
  color,
  label,
}: {
  pct: number
  color: string
  label: string
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-14 text-right text-slate-500 shrink-0">{label}</span>
      <div
        className="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label}: ${pct.toFixed(0)}%`}
      >
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </div>
      <span className="w-9 text-slate-700 font-medium tabular-nums">
        {pct.toFixed(0)}%
      </span>
    </div>
  )
}

export function PredictionAuditCard({ result }: PredictionAuditCardProps) {
  const {
    total_predictions,
    predictions_today,
    predictions_7d,
    predictions_30d,
    confidence_high_pct,
    confidence_medium_pct,
    confidence_low_pct,
    has_confidence_data,
    p50_ms,
    p95_ms,
    has_latency_data,
    sla_alert,
    quota_used,
    monthly_quota,
    quota_pct,
    quota_enabled,
    overall_status,
    summary,
  } = result

  const isEmpty = total_predictions === 0

  const borderColor =
    overall_status === "critical"
      ? "border-rose-300 bg-rose-50"
      : overall_status === "warning"
      ? "border-amber-300 bg-amber-50"
      : "border-emerald-300 bg-emerald-50"

  return (
    <Card
      className={`border ${borderColor} w-full max-w-xl`}
      aria-label="Deployment monitoring audit"
    >
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold text-slate-800">
          <span aria-hidden="true">📊</span>
          Deployment Audit
        </CardTitle>
        <div className="flex flex-wrap gap-1 mt-1">
          <StatusBadge status={overall_status} />
          <Badge className="bg-slate-200 text-slate-700 border border-slate-300 text-xs">
            {total_predictions.toLocaleString()} total predictions
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4 text-sm">
        {isEmpty ? (
          <p className="text-slate-500 italic">
            No predictions recorded yet. Metrics will appear here once the
            deployed model receives API requests.
          </p>
        ) : (
          <>
            {/* Volume */}
            <section>
              <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
                Prediction Volume
              </h3>
              <div className="grid grid-cols-3 gap-2 text-center">
                {[
                  { label: "Today", value: predictions_today },
                  { label: "7 days", value: predictions_7d },
                  { label: "30 days", value: predictions_30d },
                ].map(({ label, value }) => (
                  <div
                    key={label}
                    className="rounded border border-slate-200 bg-white py-2 px-1"
                  >
                    <div className="text-base font-semibold text-slate-800 tabular-nums">
                      {value.toLocaleString()}
                    </div>
                    <div className="text-xs text-slate-500">{label}</div>
                  </div>
                ))}
              </div>
            </section>

            {/* Confidence distribution */}
            {has_confidence_data && (
              <section>
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
                  Confidence Distribution
                </h3>
                <div className="space-y-1.5">
                  <ConfBar
                    pct={confidence_high_pct}
                    color="bg-emerald-500"
                    label="High ≥80%"
                  />
                  <ConfBar
                    pct={confidence_medium_pct}
                    color="bg-amber-400"
                    label="Med 60–79%"
                  />
                  <ConfBar
                    pct={confidence_low_pct}
                    color="bg-rose-400"
                    label="Low <60%"
                  />
                </div>
              </section>
            )}

            {/* SLA */}
            {has_latency_data && (
              <section>
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
                  Response Latency
                </h3>
                <div className="flex flex-wrap gap-3 items-center">
                  <span className="text-slate-600">
                    p50:{" "}
                    <span className="font-semibold text-slate-800">
                      {p50_ms}ms
                    </span>
                  </span>
                  <span className={sla_alert ? "text-rose-700 font-semibold" : "text-slate-600"}>
                    p95:{" "}
                    <span className="font-semibold">
                      {p95_ms}ms
                    </span>
                    {sla_alert && (
                      <span
                        role="alert"
                        className="ml-1 text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded px-1 py-0.5"
                      >
                        ⚠ above 500ms SLA
                      </span>
                    )}
                  </span>
                </div>
              </section>
            )}

            {/* Quota */}
            {quota_enabled && quota_pct !== null && quota_pct !== undefined && (
              <section>
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
                  Monthly Quota
                </h3>
                <div className="flex items-center gap-2">
                  <div
                    className="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden"
                    role="progressbar"
                    aria-valuenow={quota_pct}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={`Quota ${quota_pct.toFixed(0)}% used`}
                  >
                    <div
                      className={`h-full rounded-full transition-all ${
                        quota_pct >= 90
                          ? "bg-rose-500"
                          : quota_pct >= 70
                          ? "bg-amber-400"
                          : "bg-emerald-500"
                      }`}
                      style={{ width: `${Math.min(100, quota_pct)}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-600 tabular-nums whitespace-nowrap">
                    {quota_used.toLocaleString()} /{" "}
                    {(monthly_quota ?? 0).toLocaleString()} (
                    {quota_pct.toFixed(0)}%)
                  </span>
                </div>
              </section>
            )}

            {/* Summary */}
            <p className="text-xs text-slate-600 italic border-t border-slate-200 pt-2">
              {summary}
            </p>
          </>
        )}

        <figcaption className="sr-only">
          {isEmpty
            ? "Deployment audit: no predictions recorded yet."
            : `Deployment audit: ${total_predictions.toLocaleString()} total predictions. Status: ${overall_status}.`}
        </figcaption>
      </CardContent>
    </Card>
  )
}
