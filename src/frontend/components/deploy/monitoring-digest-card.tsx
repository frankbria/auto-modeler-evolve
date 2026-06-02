"use client"

import type { MonitoringDigestResult, MonitoringDigestSignal } from "@/lib/types"

interface MonitoringDigestCardProps {
  result: MonitoringDigestResult
}

const HEALTH_STYLES: Record<string, { border: string; badge: string; label: string }> = {
  healthy: {
    border: "border-blue-200 bg-blue-50",
    badge: "bg-blue-100 text-blue-800",
    label: "Healthy",
  },
  watching: {
    border: "border-amber-200 bg-amber-50",
    badge: "bg-amber-100 text-amber-800",
    label: "Watching",
  },
  warning: {
    border: "border-orange-200 bg-orange-50",
    badge: "bg-orange-100 text-orange-800",
    label: "Warning",
  },
  critical: {
    border: "border-rose-200 bg-rose-50",
    badge: "bg-rose-100 text-rose-800",
    label: "Critical",
  },
}

const SEVERITY_DOT: Record<string, string> = {
  green: "bg-emerald-500",
  amber: "bg-amber-500",
  red: "bg-rose-500",
  gray: "bg-slate-300",
}

const SEVERITY_ROW_BG: Record<string, string> = {
  green: "bg-white",
  amber: "bg-amber-50",
  red: "bg-rose-50",
  gray: "bg-slate-50",
}

function SignalRow({ signal }: { signal: MonitoringDigestSignal }) {
  return (
    <div
      className={`flex items-start gap-3 rounded-lg px-3 py-2 ${SEVERITY_ROW_BG[signal.severity]}`}
      data-testid={`signal-row-${signal.signal_key}`}
    >
      <span className="mt-0.5 text-base" aria-hidden="true">
        {signal.icon}
      </span>
      <div className="flex min-w-0 flex-1 items-center gap-2">
        <span className="text-sm font-medium text-slate-700 w-44 shrink-0">
          {signal.signal_label}
        </span>
        <span
          className={`inline-block h-2 w-2 rounded-full shrink-0 ${SEVERITY_DOT[signal.severity]}`}
          aria-label={`Status: ${signal.verdict_label}`}
        />
        <span className="text-xs font-medium text-slate-600 w-28 shrink-0">
          {signal.verdict_label}
        </span>
        <span className="text-xs text-slate-500 truncate">{signal.finding}</span>
      </div>
    </div>
  )
}

export function MonitoringDigestCard({ result }: MonitoringDigestCardProps) {
  const health = result.overall_health ?? "healthy"
  const styles = HEALTH_STYLES[health] ?? HEALTH_STYLES.healthy

  return (
    <div
      className={`rounded-xl border-2 p-4 ${styles.border}`}
      data-testid="monitoring-digest-card"
    >
      <figcaption className="sr-only">
        Deployment monitoring signal digest showing {result.signals_total} signals — overall health:{" "}
        {health}
      </figcaption>

      {/* Header */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-lg" aria-hidden="true">
          📡
        </span>
        <span className="text-sm font-semibold text-slate-800">
          Monitoring Signal Digest
        </span>
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-medium capitalize ${styles.badge}`}
          data-testid="overall-health-badge"
        >
          {styles.label}
        </span>
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
          Score {result.overall_score}/100
        </span>
        {result.signals_firing > 0 ? (
          <span className="rounded-full bg-rose-100 px-2 py-0.5 text-xs font-medium text-rose-700">
            {result.signals_firing} signal{result.signals_firing !== 1 ? "s" : ""} need attention
          </span>
        ) : (
          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
            All signals healthy
          </span>
        )}
        {result.target_col && (
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            {result.target_col}
          </span>
        )}
      </div>

      {/* Signal rows */}
      <div className="mb-3 flex flex-col gap-1" role="list" aria-label="Monitoring signals">
        {result.signals.map((signal) => (
          <div key={signal.signal_key} role="listitem">
            <SignalRow signal={signal} />
          </div>
        ))}
      </div>

      {/* Priority actions */}
      {result.priority_actions.length > 0 && (
        <div className="mb-3 rounded-lg border border-amber-200 bg-amber-50 p-3">
          <p className="mb-1.5 text-xs font-semibold text-amber-800 uppercase tracking-wide">
            Priority Actions
          </p>
          <ul className="space-y-1" data-testid="priority-actions">
            {result.priority_actions.map((action, i) => (
              <li key={i} className="flex items-start gap-1.5 text-xs text-amber-700">
                <span aria-hidden="true">→</span>
                <span>{action}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Summary */}
      <p className="text-xs text-slate-500" data-testid="digest-summary">
        {result.summary}
      </p>
    </div>
  )
}
