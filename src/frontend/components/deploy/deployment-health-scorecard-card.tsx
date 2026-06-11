"use client"

import type { DeploymentHealthScorecardResult, DeploymentHealthSignal } from "@/lib/types"

interface Props {
  result: DeploymentHealthScorecardResult
}

const GRADE_STYLES: Record<string, { bg: string; text: string; border: string }> = {
  A: { bg: "bg-emerald-100", text: "text-emerald-800", border: "border-emerald-300" },
  B: { bg: "bg-green-100", text: "text-green-800", border: "border-green-300" },
  C: { bg: "bg-amber-100", text: "text-amber-800", border: "border-amber-300" },
  D: { bg: "bg-orange-100", text: "text-orange-800", border: "border-orange-300" },
  F: { bg: "bg-rose-100", text: "text-rose-800", border: "border-rose-300" },
}

const HEALTH_CARD_BORDER: Record<string, string> = {
  healthy: "border-emerald-300/70 bg-emerald-50/40",
  watching: "border-amber-300/70 bg-amber-50/40",
  warning: "border-orange-300/70 bg-orange-50/40",
  critical: "border-rose-300/70 bg-rose-50/40",
}

const SEVERITY_STYLES: Record<string, { dot: string; row: string; label: string }> = {
  green: { dot: "bg-emerald-500", row: "bg-white", label: "text-emerald-700" },
  amber: { dot: "bg-amber-500", row: "bg-amber-50", label: "text-amber-700" },
  red: { dot: "bg-rose-500", row: "bg-rose-50", label: "text-rose-700" },
  gray: { dot: "bg-slate-300", row: "bg-slate-50", label: "text-slate-500" },
}

function SignalRow({ signal }: { signal: DeploymentHealthSignal }) {
  const sty = SEVERITY_STYLES[signal.severity] ?? SEVERITY_STYLES.gray
  return (
    <div
      className={`flex items-start gap-3 rounded-lg px-3 py-2 ${sty.row}`}
      data-testid={`signal-row-${signal.signal_key}`}
    >
      <span className="mt-0.5 text-base shrink-0" aria-hidden="true">
        {signal.icon}
      </span>
      <div className="flex min-w-0 flex-1 items-center gap-2">
        <span className="text-sm font-medium text-slate-700 w-44 shrink-0">
          {signal.signal_label}
        </span>
        <span
          className={`inline-block h-2 w-2 rounded-full shrink-0 ${sty.dot}`}
          aria-label={`Status: ${signal.value_label}`}
        />
        <span className={`text-xs font-medium w-28 shrink-0 ${sty.label}`}>
          {signal.value_label}
        </span>
        <span className="text-xs text-slate-500 truncate">{signal.finding}</span>
      </div>
    </div>
  )
}

export function DeploymentHealthScorecardCard({ result }: Props) {
  const grade = result.overall_grade ?? "C"
  const gradeSty = GRADE_STYLES[grade] ?? GRADE_STYLES.C
  const cardBorder = HEALTH_CARD_BORDER[result.overall_health] ?? HEALTH_CARD_BORDER.watching

  const healthLabel = {
    healthy: "Healthy",
    watching: "Watching",
    warning: "Warning",
    critical: "Critical",
  }[result.overall_health] ?? "Unknown"

  return (
    <div
      className={`rounded-xl border-2 p-4 ${cardBorder}`}
      data-testid="deployment-health-scorecard-card"
      role="region"
      aria-label="Deployment Health Scorecard"
    >
      <figcaption className="sr-only">
        Deployment health scorecard — overall grade {grade} ({result.overall_score}/100,{" "}
        {healthLabel}). {result.summary}
      </figcaption>

      {/* Header */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-lg" aria-hidden="true">
          🏥
        </span>
        <span className="text-sm font-semibold text-slate-800">
          Deployment Health Scorecard
        </span>

        {/* Grade badge */}
        <span
          className={`inline-flex items-center justify-center rounded-full px-2.5 py-0.5 text-sm font-bold border ${gradeSty.bg} ${gradeSty.text} ${gradeSty.border}`}
          data-testid="grade-badge"
          aria-label={`Grade ${grade}`}
        >
          {grade}
        </span>

        {/* Score badge */}
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
          Score {result.overall_score}/100
        </span>

        {/* Health badge */}
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-medium ${
            result.overall_health === "healthy"
              ? "bg-emerald-100 text-emerald-800"
              : result.overall_health === "watching"
                ? "bg-amber-100 text-amber-800"
                : result.overall_health === "warning"
                  ? "bg-orange-100 text-orange-800"
                  : "bg-rose-100 text-rose-800"
          }`}
          data-testid="health-badge"
        >
          {healthLabel}
        </span>

        {/* Signal counts */}
        {result.n_signals_critical > 0 && (
          <span className="rounded-full bg-rose-100 px-2 py-0.5 text-xs font-medium text-rose-700">
            {result.n_signals_critical} critical
          </span>
        )}
        {result.n_signals_warning > 0 && (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
            {result.n_signals_warning} warning
          </span>
        )}
        {result.n_signals_critical === 0 && result.n_signals_warning === 0 && (
          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
            All signals healthy
          </span>
        )}

        {/* Algorithm + target */}
        {result.algorithm && result.target_column && (
          <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
            {result.algorithm.replace(/_/g, " ")} → {result.target_column}
          </span>
        )}
      </div>

      {/* Signal rows */}
      <div className="mb-3 flex flex-col gap-1" role="list" aria-label="Health signals">
        {result.signals.map((signal) => (
          <div key={signal.signal_key} role="listitem">
            <SignalRow signal={signal} />
          </div>
        ))}
      </div>

      {/* Canary info */}
      {result.canary_info?.is_active && (
        <div className="mb-3 rounded-lg border border-violet-200 bg-violet-50 px-3 py-2">
          <p className="text-xs font-medium text-violet-700">
            🐦 Canary Active — {result.canary_info.finding}
          </p>
        </div>
      )}

      {/* Summary */}
      <p className="text-xs text-slate-500" data-testid="scorecard-summary">
        {result.summary}
      </p>
    </div>
  )
}
